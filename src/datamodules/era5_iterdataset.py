import math
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset


class ERA5Npy(IterableDataset):
    def __init__(self, file_list, variables, shuffle: bool = False) -> None:
        super().__init__()
        self.file_list = file_list
        self.variables = variables
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.file_list)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.file_list)
        else:
            if not torch.distributed.is_initialized():
                rank = 0
                world_size = 1
            else:
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
            num_workers_per_ddp = worker_info.num_workers
            num_shards = num_workers_per_ddp * world_size
            per_worker = int(math.floor(len(self.file_list) / float(num_shards)))
            worker_id = rank * num_workers_per_ddp + worker_info.id
            iter_start = worker_id * per_worker
            # iter_end = min(iter_start + per_worker, len(self.file_list))
            iter_end = iter_start + per_worker

        # print(f"rank {rank}")
        # print(f"world size {world_size}")
        # print(f"len data {len(self.file_list)}")
        # print(f"start {iter_start}, end {iter_end}")

        # count the number of data points this worker holds
        # num_data = 0
        # for idx in range(iter_start, iter_end):
        #     data = np.load(self.file_list[idx])
        #     num_data += data["t2m"].shape[0]

        # print(f"rank {rank}")
        # print(f"{num_data} data points")

        # print("==============================")

        for idx in range(iter_start, iter_end):
            path = self.file_list[idx]
            data = np.load(path)
            yield {k: data[k] for k in self.variables}


class ERA5(IterableDataset):
    def __init__(self, dataset: ERA5Npy) -> None:
        super().__init__()
        self.dataset = dataset

    def __iter__(self):
        for data in self.dataset:
            np_data = np.concatenate([data[k] for k in data.keys()], axis=1)
            yield torch.from_numpy(np_data)


class IndividualDataIter(IterableDataset):
    def __init__(self, dataset: ERA5, transforms: torch.nn.Module) -> None:
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms

    def __iter__(self):
        for data in self.dataset:
            for i in range(data.shape[0]):
                yield self.transforms(data[i])


class ERA5Video(IterableDataset):
    def __init__(self, dataset: ERA5Npy, timesteps: int = 8) -> None:
        super().__init__()
        self.dataset = dataset
        self.timesteps = timesteps

    def __iter__(self):
        for data in self.dataset:
            np_data = np.concatenate([data[k] for k in data.keys()], axis=1)
            torch_data = torch.from_numpy(np_data)
            yield self.construct_video(torch_data)

    def construct_video(self, x):
        # x: 8760, 3, 128, 256
        x = x.unsqueeze(0).repeat_interleave(self.timesteps, dim=0)
        for i in range(self.timesteps):
            x[i] = torch.roll(x[i], shifts=-i, dims=0)
        x = x[:, : -self.timesteps + 1]
        return torch.transpose(x, dim0=0, dim1=1)


class ERA5Forecast(IterableDataset):
    def __init__(self, dataset: ERA5Npy, predict_range: int = 6) -> None:
        super().__init__()
        self.dataset = dataset
        self.predict_range = predict_range

    def __iter__(self):
        # TODO: this would not get us stuff across the years
        # i.e. where inputs are from previous years and output from next
        for data in self.dataset:
            inputs = np.concatenate(
                [data[k][0 : -self.predict_range : self.predict_range] for k in data.keys()],
                axis=1,
            )
            outputs = np.concatenate(
                [data[k][self.predict_range :: self.predict_range] for k in data.keys()],
                axis=1,
            )
            yield torch.from_numpy(inputs), torch.from_numpy(outputs)


class ERA5ForecastMultiStep(IterableDataset):
    def __init__(self, dataset: ERA5Npy, pred_range: int = 6, pred_steps: int = 4) -> None:
        super().__init__()
        self.dataset = dataset
        self.pred_range = pred_range
        self.pred_steps = pred_steps

    def __iter__(self):
        # TODO: this would not get us stuff across the years
        # i.e. where inputs are from previous years and output from next
        pred_range = self.pred_range
        pred_steps = self.pred_steps
        for data in self.dataset:
            inputs = {}
            outputs = {}
            for k in data.keys():
                x = data[k]  # (730, d, 128, 256)
                interval = pred_range * pred_steps

                inputs[k] = x[0:-interval:interval]

                output_k = []
                for step in range(pred_steps):
                    start = (step + 1) * pred_range
                    end = (step - pred_steps + 1) * pred_range if step != pred_steps - 1 else -1
                    output_k.append(x[start:end:interval])

                output_k = np.stack(output_k, axis=1)
                outputs[k] = output_k

            inputs = np.concatenate(
                [inputs[k] for k in inputs.keys()],
                axis=1,
            )
            outputs = np.concatenate(
                [outputs[k] for k in outputs.keys()],
                axis=2,
            )
            yield torch.from_numpy(inputs), torch.from_numpy(outputs)


class IndividualForecastDataIter(IterableDataset):
    def __init__(self, dataset: ERA5Forecast, transforms: torch.nn.Module):
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms

    def __iter__(self):
        for (inp, out) in self.dataset:
            assert inp.shape[0] == out.shape[0]
            for i in range(inp.shape[0]):
                # TODO: should we unsqueeze the first dimension?
                if self.transforms is not None:
                    yield self.transforms(inp[i]), self.transforms(out[i])
                else:
                    yield inp[i], out[i]


class ERA5ForecastPrecip(IterableDataset):
    def __init__(self, dataset: ERA5Npy, predict_range: int = 6) -> None:
        super().__init__()
        self.dataset = dataset
        self.predict_range = predict_range

    def get_tp(self, data, len_data):
        tp = data["tp"][: len_data * self.predict_range]  # len_data*predict_range, 1, 128, 256
        tp = np.reshape(tp, (len_data, self.predict_range, *tp.shape[1:]))  # len_data, predict_range, 1, 128, 256
        tp = np.sum(tp, axis=1)  # sum over the predict range
        tp = np.log(1 + tp / 1e-5) # log-transformation
        return tp

    def __iter__(self):
        # TODO: this would not get us stuff across the years
        # i.e. where inputs are from previous years and output from next
        for data in self.dataset:
            min_len = np.min([data[k].shape[0] for k in data.keys()])
            inputs = np.concatenate(
                [data[k][0 : min_len - self.predict_range : self.predict_range] for k in data.keys() if k != "tp"],
                axis=1,
            )
            outputs = np.concatenate(
                [data[k][self.predict_range : min_len : self.predict_range] for k in data.keys() if k != "tp"],
                axis=1,
            )
            tp = self.get_tp(data, outputs.shape[0])
            yield torch.from_numpy(inputs), torch.from_numpy(outputs), torch.from_numpy(tp)


class ERA5ForecastMultiStepPrecip(IterableDataset):
    def __init__(self, dataset: ERA5Npy, pred_range: int = 6, pred_steps: int = 4) -> None:
        super().__init__()
        self.dataset = dataset
        self.pred_range = pred_range
        self.pred_steps = pred_steps

    def get_tp(self, data, len_data):
        pred_range = self.pred_range
        pred_steps = self.pred_steps
        interval = pred_range * pred_steps
        tp = data["tp"][: len_data * interval]  # x, 1, 128, 256
        tp = np.reshape(tp, (len_data, pred_steps, pred_range, *tp.shape[1:]))  # len_data, predict_range, 1, 128, 256
        tp = np.sum(tp, axis=2)  # sum over the predict range
        tp = np.log(1 + tp / 1e-5) # log-transformation
        return tp

    def __iter__(self):
        # TODO: this would not get us stuff across the years
        # i.e. where inputs are from previous years and output from next
        pred_range = self.pred_range
        pred_steps = self.pred_steps
        interval = pred_range * pred_steps
        for data in self.dataset:
            inputs = {}
            outputs = {}
            min_len = np.min([data[k].shape[0] for k in data.keys()])
            for k in data.keys():
                if k != "tp":
                    x = data[k]  # (730, d, 128, 256)

                    inputs[k] = x[0 : min_len - interval : interval]

                    output_k = []
                    for step in range(pred_steps):
                        start = (step + 1) * pred_range
                        end = (step - pred_steps + 1) * pred_range if step != pred_steps - 1 else -1
                        end = min(end, min_len)
                        output_k.append(x[start:end:interval])

                    output_k = np.stack(output_k, axis=1)
                    outputs[k] = output_k

            inputs = np.concatenate(
                [inputs[k] for k in inputs.keys()],
                axis=1,
            )
            outputs = np.concatenate(
                [outputs[k] for k in outputs.keys()],
                axis=2,
            )
            tp = self.get_tp(data, inputs.shape[0])
            yield torch.from_numpy(inputs), torch.from_numpy(outputs), torch.from_numpy(tp)


class IndividualForecastPrecipDataIter(IterableDataset):
    def __init__(self, dataset: ERA5ForecastPrecip, transforms: torch.nn.Module):
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms

    def __iter__(self):
        for (inp, out, tp) in self.dataset:
            assert inp.shape[0] == out.shape[0] and inp.shape[0] == tp.shape[0]
            for i in range(inp.shape[0]):
                # TODO: should we unsqueeze the first dimension?
                if self.transforms is not None:
                    yield self.transforms(inp[i]), self.transforms(out[i]), self.transforms(tp[i])
                else:
                    yield inp[i], out[i], tp[i]


class ShuffleIterableDataset(IterableDataset):
    def __init__(self, dataset, buffer_size: int) -> None:
        super().__init__()
        assert buffer_size > 0
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        buf = []
        for x in self.dataset:
            if len(buf) == self.buffer_size:
                idx = random.randint(0, self.buffer_size - 1)
                yield buf[idx]
                buf[idx] = x
            else:
                buf.append(x)
        random.shuffle(buf)
        while buf:
            yield buf.pop()
        # try:
        #     dataset_iter = iter(self.dataset)
        #     for i in range(self.buffer_size):
        #         shufbuf.append(next(dataset_iter))
        # except:
        #     self.buffer_size = len(shufbuf)

        # try:
        #     while True:
        #         try:
        #             item = next(dataset_iter)
        #             evict_idx = random.randint(0, self.buffer_size - 1)
        #             yield shufbuf[evict_idx]
        #             shufbuf[evict_idx] = item
        #         except StopIteration:
        #             break
        #     while len(shufbuf) > 0:
        #         yield shufbuf.pop()
        # except GeneratorExit:
        #     pass


# import os

# import torchdata.datapipes as dp

# root_dir = "/datadrive/1.40625deg_equally_np"
# lister_train = list(dp.iter.FileLister(os.path.join(root_dir, "train")))
# dataset = ShuffleIterableDataset(
#     dataset=IndividualForecastPrecipDataIter(
#         dataset=ERA5ForecastPrecip(
#             dataset=ERA5Npy(
#                 file_list=lister_train, variables=["t2m", "u10", "v10", "z", "tp"]
#             ),
#             predict_range=6,
#         ),
#         transforms=None,
#     ),
#     buffer_size=1000,
# )

# x, y, tp = next(iter(dataset))
# print(x.shape)
# print(y.shape)
# print (tp.shape)

# root_dir = "/datadrive/1.40625deg_equally_np"
# lister_train = list(dp.iter.FileLister(os.path.join(root_dir, "train")))
# dataset = ShuffleIterableDataset(
#     dataset=IndividualForecastPrecipDataIter(
#         dataset=ERA5ForecastMultiStepPrecip(
#             dataset=ERA5Npy(
#                 file_list=lister_train, variables=["t2m", "u10", "v10", "z", "tp"]
#             ),
#             pred_range=6,
#             pred_steps=4
#         ),
#         transforms=None,
#     ),
#     buffer_size=1000,
# )

# x, y, tp = next(iter(dataset))
# print(x.shape)
# print(y.shape)
# print (tp.shape)


# x = np.random.normal(size=(730, 128, 256))
# pred_range = 6
# pred_steps = 4
# interval = pred_range * pred_steps
# input = x[0:-interval:interval]
# print(input.shape)
# output = []
# for step in range(pred_steps):
#     start = (step + 1) * pred_range
#     end = (step - pred_steps + 1) * pred_range if step != pred_steps - 1 else -1
#     output.append(x[start:end:interval])

# output = np.stack(output, axis=1)
# print(output.shape)
