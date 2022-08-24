import math
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset


class ERA5Npy(IterableDataset):
    def __init__(self, file_list, variables, out_variables, shuffle: bool = False) -> None:
        super().__init__()
        self.file_list = file_list
        self.variables = variables
        self.out_variables = out_variables if out_variables is not None else variables
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
            yield {k: data[k] for k in self.variables}, self.variables, self.out_variables


class ERA5(IterableDataset):
    def __init__(self, dataset: ERA5Npy) -> None:
        super().__init__()
        self.dataset = dataset

    def __iter__(self):
        for data, variables, _ in self.dataset:
            np_data = np.concatenate([data[k] for k in data.keys()], axis=1)
            yield torch.from_numpy(np_data), variables


class IndividualDataIter(IterableDataset):
    def __init__(self, dataset: ERA5, transforms: torch.nn.Module, output_transforms: torch.nn.Module) -> None:
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms

    def __iter__(self):
        for data, variables in self.dataset:
            for i in range(data.shape[0]):
                if self.transforms is not None:
                    yield self.transforms(data[i]), variables
                else:
                    yield data[i], variables


class ERA5Video(IterableDataset):
    def __init__(self, dataset: ERA5Npy, timesteps: int = 8) -> None:
        super().__init__()
        self.dataset = dataset
        self.timesteps = timesteps

    def __iter__(self):
        for data, variables, _ in self.dataset:
            np_data = np.concatenate([data[k] for k in data.keys()], axis=1)
            torch_data = torch.from_numpy(np_data)
            yield self.construct_video(torch_data), variables

    def construct_video(self, x):
        # x: 8760, 3, 128, 256
        x = x.unsqueeze(0).repeat_interleave(self.timesteps, dim=0)
        for i in range(self.timesteps):
            x[i] = torch.roll(x[i], shifts=-i, dims=0)
        end_idx = (-self.timesteps + 1) if self.timesteps != 1 else x.shape[1]
        x = x[:, :end_idx]
        return torch.transpose(x, dim0=0, dim1=1)


class ERA5Forecast(IterableDataset):
    def __init__(self, dataset: ERA5Npy, predict_range: int = 6, history: int = 3, interval: int = 6) -> None:
        super().__init__()
        self.dataset = dataset
        self.predict_range = predict_range
        self.history = history
        self.interval = interval

    def __iter__(self):
        # TODO: this would not get us stuff across the years
        # i.e. where inputs are from previous years and output from next
        for data, variables, out_variables in self.dataset:
            x = np.concatenate([data[k] for k in data.keys()], axis=1)
            x = torch.from_numpy(x)
            y = np.concatenate([data[k] for k in out_variables], axis=1)
            y = torch.from_numpy(y)

            inputs = x.unsqueeze(0).repeat_interleave(self.history, dim=0)
            for t in range(self.history):
                inputs[t] = inputs[t].roll(-t * self.interval, dims=0)

            last_idx = -((self.history - 1) * self.interval + self.predict_range)

            outputs = y.roll(last_idx, dims=0)

            inputs = inputs[:, :last_idx].transpose(0, 1)
            outputs = outputs[:last_idx]

            yield inputs, outputs, variables, out_variables


class ERA5ForecastMultiStep(IterableDataset):
    def __init__(
        self, dataset: ERA5Npy, pred_range: int = 6, history: int = 3, interval: int = 6, pred_steps: int = 4
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.pred_range = pred_range
        self.history = history
        self.interval = interval
        self.pred_steps = pred_steps

    def __iter__(self):
        # TODO: this would not get us stuff across the years
        # i.e. where inputs are from previous years and output from next
        for data, variables, out_variables in self.dataset:
            x = np.concatenate([data[k] for k in data.keys()], axis=1)
            x = torch.from_numpy(x)
            y = np.concatenate([data[k] for k in out_variables], axis=1)
            y = torch.from_numpy(y)

            inputs = x.unsqueeze(0).repeat_interleave(self.history, dim=0)
            for t in range(self.history):
                inputs[t] = inputs[t].roll(-t * self.interval, dims=0)

            outputs = y.unsqueeze(0).repeat_interleave(self.pred_steps, dim=0)
            start_idx = -((self.history - 1) * self.interval + self.pred_range)
            for t in range(self.pred_steps):
                outputs[t] = outputs[t].roll(-(start_idx + t * self.pred_range), dims=0)

            last_idx = -((self.history - 1) * self.interval + self.pred_steps * self.pred_range)

            inputs = inputs[:, :last_idx].transpose(0, 1)
            outputs = outputs[:, :last_idx].transpose(0, 1)

            yield inputs, outputs, variables, out_variables


class IndividualForecastDataIter(IterableDataset):
    def __init__(self, dataset: ERA5Forecast, transforms: torch.nn.Module, output_transforms: torch.nn.Module):
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms
        self.output_transforms = output_transforms

    def __iter__(self):
        for (inp, out, variables, out_variables) in self.dataset:
            assert inp.shape[0] == out.shape[0]
            for i in range(inp.shape[0]):
                # TODO: should we unsqueeze the first dimension?
                if self.transforms is not None:
                    yield self.transforms(inp[i]), self.output_transforms(out[i]), variables, out_variables
                else:
                    yield inp[i], out[i], variables, out_variables


class ERA5ForecastPrecip(IterableDataset):
    def __init__(self, dataset: ERA5Npy, predict_range: int = 6) -> None:
        super().__init__()
        self.dataset = dataset
        self.predict_range = predict_range

    def get_tp(self, data, len_data):
        tp = data["tp"][: len_data * self.predict_range]  # len_data*predict_range, 1, 128, 256
        tp = np.reshape(tp, (len_data, self.predict_range, *tp.shape[1:]))  # len_data, predict_range, 1, 128, 256
        tp = np.sum(tp, axis=1)  # sum over the predict range
        tp = np.log(1 + tp / 1e-5)  # log-transformation
        return tp

    def __iter__(self):
        # TODO: this would not get us stuff across the years
        # i.e. where inputs are from previous years and output from next
        for data, variables, out_variables in self.dataset:
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
            yield torch.from_numpy(inputs), torch.from_numpy(outputs), torch.from_numpy(tp), variables, out_variables


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
        tp = np.log(1 + tp / 1e-5)  # log-transformation
        return tp

    def __iter__(self):
        # TODO: this would not get us stuff across the years
        # i.e. where inputs are from previous years and output from next
        pred_range = self.pred_range
        pred_steps = self.pred_steps
        interval = pred_range * pred_steps
        for data, variables, out_variables in self.dataset:
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
            yield torch.from_numpy(inputs), torch.from_numpy(outputs), torch.from_numpy(tp), variables, out_variables


class IndividualForecastPrecipDataIter(IterableDataset):
    def __init__(self, dataset: ERA5ForecastPrecip, transforms: torch.nn.Module, output_transforms: torch.nn.Module):
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms
        self.output_transforms = output_transforms

    def __iter__(self):
        for (inp, out, tp, variables, out_variables) in self.dataset:
            assert inp.shape[0] == out.shape[0] and inp.shape[0] == tp.shape[0]
            for i in range(inp.shape[0]):
                # TODO: should we unsqueeze the first dimension?
                if self.transforms is not None:
                    yield self.transforms(inp[i]), self.output_transforms(out[i]), tp[i]
                else:
                    yield inp[i], out[i], tp[i], variables, out_variables


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


# x = torch.randn((10, 2))
# pred_range = 2
# history = 3
# interval = 1
# pred_steps = 2

# inputs = x.unsqueeze(0).repeat_interleave(history, dim=0)
# for t in range(history):
#     inputs[t] = inputs[t].roll(-t*interval, dims=0)

# # forecast training dataset
# last_idx = -((history - 1) * interval + pred_range)

# outputs = x.roll(last_idx, dims=0)

# inputs = inputs[:, :last_idx].transpose(0, 1)
# outputs = outputs[:last_idx]

# forecast validation dataset
# outputs = x.unsqueeze(0).repeat_interleave(pred_steps, dim=0)
# start_idx = (history-1) * interval + pred_range
# for t in range(pred_steps):
#     outputs[t] = outputs[t].roll(-(start_idx + t*pred_range), dims=0)

# last_idx = - ((history-1) * interval + pred_steps * pred_range)

# inputs = inputs[:, :last_idx].transpose(0, 1)
# outputs = outputs[:, :last_idx].transpose(0, 1)

# for i in range(inputs.shape[0]):
#     print ('x', x)
#     print (i)
#     print ('in', inputs[i])
#     print ('out', outputs[i])
#     print ('=' * 20)

# import os

# import torchdata.datapipes as dp

# root_dir = "/datadrive/datasets/5.625deg_equally_np/"
# lister_train = list(dp.iter.FileLister(os.path.join(root_dir, "train")))
# dataset = ShuffleIterableDataset(
#     dataset=IndividualForecastDataIter(
#         dataset=ERA5Forecast(
#             dataset=ERA5Npy(
#                 file_list=lister_train,
#                 variables=["t2m", "u10", "v10", "z_500", "t_850"],
#                 out_variables=["z_500", "t_850"],
#             ),
#             predict_range=6,
#             history=3,
#             interval=6
#         ),
#         transforms=None,
#         output_transforms=None
#     ),
#     buffer_size=1000,
# )

# x, y, variables, out_variables = next(iter(dataset))
# print(x.shape)
# print(y.shape)
# print (variables)
# print (out_variables)

# root_dir = "/datadrive/datasets/5.625deg_equally_np/"
# lister_train = list(dp.iter.FileLister(os.path.join(root_dir, "train")))
# dataset = ShuffleIterableDataset(
#     dataset=IndividualForecastDataIter(
#         dataset=ERA5ForecastMultiStep(
#             dataset=ERA5Npy(
#                 file_list=lister_train,
#                 variables=["t2m", "u10", "v10", "z_500", "t_850"],
#                 out_variables=None
#             ),
#             pred_range=6,
#             history=3,
#             interval=6,
#             pred_steps=4,
#         ),
#         transforms=None,
#         output_transforms=None
#     ),
#     buffer_size=1000,
# )

# x, y, variables, out_variables = next(iter(dataset))
# print(x.shape)
# print(y.shape)
# print (variables)
# print (out_variables)

# root_dir = "/datadrive/5.625deg_equally_np/"
# lister_train = list(dp.iter.FileLister(os.path.join(root_dir, "train")))
# dataset = ShuffleIterableDataset(
#     dataset=IndividualDataIter(
#         dataset=ERA5(
#             dataset=ERA5Npy(
#                 file_list=lister_train, variables=["t2m", "u10", "v10", "z"]
#             ),
#         ),
#         transforms=None,
#     ),
#     buffer_size=1000,
# )

# x, variables = next(iter(dataset))
# print(x.shape)
# print (variables)


# root_dir = "/datadrive/5.625deg_equally_np/"
# lister_train = list(dp.iter.FileLister(os.path.join(root_dir, "train")))
# dataset = ShuffleIterableDataset(
#     dataset=IndividualDataIter(
#         dataset=ERA5Video(
#             dataset=ERA5Npy(
#                 file_list=lister_train, variables=["t2m", "u10", "v10", "z"]
#             ), timesteps=4
#         ),
#         transforms=None,
#     ),
#     buffer_size=1000,
# )

# x, variables = next(iter(dataset))
# print(x.shape)
# print (variables)
