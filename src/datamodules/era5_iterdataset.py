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
            per_worker = int(
                math.ceil(len(self.file_list) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.file_list))

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
