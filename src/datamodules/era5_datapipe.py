import numpy as np
import torch
import torchdata.datapipes as dp
import xarray as xr


class ERA5Npy(dp.iter.IterDataPipe):
    def __init__(self, dp: dp.iter.IterDataPipe, variables):
        super().__init__()
        self.dp = dp
        self.variables = variables

    def __iter__(self):
        for path in self.dp:
            data = np.load(path)
            yield {k: data[k] for k in self.variables}


class ERA5Zarr(dp.iter.IterDataPipe):
    def __init__(self, dp: dp.iter.IterDataPipe, variables):
        super().__init__()
        self.dp = dp
        self.variables = variables

    def __iter__(self):
        for path in self.dp:
            data = xr.open_zarr(path)
            yield {k: data[k].to_numpy() for k in self.variables}


### for pretraining
class ERA5(dp.iter.IterDataPipe):
    def __init__(self, dp: ERA5Npy):
        super().__init__()
        self.dp = dp

    def __iter__(self):
        for data in self.dp:
            np_data = np.concatenate([data[k] for k in data.keys()], axis=1)
            yield torch.from_numpy(np_data)


class IndividualDataIter(dp.iter.IterDataPipe):
    def __init__(self, dp: ERA5, transforms: torch.nn.Module):
        super().__init__()
        self.dp = dp
        self.transforms = transforms

    def __iter__(self):
        for inp in self.dp:
            for i in range(inp.shape[0]):
                # TODO: should we unsqueeze the first dimension?
                yield self.transforms(inp[i])


### for video pretraining
class ERA5Video(dp.iter.IterDataPipe):
    def __init__(self, dp: ERA5Npy, timesteps: int = 8):
        super().__init__()
        self.dp = dp
        self.timesteps = timesteps

    def __iter__(self):
        for data in self.dp:
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


### for finetuning
class ERA5Forecast(dp.iter.IterDataPipe):
    def __init__(self, dp: ERA5Npy, predict_range: int = 6) -> None:
        super().__init__()
        self.dp = dp
        self.predict_range = predict_range

    def __iter__(self):
        # TODO: this would not get us stuff across the years
        # i.e. where inputs are from previous years and output from next
        for data in self.dp:

            inputs = np.concatenate(
                [data[k][0 : -self.predict_range : self.predict_range] for k in data.keys()],
                axis=1,
            )
            outputs = np.concatenate(
                [data[k][self.predict_range :: self.predict_range] for k in data.keys()],
                axis=1,
            )
            yield torch.from_numpy(inputs), torch.from_numpy(outputs)


class IndividualForecastDataIter(dp.iter.IterDataPipe):
    def __init__(self, dp: ERA5Forecast, transforms: torch.nn.Module):
        super().__init__()
        self.dp = dp
        self.transforms = transforms

    def __iter__(self):
        for (inp, out) in self.dp:
            assert inp.shape[0] == out.shape[0]
            for i in range(inp.shape[0]):
                # TODO: should we unsqueeze the first dimension?
                yield self.transforms(inp[i]), self.transforms(out[i])


# def construct_video(x, timesteps):
#     # x: 8760, 3, 128, 256
#     x = x.unsqueeze(0).repeat_interleave(timesteps, dim=0)
#     for i in range(timesteps):
#         x[i] = torch.roll(x[i], shifts=-i, dims=0)
#     x = x[:, : -timesteps + 1]
#     return torch.transpose(x, dim0=0, dim1=1)


# steps = 4
# x = torch.randn(100, 3, 128, 256)
# v = construct_video(x, steps)
# # print(v.shape)
# i = 14
# print(x[i : i + steps] == v[i])
