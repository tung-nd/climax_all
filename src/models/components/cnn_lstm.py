import torch


class CNNLSTM(torch.nn.Module):
    def __init__(self, in_channels, img_size, patch_size):
        super().__init__()

        self.in_channels = in_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.climate_modeling = True

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels=20, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AvgPool2d(kernel_size=2),
        )
        # a global average pooling layer after this
        
        self.lstm = torch.nn.LSTM(
            input_size=20,
            hidden_size=25,
            batch_first=True
        )

        self.dense = torch.nn.Linear(25, img_size[0]*img_size[1])

    def predict(self, x: torch.Tensor, region_info):
        n, t, _, _, _ = x.shape

        if region_info is not None:
            min_h, max_h = region_info['min_h'], region_info['max_h']
            min_w, max_w = region_info['min_w'], region_info['max_w']
            x = x[:, :, :, min_h:max_h+1, min_w:max_w+1]

        x = self.cnn(x.flatten(0, 1))
        x = x.unflatten(dim=0, sizes=(n, t))
        x = torch.mean(x, dim=(-2,-1)) # N, T, C
        x, _ = self.lstm(x)
        x = x[:, -1] # N, D
        x = self.dense(x) # N, 96x144
        x = x.reshape(-1, 1, self.img_size[0], self.img_size[1])
        
        return x

    def forward(self, x: torch.Tensor, y: torch.Tensor, out_variables, region_info, metric, lat):
        pred = self.predict(x, region_info)
        min_h, max_h = region_info['min_h'], region_info['max_h']
        min_w, max_w = region_info['min_w'], region_info['max_w']
        y = y[:, :, min_h:max_h+1, min_w:max_w+1]
        lat = lat[min_h:max_h+1]
        return [m(pred, y, out_variables, lat) for m in metric], x

    def rollout(self, x, y, variables, out_variables, region_info, steps, metric, transform, lat, log_steps, log_days, clim):
        if steps > 1:
            assert len(variables) == len(out_variables)

        preds = []
        for _ in range(steps):
            x = self.predict(x, region_info)
            preds.append(x)
        preds = torch.stack(preds, dim=1)

        # extract the specified region from y and lat
        min_h, max_h = region_info['min_h'], region_info['max_h']
        min_w, max_w = region_info['min_w'], region_info['max_w']
        y = y[:, :, min_h:max_h+1, min_w:max_w+1]
        lat = lat[min_h:max_h+1]

        if clim is not None and len(clim.shape) == 3:
            clim = clim[:, min_h:max_h+1, min_w:max_w+1]

        return [m(preds, y.unsqueeze(1), transform, out_variables, lat, log_steps, log_days, clim) for m in metric], preds

# model = CNNLSTM(in_channels=4, img_size=[96, 144])
# x = torch.randn(16, 10, 4, 96, 144)
# pred = model.predict(x, None)
# print (pred.shape)
