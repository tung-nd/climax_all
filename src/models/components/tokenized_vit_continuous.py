# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import numpy as np
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import torch
import torch.nn as nn
from src.models.components.tokenized_base import TokenizedBase
from src.utils.pos_embed import (
    get_1d_sincos_pos_embed_from_grid,
    get_1d_sincos_pos_embed_from_grid_pytorch,
    get_1d_sincos_pos_embed_from_grid_pytorch_stable)


class TokenizedViTContinuous(TokenizedBase):
    def __init__(
        self,
        climate_modeling=False,
        time_history=1,
        img_size=[128, 256],
        patch_size=16,
        drop_path=0.1,
        drop_rate=0.1,
        learn_pos_emb=False,
        default_vars=[
            "geopotential_1000",
            "geopotential_850",
            "geopotential_500",
            "geopotential_50",
            "relative_humidity_850",
            "relative_humidity_500",
            "u_component_of_wind_1000",
            "u_component_of_wind_850",
            "u_component_of_wind_500",
            "v_component_of_wind_1000",
            "v_component_of_wind_850",
            "v_component_of_wind_500",
            "temperature_850",
            "temperature_500",
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        out_vars=None,
        channel_agg="mean",
        embed_dim=1024,
        depth=24,
        decoder_depth=8,
        num_heads=16,
        mlp_ratio=4.0,
        init_mode="xavier",
        freeze_encoder=False,
    ):
        super().__init__(
            img_size,
            patch_size,
            drop_path,
            drop_rate,
            learn_pos_emb,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            init_mode,
            default_vars,
            channel_agg,
        )

        self.climate_modeling = climate_modeling
        self.freeze_encoder = freeze_encoder
        self.time_history = time_history
        if climate_modeling:
            assert out_vars is not None
            self.out_vars = out_vars
        else:
            self.out_vars = default_vars

        self.time_pos_embed = nn.Parameter(torch.zeros(1, time_history, embed_dim), requires_grad=learn_pos_emb)

        self.time_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.time_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)

        # self.lead_time_embed = nn.Sequential(
        #     nn.Linear(embed_dim, embed_dim),
        #     nn.GELU(),
        #     nn.Linear(embed_dim, embed_dim),
        # )
        self.lead_time_embed = nn.Linear(1, embed_dim)

        # --------------------------------------------------------------------------
        # Decoder: either a linear or non linear prediction head
        self.head = nn.Linear(embed_dim, img_size[0]*img_size[1])
        # self.head = nn.ModuleList()
        # for i in range(decoder_depth):
        #     self.head.append(nn.Linear(embed_dim, embed_dim))
        #     self.head.append(nn.GELU())
        # self.head.append(nn.Linear(embed_dim, len(self.out_vars) * patch_size**2))
        # self.head = nn.Sequential(*self.head)
        # --------------------------------------------------------------------------

        self.initialize_weights()

        if freeze_encoder:
            # self.token_embed.requires_grad_(False)
            # self.channel_embed.requires_grad_(False)
            # self.pos_embed.requires_grad_(False)
            # self.blocks.requires_grad_(False)
            for name, p in self.blocks.named_parameters():
                name = name.lower()
                if 'norm' in name:
                    continue
                else:
                    p.requires_grad_(False)
                

    def initialize_weights(self):
        # initialization
        super().initialize_weights()

        # time embedding
        time_pos_embed = get_1d_sincos_pos_embed_from_grid(self.time_pos_embed.shape[-1], np.arange(self.time_history))
        self.time_pos_embed.data.copy_(torch.from_numpy(time_pos_embed).float().unsqueeze(0))

    def unpatchify(self, x, h=None, w=None):
        """
        x: (B, L, patch_size**2 *3)
        imgs: (B, C, H, W)
        """
        p = self.patch_size
        c = len(self.out_vars)
        h = self.img_size[0] // p if h is None else h // p
        w = self.img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def aggregate_channel(self, x: torch.Tensor):
        """
        x: B, C, L, D
        """
        b, _, l, _ = x.shape
        x = torch.einsum("bcld->blcd", x)
        x = x.flatten(0, 1)  # BxL, C, D

        if self.channel_agg is not None:
            channel_query = self.channel_query.repeat_interleave(x.shape[0], dim=0)
            x, _ = self.channel_agg(channel_query, x, x)  # BxL, D
            x = x.squeeze()
        else:
            x = torch.mean(x, dim=1)  # BxL, D

        x = x.unflatten(dim=0, sizes=(b, l))  # B, L, D
        return x

    def emb_lead_time(self, lead_times: torch.Tensor, embed_dim, device):
        # lead_times: B, 1
        sinusoidal_emb = get_1d_sincos_pos_embed_from_grid_pytorch_stable(embed_dim, lead_times, dtype=lead_times.dtype).to(device)
        return self.lead_time_embed(sinusoidal_emb) # B, D

    def forward_encoder(self, x, lead_times, variables, region_info):
        """
        x: B, T, C, H, W
        """
        b, t, _, _, _ = x.shape
        x = x.flatten(0, 1)  # BxT, C, H, W
        
        # embed tokens

        embeds = []
        var_ids = self.get_channel_ids(variables)
        for i in range(len(var_ids)):
            id = var_ids[i]
            embeds.append(self.token_embeds[id](x[:, i : i + 1]))
        x = torch.stack(embeds, dim=1)  # BxT, C, L, D

        # add channel embedding, channel_embed: 1, C, D
        channel_embed = self.get_channel_emb(self.channel_embed, variables)
        x = x + channel_embed.unsqueeze(2) # BxT, C, L, D

        valid_patch_ids = region_info['patch_ids']
        x = x[:, :, valid_patch_ids, :]

        x = self.aggregate_channel(x)  # BxT, L, D

        x = x.unflatten(0, sizes=(b, t)) # B, T, L, D

        # add time and pos embed
        # pos emb: 1, L, D
        x = x + self.pos_embed[:, valid_patch_ids, :].unsqueeze(1)
        # time emb: 1, T, D
        x = x + self.time_pos_embed.unsqueeze(2)

        # add lead time embedding
        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1)) # B, D
        # lead_time_emb = self.emb_lead_time(lead_times, x.shape[-1], x.device)
        lead_time_emb = lead_time_emb.unsqueeze(1).unsqueeze(2) # B, 1, 1, D
        x = x + lead_time_emb

        x = x.flatten(0, 1)  # BxT, L, D

        x = self.pos_drop(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x) # BxT, L, D
        x = x.unflatten(0, sizes=(b, t)) # B, T, L, D

        x = x.mean(-2) # B, T, D
        time_query = self.time_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.time_agg(time_query, x, x)  # B, 1, D
        x = self.head(x)
        x = x.reshape(-1, 1, self.img_size[0], self.img_size[1]) # B, 1, H, W

        return x

    def forward_loss(self, y, pred, variables, out_variables, region_info, metric, lat):  # metric is a list
        """
        y: [B, C, H, W]
        pred: [B, L, C*p*p]
        """
        min_h, max_h = region_info['min_h'], region_info['max_h']
        min_w, max_w = region_info['min_w'], region_info['max_w']
        # pred = self.unpatchify(pred, max_h - min_h + 1, max_w - min_w + 1)  # B, C, H, W
        y = y[:, :, min_h:max_h+1, min_w:max_w+1]
        lat = lat[min_h:max_h+1]

        # if not climate_modeling (weather forecasting), then out varibles = a subset of in variables
        # only compute loss over the variables in out_variables
        if not self.climate_modeling:
            out_var_ids = self.get_channel_ids(out_variables)
            pred = pred[:, out_var_ids]

        return [m(pred, y, out_variables, lat) for m in metric], pred

    def forward(self, x, y, lead_times, variables, out_variables, region_info, metric, lat):
        # x: N, T, C, H, W
        # y: N, C, H, W
        # lead_times: N
        preds = self.forward_encoder(x, lead_times, variables, region_info)  # B, TxL, D
        # preds = self.head(embeddings)[:, -len(region_info['patch_ids']) :]
        loss, preds = self.forward_loss(y, preds, variables, out_variables, region_info, metric, lat)
        return loss, preds

    def predict(self, x, lead_times, variables, region_info):
        min_h, max_h = region_info['min_h'], region_info['max_h']
        min_w, max_w = region_info['min_w'], region_info['max_w']
        with torch.no_grad():
            pred = self.forward_encoder(x, lead_times, variables, region_info)
            # pred = self.head(embeddings)[:, -len(region_info['patch_ids']) :]
        return pred
        # return self.unpatchify(pred, max_h - min_h + 1, max_w - min_w + 1)
        # pred = pred.unflatten(dim=1, sizes=(-1, self.num_patches))  # [B, C, L, p*p]
        # pred = pred.flatten(0, 1)  # [BxC, L, p*p]
        # return self.unpatchify(pred, variables)

    def rollout(self, x, y, lead_times, variables, out_variables, region_info, steps, metric, transform, lat, log_steps, log_days, clim):
        # transform: get back to the original range
        if steps > 1:
            # can only rollout for more than 1 step if input variables and output variables are the same
            assert len(variables) == len(out_variables)
        preds = []
        for _ in range(steps):
            x = self.predict(x, lead_times, variables, region_info).unsqueeze(1)
            preds.append(x)
        preds = torch.concat(preds, dim=1)

        # if not climate_modeling (weather forecasting), then out varibles = a subset of in variables
        # only compute loss over the variables in out_variables
        if not self.climate_modeling:
            out_var_ids = self.get_channel_ids(out_variables)
            preds = preds[:, :, out_var_ids]

        # extract the specified region from y and lat
        min_h, max_h = region_info['min_h'], region_info['max_h']
        min_w, max_w = region_info['min_w'], region_info['max_w']
        y = y[:, :, min_h:max_h+1, min_w:max_w+1]
        lat = lat[min_h:max_h+1]

        if clim is not None and len(clim.shape) == 3:
            clim = clim[:, min_h:max_h+1, min_w:max_w+1]

        return [m(preds, y.unsqueeze(1), transform, out_variables, lat, log_steps, log_days, clim) for m in metric], preds


# from src.utils.metrics import mse

# model = TokenizedViT(depth=8).cuda()
# x, y = torch.randn(2, 3, 128, 256).cuda(), torch.randn(2, 3, 128, 256).cuda()
# loss, preds = model.forward(x, y, [mse])
# print (loss)
