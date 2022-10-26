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
from src.utils.pos_embed import get_1d_sincos_pos_embed_from_grid


class TokenizedViT(TokenizedBase):
    def __init__(
        self,
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

        self.freeze_encoder = freeze_encoder
        self.time_history = time_history
        # self.out_vars = out_vars if out_vars is not None else default_vars
        self.out_vars = default_vars

        self.time_pos_embed = nn.Parameter(torch.zeros(1, time_history, embed_dim), requires_grad=learn_pos_emb)

        # --------------------------------------------------------------------------
        # Decoder: either a linear or non linear prediction head
        self.head = nn.ModuleList()
        for i in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(embed_dim, len(self.out_vars) * patch_size**2))
        self.head = nn.Sequential(*self.head)
        # --------------------------------------------------------------------------

        self.initialize_weights()

        if freeze_encoder:
            self.token_embed.requires_grad_(False)
            self.channel_embed.requires_grad_(False)
            self.pos_embed.requires_grad_(False)
            self.blocks.requires_grad_(False)

    def initialize_weights(self):
        # initialization
        super().initialize_weights()

        # time embedding
        time_pos_embed = get_1d_sincos_pos_embed_from_grid(self.time_pos_embed.shape[-1], np.arange(self.time_history))
        self.time_pos_embed.data.copy_(torch.from_numpy(time_pos_embed).float().unsqueeze(0))

    def unpatchify(self, x):
        """
        x: (B, L, patch_size**2 *3)
        imgs: (B, C, H, W)
        """
        p = self.patch_size
        c = len(self.out_vars)
        h = self.img_size[0] // p
        w = self.img_size[1] // p
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

    def forward_encoder(self, x, variables):
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

        x = self.aggregate_channel(x)  # BxT, L, D

        x = x.unflatten(0, sizes=(b, t)) # B, T, L, D

        # add time and pos embed
        # pos emb: 1, L, D
        x = x + self.pos_embed.unsqueeze(1)
        # time emb: 1, T, D
        x = x + self.time_pos_embed.unsqueeze(2)

        x = x.flatten(1, 2)  # B, TxL, D

        x = self.pos_drop(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_loss(self, y, pred, variables, out_variables, metric, lat):  # metric is a list
        """
        y: [B, C, H, W]
        pred: [B, L, C*p*p]
        """
        pred = self.unpatchify(pred)  # B, C, H, W

        # if len(self.out_vars) == len(self.default_vars):
        # only compute loss over the variables in out_variables
        out_var_ids = self.get_channel_ids(out_variables)
        pred = pred[:, out_var_ids]

        return [m(pred, y, out_variables, lat) for m in metric], pred

    def forward(self, x, y, variables, out_variables, metric, lat):
        embeddings = self.forward_encoder(x, variables)  # B, TxL, D
        preds = self.head(embeddings)[:, -self.num_patches :]
        loss, preds = self.forward_loss(y, preds, variables, out_variables, metric, lat)
        return loss, preds

    def predict(self, x, variables):
        with torch.no_grad():
            embeddings = self.forward_encoder(x, variables)
            pred = self.head(embeddings)[:, -self.num_patches :]
        return self.unpatchify(pred)
        # pred = pred.unflatten(dim=1, sizes=(-1, self.num_patches))  # [B, C, L, p*p]
        # pred = pred.flatten(0, 1)  # [BxC, L, p*p]
        # return self.unpatchify(pred, variables)

    def rollout(self, x, y, variables, out_variables, steps, metric, transform, lat, log_steps, log_days):
        # transform: get back to the original range
        if steps > 1:
            # can only rollout for more than 1 step if input variables and output variables are the same
            assert len(variables) == len(out_variables)
        preds = []
        for _ in range(steps):
            x = self.predict(x, variables).unsqueeze(1)
            preds.append(x)
        preds = torch.concat(preds, dim=1)

        # if len(self.out_vars) == len(self.default_vars):
        # only compute loss over the variables in out_variables
        out_var_ids = self.get_channel_ids(out_variables)
        preds = preds[:, :, out_var_ids]

        return [m(preds, y, transform, out_variables, lat, log_steps, log_days) for m in metric], preds


# from src.utils.metrics import mse

# model = TokenizedViT(depth=8).cuda()
# x, y = torch.randn(2, 3, 128, 256).cuda(), torch.randn(2, 3, 128, 256).cuda()
# loss, preds = model.forward(x, y, [mse])
# print (loss)
