# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------


import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, PatchEmbed

from src.utils.pos_embed import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
)


class TokenizedViT(nn.Module):
    def __init__(
        self,
        img_size=[128, 256],
        patch_size=16,
        learn_pos_emb=False,
        in_vars=[
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        # TODO: input max_num_vars, and a map from var -> var id (0, 1, 2, ...)
        embed_dim=1024,
        depth=24,
        decoder_depth=8,
        num_heads=16,
        mlp_ratio=4.0,
        out_vars=None,
        freeze_encoder=False,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size

        out_vars = out_vars if out_vars is not None else in_vars
        self.in_vars = in_vars
        self.out_vars = out_vars

        self.freeze_encoder = freeze_encoder

        # linear layer to embed each token, which is 1xpxp
        self.token_embed = PatchEmbed(img_size, patch_size, 1, embed_dim)
        self.num_patches = self.token_embed.num_patches  # assumed fixed because img_size is fixed
        # TODO: can generalize to different input resolutions

        # channel embedding and positional embedding
        self.channel_embed = nn.Parameter(torch.zeros(1, len(in_vars), embed_dim), requires_grad=learn_pos_emb)
        # TODO: len(in_vars) --> max_num_vars
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=learn_pos_emb)

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Decoder: either a linear or non linear prediction head
        self.head = nn.ModuleList()
        for i in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(embed_dim, patch_size**2))
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
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        channel_embed = get_1d_sincos_pos_embed_from_grid(self.channel_embed.shape[-1], np.arange(len(self.in_vars)))
        self.channel_embed.data.copy_(torch.from_numpy(channel_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.token_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x):
        """
        x: (BxC, L, patch_size**2)
        return: (B, C, H, W)
        """
        p = self.patch_size
        h = self.img_size[0] // p
        w = self.img_size[1] // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p))
        x = torch.einsum("nhwpq->nhpwq", x)
        imgs = x.reshape(shape=(x.shape[0], h * p, w * p))  # (BxC, H, W)
        imgs = imgs.unflatten(dim=0, sizes=(-1, len(self.in_vars)))  # (B, C, H, W)
        return imgs

    def forward_encoder(self, x):
        """
        x: B, C, H, W
        """
        # embed tokens
        b, c, _, _ = x.shape
        x = x.flatten(0, 1)  # BxC, H, W
        x = x.unsqueeze(dim=1)  # BxC, 1, H, W
        x = self.token_embed(x)  # BxC, L, D
        x = x.unflatten(dim=0, sizes=(b, c))  # B, C, L, D

        # add channel embedding, channel_embed: 1, C, D
        x = x + self.channel_embed.unsqueeze(2)
        # add pos embedding, pos_emb: 1, L, D
        x = x + self.pos_embed.unsqueeze(1)

        x = x.flatten(1, 2)  # B, CxL, D

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_loss(self, y, pred, metric, lat):  # metric is a list
        """
        y: [B, C, H, W]
        pred: [B, CxL, p*p]
        """
        pred = pred.unflatten(dim=1, sizes=(-1, self.num_patches))  # [B, C, L, p*p]
        pred = pred.flatten(0, 1)  # [BxC, L, p*p]
        pred = self.unpatchify(pred)  # [B, C, H, W]
        return [m(pred, y, self.out_vars, lat) for m in metric], pred

    def forward(self, x, y, metric, lat):
        embeddings = self.forward_encoder(x)  # B, CxL, D
        preds = self.head(embeddings)
        loss, preds = self.forward_loss(y, preds, metric, lat)
        return loss, preds

    def predict(self, x):
        with torch.no_grad():
            embeddings = self.forward_encoder(x)
            pred = self.head(embeddings)
        pred = pred.unflatten(dim=1, sizes=(-1, self.num_patches))  # [B, C, L, p*p]
        pred = pred.flatten(0, 1)  # [BxC, L, p*p]
        return self.unpatchify(pred)

    def rollout(self, x, y, steps, metric, transform, lat, log_steps, log_days):
        # transform: get back to the original range
        preds = []
        for _ in range(steps):
            x = self.predict(x)
            preds.append(x)
        preds = torch.stack(preds, dim=1)
        preds = transform(preds)
        y = transform(y)
        return [m(preds, y, self.out_vars, lat, log_steps, log_days) for m in metric], preds


# from src.utils.metrics import mse

# model = TokenizedViT(depth=8).cuda()
# x, y = torch.randn(2, 3, 128, 256).cuda(), torch.randn(2, 3, 128, 256).cuda()
# loss, preds = model.forward(x, y, [mse])
# print (loss)
