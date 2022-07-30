# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import torch
import torch.nn as nn
from src.utils.pos_embed import get_2d_sincos_pos_embed
from timm.models.vision_transformer import Block, PatchEmbed


class VisionTransformerSimple(nn.Module):
    def __init__(
        self,
        img_size=[128, 256],
        patch_size=16,
        in_vars=["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind"],
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        out_vars=["2m_temperature_max", "10m_u_component_of_wind_max", "10m_v_component_of_wind_max"],
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size

        self.in_vars = in_vars
        self.out_vars = out_vars

        # --------------------------------------------------------------------------
        # ViT encoder specifics - exactly the same to MAE
        self.patch_embed = PatchEmbed(img_size, patch_size, len(self.in_vars), embed_dim)
        num_patches = self.patch_embed.num_patches  # 128

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm) for i in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # ViT encoder specifics - exactly the same to MAE
        self.head = nn.Linear(embed_dim, len(self.out_vars))
        # --------------------------------------------------------------------------

        self.initialize_weights()

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

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
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

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h = self.img_size[0] // p
        w = self.img_size[1] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, :, :]

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_pred(self, x):
        """
        x: [N, L, embed_dim]
        """
        x = torch.mean(x, dim=1)  # [N, embed_dim]
        return self.head(x)

    def forward_loss(self, y, pred):
        """
        y: [N, 3]
        pred: [N, 3]
        """
        loss = (pred - y) ** 2  # sum over channels, [N, 3]
        loss_dict = {}

        with torch.no_grad():
            for i, var in enumerate(self.out_vars):
                loss_dict[var] = torch.mean(loss[:, i])
        loss_dict["loss"] = torch.mean(torch.sum(loss, dim=1))

        return loss_dict

    def forward(self, x, y):
        embeddings = self.forward_encoder(x)
        preds = self.forward_pred(embeddings)
        loss = self.forward_loss(y, preds)
        return loss, preds

    def predict(self, x):
        with torch.no_grad():
            embeddings = self.forward_encoder(x)
            pred = self.forward_pred(embeddings)
        return pred


# model = VisionTransformer(depth=8).cuda()
# x, y = torch.randn(2, 3, 128, 256).cuda(), torch.randn(2, 3, 128, 256).cuda()
# loss, preds = model.forward(x, y)
