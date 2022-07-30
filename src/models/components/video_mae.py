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
from src.utils.pos_embed import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
)
from timm.models.vision_transformer import Block, PatchEmbed


class VideoMAE(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone."""

    def __init__(
        self,
        timesteps=8,
        img_size=[128, 256],
        patch_size=16,
        learn_pos_emb=False,
        in_vars=[
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        out_vars=None,
        mlp_ratio=4.0,
        norm_pix_loss=False,
    ):
        super().__init__()

        self.timesteps = timesteps
        self.img_size = img_size
        self.n_channels = len(in_vars)
        self.patch_size = patch_size

        out_vars = out_vars if out_vars is not None else in_vars
        self.in_vars = in_vars
        self.out_vars = out_vars

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, len(self.in_vars), embed_dim)
        num_patches = self.patch_embed.num_patches  # 128, for each timestep

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=learn_pos_emb
        )  # fixed sin-cos embedding
        self.time_pos_embed = nn.Parameter(
            torch.zeros(1, timesteps, embed_dim), requires_grad=learn_pos_emb
        )  # fixed sin-cos embedding

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
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=learn_pos_emb
        )  # fixed sin-cos embedding
        self.decoder_time_pos_embed = nn.Parameter(
            torch.zeros(1, timesteps, decoder_embed_dim), requires_grad=learn_pos_emb
        )  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=nn.LayerNorm,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * len(self.out_vars), bias=True
        )  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        self._init_pos_embed()

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_pos_embed(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        time_pos_embed = get_1d_sincos_pos_embed_from_grid(self.time_pos_embed.shape[-1], np.arange(self.timesteps))
        self.time_pos_embed.data.copy_(torch.from_numpy(time_pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        decoder_time_pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.decoder_time_pos_embed.shape[-1], np.arange(self.timesteps)
        )
        self.decoder_time_pos_embed.data.copy_(torch.from_numpy(decoder_time_pos_embed).float().unsqueeze(0))

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
        imgs: (N, C, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h = self.img_size[0] // p
        w = self.img_size[1] // p
        c = self.n_channels
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        c = self.n_channels
        h = self.img_size[0] // p
        w = self.img_size[1] // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """Perform per-sample random masking by per-sample shuffling.

        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # B, T x num_patches, embed_dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # x: B, T, C, H, W
        b, t, c, h, w = x.shape
        x = x.flatten(0, 1)  # BxT, C, H, W

        # embed patches
        x = self.patch_embed(x)  # BxT, num_patches, embed_dim
        x = x.unflatten(dim=0, sizes=(b, t))  # B, T, num_patches, embed_dim

        # space emb
        x = x + self.pos_embed.unsqueeze(1)  # 1, 1, num_patches, embed_dim
        # time emb
        x = x + self.time_pos_embed.unsqueeze(2)  # 1, T, 1, embed_dim

        x = x.flatten(1, 2)  # B, T x num_patches, embed_dim

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)  # B, T x num_patches x mask_ratio, embed_dim

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        x = torch.gather(
            x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle, B, T x num_patches, embed_dim

        x = x.unflatten(dim=1, sizes=(self.timesteps, -1))  # B, T, num_patches, embed_dim

        # space emb
        x = x + self.decoder_pos_embed.unsqueeze(1)  # 1, 1, num_patches, embed_dim
        # time emb
        x = x + self.decoder_time_pos_embed.unsqueeze(2)  # 1, timesteps, 1, embed_dim

        x = x.flatten(1, 2)  # B, T x num_patches, embed_dim

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x

    def forward_loss(self, imgs, pred, mask, reconstruct_all):
        """
        imgs: [N, T, 3, H, W]
        pred: [N, T x num_patches, p*p*3]
        mask: [N, T x num_patches], 0 is keep, 1 is remove,
        """
        b, t, c, h, w = imgs.shape

        img_mask = mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1])  # [N, T x num_patches, p*p*3]
        img_mask = img_mask.unflatten(dim=1, sizes=(self.timesteps, -1))  # [N, T, num_patches, p*p*3]
        img_mask = img_mask.flatten(0, 1)  # [N x T, p*p*3]

        pred = pred.unflatten(dim=1, sizes=(self.timesteps, -1))  # [N, T, num_patches, p*p*3]
        pred = pred.flatten(0, 1)  # [N x T, p*p*3]

        imgs = imgs.flatten(0, 1)

        img_pred = self.unpatchify(pred)  # [NxT, 3, H, W]
        img_mask = self.unpatchify(img_mask)[:, 0]  # [N, H, W], mask is the same for all variables

        loss = (img_pred - imgs) ** 2  # [NxT, 3, H, W]
        loss_dict = {}

        if reconstruct_all:
            for i, var in enumerate(self.out_vars):
                loss_dict[var] = torch.mean(loss[:, i])
            loss_dict["loss"] = torch.mean(torch.sum(loss, dim=1))
        else:
            for i, var in enumerate(self.out_vars):
                loss_dict[var] = (loss[:, i] * img_mask).sum() / img_mask.sum()
            loss_dict["loss"] = (loss.sum(dim=1) * img_mask).sum() / img_mask.sum()

        return loss_dict

    def forward(self, imgs, mask_ratio=0.75, reconstruct_all=False):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask, reconstruct_all)
        return loss, pred, mask

    def pred(self, imgs, mask_ratio):
        _, pred, mask = self.forward(imgs, mask_ratio)
        pred = pred.unflatten(dim=1, sizes=(self.timesteps, -1)).squeeze()
        mask = mask.unflatten(dim=1, sizes=(self.timesteps, -1)).squeeze()
        return pred, mask


# x = torch.randn((2, 8, 3, 128, 256)).cuda()
# x_reshaped = x.reshape(shape=(16, 3, 128, 256))
# patch_embed = nn.Conv2d(3, 768, kernel_size=16, stride=16).cuda()

# y = patch_embed(x_reshaped)
# print(y.shape)

# y1 = y.flatten(2).transpose(1, 2)
# y1 = y1.reshape((2, -1, 768))

# y2 = y.reshape((2, 8, 768, 8, 16)).flatten(3)
# y2 = torch.einsum("ntdm->ntmd", y2)
# # print(y2.shape)
# y2 = y2.reshape((2, -1, 768))

# print(y1 == y2)

# x = torch.randn((2, 8, 3, 128, 256)).cuda()
# model = VideoMAE(
#     8, (128, 256), 16, ["a", "b", "c"], 768, 8, 16, 512, 4, 16, None, 4, False
# ).cuda()
# latent, mask, ids_restore = model.forward_encoder(x, 0.9)
# pred = model.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
# print(latent.shape)
# print(mask.shape)
# print(ids_restore.shape)
# print(pred.shape)
# loss = model.forward_loss(x, pred, mask, reconstruct_all=True)
# print(loss)
