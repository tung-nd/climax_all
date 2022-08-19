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
from timm.models.vision_transformer import Block

from src.models.components.tokenized_base import TokenizedBase
from src.utils.pos_embed import (
    get_1d_sincos_pos_embed_from_grid,
    get_2d_sincos_pos_embed,
)


class TokenizedVideoMAE(TokenizedBase):
    """Masked Autoencoder with VisionTransformer backbone."""

    def __init__(
        self,
        timesteps=8,
        img_size=[128, 256],
        patch_size=16,
        drop_path=0.0,
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
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        init_mode="xavier",
    ):
        super().__init__(
            img_size,
            patch_size,
            drop_path,
            learn_pos_emb,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            init_mode,
            default_vars,
        )

        self.timesteps = timesteps

        # time embedding
        self.time_pos_embed = nn.Parameter(torch.zeros(1, timesteps, embed_dim), requires_grad=learn_pos_emb)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # TODO: each channel has its own mask token

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_embed_dim), requires_grad=learn_pos_emb
        )
        self.decoder_time_pos_embed = nn.Parameter(
            torch.zeros(1, timesteps, decoder_embed_dim), requires_grad=learn_pos_emb
        )
        self.decoder_channel_embed, _ = self.create_channel_embedding(learn_pos_emb, decoder_embed_dim)

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
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2, bias=True)  # decoder to token
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        super().initialize_weights()

        # time embedding
        time_pos_embed = get_1d_sincos_pos_embed_from_grid(self.time_pos_embed.shape[-1], np.arange(self.timesteps))
        self.time_pos_embed.data.copy_(torch.from_numpy(time_pos_embed).float().unsqueeze(0))

        # decoder pos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        decoder_channel_embed = get_1d_sincos_pos_embed_from_grid(
            self.decoder_channel_embed.shape[-1], np.arange(len(self.default_vars))
        )
        self.decoder_channel_embed.data.copy_(torch.from_numpy(decoder_channel_embed).float().unsqueeze(0))
        decoder_time_pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.decoder_time_pos_embed.shape[-1], np.arange(self.timesteps)
        )
        self.decoder_time_pos_embed.data.copy_(torch.from_numpy(decoder_time_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=0.02)

    def unpatchify(self, x, variables):
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
        imgs = imgs.unflatten(dim=0, sizes=(-1, len(variables)))  # (B, C, H, W)
        return imgs

    def forward_encoder(self, x, variables, mask_ratio):
        """
        x: B, T, C, H, W
        """
        # embed tokens
        b, t, c, _, _ = x.shape
        x = x.flatten(0, 2)  # BxTxC, H, W
        x = x.unsqueeze(dim=1)  # BxTxC, 1, H, W
        x = self.token_embed(x)  # BxTxC, L, D
        x = x.unflatten(dim=0, sizes=(b, t, c))  # B, T, C, L, D

        # add channel embedding, channel_embed: 1, C, D
        channel_embed = self.get_channel_emb(self.channel_embed, variables)
        x = x + channel_embed.unsqueeze(1).unsqueeze(3)
        # add pos embedding, pos_emb: 1, L, D
        x = x + self.pos_embed.unsqueeze(1).unsqueeze(2)
        # add time pos embedding, time_emb: 1, T, D
        x = x + self.time_pos_embed.unsqueeze(2).unsqueeze(3)

        x = x.flatten(1, 3)  # B, TxCxL, D

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, variables, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)  # B, T x C x L x mask_ratio, D

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle, B, TxCxL, D

        x = x.unflatten(dim=1, sizes=(self.timesteps, -1, self.num_patches))  # B, C, L, D

        # add channel embedding, channel_embed: 1, C, D
        decoder_channel_embed = self.get_channel_emb(self.decoder_channel_embed, variables)
        x = x + decoder_channel_embed.unsqueeze(1).unsqueeze(3)
        # add pos embedding, pos_emb: 1, L, D
        x = x + self.decoder_pos_embed.unsqueeze(1).unsqueeze(2)
        # add time pos embedding, time_emb: 1, T, D
        x = x + self.decoder_time_pos_embed.unsqueeze(2).unsqueeze(3)

        x = x.flatten(1, 3)  # B, TxCxL, D

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x

    def forward_loss(self, imgs, pred, variables, metric, lat, mask, reconstruct_all):
        """
        imgs: [B, T, C, H, W]
        pred: [B, TxCxL, p*p]
        mask: [B, TxCxL], 0 is keep, 1 is remove,
        """
        b, t, c, h, w = imgs.shape

        imgs = imgs.flatten(0, 1)  # BxT, C, H, W

        img_mask = mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1])  # [B, TxCxL, p*p]
        img_mask = img_mask.unflatten(dim=1, sizes=(self.timesteps, -1, self.num_patches))  # [B, T, C, L, p*p]
        img_mask = img_mask.flatten(0, 2)  # [BxTxC, L, p*p]

        pred = pred.unflatten(dim=1, sizes=(self.timesteps, -1, self.num_patches))  # [B, T, C, L, p*p]
        pred = pred.flatten(0, 2)  # [BxTxC, L, p*p]

        img_pred = self.unpatchify(pred, variables)  # [BxT, C, H, W]
        img_mask = self.unpatchify(img_mask, variables)[:, 0]  # [BxT, H, W]

        if metric is None:
            return None, img_pred, img_mask

        if reconstruct_all:
            loss_dict = metric(img_pred, imgs, variables, lat, None)
        else:
            loss_dict = metric(img_pred, imgs, variables, lat, img_mask)

        return loss_dict, img_pred, img_mask

    def forward(self, imgs, variables, metric, lat, mask_ratio=0.75, reconstruct_all=False):
        latent, mask, ids_restore = self.forward_encoder(imgs, variables, mask_ratio)
        pred = self.forward_decoder(latent, variables, ids_restore)  # [B, TxCxL, p*p]
        loss, pred, mask = self.forward_loss(imgs, pred, variables, metric, lat, mask, reconstruct_all)
        return loss, pred, mask

    def pred(self, imgs, variables, mask_ratio):
        _, pred, mask = self.forward(imgs, variables, None, None, mask_ratio)
        return pred, mask


# model = TokenizedMAE(depth=4, decoder_depth=2).cuda()
# x = torch.randn(2, 3, 128, 256).cuda()
# loss, pred, mask = model(x)
# print (loss)
