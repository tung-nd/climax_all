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
from src.utils.pos_embed import (get_1d_sincos_pos_embed_from_grid,
                                 get_2d_sincos_pos_embed)
from timm.models.vision_transformer import Block, PatchEmbed


class TokenizedVideoMAE(nn.Module):
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
        # TODO: input max_num_vars, and a map from var -> var id (0, 1, 2, ...)
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
        self.patch_size = patch_size

        out_vars = out_vars if out_vars is not None else in_vars
        self.in_vars = in_vars
        self.out_vars = out_vars

        # linear layer to embed each token, which is 1xpxp
        self.token_embed = PatchEmbed(img_size, patch_size, 1, embed_dim)
        self.num_patches = self.token_embed.num_patches  # assumed fixed because img_size is fixed
        # TODO: can generalize to different input resolutions

        # channel embedding and positional embedding
        self.channel_embed = nn.Parameter(torch.zeros(1, len(in_vars), embed_dim), requires_grad=learn_pos_emb)
        # TODO: len(in_vars) --> max_num_vars
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=learn_pos_emb)
        self.time_pos_embed = nn.Parameter(torch.zeros(1, timesteps, embed_dim), requires_grad=learn_pos_emb)

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
        # TODO: each channel has its own mask token

        self.decoder_channel_embed = nn.Parameter(
            torch.zeros(1, len(in_vars), decoder_embed_dim), requires_grad=learn_pos_emb
        )
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_embed_dim), requires_grad=learn_pos_emb
        )
        self.decoder_time_pos_embed = nn.Parameter(
            torch.zeros(1, timesteps, decoder_embed_dim), requires_grad=learn_pos_emb
        )

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

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # encoder pos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        channel_embed = get_1d_sincos_pos_embed_from_grid(self.channel_embed.shape[-1], np.arange(len(self.in_vars)))
        self.channel_embed.data.copy_(torch.from_numpy(channel_embed).float().unsqueeze(0))
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
            self.decoder_channel_embed.shape[-1], np.arange(len(self.in_vars))
        )
        self.decoder_channel_embed.data.copy_(torch.from_numpy(decoder_channel_embed).float().unsqueeze(0))
        decoder_time_pos_embed = get_1d_sincos_pos_embed_from_grid(
            self.decoder_time_pos_embed.shape[-1], np.arange(self.timesteps)
        )
        self.decoder_time_pos_embed.data.copy_(torch.from_numpy(decoder_time_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.token_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=0.02)

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

    def random_masking(self, x, mask_ratio):
        """Perform per-sample random masking by per-sample shuffling.

        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
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
        x = x + self.channel_embed.unsqueeze(1).unsqueeze(3)
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

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)  # B, T x C x L x mask_ratio, D

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle, B, TxCxL, D

        x = x.unflatten(dim=1, sizes=(self.timesteps, -1, self.num_patches))  # B, C, L, D

        # add channel embedding, channel_embed: 1, C, D
        x = x + self.decoder_channel_embed.unsqueeze(1).unsqueeze(3)
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

    def forward_loss(self, imgs, pred, mask, reconstruct_all):
        """
        imgs: [B, T, C, H, W]
        pred: [B, TxCxL, p*p]
        mask: [B, TxCxL], 0 is keep, 1 is remove,
        """
        b, t, c, h, w = imgs.shape

        imgs = imgs.flatten(0, 1) # BxT, C, H, W

        img_mask = mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1])  # [B, TxCxL, p*p]
        img_mask = img_mask.unflatten(dim=1, sizes=(self.timesteps, -1, self.num_patches))  # [B, T, C, L, p*p]
        img_mask = img_mask.flatten(0, 2)  # [BxTxC, L, p*p]

        pred = pred.unflatten(dim=1, sizes=(self.timesteps, -1, self.num_patches))  # [B, T, C, L, p*p]
        pred = pred.flatten(0, 2)  # [BxTxC, L, p*p]

        img_pred = self.unpatchify(pred)  # [BxT, C, H, W]
        img_mask = self.unpatchify(img_mask)[:, 0]  # [BxT, H, W]

        loss = (img_pred - imgs) ** 2  # [BxT, C, H, W]
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
        pred = self.forward_decoder(latent, ids_restore)  # [B, TxCxL, p*p]
        loss = self.forward_loss(imgs, pred, mask, reconstruct_all)
        return loss, pred, mask

    def pred(self, imgs, mask_ratio):
        _, pred, mask = self.forward(imgs, mask_ratio)
        return pred, mask


# model = TokenizedMAE(depth=4, decoder_depth=2).cuda()
# x = torch.randn(2, 3, 128, 256).cuda()
# loss, pred, mask = model(x)
# print (loss)
