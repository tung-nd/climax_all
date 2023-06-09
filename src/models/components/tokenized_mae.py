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
from timm.models.vision_transformer import Block

from src.models.components.tokenized_base import TokenizedBase
from src.utils.pos_embed import get_2d_sincos_pos_embed


class TokenizedMAE(TokenizedBase):
    """Masked Autoencoder with VisionTransformer backbone."""

    def __init__(
        self,
        img_size=[128, 256],
        patch_size=16,
        drop_path=0.0,
        drop_rate=0.0,
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
        channel_agg="mean",
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

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # TODO: each channel has its own mask token

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_embed_dim), requires_grad=learn_pos_emb
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
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, len(default_vars) * patch_size**2, bias=True
        )  # decoder to token
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        super().initialize_weights()

        # initialize decoder components
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.img_size[0] / self.patch_size),
            int(self.img_size[1] / self.patch_size),
            cls_token=False,
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=0.02)

    def unpatchify(self, x):
        """
        x: (B, L, patch_size**2 *3)
        imgs: (B, C, H, W)
        """
        p = self.patch_size
        c = len(self.default_vars)
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

    def forward_encoder(self, x, variables, mask_ratio):
        """
        x: B, C, H, W
        """
        # embed tokens
        embeds = []
        var_ids = self.get_channel_ids(variables)
        for i in range(len(var_ids)):
            id = var_ids[i]
            embeds.append(self.token_embeds[id](x[:, i : i + 1]))
        x = torch.stack(embeds, dim=1)  # B, C, L, D

        # add channel embedding, channel_embed: 1, C, D
        channel_embed = self.get_channel_emb(self.channel_embed, variables)
        x = x + channel_embed.unsqueeze(2)

        x = self.aggregate_channel(x)  # B, L, D

        # add pos embedding, pos_emb: 1, L, D
        x = x + self.pos_embed

        x = self.pos_drop(x)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, variables, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)  # B, L x mask_ratio, D

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle, B, L, D

        # add pos embedding, pos_emb: 1, L, D
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x

    def forward_loss(self, imgs, pred, variables, metric, lat, mask, reconstruct_all):
        """
        imgs: [B, C, H, W]
        pred: [B, CxL, p*p]
        mask: [B, CxL], 0 is keep, 1 is remove,
        """

        img_mask = mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1])  # [B, L, C*p*p]
        img_mask = self.unpatchify(img_mask)[:, 0]  # [B, H, W]

        img_pred = self.unpatchify(pred)  # [B, C, H, W]
        var_ids = self.get_channel_ids(variables)
        img_pred = img_pred[:, var_ids]  # only compute loss over present variables

        if metric is None:
            return None, img_pred, img_mask

        if reconstruct_all:
            loss_dict = metric(img_pred, imgs, variables, lat, None)
        else:
            loss_dict = metric(img_pred, imgs, variables, lat, img_mask)

        return loss_dict, img_pred, img_mask

    def forward(self, imgs, variables, metric, lat, mask_ratio=0.75, reconstruct_all=False):
        latent, mask, ids_restore = self.forward_encoder(imgs, variables, mask_ratio)
        pred = self.forward_decoder(latent, variables, ids_restore)  # [B, L, p*p]
        loss, pred, mask = self.forward_loss(imgs, pred, variables, metric, lat, mask, reconstruct_all)
        return loss, pred, mask

    def pred(self, imgs, variables, mask_ratio):
        _, pred, mask = self.forward(imgs, variables, None, None, mask_ratio)
        return pred, mask


# model = TokenizedMAE(depth=4, decoder_depth=2).cuda()
# x = torch.randn(2, 3, 128, 256).cuda()
# loss, pred, mask = model(x)
# print (loss)
