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

from src.models.components.tokenized_base import TokenizedBase


class TokenizedViT(TokenizedBase):
    def __init__(
        self,
        img_size=[128, 256],
        patch_size=16,
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
        decoder_depth=8,
        num_heads=16,
        mlp_ratio=4.0,
        freeze_encoder=False,
    ):
        super().__init__(img_size, patch_size, learn_pos_emb, embed_dim, depth, num_heads, mlp_ratio, default_vars)

        self.freeze_encoder = freeze_encoder

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

    def forward_encoder(self, x, variables):
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
        channel_embed = self.get_channel_emb(self.channel_embed, variables)
        x = x + channel_embed.unsqueeze(2)
        # add pos embedding, pos_emb: 1, L, D
        x = x + self.pos_embed.unsqueeze(1)

        x = x.flatten(1, 2)  # B, CxL, D

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_loss(self, y, pred, variables, out_variables, metric, lat):  # metric is a list
        """
        y: [B, C, H, W]
        pred: [B, CxL, p*p]
        """
        pred = pred.unflatten(dim=1, sizes=(-1, self.num_patches))  # [B, C, L, p*p]
        pred = pred.flatten(0, 1)  # [BxC, L, p*p]
        pred = self.unpatchify(pred, variables)  # [B, C, H, W]

        # only compute loss over the variables in out_variables
        in_var_ids = self.get_channel_ids(variables).unsqueeze(-1)
        out_var_ids = self.get_channel_ids(out_variables).unsqueeze(-1)
        ids = (in_var_ids[:, None] == out_var_ids).all(-1).any(-1).nonzero().flatten()
        pred = pred[:, ids]

        return [m(pred, y, out_variables, lat) for m in metric], pred

    def forward(self, x, y, variables, out_variables, metric, lat):
        embeddings = self.forward_encoder(x, variables)  # B, CxL, D
        preds = self.head(embeddings)
        loss, preds = self.forward_loss(y, preds, variables, out_variables, metric, lat)
        return loss, preds

    def predict(self, x, variables):
        with torch.no_grad():
            embeddings = self.forward_encoder(x, variables)
            pred = self.head(embeddings)
        pred = pred.unflatten(dim=1, sizes=(-1, self.num_patches))  # [B, C, L, p*p]
        pred = pred.flatten(0, 1)  # [BxC, L, p*p]
        return self.unpatchify(pred, variables)

    def rollout(self, x, y, variables, out_variables, steps, metric, transform, lat, log_steps, log_days):
        # transform: get back to the original range
        if steps > 1:
            # can only rollout for more than 1 step if input variables and output variables are the same
            assert len(variables) == len(out_variables)
        preds = []
        for _ in range(steps):
            x = self.predict(x, variables)
            preds.append(x)
        preds = torch.stack(preds, dim=1)

        # only compute loss over the variables in out_variables
        in_var_ids = self.get_channel_ids(variables).unsqueeze(-1)
        out_var_ids = self.get_channel_ids(out_variables).unsqueeze(-1)
        ids = (in_var_ids[:, None] == out_var_ids).all(-1).any(-1).nonzero().flatten()
        preds = preds[:, :, ids]

        preds = transform(preds)
        y = transform(y)

        return [m(preds, y, out_variables, lat, log_steps, log_days) for m in metric], preds


# from src.utils.metrics import mse

# model = TokenizedViT(depth=8).cuda()
# x, y = torch.randn(2, 3, 128, 256).cuda(), torch.randn(2, 3, 128, 256).cuda()
# loss, preds = model.forward(x, y, [mse])
# print (loss)
