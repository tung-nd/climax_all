import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from src.utils.pos_embed import get_2d_sincos_pos_embed
from timm.models.layers import DropPath
from timm.models.vision_transformer import PatchEmbed, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AFNO2D(nn.Module):
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)


        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes]  = (
            torch.einsum('...bi,bio->...bo', o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes]  = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, H, W // 2 + 1, C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1,2), norm="ortho")
        x = x.type(dtype)

        return x + bias


class AFNOBlock(nn.Module):
    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            double_skip=True,
            num_blocks=8,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = AFNO2D(dim, num_blocks, sparsity_threshold, hard_thresholding_fraction) 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x


class AFNO(nn.Module):
    def __init__(
        self,
        img_size=[128, 256],
        patch_size=16,
        drop_path=0.1,
        learn_pos_emb=False,
        in_vars=[
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
        ],
        embed_dim=1024,
        depth=24,
        decoder_depth=8,
        num_blocks=16,
        mlp_ratio=4.0,
        out_vars=None,
        init_mode="xavier",
        sparsity_threshold=0.01,
        hard_thresholding_fraction=1.0,
    ):
        super().__init__()

        self.img_size = img_size
        self.n_channels = len(in_vars)
        self.patch_size = patch_size
        self.num_blocks = num_blocks
        self.init_mode = init_mode

        out_vars = out_vars if out_vars is not None else in_vars
        self.in_vars = in_vars
        self.out_vars = out_vars

        self.h = img_size[0] // patch_size
        self.w = img_size[1] // patch_size
        self.embed_dim = embed_dim

        # --------------------------------------------------------------------------
        # ViT encoder specifics - exactly the same to MAE
        self.patch_embed = PatchEmbed(img_size, patch_size, len(self.in_vars), embed_dim)
        num_patches = self.patch_embed.num_patches  # 128

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=learn_pos_emb
        )  # fixed sin-cos embedding

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                AFNOBlock(
                    dim=embed_dim,
                    mlp_ratio=mlp_ratio,
                    drop_path=dpr[i],
                    norm_layer=nn.LayerNorm,
                    num_blocks=self.num_blocks,
                    sparsity_threshold=sparsity_threshold,
                    hard_thresholding_fraction=hard_thresholding_fraction
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # ViT encoder specifics - exactly the same to MAE
        self.head = nn.ModuleList()
        for i in range(decoder_depth):
            self.head.append(nn.Linear(embed_dim, embed_dim))
            self.head.append(nn.GELU())
        self.head.append(nn.Linear(embed_dim, len(self.out_vars) * patch_size**2))
        self.head = nn.Sequential(*self.head)
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
        if self.init_mode == "xavier":
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        else:
            trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.init_mode == "xavier":
                torch.nn.init.xavier_uniform_(m.weight)
            else:
                trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x):
        return rearrange(
            x,
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.patch_size,
            p2=self.patch_size,
            h=self.img_size[0] // self.patch_size,
            w=self.img_size[1] // self.patch_size,
        )

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, :, :]

        x = x.reshape(x.shape[0], self.h, self.w, self.embed_dim)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_loss(self, y, pred, variables, out_variables, metric, lat):  # metric is a list
        """
        y: [N, 3, H, W]
        pred: [N, L, p*p*3]
        """
        pred = self.unpatchify(pred)
        return [m(pred, y, out_variables, lat) for m in metric], pred

    def forward(self, x, y, variables, out_variables, metric, lat):
        embeddings = self.forward_encoder(x)
        preds = self.head(embeddings)
        loss, preds = self.forward_loss(y, preds, variables, out_variables, metric, lat)
        return loss, preds

    def predict(self, x, variables):
        with torch.no_grad():
            embeddings = self.forward_encoder(x)
            pred = self.head(embeddings)
        return self.unpatchify(pred)

    def rollout(self, x, y, variables, out_variables, steps, metric, transform, lat, log_steps, log_days):
        preds = []
        for _ in range(steps):
            x = self.predict(x, variables)
            preds.append(x)
        preds = torch.stack(preds, dim=1)

        preds = transform(preds)
        y = transform(y)

        return [m(preds, y, out_variables, lat, log_steps, log_days) for m in metric], preds


# model = AFNO(depth=8, num_blocks=4).cuda()
# x, y = torch.randn(2, 3, 128, 256).cuda(), torch.randn(2, 3, 128, 256).cuda()
# pred = model.predict(x, None)
# print (pred.shape)
