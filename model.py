import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_

# A deterministic fallback would invalidate the Bayesian uncertainty claims.
try:
    from blitz.modules import BayesianLinear, BayesianConv2d
    from blitz.utils import variational_estimator
    _BLITZ_AVAILABLE = True
except ImportError as exc:
    raise ImportError(
        "BayesianEMCCUncertaintyModel requires blitz-bayesian-pytorch. Install it "
        "in this environment; silently replacing Bayesian layers with ordinary "
        "Conv2d layers would invalidate epistemic-uncertainty claims."
    ) from exc


# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------
def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


# -----------------------------------------------------------------------------
# Swin Transformer minimal components (single-stage variant used by the model)
# -----------------------------------------------------------------------------
from timm.layers import DropPath, to_2tuple, trunc_normal_  # requires timm


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        -1, window_size, window_size, C
    )
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        B, H, W, -1
    )
    return x


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) *
                        (2 * window_size[1] - 1),
                        num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(
            B_, N, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1
        )
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(
                B_ // nW, nW, self.num_heads, N, N
            ) + mask.unsqueeze(1).unsqueeze(0)
            attn = self.softmax(
                attn.view(-1, self.num_heads, N, N)
            )
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.dim, self.num_heads = dim, num_heads
        self.window_size, self.shift_size = window_size, shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop
        )
        self.H = None
        self.W = None

    def forward(self, x, mask_matrix):
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        # padding
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(
            -1, self.window_size, self.window_size, C
        )
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W
        x = x.view(B, H, W, C)
        if (H % 2 == 1) or (W % 2 == 1):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1).view(B, -1, 4 * C)
        x = self.reduction(self.norm(x))
        return x


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=(drop_path[i] if isinstance(drop_path, list) else drop_path),
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        self.downsample = downsample(dim=dim, norm_layer=norm_layer) \
            if downsample is not None else None

    def forward(self, x, H, W):
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(
            img_mask, self.window_size
        ).view(-1, self.window_size * self.window_size)
        attn_mask = (mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2))
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, float(-100.0)
        ).masked_fill(attn_mask == 0, 0.0)

        for blk in self.blocks:
            blk.H, blk.W = H, W
            x = blk(x, attn_mask)

        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=2, in_chans=3, embed_dim=64, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = self.proj(x)
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = self.norm(
                x.flatten(2).transpose(1, 2)
            ).transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x


class SwinTransformer(nn.Module):
    """
    Minimal Swin stage wrapper:
      - if feature_num == 0, apply PatchEmbed; else expect feature map as input
      - returns feature map [B, C, H, W]
    """
    def __init__(
        self,
        in_chans,
        out_chans,
        depth,
        numhead,
        feature_num,
        pretrain_img_size=512,
        patch_size=2,
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.15,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        frozen_stages=-1,
        use_checkpoint=False
    ):
        super().__init__()
        self.embed_dim = out_chans
        self.ape = ape
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.patch_embed_label = feature_num

        if feature_num == 0:
            self.patch_embed = PatchEmbed(
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=out_chans,
                norm_layer=norm_layer if patch_norm else None
            )

        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [
                pretrain_img_size[0] // patch_size[0],
                pretrain_img_size[1] // patch_size[1]
            ]
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, out_chans,
                            patches_resolution[0], patches_resolution[1])
            )
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.layers = nn.ModuleList([
            BasicLayer(
                dim=int(out_chans),
                depth=depth,
                num_heads=numhead,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint
            )
        ])
        self.one_layer = norm_layer(out_chans)

    def forward(self, x):
        if self.patch_embed_label == 0:
            x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)

        if self.ape:
            absolute_pos_embed = F.interpolate(
                self.absolute_pos_embed,
                size=(Wh, Ww),
                mode='bicubic'
            )
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)

        x = self.pos_drop(x)
        layer = self.layers[0]
        x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)
        x_out = self.one_layer(x_out)
        out_feature = x_out.view(
            -1, H, W, self.embed_dim
        ).permute(0, 3, 1, 2).contiguous()
        return out_feature


# -----------------------------------------------------------------------------
# Core conv & attention utils
# -----------------------------------------------------------------------------
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=SiLU()):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act is True \
            else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1,
            kernel_size, padding=kernel_size // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))


class EMCCBlock(nn.Module):
    """
    Grouped Depthwise Convolution Block
    Uses grouped convolutions and depthwise separable convolutions
    for efficient feature extraction.
    """
    def __init__(self, in_channels, out_channels, groups=8):
        super().__init__()
        self.groups = groups

        self.pointwise1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.depthwise = nn.Conv2d(
            out_channels, out_channels,
            3, padding=1,
            groups=min(groups, out_channels),
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.pointwise2 = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.pointwise1(x)))
        x = self.relu(self.bn2(self.depthwise(x)))
        x = self.relu(self.bn3(self.pointwise2(x)))
        return x


# -----------------------------------------------------------------------------
# Simple decoder with skip connections
# -----------------------------------------------------------------------------
class SimpleDecoder(nn.Module):
    """
    Simple decoder with upsampling and skip connections.

    Args:
        bottleneck_channels: channels at bottleneck (input to decoder)
        skip_channels: list of channels for skips in order [x3, x2, x1, x0]
        out_classes: number of output classes
        drop_rate: dropout probability

    Returns:
        feats: final feature map before last 1x1 conv
        logits: segmentation logits from final conv
    """
    def __init__(
        self,
        bottleneck_channels=1024,
        skip_channels=None,
        out_classes=3,
        drop_rate=0.1
    ):
        super().__init__()
        if skip_channels is None:
            skip_channels = [512, 256, 128, 64]

        current_channels = bottleneck_channels

        self.up_blocks = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        self.skip_EMCC_blocks = nn.ModuleList()
        self.post_concat_EMCC_blocks = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for skip_ch in skip_channels:
            self.up_blocks.append(
                nn.ConvTranspose2d(
                    current_channels, current_channels // 2,
                    2, 2, bias=False
                )
            )

            self.skip_EMCC_blocks.append(EMCCBlock(skip_ch, skip_ch))

            combined_channels = current_channels // 2 + skip_ch
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv2d(
                        combined_channels,
                        current_channels // 2,
                        3, 1, 1,
                        bias=False
                    ),
                    nn.BatchNorm2d(current_channels // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        current_channels // 2,
                        current_channels // 2,
                        3, 1, 1,
                        bias=False
                    ),
                    nn.BatchNorm2d(current_channels // 2),
                    nn.ReLU(inplace=True)
                )
            )

            self.post_concat_EMCC_blocks.append(
                EMCCBlock(current_channels // 2, current_channels // 2)
            )

            self.dropouts.append(nn.Dropout2d(p=drop_rate))

            current_channels = current_channels // 2

        self.out_channels = current_channels
        self.final_conv = nn.Conv2d(
            current_channels, out_classes,
            kernel_size=1, bias=False
        )

    def forward(self, bottleneck, skip_features):
        x = bottleneck
        skip = skip_features[::-1]  # [x3, x2, x1, x0]

        for up_block, conv_block, skip_EMCC, post_EMCC, dropout in zip(
            self.up_blocks,
            self.conv_blocks,
            self.skip_EMCC_blocks,
            self.post_concat_EMCC_blocks,
            self.dropouts
        ):
            x = up_block(x)

            s = skip.pop(0)
            s = skip_EMCC(s)

            if x.size()[2:] != s.size()[2:]:
                s = F.interpolate(
                    s, size=x.size()[2:],
                    mode='bilinear',
                    align_corners=False
                )

            x = torch.cat([x, s], dim=1)
            x = conv_block(x)
            x = post_EMCC(x)
            x = dropout(x)

        feats = x
        logits = self.final_conv(feats)
        return feats, logits


# -----------------------------------------------------------------------------
# Bayesian Transition Block (replaces DPA-MFE)
# -----------------------------------------------------------------------------
@variational_estimator
class BayesianTransitionBlock(nn.Module):
    def __init__(self, c1, c2, reduction=32):
        super().__init__()
        self.cv1 = BayesianConv2d(c1, c2, kernel_size=(1, 1), stride=1, prior_sigma_1=0.1, prior_sigma_2=0.002, prior_pi=1.0)
        self.cv2 = BayesianConv2d(c1, c2, kernel_size=(1, 1), stride=1, prior_sigma_1=0.1, prior_sigma_2=0.002, prior_pi=1.0)
        self.cv3 = BayesianConv2d(c2, c2, kernel_size=(3, 3), stride=2, padding=1, prior_sigma_1=0.1, prior_sigma_2=0.002, prior_pi=1.0)
        self.mp = nn.MaxPool2d(2)

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, 2 * c2 // reduction)
        self.conv1 = nn.Conv2d(2 * c2, mip, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mip, 2 * c2, 1, 1, 0)
        self.conv_w = nn.Conv2d(mip, 2 * c2, 1, 1, 0)

    def forward(self, x):
        x_1 = self.mp(x)
        x_1 = self.cv1(x_1)
        x_2 = self.cv2(x)
        x_2 = self.cv3(x_2)
        x_o = torch.cat([x_2, x_1], 1)

        n, c, h, w = x_o.size()
        x_h = self.pool_h(x_o)
        x_w = self.pool_w(x_o).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return x_o * a_w * a_h


# -----------------------------------------------------------------------------
# Heteroscedastic head (mean + variance)
# -----------------------------------------------------------------------------
class HeteroscedasticHead(nn.Module):
    """
    Predicts:
      - mean (μ): segmentation logits
      - var  (σ²): pixel-wise variance (softplus to ensure positivity)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mean = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.varp = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.softplus = nn.Softplus()

    def forward(self, x):
        mu = self.mean(x)
        var = self.softplus(self.varp(x)) + 1e-6
        return mu, var


# -----------------------------------------------------------------------------
# Top-level model: Bayesian encoder + EMCC + SimpleDecoder + Heteroscedastic heads
# -----------------------------------------------------------------------------
class BayesianEMCCUncertaintyModel(nn.Module):
    """
    Inputs:
      x: [B, 4, H, W] (4 MRI modalities)

    Args:
      out_channels: number of segmentation classes (e.g., 3 for WT/TC/ET)
      drop_rate: dropout probability
      base_dim: base channel width (controls model capacity)
    """
    def __init__(self, out_channels=3, drop_rate=0.1, base_dim=64):
        super().__init__()
        self.drop_rate = drop_rate
        self.base_dim = base_dim

        # Encoder stem: 4 -> base_dim
        self.conv_in = nn.Conv2d(4, base_dim, kernel_size=1, padding=0, stride=1)
        self.dropout_conv_in = nn.Dropout2d(p=drop_rate)

        # Swin stages with scalable channels
        # x0: base_dim
        # x1: 2*base_dim
        # x2: 4*base_dim
        # x3: 8*base_dim
        # x4 (bottleneck input): 16*base_dim

        self.feature_1 = SwinTransformer(
            in_chans=base_dim,
            out_chans=base_dim * 2,
            depth=4,
            numhead=4,
            feature_num=0
        )
        self.dropout_1 = nn.Dropout2d(p=drop_rate)

        self.former_2 = SwinTransformer(
            in_chans=base_dim * 2,
            out_chans=base_dim * 2,
            depth=4,
            numhead=8,
            feature_num=1
        )
        self.feature_2 = BayesianTransitionBlock(
            c1=base_dim * 2,
            c2=base_dim * 2
        )  # -> 4*base_dim
        self.dropout_2 = nn.Dropout2d(p=drop_rate)

        self.former_3 = SwinTransformer(
            in_chans=base_dim * 4,
            out_chans=base_dim * 4,
            depth=4,
            numhead=16,
            feature_num=2
        )
        self.feature_3 = BayesianTransitionBlock(
            c1=base_dim * 4,
            c2=base_dim * 4
        )  # -> 8*base_dim
        self.dropout_3 = nn.Dropout2d(p=drop_rate)

        self.former_4 = SwinTransformer(
            in_chans=base_dim * 8,
            out_chans=base_dim * 8,
            depth=4,
            numhead=32,
            feature_num=3
        )
        self.feature_4 = BayesianTransitionBlock(
            c1=base_dim * 8,
            c2=base_dim * 8
        )  # -> 16*base_dim
        self.dropout_4 = nn.Dropout2d(p=drop_rate)

        bottleneck_channels = base_dim * 16

        # Bottleneck with EMCC blocks
        self.pre_bottleneck_EMCC = EMCCBlock(bottleneck_channels, bottleneck_channels)
        self.dropout_pre_bottleneck = nn.Dropout2d(p=drop_rate)

        self.EMCC = EMCCBlock(bottleneck_channels, bottleneck_channels)
        self.dropout_bottleneck = nn.Dropout2d(p=drop_rate)

        self.post_bottleneck_EMCC = EMCCBlock(bottleneck_channels, bottleneck_channels)

        # Decoder
        self.decoder = SimpleDecoder(
            bottleneck_channels=bottleneck_channels,
            skip_channels=[
                base_dim * 8,   # x3
                base_dim * 4,   # x2
                base_dim * 2,   # x1
                base_dim        # x0
            ],
            out_classes=out_channels,
            drop_rate=drop_rate
        )

        # Heteroscedastic heads (encoder scales + final decoder)
        self.head_enc1 = HeteroscedasticHead(
            in_channels=base_dim * 8,
            out_channels=out_channels
        )
        self.head_enc2 = HeteroscedasticHead(
            in_channels=base_dim * 4,
            out_channels=out_channels
        )
        self.head_enc3 = HeteroscedasticHead(
            in_channels=base_dim * 2,
            out_channels=out_channels
        )
        self.head_dec = HeteroscedasticHead(
            in_channels=self.decoder.out_channels,
            out_channels=out_channels
        )

    def kl_divergence(self):
        """Differentiable sampled variational KL from Blitz Bayesian transition blocks.

        Call only AFTER a forward pass, because Blitz caches log-density terms
        for the weights sampled in the most recent forward pass.
        """
        terms = []
        for block in self.modules():
            if not isinstance(block, BayesianTransitionBlock):
                continue
            fn = getattr(block, "nn_kl_divergence", None)
            if not callable(fn):
                raise RuntimeError("Blitz BayesianTransitionBlock lacks nn_kl_divergence()")
            value = fn()
            if not torch.is_tensor(value) or not value.requires_grad or not torch.isfinite(value).all():
                raise RuntimeError("Bayesian KL must be a finite differentiable tensor after forward()")
            terms.append(value.reshape(()))
        if not terms:
            raise RuntimeError("No BayesianTransitionBlock found for KL regularization")
        return torch.stack(terms).sum()

    def forward(self, x):
        # Encoder
        x0 = self.conv_in(x)          # [B, base_dim, H, W]
        x0 = self.dropout_conv_in(x0)

        x1 = self.feature_1(x0)       # [B, 2*base_dim, H/2, W/2]
        x1 = self.dropout_1(x1)

        x2 = self.former_2(x1)        # [B, 2*base_dim, ...]
        x2 = self.feature_2(x2)       # [B, 4*base_dim, ...]
        # x2 = self.dropout_2(x2)

        x3 = self.former_3(x2)        # [B, 4*base_dim, ...]
        x3 = self.feature_3(x3)       # [B, 8*base_dim, ...]
        # x3 = self.dropout_3(x3)

        x4 = self.former_4(x3)        # [B, 8*base_dim, ...]
        x4 = self.feature_4(x4)       # [B, 16*base_dim, ...]
        # x4 = self.dropout_4(x4)

        # Bottleneck
        x4 = self.pre_bottleneck_EMCC(x4)
        x4 = self.dropout_pre_bottleneck(x4)

        bottleneck = self.EMCC(x4)
        # bottleneck = self.dropout_bottleneck(bottleneck)

        bottleneck = self.post_bottleneck_EMCC(bottleneck)

        # Decoder
        skip_features = [x0, x1, x2, x3]
        dec_feats, dec_logits = self.decoder(bottleneck, skip_features)

        # Heteroscedastic heads
        mu1, var1 = self.head_enc1(x3)
        mu2, var2 = self.head_enc2(x2)
        mu3, var3 = self.head_enc3(x1)

        mu_final, var_final = self.head_dec(dec_feats)

        return {
            "final_mean": mu_final,
            "final_var": var_final,
            "decoder_logits": dec_logits,
            "mean_stages": [mu1, mu2, mu3],
            "var_stages": [var1, var2, var3]
        }


# -----------------------------------------------------------------------------
# Quick shape sanity check (optional)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    with torch.no_grad():
        model = BayesianEMCCUncertaintyModel(
            out_channels=3,
            drop_rate=0.15,
            base_dim=64  # try 48 / 64 / 80 etc
        ).eval()
        x = torch.randn(2, 4, 160, 160)
        out = model(x)
        print("final_mean:", out["final_mean"].shape)
        print("final_var :", out["final_var"].shape)
        print("mean_stages:", [t.shape for t in out["mean_stages"]])
        print("var_stages :", [t.shape for t in out["var_stages"]])
        if not _BLITZ_AVAILABLE:
            print("NOTE: blitz-bayesian-pytorch not found — running with deterministic BayesianConv2d fallback.")
