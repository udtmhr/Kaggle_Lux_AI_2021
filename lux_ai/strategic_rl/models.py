from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MaskedConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 4, kernel_size: int = 5):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.expand = nn.Linear(channels, channels * expansion)
        self.project = nn.Linear(channels * expansion, channels)
        self.scale = nn.Parameter(torch.full((channels,), 1e-6))

    def forward(self, inputs):
        x, mask = inputs
        residual = x
        x = self.depthwise(x) * mask
        x = x.permute(0, 2, 3, 1)
        x = self.project(F.gelu(self.expand(self.norm(x)))) * self.scale
        x = x.permute(0, 3, 1, 2)
        return (residual + x) * mask, mask


class AxialAttentionBlock(nn.Module):
    def __init__(self, channels: int, heads: int, max_size: int = 8):
        super().__init__()
        self.row_norm = nn.LayerNorm(channels)
        self.col_norm = nn.LayerNorm(channels)
        self.row_attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.col_attn = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(nn.Linear(channels, channels * 4), nn.GELU(), nn.Linear(channels * 4, channels))
        self.row_pos = nn.Parameter(torch.zeros(1, max_size, channels))
        self.col_pos = nn.Parameter(torch.zeros(1, max_size, channels))

    @staticmethod
    def _safe_padding_mask(valid: torch.Tensor) -> torch.Tensor:
        padding = ~valid.bool()
        all_padding = padding.all(dim=1)
        padding[all_padding, 0] = False
        return padding

    def forward(self, inputs):
        x, mask = inputs
        batch, channels, height, width = x.shape
        row = x.permute(0, 2, 3, 1).reshape(batch * height, width, channels)
        row_valid = mask[:, 0].reshape(batch * height, width)
        row_in = self.row_norm(row) + self.row_pos[:, :width]
        row_out, _ = self.row_attn(row_in, row_in, row_in, key_padding_mask=self._safe_padding_mask(row_valid))
        x = (row + row_out).reshape(batch, height, width, channels).permute(0, 3, 1, 2) * mask

        col = x.permute(0, 3, 2, 1).reshape(batch * width, height, channels)
        col_valid = mask[:, 0].permute(0, 2, 1).reshape(batch * width, height)
        col_in = self.col_norm(col) + self.col_pos[:, :height]
        col_out, _ = self.col_attn(col_in, col_in, col_in, key_padding_mask=self._safe_padding_mask(col_valid))
        x = (col + col_out).reshape(batch, width, height, channels).permute(0, 3, 2, 1) * mask

        y = x.permute(0, 2, 3, 1)
        y = y + self.ffn(self.ffn_norm(y))
        return y.permute(0, 3, 1, 2) * mask, mask


class SurvivalStrategicBackbone(nn.Module):
    """Local 32x32 encoder fused with global 8x8 axial attention."""

    def __init__(self, channels: int = 96, attention_blocks: int = 4, heads: int = 6):
        super().__init__()
        mid, global_channels = channels * 3 // 2, channels * 2
        self.local = nn.Sequential(*[MaskedConvNeXtBlock(channels) for _ in range(8)])
        self.down1 = nn.Conv2d(channels, mid, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(mid, global_channels, 3, stride=2, padding=1)
        self.global_blocks = nn.Sequential(
            *[AxialAttentionBlock(global_channels, heads) for _ in range(attention_blocks)]
        )
        self.global_project = nn.Conv2d(global_channels, channels, 1)
        self.gate = nn.Sequential(nn.Conv2d(channels * 2, channels, 1), nn.Sigmoid())
        self.fusion = nn.Sequential(*[MaskedConvNeXtBlock(channels) for _ in range(4)])

    def forward(self, inputs):
        x, mask = inputs
        mask = mask.to(dtype=x.dtype)
        local, _ = self.local((x, mask))
        mask16 = F.interpolate(mask, scale_factor=0.5, mode="nearest")
        global_x = F.gelu(self.down1(local)) * mask16
        mask8 = F.interpolate(mask16, scale_factor=0.5, mode="nearest")
        global_x = F.gelu(self.down2(global_x)) * mask8
        global_x, _ = self.global_blocks((global_x, mask8))
        global_x = F.interpolate(global_x, size=local.shape[-2:], mode="bilinear", align_corners=False)
        global_x = self.global_project(global_x) * mask
        gate = self.gate(torch.cat((local, global_x), dim=1))
        fused = local + gate * global_x
        return self.fusion((fused, mask))


def compatible_attention_heads(channels: int, preferred: int) -> int:
    for heads in (preferred, 8, 6, 4, 3, 2, 1):
        if heads > 0 and channels % heads == 0:
            return heads
    return 1
