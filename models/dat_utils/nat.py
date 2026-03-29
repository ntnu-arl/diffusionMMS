#################################################################################################
# Copyright (c) 2023 Ali Hassani.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################################
# --------------------------------------------------------
# Vision Transformer with Deformable Attention
# Modified by Zhuofan Xia
# Originated from https://github.com/SHI-Labs/NATTEN/blob/main/src/natten/natten2d.py
# --------------------------------------------------------
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import pad
from torch.nn.init import trunc_normal_


def na2d_qk(q, k, kernel_size, dilation, rpb=None):
    """Compute 2D neighborhood attention QK scores (pure PyTorch).

    Args:
        q: (B, num_heads, H, W, head_dim) – already scaled.
        k: (B, num_heads, H, W, head_dim).
        kernel_size: int, odd.
        dilation: int.
        rpb: (num_heads, 2*kernel_size-1, 2*kernel_size-1) or None.

    Returns:
        attn: (B, num_heads, H, W, kernel_size**2).
    """
    B, num_heads, H, W, head_dim = q.shape
    BN = B * num_heads
    K = kernel_size
    K2 = K * K
    pad_size = (K // 2) * dilation

    # Unfold k into (BN, H*W, K*K, head_dim)
    k_4d = k.reshape(BN, H, W, head_dim).permute(0, 3, 1, 2).contiguous()
    k_uf = F.unfold(k_4d, kernel_size=K, dilation=dilation, padding=pad_size)
    k_uf = k_uf.view(BN, head_dim, K2, H * W).permute(0, 3, 2, 1)

    # Validity mask – 0 for positions that fall outside the original tensor
    mask_1 = torch.ones(1, 1, H, W, device=q.device, dtype=q.dtype)
    mask_uf = F.unfold(mask_1, kernel_size=K, dilation=dilation, padding=pad_size)
    valid = mask_uf.squeeze(0).t() > 0.5  # (H*W, K*K)

    # QK dot product
    q_flat = q.reshape(BN, H * W, head_dim)
    attn = torch.einsum("bnc,bnkc->bnk", q_flat, k_uf)  # (BN, H*W, K*K)

    # Mask invalid (padded) positions
    attn = attn.masked_fill(~valid.unsqueeze(0), float("-inf"))

    # Relative position bias – index the centre K×K block of the (2K-1)×(2K-1) table
    if rpb is not None:
        off = (K - 1) // 2  # offset to centre of RPB table
        idx = torch.arange(K, device=rpb.device) + off
        rpb_bias = rpb[:, idx][:, :, idx]  # (num_heads, K, K)
        rpb_bias = rpb_bias.reshape(1, num_heads, 1, K2)
        attn = attn.view(B, num_heads, H * W, K2) + rpb_bias
        attn = attn.reshape(BN, H * W, K2)

    return attn.view(B, num_heads, H, W, K2)


def na2d_av(attn, v, kernel_size, dilation):
    """Apply 2D neighborhood attention weights to values (pure PyTorch).

    Args:
        attn: (B, num_heads, H, W, kernel_size**2) – post-softmax.
        v:    (B, num_heads, H, W, head_dim).

    Returns:
        out:  (B, num_heads, H, W, head_dim).
    """
    B, num_heads, H, W, head_dim = v.shape
    BN = B * num_heads
    K = kernel_size
    K2 = K * K
    pad_size = (K // 2) * dilation

    v_4d = v.reshape(BN, H, W, head_dim).permute(0, 3, 1, 2).contiguous()
    v_uf = F.unfold(v_4d, kernel_size=K, dilation=dilation, padding=pad_size)
    v_uf = v_uf.view(BN, head_dim, K2, H * W).permute(0, 3, 2, 1)  # (BN, H*W, K*K, head_dim)

    a_flat = attn.reshape(BN, H * W, K2)
    out = torch.einsum("bnk,bnkc->bnc", a_flat, v_uf)
    return out.view(B, num_heads, H, W, head_dim)


class NeighborhoodAttention2D(nn.Module):
    """
    Neighborhood Attention 2D Module
    """

    def __init__(
        self, dim, kernel_size, num_heads, attn_drop=0.0, proj_drop=0.0, dilation=None
    ):
        super().__init__()
        self.fp16_enabled = False
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim**-0.5
        assert (
            kernel_size > 1 and kernel_size % 2 == 1
        ), f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        assert kernel_size in [
            3,
            5,
            7,
            9,
            11,
            13,
        ], f"CUDA kernel only supports kernel sizes 3, 5, 7, 9, 11, and 13; got {kernel_size}."
        self.kernel_size = kernel_size
        if type(dilation) is str:
            self.dilation = None
            self.window_size = None
        else:
            assert (
                dilation is None or dilation >= 1
            ), f"Dilation must be greater than or equal to 1, got {dilation}."
            self.dilation = dilation or 1
            self.window_size = self.kernel_size * self.dilation

        self.qkv = nn.Linear(dim, dim * 3)
        self.rpb = nn.Parameter(
            torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1))
        )
        trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # assert x.dtype == torch.float16, f"AMP failed!, dtype={x.dtype}"
        x = x.permute(0, 2, 3, 1)
        B, Hp, Wp, C = x.shape
        H, W = int(Hp), int(Wp)
        pad_l = pad_t = pad_r = pad_b = 0
        dilation = self.dilation
        window_size = self.window_size
        if window_size is None:
            dilation = max(min(H, W) // self.kernel_size, 1)
            window_size = dilation * self.kernel_size
        if H < window_size or W < window_size:
            pad_l = pad_t = 0
            pad_r = max(0, window_size - W)
            pad_b = max(0, window_size - H)
            x = pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, H, W, 3, self.num_heads, self.head_dim)
            .permute(3, 0, 4, 1, 2, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        # breakpoint()
        attn = na2d_qk(q, k, self.kernel_size, dilation, rpb=self.rpb)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = na2d_av(attn, v, self.kernel_size, dilation)
        x = x.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)
        if pad_r or pad_b:
            x = x[:, :Hp, :Wp, :]
        return self.proj_drop(self.proj(x)).permute(0, 3, 1, 2), None, None
