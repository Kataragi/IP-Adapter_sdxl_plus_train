# modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
# and https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/imagen_pytorch.py

import math

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )

def reshape_tensor(x, heads):
    bs, length, width = x.shape
    x = x.view(bs, length, heads, -1)
    x = x.transpose(1, 2)
    x = x.reshape(bs, heads, length, -1)
    return x

class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        # x を latents に合わせて次元拡張
        if x.dim() == 2 and latents.dim() == 3:
            x = x.unsqueeze(1)  # [batch_size, features] -> [batch_size, 1, features]
        # dtype を揃えて結合
        kv_input = torch.cat((x.to(latents.dtype), latents), dim=-2)  # [batch_size, seq_length + 1, features]
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        # 明示的に dtype を fp16 に変更
        if latents.dtype != torch.float16:
            latents = latents.to(dtype=torch.float16)

        # Process latents
        b, l, _ = latents.shape
        q = self.to_q(latents)
        kv_input = torch.cat((x.to(latents.dtype), latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # Attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)
        result = self.to_out(out)
        return result

class Resampler(nn.Module):
    def __init__(
        self,
        dim=1280, # 1024→1280に変更
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=1024, # 768→1024に変更
        output_dim=1280, # 1024→1280に変更
        ff_mult=4,
        max_seq_len: int = 577,  # CLIP tokens + CLS token
        apply_pos_emb: bool = False,
        num_latents_mean_pooled: int = 0,  # number of latents derived from mean pooled representation of the sequence
    ):
    
        # デバッグメッセージを表示
        print(f"[DEBUG] Resampler initialized with max_seq_len: {max_seq_len}")
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim) if apply_pos_emb else None

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.to_latents_from_mean_pooled_seq = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange("b (n d) -> b n d", n=num_latents_mean_pooled),
            )
            if num_latents_mean_pooled > 0
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        if self.pos_emb is not None:
            n, device = x.shape[1], x.device
            pos_emb = self.pos_emb(torch.arange(n, device=device))
            x = x + pos_emb

        latents = self.latents.repeat(x.size(0), 1, 1)

        # 入力 x を proj_in に合わせた形状にリサイズ
        expected_in_features = self.proj_in.weight.shape[1]
        if x.shape[-1] != expected_in_features:
            # print(f"Resizing input x from {x.shape[-1]} to {expected_in_features}")
            x = torch.nn.functional.pad(
                x, (0, expected_in_features - x.shape[-1])
            ) if x.shape[-1] < expected_in_features else x[:, :, :expected_in_features]

        x = self.proj_in(x)  # プロジェクション

        if self.to_latents_from_mean_pooled_seq:
            meanpooled_seq = masked_mean(x, dim=1, mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents.to(latents.dtype), latents), dim=-2)  # Ensure dtype consistency

        # Process layers
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        # Final projection
        latents = self.proj_out(latents)
        return self.norm_out(latents)


def masked_mean(t, *, dim, mask=None):
    if mask is None:
        return t.mean(dim=dim)

    denom = mask.sum(dim=dim, keepdim=True)
    mask = rearrange(mask, "b n -> b n 1")
    masked_t = t.masked_fill(~mask, 0.0)

    return masked_t.sum(dim=dim) / denom.clamp(min=1e-5)
