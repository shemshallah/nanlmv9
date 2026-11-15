#!/usr/bin/env python3
"""256-wide selective state-space (Mamba-style) trunk over a 256-expert MoE bottom.

This is a compact, CPU-runnable prototype of the architecture discussed for
NaN v9+:

  bytes -> embeddings
        -> 256 low-rank experts (top-k router, shared expert)
        -> N selective SSM blocks (d_model=256, d_state=16)
        -> tied 32k token head

The MoE is deliberately low-rank so 256 experts fit in phone RAM
(256 * 2 * d_model * rank parameters instead of 256 full MLPs).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class LowRankMoE(nn.Module):
    """256 (or N) low-rank experts + one dense shared expert."""

    def __init__(self, d_model, n_experts=256, rank=8, top_k=2, shared_hidden=512):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.down = nn.Parameter(torch.empty(n_experts, d_model, rank))
        self.up = nn.Parameter(torch.empty(n_experts, rank, d_model))
        self.shared = nn.Sequential(
            nn.Linear(d_model, shared_hidden, bias=False),
            nn.SiLU(),
            nn.Linear(shared_hidden, d_model, bias=False),
        )
        nn.init.normal_(self.router.weight, std=0.02)
        nn.init.normal_(self.down, std=1.0 / math.sqrt(d_model))
        nn.init.normal_(self.up, std=1.0 / math.sqrt(rank))
        self.aux_loss = 0.0

    def forward(self, x):
        b, l, d = x.shape
        flat = x.reshape(-1, d)
        logits = self.router(flat)
        probs = F.softmax(logits, dim=-1)
        topv, topi = torch.topk(probs, self.top_k, dim=-1)
        topv = topv / topv.sum(-1, keepdim=True).clamp_min(1e-9)

        out = self.shared(flat)
        for e in range(self.n_experts):
            mask = topi == e
            if not mask.any():
                continue
            tok_idx, slot = mask.nonzero(as_tuple=True)
            h = flat[tok_idx] @ self.down[e]
            h = F.silu(h)
            h = h @ self.up[e]
            out.index_add_(0, tok_idx, h * topv[tok_idx, slot].unsqueeze(-1))

        # Switch-style load balancing: importance x load.
        importance = probs.mean(0)
        load = F.one_hot(topi, self.n_experts).float().mean(dim=(0, 1))
        self.aux_loss = self.n_experts * (importance * load).sum()
        return out.reshape(b, l, d)


class SelectiveSSM(nn.Module):
    """Mamba-1-style selective scan, pure PyTorch sequential reference."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank=None):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = dt_rank or max(1, math.ceil(d_model / 16))
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv = nn.Conv1d(
            self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1
        )
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + 2 * self.d_state, bias=False
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        a = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(a).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        nn.init.normal_(self.in_proj.weight, std=0.02)
        nn.init.normal_(self.x_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.dt_proj.bias)
        with torch.no_grad():
            self.dt_proj.bias.copy_(
                torch.linspace(math.log(0.001), math.log(0.1), self.d_inner)
            )

    def forward(self, x):
        b, l, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = self.conv(x.transpose(1, 2))[..., :l].transpose(1, 2)
        x = F.silu(x)

        x_dbl = self.x_proj(x)
        dt = F.softplus(self.dt_proj(x_dbl[..., : self.dt_rank]))
        bm = x_dbl[..., self.dt_rank : self.dt_rank + self.d_state]
        cm = x_dbl[..., self.dt_rank + self.d_state :]
        a = -torch.exp(self.A_log)

        h = x.new_zeros(b, self.d_inner, self.d_state)
        ys = []
        for t in range(l):
            dt_t = dt[:, t]
            da = torch.exp(dt_t.unsqueeze(-1) * a)
            db = dt_t.unsqueeze(-1) * bm[:, t].unsqueeze(1)
            h = da * h + db * x[:, t].unsqueeze(-1)
            ys.append((h * cm[:, t].unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1) + x * self.D
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state=d_state, expand=expand)

    def forward(self, x):
        return x + self.ssm(self.norm(x))


class MambaMoE(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        n_layers=2,
        d_state=16,
        n_experts=256,
        rank=8,
        top_k=2,
        shared_hidden=512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.moe = LowRankMoE(d_model, n_experts, rank, top_k, shared_hidden)
        self.blocks = nn.ModuleList(
            [MambaBlock(d_model, d_state=d_state) for _ in range(n_layers)]
        )
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, idx):
        x = self.embed(idx)
        x = x + self.moe(x)
        aux = self.moe.aux_loss
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x), aux


def count_params(model):
    return sum(p.numel() for p in model.parameters())
