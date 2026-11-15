#!/usr/bin/env python3
"""Configurable 256-expert MoE + selective-SSM language model.

Ablation axes
-------------
moe_position : none | bottom | ffn | both
    - bottom: one MoE on the token embeddings
    - ffn:    a MoE FFN inside every SSM block
    - both:   both of the above
use_shared   : include a dense shared expert in each MoE
pair_matrix  : integrate a learned E x E expert-pair adjacency matrix
               through a mean-field pair adjustment of the router logits
pair_symmetric: symmetrize the pair matrix (C(256,2) semantics)
ffn_mode     : dense | moe (only used when moe_position is none/ffn/both)
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
    """E low-rank experts + optional dense shared expert + pair adjacency."""

    def __init__(
        self,
        d_model,
        n_experts=256,
        rank=8,
        top_k=2,
        shared_hidden=512,
        use_shared=True,
        pair_matrix=False,
        pair_symmetric=True,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.use_shared = use_shared
        self.pair_symmetric = pair_symmetric
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.down = nn.Parameter(torch.empty(n_experts, d_model, rank))
        self.up = nn.Parameter(torch.empty(n_experts, rank, d_model))
        nn.init.normal_(self.router.weight, std=0.02)
        nn.init.normal_(self.down, std=1.0 / math.sqrt(d_model))
        nn.init.normal_(self.up, std=1.0 / math.sqrt(rank))

        if use_shared:
            self.shared = nn.Sequential(
                nn.Linear(d_model, shared_hidden, bias=False),
                nn.SiLU(),
                nn.Linear(shared_hidden, d_model, bias=False),
            )
        else:
            self.shared = None

        if pair_matrix:
            # E x E pair adjacency.  For C(256,2) semantics we symmmetrize it.
            self.pair_bias = nn.Parameter(torch.zeros(n_experts, n_experts))
            self.pair_scale = nn.Parameter(torch.tensor(0.1))
            self.pair_mix_scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.pair_bias = None
            self.pair_scale = None
            self.pair_mix_scale = None

        self.aux_loss = 0.0
        self.pair_reg = 0.0

    def _pair_adjust(self, logits, flat):
        """Mean-field pair interaction.

        p = softmax(logits); pair = p @ B; logits += g * pair.
        Every expert's score is adjusted by its expected compatibility with the
        current expert distribution.  This uses the full E x E matrix and is
        differentiable.
        """
        if self.pair_bias is None:
            return logits
        b = self.pair_bias
        if self.pair_symmetric:
            b = 0.5 * (b + b.transpose(0, 1))
        p = F.softmax(logits, dim=-1)
        pair = p @ b
        return logits + F.softplus(self.pair_scale) * pair

    def forward(self, x):
        b, l, d = x.shape
        flat = x.reshape(-1, d)
        logits = self.router(flat)
        logits = self._pair_adjust(logits, flat)
        probs = F.softmax(logits, dim=-1)
        topv, topi = torch.topk(probs, self.top_k, dim=-1)
        topv = topv / topv.sum(-1, keepdim=True).clamp_min(1e-9)

        if self.pair_bias is not None and self.top_k >= 2:
            pb = self.pair_bias
            if self.pair_symmetric:
                pb = 0.5 * (pb + pb.transpose(0, 1))
            gates = []
            for i in range(self.top_k):
                for j in range(i + 1, self.top_k):
                    gates.append(
                        torch.sigmoid(self.pair_mix_scale * pb[topi[:, i], topi[:, j]])
                    )
            if gates:
                gate = torch.stack(gates, dim=-1).mean(-1)
                topv = topv * gate.unsqueeze(-1)
                topv = topv / topv.sum(-1, keepdim=True).clamp_min(1e-9)

        out = self.shared(flat) if self.shared is not None else torch.zeros_like(flat)
        for e in range(self.n_experts):
            mask = topi == e
            if not mask.any():
                continue
            tok_idx, slot = mask.nonzero(as_tuple=True)
            h = flat[tok_idx] @ self.down[e]
            h = F.silu(h)
            h = h @ self.up[e]
            out.index_add_(0, tok_idx, h * topv[tok_idx, slot].unsqueeze(-1))

        importance = probs.mean(0)
        load = F.one_hot(topi, self.n_experts).float().mean(dim=(0, 1))
        self.aux_loss = self.n_experts * (importance * load).sum()
        if self.pair_bias is not None:
            self.pair_reg = self.pair_bias.pow(2).mean()
        return out.view(b, l, d)


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


class DenseFFN(nn.Module):
    def __init__(self, d_model, hidden=None):
        super().__init__()
        hidden = hidden or 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, d_model, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        expand=2,
        ffn_mode="dense",
        moe_kwargs=None,
        ffn_hidden=None,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state=d_state, expand=expand)
        self.norm2 = RMSNorm(d_model)
        if ffn_mode == "moe":
            self.ffn = LowRankMoE(d_model, **(moe_kwargs or {}))
        else:
            self.ffn = DenseFFN(d_model, ffn_hidden)

    def forward(self, x):
        x = x + self.ssm(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


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
        moe_position="bottom",   # none | bottom | ffn | both
        use_shared=True,
        pair_matrix=False,
        pair_symmetric=True,
        ffn_hidden=None,
        local_context=0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.moe_position = moe_position
        self.local_context = local_context
        self.embed = nn.Embedding(vocab_size, d_model)
        if local_context:
            self.local_alpha = nn.Parameter(torch.zeros(local_context))
        nn.init.normal_(self.embed.weight, std=0.02)

        moe_kwargs = dict(
            n_experts=n_experts,
            rank=rank,
            top_k=top_k,
            shared_hidden=shared_hidden,
            use_shared=use_shared,
            pair_matrix=pair_matrix,
            pair_symmetric=pair_symmetric,
        )
        self.bottom_moe = None
        if moe_position in ("bottom", "both"):
            self.bottom_moe = LowRankMoE(d_model, **moe_kwargs)

        ffn_mode = "moe" if moe_position in ("ffn", "both") else "dense"
        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    d_model,
                    d_state=d_state,
                    ffn_mode=ffn_mode,
                    moe_kwargs=moe_kwargs,
                    ffn_hidden=ffn_hidden,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight

    def _embed(self, idx):
        x = self.embed(idx)
        if self.local_context:
            b, l = idx.shape
            for k in range(1, self.local_context + 1):
                if k >= l:
                    break
                prev = torch.cat([torch.zeros(b, k, dtype=idx.dtype, device=idx.device), idx[:, :-k]], dim=1)
                x = x + self.local_alpha[k - 1] * self.embed(prev)
        return x

    def forward(self, idx):
        x = self._embed(idx)
        aux = 0.0
        if self.bottom_moe is not None:
            x = x + self.bottom_moe(x)
            aux = aux + self.bottom_moe.aux_loss
        for blk in self.blocks:
            x = blk(x)
            if isinstance(blk.ffn, LowRankMoE):
                aux = aux + blk.ffn.aux_loss
        x = self.norm(x)
        return self.head(x), aux

    def encode(self, idx):
        """Hidden states before the token head, for SSM-state retrieval."""
        x = self._embed(idx)
        if self.bottom_moe is not None:
            x = x + self.bottom_moe(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
