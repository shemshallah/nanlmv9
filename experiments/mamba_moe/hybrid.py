#!/usr/bin/env python3
"""Neural/prior gating helpers for the CM-residual experiments.

The real target is a v9 context-mixing prior.  For fast CPU/GPU ablations this
module provides an n-gram prior with the same interface:

    prior_logits(context_ids) -> [N, vocab] log-probabilities

so we can train the neural model on the residual
`logit_final = w_nn * logit_nn + w_prior * logit_prior`
and learn the gate weights from features (prior confidence/entropy, position).
"""
import collections
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NGramPrior:
    def __init__(self, order=1, vocab_size=32000, alpha=0.1):
        self.order = order
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.unigram = None
        self.bigram = None

    @classmethod
    def from_tokens(cls, tokens, order=1, vocab_size=32000, max_tokens=2_000_000, alpha=0.1):
        p = cls(order=order, vocab_size=vocab_size, alpha=alpha)
        arr = np.asarray(tokens[:max_tokens], dtype=np.int64)
        counts = np.bincount(arr, minlength=vocab_size).astype(np.float64)
        p.unigram = counts + alpha
        p.unigram /= p.unigram.sum()
        if order >= 2:
            bg = collections.defaultdict(collections.Counter)
            prev = arr[:-1]
            cur = arr[1:]
            for a, b in zip(prev.tolist(), cur.tolist()):
                bg[a][b] += 1
            p.bigram = bg
        return p

    def logits(self, context_ids, dtype=torch.float32):
        """context_ids: [N] tensor of previous token ids (or target positions)."""
        n = context_ids.numel()
        if self.order < 2 or self.bigram is None:
            row = torch.from_numpy(np.log(self.unigram).astype(np.float32))
            return row.to(dtype)[None].expand(n, -1).clone()
        out = np.empty((n, self.vocab_size), dtype=np.float32)
        unigram = np.log(self.unigram)
        for i, ctx in enumerate(context_ids.tolist()):
            c = self.bigram.get(ctx)
            if not c:
                out[i] = unigram
                continue
            row = np.full(self.vocab_size, self.alpha, dtype=np.float64)
            for tok, cnt in c.items():
                row[tok] += cnt
            row /= row.sum()
            out[i] = np.log(row)
        return torch.from_numpy(out).to(dtype)


class ContextGate(nn.Module):
    """Learned per-position mixture weights for neural and prior sources."""

    def __init__(self, n_sources=2, hidden=64, extra_features=0):
        super().__init__()
        in_dim = n_sources * 2 + extra_features
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_sources),
        )
        self.n_sources = n_sources

    def forward(self, source_logits, extra=None):
        """source_logits: list of [N,V]; returns weights [N,S] and mixed logits."""
        stats = []
        for logits in source_logits:
            probs = F.softmax(logits, dim=-1)
            top = probs.max(-1).values
            ent = -(probs * F.log_softmax(logits, dim=-1)).sum(-1)
            stats += [top.unsqueeze(-1), ent.unsqueeze(-1)]
        feat = torch.cat(stats, dim=-1)
        if extra is not None:
            feat = torch.cat([feat, extra], dim=-1)
        weights = F.softmax(self.net(feat), dim=-1)
        mixed = sum(weights[:, i : i + 1] * logits for i, logits in enumerate(source_logits))
        return mixed, weights


def mixture_logits(nn_logits, prior_logits, gate=None, prior_weight=1.0, extra=None):
    """Residual/mixture helper.

    Without a gate: logits_nn + prior_weight * logits_prior.
    With a gate: learned per-position convex mixture.
    """
    if gate is not None:
        return gate([nn_logits, prior_logits], extra=extra)
    return nn_logits + prior_weight * prior_logits, None
