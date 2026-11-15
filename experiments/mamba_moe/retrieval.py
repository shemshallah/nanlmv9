#!/usr/bin/env python3
"""Causal retrieval for the hybrid experiments.

Three backends:
  multimatch  hash of the last N tokens -> several training positions
  suffix      optional pydivsufsort suffix array over the token byte stream,
              with multimatch fallback when the package is unavailable
  knn         kNN over SSM hidden states (requires a model encoder)

All backends are causal: they only look backward from the query position.
"""
import collections
import math

import numpy as np
import torch
import torch.nn.functional as F


def _token_bytes(tokens):
    return np.asarray(tokens, dtype=np.uint16).tobytes()


class MultiMatchIndex:
    def __init__(self, order=4, max_candidates=8, vocab_size=32000, seed=1234):
        self.order = order
        self.max_candidates = max_candidates
        self.vocab_size = vocab_size
        self.mask = (1 << 20) - 1
        self.table = collections.defaultdict(list)
        self.tokens = None
        self.unigram = None
        self.seed = seed

    def _hash(self, ctx):
        h = 0
        for t in ctx:
            h = (h * 1000003 + int(t) + 1) & 0xFFFFFFFF
        return h & self.mask

    def build(self, tokens, max_tokens=500_000):
        arr = np.asarray(tokens[:max_tokens], dtype=np.int64)
        self.tokens = arr
        counts = np.bincount(arr, minlength=self.vocab_size).astype(np.float64) + 0.1
        self.unigram = counts / counts.sum()
        for i in range(self.order, len(arr)):
            ctx = arr[i - self.order : i].tolist()
            key = self._hash(ctx)
            bucket = self.table[key]
            if len(bucket) < self.max_candidates:
                bucket.append(i)

    def logits(self, contexts):
        """contexts: [N, order] tensor; returns [N, vocab] log-probabilities."""
        n = contexts.shape[0]
        out = np.empty((n, self.vocab_size), dtype=np.float32)
        uni = np.log(self.unigram)
        for row in range(n):
            ctx = contexts[row].tolist()
            cand = self.table.get(self._hash(ctx), [])
            if not cand:
                out[row] = uni
                continue
            counts = collections.Counter()
            for pos in cand:
                if pos < len(self.tokens):
                    counts[int(self.tokens[pos])] += 1.0
            if not counts:
                out[row] = uni
                continue
            v = np.full(self.vocab_size, 1e-6, dtype=np.float64)
            for tok, c in counts.items():
                v[tok] += c
            v /= v.sum()
            out[row] = np.log(v)
        return torch.from_numpy(out)


class SuffixArrayRetriever:
    """Optional suffix-array backend.

    pydivsufsort builds the SA over the raw token byte stream.  If it is not
    installed we fall back to MultiMatchIndex so the ablation still runs.
    """

    def __init__(self, order=4, max_candidates=8, vocab_size=32000):
        self.fallback = MultiMatchIndex(order, max_candidates, vocab_size)
        self.vocab_size = vocab_size
        self.arr = None
        self.sa = None
        self.backend = "multimatch"

    def build(self, tokens, max_tokens=500_000):
        arr = np.asarray(tokens[:max_tokens], dtype=np.uint16)
        try:
            import pydivsufsort  # type: ignore

            raw = arr.tobytes()
            self.sa = pydivsufsort.divsufsort(raw)
            self.arr = arr
            self.backend = "suffix"
            print(f"suffix array: {len(arr):,} tokens")
        except Exception as e:
            print(f"suffix array unavailable ({e}); using multimatch")
            self.fallback.build(tokens, max_tokens)

    def logits(self, contexts):
        if self.backend != "suffix":
            return self.fallback.logits(contexts)
        # A compact suffix-array query: use the last `order` tokens, binary
        # search all suffixes whose byte prefix matches, then count continuations.
        n = contexts.shape[0]
        order = contexts.shape[1]
        out = np.empty((n, self.vocab_size), dtype=np.float32)
        raw = self.arr.tobytes()
        unit = 2
        for row in range(n):
            pat = contexts[row].detach().cpu().numpy().astype(np.uint16).tobytes()
            lo, hi = 0, len(self.sa)
            while lo < hi:
                mid = (lo + hi) // 2
                p = int(self.sa[mid])
                if raw[p : p + len(pat)] < pat:
                    lo = mid + 1
                else:
                    hi = mid
            counts = collections.Counter()
            j = lo
            while j < len(self.sa):
                p = int(self.sa[j])
                if raw[p : p + len(pat)] != pat:
                    break
                if p % unit:
                    j += 1
                    continue
                tok_pos = p // unit + order
                if tok_pos < len(self.arr):
                    counts[int(self.arr[tok_pos])] += 1
                j += 1
                if j - lo > 64:
                    break
            if not counts:
                out[row] = math.log(1.0 / self.vocab_size)
            else:
                v = np.full(self.vocab_size, 1e-6, dtype=np.float64)
                for tok, c in counts.items():
                    v[tok] += c
                v /= v.sum()
                out[row] = np.log(v)
        return torch.from_numpy(out)


class KNNRetriever:
    """kNN over SSM hidden states (the L2-normalized last hidden state)."""

    def __init__(self, vocab_size=32000, k=8, max_items=200_000):
        self.vocab_size = vocab_size
        self.k = k
        self.max_items = max_items
        self.keys = None
        self.vals = None

    def build(self, hidden, next_tokens):
        h = hidden.reshape(-1, hidden.shape[-1]).float()
        y = next_tokens.reshape(-1).long()
        n = min(len(h), self.max_items)
        self.keys = F.normalize(h[:n], dim=-1)
        self.vals = y[:n]

    def logits(self, hidden):
        h = F.normalize(hidden.reshape(-1, hidden.shape[-1]).float(), dim=-1)
        self.keys = self.keys.to(h.device)
        self.vals = self.vals.to(h.device)
        sim = h @ self.keys.T
        k = min(self.k, sim.shape[1])
        vals, idx = torch.topk(sim, k, dim=-1)
        weights = F.softmax(vals * 10.0, dim=-1)
        out = torch.zeros(h.shape[0], self.vocab_size)
        for j in range(k):
            out.scatter_add_(1, self.vals[idx[:, j]].unsqueeze(1), weights[:, j : j + 1])
        out = out.clamp_min(1e-9)
        out = out / out.sum(-1, keepdim=True)
        return torch.log(out)


def build_retriever(name, tokens, vocab_size, max_tokens=500_000):
    if name == "none":
        return None
    if name == "multimatch":
        r = MultiMatchIndex(vocab_size=vocab_size)
        r.build(tokens, max_tokens=max_tokens)
        return r
    if name == "suffix":
        r = SuffixArrayRetriever(vocab_size=vocab_size)
        r.build(tokens, max_tokens=max_tokens)
        return r
    if name == "knn":
        return KNNRetriever(vocab_size=vocab_size)
    raise ValueError(f"unknown retriever {name}")
