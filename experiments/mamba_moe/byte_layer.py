#!/usr/bin/env python3
"""Exact token->byte probability layer.

The 32k tokenizer is a bijection on bytes.  A token distribution P(t) induces
a byte distribution through the vocabulary trie:

    P(b | prefix) = sum_{t: bytes(t) starts with prefix+b} P(t)
                  / sum_{t: bytes(t) starts with prefix}   P(t)

This module implements the trie, exact byte NLL for a target token, and a
loader for a dumped v9 CM byte prior.
"""
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F


class TrieNode:
    __slots__ = ("children", "token_ids", "torch_ids")

    def __init__(self):
        self.children = {}
        self.token_ids = []
        self.torch_ids = None


class ByteTrie:
    def __init__(self, vocab_bytes):
        self.root = TrieNode()
        self.vocab_bytes = [bytes(b) for b in vocab_bytes]
        for tid, b in enumerate(self.vocab_bytes):
            node = self.root
            node.token_ids.append(tid)
            for byte in b:
                node = node.children.setdefault(byte, TrieNode())
                node.token_ids.append(tid)
        self.token_ids = [np.asarray(n.token_ids, dtype=np.int64) for n in self._nodes()]
        self.nodes = list(self._nodes())

    def _nodes(self):
        stack = [self.root]
        while stack:
            n = stack.pop()
            yield n
            stack.extend(n.children.values())

    @classmethod
    def from_meta(cls, meta_path):
        meta = json.load(open(meta_path))
        vocab = [bytes.fromhex(h) for h in meta["vocab_hex"]]
        return cls(vocab), meta

    def to(self, device):
        for n in self.nodes:
            n.torch_ids = torch.as_tensor(n.token_ids, dtype=torch.long, device=device)
        return self

    def _mass(self, probs, node):
        if not node.token_ids:
            return probs.new_zeros(probs.shape[0])
        ids = node.torch_ids
        if ids is None or ids.device != probs.device:
            ids = torch.as_tensor(node.token_ids, dtype=torch.long, device=probs.device)
        return probs[:, ids].sum(-1)

    def byte_nll(self, token_logits, target_ids):
        """Exact byte NLL of each target token. Returns [N] and bytes [N]."""
        probs = F.softmax(token_logits, dim=-1)
        n = probs.shape[0]
        nll = torch.zeros(n, device=probs.device)
        nbytes = torch.zeros(n, dtype=torch.long, device=probs.device)
        for i in range(n):
            tok = int(target_ids[i])
            node = self.root
            parent = self._mass(probs[i : i + 1], node)[0]
            for byte in self.vocab_bytes[tok]:
                node = node.children[byte]
                child = self._mass(probs[i : i + 1], node)[0]
                nll[i] -= torch.log((child / parent).clamp_min(1e-12))
                parent = child
                nbytes[i] += 1
        return nll, nbytes

    def byte_distribution(self, token_logits, prefix=b""):
        """Byte distribution after a byte prefix, for one token distribution.

        token_logits: [V] or [1,V].  Returns [256] probabilities and the mass
        of the prefix (0 means the prefix is impossible).
        """
        probs = F.softmax(token_logits.reshape(1, -1), dim=-1)
        node = self.root
        for byte in prefix:
            if byte not in node.children:
                return torch.zeros(256), probs.new_zeros(1)[0]
            node = node.children[byte]
        total = self._mass(probs, node)[0]
        out = probs.new_zeros(256)
        if float(total) <= 0:
            return out, total
        for byte, child in node.children.items():
            out[byte] = self._mass(probs, child)[0] / total
        return out, total

    def token_mass_under_prefix(self, token_logits, prefix=b""):
        probs = F.softmax(token_logits.reshape(1, -1), dim=-1)
        node = self.root
        for byte in prefix:
            if byte not in node.children:
                return probs.new_zeros(1)[0]
            node = node.children[byte]
        return self._mass(probs, node)[0]


def verify_token_byte_equivalence(token_logits, target_ids, trie):
    """Token NLL must equal the sum of per-byte NLL."""
    probs = F.softmax(token_logits, dim=-1)
    token_nll = -torch.log(probs.gather(1, target_ids[:, None]).clamp_min(1e-12)).squeeze(1)
    byte_nll, nbytes = trie.byte_nll(token_logits, target_ids)
    print("token nll", token_nll.tolist())
    print("byte  nll", byte_nll.tolist())
    print("bytes/token", nbytes.tolist())
    print("max abs diff", (token_nll - byte_nll).abs().max().item())


class CMBytePrior:
    """Loader for a v9 CM byte-prior dump.

    Dump format (little-endian):
      magic b'NANCM9P1', u32 n, u32 k
      u8 true_byte[n]
      u8 top_ids[n*k]
      f32 top_probs[n*k]  # top-k probabilities, rest is residual mass
    """

    def __init__(self, ids, probs, true_bytes, k):
        self.ids = ids
        self.probs = probs
        self.true_bytes = true_bytes
        self.k = k
        self.n = len(true_bytes)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            magic = f.read(8)
            if magic != b"NANCM9P1":
                raise ValueError("bad CM prior magic")
            n, k = np.frombuffer(f.read(8), dtype="<u4")
            true = np.empty(n, dtype=np.uint8)
            ids = np.empty((n, k), dtype=np.uint8)
            probs = np.empty((n, k), dtype=np.float32)
            for i in range(n):
                true[i] = np.frombuffer(f.read(1), dtype=np.uint8)[0]
                ids[i] = np.frombuffer(f.read(k), dtype=np.uint8)
                probs[i] = np.frombuffer(f.read(4 * k), dtype="<f4")
        return cls(ids, probs, true, k)

    def distribution(self, pos, vocab=256):
        """Dense [256] distribution; residual mass is spread uniformly."""
        out = torch.full((vocab,), 1e-8)
        ids = torch.from_numpy(self.ids[pos].astype(np.int64))
        p = torch.from_numpy(self.probs[pos].astype(np.float32))
        out[ids] = p
        residual = (1.0 - p.sum()).clamp_min(0.0)
        mask = torch.ones(vocab, dtype=torch.bool)
        mask[ids] = False
        if mask.any():
            out[mask] += residual / mask.sum()
        return out / out.sum()

    def logits(self, pos, vocab=256):
        return torch.log(self.distribution(pos, vocab).clamp_min(1e-12))


if __name__ == "__main__":
    meta = os.path.expanduser("~/nanlm-data/tok32k_mixed/meta.json")
    trie, meta = ByteTrie.from_meta(meta)
    V = meta["vocab_size"]
    torch.manual_seed(0)
    logits = torch.randn(4, V)
    targets = torch.randint(0, V, (4,))
    verify_token_byte_equivalence(logits, targets, trie)
    d, m = trie.byte_distribution(logits[0], b"")
    print("byte dist", d.shape, "prefix mass", float(m), "sum", float(d.sum()))
    print("BYTE LAYER SELFTEST OK")
