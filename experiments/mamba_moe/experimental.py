#!/usr/bin/env python3
"""Experimental implementations not yet promoted to the stable master.

Promotion rule: a variant moves into master.py only after a matched-compute
ablation beats the stable stack and the gain is not explained by a proxy.
"""
import math

import torch
import torch.nn.functional as F

from byte_layer import ByteTrie, CMBytePrior
from hybrid import ContextGate, mixture_logits
from model import MambaMoE, count_params


class ByteResidualLoss:
    """Train a token model through the token->byte trie + real CM prior.

    For each target token, walk its bytes.  At each byte:
      neural byte distribution = trie mass ratio from token logits
      CM distribution          = dumped v9 top-k byte prior at that byte
      gate                     = ContextGate(neural, CM)
      loss                     = -log P(true byte)
    """

    def __init__(self, trie: ByteTrie, cm_prior: CMBytePrior, gate=None, vocab_bytes=None):
        self.trie = trie
        self.cm_prior = cm_prior
        self.gate = gate
        self.vocab_bytes = vocab_bytes

    def __call__(self, token_logits, target_ids, byte_offsets, extra=None, cm_weight=1.0):
        # token_logits: [N, V], target_ids: [N], byte_offsets: [N] first byte pos
        if extra is not None:
            extra = extra.to(token_logits.device)
        nll = token_logits.new_zeros(())
        nbytes = 0
        for i in range(token_logits.shape[0]):
            target = int(target_ids[i])
            tbytes = self.vocab_bytes[target]
            prefix = b""
            for j, b in enumerate(tbytes):
                nd, mass = self.trie.byte_distribution(token_logits[i], prefix)
                if float(mass) <= 0:
                    nll = nll + 20.0
                    nbytes += 1
                    prefix += bytes([b])
                    continue
                cm = self.cm_prior.logits(int(byte_offsets[i]) + j).to(token_logits.device)
                nn = torch.log(nd.clamp_min(1e-12))
                if self.gate is not None:
                    ex = None
                    if extra is not None:
                        ex = extra[i : i + 1]
                    mixed, _ = mixture_logits(nn[None], cm[None], self.gate, cm_weight, extra=ex)
                    logp = mixed[0, int(b)]
                else:
                    logp = nn[int(b)]
                nll = nll - logp
                nbytes += 1
                prefix += bytes([b])
        return nll / max(1, nbytes), nbytes


class WordCharMamba(torch.nn.Module):
    """Word-level Mamba + char-level Mamba sidecar, custom SSM only.

    The word branch predicts word/BPE tokens.  The char branch predicts bytes
    for byte-fallback/OOV tokens.  This is the scaffold for a true
    word+character hierarchy; the byte branch can also be evaluated on the
    token->byte trie.
    """

    def __init__(self, vocab_size, d_model=256, char_model=64, layers=2, n_experts=64):
        super().__init__()
        self.word = MambaMoE(
            vocab_size, d_model=d_model, n_layers=layers, n_experts=n_experts,
            rank=8, top_k=2, shared_hidden=2 * d_model, moe_position="ffn",
            pair_matrix=True, sandwich=True,
        )
        self.char = MambaMoE(
            256, d_model=char_model, n_layers=layers, n_experts=n_experts,
            rank=8, top_k=2, shared_hidden=2 * char_model, moe_position="ffn",
            pair_matrix=True, sandwich=False,
        )
        self.word_to_char = torch.nn.Linear(d_model, char_model, bias=False)

    def forward(self, token_ids, byte_ids, token_lengths=None):
        word_hidden = self.word.encode(token_ids)
        char_extra = None
        if token_lengths is not None:
            ctx = torch.repeat_interleave(word_hidden, token_lengths, dim=1)
            if ctx.shape[1] > byte_ids.shape[1]:
                ctx = ctx[:, : byte_ids.shape[1]]
            elif ctx.shape[1] < byte_ids.shape[1]:
                pad = ctx[:, -1:].expand(-1, byte_ids.shape[1] - ctx.shape[1], -1)
                ctx = torch.cat([ctx, pad], dim=1)
            char_extra = self.word_to_char(ctx)
        word_logits, word_aux = self.word(token_ids)
        char_logits, char_aux = self.char(byte_ids, extra=char_extra)
        return word_logits, char_logits, word_aux + char_aux

    def params(self):
        return count_params(self)


class DeepRetrievalFusion:
    """Confidence-weighted neural + prior + retrieval fusion."""

    def __init__(self, temperature=1.0):
        self.temperature = temperature

    def __call__(self, nn_logits, prior_logits=None, ret_logits=None):
        sources = [nn_logits]
        if prior_logits is not None:
            sources.append(prior_logits)
        if ret_logits is not None:
            sources.append(ret_logits)
        stats = []
        for s in sources:
            p = F.softmax(s, -1)
            stats.append(p.max(-1).values)
        w = F.softmax(torch.stack(stats, -1) * self.temperature, -1)
        out = sum(w[:, i : i + 1] * s for i, s in enumerate(sources))
        return out, w


CONTEXT_SETS = {
    "local4": {"local_context": 4},
    "local8": {"local_context": 8},
    "local16": {"local_context": 16},
    "local32": {"local_context": 32},
    "local64": {"local_context": 64},
    "retrieval_multi": {"retrieval": "multimatch_multi"},
    "retrieval_multi8": {"retrieval": "multimatch_multi8"},
    "sandwich": {"sandwich": True},
    "top_extension": {"sandwich": True, "top_extension": True},
}


def smoke():
    meta = __import__("json").load(open(__import__("os").path.expanduser(
        "~/nanlm-data/tok32k_mixed/meta.json")))
    vocab = [bytes.fromhex(h) for h in meta["vocab_hex"]]
    trie = ByteTrie(vocab)
    torch.manual_seed(0)
    logits = torch.randn(4, meta["vocab_size"])
    targets = torch.randint(0, meta["vocab_size"], (4,))
    # synthetic CM dump: uniform-ish top-8 around the true bytes
    ids = torch.randint(0, 256, (64, 8)).numpy().astype("uint8")
    probs = torch.softmax(torch.randn(64, 8), -1).numpy().astype("float32")
    prior = CMBytePrior(ids, probs, torch.zeros(64, dtype=torch.uint8).numpy(), 8)
    gate = ContextGate(n_sources=2, hidden=16, extra_features=1)
    loss_fn = ByteResidualLoss(trie, prior, gate, vocab)
    offsets = torch.arange(0, 4 * 8, 8)
    loss, nbytes = loss_fn(logits, targets, offsets, extra=torch.zeros(4, 1))
    print("byte residual loss", float(loss), "bytes", nbytes)
    wc = WordCharMamba(meta["vocab_size"], d_model=32, char_model=16, layers=1, n_experts=4)
    print("word+char params", wc.params())
    tok = torch.randint(0, meta["vocab_size"], (1, 8))
    byt = torch.randint(0, 256, (1, 16))
    wl, cl, aux = wc(tok, byt)
    print("word logits", tuple(wl.shape), "char logits", tuple(cl.shape))
    print("EXPERIMENTAL SELFTEST OK")


if __name__ == "__main__":
    smoke()
