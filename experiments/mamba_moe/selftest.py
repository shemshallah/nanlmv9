#!/usr/bin/env python3
"""Shape/backward self-test for every MoE/retrieval/gating variant."""
import numpy as np
import torch
import torch.nn.functional as F

from hybrid import ContextGate, NGramPrior, mixture_logits
from model import MambaMoE, count_params
from retrieval import KNNRetriever, MultiMatchIndex


def main():
    vocab = 128
    x = torch.randint(0, vocab, (2, 6))
    y = torch.randint(0, vocab, (2, 6))
    for pos in ("none", "bottom", "ffn", "both"):
        for pair in (False, True):
            m = MambaMoE(
                vocab, d_model=32, n_layers=1, d_state=8, n_experts=8,
                rank=4, top_k=2, shared_hidden=32, moe_position=pos,
                use_shared=True, pair_matrix=pair,
            )
            logits, aux = m(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
            loss.backward()
            aux_val = float(aux.detach()) if torch.is_tensor(aux) else aux
            print(f"{pos:6s} pair={pair!s:5s} params={count_params(m):6d} aux={aux_val:.3f}")

    prior = NGramPrior.from_tokens(
        np.array([1, 2, 3, 4, 1, 2, 5, 6, 1, 2, 7], dtype=np.int64),
        order=2, vocab_size=vocab, max_tokens=100,
    )
    pl = prior.logits(torch.tensor([1, 2, 3]))
    gate = ContextGate(n_sources=2, hidden=16, extra_features=1)
    mixed, weights = mixture_logits(
        torch.zeros(3, vocab), pl, gate, 1.0, extra=torch.zeros(3, 1)
    )
    assert pl.shape == (3, vocab) and mixed.shape == (3, vocab)
    assert weights.shape == (3, 2)

    mm = MultiMatchIndex(order=2, vocab_size=vocab)
    mm.build(np.array([10, 20, 30, 40, 10, 20, 30, 50], dtype=np.int64), max_tokens=100)
    assert mm.logits(torch.tensor([[10, 20], [30, 40]])).shape == (2, vocab)

    knn = KNNRetriever(vocab, k=2)
    knn.build(torch.randn(10, 4), torch.randint(0, vocab, (10,)))
    assert knn.logits(torch.randn(3, 4)).shape == (3, vocab)
    print("SELFTEST OK")


if __name__ == "__main__":
    main()
