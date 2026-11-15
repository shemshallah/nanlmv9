# Mamba/MoE/Hybrid TODO

Status: **2026-11-15**.  `[x]` done and smoke-tested; `[~]` implemented but
only smoke-tested / needs a longer run; `[ ]` still open.

## Core architecture

- [x] 32k word + BPE + raw-byte tokenizer (`tokenizer_bpe.py`)
- [x] exact token→byte probability layer (`byte_layer.py`)
  - [x] vocabulary trie
  - [x] exact byte NLL from token logits
  - [x] token/byte equivalence self-test (max error < 1e-6)
- [x] bottom MoE, FFN MoE, both, and dense baseline (`moe_position`)
- [x] shared expert + block-level routed experts
- [x] low-rank 256-expert MoE (vectorized, no Python loop over experts)
- [x] expert-pair matrix in routing (mean-field)
- [x] expert-pair matrix in selected-expert mixture gates
- [x] two pair matrices: bottom concept cloud + top concept layer
- [x] sandwich mirror bridge with 256 learned mirror points between them
- [~] hierarchical concept-cloud → Mamba → top concept layer
  - implemented as bottom MoE + mirror points + Mamba blocks + top MoE
  - needs a longer GPU run before quality claims
- [x] local-context path (previous-token embeddings; 0/4/8/16 sweepable)
- [x] retrieval fusion: add and confidence-gated modes

## Retrieval

- [x] multi-match hash retriever
- [x] multi-order (2/4/6) retriever
- [x] real suffix array via `pydivsufsort` (multi-match fallback)
- [x] kNN over SSM hidden states
- [x] causal, eval-only mixing
- [~] retrieval fusion inside the model (currently logit mixing)

## CM prior / residual

- [x] `nancm -dumpin ... -dump ... -dumptop K` real v9 CM top-k byte dump
- [x] `CMBytePrior` loader with residual smoothing
- [x] real CM prior + token→byte + neural gate smoke
- [x] learned gate relies on CM (mean 0.791 CM weight on the smoke)
- [~] train the full Mamba model on the CM residual (gate-only smoke done)
- [ ] v9 CM prior dump for the full 1.12 MB valid split (currently 8×4k windows)
- [ ] train/validate the hybrid on the full valid split

## Contexts

- [x] v9 match-history eval fix (`-evmatch`)
- [x] online state carry (`-evcarry`)
- [x] 4k evaluation windows
- [x] local token context 0/4/8/16
- [x] multi-order retrieval contexts 2/4/6
- [ ] exhaustive extra CM contexts (sparse, indirect, record, word suffix,
      line position, case-folded word, previous-word+length)
- [ ] context-set sweep until marginal bpc turns negative
- [ ] full context ablation table on full valid

## Training / evaluation

- [x] 10M-class CPU preset (`presets.py`)
- [x] 100M-class GPU preset (112.9M parameters)
- [x] 5-minute wall-clock budget (`--max-seconds`)
- [x] CPU AMP-safe path / GPU AMP option (`--amp`)
- [x] vectorized expert matmuls
- [x] TF32 / float32 matmul precision setup on CUDA
- [x] 10M CPU shape smoke
- [x] 10M CPU 5-minute Kaggle run: 11.56M params, 200 steps, bpc 2.6189
- [x] 100M GPU 5-minute Kaggle run: 112.92M params, 330 steps, bpc 2.2559
- [x] Kaggle CPU ablation smoke (all retrieval backends execute)
- [x] Kaggle GPU 20-step architecture smoke (v4 results recorded)

## Ablations

- [x] `none`, `bottom`, `ffn`, `both`
- [x] shared vs no shared
- [x] d128 / d256 / d512
- [x] pair matrix on/off
- [x] local context
- [x] prior + context gate
- [x] multimatch / multi-order / suffix / kNN
- [x] sandwich
- [ ] longer matched-compute runs for pair matrix and sandwich
- [ ] true word+character hierarchical model with byte-level likelihood

## Word-level hierarchy

- [x] 32k word+BPE+byte tokenizer is itself a hierarchy
- [x] pure-word tokenizer ablation (`hier_word`)
- [~] word-level concept cloud → Mamba (implemented via bottom MoE + mirror)
- [ ] explicit word Transformer + character CM, exact byte likelihood

## Release / repo

- [x] code pushed to `shemshallah/nanlmv9`
- [x] self-test (`selftest.py`)
- [x] README with measured results and caveats
- [ ] longer GPU run and a new dev release with real CM-residual metrics
