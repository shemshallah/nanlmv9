# Hybrid Mamba/MoE experiments

This directory is the 32k-tokenizer + neural/hybrid branch that sits beside the
v9 C context-mixing core.  It is deliberately small enough to run on Termux for
smoke tests, while matching the architecture shape that can scale on a GPU.

## 1. What is an SSM?

A state-space model (SSM) is a linear recurrence:

```
h_t = A h_{t-1} + B x_t
y_t = C h_t + D x_t
```

`h_t` is a hidden state, `x_t` an input token/byte, `y_t` the output.  In a
**selective SSM** (Mamba), `A`, `B`, `C`, and the step size `Δ` are functions of
the current input:

```
h_t = exp(Δ_t A) h_{t-1} + Δ_t B_t x_t
y_t = C_t h_t + D x_t
```

That lets the model learn to remember or forget selectively.  The important
properties are:

- **O(n) inference** in sequence length (a recurrent state, not a growing KV cache).
- **Parallel scan** during training, so it is not strictly sequential on a GPU.
- Long context without quadratic attention.
- Weaker exact retrieval than attention; hybrid SSM + attention/MoE is often best.

Mamba-2 is the structured-state-space-duality version; this prototype uses the
Mamba-1-style selective scan for clarity.

## 2. 256-point Mamba with a 256-expert MoE bottom

Yes, this is coherent:

```
tokens
  -> 256 low-rank experts (top-k router + shared expert)
  -> N selective-SSM blocks (d_model = 256, d_state = 16)
  -> tied 32k output head
```

`model.py` implements exactly that.  The 256 experts are low-rank adapters
(`down: 256 -> 8`, `up: 8 -> 256`) plus one dense shared expert, so 256 experts
cost ~1M parameters instead of ~34M.  A router selects top-2 per token and a
Switch-style load-balancing auxiliary loss keeps experts alive.

### The 32³+32³ question

If you have **two 256-state registers**, their ordered joint state space is:

```
256 × 256 = 65,536 = 2 × 32,768 = 32³ + 32³
```

So **32³+32³ is exactly the ordered-pair space**, including the diagonal where
both registers hold the same value.

If the object is `C(256,2)` — *unordered* pairs — the natural count is:

```
C(256,2) = 32,640
```

That is close to 32³ = 32,768 but not equal (difference 128).  Adding the 256
diagonal states gives 32,896.  So:

- **ordered transitions between two 256 registers:** 65,536 = 32³+32³
- **unordered pairs:** 32,640
- **unordered pairs + diagonal:** 32,896

A 65,536-entry pair table is a very reasonable **expert-pair adjacency / router
bias** in a 256-expert MoE.  A symmetric 32,640-entry matrix is the cheaper
version if expert order does not matter.

## 3. Tokenizer and data

`tokenizer_bpe.py` builds a 32,000-token vocabulary:

- ids 0..255: raw bytes (exact fallback)
- ids 256..2255: the 2,000 v9 BPE merges
- remaining ids: top whole words, including a leading space (`" the"`)

The tokenizer is a bijection on bytes, so **bits-per-byte is exact** and there
is no `<unk>`.

NaNRouter data used (see `fetch_data.sh`):

- wordlists: google-10000-english, david47k top words, dwyl words, en-wl, wordfreq, wordset
- WikiText-103 raw train/valid/test
- domain corpora: humanities, chemistry, standards
- 150 MB mixed slice: 100 MB wt103 + 20 MB humanities + 10 MB chemistry + 20 MB standards

Measured tokenization:

| split | tokens | bytes | bytes/token |
|---|---:|---:|---:|
| mixed train (150 MB) | 46,876,778 | 150,000,004 | 3.200 |
| wt2 valid | 304,363 | 1,121,681 | 3.685 |
| wt2 test | 349,143 | 1,256,449 | 3.599 |

On wt2 valid, whole-word tokens cover 55.0% of tokens and 85.0% of bytes;
BPE pieces cover 5.4% of tokens and 4.3% of bytes; raw-byte fallback covers
39.5% of tokens but only 10.7% of bytes (mostly spaces, newlines, punctuation).

Commands:

```sh
bash fetch_data.sh
python3 tokenizer_bpe.py \
  --vocab-texts ~/nanlm-data/corpora/mixed_train.txt \
    ~/nanlm-data/wordlists/google-10000-english/google-10000-english/google-10000-english.txt \
    ~/nanlm-data/wordlists/david47k-top-english-wordlists/david47k-top-english-wordlists/top_english_words_lower_100000.txt \
    ~/nanlm-data/wordlists/dwyl-english-words/dwyl-english-words/words_alpha.txt \
  --train-texts ~/nanlm-data/corpora/mixed_train.txt \
  --valid-texts ~/nanlm-test/grok_v9/corpora/wt2_valid.txt \
  --test-texts ~/nanlm-test/grok_v9/corpora/wt2_test.txt \
  --out ~/nanlm-data/tok32k_mixed --vocab-size 32000
```

**Wordlists are not natural text.**  They are useful for vocabulary, spelling,
rare-word completion, and a word cache; training a language model only on a
wordlist concatenation teaches orthography, not grammar.  The natural-language
training signal here comes from wt103 + the domain corpora; the wordlists shape
the 32k vocabulary.

## 4. Train / evaluate

```sh
python3 train.py \
  --data-dir ~/nanlm-data/tok32k_mixed \
  --out ~/nanlm-data/runs/mamba_moe \
  --steps 5000 --batch-size 4 --seq-len 128 \
  --d-model 256 --layers 2 --experts 256 --rank 8 --top-k 2
```

For a quick CPU shape test:

```sh
python3 train.py --smoke --data-dir ~/nanlm-data/tok32k_mixed \
  --out ~/nanlm-data/runs/smoke
```

`train.py` reports **bpc** directly:

```
bpc = sum(NLL) / ln(2) / sum(bytes(target_token))
```

## 5. Ablation framework

`ablate.py` runs the architecture matrix and writes `ablation.json`:

```sh
# phone-sized shape smoke
python3 ablate.py --quick --configs none bottom ffn both ffn_pair \
  ffn_pair_noshared prior_gate retrieval_mm retrieval_suffix retrieval_knn

# GPU / Kaggle smoke
python3 ablate.py --data-dir data/tok32k --out ablations \
  --configs none bottom ffn both ffn_pair ffn_pair_noshared \
    ffn_pair_d128 ffn_pair_d256 ffn_pair_d512 \
    prior_gate retrieval_mm retrieval_suffix retrieval_knn \
  --steps 20 --batch 8 --seq-len 128 --d-model 128 --layers 2 \
  --experts 64 --rank 8 --shared-hidden 256 --device cuda
```

Configs cover:

| config | tests |
|---|---|
| `none` | dense FFN baseline |
| `bottom` | MoE only on the embedding |
| `ffn` | MoE FFN in every SSM block |
| `both` | bottom + block-level MoE |
| `ffn_pair` | block-level MoE + E×E pair matrix |
| `ffn_pair_noshared` | removes the dense shared expert |
| `ffn_pair_d128/d256/d512` | width sweep |
| `prior_gate` | n-gram prior + neural context gate (CM-residual proxy) |
| `retrieval_mm` | multi-match hash retrieval |
| `retrieval_suffix` | suffix-array backend (falls back to multi-match if `pydivsufsort` is absent) |
| `retrieval_knn` | kNN over SSM hidden states |
| `hier_word` | word-tokenizer run (skipped unless `--word-data-dir` exists) |

### Expert pair matrix

`LowRankMoE` integrates an E×E matrix as a mean-field pair adjustment:

```
p = softmax(router_logits)
pair = p @ B                 # B is E x E (symmetrized for C(256,2))
logits = router_logits + softplus(pair_scale) * pair
```

Every expert’s score is adjusted by its expected compatibility with the
current expert distribution, using the full matrix, differentiably.  The same
matrix also gates the selected expert mixture:

```
for each selected pair i<j:
    g = sigmoid(pair_mix_scale * B[e_i, e_j])
    topv_i *= g; topv_j *= 1-g
renormalize(topv)
```

For 256 experts the routing matrix is exactly the **65,536 = 32³+32³**
ordered-pair space; symmetrizing reduces it to the C(256,2) semantics.

### Retrieval

- `MultiMatchIndex`: hash of the last 4 tokens → up to 8 training positions;
  counts the next token across candidates.
- `SuffixArrayRetriever`: optional `pydivsufsort` suffix array over the token
  byte stream; falls back to multi-match.
- `KNNRetriever`: kNN over the model’s SSM hidden states; built after a few
  training steps.

Retrieval is causal and only mixed in at evaluation (`logits + w_ret * log
P_ret`), so it is a clean ablation.

### CM residual / context weights

`hybrid.py` provides:

- `NGramPrior` — unigram/bigram prior with the same interface a v9 CM prior
  would expose.
- `ContextGate` — a small MLP that takes source confidence/entropy plus a
  position feature and learns per-position mixture weights.
- `mixture_logits` — residual mixing (`nn + w*prior`) or learned convex mixing.

This is the fast proxy for “train the neural side of the CM residual” and
“neural context weights for CM”; swapping the n-gram prior for v9 logits is the
next integration step.

### Kaggle smoke

`kaggle/` contains a private GPU notebook that clones the repo, downloads wt2,
builds the 32k tokenizer, and runs the full ablation matrix for 20 steps per
config (a few minutes on a T4).  It is a shape/step smoke, not a converged
model.

## 6. Measured prototype limits

The model and tokenizer run on Termux, but CPU training is the bottleneck:

- full prototype config: **10,444,544 parameters** (32k vocab, d_model=256,
  2 SSM layers, 256 low-rank experts)
- measured full-config forward+backward: **~28 tokens/s** on this phone
- smoke config (2.16M params): forward/backward works
- benchmark config (4.35M params, 32 experts, seq 64, batch 4): **19–144 tok/s**
  depending on system load
- a 50-step sanity run reached **4.66 bpc** on a 512-token valid sample; this
  is not a converged model, it only proves the train/eval path
- at ~28 tok/s, one epoch over 47M tokens is ~19 days on this phone

The full 256-expert model is larger and slower.  This is a GPU/cloud experiment,
not a Termux training run.  The CPU path is still useful for:

- tokenizer/data validation
- architecture smoke tests
- small overfit/debug runs
- CM-sidecar inference and mixing

### Real CM prior residual smoke

`cm_residual_smoke.py` dumps a real v9 CM top-k byte prior, runs the token model
through the token→byte trie, and trains a `ContextGate` on the true bytes.
First 999 bytes of a wt2 valid window:

| source | bpc |
|---|---:|
| neural only (token→byte) | 3.5208 |
| v9 CM only | 1.9848 |
| gated hybrid | **1.7108** |

The learned mean gate weight was **0.209 neural / 0.791 CM**, and every
position put at least 0.389 weight on CM.  This is the real-logit version of
the n-gram proxy; the short sample is a wiring/signal check, not a benchmark.

## 7. Kaggle 20-step GPU smoke results

These are **20-step shape/step smokes**, not converged models.  They exist to
validate every architecture variant and to catch wiring bugs.  Do not compare
these absolute bpc values with the v9 1.49 result.

| config | params | bpc |
|---|---:|---:|
| none | 4.59M | 4.5167 |
| bottom | 4.80M | 4.5216 |
| ffn | 4.74M | 4.5943 |
| both | 4.94M | 4.6091 |
| ffn_pair | 4.75M | 4.5940 |
| ffn_pair_noshared | 4.62M | 4.6417 |
| ffn_pair_d128 | 4.75M | 4.5940 |
| ffn_pair_d256 | 9.90M | 4.5235 |
| ffn_pair_d512 | 21.4M | **4.3371** |
| prior_gate (unigram) | 4.75M | **2.9536** |
| retrieval_mm | 4.75M | **3.7847** |
| hier_word tokenizer | 4.75M | 6.5101 |

Interpretation:

- width helps immediately (d512 > d256 > d128);
- the shared expert helps (`ffn_pair` < `ffn_pair_noshared`);
- the pair matrix is neutral at 20 steps; it needs a longer run;
- the prior/gate and retrieval paths are wired correctly and give the
  expected large smoke-time gains;
- suffix/kNN failed in this version due to CUDA tensor-placement bugs; both are
  fixed and re-run in the next version.

A later **CPU Kaggle smoke** (2 steps, tiny model) validates every backend with
the fixes: `retrieval_knn` 3.604, `retrieval_multi` 3.889, `retrieval_mm` 3.918,
`suffix` 4.454, `prior_gate` 3.979, `local_ctx` 4.597 bpc.  These are wiring
checks, not converged quality comparisons.  The GPU version of the same matrix
is queued behind Kaggle's GPU scheduler.

## 8. 5-minute CPU/GPU presets

`presets.py` defines two shape presets:

- **10M CPU:** 11,559,174 params, `d_model=256`, 2 SSM layers, 256 experts,
  bottom MoE + pair matrix, local context 4.
- **100M GPU:** 112,919,321 params, `d_model=768`, 8 SSM layers, 128 experts,
  FFN MoE + bottom/top sandwich + pair matrices + 256 mirror points,
  local context 4.

Five-minute wall-clock smokes (not converged):

| preset | device | params | steps | eval bpc | eval bytes |
|---|---|---:|---:|---:|---:|
| 10M | CPU (Kaggle) | 11.56M | 200 | 2.6189 | 128 |
| 100M | T4 GPU | 112.92M | 330 | 2.2559 | 512 |

The 100M run used `--amp` bf16 autocast, TF32 matmuls, and the vectorized
low-rank expert path.  These are shape/optimization smokes, not quality claims.

- CPU kernel: https://www.kaggle.com/code/shem42/nanlmv9-10m-cpu-smoke
- 100M GPU kernel: https://www.kaggle.com/code/shem42/nanlmv9-100m-gpu-smoke

## 9. Hybrid with v9 CM

The intended production shape is:

```
P(byte) = sigmoid( w_cm * logit(P_cm) + w_nn * logit(P_nn) + w_ret * logit(P_ret) + b )
```

with the gate conditioned on match length, neural entropy, byte class, word
position, and OOV/rare-word status.  The neural side supplies word/semantic
context; v9 supplies local spelling, morphology, markup, and exact match memory.
The token-level neural distribution can be converted to exact byte
probabilities with a token trie, or the neural model can be trained directly on
bytes.

## 10. What to scale

- 100M–300M parameters on 10B–30B tokens is the first serious target.
- 1B parameters on ~100B tokens is the first frontier-relevant run.
- 7B+ MoE on 1T+ tokens is the actual frontier regime.
- Retrieval (infini-gram / kNN-LM) and online adaptation are the next
  quality levers after the neural backbone is trained.

See the top-level `README.md` for the v9 CM results and the roadmap to <1.30
bpc.
