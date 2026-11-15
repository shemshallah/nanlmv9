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

## 5. Measured prototype limits

The model and tokenizer run on Termux, but CPU training is the bottleneck:

- smoke config (2.16M params): forward/backward works
- benchmark config (4.35M params, 32 experts, seq 64, batch 4): **19–144 tok/s**
  depending on system load
- at ~20 tok/s, one epoch over 47M tokens is ~27 days on this phone

The full 256-expert model is larger and slower.  This is a GPU/cloud experiment,
not a Termux training run.  The CPU path is still useful for:

- tokenizer/data validation
- architecture smoke tests
- small overfit/debug runs
- CM-sidecar inference and mixing

## 6. Hybrid with v9 CM

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

## 7. What to scale

- 100M–300M parameters on 10B–30B tokens is the first serious target.
- 1B parameters on ~100B tokens is the first frontier-relevant run.
- 7B+ MoE on 1T+ tokens is the actual frontier regime.
- Retrieval (infini-gram / kNN-LM) and online adaptation are the next
  quality levers after the neural backbone is trained.

See the top-level `README.md` for the v9 CM results and the roadmap to <1.30
bpc.
