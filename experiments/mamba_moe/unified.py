#!/usr/bin/env python3
"""Unified byte-first multi-rate language model.

One graph:

  bytes -> byte SSM -> boundary head
                      -> gather word-end states -> byte->word feedback
  word tokens -> word/concept Mamba/MoE (bottom/top sandwich + pair matrices)
                      -> past-only word state -> word->char bridge
  bytes -> char SSM conditioned on past word state -> char logits
  word logits -> token trie -> exact normalized byte likelihood
  char/CM/retrieval logits -> calibrated fusion gate

The word branch is the normalized sequence likelihood.  The char branch is a
conditioned byte decoder (OOV/spelling), and CM/retrieval are experts.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from byte_layer import ByteTrie
from model import MambaBlock, MambaMoE, count_params
from retrieval import MultiMatchIndex, KNNRetriever


class BoundaryHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 1)
        )

    def forward(self, byte_hidden):
        return self.net(byte_hidden).squeeze(-1)


class ByteEncoder(nn.Module):
    def __init__(self, d_model=256, layers=2, d_state=16, scan_mode="chunked", scan_chunk=4):
        super().__init__()
        self.embed = nn.Embedding(256, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, ffn_mode="dense",
                       scan_mode=scan_mode, scan_chunk=scan_chunk)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, byte_ids):
        x = self.embed(byte_ids)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class CalibratedFusion(nn.Module):
    """Calibrated multi-source gate with source dropout and weight caps."""

    def __init__(self, n_sources=3, hidden=64, extra_features=1):
        super().__init__()
        self.n_sources = n_sources
        self.extra_features = extra_features
        self.net = nn.Sequential(
            nn.Linear(n_sources * 2 + extra_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_sources),
        )

    def forward(self, sources, extra=None, training=False, source_drop=0.1,
                cap=0.9):
        stats = []
        for s in sources:
            p = F.softmax(s, -1)
            stats += [p.max(-1).values.unsqueeze(-1),
                      -(p * F.log_softmax(s, -1)).sum(-1).unsqueeze(-1)]
        feat = torch.cat(stats, -1)
        if extra is not None:
            feat = torch.cat([feat, extra], -1)
        elif self.extra_features:
            feat = torch.cat([feat, feat.new_zeros(feat.shape[0], self.extra_features)], -1)
        w = F.softmax(self.net(feat), -1)
        if training and source_drop > 0:
            # Drop each non-neural source independently; this prevents the
            # gate from permanently ignoring the neural model.
            for i in range(1, self.n_sources):
                mask = (torch.rand(w.shape[0], 1, device=w.device) >= source_drop)
                w = w * torch.cat([torch.ones_like(w[:, :1])] * i +
                                  [mask] +
                                  [torch.ones_like(w[:, :1])] * (self.n_sources - i - 1), -1)
                w = w / w.sum(-1, keepdim=True).clamp_min(1e-8)
        if cap < 1.0:
            w = w.clamp(max=cap)
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-8)
        mixed = sum(w[:, i:i + 1] * s for i, s in enumerate(sources))
        ent = -(w * torch.log(w.clamp_min(1e-8))).sum(-1).mean()
        return mixed, w, ent


class CMExpert:
    """Thin adapter exposing CMBytePrior.logits as a fusion source."""
    def __init__(self, prior):
        self.prior = prior

    def logits(self, pos, vocab=256):
        return self.prior.logits(pos, vocab)


class RetrievalExpert:
    """Thin adapter for a causal retrieval backend (multimatch/suffix/kNN)."""
    def __init__(self, index):
        self.index = index

    def logits(self, contexts):
        return self.index.logits(contexts)


def fusion_features(cm_logits=None, ret_logits=None, match_len=None):
    """Extra gate features: match length bucket, optional source max/entropy."""
    feats = []
    if match_len is not None:
        m = torch.as_tensor(match_len).float().reshape(-1, 1)
        feats.append((m / 4096.0).clamp(0, 1))
    return torch.cat(feats, -1) if feats else None


class UnifiedByteLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=256,
        char_model=64,
        byte_layers=2,
        word_layers=1,
        char_layers=1,
        d_state=16,
        experts=64,
        rank=8,
        scan_mode="chunked",
        scan_chunk=4,
        mirror_points=256,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.byte_encoder = ByteEncoder(d_model, byte_layers, d_state, scan_mode, scan_chunk)
        self.boundary = BoundaryHead(d_model)
        self.byte_to_word = nn.Linear(d_model, d_model, bias=False)
        self.word = MambaMoE(
            vocab_size, d_model=d_model, n_layers=word_layers, d_state=d_state,
            n_experts=experts, rank=rank, top_k=2, shared_hidden=2 * d_model,
            moe_position="ffn", pair_matrix=True, sandwich=True,
            local_context=4, scan_mode=scan_mode, scan_chunk=scan_chunk,
            mirror_points=mirror_points,
        )
        self.word_to_char = nn.Linear(d_model, char_model, bias=False)
        self.char = MambaMoE(
            256, d_model=char_model, n_layers=char_layers, d_state=d_state,
            n_experts=experts, rank=rank, top_k=2, shared_hidden=2 * char_model,
            moe_position="ffn", pair_matrix=True, sandwich=False,
            scan_mode=scan_mode, scan_chunk=scan_chunk,
        )
        self.char_head_fusion = CalibratedFusion(3, 64, 1)
        self.norm = nn.LayerNorm(char_model)

    def gather_word_ends(self, byte_hidden, token_lengths):
        ends = torch.cumsum(token_lengths, dim=1) - 1
        ends = ends.clamp(0, byte_hidden.shape[1] - 1)
        idx = ends.unsqueeze(-1).expand(-1, -1, byte_hidden.shape[-1])
        return byte_hidden.gather(1, idx), ends

    def forward(self, byte_ids, token_ids, token_lengths):
        """byte_ids [B,Lb], token_ids [B,Lt], token_lengths [B,Lt] summing to Lb."""
        byte_h = self.byte_encoder(byte_ids)
        boundary_logits = self.boundary(byte_h)  # [B,Lb]
        end_feats, ends = self.gather_word_ends(byte_h, token_lengths)
        word_extra = self.byte_to_word(end_feats)
        word_hidden = self.word.encode(token_ids, extra=word_extra)
        word_logits = self.word.head(word_hidden)  # [B,Lt,V]

        # Past-only word state for the char decoder: shift by one token.
        past = torch.cat([torch.zeros_like(word_hidden[:, :1]), word_hidden[:, :-1]], dim=1)
        if past.shape[1] != token_lengths.shape[1]:
            past = past[:, : token_lengths.shape[1]]
        past_flat = past.reshape(-1, past.shape[-1])
        ctx_flat = torch.repeat_interleave(past_flat, token_lengths.reshape(-1), dim=0)
        ctx = ctx_flat.reshape(byte_ids.shape[0], -1, past.shape[-1])
        if ctx.shape[1] > byte_ids.shape[1]:
            ctx = ctx[:, : byte_ids.shape[1]]
        elif ctx.shape[1] < byte_ids.shape[1]:
            pad = ctx[:, -1:].expand(-1, byte_ids.shape[1] - ctx.shape[1], -1)
            ctx = torch.cat([ctx, pad], dim=1)
        char_logits, _ = self.char(byte_ids, extra=self.word_to_char(ctx))
        return {
            "word_logits": word_logits,
            "char_logits": char_logits,
            "boundary_logits": boundary_logits,
            "word_hidden": word_hidden,
            "byte_hidden": byte_h,
            "word_ends": ends,
        }

    def segment(self, byte_ids, threshold=0.5):
        """Predicted-boundary forward: returns per-batch segment lengths."""
        h = self.byte_encoder(byte_ids)
        logits = self.boundary(h)
        return boundary_lengths(logits, threshold), logits

    def fuse_bytes(self, char_logits, cm_logits=None, ret_logits=None,
                   extra=None, training=False, source_drop=0.1, cap=0.9):
        sources = [char_logits]
        if cm_logits is not None:
            sources.append(cm_logits)
        if ret_logits is not None:
            sources.append(ret_logits)
        while len(sources) < 3:
            sources.append(torch.zeros_like(char_logits))
        shape = sources[0].shape
        flat = [s.reshape(-1, s.shape[-1]) for s in sources]
        ef = extra.reshape(-1, extra.shape[-1]) if extra is not None else None
        mixed, w, ent = self.char_head_fusion(flat, extra=ef, training=training,
                                              source_drop=source_drop, cap=cap)
        return mixed.reshape(shape), w, ent


def normalized_word_bpb(word_logits, targets, bytes_per_token):
    ce = F.cross_entropy(word_logits.reshape(-1, word_logits.shape[-1]),
                        targets.reshape(-1), reduction="sum")
    nbytes = bytes_per_token.reshape(-1).sum()
    return ce / math.log(2) / nbytes, int(nbytes)


def boundary_loss(boundary_logits, lengths):
    gold = torch.zeros_like(boundary_logits)
    ends = (torch.cumsum(lengths, 1) - 1).clamp(0, boundary_logits.shape[1] - 1)
    gold.scatter_(1, ends, 1.0)
    return F.binary_cross_entropy_with_logits(boundary_logits, gold)


def predict_boundaries(boundary_logits, threshold=0.5):
    """Causal boundary prediction. Returns bool mask [B,Lb] with last byte forced."""
    mask = torch.sigmoid(boundary_logits) > threshold
    mask[:, -1] = True
    return mask


def boundary_lengths(boundary_logits, threshold=0.5):
    """Convert predicted boundary mask into per-token byte lengths."""
    mask = predict_boundaries(boundary_logits, threshold).cpu()
    out = []
    for row in mask:
        idx = torch.nonzero(row, as_tuple=False).flatten().tolist()
        prev = -1
        lens = []
        for i in idx:
            lens.append(i - prev)
            prev = i
        out.append(lens)
    return out


def tune_boundary_threshold(boundary_logits, gold_lengths, thresholds=(0.2,0.3,0.4,0.5,0.6,0.7,0.8)):
    best = None
    for t in thresholds:
        m = boundary_metrics(boundary_logits, gold_lengths, t)
        if best is None or m["boundary_f1"] > best["boundary_f1"]:
            best = dict(m); best["threshold"] = t
    return best


def boundary_metrics(boundary_logits, gold_lengths, threshold=0.5):
    mask = predict_boundaries(boundary_logits, threshold)
    gold = torch.zeros_like(mask)
    ends = (torch.cumsum(gold_lengths, 1) - 1).clamp(0, mask.shape[1] - 1)
    gold.scatter_(1, ends, True)
    tp = (mask & gold).sum().item()
    fp = (mask & ~gold).sum().item()
    fn = (~mask & gold).sum().item()
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"boundary_precision": prec, "boundary_recall": rec, "boundary_f1": f1}


def build_unified(vocab_size=32000, d_model=256, char_model=64, **kw):
    return UnifiedByteLM(vocab_size, d_model=d_model, char_model=char_model, **kw)


if __name__ == "__main__":
    B, Lt, Lb = 2, 8, 32
    lengths = torch.randint(1, 6, (B, Lt))
    lengths[:, -1] = Lb - lengths[:, :-1].sum(1)
    if (lengths <= 0).any():
        lengths[:, -1] = 1
    model = build_unified(vocab_size=128, d_model=32, char_model=16,
                          byte_layers=1, word_layers=1, char_layers=1,
                          experts=8, scan_mode="chunked", scan_chunk=4)
    byte_ids = torch.randint(0, 256, (B, Lb))
    token_ids = torch.randint(0, 128, (B, Lt))
    out = model(byte_ids, token_ids, lengths)
    print("word", tuple(out["word_logits"].shape),
          "char", tuple(out["char_logits"].shape),
          "boundary", tuple(out["boundary_logits"].shape),
          "params", f"{count_params(model):,}")
    # trim word targets to the number of tokens actually used
    word_t = torch.randint(0, 128, (B, Lt))
    bpb, nb = normalized_word_bpb(out["word_logits"][:, :Lt], word_t, lengths)
    print("normalized word bpb", round(float(bpb), 4), "bytes", nb)
    print("boundary loss", float(boundary_loss(out["boundary_logits"], lengths)))
    cm = torch.randn(B, Lb, 256)
    ret = torch.randn(B, Lb, 256)
    mixed, w, ent = model.fuse_bytes(out["char_logits"], cm, ret,
                                     extra=torch.zeros(B, Lb, 1), training=True)
    print("fused", tuple(mixed.shape), "weights", tuple(w.shape),
          "entropy", round(float(ent), 4))
    idx = MultiMatchIndex(order=2, vocab_size=256)
    idx.build(torch.randint(0, 256, (256,)).numpy(), max_tokens=256)
    ctx = byte_ids[:, :4].reshape(-1, 1)
    # contexts need order columns; duplicate the byte id as a stand-in
    ctx = torch.cat([ctx, ctx], -1)
    ret = idx.logits(ctx).reshape(B, 4, 256)
    # pad retrieval logits to byte positions for the fusion test
    ret_full = torch.zeros(B, Lb, 256)
    ret_full[:, :4] = ret[:, :4]
    mixed2, w2, e2 = model.fuse_bytes(out["char_logits"], cm, ret_full,
                                      extra=torch.zeros(B, Lb, 1), training=False)
    print("retrieval fusion", tuple(mixed2.shape), "weights", tuple(w2.shape))
    print("UNIFIED SELFTEST OK")
