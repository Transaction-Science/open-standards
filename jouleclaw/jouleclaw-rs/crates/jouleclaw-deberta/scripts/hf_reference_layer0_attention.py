#!/usr/bin/env python3
"""Dump the tensor that comes out of layer-0's attention sub-block —
the input to the FFN sub-block. Specifically:

    attn_out = SelfOutput(disentangled_attn(embedding_output, ...),
                          residual=embedding_output)
             = LayerNorm(attn_output_dense @ context + bias + embedding_output)

This is the verification target for Phase 4e: the Rust implementation
of disentangled attention + residual + LayerNorm must match this
within fp16-tolerance.

Also dumps:
    - the attention_mask 4D shape (for cross-checking my mask logic)
    - the relative_pos tensor (the bucketed [L, L] matrix)
    - the rel_embeddings tensor *after* the encoder-level LayerNorm
      (this is what disentangled_attention_bias actually consumes)

Usage:
    python3 crates/joule-deberta/scripts/hf_reference_layer0_attention.py

Writes:
    crates/joule-deberta/fixtures/layer0_attention.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

MODEL_DIR = Path("models/deberta-v3-large-mnli")
OUT_PATH = Path("crates/joule-deberta/fixtures/layer0_attention.json")

PREMISE = "Paris is the capital of France."
HYPOTHESIS = "France's capital is Paris."


def main() -> int:
    if not MODEL_DIR.exists():
        print(f"ERR: model dir {MODEL_DIR} not found", file=sys.stderr)
        return 1

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR), torch_dtype=torch.float32
    )
    model.eval()

    enc = tok(
        PREMISE, HYPOTHESIS,
        add_special_tokens=True, truncation=False, padding=False,
        return_attention_mask=True, return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    encoder = model.deberta.encoder
    emb_layer = model.deberta.embeddings
    layer0 = encoder.layer[0]

    with torch.no_grad():
        # Embedding output (verified-matching in phase 4d).
        word_embeds = emb_layer.word_embeddings(input_ids)
        ln_out = emb_layer.LayerNorm(word_embeds)
        mask_b = attention_mask.unsqueeze(-1).to(ln_out.dtype)
        embedding_output = ln_out * mask_b

        # Encoder-level prep.
        extended_attention_mask = encoder.get_attention_mask(attention_mask)
        relative_pos = encoder.get_rel_pos(embedding_output)
        rel_embeddings = encoder.get_rel_embedding()

        # Layer-0 attention only (not the FFN).
        attn_output, _ = layer0.attention(
            embedding_output,
            extended_attention_mask,
            output_attentions=False,
            query_states=None,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )

    # Strip batch dim.
    embedding_output = embedding_output.squeeze(0)
    attn_output = attn_output.squeeze(0)
    extended_mask_shape = list(extended_attention_mask.shape)
    rel_pos_flat = relative_pos.squeeze(0).flatten().to(torch.int64).tolist()

    seq_len, hidden = attn_output.shape

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump({
            "model_source": str(MODEL_DIR),
            "case_name": "simple_entail",
            "premise": PREMISE,
            "hypothesis": HYPOTHESIS,
            "input_ids": input_ids.squeeze(0).tolist(),
            "attention_mask": attention_mask.squeeze(0).tolist(),
            "seq_len": seq_len,
            "hidden_size": hidden,
            "extended_attention_mask_shape": extended_mask_shape,
            "relative_pos_flat": rel_pos_flat,
            "relative_pos_shape": [relative_pos.shape[1], relative_pos.shape[2]],
            "rel_embeddings_normed_flat":
                rel_embeddings.flatten().tolist(),
            "rel_embeddings_shape": list(rel_embeddings.shape),
            "embedding_output_flat": embedding_output.flatten().tolist(),
            "layer0_attn_output_flat": attn_output.flatten().tolist(),
            "stats": {
                "embed_abs_mean": float(embedding_output.abs().mean()),
                "attn_out_abs_mean": float(attn_output.abs().mean()),
                "attn_out_min": float(attn_output.min()),
                "attn_out_max": float(attn_output.max()),
            },
        }, f)
    print(
        f"wrote {OUT_PATH}: seq_len={seq_len} hidden={hidden} "
        f"attn_out_abs_mean={attn_output.abs().mean():.4f} "
        f"rel_pos_shape={list(relative_pos.shape)} "
        f"rel_embeddings_shape={list(rel_embeddings.shape)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
