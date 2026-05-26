#!/usr/bin/env python3
"""Dump the DeBERTa-v3 embedding-layer output for one canonical NLI
pair so the Rust implementation can be verified numerically.

For v3, the embedding layer is:
    embeddings = LayerNorm(word_embeddings[input_ids])
    embeddings *= attention_mask.unsqueeze(-1)  # mask out PAD positions

(No absolute position embeddings — position_biased_input=False; no
token-type embeddings — type_vocab_size=0.)

We run the HF model up to *just after* the embedding LayerNorm and
mask multiply, then save:
    - input_ids
    - attention_mask
    - embedding_output  (the tensor the encoder will consume)

Computed in fp32 (model is upcast on load) so the Rust comparison
isn't fighting fp16 quantization.

Usage:
    python3 crates/joule-deberta/scripts/hf_reference_embedding.py

Writes:
    crates/joule-deberta/fixtures/embedding_simple_entail.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

MODEL_DIR = Path("models/deberta-v3-large-mnli")
OUT_PATH = Path("crates/joule-deberta/fixtures/embedding_simple_entail.json")

# Same first pair from the tokenizer fixture — keeps the verification
# chain coherent.
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

    # Walk the embedding sub-module by hand so we capture the exact
    # tensor the encoder receives — not the raw word embeddings, but
    # after LayerNorm + mask multiply.
    emb_layer = model.deberta.embeddings
    with torch.no_grad():
        word_embeds = emb_layer.word_embeddings(input_ids)
        # v3: position_biased_input=False, type_vocab_size=0, so we
        # skip both branches and go straight to LayerNorm.
        ln_out = emb_layer.LayerNorm(word_embeds)
        # Mask broadcast: [B, S] -> [B, S, 1] times [B, S, H].
        mask_b = attention_mask.unsqueeze(-1).to(ln_out.dtype)
        embedding_output = ln_out * mask_b
        # No dropout at eval time.

    # Strip the batch dim — Rust side computes over a single sequence.
    embedding_output = embedding_output.squeeze(0)
    input_ids_list = input_ids.squeeze(0).tolist()
    attention_mask_list = attention_mask.squeeze(0).tolist()

    # Flatten [seq_len, hidden_size] to row-major flat list.
    seq_len, hidden = embedding_output.shape
    flat = embedding_output.flatten().tolist()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        json.dump({
            "model_source": str(MODEL_DIR),
            "case_name": "simple_entail",
            "premise": PREMISE,
            "hypothesis": HYPOTHESIS,
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "seq_len": seq_len,
            "hidden_size": hidden,
            "dtype": "float32",
            "embedding_output_flat": flat,
            # Sanity stats so the Rust side can see what it's
            # comparing against without inspecting the full tensor.
            "stats": {
                "min": float(embedding_output.min()),
                "max": float(embedding_output.max()),
                "mean": float(embedding_output.mean()),
                "abs_mean": float(embedding_output.abs().mean()),
            },
        }, f)
    print(
        f"wrote {OUT_PATH}: seq_len={seq_len} hidden={hidden} "
        f"abs_mean={embedding_output.abs().mean():.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
