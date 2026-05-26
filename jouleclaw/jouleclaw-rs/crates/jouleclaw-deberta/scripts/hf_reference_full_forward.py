#!/usr/bin/env python3
"""Dump end-to-end HF reference outputs for full-forward verification:

  - encoder_last_hidden  — [L, hidden] tensor after the 24-layer encoder
  - pooled_output         — [hidden] tensor after pooler.dense + GELU
  - logits                — [num_labels=3] tensor from the NLI head
  - predicted_label_id    — argmax of logits
  - predicted_label_name  — id2label[argmax]
  - probabilities         — softmax(logits)

Verification target for Phase 4f: the Rust full forward pass must
match logits within fp16-storage upcast tolerance.

Usage:
    python3 crates/joule-deberta/scripts/hf_reference_full_forward.py

Writes:
    crates/joule-deberta/fixtures/full_forward_simple_entail.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore

MODEL_DIR = Path("models/deberta-v3-large-mnli")
OUT_PATH = Path("crates/joule-deberta/fixtures/full_forward_simple_entail.json")

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

    with torch.no_grad():
        # Run the encoder to get the last hidden state.
        encoder_out = model.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        encoder_last_hidden = encoder_out.last_hidden_state  # [1, L, H]

        # Pooler: CLS token → dense → GELU.
        pooled = model.pooler(encoder_last_hidden)  # [1, H]

        # Classifier: pooled → linear → [num_labels].
        logits = model.classifier(pooled)  # [1, num_labels]
        probs = torch.softmax(logits, dim=-1)

    logits_list = logits.squeeze(0).tolist()
    probs_list = probs.squeeze(0).tolist()
    argmax_idx = int(logits.argmax(dim=-1).item())
    id2label = model.config.id2label
    label_name = id2label[argmax_idx] if argmax_idx in id2label else id2label[str(argmax_idx)]

    seq_len, hidden = encoder_last_hidden.shape[1], encoder_last_hidden.shape[2]

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
            "num_labels": logits.shape[-1],
            "encoder_last_hidden_flat":
                encoder_last_hidden.squeeze(0).flatten().tolist(),
            "pooled_output_flat": pooled.squeeze(0).tolist(),
            "logits": logits_list,
            "probabilities": probs_list,
            "predicted_label_id": argmax_idx,
            "predicted_label_name": label_name,
            "id2label": {str(k): v for k, v in id2label.items()},
        }, f)
    print(
        f"wrote {OUT_PATH}\n"
        f"  logits={logits_list}\n"
        f"  probs={[f'{p:.4f}' for p in probs_list]}\n"
        f"  prediction: {label_name} (id={argmax_idx})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
