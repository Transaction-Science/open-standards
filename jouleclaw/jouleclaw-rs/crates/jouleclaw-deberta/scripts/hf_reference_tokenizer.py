#!/usr/bin/env python3
"""Generate the Rust-side tokenizer fixture by running HF's
AutoTokenizer over a fixed list of (premise, hypothesis) pairs and
dumping the canonical token ids + attention masks to JSON.

The Rust tests load that JSON and assert byte-identical output from
our tokenizer wrapper. Anytime we adjust normalization, special-token
handling, or template processing, we re-run this and the diff is
the contract change.

Usage:
    python3 crates/joule-deberta/scripts/hf_reference_tokenizer.py

Writes:
    crates/joule-deberta/fixtures/tokenizer_pairs.json

Requires:
    transformers, sentencepiece (both already installed locally per
    the Phase 4a setup).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from transformers import AutoTokenizer  # type: ignore

MODEL_DIR = Path("models/deberta-v3-large-mnli")
OUT_PATH = Path("crates/joule-deberta/fixtures/tokenizer_pairs.json")

# Pairs picked to exercise: short canonical NLI, multi-sentence
# premise, punctuation/contraction edge cases, all-uppercase, mixed
# scripts (none of these models support extended Unicode in NLI but
# tokenization shouldn't break), and a "Paris is the capital of
# France" pair that mirrors the spec's running example.
TEST_PAIRS = [
    {
        "name": "simple_entail",
        "premise": "Paris is the capital of France.",
        "hypothesis": "France's capital is Paris.",
    },
    {
        "name": "simple_contradict",
        "premise": "The cat sat on the mat.",
        "hypothesis": "There was no cat anywhere.",
    },
    {
        "name": "neutral",
        "premise": "Marie Curie won the Nobel Prize in Physics in 1903.",
        "hypothesis": "She also worked on radioactive materials.",
    },
    {
        "name": "contraction_punct",
        "premise": "It's the world's biggest waterfall, isn't it?",
        "hypothesis": "Niagara Falls is the world's largest waterfall.",
    },
    {
        "name": "all_upper",
        "premise": "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG.",
        "hypothesis": "A fox jumped over a dog.",
    },
    {
        "name": "long_premise_short_hyp",
        "premise": (
            "Deep Space 1 was launched on October 24, 1998, as the first mission "
            "of NASA's New Millennium Program. It tested twelve advanced "
            "technologies, including the Remote Agent autonomous control system, "
            "the first time an AI system had primary command of a spacecraft."
        ),
        "hypothesis": "Deep Space 1 was launched in 1998.",
    },
]


def main() -> int:
    if not MODEL_DIR.exists():
        print(f"ERR: model dir {MODEL_DIR} not found", file=sys.stderr)
        return 1

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    out = {
        "tokenizer_source": str(MODEL_DIR),
        "tokenizer_class": tok.__class__.__name__,
        "vocab_size": tok.vocab_size,
        "model_max_length": getattr(tok, "model_max_length", None),
        "special_tokens": {
            "cls": tok.cls_token,
            "sep": tok.sep_token,
            "pad": tok.pad_token,
            "unk": tok.unk_token,
            "cls_id": tok.cls_token_id,
            "sep_id": tok.sep_token_id,
            "pad_id": tok.pad_token_id,
            "unk_id": tok.unk_token_id,
        },
        "pairs": [],
    }
    for case in TEST_PAIRS:
        # `encode_plus(text, text_pair)` gives us the canonical NLI
        # encoding: [CLS] premise [SEP] hypothesis [SEP]. We don't
        # truncate or pad here — we want the raw, unpadded ids so
        # the Rust side can match exactly.
        enc = tok(
            case["premise"],
            case["hypothesis"],
            add_special_tokens=True,
            truncation=False,
            padding=False,
            return_attention_mask=True,
            return_tensors=None,
        )
        out["pairs"].append({
            "name": case["name"],
            "premise": case["premise"],
            "hypothesis": case["hypothesis"],
            "token_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "token_strings": tok.convert_ids_to_tokens(enc["input_ids"]),
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"wrote {OUT_PATH} with {len(TEST_PAIRS)} pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
