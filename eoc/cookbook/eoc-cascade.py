"""
eoc-cascade.py — runnable companion to eoc-cascade-with-hf-energy-score.md

A four-stage EOC cascade (cache -> KV -> graph -> neural) with HuggingFace
AI Energy Score integration. Each resolution emits an EOC receipt carrying
the query hash, the answer hash, the resolving stage, the wall-clock duration,
and the joule cost in microjoules.

The neural stage uses microsoft/phi-3-mini-4k-instruct for demonstration.
Production deployments swap in their preferred neural backend; the cascade
contract does not depend on the model choice.

License: CC-BY-4.0 (text) / Apache-2.0 (code).
"""

from __future__ import annotations

import hashlib, json, re, time, uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Optional


# ---------- canonicalization & hashing ----------------------------------------

def canonicalize(q: str) -> str:
    return " ".join(q.strip().lower().split())

def query_hash(q: str) -> str:
    # EOC-1 mandates BLAKE3 in production; sha256 here to keep deps minimal.
    return "sha256:" + hashlib.sha256(canonicalize(q).encode("utf-8")).hexdigest()[:32]


# ---------- energy counters ---------------------------------------------------

try:
    import pyRAPL; pyRAPL.setup(); _RAPL = True
except Exception:
    _RAPL = False

# Calibrated estimator: idle-corrected CPU wattage * elapsed seconds * 1e6.
_CPU_WATTS, _IDLE_WATTS = 15.0, 3.0

@dataclass
class EnergyReport:
    microjoules: int = 0
    method: str = "estimator-v1"
    def report(self) -> dict:
        m = self.method
        return {
            "microjoules": {
                "measured":  self.microjoules if m != "estimator-v1" else 0,
                "estimated": self.microjoules if m == "estimator-v1" else 0,
            },
            "method": m,
        }

@contextmanager
def energy_counter():
    rep, t0 = EnergyReport(), time.monotonic_ns()
    meter = None
    if _RAPL:
        try:
            meter = pyRAPL.Measurement("eoc-stage"); meter.begin(); rep.method = "rapl"
        except Exception:
            meter = None
    try:
        yield rep
    finally:
        elapsed_s = (time.monotonic_ns() - t0) / 1e9
        uj = 0
        if meter is not None:
            try:
                meter.end()
                uj = int(sum(meter.result.pkg or [0]))
            except Exception:
                rep.method = "estimator-v1"
        if rep.method == "estimator-v1":
            uj = int((_CPU_WATTS - _IDLE_WATTS) * elapsed_s * 1e6)
        rep.microjoules = max(uj, 1)


# ---------- stage 1: cache ----------------------------------------------------

CACHE: dict[str, str] = {}

def cache_resolve(q: str) -> Optional[str]:
    return CACHE.get(query_hash(q))

def cache_seed(pairs: list[tuple[str, str]]) -> None:
    for q, a in pairs:
        CACHE[query_hash(q)] = a


# ---------- stage 2: KV (sentence-embedding similarity) -----------------------

_KV_MODEL = None
KV_STORE: list[tuple[Any, str]] = []
KV_THRESHOLD = 0.85

def _kv_model():
    global _KV_MODEL
    if _KV_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _KV_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _KV_MODEL

def kv_seed(pairs: list[tuple[str, str]]) -> None:
    import numpy as np
    m = _kv_model()
    for q, a in pairs:
        KV_STORE.append((np.asarray(m.encode(q, normalize_embeddings=True), dtype="float32"), a))

def kv_resolve(q: str) -> Optional[str]:
    if not KV_STORE:
        return None
    import numpy as np
    e = np.asarray(_kv_model().encode(q, normalize_embeddings=True), dtype="float32")
    sims = np.array([float(np.dot(e, s)) for s, _ in KV_STORE])
    best = int(np.argmax(sims))
    return KV_STORE[best][1] if sims[best] >= KV_THRESHOLD else None


# ---------- stage 3: graph (toy triple store + pattern matcher) ---------------

GRAPH: list[tuple[str, str, str]] = []

_GRAPH_PATTERNS: list[tuple[re.Pattern, Callable[[re.Match], tuple[str, str]]]] = [
    (re.compile(r"^what(?:'s| is) the (\w+(?:\s\w+)*) of (.+?)\??$"),
     lambda m: (m.group(2).strip().lower(), m.group(1).strip().lower())),
    (re.compile(r"^who (founded|invented|wrote|composed|painted) (.+?)\??$"),
     lambda m: (m.group(2).strip().lower(), m.group(1).strip().lower())),
]

def graph_seed(triples: list[tuple[str, str, str]]) -> None:
    GRAPH.extend((s.lower(), p.lower(), o) for s, p, o in triples)

def graph_resolve(q: str) -> Optional[str]:
    qc = canonicalize(q)
    for pat, extract in _GRAPH_PATTERNS:
        m = pat.match(qc)
        if not m:
            continue
        subject, predicate = extract(m)
        for s, p, o in GRAPH:
            if s == subject and p == predicate:
                return o
    return None


# ---------- stage 4: neural fallback (HuggingFace causal LM) ------------------

NEURAL_MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
_tok = _mdl = None

def neural_resolve(q: str) -> str:
    global _tok, _mdl
    if _mdl is None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        _tok = AutoTokenizer.from_pretrained(NEURAL_MODEL_NAME)
        _mdl = AutoModelForCausalLM.from_pretrained(
            NEURAL_MODEL_NAME, torch_dtype=torch.float32,
            device_map="cpu", trust_remote_code=True)
    import torch
    prompt = _tok.apply_chat_template(
        [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    inputs = _tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = _mdl.generate(**inputs, max_new_tokens=64, do_sample=False,
                            pad_token_id=_tok.eos_token_id)
    return _tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


# ---------- cascade -----------------------------------------------------------

_STAGES: list[tuple[str, Callable[[str], Optional[str]]]] = [
    ("cache", cache_resolve), ("kv", kv_resolve),
    ("graph", graph_resolve), ("neural", neural_resolve),
]

def _qid() -> str:
    try: return str(uuid.uuid7())
    except AttributeError: return str(uuid.uuid4())

def cascade(q: str) -> dict:
    qid, qh, t0 = _qid(), query_hash(q), time.monotonic_ns()
    for idx, (name, fn) in enumerate(_STAGES, start=1):
        with energy_counter() as ec:
            try: ans = fn(q)
            except Exception: ans = None
        if ans:
            return {
                "query": {"id": qid, "hash": qh, "canonical": canonicalize(q)},
                "resolved_at_stage": idx, "stage_name": name,
                "joule_cost": ec.report(),
                "answer": {"hash": query_hash(ans), "canonical": ans},
                "wall_clock_ns": time.monotonic_ns() - t0,
            }
    raise RuntimeError("unreachable: neural stage always answers")


# ---------- HuggingFace AI Energy Score sink ----------------------------------

def aies_emit(receipts: list[dict]) -> dict:
    bucket: dict[str, list[float]] = {n: [] for n, _ in _STAGES}
    for r in receipts:
        uj = r["joule_cost"]["microjoules"]["measured"] + r["joule_cost"]["microjoules"]["estimated"]
        bucket[r["stage_name"]].append(uj / 1e6)
    return {f"eoc-stage-{s}": {
                "energy_joules_per_inference_mean": (sum(v)/len(v)) if v else 0.0,
                "energy_joules_per_inference_count": len(v)}
            for s, v in bucket.items()}


# ---------- evaluation --------------------------------------------------------

def _joules(r: dict) -> float:
    return (r["joule_cost"]["microjoules"]["measured"]
          + r["joule_cost"]["microjoules"]["estimated"]) / 1e6

def evaluate(questions: list[dict]) -> dict:
    receipts = [cascade(it["question"]) for it in questions]
    correct = [any(ref.lower() in r["answer"]["canonical"].lower() for ref in it["references"])
               for it, r in zip(questions, receipts)]
    total_j = sum(_joules(r) for r in receipts)
    n_correct = sum(correct)
    return {
        "n_total": len(questions), "n_correct": n_correct, "total_joules": total_j,
        "joules_per_correct_answer": total_j / max(n_correct, 1),
        "per_stage_count":  {s: sum(1 for r in receipts if r["stage_name"] == s) for s, _ in _STAGES},
        "per_stage_joules": {s: sum(_joules(r) for r in receipts if r["stage_name"] == s) for s, _ in _STAGES},
        "aies": aies_emit(receipts), "receipts": receipts,
    }


# ---------- demo fixture ------------------------------------------------------

_CACHE_SEED = [
    ("What is the boiling point of water at sea level?", "100 degrees Celsius"),
    ("What is the speed of light in a vacuum?", "299792458 meters per second"),
    ("How many degrees in a circle?", "360"),
]
_KV_SEED = [
    ("Who proved Fermat's Last Theorem?", "Andrew Wiles"),
    ("What is the chemical symbol for gold?", "Au"),
]
_GRAPH_SEED = [
    ("france", "capital", "Paris"), ("japan", "capital", "Tokyo"),
    ("microsoft", "founded", "Bill Gates and Paul Allen"),
    ("apple", "founded", "Steve Jobs, Steve Wozniak, and Ronald Wayne"),
]
_EVAL_SET = [
    {"question": "What is the boiling point of water at sea level?", "references": ["100"]},
    {"question": "what is the speed of light in a vacuum",          "references": ["299792458", "3 x 10^8"]},
    {"question": "How many degrees in a circle?",                   "references": ["360"]},
    {"question": "Who proved Fermat's Last Theorem?",               "references": ["wiles"]},
    {"question": "What's the chemical symbol for gold?",            "references": ["au"]},
    {"question": "What is the capital of France?",                  "references": ["paris"]},
    {"question": "What is the capital of Japan?",                   "references": ["tokyo"]},
    {"question": "Who founded Microsoft?",                          "references": ["gates", "allen"]},
    {"question": "Who founded Apple?",                              "references": ["jobs", "wozniak"]},
    {"question": "Briefly explain why the sky is blue.",            "references": ["rayleigh", "scatter", "blue"]},
]


def main() -> None:
    cache_seed(_CACHE_SEED); kv_seed(_KV_SEED); graph_seed(_GRAPH_SEED)
    summary = evaluate(_EVAL_SET)
    print(json.dumps({k: v for k, v in summary.items() if k != "receipts"}, indent=2))
    print(f"\nReceipts: {len(summary['receipts'])} (first one shown)")
    print(json.dumps(summary["receipts"][0], indent=2))


if __name__ == "__main__":
    main()
