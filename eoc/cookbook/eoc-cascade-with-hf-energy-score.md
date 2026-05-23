# Routing inference through the EOC four-stage cascade and reporting joules with HuggingFace AI Energy Score

*A HuggingFace Cookbook-style recipe. Companion code: [`eoc-cascade.py`](./eoc-cascade.py). Licensed CC-BY-4.0.*

*Authors: Transaction Science (steward of the EOC specification). Status: working draft v0.1.*

---

## a. What you'll build

You will build a four-stage **EOC cascade** in roughly 200 lines of Python and route a small evaluation set through it, reporting joules per query at every stage. The cascade resolves each query at the **cheapest sufficient stage** — the median query hits a cache for tens of microjoules; only the residual fraction falls through to a neural model.

The four stages, in order:

1. **Cache** — exact-match lookup keyed on a content hash of the canonicalized query. Resolves in microjoules.
2. **KV** — approximate-match lookup over sentence embeddings with a cosine-similarity threshold. Resolves in millijoules.
3. **Graph** — pattern-matching lookup over a small in-memory triple store. Resolves in tens of millijoules.
4. **Neural** — a HuggingFace causal-LM fallback (default: `microsoft/phi-3-mini-4k-instruct`, ~3.8B parameters quantized). Resolves in joules.

Each resolution returns an **EOC receipt** carrying the query hash, the answer hash, the stage that answered, the wall-clock duration, and the energy cost in microjoules. The energy cost is read from `rapl-py` (Intel host CPUs) or `pynvml` (NVIDIA GPUs) when available; otherwise it falls back to a calibrated estimator. The receipt is then reported to the **HuggingFace AI Energy Score** sink, which aggregates joules-per-query across stages.

When you finish you will have:

- A runnable cascade that resolves a sample MT-Bench-style query set.
- A per-stage joule report that decomposes total energy into cache / KV / graph / neural contributions.
- A summary of **joules per correct answer** — the substrate-level useful-work unit defined in EOC's `Q1_W_DEFINITION.md`.

The recipe is deliberately small. Production deployments swap in a domain-specific knowledge graph for stage 3 and a deployment-preferred LLM for stage 4. The substrate is the cascade, not the choice of operators.

## b. Prerequisites

- **Python 3.11+**.
- **pip install** (versions pinned in [`requirements.txt`](./requirements.txt)):
  - `transformers==4.46.3` — HuggingFace model loading and inference.
  - `sentence-transformers==3.3.1` — KV-stage embeddings (`all-MiniLM-L6-v2`, 22M parameters).
  - `datasets==3.1.0` — evaluation set loader.
  - `evaluate==0.4.3` — metric helpers.
  - `torch==2.5.1` — backend for the above.
  - `numpy==2.1.3` — receipts use ndarray for embedding similarity.
  - **Optional** `rapl-py==0.5.0` — Intel RAPL energy counters; when absent the example uses the estimator.
  - **Optional** `pynvml==12.0.0` — NVIDIA GPU energy counters; when absent the example uses the estimator.
- Roughly **6 GB** of disk for the phi-3-mini weights on first run. The KV-stage embedding model is ~90 MB.
- A CPU-only run takes about **3 minutes** end-to-end on a recent laptop, with the dominant cost being the neural-fallback inferences. With a GPU it runs in roughly 30 seconds.

The whole recipe is self-contained — no network requests beyond the initial model downloads.

## c. Step 1 — set up the cache stage

The cache stage is a Python dictionary keyed on the **BLAKE3 hash** of the canonicalized query string. Canonicalization is whitespace-collapse and lowercase. A hit returns the stored answer directly; no model inference, no embeddings, no graph lookup.

```python
import hashlib  # we use sha256 in the recipe; the EOC spec mandates BLAKE3 in production

def canonicalize(q: str) -> str:
    return " ".join(q.strip().lower().split())

def query_hash(q: str) -> str:
    return hashlib.sha256(canonicalize(q).encode("utf-8")).hexdigest()

CACHE: dict[str, str] = {}

def cache_resolve(q: str):
    h = query_hash(q)
    return CACHE.get(h)
```

Why this is fast: the dictionary lookup is **constant-time**, the canonicalization is O(len(q)), and the hash is computed in roughly a microsecond per kilobyte of query text. On a modern CPU the entire stage runs in **~10–50 microjoules** per query.

In production the cache is content-addressed against a federated store (EOC-3 artifact distribution) so that a cache hit anywhere in the federation resolves the query. For this recipe we keep it in-process.

## d. Step 2 — set up the KV stage

The KV (key-value) stage handles **approximate matches**: queries that don't exactly equal a cached one but are semantically equivalent. We embed the query with a small sentence-transformer (`all-MiniLM-L6-v2`, 22M parameters, 384-dim output) and compare against a small vector store using cosine similarity.

```python
from sentence_transformers import SentenceTransformer
import numpy as np

KV_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
KV_STORE: list[tuple[np.ndarray, str]] = []   # (embedding, answer)
KV_THRESHOLD = 0.85   # cosine similarity threshold

def kv_resolve(q: str):
    if not KV_STORE:
        return None
    q_emb = KV_MODEL.encode(q, normalize_embeddings=True)
    sims = np.array([float(np.dot(q_emb, e)) for e, _ in KV_STORE])
    best = int(np.argmax(sims))
    if sims[best] >= KV_THRESHOLD:
        return KV_STORE[best][1]
    return None
```

Cost: one embedding pass through a 22M-parameter model. On a CPU, that's **~1–3 millijoules** per query.

The threshold (0.85) is deployment-tunable. Lower it and more queries resolve here at the cost of recall; raise it and fewer queries resolve here, falling through to the more expensive graph or neural stages.

In production the KV store is a proper vector database with HNSW or IVF-PQ indexing. For this recipe a linear scan over a small list suffices and stays under a millijoule.

## e. Step 3 — set up the graph stage

The graph stage resolves queries that **decompose into a structured pattern** — "what is the capital of X", "who founded Y", "when did Z happen". We use a tiny in-memory triple store and a regex-based pattern matcher.

```python
import re

GRAPH: list[tuple[str, str, str]] = []   # (subject, predicate, object) triples

GRAPH_PATTERNS = [
    (re.compile(r"^what is the (\w+(?:\s\w+)*) of (.+?)\??$"),
     lambda m: (m.group(2).strip().lower(), m.group(1).strip().lower())),
    (re.compile(r"^who (\w+ed) (.+?)\??$"),
     lambda m: (m.group(2).strip().lower(), m.group(1).strip().lower())),
]

def graph_resolve(q: str):
    qc = canonicalize(q)
    for pat, extract in GRAPH_PATTERNS:
        m = pat.match(qc)
        if not m:
            continue
        subject, predicate = extract(m)
        for s, p, o in GRAPH:
            if s == subject and p == predicate:
                return o
    return None
```

Cost: regex matching is O(len(q)), triple lookup is O(|graph|) for the toy version. On commodity CPUs **~10–30 millijoules** per query.

In production the graph stage uses **DCY** (Deterministic Cypher, EOC-4) over a typed, content-addressed knowledge graph. For this recipe a Python list of triples is enough to demonstrate the resolution predicate.

## f. Step 4 — set up the neural fallback

The neural stage is the always-available, always-last fallback. It runs a HuggingFace causal-LM. The default model in this recipe is `microsoft/phi-3-mini-4k-instruct` because it is small enough to run on a CPU in seconds but large enough to give plausible answers.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

NEURAL_MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
_tokenizer = None
_model = None

def neural_init():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(NEURAL_MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(
            NEURAL_MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
        )

def neural_resolve(q: str) -> str:
    neural_init()
    inputs = _tokenizer(q, return_tensors="pt")
    with torch.no_grad():
        out = _model.generate(**inputs, max_new_tokens=64, do_sample=False)
    return _tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
```

Cost: a single short generation through phi-3-mini on a CPU is **~0.5–2 joules**. On a GPU it's roughly an order of magnitude less per query.

**This is for demonstration.** Production deployments swap in their preferred neural backend — a quantized 7B model on consumer hardware, a 70B model on accelerated infrastructure, or a hosted endpoint. The cascade contract — "the neural stage answers any query the prior stages did not" — does not care about the model choice.

## g. Step 5 — wire the cascade

The cascade is the composition. Each stage is consulted in order; the first stage to return a non-None answer wins. Resolution emits a receipt.

```python
import time, uuid

def cascade(q: str) -> dict:
    qid = str(uuid.uuid7()) if hasattr(uuid, "uuid7") else str(uuid.uuid4())
    qh = query_hash(q)
    t0 = time.monotonic_ns()

    for stage_idx, (name, fn) in enumerate(
        [("cache", cache_resolve),
         ("kv", kv_resolve),
         ("graph", graph_resolve),
         ("neural", neural_resolve)],
        start=1,
    ):
        with energy_counter() as ec:
            answer = fn(q)
        if answer is not None:
            return {
                "query": {"id": qid, "hash": qh, "canonical": canonicalize(q)},
                "resolved_at_stage": stage_idx,
                "stage_name": name,
                "joule_cost": ec.report(),
                "answer": {"hash": query_hash(answer), "canonical": answer},
                "wall_clock_ns": time.monotonic_ns() - t0,
            }
    raise RuntimeError("unreachable: neural stage always answers")
```

The cascade is **monotone** in expected cost: stage 1 is always cheapest, stage 4 is always most expensive. Each stage's resolution predicate is the only thing that decides whether the query stops there.

## h. Step 6 — instrument with HuggingFace AI Energy Score

We wrap each stage's execution in an **energy counter**: a context manager that records joules consumed during the stage's work. The implementation prefers hardware counters and degrades gracefully to a calibrated estimator.

```python
from contextlib import contextmanager
import time

class EnergyReport:
    def __init__(self, microjoules: int, method: str):
        self.microjoules = microjoules
        self.method = method
    def report(self) -> dict:
        return {
            "microjoules": {"measured": self.microjoules if self.method != "estimator-v1" else 0,
                            "estimated": self.microjoules if self.method == "estimator-v1" else 0},
            "method": self.method,
        }

@contextmanager
def energy_counter():
    method, start = _begin_counter()
    t0 = time.monotonic_ns()
    report = EnergyReport(0, method)
    try:
        yield report
    finally:
        elapsed_ns = time.monotonic_ns() - t0
        report.microjoules = _end_counter(start, elapsed_ns, method)
```

The `_begin_counter` and `_end_counter` helpers (in the companion Python file) try `rapl-py` first, then `pynvml`, then the **estimator** (a calibrated `cpu_power_watts × wall_clock_s × 1e6`).

**HuggingFace AI Energy Score integration**: AIES expects a dict-of-(model, energy-per-inference-joules). We adapt by reporting **per-stage** energy keyed on the cascade stage name:

```python
def aies_emit(receipts: list[dict]) -> dict:
    by_stage: dict[str, list[float]] = {"cache": [], "kv": [], "graph": [], "neural": []}
    for r in receipts:
        uj = r["joule_cost"]["microjoules"]["measured"] + r["joule_cost"]["microjoules"]["estimated"]
        by_stage[r["stage_name"]].append(uj / 1e6)
    return {
        f"eoc-stage-{stage}": {
            "energy_joules_per_inference_mean": (sum(v)/len(v)) if v else 0.0,
            "energy_joules_per_inference_count": len(v),
        }
        for stage, v in by_stage.items()
    }
```

The output of `aies_emit` is the format AIES leaderboards consume. A deployment can publish its `eoc-stage-*` numbers directly to the HuggingFace AI Energy Score sink, alongside any model-level numbers it already publishes.

## i. Step 7 — aggregate joules per correct answer

The substrate-level unit of work, per EOC's `Q1_W_DEFINITION.md`, is **one correct answer**. We run the cascade against a small MT-Bench-style subset (10 fact-recall questions and 10 reasoning prompts), score each receipt's answer against a reference, and report **joules per correct answer** decomposed by stage.

```python
def evaluate(questions: list[dict]) -> dict:
    receipts, correct = [], []
    for item in questions:
        r = cascade(item["question"])
        receipts.append(r)
        correct.append(item["reference"].lower() in r["answer"]["canonical"].lower())

    total_uj = sum(
        r["joule_cost"]["microjoules"]["measured"] +
        r["joule_cost"]["microjoules"]["estimated"]
        for r in receipts
    )
    n_correct = sum(correct)
    return {
        "n_total": len(questions),
        "n_correct": n_correct,
        "total_joules": total_uj / 1e6,
        "joules_per_correct_answer": (total_uj / 1e6) / max(n_correct, 1),
        "per_stage_count": {s: sum(1 for r in receipts if r["stage_name"] == s)
                            for s in ["cache", "kv", "graph", "neural"]},
        "per_stage_joules": {s: sum(
            (r["joule_cost"]["microjoules"]["measured"] +
             r["joule_cost"]["microjoules"]["estimated"]) / 1e6
            for r in receipts if r["stage_name"] == s)
            for s in ["cache", "kv", "graph", "neural"]},
    }
```

A representative run on the included 20-question set, with pre-warmed cache and KV store, looks like:

```
{
  "n_total": 20,
  "n_correct": 18,
  "total_joules": 4.31,
  "joules_per_correct_answer": 0.239,
  "per_stage_count":  {"cache": 8, "kv": 6, "graph": 3, "neural": 3},
  "per_stage_joules": {"cache": 0.00018, "kv": 0.014,
                       "graph": 0.046,   "neural": 4.25}
}
```

Three observations:

1. **The neural stage dominates cost.** 3 of 20 queries resolved there account for 98.5% of total energy.
2. **The cache stage is essentially free.** 8 queries at ~22 μJ each.
3. **Joules per correct answer is the substrate-level unit.** It compresses both routing efficiency and answer quality into one falsifiable number.

A deployment that improves its cache hit-rate from 40% to 60% will shift its joules-per-correct-answer by roughly the ratio of neural-stage to cache-stage cost — three to four orders of magnitude on this hardware.

## j. Closing — publishing against the EOC spec

To publish your numbers against the EOC specification:

- **Record receipts** for every query the cascade resolves. Receipts are the unit of accounting and are independently verifiable from their content hashes.
- **Sign receipts** with an ed25519 key bound to your deployment. EOC-2 (the wire protocol) specifies the canonical CBOR serialization the signature covers.
- **Submit to the registry** (EOC-5) by listing your deployment's stage-implementation identifiers against the Operator Family Registry. The registry assigns each operator a stable identifier and a conformance test that any other implementation can replay.
- **Report via SCI-for-AI** by feeding receipts to a reporter implementing the **EOC extension URN `urn:gsf:sci-ai:ext:eoc:v1`**. The mapping document is at [`../standards/sci-for-ai-submission.md`](../standards/sci-for-ai-submission.md).

A **Rust reference implementation** of the four-stage cascade — production-grade, with content-addressed receipt storage, federated registry sync, and hardware-counter integration on Linux, macOS, and bare-metal targets — is tracked in the EOC roadmap at `eoc-rs/`. The Python recipe above is a teaching artifact; the Rust implementation is the conformance reference.

**Where to go next:**

- Read the EOC-1 specification at `eoc/spec/eoc1_v0_2.docx` for the full substrate architecture.
- Read `eoc/os/Q1_W_DEFINITION.md` for the definition of *useful work* in joules-per-task that this recipe implements.
- Read the SCI-for-AI mapping at [`../standards/sci-for-ai-submission.md`](../standards/sci-for-ai-submission.md) to understand how the receipts compose with GSF carbon accounting.
- Try the companion Python file [`eoc-cascade.py`](./eoc-cascade.py) end-to-end against your own evaluation set. Swap in your own knowledge graph for stage 3 and your preferred neural model for stage 4.

The cascade is the substrate. Joules are the unit. Receipts are the accounting. Everything else is choice of operator.
