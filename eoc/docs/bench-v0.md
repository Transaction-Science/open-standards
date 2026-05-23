# EOC Benchmark v0 — `joules-per-MT-Bench-point`

> Status: draft v0. Numbers in the reference table below are placeholders
> until a live deployment is benchmarked end-to-end.
> License: text CC-BY-4.0, harness code Apache-2.0.

## 1. Why a joules-per-point metric

Every other benchmark in the LLM industry answers a single question:
*how good is the answer?* MMLU, MT-Bench, ARC, HumanEval, GPQA — all of
them measure quality on a fixed input distribution and ignore the energy
it took to produce the answer. That made sense when the only inference
substrate was a transformer running on a GPU and the only knob was
"more parameters". It does not make sense for an Energy-Optimised
Compute (EOC) deployment, where the substrate is a four-stage cascade
(cache → KV → graph → neural) and the explicit goal is to answer as
many queries as possible at the cheaper stages.

In that setting, the question that matters is:

> **How many joules did it take to earn one MT-Bench point?**

A cascade that scores 7.2 on MT-Bench at 14 J/correct is doing
qualitatively different work than a 7.2 GPU-baseline that burns 480
J/correct. The benchmark exists so a deployment can show that
difference rather than claim it.

The v0 formula is:

```
joules-per-MT-Bench-point =
    total_joules / (correct_answers × mt_bench_score_per_correct)
```

`total_joules` is the wall-time energy sum reported by the cascade for
the suite, in joules. `correct_answers` is the count of cases the
harness scored as a pass. `mt_bench_score_per_correct` is the suite's
own per-question weight; for the bundled MT-Bench-style set it is `1.0`,
so for v0 the metric reduces to **joules-per-correct**. The factor is
kept in the formula so that suites that adopt MT-Bench's 10-point judge
scoring later can plug it in without reshaping the dimensional
analysis.

A complementary metric, **stage-distribution**, reports the fraction of
cases resolved at each stage. A deployment that drives
joules-per-correct down by short-circuiting at cache and KV will see
that show up directly in the distribution. A deployment that scores
high by simply running every query through the neural stage will see
its joules-per-correct number balloon, regardless of how good the
final answers are.

## 2. What the harness actually does

The harness has four small crates:

| Crate                | Role                                                                                  |
| -------------------- | ------------------------------------------------------------------------------------- |
| `eoc-bench-runner`   | `BenchCase`, `BenchResult`, `BenchReport`; `run()`; `aggregate()`.                    |
| `eoc-bench-mt`       | 20 MT-Bench-style cases bundled as JSON (`data/cases.json`).                          |
| `eoc-bench-router`   | 30 router-shaped cases chosen to exercise all four cascade stages.                    |
| `eoc-bench` (binary) | CLI: `eoc-bench run --suite {mt,router,all}` and `eoc-bench compare`.                 |

`run()` walks the supplied cases through an `eoc_cascade::Cascade` one
by one. Sequential execution is intentional: parallel queries on shared
hardware pollute the joule attribution that the cascade hands back, and
the metric we care about is **energy per query**, not throughput. A
throughput-oriented variant is a future work item, not a v0 deliverable.

For each case the runner records:

* the response payload,
* which stage resolved it (`cache | kv | graph | neural`),
* the joule cost in microjoules,
* wall-clock latency in milliseconds,
* whether the response contained the expected substring (when the
  case has an `expected` field).

The default accuracy predicate is a case-insensitive substring match.
That is sufficient for the v0 case set, which uses unambiguous facts
("Paris", "Mars", "4") rather than open-ended writing tasks. Cases
designed to elicit creative output (`mt-writing-*`, `rt-neural-*`)
deliberately leave `expected` as `null`; they count towards joules and
latency but never towards accuracy.

`aggregate()` rolls a slice of `BenchResult`s up into a `BenchReport`
containing `case_count`, `total_joules`, `joules_per_correct`,
`accuracy_pct`, `latency_p50_ms`, `latency_p95_ms`, `latency_p99_ms`,
and a `stage_distribution` map. Percentiles use the nearest-rank
method (no interpolation) so the result is always one of the observed
latencies — this matches how operators read percentiles in production
dashboards.

## 3. The case sets

**MT (`eoc-bench-mt`).** Twenty cases drawn from the broad shape of
MT-Bench's eight categories — arithmetic, multi-step reasoning,
multi-turn dialogue, code, writing, factual recall, explanation,
summarisation. Multi-turn cases are concatenated with a literal
`Follow-up:` marker; the cascade sees the full conversation as a
single prompt, which is the same convention MT-Bench uses for turn
2 grading. Twenty is deliberately small: v0 is a debug-loop tool, not
a Goodhart target.

**Router (`eoc-bench-router`).** Thirty cases sized so a well-warmed
cascade hits all four stages. Case ids carry a routing hint:

* `rt-cache-*` — six short, repeated prompts that should memoise after
  the first warmup pass.
* `rt-kv-*` — seven exact-key lookups (well-known constants, ports,
  ISO codes) the reference KV stage is seeded with.
* `rt-graph-*` — eight subject–predicate–object questions the
  reference triple store can answer.
* `rt-neural-*` — nine open-ended generation prompts no pre-populated
  store can answer.

The hints are advisory. The harness measures *where the cascade
actually resolved each case*. A deployment that lands an
`rt-graph-*` case at `neural` because its triple store was not warmed
sees that show up in the stage distribution, and the joules-per-correct
number reflects the cost.

## 4. The reference cascade the CLI ships with

`eoc-bench run` instantiates the four-stage cascade from `eoc-rs`
using deliberately small fixture data so the CLI works out of the
box. The LRU cache has a 1024-entry capacity. The KV stage is seeded
with the same constants that the router suite's `rt-kv-*` cases ask
about. The graph stage holds the triples the `rt-graph-*` cases need.
The neural stage uses the `EchoBackend` from `eoc-neural` with an
estimated cost of 50 J per inference — well above realistic GPU cost
for any single token, but a useful default for visualising the
cost spread in the report. Production deployments substitute their
own wiring; the `run` function in `eoc-bench-runner` accepts any
`eoc_cascade::Cascade`.

## 5. Reference numbers

These are placeholders. They will be replaced as soon as the harness
runs against a live deployment.

| Suite     | Cases | Accuracy %                    | Joules/correct                | p50 / p95 / p99 (ms)                                   | Stage mix (cache / kv / graph / neural)                       |
| --------- | ----- | ----------------------------- | ----------------------------- | ------------------------------------------------------ | -------------------------------------------------------------- |
| mt        | 20    | _to be filled when running against a live deployment_ | _to be filled when running against a live deployment_ | _to be filled when running against a live deployment_ | _to be filled when running against a live deployment_         |
| router    | 30    | _to be filled when running against a live deployment_ | _to be filled when running against a live deployment_ | _to be filled when running against a live deployment_ | _to be filled when running against a live deployment_         |
| all       | 50    | _to be filled when running against a live deployment_ | _to be filled when running against a live deployment_ | _to be filled when running against a live deployment_ | _to be filled when running against a live deployment_         |

Targets we publish v0 results against:

* `joules-per-correct` ≤ **5 J** on the router suite for a well-warmed
  cascade (most cases resolve at cache/KV/graph).
* `joules-per-correct` ≤ **25 J** on the MT suite when wired against a
  small open-weights neural backend on commodity GPU.
* `latency_p95_ms` ≤ **250 ms** on the router suite end-to-end.
* `accuracy_pct` ≥ **80%** on the union of scored cases.

These thresholds are aspirational, not contractual. A deployment is
free to publish lower numbers and explain why; the metric is what
matters, not the threshold.

## 6. Reproducibility

The harness is fully embedded — the case JSON is `include_str!`'d at
build time, the path dependencies on `eoc-rs` are pinned in the root
`Cargo.toml`, and the toolchain is pinned via the workspace's
`rust-toolchain.toml` to Rust 1.95.0 edition 2024.

### Command lines

Build (default features include `eoc-rs`):

```
cargo build --release -p eoc-bench
```

Run a suite and capture a report:

```
target/release/eoc-bench run --suite mt    > mt-baseline.json
target/release/eoc-bench run --suite router > router-baseline.json
target/release/eoc-bench run --suite all    > all-baseline.json
```

Compare two reports:

```
target/release/eoc-bench compare \
    --baseline all-baseline.json \
    --candidate all-candidate.json
```

`compare` prints a small JSON object with the three deltas operators
care about:

* `joules_per_correct_pct` — percent change in J/correct (negative is
  better).
* `latency_p95_pct` — percent change in p95 latency (negative is
  better).
* `accuracy_pp_delta` — absolute change in accuracy percentage points
  (positive is better).

### BenchReport JSON schema

The `BenchReport` type serialises as:

```json
{
  "case_count": 20,
  "total_joules": 0.0,
  "joules_per_correct": 0.0,
  "accuracy_pct": 0.0,
  "latency_p50_ms": 0,
  "latency_p95_ms": 0,
  "latency_p99_ms": 0,
  "stage_distribution": {
    "cache": 0,
    "kv": 0,
    "graph": 0,
    "neural": 0
  }
}
```

Field semantics:

| Field                  | Type                | Meaning                                                                 |
| ---------------------- | ------------------- | ----------------------------------------------------------------------- |
| `case_count`           | `usize`             | Number of cases in the run.                                             |
| `total_joules`         | `f64`               | Sum of stage-reported energy, expressed in joules.                      |
| `joules_per_correct`   | `f64`               | `total_joules / correct_answers`. `Infinity` if no answers were correct. |
| `accuracy_pct`         | `f64`               | Percentage of *scored* cases that passed. Cases with no expected answer are excluded from this number. |
| `latency_p50_ms`       | `u64`               | Nearest-rank p50 latency across all cases.                              |
| `latency_p95_ms`       | `u64`               | Nearest-rank p95 latency.                                               |
| `latency_p99_ms`       | `u64`               | Nearest-rank p99 latency.                                               |
| `stage_distribution`   | `map<string,usize>` | Counts keyed by stable stage id (`cache`, `kv`, `graph`, `neural`).      |

### Diffability

Reports are intended to land in a repository as text-diffable JSON.
The aggregator is deterministic: same input cases plus the same cascade
configuration produces byte-identical reports. `serde_json::to_string_pretty`
emits stable key order because the underlying maps are `BTreeMap`s.

## 7. Honest limitations

This is v0. The following are known and tracked:

1. **The bundled neural backend is `EchoBackend`.** It does not produce
   correct answers for open-ended cases; its job is to exercise the
   joule attribution pipe and let the runner end-to-end. Real numbers
   require wiring a real backend (`llama.cpp`, vLLM, an inference
   provider) at the `NeuralBackend` trait surface.

2. **Accuracy scoring is substring-based.** That works for the v0
   case set because the expected answers are short and unambiguous.
   v1 should support pluggable judges, including LLM-as-judge for the
   writing/reasoning categories (matching MT-Bench's pipeline).

3. **The case sets are small.** Twenty MT cases and thirty router
   cases are enough to debug a cascade, not enough to publish a
   leaderboard against. v1 will adopt the full 80-question MT-Bench
   set and add a router suite of at least 250 cases.

4. **Joule readings come from the cascade, not from independent
   hardware counters.** The cascade itself trusts its meter (the
   default is a `StubCounter`). Deployments that need defensible
   numbers must wire `eoc-meter` against RAPL / NVML / `powermetrics`
   and re-run.

5. **`mt_bench_score_per_correct = 1.0`.** The MT-Bench original uses a
   10-point judge score per question. v0 collapses that to a binary
   pass/fail so the harness has no dependency on a judging model. v1
   will lift this restriction.

The point of shipping v0 anyway is that the *shape* of the metric — a
joule denominator times an accuracy numerator, with a stage
distribution alongside — is what makes the cascade visible. The
absolute numbers will get sharper. The shape will not.

## 8. License

Harness code (every Rust file under `eoc-bench/`): Apache-2.0.
This document: CC-BY-4.0. Bundled case JSON: CC-BY-4.0.
