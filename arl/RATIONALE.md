# Why ARL — Trends in AI and the Rationale for a Readiness Standard

**Version 1.3** — May 2026
**Companion to:** ARL v1.3, ARL-S v1.3, LEXICON v1.3

ARL did not invent a measurement discipline. It ports working ones from fields that solved the measurement of stochastic, safety-critical systems decades ago. This document is the *why*: the trends that make a shared readiness yardstick necessary, and the reasoning behind ARL's four axes. It takes no position on who is ahead or on where the technology is going — only on what a readiness claim has to contain to mean something.

---

## 1 · The trend

Five shifts in how AI is built and reported, each pointing at the same missing thing: a yardstick that survives translation across labs, languages, and years.

**No shared yardstick.** New models often arrive with new benchmarks, and benchmarks are replaced every few months. A score reported by one effort in one year cannot be honestly compared to a score from another effort years later — and capability numbers are commonly published without error bars, without a methodology fixed before the claim, without disclosing the hardware, without energy attribution, and without saying which exact version of the system was measured.

**Reliability moves slower than headlines.** Research on operational reliability and on task time-horizons documents a widening distance between rising headline scores and how consistently systems behave under operational variation. A single-shot accuracy number does not capture variance, and variance is what determines whether a system can be relied on.

**Energy is a first-order constraint.** Global data-centre electricity grew 17% in 2025; AI-specifically grew 50%, with the IEA projecting 945 TWh by 2030, and grid + generation additions taking 5–10 years to land. Joules-per-token is now proposed as a standard efficiency metric — peer of FLOPs and latency — in current benchmarking work, and reasoning queries cost roughly 13× a standard query, an order of magnitude invisible to systems that don't measure. A readiness score that omits joules omits the part of the system the physical world actually bills for. Tokens-per-second compresses; joules don't.

**Systems are agentic, not just models.** Outputs of one step feed the inputs of the next; the harness, the tools, and the scaffolding are part of the deployed system, not accessories to the model. New attack surfaces come with that — schema-level techniques such as the Constrained Decoding Attack reach documented success rates of 94–99% by embedding intent in grammar rules while the prompt stays benign. Measurement has to cover the whole configured system.

**Governance is converging, methodology is deferred.** Risk-based AI frameworks are converging across jurisdictions, but most defer the technical *how to measure* to standards still being drafted. There is a methodology-shaped gap between "this system is high-risk" and "here is how its readiness was measured" — a gap a technical readiness framework can fill.

---

## 2 · The response

Four axes, each closing a gap the others leave open — and each borrowed from a field that already does this.

No single number characterizes a stochastic system. Pharmaceutical efficacy, radar, communications, and aircraft certification all report several quantities at once, because any one of them alone is gameable. ARL is built on the same principle. Hardware is documented alongside every claim for reproducibility, but it is not a peer axis. Three axes leave gaps; five add redundancy; four is the minimum that works.

| Axis | Borrowed from | Source | What it secures |
|------|---------------|--------|-----------------|
| **Validation Depth** | Technology Readiness Levels | NASA (Sadin 1974; Mankins 1995; ISO 16290:2013) | How thoroughly a claim has been tested, not how impressive it is. A proven toaster and a proven spacecraft both rate "proven." |
| **Convergence Class** | Pharmaceutical efficacy, radar Pd/Pfa, bit error rate | Stochastic-system characterization practice | Variance gets characterized, not waved away. Every field deploying stochastic systems abandoned "it's probabilistic, so it can't be measured" decades ago. |
| **Energy Profile** | ENERGY STAR & PUE | EPA / The Green Grid | Physical equipment energy has been measured and verified across product categories for thirty years. The methodology already exists. |
| **Security Class** | Confidentiality / integrity / auditability | NIST SP 800-53, ISO/IEC 27001, Common Criteria EAL | Fields that deploy against an adversary measure resistance to attack as a first-class property, with assurance levels, rather than asserting it. |

---

## 3 · Trends with structure

**Structure is the difference between an anecdote and a trajectory.**

A capability score by itself is a snapshot. The improvements people point to only become a *trajectory* — something you can measure, compare, and extrapolate — once a structured yardstick holds still underneath them. METR's task-horizon measurement is a clean example: by fixing the methodology, it turns "models feel more capable" into a measured cadence. The length of task a frontier agent completes at 50% reliability has roughly doubled every seven months since 2019 — with recent doublings closer to four — and the same shape now appears across nine domains. That trajectory is only visible because the measurement is structured. ARL is that discipline applied to readiness: fix the axes, and improvement becomes legible and comparable instead of asserted.

And structure is itself on a steep trajectory. Over the last few years, groups in every region have converged on disciplined evaluation and readiness frameworks — independently, and toward the same shape.

| When | Milestone | What it added |
|------|-----------|---------------|
| 1974 → 2013 | Technology Readiness Levels (NASA → ISO 16290) | the original readiness-level structure ARL adapts |
| 2019 | OECD AI Principles; Singapore Model AI Governance Framework | first national AI governance frameworks |
| 2020 → 2022 | Technology Readiness Levels adapted for machine learning (MLTRL); Stanford HELM; BIG-Bench | structured, multi-metric evaluation |
| 2022 → 2023 | China's CAC algorithm and generative-AI registry; Singapore open-sources AI Verify; ISO/IEC 42001 (AI management systems) | accountability + open testing + management systems |
| 2023 → 2024 | UK and US AI Safety Institutes; the EU AI Act; Korea's AI Basic Act; METR begins measuring task time-horizons | evaluation bodies + binding law + measured trajectories |
| 2025 | China's TC260 generative-AI security and content-labeling standards take effect; reliability and time-horizon research matures | technical standards + reliability discipline |
| 2026 | Singapore's Agentic AI framework; METR's Time Horizon 1.1 corroborates the trend across nine domains; readiness-level structures (ARL) extend the discipline to deployed systems end-to-end | structure reaches agentic systems |

This is a trend, not a scoreboard: structured measurement is becoming the global default — the same way UL marks, SOC 2, and ISO 9001 did — because comparable results let each one build on the last instead of restarting every cycle.

---

## 4 · Built to be useful

A standard earns its place by being needed, not by being mandated. UL safety marks, SOC 2 audits, ISO 9001, CE marking — each became a commercial necessity before it was ever universally required, because buyers, insurers, and procurement offices needed a defensible basis for a decision. A readiness score is for the same moment: when someone has to choose a system and be able to explain the choice later.

That is why ARL is anchored in math and physics rather than opinion. Each axis rests on a foundation that does not drift across time, languages, or institutions: a claim of "ARL 6, Class D, 12.3 kJ/task, S2" means the same thing everywhere, while a claim of "PhD-level reasoning" does not survive the trip.

---

## 5 · Composes, not competes

ARL is a technical readiness measure. It fits underneath the governance frameworks and alongside the benchmarks that already exist.

| Framework | How ARL composes |
|-----------|------------------|
| **EU — AI Act** | Defers conformity-assessment methodology to harmonized standards still in progress — the kind of technical measure ARL provides. |
| **United States — NIST AI RMF** | A voluntary risk-management frame; ARL supplies a measurable readiness score a risk process can cite. |
| **China — CAC registry + TC260** | The CAC algorithm and generative-AI filing gives provider accountability and a national inventory of deployed systems; the TC260 technical-standards stack (content-labeling GB 45438-2025, generative-AI security and training-data standards effective late 2025) addresses content security and data discipline. ARL measures per-system technical readiness — a different layer of the same goal: deployment that is disclosed and accountable. |
| **UK · Singapore — AISI / AI Verify** | Their evaluation platforms (Inspect, AI Verify) plug into the ARL Sandbox as Harnesses — the task lives in the Harness, ARL measures around it. |
| **Japan · South Korea — METI / AI Basic Act** | Risk-tiered guidance and law (Japan's METI AI Guidelines and J-AISI; Korea's AI Basic Act and KAISI) set obligations; ARL supplies the readiness scoring such obligations can reference. |
| **International — ISO/IEC JTC 1 / SC 42** | The venue where the AI standards the world interoperates with are written (ISO/IEC 42001 and the rest); ARL is the kind of technical readiness method such standards reference. |
| **Any benchmark** | MMLU, C-Eval, KMMLU, SWE-Bench, GAIA, HCAST, and language-specific suites worldwide — every benchmark is a task that plugs into ARL-S as a Harness. ARL does not replace benchmarks; it puts discipline around them. |

---

Specification text is CC-BY-4.0 (see [`LICENSE`](LICENSE)). ARL is owned by no one; Transaction Science is one steward. The rendered version of this document lives at [arl.transaction.science/why](https://arl.transaction.science/why).
