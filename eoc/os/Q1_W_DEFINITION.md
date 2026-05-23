# Q1 — Definition of `W` (useful work) per workload class

*Resolution document for THEOREM v0.2 Open Question Q1.*

**Version 0.1** — Initial proposal. Resolves Q1 for two workload classes (rendering, inference). Other classes deferred to future resolution documents.

---

## 0. Why this document exists

THEOREM v0.2 invariant I3 states: *the scheduler optimizes `J/W` where `J` is joules consumed and `W` is useful work delivered, defined per workload class.*

`J` is measurable. The energy oracle (O1) produces it. RAPL, AMU, AGX telemetry — solved.

`W` is not. It is a vibe in the theorem until each workload class supplies a concrete unit. Without `W`, the scheduler has no objective function and I3 is unimplementable.

This document defines `W` for two classes — **rendering** and **inference** — because those are the classes that bind v0.1 (graphics on Apple Silicon, joulesperbit.ai integration). Other classes are not defined here and the scheduler refuses to dispatch them until they are. That refusal is correct, not a gap.

---

## 1. The shape of a `W` definition

Every workload class definition produces five things:

1. **The unit of `W`.** A countable scalar with physical or contractual meaning. Not a benchmark score, not a relative metric — a thing that exists in the world.

2. **The counting protocol.** How `W` is incremented at runtime, who increments it, and what the kernel's role is in observing the increment. The kernel must be able to count without trusting the workload.

3. **The validity predicate.** What makes a unit of `W` count as *delivered*. A frame rendered after its deadline is not a frame; the predicate says so explicitly.

4. **The aggregation rule.** How `W` aggregates across subjects, across substrates, across time intervals. Whether `W` is monotone, whether it is fungible across workload classes (it is not).

5. **The relationship to declared latency floors (I19).** Whether the class is interactive, what its deadline structure is, and how missed deadlines affect `W`.

A class without all five is not defined. The scheduler treats it as undeclared and refuses dispatch under I16's pessimistic-defaults clause.

---

## 2. Class: `Rendering`

### 2.1 Unit

`W_render = number of frames presented to the display where the frame met its declared deadline.`

A frame is a rectangular array of pixel data scanned out to a physical display surface. A frame is "presented" when the display controller has accepted it for the next scanout cycle. A frame "meets its declared deadline" when its presentation time is no later than the deadline declared in its workload shape (I16).

### 2.2 Counting protocol

The display compositor, holding the `DisplayOutput` capability (an I20 irreversibility class), invokes a kernel-resident counter on each successful frame presentation. The counter records:

- The subject ID that produced the frame
- The substrate that rendered it (which GPU, which command queue)
- The declared deadline
- The actual presentation timestamp
- The joule cost attributed to this frame's production (O1 oracle, integrated over the frame's render interval)

The kernel does not trust the compositor's claim that a frame was rendered. The display controller's hardware presentation event is the ground truth; the compositor's role is to attribute the event to a subject, which the kernel verifies against the capability ledger (O2).

### 2.3 Validity predicate

A frame counts toward `W_render` if and only if:

- It was presented to the display (hardware event observed).
- Its presentation timestamp ≤ its declared deadline.
- The subject that produced it held a valid `DisplayOutput` capability for the surface at the moment of presentation.
- The frame's pixel content was produced within the joule budget declared by the subject's workload shape, or the subject explicitly accepted overshoot via the budget-overshoot capability.

A frame that misses its deadline is not a slow frame. It is *not a frame*. It contributes zero to `W_render` and the joules spent producing it count against `J` without offsetting `W`. This is the correctness teeth I19 needs.

A frame produced and discarded before presentation (the compositor dropped it because a newer frame arrived) also contributes zero to `W_render`. Speculative work that does not reach the display is energy spent on nothing.

### 2.4 Aggregation rule

Across subjects: `W_render` aggregates additively per subject; the system-level `W_render` is the sum.

Across substrates: a frame is attributed to the substrate that produced its pixel data, not the substrate that scanned it out. A frame rendered on the GPU and composited on the CPU is attributed to the GPU.

Across time: `W_render` is monotone non-decreasing within an interval. The scheduler's `J/W` optimization is over fixed-length sliding windows (proposed: 1 second windows for steady-state, 16ms windows for transient response). Window length is policy, not invariant.

`W_render` does not aggregate with `W_inference` or any other class. `J/W_render` and `J/W_inference` are separate optimization targets and the scheduler's job under multiple concurrent classes is the open question Q7.

### 2.5 Latency relationship (I19)

Rendering is an interactive class. Every frame carries a declared deadline. The scheduler honors deadlines as I19 hard constraints; `J/W_render` optimization happens within the deadline envelope.

Deadline declaration patterns:

- **Steady-state display:** deadline = next vsync. Frame budget = display refresh period (16.67ms at 60Hz, 8.33ms at 120Hz, etc.). The deadline regenerates every vsync.
- **Variable refresh rate:** deadline = next-allowed-vsync, which is bounded by the display's VRR window. The compositor declares the bound; the scheduler treats the upper bound as the deadline.
- **Composited animation:** deadline is the animation's frame budget, declared by the animating subject. Typically equal to vsync but may be longer for low-priority animation (subject opts into a longer deadline to save joules).

Missing a deadline is a correctness failure observable to the subject — the kernel reports the miss, the subject's `W_render` does not increment, and the subject may choose to lower its quality settings, drop frames intentionally, or release its `DisplayOutput` capability. The scheduler does not silently skip frames on the subject's behalf.

---

## 3. Class: `Inference`

### 3.1 Unit

`W_infer = number of output tokens emitted by an inference workload, where each token satisfies the workload's correctness predicate.`

An "output token" is the unit produced by the inference computation: for an LLM, a sampled token from the output distribution; for an image model, a generated pixel-block at the model's native granularity; for a sensor-classification model, an emitted classification with its confidence.

This aligns with the η = I_output / E_system formulation already in use at joulesperbit.ai, which is the prior art the project inherits. `W_infer` is `I_output` in that formulation. `J` is `E_system`. `J/W_infer` is `1/η`.

### 3.2 Counting protocol

The inference runtime, holding an `Inference` capability bound to a specific model and substrate, invokes the kernel counter on each emitted token. The counter records:

- The subject ID
- The model identity (a hash of the model weights, attested at load time)
- The substrate that produced the token
- The position in the output sequence
- The joule cost attributed to this token's production (O1 oracle, integrated over the inference step's compute interval, including any prefill cost amortized per the runtime's declared amortization scheme)

The kernel does not see token values. Token *content* is the subject's data and is not exposed to the kernel; only the *count* of valid emissions is.

### 3.3 Validity predicate

A token counts toward `W_infer` if and only if:

- It was emitted by an inference computation under a valid `Inference` capability.
- The model hash matches an attested model in the kernel's model registry.
- The token satisfies the runtime's declared correctness predicate. Examples:
  - For deterministic models: bit-exact match against a reference computation (sampled audit, not per-token).
  - For sampled models: the sampling distribution matches the model's declared output distribution within a declared statistical bound.
  - For models declared as unverifiable: the token counts on emission with no further check, but the workload's joule attribution is tagged "unverified" and the scheduler may apply a discount factor.

A token emitted by a model whose hash does not match the registry counts zero. A token emitted outside the `Inference` capability (e.g., by speculative execution that was rolled back) counts zero. A token whose emission was preempted before completion counts zero.

### 3.4 Aggregation rule

Across subjects: additive per subject.

Across substrates: a token is attributed to the substrate that produced it. A multi-substrate inference (prefill on GPU, decode on CPU) attributes prefill joules to the GPU and decode joules to the CPU; the token itself is attributed to the substrate that emitted it (the decode substrate).

Across time: monotone non-decreasing within an interval. Same windowing rules as `W_render`.

`W_infer` does not aggregate with `W_render`. They are different classes with different units; the scheduler's multi-class behavior is Q7.

### 3.5 Latency relationship (I19)

Inference may declare itself interactive or batch.

**Interactive inference** (chat assistants, real-time translation, streaming TTS) declares a per-token deadline (typically 50–200ms inter-token latency) and a first-token deadline (typically 500ms–2s). The scheduler honors both as I19 constraints. A token emitted after its deadline does not count toward `W_infer`. This is identical in shape to rendering's deadline discipline; the predicate is the same shape, the units differ.

**Batch inference** (training, large-scale evaluation, offline generation) declares no latency floor and is scheduled purely on `J/W_infer`. The scheduler is free to defer batch inference indefinitely if doing so improves system-level `J/W` — for example, scheduling batch work onto a substrate during its energy-optimal frequency window, or co-scheduling with other batch work to amortize substrate wakeup costs.

The interactive/batch distinction is declared at capability acquisition time and cannot be upgraded without releasing and reacquiring the capability. The scheduler does not infer interactivity from token-emission patterns.

---

## 4. What this document deliberately does not define

These workload classes exist and the scheduler will need definitions for them eventually. They are not defined here because v0.1 does not require them, and defining them speculatively would bake assumptions in before workloads exist to test them.

- **`Storage`** — durable bytes written. Open: how to count writes that get amplified by filesystem journaling, how to handle compression (bytes-in vs. bytes-on-medium), how to handle tiered storage where the energy cost of "durable" varies by tier.
- **`Sensing`** — samples acquired. Open: how to count samples that fail validity (sensor saturation, dropped packets), how to handle event-driven vs. periodic sensing, how to attribute energy when a sensor must be powered up to sample.
- **`Network`** — packets delivered. Open: where in the stack the packet "counts" (sent on wire, ACKed, received by application), how to handle retransmissions, how to handle multicast where one packet has many recipients.
- **`Control`** — actuator commands issued. Open: how to count commands that are issued but not physically realized (motor stalls, valves stuck), how to attribute energy when control loops are tight and individual commands are not meaningful in isolation.

Each class will get its own resolution document when a workload that needs it appears. Until then the scheduler refuses dispatch under I16's pessimistic-defaults clause, which is the correct behavior — better to refuse than to silently optimize the wrong objective.

---

## 5. Cross-class properties

These hold across all `W` definitions, present and future.

**P1 — `W` is non-fungible across classes.** A frame is not a token. A token is not a packet. The scheduler does not convert between units. Multi-class workloads have multi-objective optimization (Q7), not a unified scalar.

**P2 — `W` is countable, not estimated.** Every increment of `W` corresponds to an observable event in the world (a frame presented, a token emitted, a byte durably written). Estimation is for `J` (the energy oracle); counting is for `W`. The asymmetry is deliberate: we permit measurement error in joules because physics, but not in work delivered, because if we cannot count work we cannot say what the system did.

**P3 — `W` is attributed at the moment of observable delivery.** Not at the start of the work. Not at the moment of capability invocation. Not on intent. Speculative work that does not reach the world counts zero. This is the strongest possible discipline against the OS optimizing for activity rather than outcome — the same disease that makes Linux schedule daemons that wake up to do nothing.

**P4 — `W` increments are observed by the kernel through capability-bound channels.** The subject does not self-report `W`. The kernel observes the delivery event via the capability that gates the delivery (`DisplayOutput`, `Inference`, future capabilities for storage/network/control) and increments the counter from the trusted side. The subject cannot inflate its `W` to make its `J/W` look better.

**P5 — `J/W` is reported, not just optimized.** The kernel publishes per-subject and per-class `J/W` rolling-window statistics through a capability-gated introspection surface. Subjects can see their own efficiency. The system can report aggregate efficiency. This is the foundation for joulesperbit.ai-style external attestation: the OS itself produces the η measurements that joulesperbit.ai consumes.

---

## 6. Implementation obligations added by this resolution

These extend the THEOREM's obligation list (O1–O10).

**O11 — Frame counter.** A kernel-resident counter exposed to the `DisplayOutput` capability surface, incremented on hardware presentation events with attribution to subject and substrate. Required by `W_render` counting protocol.

**O12 — Token counter.** A kernel-resident counter exposed to the `Inference` capability surface, incremented on validated token emissions with attribution to subject, model hash, and substrate. Required by `W_infer` counting protocol.

**O13 — Model registry.** A kernel-resident structure mapping attested model hashes to validity predicates (deterministic, sampled with declared distribution, or unverified). Required by `W_infer` validity predicate.

**O14 — `J/W` reporting surface.** A capability-gated introspection interface that publishes per-subject, per-class, per-substrate, per-window `J/W` statistics. Required by P5.

---

## 7. Provenance

Q1 resolution v0.1 drafted May 8, 2026, against THEOREM v0.2.

Defines `W` for `Rendering` and `Inference`. Defers `Storage`, `Sensing`, `Network`, `Control` to future resolution documents. Adds O11–O14 to the implementation obligations.

Authors: David, Claude. To be reviewed and contested before the scheduler implementation begins. The first scheduler commit must reference this resolution document at its current version.
