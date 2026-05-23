# The OS Theorem

*A specification for an energy-first, capability-bounded, single-language operating system.*

**Version 0.2** — Refined to accommodate browser-class workloads, graphics substrates, and runtime code generation. Substrate v0.1 declared.

**Changelog from v0.1:**
- A3 / I10 refined: OS source remains single-language Rust; user-supplied content executed through in-tree Rust runtimes is out of scope.
- New axiom A7: substrates are added natively, not abstracted.
- New invariants I18 (runtime code generation is capability-gated), I19 (declared interactive workloads have latency floors honored as constraints), I20 (irreversibility classes are explicit).
- N4 refined: throughput is not promised; declared latency floors are honored.
- New non-property N6: substrate abstraction is not promised.
- New implementation obligations O8 (JIT capability surface), O9 (graphics substrate driver), O10 (substrate inventory).
- Section 8 added: substrate roadmap with v0.1 declaration.

---

## 0. Preamble

This document specifies the invariants an operating system must satisfy to be an instance of *this* OS. It is not an implementation guide. It is a theorem: a set of properties the implementation must preserve. Code that violates these properties is not a bug in the implementation — it is a refutation of the implementation's claim to be this OS.

The theorem is small on purpose. Every property here costs something to maintain. Adding properties is a one-way door. Removing them is a fork.

The theorem assumes one fact about the world: **energy is now the binding constraint on computation.** Not throughput, not latency, not transistor density, not memory bandwidth — energy. Every other design decision in this document follows from that fact. If that fact ceases to be true, the theorem is wrong and the OS should be redesigned, not patched.

---

## 1. Axioms

These are not proven. They are asserted. The rest of the document depends on them.

**A1 — Energy is a first-class physical resource.** Joules are conserved, accountable, and finite within any bounded interval. The OS treats joules the way prior OSes treated CPU cycles: as the resource to be allocated, accounted, and optimized over.

**A2 — The trusted computing base must be small enough to be reasoned about by one human.** The kernel proper shall not exceed 50,000 lines of source. This is not aesthetic. It is the threshold below which a single mind can hold the whole system in working memory, and above which the system becomes a committee artifact.

**A3 — One language across the OS source surface.** The kernel, the userland, the drivers, the build system, the in-tree runtimes — all Rust. Other languages exist outside the OS source tree or not at all. Boundaries between languages are where bugs and bloat live; collapsing the boundaries is the only structural defense. User-supplied content that the OS *executes* (JavaScript, WebAssembly, shader bytecode) is governed by capabilities (A4) and is not OS source.

**A4 — Authority is capability-bounded. There is no ambient authority.** A subject may perform an action only if it holds an unforgeable token granting that action. The token is a Rust type, not a runtime check. There is no superuser, no root, no setuid, no capability-by-virtue-of-being.

**A5 — Idle is the default state. Work is the perturbation.** The system in absence of demand consumes the minimum joules physically achievable on the substrate. All work justifies its energy cost against its output.

**A6 — Heterogeneity is native, not abstracted.** Compute substrates (general cores, accelerators, PIM, CGRA, GPUs, harvesters) are first-class scheduling targets with distinct energy curves. The OS does not pretend they are uniform. POSIX's "process on a CPU" is a special case, not the model.

**A7 — Substrates are added natively, not abstracted.** Each new substrate class brings its own driver, its own energy model, its own capability surface. The OS does not maintain a substrate-agnostic abstraction layer; the scheduler dispatches *across* substrates rather than *over* them. Adding a substrate adds code; it does not add a layer.

---

## 2. Invariants

These are properties the implementation must preserve at all times. Violating an invariant is a kernel panic, a build failure, or a refused commit — not a runtime warning.

### 2.1 Energy invariants

**I1 — Total energy accounting.** Every joule consumed by the system is attributed to a *cause*: a capability invocation, a hardware-mandated baseline, or kernel housekeeping. The sum of attributed joules over any interval equals the measured joules consumed over that interval, modulo measurement precision. There is no unattributed energy.

**I2 — Energy attribution is causal.** A subject's joule account reflects work *done on its behalf*, including work done by other subjects in service of its capability invocations. Joules do not disappear into shared infrastructure unbilled.

**I3 — Joules-per-output is the scheduler's primary objective function.** The scheduler optimizes `J/W` where `J` is joules consumed and `W` is useful work delivered, defined per workload class. Throughput, latency, fairness are constraints on this optimization, not the optimization itself.

**I4 — Continuous gearing.** The system's energy response to demand is monotonically smooth. There exist no fixed P-states, no governor staircases, no discrete frequency tiers visible to scheduling decisions. The hardware may quantize internally; the OS shall not amplify that quantization.

**I5 — True idle.** Subsystems with no pending work shall reach the lowest-power coherent state the substrate permits, without latency penalty for resumption beyond the substrate's physical floor. "Low-power" is not "off"; the OS shall reach off where off is achievable.

### 2.2 Authority invariants

**I6 — No ambient authority.** A subject possesses no rights except those granted by capabilities held in its address space. A subject with no capabilities can do nothing but terminate.

**I7 — Capabilities are unforgeable by construction.** A capability is a Rust type whose constructor is private to the kernel module that issues it. The type system prevents fabrication; the kernel prevents duplication beyond what the issuing module permits. Runtime forgery is a category error, not a security hole.

**I8 — Authority composes by intersection, not union.** When a subject delegates a capability, the delegate receives at most the rights the delegator held. There is no path by which delegation increases authority.

**I9 — Revocation is constructive.** Every issued capability has a defined revocation path. Revocation completes in bounded time and is observable to the holder. There are no zombie capabilities.

### 2.3 Structural invariants

**I10 — Single language in the OS tree.** The build system shall reject any source file in any language other than Rust, with the sole exception of architecture-specific assembly files explicitly enumerated in the kernel manifest. There is no C, no C++, no scripting language used for OS source. Configuration is Rust. Build orchestration is Rust. Tests are Rust. Runtimes for user-supplied content (JS engines, WASM runtimes, shader compilers) are themselves Rust and live in the tree; the content they consume does not.

**I11 — TCB ceiling.** The kernel proper, defined as code executing with hardware-privileged authority or producing capability tokens, shall not exceed 50,000 lines of Rust source as measured by `tokei`, excluding tests, comments, and architecture-specific assembly. Crossing this ceiling is a build failure.

**I12 — No dynamic allocation in the kernel hot path.** The scheduler, the interrupt handlers, the capability check path, and the energy accounting path shall not perform heap allocation. Memory for these paths is statically reserved at boot. Allocation failure as a kernel runtime concept is eliminated.

**I13 — Reversibility of state-changing operations.** Every operation that mutates persistent state shall produce a record sufficient to reverse the mutation, retained for a bounded window. The default lazy action is recoverable; irreversibility is opt-in and capability-gated.

**I14 — Intermittent-safe by default.** Loss of power at any instruction boundary shall not corrupt persistent state nor leave the system in an unresumable configuration. Computations that cannot tolerate intermittency declare so explicitly and acquire the energy-budget capability that guarantees their completion.

### 2.4 Substrate invariants

**I15 — Heterogeneous dispatch.** The scheduler routes work to the substrate that minimizes `J/W` for that work's shape, subject to capability constraints. The scheduler does not assume substrates are interchangeable. A work item rejected by all available substrates fails explicitly; it is never silently emulated on a wrong-fit substrate.

**I16 — Workload shape is declared.** Every schedulable unit declares its shape: predicted duration class, latency tolerance, parallelism, memory access pattern, energy budget. Undeclared shape is a build-time error for static work and a default-pessimistic shape for dynamic work. The scheduler does not infer shape from observation alone.

**I17 — Substrate energy curves are kernel-known.** For each substrate the kernel maintains a calibrated energy model: joules per work unit as a function of frequency, voltage, and workload shape. Models are measured, not assumed. Substrates without a calibrated model are ineligible for scheduling.

### 2.5 Execution invariants

**I18 — Runtime code generation is capability-gated.** A subject that emits machine code at runtime — a JIT compiler, an inline assembler, a dynamic linker — holds an explicit `CodeGen` capability bound to a specific memory region of declared size. W^X is enforced by capability state, not by ambient page-protection. A subject without `CodeGen` cannot produce executable bytes by any path. A subject with `CodeGen` cannot extend its scope beyond the capability's bound region.

**I19 — Declared interactive workloads have latency floors honored as constraints.** A workload may declare a latency floor (input-to-output deadline) under I16. The scheduler honors declared floors as hard constraints; `J/W` optimization happens within the constraint envelope, not against it. Frame deadlines, audio buffer deadlines, and input-event deadlines are not throughput; missing them is correctness failure, not performance failure.

**I20 — Irreversibility is named, not implicit.** Operations whose effects cannot be reversed (display output to a physical panel, network packet transmission, actuator commands, cryptographic key destruction) are members of declared irreversibility classes. Each class is capability-gated. I13 (reversibility log) does not apply to operations declared under I20; the capability itself is the audit trail.

---

## 3. Non-properties

These are things the OS deliberately does **not** guarantee. Listing them is as important as listing the invariants, because they bound the theorem.

**N1 — POSIX compatibility is not promised.** Programs written against POSIX may run via a translation layer or may not run at all. The translation layer is userland, optional, and outside the TCB. POSIX semantics that conflict with the invariants (ambient authority, fork's address-space duplication, signals as ambient interrupts) are not preserved.

**N2 — Linux ABI compatibility is not promised.** See N1. Lifting Linux binaries is a courtesy, not a contract.

**N3 — Backward compatibility across major versions is not promised.** The theorem is allowed to evolve. Implementations track the theorem version they satisfy. There is no stable kernel ABI promise of the kind Linus enforces; the contract is at the capability-and-invariant level, not the syscall-number level.

**N4 — Maximum throughput is not promised; declared latency floors are honored.** Where throughput conflicts with `J/W`, joules win. A workload optimized for `J/W` may run slower in wall-clock time than the same workload on Linux. Declared interactive latency floors (I19) are the exception: they are honored as constraints, and `J/W` is optimized within them.

**N5 — Generality is not promised.** The OS targets workloads where energy is the binding constraint. Workloads where energy is free and throughput is paramount are not the design center and should run on Linux.

**N6 — Substrate abstraction is not promised.** Per A7, the OS does not present a uniform compute fabric. Code that targets a substrate targets it natively. Portability is achieved by recompilation against the target substrate's capability surface, not by emulation.

---

## 4. Implementation obligations

These are the things the implementation must produce to be checkable against the theorem.

**O1 — Energy oracle.** A kernel-resident component that produces, at any scheduling decision, the per-substrate per-shape joule estimate the scheduler consumes. Backed by RAPL on x86, equivalent counters on ARM, AMU/AGX telemetry on Apple Silicon, calibrated models elsewhere. The oracle's accuracy is measured and published; below a defined threshold the substrate is ineligible (I17).

**O2 — Capability ledger.** A kernel-resident structure tracking every capability ever issued, its current holders, its delegation graph, and its revocation status. Supports I7, I8, I9. The ledger's invariants are checked at each capability operation.

**O3 — Workload shape registry.** A compile-time and runtime structure carrying the declared shape (I16) of every schedulable unit. The compiler enforces declaration for static work; the kernel rejects undeclared dynamic work or assigns pessimistic defaults.

**O4 — Reversibility log.** A bounded-retention record of state mutations sufficient to satisfy I13. Capability-gated for irreversible operations; entries for operations declared under I20 are replaced by the capability invocation record itself.

**O5 — Single-language gate.** A build-system component that fails the build on any non-Rust source outside the assembly manifest (I10).

**O6 — TCB measurement.** A CI-resident component that measures kernel-proper line count at every commit and fails on I11 violation.

**O7 — Theorem conformance suite.** A test suite that exercises each invariant and produces a pass/fail per invariant per build. The suite's failures are the only kind of failure that block a release; performance regressions are not, because the theorem does not promise performance (N4).

**O8 — JIT capability surface.** A `CodeGen` capability type with private constructor in the kernel, granting bounded executable-memory rights to its holder. JIT-bearing runtimes (V8-equivalent JS, Wasmtime, future WebGPU shader compilers) acquire and release `CodeGen` explicitly; the kernel verifies bound and lifetime. Required by I18.

**O9 — Graphics substrate driver.** A userland Rust GPU driver for the v0.1 substrate (Apple Silicon AGX), holding capabilities for MMIO regions, DMA, and firmware command submission. Exposes a Vulkan-or-equivalent capability-bounded interface. Energy model for the GPU is required under I17. Subsequent substrates (AMD RDNA, OpenIE silicon) are added natively per A7, not behind a unified abstraction.

**O10 — Substrate inventory.** A kernel-resident manifest enumerating every supported substrate, its driver location, its calibrated energy model, and its current eligibility for scheduling. Adding a substrate is an explicit manifest amendment; substrates not in the manifest are not dispatched to.

---

## 5. Open questions

These are the parts of the theorem that are not yet settled and require resolution before v1.0.

**Q1 — Definition of "useful work" in `J/W`.** Per workload class is not yet specified. Inference produces tokens; storage produces durable bytes; sensing produces samples; control produces actuator commands; rendering produces frames-meeting-deadline. Each class needs a defined unit. The scheduler cannot optimize what it cannot count.

**Q2 — Calibration protocol for substrate energy models (I17).** How models are measured, how often they are revalidated, how drift is detected, how new substrates are admitted. This is a real engineering project, not a paragraph.

**Q3 — Capability revocation latency bound (I9).** "Bounded time" is currently undefined. Hard real-time systems need microseconds; cooperative systems can accept milliseconds. The bound is policy and probably per-capability-class.

**Q4 — Reversibility window (I13).** How long the reversibility log retains records. Storage cost vs. recovery utility. Likely capability-gated and tunable per subject.

**Q5 — Lifting strategy for foreign code.** c2rust + AI tightening is the working answer. The boundary between lifted-unsafe-Rust and native-safe-Rust needs a specification: when does lifted code graduate, who certifies it, what does the type system enforce at the boundary.

**Q6 — `CodeGen` capability granularity.** Per-region, per-page, per-allocation. Performance implications for JIT throughput vs. tightness of W^X enforcement. Open.

**Q7 — The `J/W` objective under multiple concurrent workloads.** Pareto-optimal across workloads is the textbook answer; the practical answer is a scheduling policy that doesn't degenerate when workloads have orthogonal shapes. Open.

**Q8 — Latency-floor admission control (I19).** When declared latency floors cannot be jointly satisfied, the scheduler rejects work rather than miss deadlines. The admission-control policy is open: which workload yields, on what basis, and how is the rejection observable to the holder.

---

## 6. What this document is not

It is not an architecture diagram. It is not a roadmap. It is not a marketing document. It is not a manifesto.

It is a contract. Any future code, any future component, any future fork is checked against this document. Code that satisfies the invariants is the OS. Code that violates them, however clever, is something else.

The cathedral is the theorem. The code is the bazaar.

---

## 7. Provenance

Theorem v0.1 drafted May 8, 2026.
Theorem v0.2 drafted May 8, 2026, refining I10, adding A7, I18–I20, N6, O8–O10, and the substrate roadmap.

Authors: David, Claude. To be reviewed, contested, and revised before any kernel code is written. The first commit to the kernel repository must include this document, signed at its current version, in `/docs/THEOREM.md`. The kernel's CI must verify the signature.

---

## 8. Substrate roadmap

The theorem is substrate-neutral by design (A6, A7) but the implementation is not. v0.1 declares its target substrate so the energy oracle, the GPU driver, and the workload-shape registry can be concrete artifacts rather than placeholders.

**Substrate v0.1 — Apple Silicon (M-series).**

Selected because:
- The Asahi Linux project's Rust GPU driver (AGX) is the only production-quality Rust GPU driver in existence and provides the lift target for O9.
- M-series GPUs are the most energy-efficient consumer GPUs available, providing the strongest demonstration substrate for I3 (`J/W`) and I5 (true idle).
- Unified memory architecture removes a class of dispatch complexity for v0.1, allowing the scheduler-graphics integration to be built once before being generalized.
- Apple's AMU (Apple Monitoring Unit) and AGX telemetry expose the per-domain energy counters required by O1 at fidelity comparable to or exceeding RAPL.

Risks accepted:
- Apple controls the hardware roadmap and is not aligned with the project's goals.
- Each new chip generation requires reverse-engineering effort tracking Asahi upstream.
- Firmware blobs for AGX and ANE remain outside the single-language guarantee; they are bound at the capability surface (O9) and treated as opaque substrates with measured energy curves under I17.

**Substrate v0.2 — AMD RDNA.**

Added when v0.1 is stable. Selected because AMD GPU documentation is open, the architecture is available across consumer, workstation, embedded, and increasingly automotive form factors, and the existing Rust work in Mesa-adjacent drivers provides a lift path. RDNA is the path to industrial / automotive / corridor.openie.dev deployment that Apple Silicon structurally cannot reach.

**Substrate v1.0+ — OpenIE silicon.**

Added when the OpenIE chiplet SOM is fabricated and characterized. This substrate is where the theorem's heterogeneous dispatch (I15), substrate-native discipline (A7), and intermittent-safety (I14) are exercised in their design center: PIM tiles, CGRA fabric, RISC-V cores, harvesting front-end, all dispatched to natively under the same scheduler that drove v0.1 on Apple Silicon. The validation that the same OS, unchanged at the theorem level, runs across substrates of fundamentally different character is the demonstration that A7 was the right axiom.

Substrates beyond v1.0 are not declared in this document and are added by manifest amendment under O10.
