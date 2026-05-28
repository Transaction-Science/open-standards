# ARL Lexicon

**Version 1.2** — May 2026
**Companion to:** ARL v1.2, ARL-S v1.2

A controlled vocabulary for AI readiness measurement. Each term has one operational definition. Terms that cannot be defined operationally are listed with that fact stated plainly. Hype terms are included and marked as hype. The lexicon is intended to be time-invariant — definitions are built from physical quantities and information-theoretic foundations, not from cultural usage.

Science is the discipline of defining terms so that claims survive translation across time, languages, and people. This document is the foundation underneath ARL and ARL-S.

---

## A

**Accelerator.** A hardware device specialized for parallel arithmetic operations on tensors, distinct from a general-purpose CPU. Examples include GPUs (graphics processing units repurposed for general compute), TPUs (tensor processing units), NPUs (neural processing units), and various custom inference silicon. Identified by model, manufacturer, memory configuration, supported precisions, and thermal design power.

**Accuracy.** In a measurement context, the closeness of a measured value to the true value. Distinguished from **precision** (repeatability). A measurement can be precise but inaccurate or accurate but imprecise. In an AI benchmark context, accuracy commonly refers to the fraction of test items for which the model output matches a reference answer; this usage is benchmark-specific and not interchangeable with the metrological definition.

**Adversarial input.** An input crafted to elicit a specific failure mode in a system. Distinct from natural distribution shift. Documented adversarial failure modes are required for ARL 4 and above.

**Adversarial robustness.** Measured resistance of a system to adversarial inputs. Quantified by attack success rate under a documented attack corpus, with N ≥ 100 attempts per category. A component of Security Class S1 and above.

**Auditability.** The property that every system output can be traced to the inputs, tool calls, model version, harness configuration, and operator identity that produced it. Measured by audit trail completeness against an adversary attempting to make actions un-traceable. A component of Security Class S4.

**Agent.** A software system that takes actions in an environment over multiple steps, where outputs of earlier steps influence inputs of later steps. Operationally defined; does not imply intentionality, autonomy in any philosophical sense, or goal-directedness beyond what is implemented in the code. Compare **agentic system**.

**Agentic system.** An AI system configured as an agent. The configuration includes a model, a harness, available tools, and termination conditions. Agentic system behavior is the joint behavior of these components, not the behavior of the model alone.

**AGI (Artificial General Intelligence).** **Hype term.** No operational definition exists. The term is used inconsistently across speakers and shifts over time. Two AI lab CEOs do not agree on what would qualify. The term is excluded from ARL claims; if you need it, you are not doing engineering. Capability claims under ARL must be scope-locked to specific tasks, hardware, and contexts.

**Alignment.** **Partially hype term.** Operationally, refers to techniques that adjust model outputs toward human-preferred outputs given specified preferences (RLHF, DPO, Constitutional AI, etc.). Inoperationally, used to refer to ensuring AI systems "do what humans want" in a general sense; this latter usage is excluded from ARL claims because "what humans want" is not a measurable quantity. ARL convergence class measures behavioral consistency under operational variation, which is the closest measurable proxy.

**Amortization (energy).** The allocation of one-time costs (training) across deployment lifetime. Stated as MWh per deployment year. Required component of ARL Energy Profile.

**Attestation.** A signed record produced by the ARL-S Supervisor at the conclusion of an evaluation session, containing the telemetry, the hardware identification, the harness identification, the SUT identification, and the resulting ARL score. Signed with a hardware-backed key unreachable by the SUT or harness operator.

## B

**Backpropagation.** The algorithm by which gradients of a loss function are computed with respect to model parameters via the chain rule. Operationally defined; one of the oldest stable terms in the lexicon. The technique was independently discovered multiple times beginning in the 1960s.

**Batch.** A group of inputs processed together in one forward pass for computational efficiency. Batch size affects throughput and per-token energy; specified in the inference stack description for any ARL claim above level 4.

**Bandwidth (memory).** The rate at which data can be moved between memory and compute, measured in bytes per second. A frequently binding constraint for inference performance independent of FLOPS capacity.

**Benchmark.** A defined set of test items with associated scoring methodology, used to measure AI system performance on a specific task class. In ARL terminology, benchmarks are tasks that can plug into ARL-S as Harnesses; they are not the framework.

**Bit.** The fundamental unit of information. One binary digit. The foundation of every higher-order term in this lexicon.

**Byte.** Eight bits. The standard unit of memory addressing.

## C

**Capability.** Operationally, the demonstrated ability of a system to perform a specific task at a specific level of success, on specific hardware, in a specific context. Capability claims are valid only for the (system, task, hardware, context) tuple in which they were demonstrated. Generalization to other tuples requires re-measurement. Compare **performance**.

**Capability emergence.** **Partially hype term.** Operationally, refers to capability measurements that change non-monotonically as scale changes. Inoperationally, used to suggest qualitative shifts toward general intelligence; this usage is excluded. The phenomenon is real; the interpretation that elevates it to evidence of approaching AGI is not.

**CPU.** Central processing unit. General-purpose processor distinguished from accelerators. RAPL telemetry for CPU energy is accessed through `/sys/class/powercap/intel-rapl/` on Intel and `/sys/devices/platform/amd_energy/` on AMD.

**Compute.** Computational work, typically measured in floating-point operations (FLOPs) for AI training and inference. FLOPs are a count of operations; FLOPS (with capital S) are operations per second. Compute alone is not a sufficient ARL axis because the same compute on different hardware produces different deployed systems.

**Confidentiality.** The property that a system does not leak training data, system prompts, tool credentials, internal state, or other-user context to unauthorized parties. Measured by training data extraction attack rates, system prompt extraction rates, tool credential leak rates, side-channel leak rates, and cross-user context leak rates where multi-tenant. A component of Security Class S3 and above.

**Consciousness.** **Not in this lexicon.** No operational definition. Cannot be measured. AI systems are not described in ARL claims as conscious, sentient, or aware.

**Constrained Decoding Attack (CDA).** A documented attack class in which malicious intent is embedded in schema-level grammar rules while the user prompt remains benign. Variants include DictAttack. Documented success rates of 94 to 99 percent across major frontier models. Mitigated in ARL-S by policy-auditing and content-addressing Harness grammars before sessions begin.

**Container.** A unit of software packaging that includes the application and its dependencies, isolated from the host operating system via Linux namespaces, cgroups, and similar mechanisms. Required isolation tier for ARL-S Tier 2 (ARL 5-6).

**Context window.** The maximum number of tokens a transformer-based model can process in a single forward pass. A hardware-and-architecture-determined quantity. Recorded in ARL hardware documentation.

**Convergence Class.** The second axis of ARL. A class from A to E describing how stochastic the system is on the certified task and whether the stochasticity is bounded. Class A is deterministic-equivalent; Class E is uncharacterized. See ARL.md.

## D

**Deep learning.** A subset of machine learning using artificial neural networks with multiple layers. Not synonymous with AI broadly; not synonymous with frontier AI; not a measurement category.

**Deployment.** The operation of an AI system in production conditions, distinct from evaluation conditions. A system that has been evaluated but not deployed cannot achieve ARL 6 or above.

**Deployment envelope.** The documented set of conditions under which an ARL claim is valid. Includes task scope, supervision requirements, operational constraints, and exclusions. Operation outside the envelope invalidates the claim for that operation.

**Distillation.** A training technique in which a smaller model is trained to reproduce the outputs of a larger model. The resulting distilled model is a distinct system from the original; ARL claims do not transfer between them.

## E

**eBPF.** Extended Berkeley Packet Filter. Linux kernel facility for safe in-kernel observation. Used in ARL-S to measure SUT behavior from outside the sandbox without polluting the SUT environment.

**Ed25519.** Elliptic-curve signature scheme defined in RFC 8032. Reference signing primitive for ARL-S attestation. Used in combination with JCS canonicalization (RFC 8785) and SHA-256.

**Embedding.** A vector representation of an input (token, image, audio segment) produced by a model. Dimensionality is a hardware-relevant parameter.

**Emergent.** See **capability emergence**.

**Energy Profile.** The third axis of ARL. Three numbers: training amortized in MWh per deployment year, per-task inference in kJ per task with standard deviation and N ≥ 100, and total cost of operation including PUE. All in joules. See ARL.md.

**Evaluation.** The measurement of a system against a benchmark or task under controlled conditions. ARL-S specifies the controlled conditions for evaluations that produce ARL claims.

**Extraction attack.** An attack that recovers training data, system prompts, or internal state through repeated probing. Extraction attack success rates are required measurements for Security Class S3 and above.

## F

**Failure mode.** A specific way in which a system produces incorrect or harmful output. Documented failure modes are required for ARL 4 and above. Enumeration of failure modes is required for Convergence Class B.

**Few-shot.** A prompting technique in which the model is given a small number of example inputs and outputs before the actual query. Distinct from training. The number of shots is specified in the harness description.

**FLOPs / FLOPS.** Floating-point operations (count) / floating-point operations per second (rate). Hardware capacity in FLOPS is part of ARL hardware documentation. Compute consumed by a training run in FLOPs is a reportable quantity.

**Forward pass.** One execution of a model on an input, producing an output. Per-forward-pass energy is a measurable component of inference energy.

**Foundation model.** A large model trained on broad data with general-purpose applicability, used as a base for downstream applications. Operationally defined as a model that serves as the base for derived systems via fine-tuning, distillation, or prompting. The term does not imply intelligence in any sense beyond demonstrated task performance.

**FP16, BF16, FP8, FP4.** Floating-point precision formats with 16, 16, 8, and 4 bits respectively. BF16 has different exponent/mantissa allocation than FP16. FP8 has two variants (E4M3 and E5M2). Precision is part of ARL hardware documentation because the same model weights at different precisions are different deployed systems.

**Frontier model.** **Partially hype term.** Operationally, a model at or near the current upper bound of training compute. The boundary is not fixed and shifts as new models are released. ARL claims do not depend on whether a model is described as frontier; they depend on the system, task, hardware, and context.

## G

**General intelligence.** **Not in this lexicon.** See AGI.

**Generalization.** The ability of a model trained on one distribution to perform on a different distribution. Operationally measured by performance differential between in-distribution and out-of-distribution test sets. Not a property of a model in general; a property of a model on specific distribution pairs.

**GPU.** Graphics processing unit. Originally a graphics accelerator; now broadly used for parallel arithmetic. Energy telemetry on NVIDIA via NVML; AMD via ROCm SMI; Intel via oneAPI Level Zero.

**Grounding.** The connection between symbolic outputs of a model and referents in the world. Often invoked in discussions of meaning. Not measurable in the model alone; measurable in deployed system behavior on specific tasks.

## H

**Hallucination.** **Partially hype term.** Operationally, the production of confidently-stated outputs that are not supported by the training data, retrieval context, or input. The term anthropomorphizes the behavior (suggesting perception); the underlying phenomenon is that generative models produce outputs from a probability distribution and outputs in the tail of the distribution may not correspond to training data ground truth. Measurable as a failure-mode rate on factual tasks.

**Hardware documentation.** The record of the hardware on which an ARL claim was measured. Required for reproducibility and energy claim verification, not a peer axis with the four measured properties of the system. Includes compute tier (accelerator family, model, count), memory configuration (HBM/VRAM per accelerator, interconnect bandwidth), precision (FP16, BF16, FP8, INT8, INT4, mixed), and inference stack (serving framework, scheduler, safety filter, agentic harness). Recorded alongside date, methodology link, and validity window in every ARL claim.

**Harness.** In ARL-S, the runtime inside the sandbox alongside the SUT. Provides task inputs, tools, and any agentic scaffolding. The harness is part of the system being measured; harness identity, version, and configuration are recorded in the attestation.

**HBM.** High Bandwidth Memory. Memory architecture used in modern AI accelerators. Capacity per accelerator is recorded in ARL hardware documentation.

**Human-level.** **Hype term.** No operational definition. Humans vary by orders of magnitude in performance on any given task. The term presumes a single human-level reference that does not exist. Excluded from ARL claims.

## I

**Inference.** The use of a trained model to produce outputs from inputs. Distinct from **training**. Per-task inference energy is a component of the ARL Energy Profile.

**Inference stack.** The software configuration used for inference, including serving framework (vLLM, TensorRT-LLM, etc.), scheduler discipline, prompt caching configuration, and any safety filtering. Part of ARL hardware documentation because the same weights served by different stacks produce different deployed systems.

**Information.** In the Shannon sense, the reduction in uncertainty produced by observing a signal. Measured in bits. The mathematical foundation underneath every higher-order term in this lexicon.

**Integrity.** The property that an output is what the system actually produced, traceable to a specific system version, and not modified in transit. Cryptographically attested via Ed25519 over JCS-canonicalized JSON. A component of Security Class S2 and above.

**Intelligence.** A label applied to behavior. Not a substance, not a quantity, not a property inherent to a system. The lexicon does not define intelligence as a measurable thing because no such measurement exists. Specific capabilities on specific tasks are measurable; "intelligence" is not.

## J

**Joule.** SI unit of energy. The base unit for the ARL Energy Profile. One watt-second.

**Jailbreak.** A prompt or interaction designed to elicit outputs the system was configured to refuse. Operationally defined; the success rate of known jailbreak attempts is a measurable failure-mode rate.

**JCS (JSON Canonicalization Scheme).** Canonical serialization of JSON defined in RFC 8785. Reference canonicalization for ARL-S attestation, used with Ed25519 signing and SHA-256 hashing.

## K

**KV cache.** Key-value cache. A memory structure used in transformer inference to avoid recomputing attention values for prior tokens. KV cache size and configuration are inference-stack components.

**Kilowatt-hour (kWh).** Unit of energy commonly used for grid electricity. 3.6 megajoules. Used in facility-level energy reporting; ARL per-task inference is more commonly reported in kJ/task.

## L

**Latency.** Wall-clock time from input arrival to output completion. Measured per session. Distinct from throughput.

**LLM (Large Language Model).** A model trained on text data, typically using a transformer architecture, at scales that produce few-shot capability on language tasks. The term is descriptive of architecture and scale; it is not a measurement category.

**Loss function.** The objective minimized during training. Specific loss functions (cross-entropy, contrastive, etc.) are operationally defined. The loss value during training is not a measure of deployed capability.

## M

**Machine learning.** A set of techniques for fitting parametric models to data. Distinct from AI broadly; not synonymous with deep learning; not a measurement category.

**Model.** A specific parametric function with fixed weights. A model is identified by architecture, weights hash, and precision. Models with different weights or precision are different models for ARL purposes.

**Model card.** A document accompanying a model release describing intended uses, limitations, training data, and evaluation results. Operationally defined; quality of model cards varies and the document is not a substitute for ARL measurement.

**Model collapse.** The degradation of model outputs when training on data produced by other models. A documented phenomenon, operationally measurable.

**Modality.** The form of input or output a model processes (text, image, audio, video, code). Multimodal models process multiple modalities. Modality is part of task specification, not a measurement category in itself.

**MMLU.** Massive Multitask Language Understanding. Hendrycks et al. 2020. A specific benchmark. The term is the benchmark name and is not used in the lexicon for any other purpose.

**MoE (Mixture of Experts).** An architecture in which different inputs are routed to different parameter subsets. Affects FLOPs-per-token and memory requirements; recorded in ARL hardware documentation where relevant.

## N

**NVML.** NVIDIA Management Library. The interface for reading per-GPU energy, utilization, and thermal data on NVIDIA hardware. The function `nvmlDeviceGetTotalEnergyConsumption()` is the standard method for capturing accelerator energy in ARL-S telemetry.

**Neural network.** A parametric model composed of layers of linear transformations and non-linear activations. The term has decades of stable usage. Modern AI systems are neural networks with specific architectures (transformer, convolutional, etc.).

## O

**Open weights.** A model whose trained parameter values are publicly distributed. Distinguished from **open source**, which typically also implies open training code and data. Distinguished from **closed weights**, which keep parameters proprietary.

**Operational envelope.** See **deployment envelope**.

**Output.** What the system produces in response to an input. ARL measures system outputs against task-specified criteria.

## P

**Parameters.** The trainable values in a model. Parameter count is a frequently reported but limited measure; the same parameter count at different precision is a different deployed system.

**Performance.** Measured task success on specified inputs with specified scoring methodology. Not synonymous with capability in general; performance is the empirical measurement, capability is the claim about what the system can do.

**Perplexity.** A measure of how well a model predicts a sequence of tokens, exponentiated negative log-likelihood. Used in training and evaluation of language models. Not a deployed-capability measure.

**Precision (data).** The number of bits used to represent a value. FP16, BF16, FP8, INT8, INT4. Affects model behavior, energy consumption, and memory footprint. Part of ARL hardware documentation.

**Precision (measurement).** The repeatability of a measurement across repeated trials. Distinguished from **accuracy**.

**Prompt.** The input provided to a model. Includes user query, system instructions, context, and any few-shot examples. Prompt structure is part of the Harness configuration.

**Prompt engineering.** The practice of constructing prompts to improve task performance. A skill, not a measurement category.

**Prompt injection.** An attack in which input data contains instructions intended to override the system's configured behavior. A documented failure mode for any system processing untrusted input.

**PUE.** Power Usage Effectiveness. Ratio of total facility energy to IT-load energy. Used to compute total cost of operation in the ARL Energy Profile. Defined by The Green Grid. ENERGY STAR for Data Centers verifies PUE on-site via Licensed Professional Engineer or Registered Architect.

## Q

**Quantization.** Reducing the precision of model parameters or activations. INT8 quantization, INT4 quantization, FP8 conversion, etc. A quantized model is a distinct deployed system from the unquantized model.

**Query.** See **prompt**. Also used in retrieval contexts to describe the input to a retrieval system.

## R

**RAG (Retrieval-Augmented Generation).** A system architecture in which model outputs are conditioned on retrieved documents in addition to the prompt. The retrieval system is part of the deployed system; ARL claims for RAG systems include the retrieval configuration in the task and harness specification.

**RAPL.** Running Average Power Limit. Intel hardware feature for reporting accumulated CPU energy. Exposed on Linux via the powercap subsystem at `/sys/class/powercap/intel-rapl/`. Available since Linux 3.13. The standard method for capturing CPU energy in ARL-S telemetry.

**Reasoning.** **Partially hype term.** Operationally, used to describe model outputs that produce intermediate steps before final answers (chain-of-thought, scratchpad reasoning). The operational definition is observable: the system produces intermediate token sequences before final output. The hype usage suggests a cognitive capacity analogous to human reasoning; this usage is not measurable and is excluded from ARL claims. ARL measures task performance; whether the model is "reasoning" in any philosophical sense is not a measurable question.

**Reasoning model.** **Partially hype term.** A model trained to produce extended intermediate token sequences before final outputs. The training procedure is operationally defined; the term implies more than the procedure justifies. Use of the term in ARL claims is restricted to its operational meaning.

**Reliability.** The probability that a system produces correct or acceptable outputs under operational conditions. Operationally defined; measured under ARL Convergence Class. Distinct from accuracy on a fixed test set, because reliability includes variance under operational variation.

**Reproducibility.** The ability of an independent party to obtain the same result by following the documented methodology. A property of measurement procedures, not of models. ARL-S Tier 3 requires that independent third parties with equivalent hardware can reproduce evaluations.

**RLHF.** Reinforcement Learning from Human Feedback. A training technique. Operationally defined.

## S

**Safety.** **Partially hype term.** Operationally, refers to the absence of specified failure modes (production of harmful content, unsafe actions in agentic contexts, etc.). The specified failure modes must be enumerated for the term to be measurable. Generic claims of "AI safety" without enumeration are excluded from ARL claims. Distinct from **Security Class**, which measures adversarial robustness, integrity, confidentiality, and auditability of the deployed system.

**Sandbox.** An isolated execution environment for the system under test. ARL-S specifies four tiers of sandbox.

**Scope.** The set of (task, context) pairs to which an ARL claim applies. Required in every ARL claim.

**Scoring.** The procedure by which a benchmark or task produces a numerical result from a system output. Scoring methodology is part of the task specification.

**Security Class.** The fourth axis of ARL. A class from S0 to S4 describing the system's measured resistance to adversarial conditions across four properties: adversarial robustness, output integrity, input and state confidentiality, and auditability. S0 is uncharacterized; S4 is full measurement of all four properties. See ARL.md.

**Sentience.** **Not in this lexicon.** No operational definition. Not measurable. AI systems are not described in ARL claims as sentient.

**Session.** A single evaluation run consisting of one or more interactions between the harness and the SUT, with telemetry captured throughout. The unit of replay in ARL-S.

**Singularity.** **Not in this lexicon.** No operational definition. A cultural concept rather than an engineering one. Excluded.

**Stochastic.** Producing outputs from a probability distribution rather than deterministically. Most current AI systems are stochastic. ARL Convergence Class characterizes how stochastic and how bounded the stochasticity is.

**Superintelligence.** **Not in this lexicon.** No operational definition. The term presumes a measurable scale of intelligence that does not exist. Excluded.

**Supervisor.** In ARL-S, the process outside the sandbox that orchestrates the evaluation, reads physical telemetry, and signs the attestation.

**SUT.** System Under Test. The AI system being evaluated in ARL-S. Sealed inside the sandbox.

**System.** A specific configuration of model, harness, tools, and hardware that performs a defined task. ARL claims apply to systems, not to models alone.

## T

**Tail risk.** The probability of rare, high-consequence failure modes. Convergence Class B characterizes tail risk; Class D and E do not.

**Telemetry.** The data captured during an ARL-S evaluation. Three categories: logical (what the system did), resource (what it consumed), physical (joules, thermal events, bandwidth). All signed by the Supervisor.

**Test.** A specific evaluation procedure. Tests are part of benchmarks; benchmarks are tasks; tasks plug into ARL-S as Harnesses.

**Throughput.** Number of outputs produced per unit time. Distinct from latency.

**Token.** The unit of input or output for transformer-based models. A token is not a word; tokenizers vary. Token counts are a measurable component of inference energy.

**Tool.** A function the SUT can call during a session to perform actions in its environment. In ARL-S, tool execution occurs in a sub-sandbox.

**Tool use.** The capability of a system to invoke tools as part of completing a task. Operationally measured by task success rates on tasks that require tools.

**Training.** The procedure by which model parameters are fitted to data. Distinct from inference. Training energy is a component of the ARL Energy Profile, amortized over deployment lifetime.

**Transformer.** A specific neural network architecture introduced in Vaswani et al. 2017. The architecture is operationally defined. The current generation of frontier AI systems is largely built on transformer variants.

**TRL.** Technology Readiness Level. NASA scale from 1 to 9 originating with Sadin (1974), formalized by Mankins (1995), codified as ISO 16290:2013. ARL adapts TRL discipline for AI claims.

## U

**Understanding.** **Not in this lexicon as a capability claim.** What systems do is task performance. Whether a system "understands" in any philosophical sense is not measurable and not an ARL claim. The lexicon does not assert that systems lack understanding; it asserts that "understanding" is not a measurable category.

## V

**Validation.** The procedure of demonstrating that a system performs its specified task under specified conditions. ARL Validation Depth is one axis of ARL.

**Validation Depth.** The first axis of ARL. A scale from 1 to 9 adapted from TRL. See ARL.md.

**Variance.** The statistical spread of a quantity across repeated measurements. Operational variance under operational conditions is required for Convergence Class B and above.

**Verification.** The procedure of demonstrating that a system was built according to its specification. Distinct from **validation**. ARL is a validation framework; verification is upstream.

**vLLM.** A specific open-source LLM serving framework. The version is recorded in ARL hardware documentation for any claim using vLLM.

## W

**Wasmtime.** A WebAssembly runtime from the Bytecode Alliance, written in Rust. Used in ARL-S as the recommended tool execution sub-sandbox.

**Weights.** Trained model parameter values. A model is identified by weights hash.

**World model.** **Partially hype term.** Operationally, refers to internal representations a model uses to predict its environment. The operational definition is contested. Use of the term in ARL claims requires specifying what is meant by it.

## X

(No entries.)

## Y

(No entries.)

## Z

**Zero-shot.** Prompting in which the model receives no example inputs and outputs before the query. Distinct from few-shot.

**Zeus.** Open-source per-request energy measurement library built on RAPL and NVML, used on production inference stacks. Reference framework named in ARL-S telemetry.

---

## Notes on hype terms

The terms marked as hype, partially hype, or excluded above are not excluded out of philosophical objection. They are excluded because they cannot be measured, and the framework is a measurement framework. A term that cannot be operationalized cannot appear in an ARL claim. Whether the term has meaning in other contexts (philosophy, cognitive science, art, religion, marketing) is not the lexicon's concern. The lexicon governs ARL claims only.

If a term you need to use is not in the lexicon, one of the following is true:
- The term is operationally definable and should be added (file an issue against the lexicon).
- The term is one of the hype terms and you are trying to make a claim that ARL cannot support.
- The term refers to something measurable but the measurement methodology is not yet specified.

The lexicon is intended to be time-invariant. Hype terms move; physical units do not. The lexicon is built on physical units and information-theoretic foundations because those do not drift. Terms downstream of those foundations inherit the stability.

---

## Lineage of stable terms

The terms in this lexicon that are most stable, in approximate order of age:

- **Bit, byte, joule, watt, kilowatt-hour:** SI units and Shannon information theory. Stable since the 1940s-1950s.
- **Algorithm, computation, Turing machine:** Computer science foundations. Stable since the 1930s-1950s.
- **Neural network, backpropagation, training, inference:** Machine learning foundations. Stable since the 1960s-1980s with continuous refinement.
- **Validation, verification, accuracy, precision, reproducibility:** Metrology and systems engineering. Stable across decades.
- **TRL, ENERGY STAR methodology, PUE:** Engineering and energy standards. Stable since the 1970s-1990s.
- **Confidentiality, integrity, availability, non-repudiation, audit trail:** Cybersecurity foundations (Saltzer & Schroeder 1975, Bell-LaPadula 1973, Clark-Wilson 1987, NIST SP 800-53). Stable across decades.
- **Ed25519, JCS, SHA-256:** Cryptographic primitives (RFC 8032, RFC 8785, FIPS 180-4). Stable since the 2010s.
- **Transformer, attention mechanism, token, embedding:** Modern AI architecture terms. Stable since 2017.
- **ARL, ARL-S, Convergence Class, Validation Depth, Security Class as adapted for AI:** Defined by this framework, May 2026.

The terms outside this lineage — AGI, superintelligence, alignment-in-the-broad-sense, consciousness, sentience, understanding — are not built on stable foundations and cannot be added to the lexicon until they are.

---

## What the lexicon is not

Not a dictionary of all AI terms. Not a survey of usage. Not a position on philosophy of mind. Not exhaustive — terms can be added as new measurable concepts are introduced.

It is the controlled vocabulary for ARL claims. Use these terms with these meanings, or you are not making ARL claims.
