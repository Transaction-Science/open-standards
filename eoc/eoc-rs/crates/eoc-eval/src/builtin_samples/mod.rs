//! Built-in tiny samples for each harness.
//!
//! Each constant is a JSON document with the schema the harness expects
//! in [`crate::harness::DatasetSource::BuiltinSample`]. The samples are
//! real questions / problems from the published datasets, kept small so
//! the crate can run out-of-the-box without an external dataset fetch.

/// MMLU sample — 20 items across diverse subjects.
pub const MMLU: &str = include_str!("mmlu.json");
/// MMLU-Pro sample — 10-way MCQ items.
pub const MMLU_PRO: &str = include_str!("mmlu_pro.json");
/// GPQA sample — graduate-level science questions.
pub const GPQA: &str = include_str!("gpqa.json");
/// HumanEval sample — Python coding problems with held-out tests.
pub const HUMANEVAL: &str = include_str!("humaneval.json");
/// BIG-Bench Hard sample — one item per task, 23 tasks.
pub const BBH: &str = include_str!("bbh.json");
/// IFEval sample — verifiable instruction-following constraints.
pub const IFEVAL: &str = include_str!("ifeval.json");
/// AlpacaEval sample — head-to-head reference comparisons.
pub const ALPACA_EVAL: &str = include_str!("alpaca_eval.json");
/// AGIEval sample — academic / professional standardised tests.
pub const AGI_EVAL: &str = include_str!("agi_eval.json");
/// HellaSwag sample — commonsense sentence completion.
pub const HELLASWAG: &str = include_str!("hellaswag.json");
/// ARC sample — Easy + Challenge mixed.
pub const ARC: &str = include_str!("arc.json");
/// TruthfulQA sample — adversarial questions with MC1/MC2 options.
pub const TRUTHFULQA: &str = include_str!("truthfulqa.json");
/// BoolQ sample — yes/no reading comprehension.
pub const BOOLQ: &str = include_str!("boolq.json");
/// GSM8K sample — grade-school math word problems.
pub const GSM8K: &str = include_str!("gsm8k.json");
/// MATH sample — competition mathematics.
pub const MATH: &str = include_str!("math.json");
/// Winogrande sample — pronoun resolution.
pub const WINOGRANDE: &str = include_str!("winogrande.json");
