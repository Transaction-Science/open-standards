//! # jouleclaw-ssm-reader — L1.5 SSM reader tier
//!
//! Local reading-comprehension QA over retrieved passages. Sits between
//! the L1.375 structural-contrast tier and the L2 federation tier in the
//! JouleClaw cascade. Class-typical cost: ~20,000 µJ, ~10 ms.
//!
//! Higher-energy than the L0.75 router (~100 µJ) because the tier does
//! full reading over passages, not just intent classification. Lower-
//! energy than an L3 frontier model because the SSM is Mamba-3 class
//! (≤ ~1B params, int8) and reads only a handful of short passages.
//!
//! ## Architecture
//!
//! The donor (`verity-cascade::layers::l15_ssm_reader`) calls into a real
//! SSM engine (Mamba-3 / Liquid) when one is loaded. JouleClaw is the
//! open-standard layer: it carries a [`ReadingComprehender`] trait so
//! production deployments can plug in any reader backend, and a
//! deterministic [`ExtractiveReader`] default for v0.1.
//!
//! The default reader is *deterministic* (same passages + question →
//! same answer, always) and *zero-energy* (pure CPU token-overlap). It
//! picks the sentence in the passage set with the largest question-token
//! overlap and returns it verbatim. Suitable for conformance vectors and
//! tests; production should swap in a real SSM.
//!
//! ## Query envelope
//!
//! The L1.5 tier consumes a structured envelope, not a raw text query.
//! Callers wrap the question and the retrieved passages in JSON via
//! [`QueryInput::Structured`]:
//!
//! ```json
//! {
//!   "question": "What is the capital of France?",
//!   "passages": [
//!     { "text": "Paris is the capital of France.", "source": "wp:Paris" },
//!     { "text": "France is in western Europe." }
//!   ]
//! }
//! ```
//!
//! Pure text queries — without a `passages` array — are inapplicable for
//! this tier; `estimate_cost` returns `None` and the runtime moves on.
//!
//! ## Wiring
//!
//! ```ignore
//! use jouleclaw_cascade::tier::{Cascade, Runtime};
//! use jouleclaw_ssm_reader::SsmReaderTier;
//!
//! let mut cascade = Cascade::new();
//! cascade.register(Box::new(SsmReaderTier::new()));
//! let mut rt = Runtime::new_without_l0(cascade);
//! ```
//!
//! Plug in a custom reader:
//!
//! ```ignore
//! struct MyMambaReader { /* … */ }
//! impl jouleclaw_ssm_reader::ReadingComprehender for MyMambaReader { /* … */ }
//!
//! let tier = SsmReaderTier::with_reader(Box::new(MyMambaReader { /* … */ }));
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod reader;
mod tier;

pub use reader::{ExtractiveReader, Passage, Reading, ReaderError, ReadingComprehender};
pub use tier::{
    SSM_READER_CONFIDENCE_FLOOR, SSM_READER_JOULES, SSM_READER_LATENCY, SsmReaderError,
    SsmReaderTier, parse_envelope,
};
