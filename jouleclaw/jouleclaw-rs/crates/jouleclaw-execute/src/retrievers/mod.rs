//! Concrete retriever implementations (spec §5.3).
//!
//! Each module here ships a single [`crate::retriever::Retriever`]
//! impl with the methods named in its RAP. The orchestrator never
//! depends on which retrievers are in the registry — they're
//! injected via [`crate::retriever::RetrieverRegistry`].

pub mod fixture;
pub mod http_cache;
pub mod wikidata;
pub mod wikipedia;
