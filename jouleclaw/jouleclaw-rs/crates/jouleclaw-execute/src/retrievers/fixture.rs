//! In-memory programmable retriever for tests and offline acceptance
//! runs. The orchestrator never inspects the retriever type — a
//! `FixtureRetriever` slots in wherever a real backend would.

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;

use jouleclaw_schema::{RetrievedItem, SubQuery};

use crate::retriever::{Retriever, RetrieverError};

type MethodResult = Result<Vec<RetrievedItem>, RetrieverError>;
type MethodFn = Box<dyn Fn() -> MethodResult + Send + Sync>;

pub struct FixtureRetriever {
    id: String,
    methods: Mutex<HashMap<String, MethodFn>>,
}

fn clone_err(e: &RetrieverError) -> RetrieverError {
    match e {
        RetrieverError::UnknownMethod(s) => RetrieverError::UnknownMethod(s.clone()),
        RetrieverError::Backend(s) => RetrieverError::Backend(s.clone()),
        RetrieverError::ParseFailed(s) => RetrieverError::ParseFailed(s.clone()),
        RetrieverError::Refused(s) => RetrieverError::Refused(s.clone()),
    }
}

impl FixtureRetriever {
    pub fn new(id: &str) -> Self {
        Self {
            id: id.into(),
            methods: Mutex::new(HashMap::new()),
        }
    }

    /// Register a method that returns the same value every call.
    pub fn set_method(&self, method: &str, result: MethodResult) {
        let stored = result;
        let f: MethodFn = Box::new(move || match &stored {
            Ok(items) => Ok(items.clone()),
            Err(e) => Err(clone_err(e)),
        });
        self.methods.lock().unwrap().insert(method.into(), f);
    }

    /// Register a method that runs a custom closure each call.
    pub fn set_method_fn<F>(&self, method: &str, f: F)
    where
        F: Fn() -> MethodResult + Send + Sync + 'static,
    {
        self.methods
            .lock()
            .unwrap()
            .insert(method.into(), Box::new(f));
    }
}

#[async_trait]
impl Retriever for FixtureRetriever {
    fn retriever_id(&self) -> &str {
        &self.id
    }

    async fn call(
        &self,
        method: &str,
        _subquery: &SubQuery,
        _parameters: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<Vec<RetrievedItem>, RetrieverError> {
        // Invoke the closure inside the lock so we never carry the
        // guard across an await. The closures are synchronous by
        // contract — see `set_method_fn`.
        let guard = self.methods.lock().unwrap();
        match guard.get(method) {
            Some(f) => f(),
            None => Err(RetrieverError::UnknownMethod(method.into())),
        }
    }
}
