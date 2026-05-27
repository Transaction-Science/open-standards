//! Error types for jouleclaw-program.

use thiserror::Error;

/// Crate-level result alias.
pub type Result<T> = core::result::Result<T, Error>;

/// All errors surfaced by jouleclaw-program.
///
/// Errors fall into four families:
///
/// 1. **Signature** — declared signatures are malformed (duplicate field names,
///    empty input/output lists, …).
/// 2. **Program** — the wiring graph references modules or fields that don't
///    exist, or contains cycles.
/// 3. **Compile** — the compiler couldn't lower a program (unbound input,
///    missing edge, conflicting types).
/// 4. **Runtime** — execution failed (backend error, type mismatch in record,
///    code-runner error, …).
#[derive(Debug, Error)]
pub enum Error {
    // --- signature ---
    #[error("signature `{name}` has no input fields")]
    SignatureMissingInputs { name: String },

    #[error("signature `{name}` has no output fields")]
    SignatureMissingOutputs { name: String },

    #[error("signature `{name}` has duplicate field `{field}`")]
    DuplicateField { name: String, field: String },

    #[error("field `{field}` has an empty name")]
    EmptyFieldName { field: String },

    // --- program ---
    #[error("program has duplicate module name `{name}`")]
    DuplicateModule { name: String },

    #[error("program edge references unknown module `{module}`")]
    UnknownModule { module: String },

    #[error("program edge references unknown field `{field}` on module `{module}`")]
    UnknownField { module: String, field: String },

    #[error(
        "program edge wires output field `{from_module}.{from_field}` ({from_ty}) into input \
         field `{to_module}.{to_field}` ({to_ty}) — types do not match"
    )]
    EdgeTypeMismatch {
        from_module: String,
        from_field: String,
        from_ty: String,
        to_module: String,
        to_field: String,
        to_ty: String,
    },

    #[error("program graph contains a cycle including module `{module}`")]
    ProgramCycle { module: String },

    // --- compile ---
    #[error(
        "module `{module}` input field `{field}` has no incoming edge and is not declared as a \
         program input"
    )]
    UnboundInput { module: String, field: String },

    #[error("program has no modules")]
    EmptyProgram,

    // --- runtime ---
    #[error("backend error in module `{module}`: {message}")]
    Backend { module: String, message: String },

    #[error("code runner error in module `{module}`: {message}")]
    CodeRunner { module: String, message: String },

    #[error("module `{module}` produced no value for output field `{field}`")]
    MissingOutput { module: String, field: String },

    #[error(
        "module `{module}` produced value for field `{field}` with wrong type: expected \
         {expected}, got {actual}"
    )]
    TypeMismatch {
        module: String,
        field: String,
        expected: String,
        actual: String,
    },

    #[error("missing program input `{field}`")]
    MissingInput { field: String },
}
