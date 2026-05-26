//! Error types for efficient-genai.

use alloc::string::String;
use core::fmt;

#[cfg(feature = "std")]
use std::backtrace::Backtrace;

/// Result type alias using our Error type.
pub type Result<T> = core::result::Result<T, Error>;

/// Error source information for chaining errors.
#[cfg(feature = "std")]
#[derive(Debug)]
pub struct ErrorSource {
    /// The underlying error
    pub error: Box<dyn std::error::Error + Send + Sync>,
}

/// Main error type for the library.
#[derive(Debug)]
pub enum Error {
    /// Memory allocation failed
    OutOfMemory {
        /// Requested allocation size in bytes
        requested: usize,
        /// Available memory in bytes
        available: usize,
        /// Optional source error
        #[cfg(feature = "std")]
        source: Option<ErrorSource>,
    },

    /// Shape mismatch in operation
    ShapeMismatch {
        /// Expected shape description
        expected: String,
        /// Actual shape description
        got: String,
    },

    /// Data type mismatch
    DTypeMismatch {
        /// Expected dtype
        expected: String,
        /// Actual dtype
        got: String,
    },

    /// Device not available or not supported
    DeviceNotAvailable {
        /// Device identifier
        device: String,
        /// Reason for unavailability
        reason: String,
        /// Optional source error
        #[cfg(feature = "std")]
        source: Option<ErrorSource>,
    },

    /// Kernel compilation failed
    KernelCompilation {
        /// Kernel name
        kernel: String,
        /// Compilation error message
        message: String,
    },

    /// Runtime execution error
    Execution {
        /// Operation that failed
        operation: String,
        /// Error details
        message: String,
        /// Optional source error
        #[cfg(feature = "std")]
        source: Option<ErrorSource>,
    },

    /// Invalid argument
    InvalidArgument {
        /// Argument name
        name: String,
        /// Error message
        message: String,
    },

    /// I/O error
    Io {
        /// Operation description
        operation: String,
        /// Error message
        message: String,
        /// Optional source error
        #[cfg(feature = "std")]
        source: Option<ErrorSource>,
    },

    /// Model loading error
    ModelLoad {
        /// Model path or identifier
        model: String,
        /// Error message
        message: String,
        /// Optional source error
        #[cfg(feature = "std")]
        source: Option<ErrorSource>,
    },

    /// Unsupported operation
    Unsupported {
        /// Feature or operation name
        feature: String,
    },

    /// Internal error (should not happen)
    Internal {
        /// Error message
        message: String,
        /// Backtrace captured when error was created
        #[cfg(feature = "std")]
        backtrace: Option<Backtrace>,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfMemory { requested, available, .. } => {
                write!(
                    f,
                    "out of memory: requested {} bytes, available {} bytes",
                    requested, available
                )
            }
            Self::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {expected}, got {got}")
            }
            Self::DTypeMismatch { expected, got } => {
                write!(f, "dtype mismatch: expected {expected}, got {got}")
            }
            Self::DeviceNotAvailable { device, reason, .. } => {
                write!(f, "device '{device}' not available: {reason}")
            }
            Self::KernelCompilation { kernel, message } => {
                write!(f, "kernel '{kernel}' compilation failed: {message}")
            }
            Self::Execution { operation, message, .. } => {
                write!(f, "execution of '{operation}' failed: {message}")
            }
            Self::InvalidArgument { name, message } => {
                write!(f, "invalid argument '{name}': {message}")
            }
            Self::Io { operation, message, .. } => {
                write!(f, "I/O error in '{operation}': {message}")
            }
            Self::ModelLoad { model, message, .. } => {
                write!(f, "failed to load model '{model}': {message}")
            }
            Self::Unsupported { feature } => {
                write!(f, "unsupported: {feature}")
            }
            Self::Internal { message, .. } => {
                write!(f, "internal error: {message}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::OutOfMemory { source, .. } => {
                source.as_ref().map(|s| s.error.as_ref() as &(dyn std::error::Error + 'static))
            }
            Self::DeviceNotAvailable { source, .. } => {
                source.as_ref().map(|s| s.error.as_ref() as &(dyn std::error::Error + 'static))
            }
            Self::Execution { source, .. } => {
                source.as_ref().map(|s| s.error.as_ref() as &(dyn std::error::Error + 'static))
            }
            Self::Io { source, .. } => {
                source.as_ref().map(|s| s.error.as_ref() as &(dyn std::error::Error + 'static))
            }
            Self::ModelLoad { source, .. } => {
                source.as_ref().map(|s| s.error.as_ref() as &(dyn std::error::Error + 'static))
            }
            _ => None,
        }
    }
}

impl Error {
    /// Create an out of memory error.
    #[inline]
    pub fn out_of_memory(requested: usize, available: usize) -> Self {
        Self::OutOfMemory {
            requested,
            available,
            #[cfg(feature = "std")]
            source: None,
        }
    }

    /// Create a shape mismatch error.
    #[inline]
    pub fn shape_mismatch(expected: impl Into<String>, got: impl Into<String>) -> Self {
        Self::ShapeMismatch {
            expected: expected.into(),
            got: got.into(),
        }
    }

    /// Create a device not available error.
    #[inline]
    pub fn device_not_available(device: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::DeviceNotAvailable {
            device: device.into(),
            reason: reason.into(),
            #[cfg(feature = "std")]
            source: None,
        }
    }

    /// Create an execution error.
    #[inline]
    pub fn execution(operation: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Execution {
            operation: operation.into(),
            message: message.into(),
            #[cfg(feature = "std")]
            source: None,
        }
    }

    /// Create an unsupported error.
    #[inline]
    pub fn unsupported(feature: impl Into<String>) -> Self {
        Self::Unsupported {
            feature: feature.into(),
        }
    }

    /// Create an internal error with automatic backtrace capture.
    #[inline]
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
            #[cfg(feature = "std")]
            backtrace: Some(Backtrace::capture()),
        }
    }

    /// Create an invalid argument error.
    #[inline]
    pub fn invalid_argument(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self::InvalidArgument {
            name: name.into(),
            message: message.into(),
        }
    }

    /// Create an invalid input error (shorthand for invalid_argument with name="input").
    #[inline]
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidArgument {
            name: "input".into(),
            message: message.into(),
        }
    }

    /// Create an I/O error.
    #[inline]
    pub fn io(operation: impl Into<String>, message: impl Into<String>) -> Self {
        Self::Io {
            operation: operation.into(),
            message: message.into(),
            #[cfg(feature = "std")]
            source: None,
        }
    }

    /// Create an I/O error with a source error.
    #[cfg(feature = "std")]
    #[inline]
    pub fn io_with_source(
        operation: impl Into<String>,
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Io {
            operation: operation.into(),
            message: message.into(),
            source: Some(ErrorSource {
                error: Box::new(source),
            }),
        }
    }

    /// Create a model load error.
    #[inline]
    pub fn model_load(model: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ModelLoad {
            model: model.into(),
            message: message.into(),
            #[cfg(feature = "std")]
            source: None,
        }
    }

    /// Create a model load error with a source error.
    #[cfg(feature = "std")]
    #[inline]
    pub fn model_load_with_source(
        model: impl Into<String>,
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::ModelLoad {
            model: model.into(),
            message: message.into(),
            source: Some(ErrorSource {
                error: Box::new(source),
            }),
        }
    }

    /// Create a dtype mismatch error.
    #[inline]
    pub fn dtype_mismatch(expected: impl Into<String>, got: impl Into<String>) -> Self {
        Self::DTypeMismatch {
            expected: expected.into(),
            got: got.into(),
        }
    }

    /// Create a kernel compilation error.
    #[inline]
    pub fn kernel_compilation(kernel: impl Into<String>, message: impl Into<String>) -> Self {
        Self::KernelCompilation {
            kernel: kernel.into(),
            message: message.into(),
        }
    }

    /// Get the backtrace if available (for Internal errors).
    #[cfg(feature = "std")]
    pub fn backtrace(&self) -> Option<&Backtrace> {
        match self {
            Self::Internal { backtrace, .. } => backtrace.as_ref(),
            _ => None,
        }
    }
}
