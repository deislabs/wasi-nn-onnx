pub mod onnx_runtime;
pub mod witx;

pub use onnx_runtime::WasiNnCtx;
pub use witx::wasi_ephemeral_nn::add_to_linker;

#[derive(Debug, thiserror::Error)]
pub enum WasiNnError {
    #[error("guest error")]
    GuestError(#[from] wiggle::GuestError),

    #[error("runtime error")]
    RuntimeError,

    #[error("ONNX error")]
    OnnxError,

    #[error("Invalid encoding")]
    InvalidEncodingError,
}

pub type WasiNnResult<T> = Result<T, WasiNnError>;
