pub mod bytes;
#[cfg(feature = "c_onnxruntime")]
pub mod onnx_runtime;

pub mod tract;
pub mod witx;

#[cfg(feature = "c_onnxruntime")]
pub use onnx_runtime::WasiNnOnnxCtx;
pub use tract::WasiNnTractCtx;
pub use witx::wasi_ephemeral_nn::add_to_linker;

pub type WasiNnResult<T> = Result<T, WasiNnError>;

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
