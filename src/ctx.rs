use thiserror::Error;
use wiggle::GuestError;

#[derive(Debug, Error)]
pub enum WasiNnError {
    #[error("guest error")]
    GuestError(#[from] GuestError),
}

pub(crate) type WasiNnResult<T> = Result<T, WasiNnError>;

pub struct WasiNnCtx {}
