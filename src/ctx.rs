use std::{
    collections::{hash_map::Keys, HashMap},
    sync::{Arc, PoisonError, RwLock},
};

use crate::witx::types::{Graph, GraphExecutionContext};
use onnxruntime::{session::OwnedSession, OrtError};
use thiserror::Error;
use wiggle::GuestError;

#[derive(Debug, Error)]
pub enum WasiNnError {
    #[error("guest error")]
    GuestError(#[from] GuestError),

    #[error("runtime error")]
    RuntimeError,

    #[error("ONNX error")]
    OnnxError,

    #[error("Invalid encoding")]
    InvalidEncodingError,
}

impl From<OrtError> for WasiNnError {
    fn from(_: OrtError) -> Self {
        WasiNnError::OnnxError
    }
}

impl From<PoisonError<std::sync::RwLockReadGuard<'_, State>>> for WasiNnError {
    fn from(_: PoisonError<std::sync::RwLockReadGuard<'_, State>>) -> Self {
        WasiNnError::RuntimeError
    }
}

impl From<PoisonError<std::sync::RwLockWriteGuard<'_, State>>> for WasiNnError {
    fn from(_: PoisonError<std::sync::RwLockWriteGuard<'_, State>>) -> Self {
        WasiNnError::RuntimeError
    }
}

impl From<PoisonError<&mut State>> for WasiNnError {
    fn from(_: PoisonError<&mut State>) -> Self {
        WasiNnError::RuntimeError
    }
}

pub(crate) type WasiNnResult<T> = Result<T, WasiNnError>;

pub struct WasiNnCtx {
    pub state: Arc<RwLock<State>>,
}

pub struct State {
    pub sessions: HashMap<GraphExecutionContext, OwnedSession>,
    pub models: HashMap<Graph, Vec<u8>>,
}

impl State {
    pub fn key<K: Into<u32> + From<u32> + Copy, V>(&self, keys: Keys<K, V>) -> K {
        match keys.last() {
            Some(&k) => {
                let last: u32 = k.into();
                K::from(last + 1)
            }
            None => K::from(0),
        }
    }
}
