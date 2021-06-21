use crate::witx::types::{Graph, GraphExecutionContext, TensorType};
use onnxruntime::{
    ndarray::{Array, Dim, IxDynImpl, ShapeError},
    session::OwnedSession,
    OrtError, TensorElementDataType,
};
use std::{
    collections::{btree_map::Keys, BTreeMap},
    fmt::Debug,
    sync::{Arc, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard},
};
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

impl<'a> From<PoisonError<std::sync::RwLockReadGuard<'_, State>>> for WasiNnError {
    fn from(_: PoisonError<RwLockReadGuard<'_, State>>) -> Self {
        WasiNnError::RuntimeError
    }
}

impl<'a> From<PoisonError<RwLockWriteGuard<'_, State>>> for WasiNnError {
    fn from(_: PoisonError<RwLockWriteGuard<'_, State>>) -> Self {
        WasiNnError::RuntimeError
    }
}

impl<'a> From<PoisonError<&mut State>> for WasiNnError {
    fn from(_: PoisonError<&mut State>) -> Self {
        WasiNnError::RuntimeError
    }
}

impl From<ShapeError> for WasiNnError {
    fn from(_: ShapeError) -> Self {
        WasiNnError::RuntimeError
    }
}

pub(crate) type WasiNnResult<T> = Result<T, WasiNnError>;

pub struct WasiNnCtx {
    pub state: Arc<RwLock<State>>,
}

pub struct State {
    pub executions: BTreeMap<GraphExecutionContext, OnnxSession>,
    pub models: BTreeMap<Graph, Vec<u8>>,
}

#[derive(Debug)]
pub struct OnnxSession {
    pub session: OwnedSession,
    pub input_arrays: Option<Vec<Array<f32, Dim<IxDynImpl>>>>,
    pub output_arrays: Option<Vec<Array<f32, Dim<IxDynImpl>>>>,
}

impl OnnxSession {
    pub fn with_session(session: OwnedSession) -> WasiNnResult<Self> {
        Ok(Self {
            session,
            input_arrays: None,
            output_arrays: None,
        })
    }
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

impl From<TensorType> for TensorElementDataType {
    fn from(tt: TensorType) -> Self {
        match tt {
            TensorType::F16 | TensorType::F32 => Self::Float,

            TensorType::U8 => Self::Uint8,
            TensorType::I32 => Self::Int32,
        }
    }
}
