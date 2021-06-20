use crate::witx::types::{Graph, GraphExecutionContext};
use onnxruntime::{
    ndarray::{Array, Dimension},
    session::OwnedSession,
    OrtError, TypeToTensorElementDataType,
};
use std::{
    collections::{btree_map::Keys, BTreeMap},
    fmt::Debug,
    sync::{Arc, PoisonError, RwLock},
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
    pub sessions: BTreeMap<GraphExecutionContext, OwnedSession>,
    pub models: BTreeMap<Graph, Vec<u8>>,
}

// TODO
//
// The actual session we keep track of should contain the inputs and
// outputs as well.
// There should be a way to introduce this without carrying around all
// the generic types in State and WasiNnCtx.
#[derive(Debug)]
pub struct OnnxSession<TIn, TOut, D>
where
    TIn: TypeToTensorElementDataType + Debug + Clone,
    TOut: TypeToTensorElementDataType + Debug + Clone,
    D: Dimension,
{
    pub session: OwnedSession,
    pub input_arrays: Option<Vec<Array<TIn, D>>>,
    pub output_arrays: Option<Vec<Array<TOut, D>>>,
}

impl<TIn, TOut, D> OnnxSession<TIn, TOut, D>
where
    TIn: TypeToTensorElementDataType + Debug + Clone,
    TOut: TypeToTensorElementDataType + Debug + Clone,
    D: Dimension,
{
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
