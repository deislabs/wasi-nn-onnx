use crate::witx::types::{Graph, GraphExecutionContext, TensorType};
use onnxruntime::{
    ndarray::{Array, Dimension, ShapeError},
    session::OwnedSession,
    OrtError, TensorElementDataType, TypeToTensorElementDataType,
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

impl<'a, TIn, TOut, D> From<PoisonError<std::sync::RwLockReadGuard<'_, State<TIn, TOut, D>>>>
    for WasiNnError
where
    TIn: TypeToTensorElementDataType + Debug + Clone,
    TOut: TypeToTensorElementDataType + Debug + Clone,
    D: Dimension,
{
    fn from(_: PoisonError<RwLockReadGuard<'_, State<TIn, TOut, D>>>) -> Self {
        WasiNnError::RuntimeError
    }
}

impl<'a, TIn, TOut, D> From<PoisonError<RwLockWriteGuard<'_, State<TIn, TOut, D>>>> for WasiNnError
where
    TIn: TypeToTensorElementDataType + Debug + Clone,
    TOut: TypeToTensorElementDataType + Debug + Clone,
    D: Dimension,
{
    fn from(_: PoisonError<RwLockWriteGuard<'_, State<TIn, TOut, D>>>) -> Self {
        WasiNnError::RuntimeError
    }
}

impl<'a, TIn, TOut, D> From<PoisonError<&mut State<TIn, TOut, D>>> for WasiNnError
where
    TIn: TypeToTensorElementDataType + Debug + Clone,
    TOut: TypeToTensorElementDataType + Debug + Clone,
    D: Dimension,
{
    fn from(_: PoisonError<&mut State<TIn, TOut, D>>) -> Self {
        WasiNnError::RuntimeError
    }
}

impl From<ShapeError> for WasiNnError {
    fn from(_: ShapeError) -> Self {
        WasiNnError::RuntimeError
    }
}

pub(crate) type WasiNnResult<T> = Result<T, WasiNnError>;

pub struct WasiNnCtx<TIn, TOut, D>
where
    TIn: TypeToTensorElementDataType + Debug + Clone,
    TOut: TypeToTensorElementDataType + Debug + Clone,
    D: Dimension,
{
    pub state: Arc<RwLock<State<TIn, TOut, D>>>,
}

pub struct State<TIn, TOut, D>
where
    TIn: TypeToTensorElementDataType + Debug + Clone,
    TOut: TypeToTensorElementDataType + Debug + Clone,
    D: Dimension,
{
    pub executions: BTreeMap<GraphExecutionContext, OnnxSession<TIn, TOut, D>>,
    pub models: BTreeMap<Graph, Vec<u8>>,
}

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

impl<TIn, TOut, D> State<TIn, TOut, D>
where
    TIn: TypeToTensorElementDataType + Debug + Clone,
    TOut: TypeToTensorElementDataType + Debug + Clone,
    D: Dimension,
{
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
