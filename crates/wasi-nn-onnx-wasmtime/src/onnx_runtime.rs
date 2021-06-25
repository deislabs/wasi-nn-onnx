use crate::{
    bytes::{bytes_to_f32_vec, f32_vec_to_bytes},
    witx::{
        types::{
            BufferSize, ExecutionTarget, Graph, GraphBuilderArray, GraphEncoding,
            GraphExecutionContext, Tensor, TensorType,
        },
        wasi_ephemeral_nn::WasiEphemeralNn,
    },
    WasiNnError, WasiNnResult as Result,
};
use ndarray::{Array, Dim, IxDynImpl, ShapeError};
use onnxruntime::{
    environment::Environment, session::OwnedSession, tensor::OrtOwnedTensor,
    GraphOptimizationLevel, LoggingLevel, OrtError, TensorElementDataType,
};
use std::{
    borrow::BorrowMut,
    collections::{btree_map::Keys, BTreeMap},
    fmt::Debug,
    sync::{Arc, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

/// Main context struct for which we implement the WasiEphemeralNn trait.
#[derive(Default)]
pub struct WasiNnOnnxCtx {
    pub state: Arc<RwLock<State>>,
}

/// Struct to hold the non-instantiated models and execution sessions.
#[derive(Default)]
pub struct State {
    pub executions: BTreeMap<GraphExecutionContext, OnnxSession>,
    pub models: BTreeMap<Graph, Vec<u8>>,
}

/// ONNX session that owns its environment and
/// the input and output tensors.
#[derive(Debug)]
pub struct OnnxSession {
    pub session: OwnedSession,

    // Currently, the input and output tensors are hardcoded to
    // F32. However, this should not always be the case.
    // But because each input or output tensor can have a *different*
    // tensor type, introducing a struct-level generic type
    // parameter does not solve the issue.
    // The most straightforward way would be to (re)introduce
    // trait constraints, as shown below:

    // where
    // TIn: TypeToTensorElementDataType + Debug + Clone,
    // TOut: TypeToTensorElementDataType + Debug + Clone,
    // D: ndarray::Dimension,

    // The main goal here is to be able to constrain the
    // input and output tensor types, but without introducing
    // the constraint to State and WasiNnCtx.
    //
    // A workaround for this could be to store all inputs and
    // outputs as Vec<u8>, and only transform them
    // to their semantically relevant types just-in-time (for
    // inputs that is when compute is called, for outputs
    // when get_output is called).
    // This has some obvious performance downsides, but would
    // significantly simplify the session struct without
    // sacrificing the input and output type options.
    pub input_tensors: Option<Vec<Array<f32, Dim<IxDynImpl>>>>,
    pub output_tensors: Option<Vec<Array<f32, Dim<IxDynImpl>>>>,
}

impl OnnxSession {
    pub fn with_session(session: OwnedSession) -> Result<Self> {
        Ok(Self {
            session,
            input_tensors: None,
            output_tensors: None,
        })
    }
}

impl State {
    /// Helper function that returns the key that is supposed to be inserted next.
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

impl WasiEphemeralNn for WasiNnOnnxCtx {
    fn load(
        &mut self,
        builder: &GraphBuilderArray,
        encoding: GraphEncoding,
        target: ExecutionTarget,
    ) -> Result<Graph> {
        log::info!("load: encoding: {:#?}, target: {:#?}", encoding, target);

        if encoding != GraphEncoding::Onnx {
            log::error!("load current implementation can only load ONNX models");
            return Err(WasiNnError::InvalidEncodingError);
        }

        let model_bytes = builder.as_ptr().read()?.as_slice()?.to_vec();
        let mut state = self.state.write()?;
        let graph = state.key(state.models.keys());
        log::info!(
            "load: inserting graph: {:#?} with size {:#?}",
            graph,
            model_bytes.len()
        );
        state.models.insert(graph, model_bytes);

        log::info!("load: current number of models: {:#?}", state.models.len());

        Ok(graph)
    }

    fn init_execution_context(&mut self, graph: Graph) -> Result<GraphExecutionContext> {
        log::info!("init_execution_context: graph: {:#?}", graph);

        let mut state = self.state.write()?;
        let model_bytes = match state.models.get(&graph) {
            Some(mb) => mb,
            None => {
                log::error!(
                    "init_execution_context: cannot find model in state with graph {:#?}",
                    graph
                );
                return Err(WasiNnError::RuntimeError);
            }
        };

        let environment = Environment::builder()
            .with_log_level(LoggingLevel::Verbose)
            .build()?;
        let session = environment
            .new_owned_session_builder()?
            .with_optimization_level(GraphOptimizationLevel::All)?
            .with_model_from_memory(model_bytes)?;
        let session = OnnxSession::with_session(session)?;
        let gec = state.key(state.executions.keys());
        log::info!(
            "init_execution_context: inserting graph execution context: {:#?}",
            gec
        );

        state.executions.insert(gec, session);

        Ok(gec)
    }

    // If there are multiple input tensors, the guest
    // should call this function in order, as this actually
    // constructs the final input tensor used for the inference.
    // If we wanted to avoid this, we could create an intermediary
    // HashMap<u32, Array<TIn, D>> and collapse it into a Vec<Array<TIn, D>>
    // when performing the inference.
    fn set_input(
        &mut self,
        context: GraphExecutionContext,
        index: u32,
        tensor: &Tensor,
    ) -> Result<()> {
        let mut state = self.state.write()?;
        let mut execution = match state.executions.get_mut(&context) {
            Some(s) => s,
            None => {
                log::error!(
                    "set_input: cannot find session in state with context {:#?}",
                    context
                );

                return Err(WasiNnError::RuntimeError);
            }
        };

        let input = tensor
            .dimensions
            .as_slice()?
            .iter()
            .map(|d| *d as usize)
            .collect::<Vec<_>>();

        let input_type: TensorElementDataType = tensor.type_.into();

        log::info!(
            "set_input: input dimensions: {:#?}, input type: {:#?}",
            input,
            input_type
        );

        // TODO
        // Some models only set partial dimensions here, so checking the input and
        // expected dimensions are equal will have to take that into account.
        if let Some(exp) = execution.session.inputs.get(index as usize) {
            log::info!(
                "set_input: model expects input tensor dimensions: {:#?}",
                exp
            );
        };

        // for simplicity, the input and output tensors supported for now are F32 only.
        // In the future, we want to match here and coerce the input data to the correct
        // data type -- however, this requires the OnnxSession struct's input and output tensors
        // to be generic over the data types as well, not only over the dimention.
        match tensor.type_ {
            TensorType::F16 | TensorType::I32 | TensorType::U8 => {
                log::error!("set_input: only F32 inputs and outputs are supported for now");
                return Err(WasiNnError::RuntimeError);
            }
            _ => {}
        };
        // TODO
        // match on the tensor data type and coerce input to right type
        let data = bytes_to_f32_vec(tensor.data.as_slice()?.to_vec())?;
        let input = Array::from_shape_vec(input, data)?;

        match execution.input_tensors {
            Some(ref mut input_tensors) => {
                input_tensors.push(input);
                log::info!(
                    "set_input: input tensors now contains {} items",
                    input_tensors.len(),
                );
            }
            None => {
                execution.input_tensors = Some(vec![input]);
            }
        };

        Ok(())
    }

    fn get_output(
        &mut self,
        context: GraphExecutionContext,
        index: u32,
        out_buffer: &wiggle::GuestPtr<'_, u8>,
        out_buffer_max_size: BufferSize,
    ) -> Result<BufferSize> {
        let mut state = self.state.write()?;
        let execution = match state.executions.get_mut(&context) {
            Some(s) => s,
            None => {
                log::error!(
                    "get_output: cannot find session in state with context {:#?}",
                    context
                );

                return Err(WasiNnError::RuntimeError);
            }
        };

        let output_tensors = match execution.output_tensors {
            Some(ref oa) => oa,
            None => {
                log::error!("get_output: output_tensors for session is none. Perhaps you haven't called compute yet?");
                return Err(WasiNnError::RuntimeError);
            }
        };

        let tensor = match output_tensors.get(index as usize) {
            Some(a) => a,
            None => {
                log::error!(
                    "get_output: output_tensors does not contain index {}",
                    index
                );
                return Err(WasiNnError::RuntimeError);
            }
        };

        let bytes = f32_vec_to_bytes(tensor.as_slice().unwrap().to_vec());
        let size = bytes.len();
        let mut out_slice = out_buffer.as_array(out_buffer_max_size).as_slice_mut()?;
        (&mut out_slice[..size]).copy_from_slice(&bytes);

        Ok(size as BufferSize)
    }

    fn compute(&mut self, context: GraphExecutionContext) -> Result<()> {
        let mut state = self.state.write()?;
        let mut execution = match state.executions.get_mut(&context) {
            Some(s) => s,
            None => {
                log::error!(
                    "compute: cannot find session in state with context {:#?}",
                    context
                );

                return Err(WasiNnError::RuntimeError);
            }
        };
        let input_tensors = execution
            .input_tensors
            .borrow_mut()
            .as_ref()
            .unwrap_or(&vec![])
            .clone();

        log::info!(
            "compute: input tensors contains {} elements",
            input_tensors.len()
        );

        let output_tensors: Vec<OrtOwnedTensor<'_, '_, f32, Dim<IxDynImpl>>> =
            execution.session.run(input_tensors)?;

        log::info!(
            "compute: output tensors contains {} elements",
            output_tensors.len()
        );

        log::info!(
            "compute: dimensions of first output tensor: {:#?}",
            output_tensors.get(0).unwrap().dim()
        );

        match execution.output_tensors {
            Some(_) => {
                log::error!("compute: existing data in output_tensors, aborting");
                return Err(WasiNnError::RuntimeError);
            }
            None => {
                execution.output_tensors = Some(output_vec_from_tensors(output_tensors)?);
            }
        };
        Ok(())
    }
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

impl From<std::io::Error> for WasiNnError {
    fn from(_: std::io::Error) -> Self {
        WasiNnError::RuntimeError
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

fn output_vec_from_tensors(
    tensors: Vec<OrtOwnedTensor<'_, '_, f32, Dim<IxDynImpl>>>,
) -> Result<Vec<Array<f32, Dim<IxDynImpl>>>> {
    let mut res = Vec::new();
    for (_, tensor) in tensors.into_iter().enumerate() {
        res.push(Array::from_shape_vec(
            tensor.shape(),
            tensor.as_slice().unwrap().to_vec(),
        )?);
    }

    Ok(res)
}
