use std::{
    collections::{btree_map::Keys, BTreeMap},
    io::Cursor,
    sync::{Arc, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use crate::{
    bytes::{bytes_to_f32_vec, f32_vec_to_bytes},
    witx::{
        types::{
            BufferSize, ExecutionTarget, Graph, GraphBuilderArray, GraphEncoding,
            GraphExecutionContext, NnErrno, Tensor, UserErrorConversion,
        },
        wasi_ephemeral_nn::WasiEphemeralNn,
    },
    WasiNnError, WasiNnResult as Result,
};

use ndarray::Array;
use tract_onnx::prelude::Tensor as TractTensor;
use tract_onnx::prelude::*;
use tract_onnx::{prelude::Graph as TractGraph, tract_hir::infer::InferenceOp};

/// Main context struct for which we implement the WasiEphemeralNn trait.
#[derive(Default)]
pub struct WasiNnTractCtx {
    pub state: Arc<RwLock<State>>,
}

#[derive(Default)]
pub struct State {
    pub executions: BTreeMap<GraphExecutionContext, TractSession>,
    pub models: BTreeMap<Graph, Vec<u8>>,
}

#[derive(Debug)]
pub struct TractSession {
    pub graph: TractGraph<InferenceFact, Box<dyn InferenceOp>>,
    pub input_tensors: Option<Vec<TractTensor>>,
    pub output_tensors: Option<Vec<Arc<TractTensor>>>,
}

impl TractSession {
    pub fn with_graph(graph: TractGraph<InferenceFact, Box<dyn InferenceOp>>) -> Self {
        Self {
            graph,
            input_tensors: None,
            output_tensors: None,
        }
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

impl WasiEphemeralNn for WasiNnTractCtx {
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
        let mut model_bytes = match state.models.get(&graph) {
            Some(mb) => Cursor::new(mb),
            None => {
                log::error!(
                    "init_execution_context: cannot find model in state with graph {:#?}",
                    graph
                );
                return Err(WasiNnError::RuntimeError);
            }
        };

        let model = tract_onnx::onnx().model_for_read(&mut model_bytes).unwrap();

        let gec = state.key(state.executions.keys());
        log::info!(
            "init_execution_context: inserting graph execution context: {:#?}",
            gec
        );

        state
            .executions
            .insert(gec, TractSession::with_graph(model));

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
        let execution = match state.executions.get_mut(&context) {
            Some(s) => s,
            None => {
                log::error!(
                    "set_input: cannot find session in state with context {:#?}",
                    context
                );

                return Err(WasiNnError::RuntimeError);
            }
        };

        let shape = tensor
            .dimensions
            .as_slice()?
            .iter()
            .map(|d| *d as usize)
            .collect::<Vec<_>>();

        execution.graph.set_input_fact(
            index as usize,
            InferenceFact::dt_shape(f32::datum_type(), shape.clone()),
        )?;

        let data = bytes_to_f32_vec(tensor.data.as_slice()?.to_vec())?;
        let input: TractTensor = Array::from_shape_vec(shape, data)?.into();

        match execution.input_tensors {
            Some(ref mut input_arrays) => {
                input_arrays.push(input);
                log::info!(
                    "set_input: input arrays now contains {} items",
                    input_arrays.len(),
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
        let state = self.state.read()?;
        let execution = match state.executions.get(&context) {
            Some(s) => s,
            None => {
                log::error!(
                    "compute: cannot find session in state with context {:#?}",
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

        // TODO
        //
        // There are two `.clone()` calls here that could prove
        // to be *very* inneficient, one in getting the input tensors,
        // the other in making the model runnable.
        let input_tensors: Vec<TractTensor> = execution
            .input_tensors
            .as_ref()
            .unwrap_or(&vec![])
            .clone()
            .into_iter()
            .collect();

        log::info!(
            "compute: input tensors contains {} elements",
            input_tensors.len()
        );

        // Some ONNX models don't specify their input tensor
        // shapes completely, so we can only call `.into_optimized()` after we
        // have set the input tensor shapes.
        let output_tensors = execution
            .graph
            .clone()
            .into_optimized()?
            .into_runnable()?
            .run(input_tensors.into())?;

        log::info!(
            "compute: output tensors contains {} elements",
            output_tensors.len()
        );
        match execution.output_tensors {
            Some(_) => {
                log::error!("compute: existing data in output_tensors, aborting");
                return Err(WasiNnError::RuntimeError);
            }
            None => {
                execution.output_tensors = Some(output_tensors.into_iter().collect());
            }
        };

        Ok(())
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

impl From<wiggle::anyhow::Error> for WasiNnError {
    fn from(_: wiggle::anyhow::Error) -> Self {
        WasiNnError::RuntimeError
    }
}

impl<'a> UserErrorConversion for WasiNnTractCtx {
    fn nn_errno_from_wasi_nn_error(
        &mut self,
        e: WasiNnError,
    ) -> std::result::Result<NnErrno, wiggle::Trap> {
        eprintln!("Host error: {:?}", e);
        match e {
            WasiNnError::GuestError(_) => unimplemented!(),
            WasiNnError::RuntimeError => unimplemented!(),
            WasiNnError::OnnxError => unimplemented!(),
            WasiNnError::InvalidEncodingError => unimplemented!(),
        }
    }
}
