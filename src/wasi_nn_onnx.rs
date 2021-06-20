use crate::{
    ctx::{WasiNnCtx, WasiNnError, WasiNnResult as Result},
    witx::{
        types::{
            BufferSize, ExecutionTarget, Graph, GraphBuilderArray, GraphEncoding,
            GraphExecutionContext, Tensor, TensorType,
        },
        wasi_ephemeral_nn::WasiEphemeralNn,
    },
};
use onnxruntime::{environment::Environment, GraphOptimizationLevel, LoggingLevel};

impl WasiEphemeralNn for WasiNnCtx {
    fn load(
        &mut self,
        builder: &GraphBuilderArray,
        encoding: GraphEncoding,
        target: ExecutionTarget,
    ) -> Result<Graph> {
        log::info!(
            "wasi_nn_onnx::load: encoding: {:#?}, target: {:#?}",
            encoding,
            target
        );

        if encoding != GraphEncoding::Onnx {
            log::error!("wasi_nn_onnx::load current implementation can only load ONNX models");
            return Err(WasiNnError::InvalidEncodingError);
        }

        let model_bytes = builder.as_ptr().read()?.as_slice()?.to_vec();
        let mut state = self.state.write()?;
        let graph = state.key(state.models.keys());
        log::info!(
            "wasi_nn_onnx::load: inserting graph: {:#?} with size {:#?}",
            graph,
            model_bytes.len()
        );
        state.models.insert(graph, model_bytes);

        Ok(graph)
    }

    fn init_execution_context(&mut self, graph: Graph) -> Result<GraphExecutionContext> {
        log::info!("wasi_nn_onnx::init_execution_context: graph: {:#?}", graph);

        let mut state = self.state.write()?;
        let model_bytes = match state.models.get(&graph) {
            Some(mb) => mb,
            None => {
                log::error!("wasi_nn_onnx::init_execution_context: cannot find model in state with graph {:#?}", graph);
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
        let gec = state.key(state.sessions.keys());
        log::info!(
            "wasi_nn_onnx::init_execution_context: inserting graph execution context with session: {:#?} with size {:#?}",
            gec,
            session
        );

        state.sessions.insert(gec, session);

        Ok(gec)
    }

    fn set_input(
        &mut self,
        context: GraphExecutionContext,
        index: u32,
        tensor: &Tensor,
    ) -> Result<()> {
        let mut state = self.state.write()?;
        let mut session = match state.sessions.get_mut(&context) {
            Some(s) => s,
            None => {
                log::error!(
                    "wasi_nn_onnx::set_input: cannot find session in state with context {:#?}",
                    context
                );

                return Err(WasiNnError::RuntimeError);
            }
        };

        log::info!("wasi_nn_onnx::set_input: session: {:#?}", session);

        let expected = session
            .inputs
            .get(index as usize)
            .unwrap()
            .dimensions()
            .map(|d| d.unwrap())
            .collect::<Vec<usize>>();

        let input = tensor
            .dimensions
            .as_slice()?
            .iter()
            .map(|d| *d as usize)
            .collect::<Vec<_>>();

        log::info!(
            "wasi_nn_onnx::set_input: expected dimensions: {:#?}, input dimensions: {:#?}",
            expected,
            input
        );

        // TODO
        // assume f32 for now?

        todo!()
    }

    fn get_output(
        &mut self,
        _context: GraphExecutionContext,
        _index: u32,
        _out_buffer: &wiggle::GuestPtr<'_, u8>,
        _out_buffer_max_size: BufferSize,
    ) -> Result<BufferSize> {
        todo!()
    }

    fn compute(&mut self, _context: GraphExecutionContext) -> Result<()> {
        todo!()
    }
}
