use crate::ctx::{WasiNnCtx, WasiNnResult as Result};
use crate::witx::types;
use crate::witx::wasi_ephemeral_nn::WasiEphemeralNn;
use types::{
    ExecutionTarget, Graph, GraphBuilderArray, GraphEncoding, GraphExecutionContext, Tensor,
};
use wiggle::GuestPtr;

impl<'a> WasiEphemeralNn for WasiNnCtx {
    fn load<'b>(
        &mut self,
        builder: &GraphBuilderArray<'_>,
        encoding: GraphEncoding,
        target: ExecutionTarget,
    ) -> Result<Graph> {
        todo!()
    }

    fn init_execution_context(&mut self, graph: Graph) -> Result<GraphExecutionContext> {
        todo!()
    }

    fn set_input<'b>(
        &mut self,
        context: GraphExecutionContext,
        index: u32,
        tensor: &Tensor<'b>,
    ) -> Result<()> {
        todo!()
    }

    fn get_output<'b>(
        &mut self,
        context: GraphExecutionContext,
        index: u32,
        out_buffer: &GuestPtr<'_, u8>,
        out_buffer_max_size: u32,
    ) -> Result<u32> {
        todo!()
    }

    fn compute(&mut self, context: GraphExecutionContext) -> Result<()> {
        todo!()
    }
}
