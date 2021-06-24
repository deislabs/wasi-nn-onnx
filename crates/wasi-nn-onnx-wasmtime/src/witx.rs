use crate::onnx_runtime::WasiNnOnnxCtx;
use crate::tract::WasiNnTractCtx;
use crate::WasiNnError;
use std::cmp::Ordering;
use types::{Graph, GraphExecutionContext, NnErrno, UserErrorConversion};
use wiggle::GuestErrorType;

wiggle::from_witx!({
    witx: ["$WASI_ROOT/phases/ephemeral/witx/wasi_ephemeral_nn.witx"],
    errors: { nn_errno => WasiNnError }
});

impl<'a> UserErrorConversion for WasiNnTractCtx {
    fn nn_errno_from_wasi_nn_error(&mut self, e: WasiNnError) -> Result<NnErrno, wiggle::Trap> {
        eprintln!("Host error: {:?}", e);
        match e {
            WasiNnError::GuestError(_) => unimplemented!(),
            WasiNnError::RuntimeError => unimplemented!(),
            WasiNnError::OnnxError => unimplemented!(),
            WasiNnError::InvalidEncodingError => unimplemented!(),
        }
    }
}

impl<'a> UserErrorConversion for WasiNnOnnxCtx {
    fn nn_errno_from_wasi_nn_error(&mut self, e: WasiNnError) -> Result<NnErrno, wiggle::Trap> {
        eprintln!("Host error: {:?}", e);
        match e {
            WasiNnError::GuestError(_) => unimplemented!(),
            WasiNnError::RuntimeError => unimplemented!(),
            WasiNnError::OnnxError => unimplemented!(),
            WasiNnError::InvalidEncodingError => unimplemented!(),
        }
    }
}

impl GuestErrorType for NnErrno {
    fn success() -> Self {
        Self::Success
    }
}

impl Ord for Graph {
    fn cmp(&self, other: &Self) -> Ordering {
        let (s, o) = (*self, *other);
        let s: u32 = s.into();
        let o: u32 = o.into();
        s.cmp(&o)
    }
}

impl PartialOrd for Graph {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GraphExecutionContext {
    fn cmp(&self, other: &Self) -> Ordering {
        let (s, o) = (*self, *other);
        let s: u32 = s.into();
        let o: u32 = o.into();
        s.cmp(&o)
    }
}

impl PartialOrd for GraphExecutionContext {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
