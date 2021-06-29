use crate::WasiNnError;
use ndarray::ShapeError;
use std::cmp::Ordering;
use types::{Graph, GraphExecutionContext, NnErrno};
use wiggle::GuestErrorType;

wiggle::from_witx!({
    witx: ["$WASI_ROOT/phases/ephemeral/witx/wasi_ephemeral_nn.witx"],
    errors: { nn_errno => WasiNnError }
});

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
