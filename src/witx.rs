use crate::ctx::{WasiNnCtx, WasiNnError};
use types::{NnErrno, UserErrorConversion};
use wiggle::GuestErrorType;

wiggle::from_witx!({
    witx: ["$WASI_ROOT/phases/ephemeral/witx/wasi_ephemeral_nn.witx"],
    errors: { nn_errno => WasiNnError }
});

impl<'a> UserErrorConversion for WasiNnCtx {
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
