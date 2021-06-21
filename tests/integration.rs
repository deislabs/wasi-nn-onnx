use wasi_common::WasiCtx;
use wasi_nn_onnx_wasmtime::WasiNnCtx;

#[cfg(test)]
mod tests {
    use crate::Ctx;
    use anyhow::Error;
    use wasi_nn_onnx_wasmtime::ctx::WasiNnCtx;
    use wasmtime::*;
    use wasmtime_wasi::sync::WasiCtxBuilder;

    #[test]
    fn test_rust_load() {
        init();

        run_func(
            "target/wasm32-wasi/release/rust.wasm".to_string(),
            "_start".to_string(),
        )
        .unwrap();
    }

    fn init() {
        env_logger::init();
    }

    fn run_func(filename: String, name: String) -> Result<(), Error> {
        let engine = Engine::default();
        let mut store = Store::new(&engine, Ctx::default());
        let mut linker = Linker::new(&engine);
        linker.allow_unknown_exports(true);

        populate_with_wasi(&mut store, &mut linker)?;

        let module = Module::from_file(linker.engine(), filename)?;

        let instance = linker.instantiate(&mut store, &module)?;
        let func = instance
            .get_func(&mut store, name.as_str())
            .unwrap_or_else(|| panic!("cannot find _start"));
        func.call(&mut store, &[])?;
        Ok(())
    }

    fn populate_with_wasi(store: &mut Store<Ctx>, linker: &mut Linker<Ctx>) -> Result<(), Error> {
        wasmtime_wasi::add_to_linker(linker, |host| host.wasi_ctx.as_mut().unwrap())?;

        store.data_mut().wasi_ctx = Some(
            WasiCtxBuilder::new()
                .inherit_stdin()
                .inherit_stdout()
                .inherit_stderr()
                .build(),
        );

        wasi_nn_onnx_wasmtime::add_to_linker(linker, |host| host.nn_ctx.as_mut().unwrap())?;
        store.data_mut().nn_ctx = Some(WasiNnCtx::new()?);

        Ok(())
    }
}

#[derive(Default)]
struct Ctx {
    pub wasi_ctx: Option<WasiCtx>,
    pub nn_ctx: Option<WasiNnCtx>,
}
