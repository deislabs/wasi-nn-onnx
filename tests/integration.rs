#[cfg(test)]
mod tests {
    use anyhow::{Context, Error};
    use std::path::Path;
    use wasi_cap_std_sync::{ambient_authority, Dir};
    use wasi_common::WasiCtx;
    use wasi_nn_onnx_wasmtime::{WasiNnOnnxCtx, WasiNnTractCtx};
    use wasmtime::*;
    use wasmtime_wasi::sync::WasiCtxBuilder;

    #[derive(Default)]
    struct Ctx {
        pub wasi_ctx: Option<WasiCtx>,
        pub nn_ctx: Option<WasiNnOnnxCtx>,
        pub tract_ctx: Option<WasiNnTractCtx>,
    }

    enum Runtime {
        C,
        Tract,
    }

    const RUST_WASM_TEST: &str = "tests/rust/target/wasm32-wasi/release/wasi-nn-rust.wasm";
    const INFERENCE_TESTS: [&str; 10] = [
        "load_empty",
        "load_model",
        "init_execution_context",
        "set_input",
        "compute",
        "test_squeezenet",
        "batch_squeezenet",
        "test_mobilenetv2",
        "batch_mobilenetv2",
        "infernece_identity_model",
    ];

    #[test]
    fn test_c_api() {
        init();
        run_tests(RUST_WASM_TEST, INFERENCE_TESTS.to_vec(), Runtime::C).unwrap();
    }

    #[test]
    fn test_tract() {
        init();
        run_tests(RUST_WASM_TEST, INFERENCE_TESTS.to_vec(), Runtime::Tract).unwrap();
    }

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    fn run_tests<S: Into<String> + AsRef<Path> + Clone>(
        filename: S,
        funcs: Vec<S>,
        r: Runtime,
    ) -> Result<(), Error> {
        let (instance, mut store) = create_instance(filename.clone(), r)?;
        for f in funcs {
            log::info!(
                "executing {} for module {}",
                f.clone().into(),
                filename.clone().into()
            );
            let func = instance
                .get_func(&mut store, f.into().as_str())
                .unwrap_or_else(|| panic!("cannot find function"));
            func.call(&mut store, &[], &mut vec![])?;
        }

        Ok(())
    }

    fn create_instance<S: Into<String> + AsRef<Path>>(
        filename: S,
        r: Runtime,
    ) -> Result<(Instance, Store<Ctx>), Error> {
        let engine = Engine::default();
        let mut store = Store::new(&engine, Ctx::default());
        let mut linker = Linker::new(&engine);
        linker.allow_unknown_exports(true);

        populate_with_wasi(&mut store, &mut linker, vec!["tests/testdata"], r)?;

        let module = Module::from_file(linker.engine(), filename)?;
        let instance = linker.instantiate(&mut store, &module)?;

        Ok((instance, store))
    }

    fn populate_with_wasi(
        store: &mut Store<Ctx>,
        linker: &mut Linker<Ctx>,
        dirs: Vec<&str>,
        r: Runtime,
    ) -> Result<(), Error> {
        wasmtime_wasi::add_to_linker(linker, |host| host.wasi_ctx.as_mut().unwrap())?;

        let mut builder = WasiCtxBuilder::new()
            .inherit_stdin()
            .inherit_stdout()
            .inherit_stderr();

        for dir in dirs.iter() {
            builder = builder.preopened_dir(
                Dir::open_ambient_dir(dir, ambient_authority())
                    .with_context(|| format!("failed to open directory '{}'", dir))?,
                dir,
            )?;
        }

        store.data_mut().wasi_ctx = Some(builder.build());

        match r {
            Runtime::C => {
                wasi_nn_onnx_wasmtime::add_to_linker(linker, |host| host.nn_ctx.as_mut().unwrap())?;
                store.data_mut().nn_ctx = Some(WasiNnOnnxCtx::default());
            }
            Runtime::Tract => {
                wasi_nn_onnx_wasmtime::add_to_linker(linker, |host| {
                    host.tract_ctx.as_mut().unwrap()
                })?;
                store.data_mut().tract_ctx = Some(WasiNnTractCtx::default());
            }
        };

        Ok(())
    }
}
