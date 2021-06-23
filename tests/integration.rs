use wasi_common::WasiCtx;
use wasi_nn_onnx_wasmtime::WasiNnCtx;

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::Ctx;
    use anyhow::{Context, Error};
    use wasi_cap_std_sync::Dir;
    use wasi_nn_onnx_wasmtime::WasiNnCtx;
    use wasmtime::*;
    use wasmtime_wasi::sync::WasiCtxBuilder;

    #[test]
    fn rust_tests() {
        init();
        run_tests(
            "target/wasm32-wasi/release/wasi-nn-rust.wasm",
            vec![
                "load_empty",
                "load_model",
                "init_execution_context",
                "set_input",
                "compute",
                "get_output",
                "batch",
            ],
        )
        .unwrap();
    }

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    fn run_tests<S: Into<String> + AsRef<Path> + Clone>(
        filename: S,
        funcs: Vec<S>,
    ) -> Result<(), Error> {
        let (instance, mut store) = create_instance(filename.clone())?;
        for f in funcs {
            log::info!(
                "executing {} for module {}",
                f.clone().into(),
                filename.clone().into()
            );
            let func = instance
                .get_func(&mut store, f.into().as_str())
                .unwrap_or_else(|| panic!("cannot find function"));
            func.call(&mut store, &[])?;
        }

        Ok(())
    }

    fn create_instance<S: Into<String> + AsRef<Path>>(
        filename: S,
    ) -> Result<(Instance, Store<Ctx>), Error> {
        let engine = Engine::default();
        let mut store = Store::new(&engine, Ctx::default());
        let mut linker = Linker::new(&engine);
        linker.allow_unknown_exports(true);

        populate_with_wasi(&mut store, &mut linker, vec!["tests/testdata"])?;

        let module = Module::from_file(linker.engine(), filename)?;
        let instance = linker.instantiate(&mut store, &module)?;

        Ok((instance, store))
    }

    fn populate_with_wasi(
        store: &mut Store<Ctx>,
        linker: &mut Linker<Ctx>,
        dirs: Vec<&str>,
    ) -> Result<(), Error> {
        wasmtime_wasi::add_to_linker(linker, |host| host.wasi_ctx.as_mut().unwrap())?;

        let mut builder = WasiCtxBuilder::new()
            .inherit_stdin()
            .inherit_stdout()
            .inherit_stderr();

        for dir in dirs.iter() {
            builder = builder.preopened_dir(
                unsafe { Dir::open_ambient_dir(dir) }
                    .with_context(|| format!("failed to open directory '{}'", dir))?,
                dir,
            )?;
        }

        store.data_mut().wasi_ctx = Some(builder.build());

        wasi_nn_onnx_wasmtime::add_to_linker(linker, |host| host.nn_ctx.as_mut().unwrap())?;
        store.data_mut().nn_ctx = Some(WasiNnCtx::default());

        Ok(())
    }
}

#[derive(Default)]
struct Ctx {
    pub wasi_ctx: Option<WasiCtx>,
    pub nn_ctx: Option<WasiNnCtx>,
}
