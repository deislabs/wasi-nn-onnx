use std::time::Instant;

use anyhow::{bail, Context, Error};
use structopt::StructOpt;
use wasi_cap_std_sync::WasiCtxBuilder;
use wasi_nn_onnx_wasmtime::ctx::WasiNnCtx;
use wasmtime::{AsContextMut, Config, Engine, Func, Instance, Linker, Module, Store, Val, ValType};
use wasmtime_wasi::*;

#[derive(Debug, StructOpt)]
#[structopt(name = "wasmtime-onnx")]
struct Opt {
    #[structopt(help = "The path of the WebAssembly module to run")]
    module: String,

    #[structopt(
        short = "i",
        long = "invoke",
        default_value = "_start",
        help = "The name of the function to run"
    )]
    invoke: String,

    #[structopt(short = "c", long = "cache", help = "Path to cache config file")]
    cache: Option<String>,

    #[structopt(
        short = "e",
        long = "env",
        value_name = "NAME=VAL",
        parse(try_from_str = parse_env_var),
        help = "Pass an environment variable to the program"
    )]
    vars: Vec<(String, String)>,

    /// Grant access to the given host directory
    #[structopt(long = "dir", number_of_values = 1, value_name = "DIRECTORY")]
    dirs: Vec<String>,

    /// Grant access to a guest directory mapped as a host directory
    #[structopt(long = "mapdir", number_of_values = 1, value_name = "GUEST_DIR::HOST_DIR", parse(try_from_str = parse_map_dirs))]
    map_dirs: Vec<(String, String)>,

    #[structopt(value_name = "ARGS", help = "The arguments to pass to the module")]
    module_args: Vec<String>,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Error> {
    env_logger::init();

    let opt = Opt::from_args();
    let method = opt.invoke.clone();

    let dirs = compute_preopen_dirs(opt.dirs, opt.map_dirs)?;

    let start = Instant::now();

    let (instance, mut store) = create_instance(opt.module, opt.vars, dirs, opt.cache)?;
    let func = instance
        .get_func(&mut store, method.as_str())
        .unwrap_or_else(|| panic!("cannot find function {}", method));

    invoke_func(func, opt.module_args, &mut store)?;
    let duration = start.elapsed();
    log::info!("execution time: {:#?}", duration);
    Ok(())
}

fn create_instance(
    filename: String,
    vars: Vec<(String, String)>,
    preopen_dirs: Vec<(String, Dir)>,
    cache_config: Option<String>,
) -> Result<(Instance, Store<Ctx>), Error> {
    let mut config = Config::default();
    if let Some(c) = cache_config {
        if let Ok(p) = std::fs::canonicalize(c) {
            config.cache_config_load(p)?;
        };
    };
    let engine = Engine::new(&config)?;
    let mut store = Store::new(&engine, Ctx::default());
    let mut linker = Linker::new(&engine);
    linker.allow_unknown_exports(true);

    populate_with_wasi(&mut store, &mut linker, vars, preopen_dirs)?;

    let module = Module::from_file(linker.engine(), filename)?;
    let instance = linker.instantiate(&mut store, &module)?;

    Ok((instance, store))
}

fn populate_with_wasi(
    store: &mut Store<Ctx>,
    linker: &mut Linker<Ctx>,
    vars: Vec<(String, String)>,
    preopen_dirs: Vec<(String, Dir)>,
) -> Result<(), Error> {
    wasmtime_wasi::add_to_linker(linker, |host| host.wasi_ctx.as_mut().unwrap())?;

    let mut builder = WasiCtxBuilder::new()
        .inherit_stdin()
        .inherit_stdout()
        .inherit_stderr()
        .envs(&vars)?;

    for (name, dir) in preopen_dirs.into_iter() {
        builder = builder.preopened_dir(dir, name)?;
    }
    store.data_mut().wasi_ctx = Some(builder.build());

    wasi_nn_onnx_wasmtime::add_to_linker(linker, |host| host.nn_ctx.as_mut().unwrap())?;
    store.data_mut().nn_ctx = Some(WasiNnCtx::new()?);

    Ok(())
}

fn compute_preopen_dirs(
    dirs: Vec<String>,
    map_dirs: Vec<(String, String)>,
) -> Result<Vec<(String, Dir)>, Error> {
    let mut preopen_dirs = Vec::new();

    for dir in dirs.iter() {
        preopen_dirs.push((
            dir.clone(),
            unsafe { Dir::open_ambient_dir(dir) }
                .with_context(|| format!("failed to open directory '{}'", dir))?,
        ));
    }

    for (guest, host) in map_dirs.iter() {
        preopen_dirs.push((
            guest.clone(),
            unsafe { Dir::open_ambient_dir(host) }
                .with_context(|| format!("failed to open directory '{}'", host))?,
        ));
    }

    Ok(preopen_dirs)
}

// Invoke function given module arguments and print results.
// Adapted from https://github.com/bytecodealliance/wasmtime/blob/main/src/commands/run.rs.
fn invoke_func(func: Func, args: Vec<String>, mut store: impl AsContextMut) -> Result<(), Error> {
    let ty = func.ty(&mut store);

    let mut args = args.iter();
    let mut values = Vec::new();
    for ty in ty.params() {
        let val = match args.next() {
            Some(s) => s,
            None => {
                bail!("not enough arguments for invocation")
            }
        };
        values.push(match ty {
            ValType::I32 => Val::I32(val.parse()?),
            ValType::I64 => Val::I64(val.parse()?),
            ValType::F32 => Val::F32(val.parse()?),
            ValType::F64 => Val::F64(val.parse()?),
            t => bail!("unsupported argument type {:?}", t),
        });
    }

    let results = func.call(&mut store, &values)?;
    for result in results.into_vec() {
        match result {
            Val::I32(i) => println!("{}", i),
            Val::I64(i) => println!("{}", i),
            Val::F32(f) => println!("{}", f32::from_bits(f)),
            Val::F64(f) => println!("{}", f64::from_bits(f)),
            Val::ExternRef(_) => println!("<externref>"),
            Val::FuncRef(_) => println!("<funcref>"),
            Val::V128(i) => println!("{}", i),
        };
    }

    Ok(())
}

fn parse_env_var(s: &str) -> Result<(String, String), Error> {
    let parts: Vec<_> = s.splitn(2, '=').collect();
    if parts.len() != 2 {
        bail!("must be of the form `key=value`");
    }
    Ok((parts[0].to_owned(), parts[1].to_owned()))
}

fn parse_map_dirs(s: &str) -> Result<(String, String), Error> {
    let parts: Vec<&str> = s.split("::").collect();
    if parts.len() != 2 {
        bail!("must contain exactly one double colon ('::')");
    }
    Ok((parts[0].into(), parts[1].into()))
}

#[derive(Default)]
struct Ctx {
    pub wasi_ctx: Option<WasiCtx>,
    pub nn_ctx: Option<WasiNnCtx>,
}
