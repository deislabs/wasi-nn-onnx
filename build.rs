use std::process::{self, Command};

const TESTS_DIR: &str = "tests";
const RUST_EXAMPLE: &str = "rust";
// const AS_EXAMPLE: &str = "as";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=tests/rust/src/main.rs");
    println!("cargo:rerun-if-changed=crates/wasi-nn-onnx-wasmtime/src/lib.rs");
    println!("cargo:rerun-if-changed=crates/wasi-nn-onnx-wasmtime/src/wasi_nn_onnx.rs");
    println!("cargo:rerun-if-changed=crates/wasi-nn-onnx-wasmtime/src/ctx.rs");
    println!("cargo:rerun-if-changed=crates/wasi-nn-onnx-wasmtime/src/witx.rs");

    println!("cargo:rustc-link-search=native={}", "./target");
    // println!("cargo:rustc-link-lib=dylib={}", "onnxruntime");
    println!("cargo:rustc-link-lib=onnxruntime");

    cargo_build_example(TESTS_DIR, RUST_EXAMPLE);
}
fn cargo_build_example(dir: &str, example: &str) {
    let dir = format!("{}/{}", dir, example);

    run(
        vec!["cargo", "build", "--target", "wasm32-wasi", "--release"],
        Some(dir),
    );
}

// fn as_build_example(dir: &str, example: &str) {
//     let dir = format!("{}/{}", dir, example);

//     run(vec!["npm", "install"], Some(dir.clone()));
//     run(vec!["npm", "run", "asbuild"], Some(dir));
// }

fn run<S: Into<String> + AsRef<std::ffi::OsStr>>(args: Vec<S>, dir: Option<String>) {
    let mut cmd = Command::new(get_os_process());
    cmd.stdout(process::Stdio::piped());
    cmd.stderr(process::Stdio::piped());

    if let Some(dir) = dir {
        cmd.current_dir(dir);
    };

    cmd.arg("-c");
    cmd.arg(
        args.into_iter()
            .map(Into::into)
            .collect::<Vec<String>>()
            .join(" "),
    );

    println!("running {:#?}", cmd);

    cmd.output().unwrap();
}

fn get_os_process() -> String {
    if cfg!(target_os = "windows") {
        String::from("powershell.exe")
    } else {
        String::from("/bin/bash")
    }
}
