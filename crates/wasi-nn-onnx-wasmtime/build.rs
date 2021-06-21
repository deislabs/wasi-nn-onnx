fn main() {
    let cwd = std::env::current_dir().unwrap();
    let wasi_root = cwd.join("../../spec");
    println!("cargo:rustc-env=WASI_ROOT={}", wasi_root.display());
}
