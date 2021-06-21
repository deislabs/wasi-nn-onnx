fn main() {}

#[no_mangle]
fn load_empty() {
    let _ = unsafe {
        wasi_nn::load(
            &[&[]],
            wasi_nn::GRAPH_ENCODING_ONNX,
            wasi_nn::EXECUTION_TARGET_CPU,
        )
        .unwrap()
    };
}
