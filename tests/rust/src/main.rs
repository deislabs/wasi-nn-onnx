use wasi_nn;

fn main() {
    let _ = unsafe {
        wasi_nn::load(
            &[&[]],
            wasi_nn::GRAPH_ENCODING_ONNX,
            wasi_nn::EXECUTION_TARGET_CPU,
        )
        .unwrap()
    };
}
