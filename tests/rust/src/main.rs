fn main() {}

#[no_mangle]
fn load_empty() {
    let g = unsafe {
        wasi_nn::load(
            &[&[]],
            wasi_nn::GRAPH_ENCODING_ONNX,
            wasi_nn::EXECUTION_TARGET_CPU,
        )
        .unwrap()
    };

    assert_eq!(g, 0);
}

#[no_mangle]
fn load_model() {
    let model = std::fs::read("tests/testdata/squeezenet1.0-8.onnx").unwrap();
    println!("integration::load_model: loaded {} bytes", model.len());
    let g = unsafe {
        wasi_nn::load(
            &[&model],
            wasi_nn::GRAPH_ENCODING_ONNX,
            wasi_nn::EXECUTION_TARGET_CPU,
        )
        .unwrap()
    };

    assert_eq!(g, 1);
}

#[no_mangle]
fn init_execution_context() {
    let model = std::fs::read("tests/testdata/squeezenet1.0-8.onnx").unwrap();
    println!(
        "integration::init_execution_context: loaded {} bytes",
        model.len()
    );
    let g = unsafe {
        wasi_nn::load(
            &[&model],
            wasi_nn::GRAPH_ENCODING_ONNX,
            wasi_nn::EXECUTION_TARGET_CPU,
        )
        .unwrap()
    };
    assert_eq!(g, 2);

    let gec = unsafe { wasi_nn::init_execution_context(g) }.unwrap();
    println!(
        "integration::init_execution_context: received graph execution context {}",
        gec
    );

    assert_eq!(gec, 0);
}
