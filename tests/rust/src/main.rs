use anyhow::Error;
use ndarray::Array;
use std::{
    fmt::Debug,
    io::{self, BufRead, BufReader},
    path::PathBuf,
};
use wasi_nn_rust::{
    assert_contains_class, bytes_to_f32_vec, f32_vec_to_bytes, image_to_tensor, NdArrayTensor,
};

const SQUEEZENET_PATH: &str = "tests/testdata/models/squeezenet1.1-7.onnx";
const MOBILENETV2_PATH: &str = "tests/testdata/models/mobilenetv2-7.onnx";
const LABELS_PATH: &str = "tests/testdata/models/squeezenet_labels.txt";
const IMG_PATH: &str = "tests/testdata/images/n04350905.jpg";
const IMG_DIR: &str = "tests/testdata/images/";

fn main() {}

#[no_mangle]
fn load_simple_tensor() {
    // the model is not important here, we just want to make sure
    // the input tensor is reconstructed properly on the runtime.
    let model = std::fs::read(SQUEEZENET_PATH).unwrap();

    let g = unsafe {
        wasi_nn::load(
            &[&model],
            wasi_nn::GRAPH_ENCODING_ONNX,
            wasi_nn::EXECUTION_TARGET_CPU,
        )
        .unwrap()
    };

    let gec = unsafe { wasi_nn::init_execution_context(g) }.unwrap();
    println!(
        "integration::load_simple_tensor: received graph execution context {}",
        gec
    );

    let tensor = Array::from(vec![0.0_f32, 1.0, 2.0]);
    let shape: Vec<u32> = tensor.shape().iter().map(|u| *u as u32).collect();
    println!(
        "integration::load_simple_tensor: sending simple tensor: {:#?}",
        tensor
    );

    let tensor = f32_vec_to_bytes(tensor.as_slice().unwrap().to_vec());
    let tensor = wasi_nn::Tensor {
        dimensions: &shape,
        r#type: wasi_nn::TENSOR_TYPE_F32,
        data: &tensor,
    };

    unsafe {
        wasi_nn::set_input(gec, 0, tensor).unwrap();
    }
}

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

#[no_mangle]
fn load_model() {
    let model = std::fs::read(SQUEEZENET_PATH).unwrap();
    println!("integration::load_model: loaded {} bytes", model.len());
    let _ = unsafe {
        wasi_nn::load(
            &[&model],
            wasi_nn::GRAPH_ENCODING_ONNX,
            wasi_nn::EXECUTION_TARGET_CPU,
        )
        .unwrap()
    };
}

#[no_mangle]
fn init_execution_context() {
    let model = std::fs::read(SQUEEZENET_PATH).unwrap();
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

    let gec = unsafe { wasi_nn::init_execution_context(g) }.unwrap();
    println!(
        "integration::init_execution_context: received graph execution context {}",
        gec
    );
}

#[no_mangle]
fn set_input() {
    let model = std::fs::read(SQUEEZENET_PATH).unwrap();
    println!("integration::set_input: loaded {} bytes", model.len());
    let g = unsafe {
        wasi_nn::load(
            &[&model],
            wasi_nn::GRAPH_ENCODING_ONNX,
            wasi_nn::EXECUTION_TARGET_CPU,
        )
        .unwrap()
    };

    let gec = unsafe { wasi_nn::init_execution_context(g) }.unwrap();
    println!(
        "integration::set_input: received graph execution context {}",
        gec
    );

    let tensor_data = image_to_tensor(IMG_PATH, 224, 224).unwrap();
    let tensor = wasi_nn::Tensor {
        dimensions: &[1, 3, 224, 224],
        r#type: wasi_nn::TENSOR_TYPE_F32,
        data: &tensor_data,
    };

    unsafe {
        wasi_nn::set_input(gec, 0, tensor).unwrap();
    }
}

#[no_mangle]
fn compute() {
    let model = std::fs::read(SQUEEZENET_PATH).unwrap();
    println!("integration::compute: loaded {} bytes", model.len());
    let g = unsafe {
        wasi_nn::load(
            &[&model],
            wasi_nn::GRAPH_ENCODING_ONNX,
            wasi_nn::EXECUTION_TARGET_CPU,
        )
        .unwrap()
    };

    let gec = unsafe { wasi_nn::init_execution_context(g) }.unwrap();
    println!(
        "integration::compute: received graph execution context {}",
        gec
    );

    let tensor_data = image_to_tensor(IMG_PATH, 224, 224).unwrap();
    let tensor = wasi_nn::Tensor {
        dimensions: &[1, 3, 224, 224],
        r#type: wasi_nn::TENSOR_TYPE_F32,
        data: &tensor_data,
    };

    unsafe {
        wasi_nn::set_input(gec, 0, tensor).unwrap();
    }

    unsafe {
        wasi_nn::compute(gec).unwrap();
    }
}

#[no_mangle]
fn test_squeezenet() {
    let model = std::fs::read(SQUEEZENET_PATH).unwrap();
    inference_image(model, IMG_PATH).unwrap();
}

#[no_mangle]
fn batch_squeezenet() {
    run_batch(SQUEEZENET_PATH, IMG_DIR).unwrap();
}

fn run_batch<S: Into<String> + AsRef<std::path::Path> + Copy>(
    model: S,
    dir: S,
) -> Result<(), Error> {
    let mut entries = std::fs::read_dir(dir)?
        .map(|res| res.map(|e| e.path()))
        .collect::<Result<Vec<_>, io::Error>>()?;
    entries.sort();

    let model = std::fs::read(model).unwrap();

    for path in entries.iter() {
        inference_image::<String>(model.clone(), path.to_string_lossy().to_string())?;
    }

    Ok(())
}

#[no_mangle]
fn test_mobilenetv2() {
    let model = std::fs::read(MOBILENETV2_PATH).unwrap();
    inference_image(model, IMG_PATH).unwrap();
}

#[no_mangle]
fn batch_mobilenetv2() {
    run_batch(MOBILENETV2_PATH, IMG_DIR).unwrap();
}

fn inference_image<S: Into<String> + AsRef<std::path::Path> + Clone + Debug>(
    model: Vec<u8>,
    img: S,
) -> Result<(), Error>
where
    PathBuf: From<S>,
{
    println!(
        "integration::inference_image: loaded module {}  with size {} bytes",
        SQUEEZENET_PATH,
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

    let gec = unsafe { wasi_nn::init_execution_context(g) }.unwrap();
    println!(
        "integration::inference_image: received graph execution context {}",
        gec
    );

    let tensor_data = image_to_tensor(img.clone(), 224, 224).unwrap();
    let tensor = wasi_nn::Tensor {
        dimensions: &[1, 3, 224, 224],
        r#type: wasi_nn::TENSOR_TYPE_F32,
        data: &tensor_data,
    };

    unsafe {
        wasi_nn::set_input(gec, 0, tensor).unwrap();
    }

    unsafe {
        wasi_nn::compute(gec).unwrap();
    }

    let mut buffer = vec![0u8; 4000];
    unsafe {
        wasi_nn::get_output(gec, 0, buffer.as_mut_ptr(), buffer.len() as u32).unwrap();
    }

    let output_f32 = bytes_to_f32_vec(buffer);

    let output_tensor = Array::from_shape_vec((1, 1000, 1, 1), output_f32).unwrap();
    let mut probabilities: Vec<(usize, f32)> = output_tensor
        .softmax(ndarray::Axis(1))
        .into_iter()
        .enumerate()
        .collect::<Vec<_>>();
    probabilities.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let labels = BufReader::new(std::fs::File::open(LABELS_PATH).unwrap());

    let mut actual: Vec<String> = Vec::new();
    let labels: Vec<String> = labels.lines().map(|line| line.unwrap()).collect();
    println!("integration::inference_image: results for image {:#?}", img);
    for i in 0..5 {
        let c = labels[probabilities[i].0].clone();
        actual.push(c);
        println!(
            "class={} ({}); probability={}",
            labels[probabilities[i].0], probabilities[i].0, probabilities[i].1
        );
    }

    assert_contains_class(img.into(), actual);

    println!("\n");

    Ok(())
}
