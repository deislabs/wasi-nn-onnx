use anyhow::Error;
use ndarray::Array4;

const MODEL_PATH: &str = "tests/testdata/resnet18-v1-7.onnx";
const IMG_PATH: &str = "tests/testdata/husky.jpg";

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

    // assert_eq!(g, 0);
}

#[no_mangle]
fn load_model() {
    let model = std::fs::read(MODEL_PATH).unwrap();
    println!("integration::load_model: loaded {} bytes", model.len());
    let _ = unsafe {
        wasi_nn::load(
            &[&model],
            wasi_nn::GRAPH_ENCODING_ONNX,
            wasi_nn::EXECUTION_TARGET_CPU,
        )
        .unwrap()
    };

    // assert_eq!(g, 1);
}

#[no_mangle]
fn init_execution_context() {
    let model = std::fs::read(MODEL_PATH).unwrap();
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
    // assert_eq!(g, 2);

    let gec = unsafe { wasi_nn::init_execution_context(g) }.unwrap();
    println!(
        "integration::init_execution_context: received graph execution context {}",
        gec
    );

    // assert_eq!(gec, 0);
}

#[no_mangle]
fn set_input() {
    let model = std::fs::read(MODEL_PATH).unwrap();
    println!("integration::set_input: loaded {} bytes", model.len());
    let g = unsafe {
        wasi_nn::load(
            &[&model],
            wasi_nn::GRAPH_ENCODING_ONNX,
            wasi_nn::EXECUTION_TARGET_CPU,
        )
        .unwrap()
    };
    // assert_eq!(g, 3);

    let gec = unsafe { wasi_nn::init_execution_context(g) }.unwrap();
    println!(
        "integration::set_input: received graph execution context {}",
        gec
    );

    // assert_eq!(gec, 1);

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
    let model = std::fs::read(MODEL_PATH).unwrap();
    println!("integration::compute: loaded {} bytes", model.len());
    let g = unsafe {
        wasi_nn::load(
            &[&model],
            wasi_nn::GRAPH_ENCODING_ONNX,
            wasi_nn::EXECUTION_TARGET_CPU,
        )
        .unwrap()
    };
    // assert_eq!(g, 4);

    let gec = unsafe { wasi_nn::init_execution_context(g) }.unwrap();
    println!(
        "integration::compute: received graph execution context {}",
        gec
    );

    // assert_eq!(gec, 2);

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
fn get_output() {
    let model = std::fs::read(MODEL_PATH).unwrap();
    println!("integration::compute: loaded {} bytes", model.len());
    let g = unsafe {
        wasi_nn::load(
            &[&model],
            wasi_nn::GRAPH_ENCODING_ONNX,
            wasi_nn::EXECUTION_TARGET_CPU,
        )
        .unwrap()
    };
    // assert_eq!(g, 5);

    let gec = unsafe { wasi_nn::init_execution_context(g) }.unwrap();
    println!(
        "integration::compute: received graph execution context {}",
        gec
    );

    // assert_eq!(gec, 3);

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

    let mut output_buffer = vec![0f32; 1001];
    unsafe {
        wasi_nn::get_output(
            gec,
            0,
            &mut output_buffer[..] as *mut [f32] as *mut u8,
            std::convert::TryInto::try_into(output_buffer.len() * 4).unwrap(),
        )
        .unwrap();
    }

    let results = sort_results(&output_buffer);
    println!("Found results, sorted top 5: {:?}", &results[..5]);
}

// Sort the buffer of probabilities. The graph places the match probability for each class at the
// index for that class (e.g. the probability of class 42 is placed at buffer[42]). Here we convert
// to a wrapping InferenceResult and sort the results.
fn sort_results(buffer: &[f32]) -> Vec<InferenceResult> {
    let mut results: Vec<InferenceResult> = buffer
        .iter()
        .skip(1)
        .enumerate()
        .map(|(c, p)| InferenceResult(c, *p))
        .collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    results
}

fn image_to_tensor<S: Into<String> + AsRef<std::path::Path>>(
    path: S,
    height: u32,
    width: u32,
) -> Result<Vec<u8>, Error> {
    let image = image::load_from_memory(&std::fs::read(path)?)
        .unwrap()
        .to_rgb8();
    // The model was trained on 224 x 224 RGB images, so we are resizing the input image to this dimension.
    let resized = image::imageops::resize(
        &image,
        width,
        height,
        ::image::imageops::FilterType::Triangle,
    );
    let tensor = Array4::from_shape_fn((1, 224, 224, 3), |(_, y, x, c)| {
        resized[(x as _, y as _)][c] as f32 / 255.0
    });

    Ok(f32_vec_to_bytes(tensor.as_slice().unwrap().to_vec()))
}

fn f32_vec_to_bytes(data: Vec<f32>) -> Vec<u8> {
    let chunks: Vec<[u8; 4]> = data.into_iter().map(|f| f.to_le_bytes()).collect();
    let mut result: Vec<u8> = Vec::new();

    // TODO
    // simplify this to potentially a single map.
    for c in chunks {
        for u in c.iter() {
            result.push(*u);
        }
    }
    result
}

// Take the image located at 'path', open it, resize it to height x width, and then converts
// the pixel precision to FP32. The resulting BGR pixel vector is then returned.
// fn image_to_tensor<S: Into<String> + AsRef<std::path::Path>>(
//     path: S,
//     height: u32,
//     width: u32,
// ) -> Vec<u8> {
//     let pixels = Reader::open(path).unwrap().decode().unwrap();
//     let dyn_img: DynamicImage = pixels.resize_exact(width, height, image::imageops::Triangle);
//     let bgr_img = dyn_img.to_bgr8();
//     // Get an array of the pixel values
//     let raw_u8_arr: &[u8] = &bgr_img.as_raw()[..];
//     // Create an array to hold the f32 value of those pixels
//     let bytes_required = raw_u8_arr.len() * 4;
//     let mut u8_f32_arr: Vec<u8> = vec![0; bytes_required];

//     for i in 0..raw_u8_arr.len() {
//         // Read the number as a f32 and break it into u8 bytes
//         let u8_f32: f32 = raw_u8_arr[i] as f32;
//         let u8_bytes = u8_f32.to_ne_bytes();

//         for j in 0..4 {
//             u8_f32_arr[(i * 4) + j] = u8_bytes[j];
//         }
//     }
//     u8_f32_arr
// }

// A wrapper for class ID and match probabilities.
#[derive(Debug, PartialEq)]
struct InferenceResult(usize, f32);
