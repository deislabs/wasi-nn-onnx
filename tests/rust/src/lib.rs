use anyhow::Error;
use byteorder::{LittleEndian, ReadBytesExt};
use image::Pixel;
use ndarray::{s, Array, ArrayBase};
use std::{fmt::Debug, io::Cursor};

pub trait NdArrayTensor<S, T, D> {
    /// https://en.wikipedia.org/wiki/Softmax_function)
    fn softmax(&self, axis: ndarray::Axis) -> Array<T, D>
    where
        D: ndarray::RemoveAxis,
        S: ndarray::RawData + ndarray::Data + ndarray::RawData<Elem = T>,
        <S as ndarray::RawData>::Elem: std::clone::Clone,
        T: ndarray::NdFloat + std::ops::SubAssign + std::ops::DivAssign;
}

impl<S, T, D> NdArrayTensor<S, T, D> for ArrayBase<S, D>
where
    D: ndarray::RemoveAxis,
    S: ndarray::RawData + ndarray::Data + ndarray::RawData<Elem = T>,
    <S as ndarray::RawData>::Elem: std::clone::Clone,
    T: ndarray::NdFloat + std::ops::SubAssign + std::ops::DivAssign,
{
    fn softmax(&self, axis: ndarray::Axis) -> Array<T, D> {
        let mut new_array: Array<T, D> = self.to_owned();
        new_array.map_inplace(|v| *v = v.exp());
        let sum = new_array.sum_axis(axis).insert_axis(axis);
        new_array /= &sum;

        new_array
    }
}

pub fn f32_vec_to_bytes(data: Vec<f32>) -> Vec<u8> {
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

pub fn bytes_to_f32_vec(data: Vec<u8>) -> Vec<f32> {
    let chunks: Vec<&[u8]> = data.chunks(4).collect();
    let result: Vec<f32> = chunks
        .into_iter()
        .map(|c| {
            let mut rdr = Cursor::new(c);
            rdr.read_f32::<LittleEndian>().unwrap()
        })
        .collect();
    let sum: f32 = result.iter().sum();
    println!(
        "wasi_nn_onnx: bytes_to_f32_vec: flattened output tensor contains {} elements with sum {}",
        result.len(),
        sum
    );

    result
}

pub fn image_to_tensor<S: Into<String> + AsRef<std::path::Path> + Debug>(
    path: S,
    height: u32,
    width: u32,
) -> Result<Vec<u8>, Error> {
    println!("trying to load image {:#?}", path);
    let image = image::imageops::resize(
        &image::open(path)?,
        width,
        height,
        ::image::imageops::FilterType::Triangle,
    );

    println!("resized image: {:#?}", image.dimensions());

    let mut array = ndarray::Array::from_shape_fn((1, 3, 224, 224), |(_, c, j, i)| {
        let pixel = image.get_pixel(i as u32, j as u32);
        let channels = pixel.channels();

        // range [0, 255] -> range [0, 1]
        (channels[c] as f32) / 255.0
    });

    // Normalize channels to mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];
    for c in 0..3 {
        let mut channel_array = array.slice_mut(s![0, c, .., ..]);
        channel_array -= mean[c];
        channel_array /= std[c];
    }

    Ok(f32_vec_to_bytes(array.as_slice().unwrap().to_vec()))
}

pub fn assert_inferred_class(img: String, exp: &String) {
    let class = std::path::PathBuf::from(img);
    let class = class
        .file_name()
        .unwrap()
        .to_string_lossy()
        .split('.')
        .collect::<Vec<&str>>()[0]
        .to_string();

    let exp = exp.split(' ').collect::<Vec<&str>>()[0].to_string();

    assert_eq!(class, exp);
}

#[test]
fn test_class_from_path() {
    assert_inferred_class(
        "tests/testdata/images/n04350905.jpg".to_string(),
        &"n04350905 suit, suit of clothes".to_string(),
    );
}
