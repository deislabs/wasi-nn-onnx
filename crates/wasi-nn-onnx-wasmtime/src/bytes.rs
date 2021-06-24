use crate::WasiNnResult as Result;
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::Cursor;

pub fn bytes_to_f32_vec(data: Vec<u8>) -> Result<Vec<f32>> {
    let chunks: Vec<&[u8]> = data.chunks(4).collect();
    let v: Vec<Result<f32>> = chunks
        .into_iter()
        .map(|c| {
            let mut rdr = Cursor::new(c);
            Ok(rdr.read_f32::<LittleEndian>()?)
        })
        .collect();

    v.into_iter().collect()
}

pub fn f32_vec_to_bytes(data: Vec<f32>) -> Vec<u8> {
    let sum: f32 = data.iter().sum();
    log::info!(
        "f32_vec_to_bytes: flatten output tensor contains {} elements with sum {}",
        data.len(),
        sum
    );
    let chunks: Vec<[u8; 4]> = data.into_iter().map(|f| f.to_le_bytes()).collect();
    let result: Vec<u8> = chunks.iter().flatten().copied().collect();

    log::info!(
        "f32_vec_to_bytes: flatten byte output tensor contains {} elements",
        result.len()
    );
    result
}

#[test]
fn test_f32_bytes_array_and_back() {
    let case = vec![0.0_f32, 1.1, 2.2, 3.3];
    let bytes = f32_vec_to_bytes(case.clone());
    let res = bytes_to_f32_vec(bytes).unwrap();
    assert_eq!(case, res);
}

#[test]
fn test_bytes_array_to_f32_array() {
    let bytes = vec![0x00, 0x00, 0x48, 0x41, 0x00, 0x00, 0x48, 0x41];
    let res = bytes_to_f32_vec(bytes).unwrap();
    assert!((12.5 - res[0]).abs() < f32::EPSILON);
    assert!((12.5 - res[1]).abs() < f32::EPSILON);
}
