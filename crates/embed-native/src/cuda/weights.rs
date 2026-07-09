use std::collections::BTreeMap;

use crate::cuda::ffi::{CudaDevice, DeviceBuffer};
use crate::gguf::GgufModel;
use crate::quant::GgmlDType;
use crate::{Error, Result};

pub struct CudaWeights {
    tensors: BTreeMap<String, CudaTensor>,
}

pub struct CudaTensor {
    pub dtype: GgmlDType,
    pub shape: Vec<usize>,
    pub buffer: DeviceBuffer,
    row_stride_blocks: usize,
}

impl CudaWeights {
    pub fn load(dev: &CudaDevice, model: &GgufModel) -> Result<Self> {
        let mut tensors = BTreeMap::new();
        for (name, info) in model.tensor_infos() {
            let view = model.tensor(name)?;
            let (bytes, row_stride_blocks) = pad_for_mmq(info.dtype, &info.shape, view.raw_bytes)?;
            let buffer = dev.upload_bytes(&bytes)?;
            tensors.insert(
                name.clone(),
                CudaTensor {
                    dtype: info.dtype,
                    shape: info.shape.clone(),
                    buffer,
                    row_stride_blocks,
                },
            );
        }
        dev.sync()?;
        Ok(Self { tensors })
    }

    pub fn require(&self, name: &str) -> Result<&CudaTensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| Error::MissingTensor(name.to_string()))
    }
}

impl CudaTensor {
    pub fn rows(&self) -> Result<usize> {
        self.shape.first().copied().ok_or_else(|| {
            Error::InvalidGguf(format!(
                "CUDA tensor shape {:?} has no row dimension",
                self.shape
            ))
        })
    }

    pub fn cols(&self) -> Result<usize> {
        self.shape.get(1).copied().ok_or_else(|| {
            Error::InvalidGguf(format!(
                "CUDA tensor shape {:?} has no column dimension",
                self.shape
            ))
        })
    }

    pub fn row_stride_blocks(&self) -> usize {
        self.row_stride_blocks
    }

    pub fn ggml_type_id(&self) -> Result<i32> {
        Ok(match self.dtype {
            GgmlDType::Q5_0 => 6,
            GgmlDType::Q8_0 => 8,
            GgmlDType::Q4K => 12,
            GgmlDType::Q5K => 13,
            GgmlDType::Q6K => 14,
            other => return Err(Error::UnsupportedDType(other)),
        })
    }
}

fn pad_for_mmq(dtype: GgmlDType, shape: &[usize], raw: &[u8]) -> Result<(Vec<u8>, usize)> {
    if shape.len() != 2 || !is_mmq_quant(dtype) {
        return Ok((raw.to_vec(), raw.len().max(1)));
    }

    let rows = shape[0];
    let cols = shape[1];
    let qk = dtype.block_size();
    let type_size = dtype.type_size();
    let row_blocks = cols / qk;
    let row_bytes = row_blocks * type_size;
    let expected = rows * row_bytes;
    if raw.len() != expected {
        return Err(Error::InvalidGguf(format!(
            "CUDA tensor raw len {} for shape {:?} dtype {} did not match expected {expected}",
            raw.len(),
            shape,
            dtype
        )));
    }

    let iter_blocks = (256 / qk).max(1);
    let padded_row_blocks = row_blocks.div_ceil(iter_blocks) * iter_blocks;
    if padded_row_blocks == row_blocks {
        return Ok((raw.to_vec(), row_blocks));
    }

    let padded_row_bytes = padded_row_blocks * type_size;
    let mut padded = vec![0u8; rows * padded_row_bytes];
    for row in 0..rows {
        let src = &raw[row * row_bytes..(row + 1) * row_bytes];
        let dst = &mut padded[row * padded_row_bytes..row * padded_row_bytes + row_bytes];
        dst.copy_from_slice(src);
    }
    Ok((padded, padded_row_blocks))
}

fn is_mmq_quant(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q5_0 | GgmlDType::Q8_0 | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q6K
    )
}
