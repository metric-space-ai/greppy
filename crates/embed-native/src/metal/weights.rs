//! GGUF weight upload for the Metal backend.
//!
//! Keeps the product-path invariant that quantized GGUF tensors stay quantized:
//! all raw tensor bytes are copied once into a single shared `MTLBuffer`, then
//! each tensor is exposed as a view into that slab.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::gguf::GgufModel;
use crate::metal::ffi::{Buffer, Device};
use crate::metal::tensor::{GgmlType, Tensor};
use crate::quant::GgmlDType;
use crate::{Error, Result};

pub struct MetalWeights {
    pub tensors: BTreeMap<String, Tensor>,
    pub buffers: Vec<Arc<Buffer>>,
}

impl MetalWeights {
    pub fn load(dev: &Device, model: &GgufModel) -> Result<Self> {
        const ALIGN: usize = 256;
        fn align_up(value: usize, align: usize) -> Result<usize> {
            value
                .checked_add(align - 1)
                .map(|v| v / align * align)
                .ok_or_else(|| Error::InvalidGguf("Metal weight slab alignment overflows".into()))
        }

        let mut total_bytes = 0usize;
        let mut layout = Vec::with_capacity(model.tensor_infos().len());
        for (name, info) in model.tensor_infos() {
            let view = model.tensor(&name)?;
            let offset = align_up(total_bytes, ALIGN)?;
            total_bytes = offset.checked_add(view.raw_bytes.len()).ok_or_else(|| {
                Error::InvalidGguf(format!("Metal weight slab size overflows at tensor {name}"))
            })?;
            layout.push((name.clone(), info.clone(), offset, view.raw_bytes.len()));
        }

        let slab = Arc::new(dev.new_buffer(total_bytes).ok_or_else(|| {
            Error::InvalidGguf(format!(
                "failed to allocate Metal weight slab ({total_bytes} bytes)"
            ))
        })?);

        let mut tensors = BTreeMap::new();
        for (name, info, offset, _len) in layout {
            let view = model.tensor(&name)?;
            let dtype = map_dtype(info.dtype)?;
            let mut ne = [1i64; 4];
            for (idx, &dim) in info.shape.iter().rev().enumerate().take(4) {
                ne[idx] = i64::try_from(dim).map_err(|_| {
                    Error::InvalidGguf(format!(
                        "tensor {name} dimension {dim} does not fit Metal i64 shape"
                    ))
                })?;
            }
            let nb = Tensor::make_contiguous_strides(dtype, ne);
            unsafe {
                slab.write(offset, view.raw_bytes);
            }
            tensors.insert(
                name.clone(),
                Tensor {
                    name: name.clone(),
                    dtype,
                    ne,
                    nb,
                    buffer: slab.clone(),
                    offset,
                },
            );
        }

        Ok(Self {
            tensors,
            buffers: vec![slab],
        })
    }

    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    pub fn require(&self, name: &str) -> Result<&Tensor> {
        self.get(name)
            .ok_or_else(|| Error::MissingTensor(name.to_string()))
    }
}

fn map_dtype(dtype: GgmlDType) -> Result<GgmlType> {
    Ok(match dtype {
        GgmlDType::F32 => GgmlType::F32,
        GgmlDType::F16 => GgmlType::F16,
        GgmlDType::BF16 => GgmlType::Bf16,
        GgmlDType::Q5_0 => GgmlType::Q5_0,
        GgmlDType::Q8_0 => GgmlType::Q8_0,
        GgmlDType::Q4K => GgmlType::Q4_K,
        GgmlDType::Q5K => GgmlType::Q5_K,
        GgmlDType::Q6K => GgmlType::Q6_K,
        other => return Err(Error::UnsupportedDType(other)),
    })
}
