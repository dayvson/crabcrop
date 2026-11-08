//! The aesthetic head's checkpoint reader: a PyTorch
//! save (a zip archive holding a pickle and raw float
//! storages), parsed into the head's weights and bias.

mod pickle;
mod zip;

use pickle::{PickleTensor, PickleValue, unpickle};
use zip::read_zip_entries;

/// Extract the (weights, bias) pair from a head
/// file's bytes: the zip entries, the pickle's state
/// dict, and the float storages it names. Every
/// deviation from the expected shape is an error
/// saying what was found instead.
pub(super) fn parse_aesthetic_head(bytes: &[u8]) -> Result<(Vec<f32>, f32), String> {
    let entries = read_zip_entries(bytes)?;
    let entry = |name: &str| {
        entries
            .iter()
            .find(|e| e.name == name)
            .map(|e| e.data.as_slice())
    };
    let pkl = entry("archive/data.pkl")
        .ok_or("the archive holds no archive/data.pkl: not a PyTorch save")?;
    let root = unpickle(pkl)?;
    let PickleValue::Map(pairs) = root else {
        return Err("the save's root is not a state dict".to_string());
    };
    let mut weight: Option<PickleTensor> = None;
    let mut bias: Option<PickleTensor> = None;
    for (key, value) in &pairs {
        let PickleValue::Tensor(tensor) = value else {
            return Err(format!(
                "the state dict entry {key:?} is not a tensor: only a single linear head (weight and bias tensors) is supported"
            ));
        };
        match key.as_str() {
            "weight" => weight = Some(tensor.clone()),
            "bias" => bias = Some(tensor.clone()),
            _ => {
                return Err(format!(
                    "the state dict holds an unexpected entry {key:?}: only a single linear head (weight and bias tensors) is supported"
                ));
            }
        }
    }
    let weight = weight.ok_or("the state dict holds no weight tensor")?;
    let bias = bias.ok_or("the state dict holds no bias tensor")?;
    let storage_floats = |tensor: &PickleTensor| -> Result<Vec<f32>, String> {
        let name = format!("archive/data/{}", tensor.storage);
        let data = entry(&name).ok_or_else(|| {
            format!("the archive holds no {name}: the tensor's storage is missing")
        })?;
        if data.len() % 4 != 0 {
            return Err(format!(
                "the storage {name} is {} bytes, not a whole number of float32 values",
                data.len()
            ));
        }
        let floats: Vec<f32> = data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect();
        let count = tensor.size.iter().product::<usize>();
        let end = tensor
            .offset
            .checked_add(count)
            .ok_or("the tensor size overflows")?;
        if end > floats.len() {
            return Err(format!(
                "the tensor overruns its storage: offset {} plus {count} values against {} stored",
                tensor.offset,
                floats.len()
            ));
        }
        Ok(floats[tensor.offset..end].to_vec())
    };
    let weights = storage_floats(&weight)?;
    if weights.is_empty() {
        return Err("the weight tensor is empty".to_string());
    }
    let bias = storage_floats(&bias)?;
    if bias.len() != 1 {
        return Err(format!(
            "the bias tensor holds {} values: a linear head's bias is a single value",
            bias.len()
        ));
    }
    Ok((weights, bias[0]))
}

#[cfg(test)]
mod tests;
