//! Native Hugging Face `.safetensors` parser, HF->GGUF tensor name mapper,
//! and direct Safetensors-to-GGUF quantization (`F16`, `Q8_0`, `Q4_K_M`, `Q6_K`, `Q4_0`).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use rayon::prelude::*;
use serde_json::Value as JsonValue;
use swizzle_gguf::{TensorOut, TensorType, Value};

use crate::Result;

/// One tensor entry inside a `.safetensors` JSON header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SafeTensor {
    pub name: String,
    pub dtype: String,
    /// Shape in PyTorch/Safetensors row-major order (outermost dimension first).
    pub shape: Vec<u64>,
    /// Start and end byte offsets relative to the start of the data buffer.
    pub data_offsets: (usize, usize),
}

/// Loaded `.safetensors` file.
pub struct SafetensorsFile {
    buf: Vec<u8>,
    pub data_start: usize,
    pub metadata: BTreeMap<String, String>,
    pub tensors: Vec<SafeTensor>,
}

impl SafetensorsFile {
    pub fn open(path: &Path) -> Result<Self> {
        let buf = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_bytes(buf)
    }

    pub fn from_bytes(buf: Vec<u8>) -> Result<Self> {
        let (data_start, metadata, tensors) = parse_header(&buf)?;
        Ok(SafetensorsFile {
            buf,
            data_start,
            metadata,
            tensors,
        })
    }

    pub fn tensor_bytes(&self, t: &SafeTensor) -> &[u8] {
        let start = self.data_start + t.data_offsets.0;
        let end = self.data_start + t.data_offsets.1;
        &self.buf[start..end]
    }
}

/// Parses the 8-byte little-endian header length and JSON header of a `.safetensors` buffer.
pub fn parse_header(buf: &[u8]) -> Result<(usize, BTreeMap<String, String>, Vec<SafeTensor>)> {
    if buf.len() < 8 {
        return Err("safetensors file is shorter than 8 bytes".into());
    }
    let header_len = u64::from_le_bytes(buf[..8].try_into().unwrap()) as usize;
    if header_len == 0 || buf.len() < 8 + header_len {
        return Err("safetensors header length exceeds file size".into());
    }
    let data_start = 8 + header_len;
    let data_len = buf.len() - data_start;
    let root: serde_json::Map<String, JsonValue> =
        serde_json::from_slice(&buf[8..data_start])
            .map_err(|e| format!("invalid safetensors JSON header: {e}"))?;

    let mut metadata = BTreeMap::new();
    let mut tensors = Vec::new();
    for (key, val) in root {
        if key == "__metadata__" {
            if let Some(obj) = val.as_object() {
                for (mk, mv) in obj {
                    if let Some(s) = mv.as_str() {
                        metadata.insert(mk.clone(), s.to_owned());
                    }
                }
            }
            continue;
        }
        let obj = val
            .as_object()
            .ok_or_else(|| format!("safetensors tensor {key} is not a JSON object"))?;
        let dtype = obj
            .get("dtype")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| format!("safetensors tensor {key} missing dtype"))?
            .to_owned();
        let shape = obj
            .get("shape")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| format!("safetensors tensor {key} missing shape"))?
            .iter()
            .map(|d| {
                d.as_u64()
                    .ok_or_else(|| format!("safetensors tensor {key} has invalid shape dim"))
            })
            .collect::<std::result::Result<Vec<u64>, String>>()?;
        let offsets = obj
            .get("data_offsets")
            .and_then(JsonValue::as_array)
            .filter(|a| a.len() == 2)
            .ok_or_else(|| format!("safetensors tensor {key} missing data_offsets"))?;
        let start = offsets[0]
            .as_u64()
            .ok_or_else(|| format!("safetensors tensor {key} has invalid start offset"))?
            as usize;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| format!("safetensors tensor {key} has invalid end offset"))?
            as usize;
        if start > end || end > data_len {
            return Err(format!("safetensors tensor {key} has out-of-bounds data_offsets").into());
        }
        tensors.push(SafeTensor {
            name: key,
            dtype,
            shape,
            data_offsets: (start, end),
        });
    }
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    Ok((data_start, metadata, tensors))
}

/// Maps a standard Hugging Face Transformer tensor name (`model.layers.0.self_attn.q_proj.weight`)
/// to its canonical GGUF tensor name (`blk.0.attn_q.weight`).
pub fn hf_to_gguf_name(hf_name: &str) -> String {
    match hf_name {
        "model.embed_tokens.weight" | "transformer.wte.weight" => "token_embd.weight".into(),
        "model.norm.weight" | "transformer.ln_f.weight" => "output_norm.weight".into(),
        "model.norm.bias" | "transformer.ln_f.bias" => "output_norm.bias".into(),
        "lm_head.weight" => "output.weight".into(),
        _ => {
            if let Some(rest) = hf_name
                .strip_prefix("model.layers.")
                .or_else(|| hf_name.strip_prefix("transformer.h."))
            {
                if let Some((idx, tail)) = rest.split_once('.') {
                    let mapped_tail = match tail {
                        "input_layernorm.weight" | "ln_1.weight" => "attn_norm.weight",
                        "input_layernorm.bias" | "ln_1.bias" => "attn_norm.bias",
                        "post_attention_layernorm.weight" | "ln_2.weight" => "ffn_norm.weight",
                        "post_attention_layernorm.bias" | "ln_2.bias" => "ffn_norm.bias",
                        "self_attn.q_proj.weight" => "attn_q.weight",
                        "self_attn.q_proj.bias" => "attn_q.bias",
                        "self_attn.k_proj.weight" => "attn_k.weight",
                        "self_attn.k_proj.bias" => "attn_k.bias",
                        "self_attn.v_proj.weight" => "attn_v.weight",
                        "self_attn.v_proj.bias" => "attn_v.bias",
                        "self_attn.o_proj.weight" => "attn_output.weight",
                        "self_attn.o_proj.bias" => "attn_output.bias",
                        "self_attn.q_norm.weight" => "attn_q_norm.weight",
                        "self_attn.k_norm.weight" => "attn_k_norm.weight",
                        "mlp.gate_proj.weight" => "ffn_gate.weight",
                        "mlp.up_proj.weight" => "ffn_up.weight",
                        "mlp.down_proj.weight" => "ffn_down.weight",
                        other => other,
                    };
                    return format!("blk.{idx}.{mapped_tail}");
                }
            }
            hf_name.to_owned()
        }
    }
}

/// Converts a `.safetensors` file (and optional sibling `config.json`) into a quantized `.gguf` file.
pub fn convert_to_gguf(input: &Path, output: &Path, kind: &str) -> Result<()> {
    let st = SafetensorsFile::open(input)?;
    let target_ty = match kind.to_ascii_uppercase().as_str() {
        "F32" => TensorType::F32,
        "F16" => TensorType::F16,
        "Q8_0" => TensorType::Q8_0,
        "Q4_0" => TensorType::Q4_0,
        "Q4_K" | "Q4_K_M" => TensorType::Q4K,
        "Q5_K" | "Q5_K_M" => TensorType::Q5K,
        "Q6_K" => TensorType::Q6K,
        _ => {
            return Err(
                format!("unknown target type {kind}; use F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M or Q4_0")
                    .into(),
            )
        }
    };

    let mut metadata = Vec::new();
    let arch = detect_arch(input, &st);
    metadata.push(("general.architecture".into(), Value::String(arch.clone())));
    for (mk, mv) in &st.metadata {
        metadata.push((format!("safetensors.{mk}"), Value::String(mv.clone())));
    }
    metadata.push((
        "general.name".into(),
        Value::String(
            input
                .file_stem()
                .map_or_else(|| "swizzle-safetensors".into(), |s| s.to_string_lossy().into_owned()),
        ),
    ));
    metadata.push(("general.quantization_version".into(), Value::U32(2)));
    append_config_metadata(input, &arch, &mut metadata);

    let mut out_tensors = Vec::with_capacity(st.tensors.len());
    for t in &st.tensors {
        let gguf_name = hf_to_gguf_name(&t.name);
        // Safetensors shapes are outermost-first [rows, cols]; GGUF dims are innermost-first [cols, rows]
        let mut dims: Vec<u64> = t.shape.iter().copied().rev().collect();
        if dims.is_empty() {
            dims.push(1);
        }
        let cols = dims[0];
        let is_2d_weight = dims.len() >= 2
            && gguf_name.ends_with(".weight")
            && !gguf_name.contains("_norm.weight");
        let ty = if is_2d_weight && target_ty != TensorType::F32 && target_ty != TensorType::F16 {
            if cols.is_multiple_of(target_ty.block_len() as u64) {
                target_ty
            } else if cols.is_multiple_of(32) {
                TensorType::Q8_0
            } else {
                TensorType::F16
            }
        } else if dims.len() == 1 || gguf_name.contains("_norm") || gguf_name.ends_with(".bias") {
            TensorType::F32
        } else {
            target_ty
        };
        out_tensors.push(TensorOut {
            name: gguf_name,
            dims,
            ty,
        });
    }

    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut sink =
        BufWriter::new(File::create(output).map_err(|e| format!("{}: {e}", output.display()))?);
    let total = out_tensors.len();
    swizzle_gguf::write(&mut sink, &metadata, &out_tensors, |i| {
        let src_t = &st.tensors[i];
        let dst_t = &out_tensors[i];
        eprint!(
            "\r[{:>4}/{total}] {:<40} {:>4} -> {:<6}",
            i + 1,
            dst_t.name,
            src_t.dtype,
            dst_t.ty.name()
        );
        encode_tensor(&st, src_t, dst_t)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
    })?;
    sink.flush()?;
    eprintln!();
    Ok(())
}

fn detect_arch(input: &Path, st: &SafetensorsFile) -> String {
    if let Some(cfg) = read_sibling_config(input) {
        if let Some(mt) = cfg.get("model_type").and_then(JsonValue::as_str) {
            return match mt {
                "qwen2" => "qwen2".into(),
                "qwen3" => "qwen3".into(),
                "mistral" | "llama" => "llama".into(),
                "gemma" | "gemma2" | "gemma3" => mt.into(),
                other => other.into(),
            };
        }
    }
    if st.tensors.iter().any(|t| t.name.contains("q_norm.weight")) {
        "qwen3".into()
    } else {
        "llama".into()
    }
}

fn read_sibling_config(input: &Path) -> Option<JsonValue> {
    let dir = input.parent()?;
    let text = std::fs::read_to_string(dir.join("config.json")).ok()?;
    serde_json::from_str(&text).ok()
}

fn append_config_metadata(input: &Path, arch: &str, metadata: &mut Vec<(String, Value)>) {
    let Some(cfg) = read_sibling_config(input) else {
        return;
    };
    let mut push_u32 = |suffix: &str, json_key: &str| {
        if let Some(v) = cfg.get(json_key).and_then(JsonValue::as_u64) {
            metadata.push((format!("{arch}.{suffix}"), Value::U32(v as u32)));
        }
    };
    push_u32("context_length", "max_position_embeddings");
    push_u32("embedding_length", "hidden_size");
    push_u32("block_count", "num_hidden_layers");
    push_u32("feed_forward_length", "intermediate_size");
    push_u32("attention.head_count", "num_attention_heads");
    push_u32("attention.head_count_kv", "num_key_value_heads");
    if let Some(eps) = cfg.get("rms_norm_eps").and_then(JsonValue::as_f64) {
        metadata.push((
            format!("{arch}.attention.layer_norm_rms_epsilon"),
            Value::F32(eps as f32),
        ));
    }
    if let Some(theta) = cfg.get("rope_theta").and_then(JsonValue::as_f64) {
        metadata.push((format!("{arch}.rope.freq_base"), Value::F32(theta as f32)));
    }
}

fn encode_tensor(st: &SafetensorsFile, src_t: &SafeTensor, dst_t: &TensorOut) -> Result<Vec<u8>> {
    let raw = st.tensor_bytes(src_t);
    let elements: usize = dst_t.dims.iter().product::<u64>() as usize;
    let floats = decode_to_f32(&src_t.dtype, raw, elements)?;
    match dst_t.ty {
        TensorType::F32 => {
            let mut out = Vec::with_capacity(floats.len() * 4);
            for v in floats {
                out.extend_from_slice(&v.to_le_bytes());
            }
            Ok(out)
        }
        TensorType::F16 => {
            let mut out = Vec::with_capacity(floats.len() * 2);
            for v in floats {
                out.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes());
            }
            Ok(out)
        }
        ty => {
            let cols = dst_t.dims[0] as usize;
            let rows = elements / cols;
            let out_row = ty.bytes_for(cols as u64).ok_or("invalid row block size")? as usize;
            let mut out = vec![0u8; rows * out_row];
            out.par_chunks_mut(out_row)
                .zip(floats.par_chunks(cols))
                .for_each(|(dst, row)| {
                    swizzle_quant::quantize(ty, row, dst);
                });
            Ok(out)
        }
    }
}

fn decode_to_f32(dtype: &str, raw: &[u8], elements: usize) -> Result<Vec<f32>> {
    match dtype {
        "F32" => {
            if raw.len() != elements * 4 {
                return Err("F32 tensor byte length mismatch".into());
            }
            Ok(raw
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        }
        "F16" => {
            if raw.len() != elements * 2 {
                return Err("F16 tensor byte length mismatch".into());
            }
            let mut out = vec![0.0f32; elements];
            swizzle_quant::dequantize(TensorType::F16, raw, &mut out)
                .map_err(|e| format!("F16 dequantize: {e}"))?;
            Ok(out)
        }
        "BF16" => {
            if raw.len() != elements * 2 {
                return Err("BF16 tensor byte length mismatch".into());
            }
            let mut out = vec![0.0f32; elements];
            swizzle_quant::dequantize(TensorType::BF16, raw, &mut out)
                .map_err(|e| format!("BF16 dequantize: {e}"))?;
            Ok(out)
        }
        other => Err(format!("unsupported safetensors dtype {other}; expected F32, F16 or BF16").into()),
    }
}

/// IEEE-754 single-precision (`f32`) to half-precision (`f16`) bits with round-to-nearest-even.
fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp == 255 {
        return sign | 0x7c00 | ((mant != 0) as u16);
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return sign | 0x7c00;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign;
        }
        let m = mant | 0x0080_0000;
        let shift = 14 - new_exp;
        return sign | ((m >> shift) as u16);
    }
    sign | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_hf_tensor_names_to_gguf() {
        assert_eq!(hf_to_gguf_name("model.embed_tokens.weight"), "token_embd.weight");
        assert_eq!(
            hf_to_gguf_name("model.layers.3.self_attn.q_proj.weight"),
            "blk.3.attn_q.weight"
        );
        assert_eq!(
            hf_to_gguf_name("model.layers.11.mlp.down_proj.weight"),
            "blk.11.ffn_down.weight"
        );
        assert_eq!(hf_to_gguf_name("lm_head.weight"), "output.weight");
    }

    #[test]
    fn parses_safetensors_header_and_converts_to_gguf() {
        let header = serde_json::json!({
            "__metadata__": {"format": "pt"},
            "model.norm.weight": {
                "dtype": "F32",
                "shape": [4],
                "data_offsets": [0, 16]
            },
            "model.layers.0.self_attn.q_proj.weight": {
                "dtype": "F32",
                "shape": [2, 32],
                "data_offsets": [16, 272]
            }
        })
        .to_string();
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        for i in 0..68 {
            buf.extend_from_slice(&((i as f32) * 0.05).to_le_bytes());
        }
        let (start, meta, tensors) = parse_header(&buf).unwrap();
        assert_eq!(start, 8 + header.len());
        assert_eq!(meta.get("format").map(String::as_str), Some("pt"));
        assert_eq!(tensors.len(), 2);

        let dir = std::env::temp_dir().join("swizzle_st_test");
        let _ = std::fs::create_dir_all(&dir);
        let st_path = dir.join("model.safetensors");
        let gguf_path = dir.join("model-q8_0.gguf");
        std::fs::write(&st_path, &buf).unwrap();
        convert_to_gguf(&st_path, &gguf_path, "Q8_0").unwrap();
        let gguf = swizzle_gguf::GgufFile::open(&gguf_path).unwrap();
        assert_eq!(gguf.header().tensors.len(), 2);
        let q = gguf.header().tensor("blk.0.attn_q.weight").unwrap();
        assert_eq!(q.ty, TensorType::Q8_0);
        assert_eq!(q.dims, vec![32, 2]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
