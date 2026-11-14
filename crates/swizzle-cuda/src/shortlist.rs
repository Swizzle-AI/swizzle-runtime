use std::rc::Rc;

use swizzle_quant::TensorType;

use crate::argmax::Argmax;
use crate::context::Context;
use crate::error::{Error, Result};
use crate::gemv::{DeviceMatrix, GemvInput};
use crate::kernels::{arg, Kernels};
use crate::memory::DeviceBuffer;
use crate::module::LaunchConfig;
use crate::stream::Stream;

/// Scratch device buffers required to screen an output head of `vocab` rows
/// via GPU Shortlist without host synchronization.
pub struct ShortlistWork {
    approx: DeviceBuffer<f32>,
    approx_best: Argmax,
    floor: DeviceBuffer<f32>,
    norm: DeviceBuffer<f32>,
    count: DeviceBuffer<u32>,
    kept_idx: DeviceBuffer<u32>,
    kept_val: DeviceBuffer<f32>,
}

impl ShortlistWork {
    pub fn new(ctx: &Rc<Context>, vocab: usize) -> Result<Self> {
        Ok(Self {
            approx: DeviceBuffer::new(ctx, vocab)?,
            approx_best: Argmax::new(ctx, 1, vocab)?,
            floor: DeviceBuffer::new(ctx, 1)?,
            norm: DeviceBuffer::new(ctx, 1)?,
            count: DeviceBuffer::new(ctx, 1)?,
            kept_idx: DeviceBuffer::new(ctx, vocab)?,
            kept_val: DeviceBuffer::new(ctx, vocab)?,
        })
    }

    pub fn size_bytes(&self) -> usize {
        self.approx.len() * size_of::<f32>() * 2
            + self.kept_idx.len() * size_of::<u32>()
            + 3 * size_of::<f32>()
    }
}

/// Interleaved `[head_radius, spread_radius]` per vocabulary row, coalesced in device memory.
pub fn pack_radii(pairs: &[(f32, f32)]) -> Vec<f32> {
    let mut out = Vec::with_capacity(pairs.len() * 2);
    for &(h, s) in pairs {
        out.push(h);
        out.push(s);
    }
    out
}

impl Kernels {
    /// Computes the exact greedy token of `head` (`Q8_0` or `Q6_K`) for the
    /// first vector of `x`, screening rows through `copy` (`Q4_K`) and packed
    /// `radii` (`[head, spread]` pairs) without reading unselected weight rows.
    #[allow(clippy::too_many_arguments)]
    pub fn shortlist_greedy(
        &self,
        head: &DeviceMatrix,
        copy: &DeviceMatrix,
        radii: &DeviceBuffer<f32>,
        slack: f32,
        x: &GemvInput,
        work: &mut ShortlistWork,
        out: &mut Argmax,
        stream: &Stream,
    ) -> Result<()> {
        let vocab = head.rows();
        if vocab == 0
            || copy.rows() != vocab
            || radii.len() != vocab * 2
            || copy.cols() != head.cols()
        {
            return Err(Error::Shape("shortlist shape mismatch".into()));
        }
        let (prepare, eval) = match head.ty() {
            TensorType::Q8_0 => (&self.shortlist_prepare_q8_0, &self.shortlist_eval_q8_0),
            TensorType::Q6K => (&self.shortlist_prepare_q6_k, &self.shortlist_eval_q6_k),
            ty => {
                return Err(Error::Shape(format!(
                    "unsupported CUDA shortlist head {ty}"
                )))
            }
        };

        // Stage 1: Q4_K screening GEMV + candidate winner.
        self.gemv_rows(copy, vocab, x, 0..1, &mut work.approx, stream)?;
        self.argmax(&work.approx, vocab, 1, &mut work.approx_best, stream)?;

        // Stage 2: Exact score for candidate winner + Q8 activation norm.
        let head_ptr = head.data().device_ptr();
        let total = head.total_blocks();
        let blocks_per_row = (head.cols() / head.ty().block_len()) as u32;
        let q8_blocks = (head.cols() / 32) as u32;
        let xq = x.quantized().device_ptr();
        let xds = x.block_scales().device_ptr();
        let best_ptr = work.approx_best.best().device_ptr();
        let floor_ptr = work.floor.device_ptr();
        let norm_ptr = work.norm.device_ptr();
        let count_ptr = work.count.device_ptr();
        let mut prep_args = [
            arg(&head_ptr),
            arg(&total),
            arg(&blocks_per_row),
            arg(&q8_blocks),
            arg(&xq),
            arg(&xds),
            arg(&best_ptr),
            arg(&floor_ptr),
            arg(&norm_ptr),
            arg(&count_ptr),
        ];
        // SAFETY: arguments match `shortlist_prepare_*` signature.
        unsafe {
            prepare.launch(
                LaunchConfig {
                    grid: [1, 1, 1],
                    block: [32, 1, 1],
                    shared_bytes: 0,
                },
                stream,
                &mut prep_args,
            )?;
        }

        // Stage 3: Bound-based row filter using warp-aggregated atomics.
        let approx_ptr = work.approx.device_ptr();
        let rad_ptr = radii.device_ptr();
        let vocab_u32 = vocab as u32;
        let kept_idx_ptr = work.kept_idx.device_ptr();
        let mut filter_args = [
            arg(&approx_ptr),
            arg(&rad_ptr),
            arg(&vocab_u32),
            arg(&slack),
            arg(&best_ptr),
            arg(&floor_ptr),
            arg(&norm_ptr),
            arg(&count_ptr),
            arg(&kept_idx_ptr),
        ];
        // SAFETY: arguments match `shortlist_filter` signature.
        unsafe {
            self.shortlist_filter.launch(
                LaunchConfig {
                    grid: [vocab_u32.div_ceil(256), 1, 1],
                    block: [256, 1, 1],
                    shared_bytes: 0,
                },
                stream,
                &mut filter_args,
            )?;
        }

        // Stage 4: Exact row evaluation over only surviving candidates.
        let kept_val_ptr = work.kept_val.device_ptr();
        let eval_blocks = (self.multiprocessors * 4).clamp(16, 128);
        let mut eval_args = [
            arg(&head_ptr),
            arg(&total),
            arg(&blocks_per_row),
            arg(&xq),
            arg(&xds),
            arg(&count_ptr),
            arg(&kept_idx_ptr),
            arg(&kept_val_ptr),
        ];
        // SAFETY: arguments match `shortlist_eval_*` signature.
        unsafe {
            eval.launch(
                LaunchConfig {
                    grid: [eval_blocks, 1, 1],
                    block: [128, 1, 1],
                    shared_bytes: 0,
                },
                stream,
                &mut eval_args,
            )?;
        }

        // Stage 5: Tie-preserving argmax over surviving exact logits.
        let out_ptr = out.best().device_ptr();
        let mut arg_args = [
            arg(&count_ptr),
            arg(&kept_idx_ptr),
            arg(&kept_val_ptr),
            arg(&out_ptr),
        ];
        // SAFETY: arguments match `shortlist_argmax` signature.
        unsafe {
            self.shortlist_argmax.launch(
                LaunchConfig {
                    grid: [1, 1, 1],
                    block: [256, 1, 1],
                    shared_bytes: 0,
                },
                stream,
                &mut arg_args,
            )
        }
    }
}
