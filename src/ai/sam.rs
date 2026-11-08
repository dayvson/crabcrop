//! Segmentation (issue #175): the SAM-family encoder and
//! decoder pair, the box-prompt decode, and mask geometry.

use image::{DynamicImage, GrayImage};
use ort::session::Session;
use std::sync::Mutex;

use super::session::{build_session, fir_resize};
use super::{AiError, SegmentEmbedding, SegmentProvider};

/// The SAM encoder's fixed input edge.
const SEGMENT_INPUT: u32 = 1024;

/// The SAM embedding's channel count and grid edge: the
/// fixed output shape of the encoder exports the segment
/// contract targets.
const SEGMENT_CHANNELS: usize = 256;

const SEGMENT_GRID: usize = 64;

/// The decoder's mask-prompt input edge (4x the grid): the
/// contract takes a mask prompt slot even when no mask is
/// being prompted, so it is always fed, empty.
const SEGMENT_MASK_INPUT: usize = 256;

/// SAM input preprocessing, the reference pipeline: resize
/// the longest side to the encoder's edge keeping aspect,
/// normalize with the SAM pixel statistics (which are in
/// 0..255 scale), and zero-pad to the square in normalized
/// space. Output is planar CHW at the encoder's edge.
pub fn segment_preprocess(img: &DynamicImage) -> Vec<f32> {
    const MEAN: [f32; 3] = [123.675, 116.28, 103.53];
    const STD: [f32; 3] = [58.395, 57.12, 57.375];
    let (nw, nh) = segment_input_dims(img.width(), img.height());
    let resized = fir_resize(&img.to_rgb8(), nw, nh);
    let size = SEGMENT_INPUT as usize;
    let mut out = vec![0f32; 3 * size * size];
    for (i, px) in resized.pixels().enumerate() {
        let (rx, ry) = ((i as u32 % nw) as usize, (i as u32 / nw) as usize);
        for c in 0..3 {
            out[c * size * size + ry * size + rx] = (px[c] as f32 - MEAN[c]) / STD[c];
        }
    }
    out
}

/// The decoder's box-prompt coordinates for one box, as
/// (x0, y0, x1, y1) in the encoder's input space. The
/// exported decoder does not rescale prompts (its point
/// embedding only shifts by half a pixel and divides by
/// the input edge), so the caller applies the same
/// longest-side scale the image itself got, exactly as
/// the reference ONNX pipeline does.
pub fn segment_box_coords(img_w: u32, img_h: u32, bbox: (u32, u32, u32, u32)) -> [f32; 4] {
    let scale = SEGMENT_INPUT as f32 / img_w.max(img_h) as f32;
    let (x, y, w, h) = bbox;
    [
        x as f32 * scale,
        y as f32 * scale,
        x.saturating_add(w) as f32 * scale,
        y.saturating_add(h) as f32 * scale,
    ]
}

/// The size `segment_preprocess` resizes a `w` x `h` frame
/// to before padding: the longest side at the encoder's
/// edge, aspect kept, at least 1 px a side.
fn segment_input_dims(w: u32, h: u32) -> (u32, u32) {
    let scale = SEGMENT_INPUT as f32 / w.max(h) as f32;
    (
        ((w as f32 * scale).round() as u32).max(1),
        ((h as f32 * scale).round() as u32).max(1),
    )
}

/// The mask plane, as (w, h), the decoder is asked to
/// answer for an `img_w` x `img_h` frame (its
/// `orig_im_size` input): the frame itself when it fits the
/// encoder's input, else the frame at the encoder's scale,
/// the size `segment_preprocess` resized it to. The decoder
/// answers one f32 plane of that size per mask candidate,
/// upsampled from a 256x256 field, so asking for the full
/// frame added no detail and made every box cost hundreds
/// of MB on large frames (issue #410). Either way the plane
/// is at most `SEGMENT_INPUT` px a side.
fn segment_mask_dims(img_w: u32, img_h: u32) -> (u32, u32) {
    if img_w.max(img_h) <= SEGMENT_INPUT {
        (img_w.max(1), img_h.max(1))
    } else {
        segment_input_dims(img_w, img_h)
    }
}

/// Charge the decoder's mask plane against the shared
/// output pixel budget before the decoder runs (issues
/// #359 and #410). `segment_mask_dims` keeps the plane at
/// the encoder's scale, so this cannot bind today; it
/// keeps the invariant explicit if that bound ever moves.
fn check_mask_plane(mw: u32, mh: u32) -> Result<(), AiError> {
    if mw as u64 * mh as u64 > crate::engine::MAX_OUTPUT_PIXELS {
        return Err(AiError::BadOutput(format!(
            "mask shape {mw}x{mh} exceeds the {} pixel budget",
            crate::engine::MAX_OUTPUT_PIXELS
        )));
    }
    Ok(())
}

/// One decoder mask logit as a soft strength: the logistic
/// function scaled to 0..255.
fn sigmoid_strength(logit: f32) -> u8 {
    (255.0 / (1.0 + (-logit).exp())).round() as u8
}

/// One box prompt in the decoder's terms: the box's
/// corners in encoder space, and the mask plane it is
/// asked to answer (`orig_im_size`).
#[derive(Debug, Clone, Copy, PartialEq)]
struct MaskPrompt {
    coords: [f32; 4],
    mask_w: u32,
    mask_h: u32,
}

/// The decoder's answer to one prompt, borrowed from its
/// output tensors: the mask logits' dims (1 x candidates x
/// h x w) and values, and the predicted-IoU scores, one per
/// candidate.
struct MaskAnswer<'a> {
    dims: &'a [i64],
    logits: &'a [f32],
    scores: &'a [f32],
}

/// The callback a decoder run hands its answer to.
type MaskFinish<'f> = dyn FnMut(MaskAnswer<'_>) -> Result<GrayImage, AiError> + 'f;

/// Decode one box prompt into the box's soft mask. `run`
/// executes the decoder for the prompt and passes its
/// answer, still borrowed from the output tensors, to the
/// callback it is given: the ONNX session in production, a
/// fake in tests. The plane is bounded and charged before
/// `run`, and only the best candidate's box region is read
/// back, so neither the decoder's answer nor anything
/// copied from it grows with the frame (issue #410).
fn decode_box(
    img_w: u32,
    img_h: u32,
    bbox: (u32, u32, u32, u32),
    run: impl FnOnce(&MaskPrompt, &mut MaskFinish<'_>) -> Result<GrayImage, AiError>,
) -> Result<GrayImage, AiError> {
    let (_, _, w, h) = bbox;
    if w == 0 || h == 0 {
        return Ok(GrayImage::new(w, h));
    }
    let (mask_w, mask_h) = segment_mask_dims(img_w, img_h);
    check_mask_plane(mask_w, mask_h)?;
    let prompt = MaskPrompt {
        coords: segment_box_coords(img_w, img_h, bbox),
        mask_w,
        mask_h,
    };
    run(&prompt, &mut |answer| {
        box_mask(&answer, &prompt, img_w, img_h, bbox)
    })
}

/// The box's soft mask from a decoder answer: the best
/// candidate by predicted IoU, read at the box's pixel
/// centers and passed through the logistic function. The
/// plane covers the whole frame; when it is smaller than
/// the frame, each pixel is sampled bilinearly with the
/// half-pixel mapping the decoder's own resize uses, so
/// only the box region is ever resampled. A plane larger
/// than the one asked for, or one whose claimed shape the
/// values do not fill, is a bad output. A box reaching
/// past the frame (a detector's rounding) clips, and the
/// missing strip stays background, so the answer is always
/// the box's own size.
fn box_mask(
    answer: &MaskAnswer<'_>,
    prompt: &MaskPrompt,
    img_w: u32,
    img_h: u32,
    bbox: (u32, u32, u32, u32),
) -> Result<GrayImage, AiError> {
    let dims = answer.dims;
    let bad_shape = || AiError::BadOutput(format!("unexpected mask output shape: {dims:?}"));
    if dims.len() != 4 || dims[0] != 1 {
        return Err(bad_shape());
    }
    let candidates = usize::try_from(dims[1]).map_err(|_| bad_shape())?;
    let mh = u32::try_from(dims[2]).map_err(|_| bad_shape())?;
    let mw = u32::try_from(dims[3]).map_err(|_| bad_shape())?;
    if candidates == 0 || answer.scores.len() < candidates {
        return Err(AiError::BadOutput(format!(
            "the model produced {candidates} mask candidates but {} scores",
            answer.scores.len()
        )));
    }
    if mw == 0 || mh == 0 || mw > prompt.mask_w || mh > prompt.mask_h {
        return Err(AiError::BadOutput(format!(
            "the decoder answered a {mw}x{mh} mask plane for a {}x{} request",
            prompt.mask_w, prompt.mask_h
        )));
    }
    // The predicted-IoU scores are unbounded model
    // predictions (a near-perfect mask can score over
    // 1.0): they rank the candidates, nothing more.
    let best = answer.scores[..candidates]
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    // The claimed shape must fit the values actually
    // returned before anything slices by it (issue #267).
    let plane = mw as usize * mh as usize;
    let end = (best + 1)
        .checked_mul(plane)
        .ok_or_else(|| AiError::BadOutput(format!("mask shape overflows: {dims:?}")))?;
    if answer.logits.len() < end {
        return Err(AiError::BadOutput(format!(
            "mask output has {} values, short of the {end} its shape claims",
            answer.logits.len()
        )));
    }
    let channel = &answer.logits[best * plane..end];
    let (x, y, w, h) = bbox;
    let cx = x.min(img_w);
    let cy = y.min(img_h);
    let cw = w.min(img_w - cx);
    let ch = h.min(img_h - cy);
    let mut boxed = GrayImage::new(w, h);
    let (mw_us, mh_us) = (mw as usize, mh as usize);
    if (mw, mh) == (img_w, img_h) {
        for by in 0..ch {
            let row = (cy + by) as usize * mw_us;
            for bx in 0..cw {
                let v = channel[row + (cx + bx) as usize];
                boxed.put_pixel(bx, by, image::Luma([sigmoid_strength(v)]));
            }
        }
        return Ok(boxed);
    }
    // Source taps for one output coordinate: the pixel
    // center mapped into the plane, clamped to its edges.
    let taps = |dst: u32, scale: f32, n: usize| -> (usize, usize, f32) {
        let s = ((dst as f32 + 0.5) * scale - 0.5).max(0.0);
        let i0 = (s.floor() as usize).min(n - 1);
        let i1 = (i0 + 1).min(n - 1);
        (i0, i1, (s - i0 as f32).clamp(0.0, 1.0))
    };
    let (kx, ky) = (mw as f32 / img_w as f32, mh as f32 / img_h as f32);
    let columns: Vec<_> = (0..cw).map(|bx| taps(cx + bx, kx, mw_us)).collect();
    for by in 0..ch {
        let (y0, y1, fy) = taps(cy + by, ky, mh_us);
        let (r0, r1) = (
            &channel[y0 * mw_us..][..mw_us],
            &channel[y1 * mw_us..][..mw_us],
        );
        for (bx, &(x0, x1, fx)) in (0..cw).zip(&columns) {
            let top = r0[x0] + (r0[x1] - r0[x0]) * fx;
            let bottom = r1[x0] + (r1[x1] - r1[x0]) * fx;
            let v = top + (bottom - top) * fy;
            boxed.put_pixel(bx, by, image::Luma([sigmoid_strength(v)]));
        }
    }
    Ok(boxed)
}

/// The geometry of one segmentation mask (issue #175):
/// where the silhouette sits inside the mask, how much of
/// it the silhouette covers, and where its mass centers.
/// This is the mask geometry crop anchoring and meta
/// reporting consume: a tighter statement of an object
/// than its detection rectangle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaskGeometry {
    /// The silhouette's tight box, in mask coordinates:
    /// (x, y, w, h) spanning the pixels over half strength.
    pub tight: (u32, u32, u32, u32),
    /// The silhouette's soft area share of the mask,
    /// 0..=1: every pixel weighted by its strength.
    pub coverage: f32,
    /// The mask's center of mass, as fractions of the
    /// mask's width and height.
    pub centroid: (f32, f32),
}

/// Measure a soft segment mask (0 is background). `None`
/// when nothing in the mask passes half strength: the
/// model found no object there.
pub fn mask_geometry(mask: &GrayImage) -> Option<MaskGeometry> {
    let (w, h) = (mask.width(), mask.height());
    if w == 0 || h == 0 {
        return None;
    }
    let (mut total, mut sx, mut sy) = (0f64, 0f64, 0f64);
    let (mut min_x, mut min_y) = (w, h);
    let (mut max_x, mut max_y) = (0u32, 0u32);
    let mut any = false;
    for (x, y, p) in mask.enumerate_pixels() {
        let v = p[0] as f64;
        if v > 0.0 {
            total += v;
            sx += v * x as f64;
            sy += v * y as f64;
        }
        if p[0] > 127 {
            any = true;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    if !any || total == 0.0 {
        return None;
    }
    Some(MaskGeometry {
        tight: (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1),
        coverage: (total / (w as f64 * h as f64 * 255.0)) as f32,
        centroid: (
            (sx / total / w as f64) as f32,
            (sy / total / h as f64) as f32,
        ),
    })
}

/// Box-prompted segmentation backed by a SAM-family ONNX
/// export pair (issue #175): an image encoder producing a
/// 256x64x64 embedding from a 1024x1024 input, and a
/// prompt decoder (the `SamOnnxModel` export shape) taking
/// the embedding plus point prompts and answering mask
/// logits at the size its `orig_im_size` input names, one
/// candidate per mask token, with predicted-IoU scores.
/// That size is the frame's, capped at the encoder's scale
/// (`segment_mask_dims`, issue #410). The reference
/// artifact is MobileSAM (Apache-2.0): a 28 MB encoder and
/// a 16.5 MB decoder exported from the project's
/// checkpoint with its own ONNX export path. Box prompts
/// are two corner points with labels 2 and 3; the best
/// candidate by predicted IoU wins, and its logits become
/// the soft mask through the logistic function.
pub struct OnnxSegmentProvider {
    encoder: Mutex<Session>,
    decoder: Mutex<Session>,
    encoder_input: String,
    embeddings_input: String,
    coords_input: String,
    labels_input: String,
    mask_input: String,
    has_mask_input: String,
    size_input: String,
    masks_output: String,
    scores_output: String,
}

impl OnnxSegmentProvider {
    /// Load an encoder/decoder pair. Input and output
    /// names are found by their conventional names,
    /// falling back to declaration order, like the
    /// inpainting provider. A model that does not fit the
    /// pair contract (wrong input count, an embedding of
    /// the wrong shape, a probe mask of the wrong size)
    /// is rejected here, at boot.
    pub fn load(
        dylib: Option<&str>,
        encoder_path: &str,
        decoder_path: &str,
    ) -> Result<Self, AiError> {
        let (encoder, encoder_input) = build_session(dylib, encoder_path)?;
        let (decoder, _) = build_session(dylib, decoder_path)?;
        let names: Vec<String> = decoder
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        if names.len() < 6 {
            return Err(AiError::Load(format!(
                "the model {decoder_path} has {} input(s); a segment decoder \
                 takes an embedding, point coordinates and labels, a mask \
                 prompt slot, its presence flag, and the image size",
                names.len()
            )));
        }
        let pick = |want: &str, fallback: usize| -> String {
            names
                .iter()
                .find(|n| n.eq_ignore_ascii_case(want))
                .cloned()
                .unwrap_or_else(|| names[fallback].clone())
        };
        let out_names: Vec<String> = decoder
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        let pick_out = |want: &str, fallback: usize| -> Result<String, AiError> {
            out_names
                .iter()
                .find(|n| n.eq_ignore_ascii_case(want))
                .cloned()
                .or_else(|| out_names.get(fallback).cloned())
                .ok_or_else(|| {
                    AiError::Load(format!("the model {decoder_path} has no output for {want}"))
                })
        };
        let provider = Self {
            encoder: Mutex::new(encoder),
            decoder: Mutex::new(decoder),
            encoder_input,
            embeddings_input: pick("image_embeddings", 0),
            coords_input: pick("point_coords", 1),
            labels_input: pick("point_labels", 2),
            mask_input: pick("mask_input", 3),
            has_mask_input: pick("has_mask_input", 4),
            size_input: pick("orig_im_size", 5),
            masks_output: pick_out("masks", 0)?,
            scores_output: pick_out("iou_predictions", 1)?,
        };
        // Probe the pair end to end, like the other
        // providers probe their contracts: encode a small
        // image and segment a box inside it.
        let probe = DynamicImage::new_rgb8(64, 48);
        let mask = provider
            .segment(&probe, (8, 8, 32, 24))
            .map_err(|e| AiError::Load(format!("the segment model pair failed its probe: {e}")))?;
        if (mask.width(), mask.height()) != (32, 24) {
            return Err(AiError::Load(format!(
                "the segment model pair failed its probe: a 32x24 box came \
                 back as a {}x{} mask",
                mask.width(),
                mask.height()
            )));
        }
        Ok(provider)
    }
}

impl std::fmt::Debug for OnnxSegmentProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxSegmentProvider")
            .field("encoder_input", &self.encoder_input)
            .field("embeddings_input", &self.embeddings_input)
            .field("masks_output", &self.masks_output)
            .finish_non_exhaustive()
    }
}

impl SegmentProvider for OnnxSegmentProvider {
    fn encode(&self, img: &DynamicImage) -> Result<SegmentEmbedding, AiError> {
        let size = SEGMENT_INPUT as usize;
        let input = segment_preprocess(img);
        let tensor = ort::value::Tensor::from_array(([1, 3, size, size], input.into_boxed_slice()))
            .map_err(|e| AiError::Inference(format!("cannot build the input tensor: {e}")))?;
        let mut session = self.encoder.lock().map_err(|_| AiError::LockPoisoned)?;
        let outputs = session
            .run(ort::inputs![self.encoder_input.as_str() => tensor])
            .map_err(|e| AiError::Inference(format!("inference failed: {e}")))?;
        let (_, first) = outputs
            .into_iter()
            .next()
            .ok_or_else(|| AiError::BadOutput("the model produced no outputs".to_string()))?;
        let (shape, data) = first
            .try_extract_tensor::<f32>()
            .map_err(|e| AiError::BadOutput(format!("cannot read the model output: {e}")))?;
        let want = SEGMENT_CHANNELS * SEGMENT_GRID * SEGMENT_GRID;
        if shape.len() != 4
            || shape[1] as usize != SEGMENT_CHANNELS
            || shape[2] as usize != SEGMENT_GRID
            || shape[3] as usize != SEGMENT_GRID
            || data.len() != want
        {
            return Err(AiError::BadOutput(format!(
                "unexpected embedding shape: {shape:?} ({} values); the \
                 contract takes a 256x64x64 field",
                data.len()
            )));
        }
        Ok(SegmentEmbedding {
            data: data.to_vec(),
        })
    }

    fn decode(
        &self,
        embedding: &SegmentEmbedding,
        img_w: u32,
        img_h: u32,
        bbox: (u32, u32, u32, u32),
    ) -> Result<GrayImage, AiError> {
        decode_box(img_w, img_h, bbox, |prompt, finish| {
            self.run_decoder(embedding, prompt, finish)
        })
    }
}

impl OnnxSegmentProvider {
    /// Run the decoder for one prompt and pass its masks
    /// and predicted-IoU outputs to `finish`, borrowed from
    /// the session's output tensors: no output is copied
    /// here.
    fn run_decoder(
        &self,
        embedding: &SegmentEmbedding,
        prompt: &MaskPrompt,
        finish: &mut MaskFinish<'_>,
    ) -> Result<GrayImage, AiError> {
        let want = SEGMENT_CHANNELS * SEGMENT_GRID * SEGMENT_GRID;
        if embedding.data.len() != want {
            return Err(AiError::BadOutput(format!(
                "the embedding carries {} values; the segment contract \
                 takes {want}",
                embedding.data.len()
            )));
        }
        let coords = prompt.coords;
        let embeddings_tensor = ort::value::Tensor::from_array((
            [1, SEGMENT_CHANNELS, SEGMENT_GRID, SEGMENT_GRID],
            embedding.data.clone().into_boxed_slice(),
        ))
        .map_err(|e| AiError::Inference(format!("cannot build the embedding tensor: {e}")))?;
        let coords_tensor = ort::value::Tensor::from_array((
            [1, 2, 2],
            vec![coords[0], coords[1], coords[2], coords[3]].into_boxed_slice(),
        ))
        .map_err(|e| AiError::Inference(format!("cannot build the prompt tensor: {e}")))?;
        let labels_tensor =
            ort::value::Tensor::from_array(([1, 2], vec![2.0f32, 3.0].into_boxed_slice()))
                .map_err(|e| AiError::Inference(format!("cannot build the prompt tensor: {e}")))?;
        let mask_tensor = ort::value::Tensor::from_array((
            [1, 1, SEGMENT_MASK_INPUT, SEGMENT_MASK_INPUT],
            vec![0f32; SEGMENT_MASK_INPUT * SEGMENT_MASK_INPUT].into_boxed_slice(),
        ))
        .map_err(|e| AiError::Inference(format!("cannot build the mask prompt tensor: {e}")))?;
        let has_mask_tensor =
            ort::value::Tensor::from_array(([1], vec![0.0f32].into_boxed_slice())).map_err(
                |e| AiError::Inference(format!("cannot build the mask flag tensor: {e}")),
            )?;
        let size_tensor = ort::value::Tensor::from_array((
            [2],
            vec![prompt.mask_h as f32, prompt.mask_w as f32].into_boxed_slice(),
        ))
        .map_err(|e| AiError::Inference(format!("cannot build the size tensor: {e}")))?;
        let mut session = self.decoder.lock().map_err(|_| AiError::LockPoisoned)?;
        let outputs = session
            .run(ort::inputs![
                self.embeddings_input.as_str() => embeddings_tensor,
                self.coords_input.as_str() => coords_tensor,
                self.labels_input.as_str() => labels_tensor,
                self.mask_input.as_str() => mask_tensor,
                self.has_mask_input.as_str() => has_mask_tensor,
                self.size_input.as_str() => size_tensor
            ])
            .map_err(|e| AiError::Inference(format!("inference failed: {e}")))?;
        fn read(value: &ort::value::DynValue) -> Result<(&ort::value::Shape, &[f32]), AiError> {
            value
                .try_extract_tensor::<f32>()
                .map_err(|e| AiError::BadOutput(format!("cannot read the model output: {e}")))
        }
        // By name, falling back to the outputs' order.
        let output = |name: &str, index: usize| {
            outputs
                .get(name)
                .or_else(|| outputs.keys().nth(index).and_then(|k| outputs.get(k)))
        };
        let masks = output(&self.masks_output, 0)
            .ok_or_else(|| AiError::BadOutput("the model produced no outputs".to_string()))?;
        let scores = output(&self.scores_output, 1).ok_or_else(|| {
            AiError::BadOutput("the model produced no IoU predictions".to_string())
        })?;
        let (mask_shape, logits) = read(masks)?;
        let (_, scores) = read(scores)?;
        let dims: Vec<i64> = mask_shape.iter().copied().collect();
        finish(MaskAnswer {
            dims: &dims,
            logits,
            scores,
        })
    }
}

#[cfg(test)]
mod tests;
