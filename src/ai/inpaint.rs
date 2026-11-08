//! Region removal: the inpainting provider and the hole
//! compositing around it.

use image::imageops::FilterType;
use image::{DynamicImage, GrayImage};
use ort::session::Session;
use std::sync::Mutex;

use super::session::build_session;
use super::{AiError, InpaintProvider};

/// The LaMa export's fixed input size: image and mask are
/// both fed at this square resolution, whatever the crop
/// window's shape (issue #176).
pub const INPAINT_INPUT_SIZE: u32 = 512;

/// A white-on-black mask of one rectangle, at the image's
/// own resolution: the mask `remove_region` hands to the
/// inpainting provider (white is the hole to fill).
pub fn region_mask(width: u32, height: u32, x: u32, y: u32, w: u32, h: u32) -> GrayImage {
    let mut mask = GrayImage::new(width, height);
    for py in y..y.saturating_add(h).min(height) {
        for px in x..x.saturating_add(w).min(width) {
            mask.put_pixel(px, py, image::Luma([255]));
        }
    }
    mask
}

/// The bounding box of the mask's nonzero pixels, as
/// (x, y, w, h); `None` when the mask removes nothing.
fn mask_bbox(mask: &GrayImage) -> Option<(u32, u32, u32, u32)> {
    let (mut min_x, mut min_y) = (u32::MAX, u32::MAX);
    let (mut max_x, mut max_y) = (0u32, 0u32);
    for (x, y, p) in mask.enumerate_pixels() {
        if p[0] > 0 {
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    if min_x == u32::MAX {
        return None;
    }
    Some((min_x, min_y, max_x - min_x + 1, max_y - min_y + 1))
}

/// The crop window an inpainting run works on: a square
/// centered on the mask's bounding box, with a side of
/// twice the box's larger axis so the model sees as much
/// surrounding context as hole, clamped into the image (a
/// window axis stops at the image edge, so an image
/// smaller than the square on one axis makes the window
/// non-square there; the resize to the model's square
/// input absorbs the difference). Pure, so the geometry
/// is testable without the runtime.
fn inpaint_window(img_w: u32, img_h: u32, bbox: (u32, u32, u32, u32)) -> (u32, u32, u32, u32) {
    let (bx, by, bw, bh) = bbox;
    let side = bw
        .max(bh)
        .saturating_mul(2)
        .min(img_w.max(img_h))
        .max(bw.max(bh));
    let (w, h) = (side.min(img_w), side.min(img_h));
    let (cx, cy) = (bx + bw / 2, by + bh / 2);
    let x = cx.saturating_sub(w / 2).min(img_w - w);
    let y = cy.saturating_sub(h / 2).min(img_h - h);
    (x, y, w, h)
}

/// Composite a filled crop window back into the image:
/// pixels under the mask (at the image's own resolution)
/// come from the filled window, every other pixel is the
/// original, bit for bit. The model's output is a
/// regeneration of the whole window, plausible everywhere
/// but faithful nowhere, so only the hole may be taken
/// from it. The source's alpha, when it has one, is kept:
/// the model sees RGB only, and inpainting fills color,
/// not transparency.
pub fn composite_inpaint(
    img: &DynamicImage,
    mask: &GrayImage,
    window: (u32, u32, u32, u32),
    filled: &image::RgbImage,
) -> DynamicImage {
    let (wx, wy, ww, wh) = window;
    let mut rgb = img.to_rgb8();
    let fitted;
    let filled = if filled.width() == ww && filled.height() == wh {
        filled
    } else {
        fitted = image::imageops::resize(filled, ww, wh, FilterType::Triangle);
        &fitted
    };
    for ly in 0..wh {
        for lx in 0..ww {
            let (gx, gy) = (wx + lx, wy + ly);
            if mask.get_pixel(gx, gy)[0] > 0 {
                *rgb.get_pixel_mut(gx, gy) = *filled.get_pixel(lx, ly);
            }
        }
    }
    let out = DynamicImage::ImageRgb8(rgb);
    if matches!(
        img,
        DynamicImage::ImageRgba8(_) | DynamicImage::ImageLumaA8(_)
    ) {
        let src = img.to_rgba8();
        let mut rgba = out.to_rgba8();
        for (px, s) in rgba.pixels_mut().zip(src.pixels()) {
            px[3] = s[3];
        }
        return DynamicImage::ImageRgba8(rgba);
    }
    out
}

/// Region removal backed by a LaMa-family ONNX model
/// (issue #176): a two-input session (image plus mask) at
/// a fixed 512x512 working resolution, e.g. the OpenCV zoo
/// export of big-lama. The export's contract, verified
/// against the artifact: the image goes in as RGB in
/// 0..1, the mask as 0..1 with 1 the hole, and the output
/// comes back in 0..255 (the asymmetry is the export's,
/// not a typo).
pub struct OnnxInpaintProvider {
    session: Mutex<Session>,
    image_input: String,
    mask_input: String,
}

impl OnnxInpaintProvider {
    /// Load a model file. The two inputs are found by
    /// their conventional names (`image`, `mask`), falling
    /// back to declaration order; a model without two
    /// inputs is rejected here. The load probes the model
    /// with a small removal, like the upscale load probes
    /// its factor: a model that cannot run it at all has
    /// failed to load, not failed an inference.
    pub fn load(dylib: Option<&str>, model_path: &str) -> Result<Self, AiError> {
        let (session, _) = build_session(dylib, model_path)?;
        let names: Vec<String> = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        if names.len() < 2 {
            return Err(AiError::Load(format!(
                "the model {model_path} has {} input(s); an inpainting model \
                 takes an image and a mask",
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
        let probe = Self {
            session: Mutex::new(session),
            image_input: pick("image", 0),
            mask_input: pick("mask", 1),
        };
        let img = DynamicImage::new_rgb8(64, 64);
        let mask = region_mask(64, 64, 24, 24, 16, 16);
        probe.inpaint(&img, &mask).map_err(|e| {
            AiError::Load(format!(
                "the model {model_path} failed its inpainting probe: {e}"
            ))
        })?;
        Ok(probe)
    }
}

impl std::fmt::Debug for OnnxInpaintProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxInpaintProvider")
            .field("image_input", &self.image_input)
            .field("mask_input", &self.mask_input)
            .finish_non_exhaustive()
    }
}

impl InpaintProvider for OnnxInpaintProvider {
    fn inpaint(&self, img: &DynamicImage, mask: &GrayImage) -> Result<DynamicImage, AiError> {
        let fitted;
        let mask = if mask.width() == img.width() && mask.height() == img.height() {
            mask
        } else {
            fitted = image::imageops::resize(mask, img.width(), img.height(), FilterType::Triangle);
            &fitted
        };
        let Some(bbox) = mask_bbox(mask) else {
            return Ok(img.clone());
        };
        let window = inpaint_window(img.width(), img.height(), bbox);
        let (wx, wy, ww, wh) = window;
        let size = INPAINT_INPUT_SIZE as usize;
        let crop = img.crop_imm(wx, wy, ww, wh).to_rgb8();
        let crop = image::imageops::resize(
            &crop,
            INPAINT_INPUT_SIZE,
            INPAINT_INPUT_SIZE,
            FilterType::Triangle,
        );
        let mask_crop = image::imageops::crop_imm(mask, wx, wy, ww, wh).to_image();
        let mask_crop = image::imageops::resize(
            &mask_crop,
            INPAINT_INPUT_SIZE,
            INPAINT_INPUT_SIZE,
            FilterType::Triangle,
        );
        let mut image_input = vec![0f32; 3 * size * size];
        for (i, px) in crop.pixels().enumerate() {
            for c in 0..3 {
                image_input[c * size * size + i] = px[c] as f32 / 255.0;
            }
        }
        let mut mask_input = vec![0f32; size * size];
        for (i, px) in mask_crop.pixels().enumerate() {
            mask_input[i] = px[0] as f32 / 255.0;
        }
        let image_tensor =
            ort::value::Tensor::from_array(([1, 3, size, size], image_input.into_boxed_slice()))
                .map_err(|e| AiError::Inference(format!("cannot build the image tensor: {e}")))?;
        let mask_tensor =
            ort::value::Tensor::from_array(([1, 1, size, size], mask_input.into_boxed_slice()))
                .map_err(|e| AiError::Inference(format!("cannot build the mask tensor: {e}")))?;
        let mut session = self.session.lock().map_err(|_| AiError::LockPoisoned)?;
        let outputs = session
            .run(ort::inputs![
                self.image_input.as_str() => image_tensor,
                self.mask_input.as_str() => mask_tensor
            ])
            .map_err(|e| AiError::Inference(format!("inference failed: {e}")))?;
        let (_, first) = outputs
            .into_iter()
            .next()
            .ok_or_else(|| AiError::BadOutput("the model produced no outputs".to_string()))?;
        let (shape, data) = first
            .try_extract_tensor::<f32>()
            .map_err(|e| AiError::BadOutput(format!("cannot read the model output: {e}")))?;
        if shape.len() != 4 || shape[0] != 1 || shape[1] != 3 {
            return Err(AiError::BadOutput(format!(
                "unexpected model output shape: {shape:?}"
            )));
        }
        let (ow, oh) = (shape[3] as u32, shape[2] as u32);
        // The claimed shape must fit the values actually
        // returned before anything indexes by it (issue
        // #267): a corrupt-but-loadable model can claim
        // dimensions its tensor does not carry.
        let plane = (ow as usize).checked_mul(oh as usize).ok_or_else(|| {
            AiError::BadOutput(format!("model output shape overflows: {shape:?}"))
        })?;
        if data.len() < plane * 3 {
            return Err(AiError::BadOutput(format!(
                "model output has {} values, short of the {} its shape claims",
                data.len(),
                plane * 3
            )));
        }
        // The export answers in 0..255 (see the type's
        // contract note), so the values clamp there, not
        // at 1.
        let mut filled = image::RgbImage::new(ow, oh);
        for (i, px) in filled.pixels_mut().enumerate() {
            for c in 0..3 {
                let v = data[c * plane + i];
                px[c] = v.clamp(0.0, 255.0).round() as u8;
            }
        }
        Ok(composite_inpaint(img, mask, window, &filled))
    }
}

#[cfg(test)]
mod tests;
