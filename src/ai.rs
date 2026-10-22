//! In-process AI models (ONNX). The engine stays free of the
//! runtime: it depends only on the provider traits defined
//! here. The server builds the real providers at boot from
//! operator configuration (a model is preloaded when one is
//! configured, and a broken configuration fails startup).
//!
//! The plumbing is shared with AI upscale: a model is loaded
//! once against a process-wide ONNX environment, and inference
//! runs behind the server-wide concurrency gate.

use image::{DynamicImage, GrayImage, imageops::FilterType};
use ort::session::{Session, builder::GraphOptimizationLevel};
use std::sync::{Mutex, OnceLock};

/// Produces a foreground mask for an image: bright where the
/// subject is, dark where the background is. Implemented by the
/// ONNX provider in production and by fakes in tests.
pub trait MaskProvider: Send + Sync + std::fmt::Debug {
    fn mask(&self, img: &DynamicImage) -> Result<GrayImage, String>;
}

/// The process-wide ONNX environment, created on first use:
/// in practice that is at boot, when the server preloads its
/// configured models.
static ENVIRONMENT: OnceLock<ort::environment::Environment> = OnceLock::new();

fn environment(dylib: Option<&str>) -> Result<&'static ort::environment::Environment, String> {
    if let Some(env) = ENVIRONMENT.get() {
        return Ok(env);
    }
    let builder = match dylib {
        Some(path) => ort::init_from(path)
            .map_err(|e| format!("cannot load the ONNX runtime from {path}: {e}"))?,
        None => ort::init(),
    };
    let env = builder
        .build()
        .map_err(|e| format!("cannot initialize the ONNX runtime: {e}"))?;
    let _ = ENVIRONMENT.set(env);
    ENVIRONMENT
        .get()
        .ok_or_else(|| "ONNX environment unavailable".to_string())
}

/// Background removal backed by a U²-Net-family ONNX model
/// (single 320x320 input), e.g. the u2netp model the `ai`
/// container image ships.
pub struct OnnxMaskProvider {
    session: Mutex<Session>,
    input_name: String,
}

impl OnnxMaskProvider {
    /// Load a model file against the runtime at `dylib` (or the
    /// runtime's own search when `None`). Every failure is a
    /// plain message: at boot it becomes the startup diagnostic.
    pub fn load(dylib: Option<&str>, model_path: &str) -> Result<Self, String> {
        let env = environment(dylib)?;
        let session = Session::builder(env)
            .map_err(|e| format!("cannot create a session: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("cannot configure the session: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| format!("cannot load the model {model_path}: {e}"))?;
        let input_name = session
            .inputs()
            .first()
            .map(|i| i.name().to_string())
            .ok_or_else(|| format!("the model {model_path} has no inputs"))?;
        Ok(Self {
            session: Mutex::new(session),
            input_name,
        })
    }
}

impl std::fmt::Debug for OnnxMaskProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxMaskProvider")
            .field("input_name", &self.input_name)
            .finish_non_exhaustive()
    }
}

impl MaskProvider for OnnxMaskProvider {
    fn mask(&self, img: &DynamicImage) -> Result<GrayImage, String> {
        let input = preprocess(img);
        let tensor = ort::value::Tensor::from_array(([1, 3, 320, 320], input.into_boxed_slice()))
            .map_err(|e| format!("cannot build the input tensor: {e}"))?;
        let mut session = self
            .session
            .lock()
            .map_err(|_| "mask provider lock poisoned".to_string())?;
        let outputs = session
            .run(ort::inputs![self.input_name.as_str() => tensor])
            .map_err(|e| format!("inference failed: {e}"))?;
        let (_, first) = outputs
            .into_iter()
            .next()
            .ok_or_else(|| "the model produced no outputs".to_string())?;
        let (_shape, data) = first
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("cannot read the model output: {e}"))?;
        Ok(postprocess(data, img.width(), img.height()))
    }
}

/// U²-Net preprocessing, matching the reference pipeline:
/// resize to the model's 320x320, scale by the image maximum,
/// then normalize per channel with ImageNet statistics. Output
/// is planar CHW.
fn preprocess(img: &DynamicImage) -> Vec<f32> {
    let small = img.resize_exact(320, 320, FilterType::Triangle).to_rgb8();
    let max = small.pixels().flat_map(|p| p.0).max().unwrap_or(0);
    let scale = if max == 0 { 1.0 } else { max as f32 };
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    let mut out = vec![0f32; 3 * 320 * 320];
    for (i, px) in small.pixels().enumerate() {
        for (c, (mean, std)) in MEAN.iter().zip(STD.iter()).enumerate() {
            out[c * 320 * 320 + i] = ((px[c] as f32 / scale) - mean) / std;
        }
    }
    out
}

/// Turn the model's first output into a mask: min-max
/// normalize to 0..255 (the reference pipeline's behavior),
/// then resize back to the source dimensions.
fn postprocess(data: &[f32], w: u32, h: u32) -> GrayImage {
    let (min, max) = data
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let range = (max - min).max(f32::EPSILON);
    let mask = GrayImage::from_fn(320, 320, |x, y| {
        let v = data[(y * 320 + x) as usize];
        image::Luma([((v - min) / range * 255.0).round() as u8])
    });
    if (w, h) == (320, 320) {
        mask
    } else {
        image::imageops::resize(&mask, w, h, FilterType::Triangle)
    }
}

/// Apply a foreground mask: the output alpha is the source
/// alpha scaled by the mask, so the subject stays and the
/// background becomes transparent.
pub fn apply_mask(img: &DynamicImage, mask: &GrayImage) -> DynamicImage {
    let mut rgba = img.to_rgba8();
    let fitted;
    let mask = if mask.width() == rgba.width() && mask.height() == rgba.height() {
        mask
    } else {
        fitted = image::imageops::resize(mask, rgba.width(), rgba.height(), FilterType::Triangle);
        &fitted
    };
    for (px, m) in rgba.pixels_mut().zip(mask.pixels()) {
        px[3] = ((px[3] as u16 * m[0] as u16) / 255) as u8;
    }
    DynamicImage::ImageRgba8(rgba)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(r: u8, g: u8, b: u8, w: u32, h: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(w, h, image::Rgb([r, g, b])))
    }

    #[test]
    fn apply_mask_scales_alpha_by_the_mask() {
        // Left half background (mask 0), right half subject (255).
        let mask = GrayImage::from_fn(4, 2, |x, _| image::Luma([if x < 2 { 0 } else { 255 }]));
        let out = apply_mask(&solid(200, 30, 30, 4, 2), &mask).to_rgba8();
        assert_eq!(out.get_pixel(0, 0)[3], 0);
        assert_eq!(out.get_pixel(3, 0)[3], 255);
        // Color is untouched.
        assert_eq!(out.get_pixel(3, 0)[0], 200);
    }

    #[test]
    fn apply_mask_respects_existing_alpha() {
        let mut img = DynamicImage::new_rgba8(2, 1);
        img.as_mut_rgba8()
            .unwrap()
            .put_pixel(0, 0, image::Rgba([10, 20, 30, 128]));
        let mask = GrayImage::from_pixel(2, 1, image::Luma([128]));
        let out = apply_mask(&img, &mask).to_rgba8();
        // 128 * 128 / 255 = 64.
        assert_eq!(out.get_pixel(0, 0)[3], 64);
    }

    #[test]
    fn preprocess_produces_a_normalized_chw_tensor() {
        let input = preprocess(&solid(255, 255, 255, 8, 8));
        assert_eq!(input.len(), 3 * 320 * 320);
        // White scales to 1.0 by the max, then normalizes:
        // (1 - 0.485) / 0.229 = 2.2489... in the red channel.
        assert!((input[0] - 2.2489).abs() < 0.001, "got {}", input[0]);
        // A black image must not divide by zero; its red channel
        // is (0 - 0.485) / 0.229 = -2.1179...
        let input = preprocess(&solid(0, 0, 0, 8, 8));
        assert!((input[0] - -2.1179).abs() < 0.001, "got {}", input[0]);
    }

    /// Real-model verification, run manually with the deployment
    /// artifacts in place:
    /// CRABCROP_TEST_ORT_DYLIB=... CRABCROP_TEST_REMOVEBG_MODEL=...
    /// cargo test -- --ignored removebg_with_the_real_model
    #[test]
    #[ignore = "needs the ONNX runtime and a model file (env-configured)"]
    fn removebg_with_the_real_model() {
        let dylib = std::env::var("CRABCROP_TEST_ORT_DYLIB").unwrap();
        let model = std::env::var("CRABCROP_TEST_REMOVEBG_MODEL").unwrap();
        let provider = OnnxMaskProvider::load(Some(&dylib), &model).unwrap();
        // A dark square on a light background: the model should
        // call the square the subject.
        let mut img = DynamicImage::new_rgb8(200, 200);
        for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            *p = if (60..140).contains(&x) && (60..140).contains(&y) {
                image::Rgb([30, 30, 30])
            } else {
                image::Rgb([240, 240, 240])
            };
        }
        let mask = provider.mask(&img).unwrap();
        assert_eq!((mask.width(), mask.height()), (200, 200));
        assert!(
            mask.get_pixel(100, 100)[0] > 200,
            "center (subject) mask = {}",
            mask.get_pixel(100, 100)[0]
        );
        assert!(
            mask.get_pixel(5, 5)[0] < 55,
            "corner (background) mask = {}",
            mask.get_pixel(5, 5)[0]
        );
    }
}
