//! OCR (issue #177): the text detector, the quad geometry,
//! and the CTC recognizer.

use image::DynamicImage;
use image::imageops::FilterType;
use ort::session::Session;
use std::sync::Mutex;

use super::session::{build_session, collect_outputs, fir_resize_rgb};
use super::{AiError, OcrLine, OcrProvider, OcrReport};

/// The detection stage's side limit: the image is scaled
/// so its longer side does not exceed this before the
/// detection model runs (the RapidOCR pipeline default).
const OCR_DET_LIMIT_SIDE: u32 = 960;

/// A detection-map pixel is text at this probability
/// (the DB detector's binarization threshold).
const OCR_DET_THRESHOLD: f32 = 0.3;

/// A candidate box survives when the mean probability of
/// the pixels that fired for it reaches this (the DB
/// postprocess's box threshold).
const OCR_BOX_THRESHOLD: f32 = 0.5;

/// How far a detected box grows before recognition: the
/// DB unclip distance is the box's area times this ratio
/// over its perimeter.
const OCR_UNCLIP_RATIO: f64 = 1.5;

/// The recognition model's fixed input height.
const OCR_REC_HEIGHT: u32 = 48;

/// The widest recognition input: a crop whose aspect
/// would ask for more is squeezed to this width instead,
/// which only absurd slivers ever reach.
const OCR_REC_MAX_WIDTH: u32 = 3200;

/// The most lines one response reports: a guard on meta
/// size for pathological inputs, far past any real page
/// the detector's side limit can resolve.
pub const OCR_MAX_LINES: usize = 512;

/// Read an OCR character dictionary: one character per
/// line, in the recognition model's class order. Unlike
/// the label files, nothing is trimmed or skipped: every
/// line is one output class of the model, so a stray
/// blank line would misalign the whole table, and a
/// character that is itself whitespace is data. Only the
/// line terminators come off, and one final empty line
/// from the file's trailing newline. The PP-OCRv6 file
/// this ships against was verified identical to the
/// model's own embedded table; the load probe checks the
/// count against the model's output width.
pub fn read_ocr_dict(path: &str) -> Result<Vec<String>, AiError> {
    let raw = std::fs::read_to_string(path)
        .map_err(|e| AiError::Load(format!("cannot read the OCR dictionary {path}: {e}")))?;
    let mut chars: Vec<String> = raw
        .split('\n')
        .map(|line| line.strip_suffix('\r').unwrap_or(line).to_string())
        .collect();
    if chars.last().is_some_and(|line| line.is_empty()) {
        chars.pop();
    }
    if chars.is_empty() {
        return Err(AiError::Load(format!(
            "the OCR dictionary {path} contains no characters"
        )));
    }
    Ok(chars)
}

/// The detection model's input size for an image: the
/// longer side scaled to the side limit (never up), each
/// side then rounded up to a multiple of 32, the stride
/// the detector's maps come back on.
fn ocr_det_size(img_w: u32, img_h: u32) -> (u32, u32) {
    let max_side = img_w.max(img_h);
    let ratio = if max_side > OCR_DET_LIMIT_SIDE {
        OCR_DET_LIMIT_SIDE as f64 / max_side as f64
    } else {
        1.0
    };
    let side = |v: u32| {
        let scaled = (v as f64 * ratio / 32.0).ceil() as u32 * 32;
        scaled.max(32)
    };
    (side(img_w), side(img_h))
}

/// The detection input tensor: the image resized to the
/// detector size and normalized with the ImageNet
/// statistics the PP-OCR detectors are trained with,
/// planar CHW. Returns the tensor and the size it was
/// made at.
fn ocr_det_preprocess(img: &DynamicImage) -> (Vec<f32>, u32, u32) {
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    let (rw, rh) = ocr_det_size(img.width(), img.height());
    let resized = fir_resize_rgb(&img.to_rgb8(), rw, rh);
    let plane = (rw * rh) as usize;
    let mut out = vec![0f32; 3 * plane];
    for (i, px) in resized.pixels().enumerate() {
        for c in 0..3 {
            out[c * plane + i] = (px[c] as f32 / 255.0 - MEAN[c]) / STD[c];
        }
    }
    (out, rw, rh)
}

/// The connected components of the binarized detection
/// map (4-connected, over the DB threshold), each as its
/// pixel coordinates in map space.
fn ocr_components(prob: &[f32], w: u32, h: u32) -> Vec<Vec<(u32, u32)>> {
    let (w, h) = (w as usize, h as usize);
    let mut seen = vec![false; w * h];
    let mut components = Vec::new();
    for start in 0..w * h {
        if seen[start] || prob[start] <= OCR_DET_THRESHOLD {
            continue;
        }
        let mut pixels = Vec::new();
        let mut stack = vec![start];
        seen[start] = true;
        while let Some(idx) = stack.pop() {
            let (x, y) = (idx % w, idx / w);
            pixels.push((x as u32, y as u32));
            for (nx, ny) in [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ] {
                if nx >= w || ny >= h {
                    continue;
                }
                let n = ny * w + nx;
                if !seen[n] && prob[n] > OCR_DET_THRESHOLD {
                    seen[n] = true;
                    stack.push(n);
                }
            }
        }
        components.push(pixels);
    }
    components
}

/// The convex hull of a point set (Andrew's monotone
/// chain), extreme points only: collinear points along an
/// edge are dropped, which is what the rotating calipers
/// want.
fn convex_hull(points: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let cross = |o: (f32, f32), a: (f32, f32), b: (f32, f32)| {
        (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
    };
    let mut pts: Vec<(f32, f32)> = points.to_vec();
    pts.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    });
    pts.dedup();
    if pts.len() < 3 {
        return pts;
    }
    let mut lower: Vec<(f32, f32)> = Vec::new();
    for &p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<(f32, f32)> = Vec::new();
    for &p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// The minimum-area enclosing rectangle of a convex hull
/// (rotating calipers over the hull's edges), as four
/// corners in cyclic order. `None` for a degenerate hull.
fn min_area_rect(hull: &[(f32, f32)]) -> Option<[(f32, f32); 4]> {
    if hull.len() < 3 {
        return None;
    }
    let mut best: Option<(f32, [(f32, f32); 4])> = None;
    for i in 0..hull.len() {
        let a = hull[i];
        let b = hull[(i + 1) % hull.len()];
        let (ex, ey) = (b.0 - a.0, b.1 - a.1);
        let len = (ex * ex + ey * ey).sqrt();
        if len == 0.0 {
            continue;
        }
        let (ux, uy) = (ex / len, ey / len);
        let (vx, vy) = (-uy, ux);
        let (mut min_u, mut max_u, mut min_v, mut max_v) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
        for &p in hull {
            let du = p.0 * ux + p.1 * uy;
            let dv = p.0 * vx + p.1 * vy;
            min_u = min_u.min(du);
            max_u = max_u.max(du);
            min_v = min_v.min(dv);
            max_v = max_v.max(dv);
        }
        let area = (max_u - min_u) * (max_v - min_v);
        if best.as_ref().is_none_or(|(best_area, _)| area < *best_area) {
            let corner = |du: f32, dv: f32| (du * ux + dv * vx, du * uy + dv * vy);
            best = Some((
                area,
                [
                    corner(min_u, min_v),
                    corner(max_u, min_v),
                    corner(max_u, max_v),
                    corner(min_u, max_v),
                ],
            ));
        }
    }
    best.map(|(_, rect)| rect)
}

fn quad_edge(a: (f32, f32), b: (f32, f32)) -> f64 {
    let (dx, dy) = ((b.0 - a.0) as f64, (b.1 - a.1) as f64);
    (dx * dx + dy * dy).sqrt()
}

fn quad_area(quad: &[(f32, f32); 4]) -> f64 {
    let mut sum = 0.0f64;
    for i in 0..4 {
        let a = quad[i];
        let b = quad[(i + 1) % 4];
        sum += (a.0 * b.1 - b.0 * a.1) as f64;
    }
    (sum / 2.0).abs()
}

/// Grow a quadrilateral outward by the DB unclip
/// distance (area x ratio / perimeter): each edge moves
/// along its outward normal by that distance, and the
/// new corners are the intersections of adjacent moved
/// edges. Parallel neighbors keep the moved corner.
fn unclip_quad(quad: &[(f32, f32); 4], ratio: f64) -> [(f32, f32); 4] {
    let perimeter: f64 = (0..4).map(|i| quad_edge(quad[i], quad[(i + 1) % 4])).sum();
    let dist = if perimeter > 0.0 {
        quad_area(quad) * ratio / perimeter
    } else {
        0.0
    };
    if dist <= 0.0 || !dist.is_finite() {
        return *quad;
    }
    let centroid = (
        (quad[0].0 + quad[1].0 + quad[2].0 + quad[3].0) / 4.0,
        (quad[0].1 + quad[1].1 + quad[2].1 + quad[3].1) / 4.0,
    );
    // Each edge's moved line: a point and a direction.
    let mut lines = [((0.0f32, 0.0f32), (0.0f32, 0.0f32)); 4];
    for i in 0..4 {
        let a = quad[i];
        let b = quad[(i + 1) % 4];
        let edge = ((b.0 - a.0) as f64, (b.1 - a.1) as f64);
        let elen = (edge.0 * edge.0 + edge.1 * edge.1).sqrt();
        if elen == 0.0 {
            lines[i] = ((a.0, a.1), (1.0, 0.0));
            continue;
        }
        let mut normal = (-edge.1 / elen, edge.0 / elen);
        let mid = ((a.0 + b.0) as f64 / 2.0, (a.1 + b.1) as f64 / 2.0);
        if normal.0 * (mid.0 - centroid.0 as f64) + normal.1 * (mid.1 - centroid.1 as f64) < 0.0 {
            normal = (-normal.0, -normal.1);
        }
        lines[i] = (
            (
                (a.0 as f64 + normal.0 * dist) as f32,
                (a.1 as f64 + normal.1 * dist) as f32,
            ),
            ((edge.0 / elen) as f32, (edge.1 / elen) as f32),
        );
    }
    let mut out = [(0.0f32, 0.0f32); 4];
    for i in 0..4 {
        let (p1, d1) = lines[(i + 3) % 4];
        let (p2, d2) = lines[i];
        let cross = d1.0 * d2.1 - d1.1 * d2.0;
        out[i] = if cross.abs() < 1e-6 {
            p2
        } else {
            let t = ((p2.0 - p1.0) * d2.1 - (p2.1 - p1.1) * d2.0) / cross;
            (p1.0 + d1.0 * t, p1.1 + d1.1 * t)
        };
    }
    out
}

/// Order a quadrilateral's corners: first the top-left
/// (smallest x+y), then clockwise in image coordinates,
/// so the warp reads the text upright.
fn order_quad(quad: &mut [(f32, f32); 4]) {
    let start = (0..4)
        .min_by(|&a, &b| {
            let sa = quad[a].0 + quad[a].1;
            let sb = quad[b].0 + quad[b].1;
            sa.partial_cmp(&sb)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(
                    quad[a]
                        .0
                        .partial_cmp(&quad[b].0)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        })
        .unwrap_or(0);
    quad.rotate_left(start);
    let u = (quad[1].0 - quad[0].0, quad[1].1 - quad[0].1);
    let v = (quad[3].0 - quad[0].0, quad[3].1 - quad[0].1);
    if u.0 * v.1 - u.1 * v.0 < 0.0 {
        quad.swap(1, 3);
    }
}

/// The DB postprocess: from a detection probability map
/// to text quadrilaterals in image coordinates, in
/// reading order. Mirrors the reference pipeline
/// (RapidOCR's DB postprocess at its defaults): binarize,
/// take each component's minimum-area rectangle, drop
/// weak or tiny boxes, unclip, then sort the way a reader
/// scans (top to bottom, same-line boxes left to right,
/// the reference's own 10-pixel line tolerance).
fn ocr_det_quads(
    prob: &[f32],
    map_w: u32,
    map_h: u32,
    img_w: u32,
    img_h: u32,
) -> Vec<[(f32, f32); 4]> {
    let mut quads = Vec::new();
    for component in ocr_components(prob, map_w, map_h) {
        if component.len() < 4 {
            continue;
        }
        let sum: f32 = component
            .iter()
            .map(|&(x, y)| prob[(y * map_w + x) as usize])
            .sum();
        if sum / (component.len() as f32) < OCR_BOX_THRESHOLD {
            continue;
        }
        let points: Vec<(f32, f32)> = component
            .iter()
            .map(|&(x, y)| (x as f32, y as f32))
            .collect();
        let hull = convex_hull(&points);
        let Some(rect) = min_area_rect(&hull) else {
            continue;
        };
        let short_side = quad_edge(rect[0], rect[1]).min(quad_edge(rect[1], rect[2]));
        if short_side < 3.0 {
            continue;
        }
        let mut quad = unclip_quad(&rect, OCR_UNCLIP_RATIO);
        for p in &mut quad {
            p.0 = p.0.clamp(0.0, (map_w - 1) as f32) * img_w as f32 / map_w as f32;
            p.1 = p.1.clamp(0.0, (map_h - 1) as f32) * img_h as f32 / map_h as f32;
        }
        order_quad(&mut quad);
        // A box spanning essentially the whole frame is
        // the detector firing on a smooth background (a
        // gradient reads as one giant component), never a
        // text line: a real line keeps margins on some
        // side. Dropped, like the weak and tiny boxes.
        let (qx0, qy0, qx1, qy1) = quad.iter().fold(
            (f32::MAX, f32::MAX, f32::MIN, f32::MIN),
            |(x0, y0, x1, y1), p| (x0.min(p.0), y0.min(p.1), x1.max(p.0), y1.max(p.1)),
        );
        if ((qx1 - qx0) * (qy1 - qy0)) * 100.0 >= (img_w as f32 * img_h as f32) * 95.0 {
            continue;
        }
        quads.push(quad);
    }
    quads.sort_by(|a, b| {
        a[0].1
            .partial_cmp(&b[0].1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                a[0].0
                    .partial_cmp(&b[0].0)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });
    for i in (1..quads.len()).rev() {
        for j in 0..i {
            if (quads[j + 1][0].1 - quads[j][0].1).abs() < 10.0 && quads[j + 1][0].0 < quads[j][0].0
            {
                quads.swap(j, j + 1);
            }
        }
    }
    quads
}

/// Solve the homography mapping `from` corners to `to`
/// corners (both in order), as a row-major 3x3 with the
/// last element fixed at 1. Gaussian elimination with
/// partial pivoting; `None` for a degenerate
/// correspondence.
fn homography(from: &[(f32, f32); 4], to: &[(f32, f32); 4]) -> Option<[f64; 9]> {
    let mut a = [[0.0f64; 9]; 8];
    for i in 0..4 {
        let (x, y) = (from[i].0 as f64, from[i].1 as f64);
        let (tx, ty) = (to[i].0 as f64, to[i].1 as f64);
        a[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -tx * x, -tx * y, tx];
        a[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -ty * x, -ty * y, ty];
    }
    for col in 0..8 {
        let pivot = (col..8).max_by(|&r1, &r2| {
            a[r1][col]
                .abs()
                .partial_cmp(&a[r2][col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        let div = a[col][col];
        for v in &mut a[col][col..] {
            *v /= div;
        }
        let prow = a[col];
        for (row, r) in a.iter_mut().enumerate() {
            if row != col && r[col] != 0.0 {
                let factor = r[col];
                for (j, v) in r.iter_mut().enumerate().skip(col) {
                    *v -= factor * prow[j];
                }
            }
        }
    }
    Some([
        a[0][8], a[1][8], a[2][8], a[3][8], a[4][8], a[5][8], a[6][8], a[7][8], 1.0,
    ])
}

fn bilinear(img: &image::RgbImage, sx: f32, sy: f32) -> image::Rgb<u8> {
    let (w, h) = (img.width(), img.height());
    let sx = sx.clamp(0.0, (w - 1) as f32);
    let sy = sy.clamp(0.0, (h - 1) as f32);
    let (x0, y0) = (sx.floor() as u32, sy.floor() as u32);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = (sx - x0 as f32, sy - y0 as f32);
    let corners = [
        img.get_pixel(x0, y0).0,
        img.get_pixel(x1, y0).0,
        img.get_pixel(x0, y1).0,
        img.get_pixel(x1, y1).0,
    ];
    let mut out = [0u8; 3];
    for (c, slot) in out.iter_mut().enumerate() {
        let top = corners[0][c] as f32 * (1.0 - fx) + corners[1][c] as f32 * fx;
        let bottom = corners[2][c] as f32 * (1.0 - fx) + corners[3][c] as f32 * fx;
        *slot = (top * (1.0 - fy) + bottom * fy).round() as u8;
    }
    image::Rgb(out)
}

/// Rectify a text quadrilateral into an upright crop:
/// the perspective warp the recognition model reads.
/// The output's far corners are the quad's corners, so
/// the quad's own edge lengths set the crop size.
fn warp_quad(
    img: &image::RgbImage,
    quad: &[(f32, f32); 4],
    out_w: u32,
    out_h: u32,
) -> Option<image::RgbImage> {
    let dst = [
        (0.0, 0.0),
        (out_w as f32, 0.0),
        (out_w as f32, out_h as f32),
        (0.0, out_h as f32),
    ];
    // Solve destination -> source, so each output pixel
    // samples where it came from.
    let m = homography(&dst, quad)?;
    let mut out = image::RgbImage::new(out_w, out_h);
    for y in 0..out_h {
        for x in 0..out_w {
            let (xf, yf) = (x as f64, y as f64);
            let d = m[6] * xf + m[7] * yf + 1.0;
            let sx = ((m[0] * xf + m[1] * yf + m[2]) / d) as f32;
            let sy = ((m[3] * xf + m[4] * yf + m[5]) / d) as f32;
            out.put_pixel(x, y, bilinear(img, sx, sy));
        }
    }
    Some(out)
}

/// Rotate an image 90 degrees counterclockwise: the
/// rectification step for vertical text (a crop much
/// taller than wide is read sideways, the reference
/// pipeline's rule).
fn rotate_ccw(img: &image::RgbImage) -> image::RgbImage {
    let (w, h) = (img.width(), img.height());
    let mut out = image::RgbImage::new(h, w);
    for y in 0..h {
        for x in 0..w {
            out.put_pixel(y, w - 1 - x, *img.get_pixel(x, y));
        }
    }
    out
}

/// The recognition input for one crop: resized to the
/// model's fixed height keeping aspect (width capped),
/// normalized to -1..1, planar CHW, with its shape.
fn ocr_rec_preprocess(crop: &image::RgbImage) -> (Vec<f32>, [usize; 4]) {
    let width = ((OCR_REC_HEIGHT as f64 * crop.width() as f64 / crop.height() as f64).ceil()
        as u32)
        .clamp(1, OCR_REC_MAX_WIDTH);
    let resized = image::imageops::resize(crop, width, OCR_REC_HEIGHT, FilterType::Triangle);
    let plane = (width * OCR_REC_HEIGHT) as usize;
    let mut out = vec![0f32; 3 * plane];
    for (i, px) in resized.pixels().enumerate() {
        for c in 0..3 {
            out[c * plane + i] = (px[c] as f32 / 255.0 - 0.5) / 0.5;
        }
    }
    (out, [1, 3, OCR_REC_HEIGHT as usize, width as usize])
}

/// Decode one recognition output (CTC): greedy argmax
/// per step, repeats collapsed, blanks dropped. The
/// class table is the dictionary with blank at 0 and a
/// space appended last, the PP-OCR layout the dictionary
/// contract fixes. The score is the mean probability of
/// the steps that produced a character. `None` when the
/// crop read as no characters at all: a detection with
/// nothing legible in it is not a text line.
fn ocr_ctc_decode(
    probs: &[f32],
    steps: usize,
    classes: usize,
    dict: &[String],
) -> Option<(String, f32)> {
    let mut text = String::new();
    let (mut conf_sum, mut conf_count) = (0.0f32, 0usize);
    let mut prev = usize::MAX;
    for t in 0..steps {
        let row = &probs[t * classes..(t + 1) * classes];
        let (idx, &conf) = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))?;
        if idx != 0 && idx != prev {
            if idx <= dict.len() {
                text.push_str(&dict[idx - 1]);
            } else {
                text.push(' ');
            }
            conf_sum += conf;
            conf_count += 1;
        }
        prev = idx;
    }
    if text.is_empty() {
        None
    } else {
        Some((text, conf_sum / conf_count as f32))
    }
}

/// OCR backed by a PP-OCR-family ONNX model pair
/// (issue #177): a DB detection model (image in,
/// probability map out) and a CTC recognition model
/// (48-pixel-high crops in, class probabilities out),
/// plus the recognition model's character dictionary as
/// operator configuration. The reference artifacts are
/// the RapidOCR exports of PP-OCRv6 small (Apache-2.0).
/// Detection runs once per image; recognition once per
/// detected line.
pub struct OnnxOcrProvider {
    detect: Mutex<Session>,
    recognize: Mutex<Session>,
    detect_input: String,
    recognize_input: String,
    dict: Vec<String>,
    model: String,
    language: String,
}

impl OnnxOcrProvider {
    /// Load the model pair and the dictionary. Both
    /// models are probed before returning: detection
    /// runs on a blank image and recognition on a blank
    /// crop, and the recognition output width is checked
    /// against the dictionary (one class per character,
    /// plus blank and space). The two probes are
    /// separate because detection alone cannot prove the
    /// pair: a blank image yields no crops, so a full
    /// pipeline run would never reach the recognizer,
    /// and a mismatched dictionary would slip through to
    /// the first real request.
    pub fn load(
        dylib: Option<&str>,
        detect_path: &str,
        recognize_path: &str,
        dict_path: &str,
        language: &str,
    ) -> Result<Self, AiError> {
        let dict = read_ocr_dict(dict_path)?;
        let (detect, detect_input) = build_session(dylib, detect_path)?;
        let (recognize, recognize_input) = build_session(dylib, recognize_path)?;
        let provider = Self {
            detect: Mutex::new(detect),
            recognize: Mutex::new(recognize),
            detect_input,
            recognize_input,
            dict,
            // Meta reports the model label, never the
            // operator's filesystem layout (issue #268):
            // the file name identifies the model without
            // disclosing where it lives.
            model: std::path::Path::new(recognize_path)
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| "ocr".to_string()),
            language: language.to_string(),
        };
        let probe = DynamicImage::new_rgb8(64, 64);
        provider.detect_probs(&probe).map_err(|e| {
            AiError::Load(format!(
                "the OCR models {detect_path} / {recognize_path} failed their load probe: {e}"
            ))
        })?;
        let crop = image::RgbImage::new(96, OCR_REC_HEIGHT);
        provider.recognize_crop(&crop).map_err(|e| {
            AiError::Load(format!(
                "the OCR models {detect_path} / {recognize_path} failed their load probe: {e}"
            ))
        })?;
        Ok(provider)
    }

    /// The recognized-line count cap and dictionary size
    /// are boot facts worth reporting beside the model.
    pub fn dict_len(&self) -> usize {
        self.dict.len()
    }

    fn detect_probs(&self, img: &DynamicImage) -> Result<(Vec<f32>, u32, u32), AiError> {
        let (input, rw, rh) = ocr_det_preprocess(img);
        let tensor = ort::value::Tensor::from_array((
            [1usize, 3, rh as usize, rw as usize],
            input.into_boxed_slice(),
        ))
        .map_err(|e| AiError::Inference(format!("cannot build the detection tensor: {e}")))?;
        let mut session = self.detect.lock().map_err(|_| AiError::LockPoisoned)?;
        let outputs = session
            .run(ort::inputs![self.detect_input.as_str() => tensor])
            .map_err(|e| AiError::Inference(format!("detection failed: {e}")))?;
        let outputs = collect_outputs(outputs)?;
        let (_, dims, data) = outputs.first().ok_or_else(|| {
            AiError::BadOutput("the detection model produced no outputs".to_string())
        })?;
        if dims.len() != 4
            || dims[0] != 1
            || dims[1] != 1
            || dims[2] != rh as i64
            || dims[3] != rw as i64
        {
            return Err(AiError::BadOutput(format!(
                "unexpected detection output shape: {dims:?}"
            )));
        }
        Ok((data.clone(), rw, rh))
    }

    fn recognize_crop(&self, crop: &image::RgbImage) -> Result<Option<(String, f32)>, AiError> {
        let (input, shape) = ocr_rec_preprocess(crop);
        let tensor = ort::value::Tensor::from_array((shape, input.into_boxed_slice()))
            .map_err(|e| AiError::Inference(format!("cannot build the recognition tensor: {e}")))?;
        let mut session = self.recognize.lock().map_err(|_| AiError::LockPoisoned)?;
        let outputs = session
            .run(ort::inputs![self.recognize_input.as_str() => tensor])
            .map_err(|e| AiError::Inference(format!("recognition failed: {e}")))?;
        let outputs = collect_outputs(outputs)?;
        let (_, dims, data) = outputs.first().ok_or_else(|| {
            AiError::BadOutput("the recognition model produced no outputs".to_string())
        })?;
        if dims.len() != 3 || dims[0] != 1 {
            return Err(AiError::BadOutput(format!(
                "unexpected recognition output shape: {dims:?}"
            )));
        }
        let (steps, classes) = (dims[1] as usize, dims[2] as usize);
        if classes != self.dict.len() + 2 {
            return Err(AiError::BadOutput(format!(
                "the recognition model produced {classes} classes for {} dictionary \
                 characters (plus blank and space): the dictionary does not match the model",
                self.dict.len()
            )));
        }
        Ok(ocr_ctc_decode(data, steps, classes, &self.dict))
    }
}

impl std::fmt::Debug for OnnxOcrProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxOcrProvider")
            .field("dict", &self.dict.len())
            .field("model", &self.model)
            .field("language", &self.language)
            .finish_non_exhaustive()
    }
}

impl OnnxOcrProvider {
    /// The recognition walk, with an optional
    /// deadline: detection runs once, then each
    /// detected line costs a recognition run, so the
    /// deadline is checked between lines and the
    /// report carries the lines gathered so far.
    fn recognize_within(
        &self,
        img: &DynamicImage,
        deadline: Option<std::time::Instant>,
    ) -> Result<OcrReport, AiError> {
        let (prob, map_w, map_h) = self.detect_probs(img)?;
        let quads = ocr_det_quads(&prob, map_w, map_h, img.width(), img.height());
        let rgb = img.to_rgb8();
        let mut lines = Vec::new();
        for quad in &quads {
            if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                break;
            }
            if lines.len() >= OCR_MAX_LINES {
                break;
            }
            let out_w = quad_edge(quad[0], quad[1])
                .max(quad_edge(quad[2], quad[3]))
                .round() as u32;
            let out_h = quad_edge(quad[0], quad[3])
                .max(quad_edge(quad[1], quad[2]))
                .round() as u32;
            if out_w == 0 || out_h == 0 {
                continue;
            }
            let Some(mut crop) = warp_quad(&rgb, quad, out_w, out_h) else {
                continue;
            };
            if crop.height() * 2 >= crop.width() * 3 {
                crop = rotate_ccw(&crop);
            }
            let Some((text, score)) = self.recognize_crop(&crop)? else {
                continue;
            };
            let min_x = quad.iter().map(|p| p.0).fold(f32::MAX, f32::min);
            let min_y = quad.iter().map(|p| p.1).fold(f32::MAX, f32::min);
            let max_x = quad.iter().map(|p| p.0).fold(f32::MIN, f32::max);
            let max_y = quad.iter().map(|p| p.1).fold(f32::MIN, f32::max);
            let x = (min_x.floor() as u32).min(img.width());
            let y = (min_y.floor() as u32).min(img.height());
            let w = ((max_x.ceil() as u32).min(img.width())).saturating_sub(x);
            let h = ((max_y.ceil() as u32).min(img.height())).saturating_sub(y);
            lines.push(OcrLine {
                text,
                score,
                x,
                y,
                w,
                h,
            });
        }
        Ok(OcrReport {
            model: self.model.clone(),
            language: self.language.clone(),
            lines,
        })
    }
}

impl OcrProvider for OnnxOcrProvider {
    fn recognize(&self, img: &DynamicImage) -> Result<OcrReport, AiError> {
        self.recognize_within(img, None)
    }

    fn recognize_deadline(
        &self,
        img: &DynamicImage,
        deadline: std::time::Instant,
    ) -> Result<OcrReport, AiError> {
        self.recognize_within(img, Some(deadline))
    }
}

#[cfg(test)]
mod tests;
