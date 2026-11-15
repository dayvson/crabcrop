use super::*;
use image::Luma;

/// Deterministic noise: smooth ramps would hide a pixel
/// read from the wrong place.
fn noise(len: usize, seed: u32) -> Vec<u8> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 24) as u8
        })
        .collect()
}

fn noise_rgb(w: u32, h: u32, seed: u32) -> DynamicImage {
    let raw = noise((w * h * 3) as usize, seed);
    DynamicImage::ImageRgb8(image::RgbImage::from_raw(w, h, raw).unwrap())
}

fn noise_rgba(w: u32, h: u32, seed: u32) -> RgbaImage {
    RgbaImage::from_raw(w, h, noise((w * h * 4) as usize, seed)).unwrap()
}

/// A mask of noise in which about a third of the pixels are
/// uncovered, a third fully covered, and the rest soft.
fn soft_mask(w: u32, h: u32, seed: u32) -> GrayImage {
    let raw = noise((w * h) as usize, seed)
        .into_iter()
        .map(|v| match v % 3 {
            0 => 0,
            1 => 255,
            _ => v,
        })
        .collect();
    GrayImage::from_raw(w, h, raw).unwrap()
}

/// A subject in the middle of the frame: covered inside an
/// ellipse, a soft ramp around it, uncovered beyond.
fn blob_mask(w: u32, h: u32) -> GrayImage {
    GrayImage::from_fn(w, h, |x, y| {
        let dx = (x as f32 + 0.5 - w as f32 / 2.0) / (w as f32 * 0.3);
        let dy = (y as f32 + 0.5 - h as f32 / 2.0) / (h as f32 * 0.3);
        let d = (dx * dx + dy * dy).sqrt();
        Luma([((1.25 - d) * 4.0 * 255.0).clamp(0.0, 255.0) as u8])
    })
}

/// `mask_bbox` as it was: every pixel with its coordinates.
fn mask_bbox_by_pixels(mask: &GrayImage, threshold: u8) -> Option<(u32, u32, u32, u32)> {
    let (mut min_x, mut min_y) = (u32::MAX, u32::MAX);
    let (mut max_x, mut max_y) = (0u32, 0u32);
    let mut any = false;
    for (x, y, px) in mask.enumerate_pixels() {
        if px[0] >= threshold {
            any = true;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }
    any.then(|| (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1))
}

/// `defringe` as it was: a copy of the frame in, the grid's
/// nine cells added up again for every edge pixel.
fn defringe_by_pixels(img: &DynamicImage, mask: &GrayImage) -> DynamicImage {
    let rgba = img.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());
    let (mut total, mut count) = ([0u64; 3], 0u64);
    for (px, m) in rgba.pixels().zip(mask.pixels()) {
        if m[0] == 0 {
            for c in 0..3 {
                total[c] += px[c] as u64;
            }
            count += 1;
        }
    }
    if count == 0 {
        return img.clone();
    }
    let global = [
        (total[0] / count) as f32,
        (total[1] / count) as f32,
        (total[2] / count) as f32,
    ];
    let cell = (w.max(h) / 64).max(1);
    let (gw, gh) = (w.div_ceil(cell), h.div_ceil(cell));
    let mut sums = vec![[0u64; 3]; (gw * gh) as usize];
    let mut counts = vec![0u64; (gw * gh) as usize];
    for (x, y, px) in rgba.enumerate_pixels() {
        if mask.get_pixel(x, y)[0] == 0 {
            let i = (y / cell * gw + x / cell) as usize;
            for c in 0..3 {
                sums[i][c] += px[c] as u64;
            }
            counts[i] += 1;
        }
    }
    let local_bg = |x: u32, y: u32| -> [f32; 3] {
        let (cx, cy) = (x / cell, y / cell);
        let (mut s, mut n) = ([0u64; 3], 0u64);
        for gy in cy.saturating_sub(1)..=(cy + 1).min(gh - 1) {
            for gx in cx.saturating_sub(1)..=(cx + 1).min(gw - 1) {
                let i = (gy * gw + gx) as usize;
                for c in 0..3 {
                    s[c] += sums[i][c];
                }
                n += counts[i];
            }
        }
        if n == 0 {
            global
        } else {
            [
                s[0] as f32 / n as f32,
                s[1] as f32 / n as f32,
                s[2] as f32 / n as f32,
            ]
        }
    };
    let mut out = rgba;
    for (x, y, px) in out.enumerate_pixels_mut() {
        let m = mask.get_pixel(x, y)[0];
        if m == 0 || m == 255 {
            continue;
        }
        let a = m as f32 / 255.0;
        let divisor = a.max(0.2);
        let bg = local_bg(x, y);
        for c in 0..3 {
            let fg = (px[c] as f32 - (1.0 - a) * bg[c]) / divisor;
            px[c] = fg.round().clamp(0.0, 255.0) as u8;
        }
    }
    DynamicImage::ImageRgba8(out)
}

/// `place` as it was: the image crate resizes the piece.
fn place_by_the_crate(
    piece: &RgbaImage,
    subject: (u32, u32, u32, u32),
    canvas: (u32, u32),
    pad: u8,
) -> (RgbaImage, i64, i64) {
    let (w, h) = canvas;
    let (sx, sy, sw, sh) = subject;
    let keep = (100.0 - 2.0 * pad as f32) / 100.0;
    let (iw, ih) = (w as f32 * keep, h as f32 * keep);
    let fit = (iw / sw as f32).min(ih / sh as f32);
    let by_axis =
        (2.0 * w as f32 / piece.width() as f32).min(2.0 * h as f32 / piece.height() as f32);
    let piece_area = piece.width() as f32 * piece.height() as f32;
    let by_area = (crate::engine::MAX_OUTPUT_PIXELS as f32 / piece_area).sqrt();
    let scale = fit.min(by_axis).min(by_area);
    let scaled = image::imageops::resize(
        piece,
        ((piece.width() as f32 * scale).round() as u32).max(1),
        ((piece.height() as f32 * scale).round() as u32).max(1),
        FilterType::Triangle,
    );
    let bcx = (sx as f32 + sw as f32 / 2.0) * scale;
    let bcy = (sy as f32 + sh as f32 / 2.0) * scale;
    let dx = (w as f32 / 2.0 - bcx).round() as i64;
    let dy = (h as f32 / 2.0 - bcy).round() as i64;
    (scaled, dx, dy)
}

/// The shadow as it was: an RGBA layer the size of the
/// canvas, for the image crate to overlay.
fn shadow_layer(
    alpha: &GrayImage,
    at: (i64, i64),
    canvas: (u32, u32),
    shadow: StudioShadow,
    deadline: Option<std::time::Instant>,
) -> Result<RgbaImage, DeadlinePassed> {
    let (w, h) = canvas;
    let min_dim = w.min(h) as f32;
    let (sigma, peak) = match shadow {
        StudioShadow::Soft => ((min_dim * 0.02).max(1.0), 0.35),
        StudioShadow::Strong => ((min_dim * 0.006).max(1.0), 0.55),
        StudioShadow::None => unreachable!("shadow layers are only built for soft or strong"),
    };
    let blurred = crate::filters::blur_gray_within(alpha, sigma, deadline)?;
    let drop = ((h as f32 * 0.01).round() as i64).max(1);
    let mut layer = RgbaImage::from_pixel(w, h, Rgba([0, 0, 0, 0]));
    for (x, y, px) in blurred.enumerate_pixels() {
        let a = (px[0] as f32 / 255.0 * peak * 255.0).round() as u8;
        if a == 0 {
            continue;
        }
        let (tx, ty) = (x as i64 + at.0, y as i64 + at.1 + drop);
        if tx >= 0 && ty >= 0 && tx < w as i64 && ty < h as i64 {
            layer.put_pixel(tx as u32, ty as u32, Rgba([0, 0, 0, a]));
        }
    }
    Ok(layer)
}

/// The backdrop with its shadow as it was: the layer, laid
/// on the plain canvas by the image crate's overlay.
fn backdrop_under_a_layer(
    canvas: (u32, u32),
    backdrop: [u8; 3],
    alpha: &GrayImage,
    at: (i64, i64),
    shadow: StudioShadow,
) -> RgbaImage {
    let (w, h) = canvas;
    let mut canvas =
        RgbaImage::from_pixel(w, h, Rgba([backdrop[0], backdrop[1], backdrop[2], 255]));
    let layer = shadow_layer(alpha, at, (w, h), shadow, None).unwrap();
    image::imageops::overlay(&mut canvas, &layer, 0, 0);
    canvas
}

/// The product composite as it was, from the steps above
/// and the image crate's overlay.
fn composite_by_layers(
    img: &DynamicImage,
    mask: &GrayImage,
    spec: &StudioSpec,
    brand_color: Option<[u8; 3]>,
) -> StudioOutcome {
    let (w, h) = (img.width(), img.height());
    let resolved = resolve_bg(spec, brand_color);
    let mut fallbacks: Vec<&'static str> = Vec::new();
    if resolved.brand_fallback {
        fallbacks.push("bg_brand_unset");
    }
    let mut steps: Vec<&'static str> = vec!["mask"];
    let fitted;
    let mask = if mask.width() == w && mask.height() == h {
        mask
    } else {
        fitted = image::imageops::resize(mask, w, h, FilterType::Triangle);
        &fitted
    };
    let subject = match (
        mask_bbox_by_pixels(mask, SUBJECT_THRESHOLD),
        mask_bbox_by_pixels(mask, 1),
    ) {
        (Some(bbox), Some(region)) if !degenerate_subject(bbox, region) => Some((bbox, region)),
        _ => None,
    };
    let (piece, subject_in_piece, shadow_ok) = match subject {
        Some((bbox, region)) => {
            steps.push("defringe");
            let masked = crate::ai::apply_mask(img, mask);
            let clean = defringe_by_pixels(&masked, mask).to_rgba8();
            let (rx, ry, rw, rh) = region;
            let crop = image::imageops::crop_imm(&clean, rx, ry, rw, rh).to_image();
            (
                crop,
                (bbox.0 - rx, bbox.1 - ry, bbox.2, bbox.3),
                spec.shadow != StudioShadow::None,
            )
        }
        None => {
            steps.push("fallback");
            fallbacks.push("no_subject");
            (img.to_rgba8(), (0, 0, w, h), false)
        }
    };
    steps.push("backdrop");
    let (scaled, dx, dy) = place_by_the_crate(&piece, subject_in_piece, (w, h), spec.pad);
    let mut canvas = match resolved.color {
        Some(c) => RgbaImage::from_pixel(w, h, Rgba([c[0], c[1], c[2], 255])),
        None => RgbaImage::from_pixel(w, h, Rgba([0, 0, 0, 0])),
    };
    if shadow_ok && resolved.color.is_some() {
        let alpha = GrayImage::from_fn(scaled.width(), scaled.height(), |x, y| {
            Luma([scaled.get_pixel(x, y)[3]])
        });
        let layer = shadow_layer(&alpha, (dx, dy), (w, h), spec.shadow, None).unwrap();
        image::imageops::overlay(&mut canvas, &layer, 0, 0);
        steps.push("shadow");
    }
    image::imageops::overlay(&mut canvas, &scaled, dx, dy);
    steps.push("pad");
    if resolved.color.is_some() {
        for px in canvas.pixels_mut() {
            px[3] = 255;
        }
    }
    StudioOutcome {
        image: DynamicImage::ImageRgba8(canvas),
        record: StudioRecord {
            mode: StudioMode::Product,
            bg: resolved.color,
            bg_from: resolved.from,
            shadow: spec.shadow,
            pad: spec.pad,
            steps,
            fallbacks,
            confidence: None,
            scene: None,
        },
    }
}

#[test]
fn mask_bbox_matches_the_per_pixel_scan() {
    // Empty masks, down to no pixels at all.
    for (w, h) in [(0, 0), (0, 5), (5, 0), (1, 1), (7, 3)] {
        let empty = GrayImage::new(w, h);
        for threshold in [0, 1, 128, 255] {
            assert_eq!(
                mask_bbox(&empty, threshold),
                mask_bbox_by_pixels(&empty, threshold),
                "empty {w}x{h} at {threshold}"
            );
        }
    }
    assert_eq!(mask_bbox(&GrayImage::new(7, 3), 1), None);
    // One pixel, in each corner and inside.
    for (x, y) in [(0, 0), (8, 0), (0, 4), (8, 4), (3, 2)] {
        let mut one = GrayImage::new(9, 5);
        one.put_pixel(x, y, Luma([200]));
        assert_eq!(mask_bbox(&one, 128), Some((x, y, 1, 1)));
        for threshold in [1, 128, 200, 201, 255] {
            assert_eq!(
                mask_bbox(&one, threshold),
                mask_bbox_by_pixels(&one, threshold),
                "one pixel at {x},{y} at {threshold}"
            );
        }
    }
    // A full frame.
    let full = GrayImage::from_pixel(9, 5, Luma([255]));
    assert_eq!(mask_bbox(&full, 255), Some((0, 0, 9, 5)));
    assert_eq!(mask_bbox(&full, 255), mask_bbox_by_pixels(&full, 255));
    // Noise, soft masks and a blob, at every threshold the
    // composite and the scene use and the two ends.
    let masks = [
        GrayImage::from_raw(31, 17, noise(31 * 17, 3)).unwrap(),
        soft_mask(64, 48, 5),
        blob_mask(120, 90),
        blob_mask(1, 1),
        // A buffer longer than the frame: what follows the
        // frame is not the frame.
        GrayImage::from_raw(4, 3, [vec![0; 12], vec![255; 4]].concat()).unwrap(),
    ];
    for mask in &masks {
        for threshold in [0, 1, 2, 127, 128, 129, 254, 255] {
            assert_eq!(
                mask_bbox(mask, threshold),
                mask_bbox_by_pixels(mask, threshold),
                "{:?} at {threshold}",
                mask.dimensions()
            );
        }
    }
}

#[test]
fn defringe_matches_the_per_pixel_form() {
    // One pixel, a few pixels, the largest frame with
    // one-pixel cells, and frames whose cells are 2 and 3
    // pixels with a last column and row of cells cut short.
    for (w, h) in [(1, 1), (5, 3), (64, 64), (129, 70), (200, 33)] {
        let masks = [
            (
                "covers everything",
                GrayImage::from_pixel(w, h, Luma([255])),
            ),
            ("covers nothing", GrayImage::new(w, h)),
            ("soft with nothing uncovered", {
                let mut m = soft_mask(w, h, w + h);
                m.pixels_mut().for_each(|p| p[0] = p[0].max(1));
                m
            }),
            ("soft noise", soft_mask(w, h, w * 3 + h)),
            ("a blob", blob_mask(w, h)),
        ];
        for (what, mask) in &masks {
            // The cut-out as the composite makes it, from an
            // opaque frame and from one with alpha of its
            // own.
            for source in [
                noise_rgb(w, h, w * 7 + h),
                DynamicImage::ImageRgba8(noise_rgba(w, h, w * 11 + h)),
            ] {
                let masked = crate::ai::apply_mask(&source, mask);
                let want = defringe_by_pixels(&masked, mask).to_rgba8();
                let got = defringe(masked.into_rgba8(), mask);
                assert_eq!(got.dimensions(), want.dimensions(), "{what} {w}x{h}");
                assert_eq!(got.as_raw(), want.as_raw(), "{what} {w}x{h}");
            }
        }
    }
}

#[test]
fn overlay_rgba_matches_the_crate_overlay() {
    let mut mixed = noise_rgba(7, 5, 21);
    for (i, px) in mixed.pixels_mut().enumerate() {
        // A third transparent, a third opaque, a third soft.
        px[3] = match i % 3 {
            0 => 0,
            1 => 255,
            _ => px[3],
        };
    }
    let tops = [
        mixed,
        noise_rgba(20, 15, 22),
        noise_rgba(1, 1, 23),
        RgbaImage::new(0, 0),
        RgbaImage::from_pixel(4, 4, Rgba([9, 8, 7, 255])),
    ];
    let bottoms = [
        RgbaImage::from_pixel(13, 9, Rgba([250, 240, 230, 255])),
        RgbaImage::new(13, 9),
        noise_rgba(13, 9, 24),
    ];
    // Inside, off each edge, off two edges, touching an
    // edge from outside, one column or row in, nowhere
    // near, and the ends of the range.
    let offsets = [
        (0, 0),
        (3, 2),
        (-3, -2),
        (10, -3),
        (-4, 6),
        (12, 8),
        (13, 0),
        (14, 0),
        (0, 9),
        (0, 10),
        (-7, 0),
        (-6, 0),
        (0, -5),
        (0, -4),
        (-19, -14),
        (500, 500),
        (i64::MAX, 0),
        (i64::MIN, 0),
        (0, i64::MAX),
        (0, i64::MIN),
    ];
    for bottom in &bottoms {
        for top in &tops {
            for (x, y) in offsets {
                let mut got = bottom.clone();
                overlay_rgba(&mut got, top, x, y);
                let mut want = bottom.clone();
                image::imageops::overlay(&mut want, top, x, y);
                assert_eq!(
                    got.as_raw(),
                    want.as_raw(),
                    "a {:?} top at {x},{y}",
                    top.dimensions()
                );
            }
        }
    }
}

#[test]
fn the_shadow_matches_the_layer_and_overlay_form() {
    // Both strengths, on white, black and a mid color, for
    // a soft blob, a solid plane and noise, placed inside
    // the canvas, off each of its edges and wholly outside
    // it.
    let alphas = [
        blob_mask(20, 14),
        GrayImage::from_pixel(9, 6, Luma([255])),
        GrayImage::from_raw(17, 11, noise(17 * 11, 31)).unwrap(),
    ];
    let compare = |canvas: (u32, u32), backdrop, shadow, alpha: &GrayImage, at| {
        let got = shadowed_backdrop(canvas, backdrop, alpha, at, shadow, None).unwrap();
        let want = backdrop_under_a_layer(canvas, backdrop, alpha, at, shadow);
        assert_eq!(got.dimensions(), want.dimensions());
        assert_eq!(
            got.as_raw(),
            want.as_raw(),
            "{shadow:?} on {backdrop:?}, {canvas:?} canvas, at {at:?}"
        );
    };
    for canvas in [(60, 40), (33, 57)] {
        let (w, h) = (canvas.0 as i64, canvas.1 as i64);
        for backdrop in [[255, 255, 255], [0, 0, 0], [120, 90, 200]] {
            for shadow in [StudioShadow::Soft, StudioShadow::Strong] {
                for alpha in &alphas {
                    let (aw, ah) = (alpha.width() as i64, alpha.height() as i64);
                    for at in [
                        (5, 4),
                        (-aw / 2, 3),
                        (w - aw / 2, 3),
                        (5, -ah / 2),
                        (5, h - ah / 2),
                        (-aw / 2, h - ah / 2),
                        (-aw - 3, 0),
                        (w + 3, 0),
                        (0, -ah - 9),
                        (0, h + 3),
                    ] {
                        compare(canvas, backdrop, shadow, alpha, at);
                    }
                }
            }
        }
    }
    // A canvas large enough for a soft shadow's sigma to
    // pass its floor (4 here), and a drop of 2 rows.
    let large = blob_mask(90, 60);
    for shadow in [StudioShadow::Soft, StudioShadow::Strong] {
        for at in [(40, 30), (-30, 160), (250, -20)] {
            compare((300, 200), [250, 240, 230], shadow, &large, at);
        }
    }
    // The shadow darkened something: the comparison above
    // is not two plain backdrops.
    let shaded = shadowed_backdrop(
        (60, 40),
        [255, 255, 255],
        &alphas[1],
        (5, 4),
        StudioShadow::Strong,
        None,
    )
    .unwrap();
    assert!(shaded.pixels().any(|p| p[0] < 200));
    // Issue #614: the blur still stops at the deadline.
    let past = Some(std::time::Instant::now());
    assert_eq!(
        shadowed_backdrop(
            (60, 40),
            [255, 255, 255],
            &alphas[0],
            (5, 4),
            StudioShadow::Soft,
            past
        )
        .map(|_| ()),
        Err(DeadlinePassed)
    );
}

/// A piece's size, the subject box in it, and the canvas.
type PlaceCase = ((u32, u32), (u32, u32, u32, u32), (u32, u32));

#[test]
fn place_matches_the_crate_resize() {
    // Pieces scaled up and down, a subject box that is the
    // whole piece and one that is a corner of it, at each
    // pad, and issue #243's piece, whose scale the bounds
    // hold down.
    let cases: [PlaceCase; 7] = [
        ((40, 30), (5, 4, 20, 15), (80, 60)),
        ((40, 30), (0, 0, 40, 30), (33, 57)),
        ((40, 30), (30, 20, 10, 10), (200, 120)),
        ((90, 70), (10, 10, 60, 40), (31, 17)),
        ((1, 1), (0, 0, 1, 1), (9, 9)),
        ((7, 5), (2, 1, 3, 3), (1, 1)),
        ((400, 300), (0, 0, 3, 2), (80, 60)),
    ];
    for (size, subject, canvas) in cases {
        let piece = noise_rgba(size.0, size.1, size.0 * 5 + size.1);
        for pad in [0, 10, 40] {
            let (got, dx, dy) = place(&piece, subject, canvas, pad);
            let (want, want_dx, want_dy) = place_by_the_crate(&piece, subject, canvas, pad);
            let what = format!("{size:?} piece, {subject:?} on {canvas:?}, pad {pad}");
            assert_eq!(got.dimensions(), want.dimensions(), "{what}");
            assert_eq!(got.as_raw(), want.as_raw(), "{what}");
            assert_eq!((dx, dy), (want_dx, want_dy), "{what}");
        }
    }
}

#[test]
fn the_composite_matches_the_layered_form() {
    // The whole product composite, image and record, for
    // every backdrop and shadow the grammar names, over a
    // noisy photo in each layout with a soft subject, a
    // mask smaller than the frame, and no subject at all.
    let brand = Some([14, 124, 123]);
    let specs = [
        (StudioBg::White, StudioShadow::Soft, 10, None),
        (
            StudioBg::Color([20, 40, 200]),
            StudioShadow::Strong,
            0,
            None,
        ),
        (StudioBg::None, StudioShadow::Soft, 40, None),
        (StudioBg::Brand, StudioShadow::Soft, 25, None),
        (StudioBg::Brand, StudioShadow::Strong, 25, brand),
        (StudioBg::Default, StudioShadow::None, 5, brand),
    ];
    let rgb = noise_rgb(120, 90, 41);
    let sources = [
        rgb.clone(),
        DynamicImage::ImageRgba8(noise_rgba(120, 90, 42)),
        DynamicImage::ImageLuma8(rgb.to_luma8()),
        DynamicImage::ImageRgb16(rgb.to_rgb16()),
        noise_rgb(31, 57, 43),
    ];
    for source in &sources {
        let (w, h) = (source.width(), source.height());
        let masks = [
            ("a blob", blob_mask(w, h)),
            ("a small mask", blob_mask(w / 3, h / 3)),
            ("soft noise", soft_mask(w, h, 44)),
            ("no subject", GrayImage::new(w, h)),
        ];
        for (what, mask) in &masks {
            for (bg, shadow, pad, brand) in specs {
                let spec = StudioSpec {
                    mode: StudioMode::Product,
                    bg,
                    shadow,
                    pad,
                };
                let got = composite(source, mask, None, &spec, brand);
                let want = composite_by_layers(source, mask, &spec, brand);
                let label = format!("{what}, {spec:?}, brand {brand:?}, {w}x{h}");
                assert_eq!(got.record, want.record, "{label}");
                assert_eq!(
                    std::mem::discriminant(&got.image),
                    std::mem::discriminant(&want.image),
                    "{label}"
                );
                assert_eq!(got.image.color_space(), want.image.color_space());
                assert_eq!(
                    (got.image.width(), got.image.height()),
                    (want.image.width(), want.image.height()),
                    "{label}"
                );
                assert_eq!(got.image.as_bytes(), want.image.as_bytes(), "{label}");
            }
        }
    }
}
