use super::*;

fn rand_rgb(w: u32, h: u32, seed: u64) -> image::RgbImage {
    let mut s = seed;
    let mut buf = vec![0u8; (w * h * 3) as usize];
    for b in buf.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *b = (s >> 33) as u8;
    }
    image::RgbImage::from_raw(w, h, buf).unwrap()
}

pub(crate) fn old_convolve(img: &DynamicImage, kernel: &[f64], normalize: bool) -> DynamicImage {
    let size = if kernel.len() == 9 { 3 } else { 5 };
    let radius = (size / 2) as i32;
    let sum: f64 = kernel.iter().sum();
    let scale = if normalize && sum != 0.0 {
        1.0 / sum
    } else {
        1.0
    };
    let has_alpha = img.color().has_alpha();
    let rgba = img.to_rgba8();
    let (width, height) = (rgba.width() as i32, rgba.height() as i32);
    let mut out = rgba.clone();
    for y in 0..height {
        for x in 0..width {
            let mut acc = [0.0f64; 3];
            for ky in 0..size as i32 {
                let sy = (y + ky - radius).clamp(0, height - 1) as u32;
                for kx in 0..size as i32 {
                    let k = kernel[(ky as usize) * size + kx as usize] * scale;
                    if k == 0.0 {
                        continue;
                    }
                    let sx = (x + kx - radius).clamp(0, width - 1) as u32;
                    let p = rgba.get_pixel(sx, sy);
                    for (c, a) in acc.iter_mut().enumerate() {
                        *a += k * f64::from(p[c]);
                    }
                }
            }
            let o = out.get_pixel_mut(x as u32, y as u32);
            for (c, a) in acc.iter().enumerate() {
                o[c] = round_clamp(*a);
            }
        }
    }
    if has_alpha {
        DynamicImage::ImageRgba8(out)
    } else {
        DynamicImage::ImageRgb8(DynamicImage::ImageRgba8(out).to_rgb8())
    }
}

#[test]
fn filter3x3_rgb_matches_image_crate_in_the_interior_and_replicates_edges() {
    // Interior pixels match the image crate byte for
    // byte; the border is the deliberate divergence of
    // issue #378 (the crate zeroes it), checked here
    // against a clamped-tap reference computed pixel
    // by pixel, for every size including the under-3px
    // ones the crate turns entirely black.
    let reference = |img: &image::RgbImage, kernel: &[f32; 9]| {
        let (w, h) = (img.width() as i32, img.height() as i32);
        let sum: f32 = kernel.iter().sum();
        let inverse_sum = if sum == 0.0 { 1.0 } else { 1.0 / sum };
        let mut out = image::RgbImage::new(w as u32, h as u32);
        for y in 0..h {
            for x in 0..w {
                let mut t = [0f32; 3];
                for (ki, k) in kernel.iter().enumerate().filter(|(_, k)| **k != 0.0) {
                    let sx = (x + (ki % 3) as i32 - 1).clamp(0, w - 1);
                    let sy = (y + (ki / 3) as i32 - 1).clamp(0, h - 1);
                    let px = img.get_pixel(sx as u32, sy as u32);
                    for (c, tc) in t.iter_mut().enumerate() {
                        *tc += px.0[c] as f32 * k;
                    }
                }
                out.put_pixel(
                    x as u32,
                    y as u32,
                    image::Rgb([
                        (t[0] * inverse_sum).clamp(0.0, 255.0) as u8,
                        (t[1] * inverse_sum).clamp(0.0, 255.0) as u8,
                        (t[2] * inverse_sum).clamp(0.0, 255.0) as u8,
                    ]),
                );
            }
        }
        out
    };
    let a = 1.5f32;
    let kernel: [f32; 9] = [0.0, -a, 0.0, -a, 1.0 + 4.0 * a, -a, 0.0, -a, 0.0];
    let blur_kernel: [f32; 9] = [1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0];
    for (w, h) in [(1, 1), (2, 2), (3, 3), (4, 7), (17, 13), (64, 48)] {
        let img = rand_rgb(w, h, 42 + w as u64);
        for k in [kernel, blur_kernel] {
            let mine = filter3x3_rgb(&img, &k);
            assert_eq!(
                mine.as_raw(),
                reference(&img, &k).as_raw(),
                "replicated edges at {w}x{h}"
            );
            if w >= 3 && h >= 3 {
                let theirs = image::imageops::filter3x3(&img, &k);
                for y in 1..h - 1 {
                    for x in 1..w - 1 {
                        assert_eq!(
                            mine.get_pixel(x, y),
                            theirs.get_pixel(x, y),
                            "interior pixel {x},{y} at {w}x{h}"
                        );
                    }
                }
            }
        }
    }
    // The all-black cases of the issue: a solid image
    // under 3px on an axis sharpens to itself, since
    // every tap reads the same replicated pixel.
    for (w, h) in [(1, 1), (2, 2), (1, 7), (7, 1)] {
        let img = image::RgbImage::from_pixel(w, h, image::Rgb([120, 30, 200]));
        let out = filter3x3_rgb(&img, &kernel);
        assert_eq!(out.as_raw(), img.as_raw(), "solid {w}x{h} is unchanged");
    }
}

#[test]
fn convolve_matches_old_implementation() {
    let k3: Vec<f64> = vec![1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0];
    let k5: Vec<f64> = (0..25).map(|i| (i % 7) as f64 - 2.0).collect();
    for (w, h) in [(1, 1), (3, 2), (9, 5), (31, 17)] {
        let rgb = DynamicImage::ImageRgb8(rand_rgb(w, h, 7 + w as u64));
        let mut rgba_buf = rgb.to_rgba8();
        for (i, p) in rgba_buf.pixels_mut().enumerate() {
            p[3] = ((i * 37) % 256) as u8;
        }
        let rgba = DynamicImage::ImageRgba8(rgba_buf);
        for img in [&rgb, &rgba] {
            for (k, norm) in [(&k3, true), (&k3, false), (&k5, true)] {
                let mine = convolve(img, k, norm);
                let theirs = old_convolve(img, k, norm);
                assert_eq!(
                    mine.to_rgba8().as_raw(),
                    theirs.to_rgba8().as_raw(),
                    "size {w}x{h} alpha={} norm={norm}",
                    img.color().has_alpha()
                );
            }
        }
    }
}

#[test]
fn contrast_lut_matches_image_crate() {
    for amount in [-100.0f32, -25.0, 0.0, 10.0, 25.0, 100.0] {
        let rgb = rand_rgb(64, 48, 7);
        let mine = contrast_in_place(rgb.clone(), amount);
        let theirs = image::imageops::contrast(&rgb, amount);
        assert_eq!(mine.as_raw(), theirs.as_raw(), "amount {amount}");
    }
}

/// Every color an 8-bit RGB pixel holds, once each: a 4096 by
/// 4096 frame.
fn rgb_cube() -> image::RgbImage {
    image::RgbImage::from_fn(4096, 4096, |x, y| {
        let i = y * 4096 + x;
        image::Rgb([(i >> 16) as u8, (i >> 8) as u8, i as u8])
    })
}

/// Makes the filters run on this thread where they would be
/// skipped, until dropped.
struct RunsSkipped;

impl RunsSkipped {
    fn new() -> Self {
        RUNS_SKIPPED.set(true);
        RunsSkipped
    }
}

impl Drop for RunsSkipped {
    fn drop(&mut self) {
        RUNS_SKIPPED.set(false);
    }
}

#[test]
fn the_arguments_skipped_change_no_pixel() {
    // Issue #810: each filter skipped for its arguments, run
    // anyway, hands every color back as it was: 16.7 million,
    // the whole RGB cube.
    use PlainFilter as F;
    let cube = DynamicImage::ImageRgb8(rgb_cube());
    let _runs = RunsSkipped::new();
    for filter in [
        F::Saturation(1.0),
        F::HueRotate(0),
        F::HueRotate(720),
        F::Rgb(0, 0, 0),
        F::Brighten(0),
        F::Gamma(1.0),
        F::Exposure(0.0),
    ] {
        assert!(leaves_unchanged(&filter, &cube), "{filter:?}");
        let (out, _) = apply_filters(cube.clone(), std::slice::from_ref(&filter));
        assert!(
            out.as_bytes() == cube.as_bytes(),
            "{filter:?} moved a color"
        );
    }
    // A whole number of turns reduces to no angle, whose matrix
    // is the identity exactly; the tables map every value, the
    // whole of what a table holds, to itself.
    for turns in (i32::MIN / 360)..=(i32::MAX / 360) {
        assert_eq!(hue_degrees(turns * 360), 0, "{turns} turns");
    }
    assert_eq!(hue_matrix(0), [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    assert!(is_identity(&brighten_lut(0)));
    assert!(is_identity(&gamma_lut(1.0)));
    assert!(is_identity(&exposure_lut(0.0)));
    assert!(is_identity(&exposure_lut(-0.0)));
    // Any other brightness moves an end of the range.
    for value in (-255..=255).filter(|&v| v != 0) {
        assert!(!is_identity(&brighten_lut(value)), "brighten({value})");
    }
}

/// Small frames in the layouts a filter can receive, the ones
/// with alpha under every alpha level.
fn filter_layouts() -> Vec<(&'static str, DynamicImage)> {
    let rgb = DynamicImage::ImageRgb8(rand_rgb(31, 9, 810));
    let mut rgba = rgb.to_rgba8();
    for (i, p) in rgba.pixels_mut().enumerate() {
        p[3] = (i * 37 % 256) as u8;
    }
    let rgba = DynamicImage::ImageRgba8(rgba);
    vec![
        ("luma8", DynamicImage::ImageLuma8(rgb.to_luma8())),
        ("lumaa8", DynamicImage::ImageLumaA8(rgba.to_luma_alpha8())),
        ("rgb16", DynamicImage::ImageRgb16(rgb.to_rgb16())),
        ("rgba16", DynamicImage::ImageRgba16(rgba.to_rgba16())),
        ("rgb32f", DynamicImage::ImageRgb32F(rgb.to_rgb32f())),
        ("rgb8", rgb),
        ("rgba8", rgba),
    ]
}

#[test]
fn identity_arguments_skip_only_on_the_layouts_they_keep() {
    // Issue #810: which filters `leaves_unchanged` skips, and on
    // which layouts. Where one is skipped, running it anyway
    // hands the frame back as it came, layout and bytes, alpha
    // included. At any other argument, and on a layout the
    // filter turns into RGB8 or RGBA8 as it runs, it runs.
    use PlainFilter as F;
    let eight_bit = ["luma8", "lumaa8", "rgb8", "rgba8"].as_slice();
    let rgb = ["rgb8", "rgba8"].as_slice();
    let cases: [(PlainFilter, &[&str]); 17] = [
        (F::Brighten(0), eight_bit),
        (F::Gamma(1.0), rgb),
        (F::Exposure(0.0), rgb),
        (F::Saturation(1.0), rgb),
        (F::HueRotate(0), rgb),
        (F::HueRotate(-720), rgb),
        (F::Rgb(0, 0, 0), rgb),
        (F::Brighten(1), &[]),
        (F::Gamma(1.5), &[]),
        (F::Exposure(0.5), &[]),
        (F::Saturation(1.5), &[]),
        (F::HueRotate(90), &[]),
        (F::Rgb(0, 0, 1), &[]),
        (F::Contrast(0.0), &[]),
        (F::Contrast(25.0), &[]),
        (F::Fill([1, 2, 3]), &[]),
        (F::NoUpscale, &[]),
    ];
    let mut skipped = 0;
    for (filter, skips_on) in &cases {
        for (name, img) in filter_layouts() {
            let skips = leaves_unchanged(filter, &img);
            assert_eq!(skips, skips_on.contains(&name), "{filter:?} on {name}");
            if skips {
                let _runs = RunsSkipped::new();
                let (out, _) = apply_filters(img.clone(), std::slice::from_ref(filter));
                assert_eq!(out.color(), img.color(), "{filter:?} on {name}");
                assert!(out.as_bytes() == img.as_bytes(), "{filter:?} on {name}");
                skipped += 1;
            }
        }
    }
    assert_eq!(skipped, 4 + 6 * 2);
    // On another layout the skip would not be exact: gray comes
    // back from `gamma(1)` and the others as RGB8.
    let gray = DynamicImage::ImageLuma8(DynamicImage::ImageRgb8(rand_rgb(9, 7, 815)).to_luma8());
    for filter in [
        F::Gamma(1.0),
        F::Exposure(0.0),
        F::Saturation(1.0),
        F::Rgb(0, 0, 0),
    ] {
        let (out, _) = apply_filters(gray.clone(), std::slice::from_ref(&filter));
        assert_eq!(out.color(), image::ColorType::Rgb8, "{filter:?}");
    }
}

#[test]
fn contrast_zero_still_runs_since_its_table_drops_levels() {
    // Issue #815: `contrast(0)`'s table maps 1 to 63 one level
    // down and every other value to itself, so `contrast(0)`
    // changes pixels and is not skipped. Skipping it, or making
    // its table the identity, would move rendered bytes.
    let lut = contrast_lut(0.0);
    let moved: Vec<usize> = (0..256).filter(|&i| usize::from(lut[i]) != i).collect();
    assert_eq!(moved, (1..=63).collect::<Vec<_>>());
    assert!(moved.iter().all(|&i| usize::from(lut[i]) == i - 1));
    let ramp = DynamicImage::ImageRgb8(image::RgbImage::from_fn(256, 1, |x, _| {
        image::Rgb([x as u8; 3])
    }));
    assert!(!leaves_unchanged(&PlainFilter::Contrast(0.0), &ramp));
    let (out, _) = apply_filters(ramp.clone(), &[PlainFilter::Contrast(0.0)]);
    assert_ne!(out.as_bytes(), ramp.as_bytes());
    assert_eq!(out.as_bytes()[3 * 63], 62);
}
