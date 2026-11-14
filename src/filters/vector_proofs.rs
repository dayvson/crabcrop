use super::*;

/// Smooth content with a little noise on top.
fn smooth(x: u32, y: u32, c: u32) -> u8 {
    let ramp = (x * 3 + y * 5 + c * 40) % 256;
    let mut h = x
        .wrapping_mul(0x9E37_79B1)
        .wrapping_add(y.wrapping_mul(0x85EB_CA77))
        .wrapping_add(c.wrapping_mul(0xC2B2_AE3D));
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    ((ramp * 3 + (h % 256)) / 4) as u8
}

/// Hard edges between the extremes, where a sum overshoots
/// and the clamp has work to do.
fn harsh(x: u32, y: u32, c: u32) -> u8 {
    match (x / 3 + y / 2 + c) % 5 {
        0 => 0,
        1 => 255,
        2 => 1,
        3 => 254,
        _ => smooth(x, y, c),
    }
}

fn rgb(w: u32, h: u32, content: fn(u32, u32, u32) -> u8) -> image::RgbImage {
    image::RgbImage::from_fn(w, h, |x, y| {
        image::Rgb([content(x, y, 0), content(x, y, 1), content(x, y, 2)])
    })
}

/// `filter3x3_rgb` as it stood before issue #658: every
/// output sample summed on its own, the taps as the inner
/// loop.
fn old_filter3x3_rgb(img: &image::RgbImage, kernel: &[f32; 9]) -> image::RgbImage {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut out = image::RgbImage::new(w as u32, h as u32);
    if w == 0 || h == 0 {
        return out;
    }
    let src = img.as_raw();
    let dst: &mut [u8] = out.as_mut();
    let stride = w * 3;
    let sum: f32 = kernel.iter().sum();
    let inverse_sum = if sum == 0.0 { 1.0 } else { 1.0 / sum };
    let taps: Vec<(usize, usize, f32)> = kernel
        .iter()
        .enumerate()
        .filter(|(_, k)| **k != 0.0)
        .map(|(ki, k)| (ki / 3, (ki % 3) * 3, *k))
        .collect();
    if w >= 3 && h >= 3 {
        for y in 1..h - 1 {
            let rows = [
                &src[(y - 1) * stride..y * stride],
                &src[y * stride..(y + 1) * stride],
                &src[(y + 1) * stride..(y + 2) * stride],
            ];
            for x in 1..w - 1 {
                let xb = x * 3;
                let mut t = [0f32; 3];
                for &(rd, coff, k) in &taps {
                    let b = xb + coff - 3;
                    let row = rows[rd];
                    t[0] += row[b] as f32 * k;
                    t[1] += row[b + 1] as f32 * k;
                    t[2] += row[b + 2] as f32 * k;
                }
                let base = y * stride + xb;
                for c in 0..3 {
                    dst[base + c] = (t[c] * inverse_sum).clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
    for y in 0..h {
        for x in 0..w {
            if w >= 3 && h >= 3 && (1..w - 1).contains(&x) && (1..h - 1).contains(&y) {
                continue;
            }
            let mut t = [0f32; 3];
            for &(rd, coff, k) in &taps {
                let sy = (y + rd).saturating_sub(1).min(h - 1);
                let sx = (x + coff / 3).saturating_sub(1).min(w - 1);
                let b = sy * stride + sx * 3;
                t[0] += src[b] as f32 * k;
                t[1] += src[b + 1] as f32 * k;
                t[2] += src[b + 2] as f32 * k;
            }
            let base = y * stride + x * 3;
            for c in 0..3 {
                dst[base + c] = (t[c] * inverse_sum).clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

fn sharpen_kernel(a: f32) -> [f32; 9] {
    [0.0, -a, 0.0, -a, 1.0 + 4.0 * a, -a, 0.0, -a, 0.0]
}

#[test]
fn row_pass_convolution_matches_the_per_sample_sums() {
    // Issue #658. The kernels: sharpen from nothing to far
    // past the clamp, a gaussian, an edge kernel whose
    // weights sum to zero, lone taps, weights that span
    // sixteen orders of magnitude (the sum of those depends
    // on the order of the additions, so a tap out of turn
    // shows), and weights that overflow to infinity and to
    // NaN, which must come out as the same bytes too.
    let mut kernels: Vec<[f32; 9]> = [0.0f32, 0.01, 0.3, 1.0, 1.2, 1.5, 2.7, 10.0, 255.0]
        .map(sharpen_kernel)
        .to_vec();
    kernels.extend([
        [
            0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625,
        ],
        [-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.5],
        [1e8, 1.0, -1e8, 0.5, 1.0, 0.25, 3.0, -2.0, 1e-8],
        [0.1, 0.7, -0.3, 1e7, 0.2, -1e7, 0.9, 0.3, 0.6],
        [f32::MAX, f32::MAX, 1.0, -1.0, 0.5, 0.0, 0.0, f32::MIN, 1.0],
        [
            f32::INFINITY,
            1.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            1.0,
            f32::NEG_INFINITY,
        ],
        [f32::NAN, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
    ]);
    // Sizes with no interior, with one interior pixel, every
    // width whose interior leaves a different remainder
    // against a vector block, and a frame with many rows.
    let mut sizes = vec![
        (1, 1),
        (1, 5),
        (5, 1),
        (2, 2),
        (2, 9),
        (3, 3),
        (3, 7),
        (7, 3),
    ];
    sizes.extend((4..=20).map(|w| (w, 4)));
    sizes.extend([(35, 6), (101, 53)]);
    let mut cases = 0;
    for &(w, h) in &sizes {
        for content in [smooth as fn(u32, u32, u32) -> u8, harsh] {
            let img = rgb(w, h, content);
            for kernel in &kernels {
                assert_eq!(
                    old_filter3x3_rgb(&img, kernel).as_raw(),
                    filter3x3_rgb(&img, kernel).as_raw(),
                    "{w}x{h} {kernel:?}"
                );
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 27 * 2 * 18);
    // The size the docs' sharpen example runs at.
    let img = rgb(800, 533, harsh);
    for kernel in [sharpen_kernel(1.5), kernels[9]] {
        assert_eq!(
            old_filter3x3_rgb(&img, &kernel).as_raw(),
            filter3x3_rgb(&img, &kernel).as_raw(),
            "800x533 {kernel:?}"
        );
    }
}

#[test]
fn sharpen_through_the_filter_matches_the_per_sample_sums() {
    // Issue #658, through `filters:sharpen(...)`: an image
    // with alpha keeps it, and its colors are the old sums.
    let img = rgb(64, 48, harsh);
    let rgba = image::RgbaImage::from_fn(64, 48, |x, y| {
        let p = img.get_pixel(x, y);
        image::Rgba([p[0], p[1], p[2], smooth(y, x, 3)])
    });
    for amount in [0.5f32, 1.5, 4.0] {
        let expected = old_filter3x3_rgb(&img, &sharpen_kernel(amount));
        let (out, _) = apply_filters(
            DynamicImage::ImageRgb8(img.clone()),
            &[Filter::Sharpen(amount)],
        );
        assert_eq!(out.as_bytes(), expected.as_raw().as_slice(), "{amount}");
        let (out, _) = apply_filters(
            DynamicImage::ImageRgba8(rgba.clone()),
            &[Filter::Sharpen(amount)],
        );
        let out = out.to_rgba8();
        for (ours, (color, original)) in out.pixels().zip(expected.pixels().zip(rgba.pixels())) {
            assert_eq!(ours.0, [color[0], color[1], color[2], original[3]]);
        }
    }
}
