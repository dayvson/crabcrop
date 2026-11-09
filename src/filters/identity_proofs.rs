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
