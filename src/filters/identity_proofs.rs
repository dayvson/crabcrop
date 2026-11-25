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

/// [`filter3x3`] over an RGB8 image: the form these proofs and
/// the vector proofs compare.
pub(crate) fn filter3x3_rgb(img: &image::RgbImage, kernel: &[f32; 9]) -> image::RgbImage {
    let (w, h) = img.dimensions();
    let out = filter3x3::<3>(img.as_raw(), (w as usize, h as usize), kernel);
    image::RgbImage::from_raw(w, h, out).expect("the image's size")
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
    // Every amount but zero stays byte-identical to the image
    // crate's contrast; `contrast(0)` is the identity instead
    // (issue #815), so it is checked on its own below and left
    // out here, where the crate drops a level at 0.
    for amount in [-100.0f32, -25.0, 10.0, 25.0, 100.0] {
        let rgb = rand_rgb(64, 48, 7);
        let mine = contrast(DynamicImage::ImageRgb8(rgb.clone()), amount);
        let theirs = image::imageops::contrast(&rgb, amount);
        assert_eq!(
            mine.as_bytes(),
            theirs.as_raw().as_slice(),
            "amount {amount}"
        );
    }
}

/// Issue #815 acceptance: the tables for the other arguments
/// follow the documented formula with its rounding. The formula
/// scales each byte about the midpoint by `((100 + amount)/100)^2`
/// in f32 and truncates; this recomputes it independently and
/// pins every entry, so a later change to the rounding would be
/// caught here, not only through the image crate.
#[test]
fn contrast_lut_follows_the_formula_for_nonzero_amounts() {
    for amount in [-100.0f32, -50.0, -25.0, -1.0, 1.0, 10.0, 25.0, 50.0, 100.0] {
        let percent = ((100.0 + amount) / 100.0).powi(2);
        let want: [u8; 256] = std::array::from_fn(|i| {
            let d = ((i as f32 / 255.0 - 0.5) * percent + 0.5) * 255.0;
            d.clamp(0.0, 255.0) as u8
        });
        assert_eq!(contrast_lut(amount), want, "amount {amount}");
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
        F::Contrast(0.0),
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
        // contrast(0) is the identity now (issue #815), so it
        // skips on the two RGB layouts like the others at their
        // no-op argument; any other amount still runs everywhere.
        (F::Contrast(0.0), rgb),
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
    assert_eq!(skipped, 4 + 7 * 2);
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
fn contrast_zero_is_the_identity() {
    // Issue #815: `contrast(0)` scales about the midpoint by one,
    // which is the identity, so its table must map every value to
    // itself. Before the fix the f32 formula truncated 1 to 63 one
    // level down. This is the exhaustive check the issue asks for.
    let lut = contrast_lut(0.0);
    let moved: Vec<usize> = lut
        .iter()
        .enumerate()
        .filter(|&(i, &v)| usize::from(v) != i)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(moved, Vec::<usize>::new(), "contrast(0) moves no value");
    for (i, &v) in lut.iter().enumerate() {
        assert_eq!(usize::from(v), i, "contrast(0) maps {i}");
    }
    // The identity table means `contrast(0)` is skipped on the two
    // 8-bit RGB layouts, like the other filters at their no-op
    // argument, and leaves a full ramp untouched when it does run.
    let ramp = DynamicImage::ImageRgb8(image::RgbImage::from_fn(256, 1, |x, _| {
        image::Rgb([x as u8; 3])
    }));
    assert!(leaves_unchanged(&PlainFilter::Contrast(0.0), &ramp));
    let _runs = RunsSkipped::new();
    let (out, _) = apply_filters(ramp.clone(), &[PlainFilter::Contrast(0.0)]);
    assert_eq!(out.as_bytes(), ramp.as_bytes());
}

/// What a filter over RGB8 made of an image before issue #812:
/// `map_rgb_image` ran it on an RGB8 copy of an image with alpha
/// and copied the colors into the image's RGBA8 form, one pixel
/// at a time; an image without alpha it ran on as RGB8.
fn map_rgb_image_as_it_was(
    img: DynamicImage,
    f: impl FnOnce(image::RgbImage) -> image::RgbImage,
) -> DynamicImage {
    if img.color().has_alpha() {
        let mapped = f(img.to_rgb8());
        let mut out = img.into_rgba8();
        for (o, m) in out.pixels_mut().zip(mapped.pixels()) {
            o[0] = m[0];
            o[1] = m[1];
            o[2] = m[2];
        }
        DynamicImage::ImageRgba8(out)
    } else {
        DynamicImage::ImageRgb8(f(img.into_rgb8()))
    }
}

/// The layouts an RGB filter can be handed, at `w` by `h`:
/// RGBA8 opaque, translucent at every level, clear over colors,
/// and a GIF frame's opaque or clear black pixels; the other
/// layouts with alpha, translucent, their samples off the 8-bit
/// grid; and two layouts without alpha.
fn alpha_layouts(w: u32, h: u32) -> Vec<(&'static str, DynamicImage)> {
    let rgb = DynamicImage::ImageRgb8(rand_rgb(w, h, 812 + u64::from(w)));
    let plane = |alpha: &dyn Fn(usize) -> u8, gif: bool| {
        let mut rgba = rgb.to_rgba8();
        for (i, p) in rgba.pixels_mut().enumerate() {
            p[3] = alpha(i);
            if gif && p[3] == 0 {
                p.0 = [0, 0, 0, 0];
            }
        }
        DynamicImage::ImageRgba8(rgba)
    };
    let translucent = plane(&|i| (i * 37 % 256) as u8, false);
    let mut rgba16 = translucent.to_rgba16();
    for (i, s) in rgba16.iter_mut().enumerate() {
        *s = s.saturating_add((i * 29 % 200) as u16);
    }
    let mut rgba32f = translucent.to_rgba32f();
    for (i, s) in rgba32f.iter_mut().enumerate() {
        *s = (*s + (i % 7) as f32 / 1000.0).min(1.0);
    }
    vec![
        ("rgba8 opaque", plane(&|_| 255, false)),
        ("rgba8 clear", plane(&|_| 0, false)),
        (
            "rgba8 gif",
            plane(&|i| if i % 3 == 0 { 0 } else { 255 }, true),
        ),
        (
            "lumaa8",
            DynamicImage::ImageLumaA8(translucent.to_luma_alpha8()),
        ),
        (
            "lumaa16",
            DynamicImage::ImageLumaA16(translucent.to_luma_alpha16()),
        ),
        ("rgba16", DynamicImage::ImageRgba16(rgba16)),
        ("rgba32f", DynamicImage::ImageRgba32F(rgba32f)),
        ("luma8", DynamicImage::ImageLuma8(rgb.to_luma8())),
        ("rgba8 translucent", translucent),
        ("rgb8", rgb),
    ]
}

#[test]
fn the_rgb_filters_keep_the_bytes_of_the_rgb_copy_they_ran_on() {
    // Issue #812: contrast, sharpen, saturation, vibrance,
    // equalize and enhance filter an image with alpha where its
    // pixels lie, as RGBA8, with no RGB8 copy out and back.
    // Every layout, RGBA8 under each alpha plane, at sizes with
    // no interior for sharpen and odd ones, and a frame past the
    // banding thresholds: the bytes the RGB copy gave, alpha as
    // the RGBA8 form had it.
    use PlainFilter as F;
    let filters = [
        F::Contrast(20.0),
        F::Contrast(-50.0),
        F::Sharpen(1.5),
        F::Saturation(0.5),
        F::Saturation(1.7),
        F::Vibrance(40.0),
        F::Vibrance(-60.0),
        F::Equalize,
        F::Enhance,
    ];
    let check = |name: &str, img: &DynamicImage, filter: &PlainFilter| {
        let (ours, _) = apply_filters(img.clone(), std::slice::from_ref(filter));
        let theirs = map_rgb_image_as_it_was(img.clone(), |rgb| {
            let (rgb, _) =
                apply_filters(DynamicImage::ImageRgb8(rgb), std::slice::from_ref(filter));
            rgb.into_rgb8()
        });
        assert_eq!(ours.color(), theirs.color(), "{filter:?} on {name}");
        assert_eq!(
            (ours.width(), ours.height()),
            (theirs.width(), theirs.height())
        );
        assert!(ours.as_bytes() == theirs.as_bytes(), "{filter:?} on {name}");
    };
    let mut cases = 0;
    for (w, h) in [(1, 1), (2, 5), (3, 3), (17, 9), (64, 48)] {
        for (name, img) in alpha_layouts(w, h) {
            for filter in &filters {
                check(&format!("{name} {w}x{h}"), &img, filter);
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 5 * 10 * 9);
    // Past 3,000,000 samples and 256 rows the maps and the
    // histogram run in bands, on the RGBA8 frame as on its copy.
    let big = alpha_layouts(1003, 1001)
        .into_iter()
        .find(|(name, _)| *name == "rgba8 translucent")
        .unwrap()
        .1;
    for filter in [F::Contrast(20.0), F::Saturation(1.7), F::Enhance] {
        check("rgba8 translucent 1003x1001", &big, &filter);
    }
}
