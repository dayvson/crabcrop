use super::*;
use image::imageops::FilterType;

/// A deterministic noisy buffer: a smooth ramp would hide a
/// misplaced tap.
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

fn crate_filter(kernel: Kernel) -> FilterType {
    match kernel {
        Kernel::Triangle => FilterType::Triangle,
        Kernel::CatmullRom => FilterType::CatmullRom,
    }
}

const KERNELS: [Kernel; 2] = [Kernel::Triangle, Kernel::CatmullRom];

/// Whether a case is one of the largest frames. The image
/// crate's resize of one takes most of a second in a debug
/// build, and the kernels and layouts differ in nothing a
/// smaller frame does not show, so these run once: with the
/// kernel and the layout of the caller that reaches them,
/// the 224-crop's CatmullRom on RGB.
fn large((sw, sh): (u32, u32), (dw, dh): (u32, u32)) -> bool {
    (sw * sh).max(dw * dh) > 100_000
}

/// Source and target sizes: one pixel either way, upscales,
/// downscales, an unchanged size, exact halvings and
/// doublings (whose sums land on half values), a long thin
/// frame, and the sizes the callers reach.
const SIZES: [((u32, u32), (u32, u32)); 18] = [
    ((1, 1), (1, 1)),
    ((1, 1), (5, 3)),
    ((2, 3), (1, 1)),
    ((3, 2), (7, 9)),
    ((17, 5), (16, 6)),
    ((64, 48), (64, 48)),
    ((64, 48), (32, 24)),
    ((32, 24), (64, 48)),
    ((97, 61), (33, 224)),
    ((61, 97), (224, 31)),
    ((500, 3), (7, 300)),
    ((200, 150), (336, 224)),
    ((400, 250), (168, 112)),
    ((413, 277), (512, 512)),
    ((512, 512), (360, 361)),
    ((800, 533), (336, 224)),
    ((300, 31), (464, 48)),
    ((1, 700), (224, 4096)),
];

#[test]
fn resize_matches_the_image_crate_byte_for_byte() {
    // Issue #673: one, three and four channels, every size
    // in the table. A size the module declines would come
    // back `None`: none does.
    for kernel in KERNELS {
        for ((sw, sh), (dw, dh)) in SIZES {
            let large = large((sw, sh), (dw, dh));
            if large && kernel != Kernel::CatmullRom {
                continue;
            }
            let filter = crate_filter(kernel);
            let rgb = noise((sw * sh * 3) as usize, sw * 31 + sh);
            let img = image::RgbImage::from_raw(sw, sh, rgb.clone()).unwrap();
            assert_eq!(
                resize::<3>(&rgb, (sw, sh), (dw, dh), kernel),
                Some(image::imageops::resize(&img, dw, dh, filter).into_raw()),
                "RGB {sw}x{sh} to {dw}x{dh} with {kernel:?}"
            );
            if large {
                continue;
            }
            let rgba = noise((sw * sh * 4) as usize, sw * 17 + sh);
            let img = image::RgbaImage::from_raw(sw, sh, rgba.clone()).unwrap();
            assert_eq!(
                resize::<4>(&rgba, (sw, sh), (dw, dh), kernel),
                Some(image::imageops::resize(&img, dw, dh, filter).into_raw()),
                "RGBA {sw}x{sh} to {dw}x{dh} with {kernel:?}"
            );
            let gray = noise((sw * sh) as usize, sw * 7 + sh);
            let img = image::GrayImage::from_raw(sw, sh, gray.clone()).unwrap();
            assert_eq!(
                resize::<1>(&gray, (sw, sh), (dw, dh), kernel),
                Some(image::imageops::resize(&img, dw, dh, filter).into_raw()),
                "gray {sw}x{sh} to {dw}x{dh} with {kernel:?}"
            );
        }
    }
}

/// A source size, a target size, and a window of the target.
type WindowCase = ((u32, u32), (u32, u32), (u32, u32, u32, u32));

#[test]
fn a_window_is_the_crop_of_the_full_resize() {
    // The 224-crop's windows (wide, tall, and at the long
    // side's cap), a window in a corner, one against the far
    // edges, a single sample, and a window of an unchanged
    // size.
    let cases: [WindowCase; 9] = [
        ((300, 200), (336, 224), (56, 0, 224, 224)),
        ((800, 533), (336, 224), (56, 0, 224, 224)),
        ((533, 800), (224, 336), (0, 56, 224, 224)),
        ((100, 2000), (224, 4096), (0, 1936, 224, 224)),
        ((97, 61), (300, 250), (0, 0, 224, 224)),
        ((97, 61), (300, 250), (76, 26, 224, 224)),
        ((97, 61), (300, 250), (299, 249, 1, 1)),
        ((300, 250), (97, 61), (13, 7, 40, 30)),
        ((97, 61), (97, 61), (10, 20, 30, 40)),
    ];
    for kernel in KERNELS {
        for ((sw, sh), (dw, dh), window) in cases {
            let large = large((sw, sh), (dw, dh));
            if large && kernel != Kernel::CatmullRom {
                continue;
            }
            let (wx, wy, ww, wh) = window;
            let rgb = noise((sw * sh * 3) as usize, sw + sh);
            let img = image::RgbImage::from_raw(sw, sh, rgb.clone()).unwrap();
            let full = image::imageops::resize(&img, dw, dh, crate_filter(kernel));
            assert_eq!(
                resize_window::<3>(&rgb, (sw, sh), (dw, dh), kernel, window),
                Some(
                    image::imageops::crop_imm(&full, wx, wy, ww, wh)
                        .to_image()
                        .into_raw()
                ),
                "RGB {sw}x{sh} to {dw}x{dh}, window {window:?}, {kernel:?}"
            );
            if large {
                continue;
            }
            let rgba = noise((sw * sh * 4) as usize, sw * 3 + sh);
            let img = image::RgbaImage::from_raw(sw, sh, rgba.clone()).unwrap();
            let full = image::imageops::resize(&img, dw, dh, crate_filter(kernel));
            assert_eq!(
                resize_window::<4>(&rgba, (sw, sh), (dw, dh), kernel, window),
                Some(
                    image::imageops::crop_imm(&full, wx, wy, ww, wh)
                        .to_image()
                        .into_raw()
                ),
                "RGBA {sw}x{sh} to {dw}x{dh}, window {window:?}, {kernel:?}"
            );
        }
    }
}

#[test]
fn sums_on_a_half_round_as_the_crate_rounds_them() {
    // Doubling two samples two apart: with the Triangle
    // kernel the second output is three quarters of one and
    // a quarter of the other, a half exactly, where the two
    // ways of rounding could part. Up from a low sample and
    // down from a high one.
    for low in [0u8, 1, 126, 127, 252, 253] {
        for gray in [[low, low + 2], [low + 2, low]] {
            let img = image::GrayImage::from_raw(2, 1, gray.to_vec()).unwrap();
            for kernel in KERNELS {
                let want = image::imageops::resize(&img, 4, 1, crate_filter(kernel)).into_raw();
                assert_eq!(
                    resize::<1>(&gray, (2, 1), (4, 1), kernel),
                    Some(want.clone()),
                    "{gray:?} with {kernel:?}"
                );
                if kernel == Kernel::Triangle {
                    // The half goes away from zero.
                    let half = gray[0].min(gray[1]) + 1;
                    assert!(want[1] == half || want[2] == half, "{want:?}");
                }
            }
        }
    }
}

#[test]
fn empty_sources_and_targets_match_the_crate() {
    for kernel in KERNELS {
        let filter = crate_filter(kernel);
        let empty = image::RgbImage::new(0, 0);
        assert_eq!(
            resize::<3>(&[], (0, 0), (4, 3), kernel),
            Some(image::imageops::resize(&empty, 4, 3, filter).into_raw())
        );
        let rgb = noise(5 * 4 * 3, 9);
        let img = image::RgbImage::from_raw(5, 4, rgb.clone()).unwrap();
        for (dw, dh) in [(0, 0), (0, 7), (7, 0)] {
            assert_eq!(
                resize::<3>(&rgb, (5, 4), (dw, dh), kernel),
                Some(image::imageops::resize(&img, dw, dh, filter).into_raw()),
                "to {dw}x{dh}"
            );
        }
    }
}

#[test]
fn what_the_module_cannot_read_is_left_to_the_caller() {
    let rgb = noise(5 * 4 * 3, 11);
    let kernel = Kernel::CatmullRom;
    // A buffer shorter than its size.
    assert_eq!(resize::<3>(&rgb[..59], (5, 4), (9, 9), kernel), None);
    // A window that is not inside the output.
    assert_eq!(
        resize_window::<3>(&rgb, (5, 4), (9, 9), kernel, (5, 0, 5, 9)),
        None
    );
    assert_eq!(
        resize_window::<3>(&rgb, (5, 4), (9, 9), kernel, (0, u32::MAX, 1, 2)),
        None
    );
    // More channels than a pixel has.
    assert_eq!(
        resize::<5>(&noise(5 * 4 * 5, 3), (5, 4), (9, 9), kernel),
        None
    );
}

/// The crate's conversion of a sum to a `u8`: its clamp,
/// which like `f32::clamp` hands a NaN on, `round`, and the
/// checked cast, which refuses a NaN (the crate unwraps it;
/// a bounded weighted sum of 8-bit samples is never one).
fn crate_to_u8(t: f32) -> Option<u8> {
    let x = t.clamp(0.0, 255.0);
    let rounded = x.round();
    (rounded > -1.0 && rounded < 256.0).then_some(rounded as u8)
}

#[test]
fn to_u8_rounds_as_the_crate_does() {
    // The comparison stands in for `round`. Every f32 cannot
    // be walked in a unit test, so: both zeros, every half
    // from 0.5 to 254.5 with the floats on either side of
    // it, every whole number with its neighbours, the ends
    // of the range, the infinities, subnormals, and a sample
    // of every sign and exponent with 4,096 mantissas each.
    let mut values: Vec<f32> = vec![
        0.0,
        -0.0,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::from_bits(0x8000_0001),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        f32::MIN,
        255.0,
        255.5,
        256.0,
        -0.5,
        -1.0,
    ];
    for whole in 0..=255u32 {
        for base in [whole as f32, whole as f32 + 0.5] {
            values.extend([base, base.next_up(), base.next_down()]);
        }
    }
    for sign_and_exponent in 0..512u32 {
        for step in 0..4096u32 {
            // Spread over the 23 mantissa bits, with the low
            // bits varied too.
            let mantissa = (step << 11) | (step.wrapping_mul(2_654_435_761) >> 21);
            values.push(f32::from_bits(
                (sign_and_exponent << 23) | (mantissa & 0x7f_ffff),
            ));
        }
    }
    let mut checked = 0;
    for v in values {
        match crate_to_u8(v) {
            Some(want) => assert_eq!(to_u8(v), want, "{v:e} ({:#x})", v.to_bits()),
            None => assert!(v.is_nan(), "{v:e}"),
        }
        checked += 1;
    }
    assert!(checked > 2_000_000);
    // A NaN reads as 0 here, where the crate would stop.
    assert_eq!(to_u8(f32::NAN), 0);
    assert_eq!(crate_to_u8(f32::NAN), None);
}

#[test]
fn the_weights_are_the_bounded_ones_the_module_computes_with() {
    // Every run of normalized weights, over sizes from one
    // sample to a few thousand in both directions, stays
    // well inside the limit that rules out an overflow, so
    // no size is turned away for its weights.
    for kernel in KERNELS {
        let mut largest = 0f32;
        for src in [1u32, 2, 3, 7, 61, 224, 533, 4096, 65_535] {
            for dst in [1u32, 2, 3, 7, 61, 224, 533, 4096] {
                let taps = axis_taps(src, dst, kernel, 0, dst).expect("bounded weights");
                assert_eq!(taps.runs.len(), dst as usize);
                for &w in &taps.weights {
                    largest = largest.max(w.abs());
                }
            }
        }
        assert!(largest <= 3.0, "{largest} with {kernel:?}");
    }
}

/// An image in each layout the stand-ins compute over rows,
/// RGBA8 with every alpha the noise gives, and in two layouts
/// they leave to the crate, with whether its rows are this
/// module's.
fn images((w, h): (u32, u32), seed: u32) -> Vec<(DynamicImage, bool)> {
    let rgb = image::RgbImage::from_raw(w, h, noise((w * h * 3) as usize, seed)).unwrap();
    let rgba = image::RgbaImage::from_raw(w, h, noise((w * h * 4) as usize, seed + 1)).unwrap();
    let rgb = DynamicImage::ImageRgb8(rgb);
    vec![
        (DynamicImage::ImageLuma8(rgb.to_luma8()), false),
        (DynamicImage::ImageRgb16(rgb.to_rgb16()), false),
        (rgb, true),
        (DynamicImage::ImageRgba8(rgba), true),
    ]
}

/// The layout, the size and every byte.
fn same(a: &DynamicImage, b: &DynamicImage) -> bool {
    a.color() == b.color()
        && (a.width(), a.height()) == (b.width(), b.height())
        && a.as_bytes() == b.as_bytes()
}

#[test]
fn resize_to_fill_matches_the_crate_byte_for_byte() {
    // Issue #806, the backdrop of `pad:WxH:blur`: a pixel, wide,
    // tall and odd-sized sources, into boxes that cut the sides,
    // cut the top and bottom, shrink the source, enlarge it past
    // its own size, miss square by a pixel, and match it. The
    // crate's layouts run with the pad's kernel only: their
    // bytes are the crate's own call.
    let mut cases = 0;
    for (sw, sh) in [(1, 1), (3, 2), (97, 61), (240, 160), (161, 241)] {
        for (tw, th) in [
            (1, 1),
            (80, 80),
            (120, 30),
            (30, 120),
            (64, 64),
            (79, 81),
            (300, 300),
            (240, 160),
        ] {
            for (img, rows) in images((sw, sh), sw * 13 + sh) {
                for kernel in KERNELS {
                    if !rows && kernel != Kernel::Triangle {
                        continue;
                    }
                    let ours = resize_to_fill(&img, tw, th, kernel);
                    let theirs = img.resize_to_fill(tw, th, crate_filter(kernel));
                    assert!(
                        same(&ours, &theirs),
                        "{:?} {sw}x{sh} to {tw}x{th} with {kernel:?}",
                        img.color()
                    );
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 5 * 8 * (2 * 2 + 2));
}

#[test]
fn the_layouts_filled_by_window_are_the_ones_computed_over_rows() {
    // A blurred pad holds the cover of an image to the byte
    // budget only where `resize_to_fill` builds it whole (issue
    // #784), and asks `fills_by_window` which covers those are:
    // every layout `image_window` does not compute.
    use image::ColorType;
    for layout in [
        ColorType::L8,
        ColorType::La8,
        ColorType::Rgb8,
        ColorType::Rgba8,
        ColorType::L16,
        ColorType::La16,
        ColorType::Rgb16,
        ColorType::Rgba16,
        ColorType::Rgb32F,
        ColorType::Rgba32F,
    ] {
        let img = DynamicImage::new(3, 2, layout);
        let computed = image_window(&img, (6, 4), Kernel::Triangle, (0, 1, 6, 2)).is_some();
        assert_eq!(fills_by_window(layout), computed, "{layout:?}");
    }
}

#[test]
fn resize_dimensions_are_the_crate_s() {
    // The size `DynamicImage::resize` fits a box at, as its
    // output shows it. `resize_to_fill` takes the larger ratio
    // where this takes the smaller, which the bytes of its cut
    // check above. Shapes from a pixel to a long strip, into
    // boxes on either side of them.
    let mut cases = 0;
    for (w, h) in [
        (1, 1),
        (1, 7),
        (7, 1),
        (3, 2),
        (97, 61),
        (800, 533),
        (533, 800),
        (16_383, 1),
    ] {
        for (nw, nh) in [
            (1, 1),
            (100, 100),
            (16, 9),
            (9, 16),
            (2, 1000),
            (1000, 2),
            (w, h),
            (w + 1, h),
        ] {
            let fit = DynamicImage::new_luma8(w, h).resize(nw, nh, FilterType::Nearest);
            assert_eq!(
                resize_dimensions(w, h, nw, nh, false),
                (fit.width(), fit.height()),
                "{w}x{h} into {nw}x{nh}"
            );
            cases += 1;
        }
    }
    assert_eq!(cases, 8 * 8);
    // A strip turned across its box covers it at its length
    // squared; past `u32` that axis is held at it, and the
    // other follows at the ratio that reaches it, as the crate
    // holds them.
    assert_eq!(
        resize_dimensions(16_383, 1, 1, 16_383, true),
        (268_402_689, 16_383)
    );
    assert_eq!(
        resize_dimensions(1, 65_536, 65_536, 1, true),
        (65_536, u32::MAX)
    );
}
