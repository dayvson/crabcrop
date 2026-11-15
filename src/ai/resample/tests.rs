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
        Kernel::CatmullRom => FilterType::CatmullRom,
    }
}

const KERNELS: [Kernel; 1] = [Kernel::CatmullRom];

/// The whole output as one window.
fn resize<const CH: usize>(
    src: &[u8],
    src_dims: (u32, u32),
    (dw, dh): (u32, u32),
    kernel: Kernel,
) -> Option<Vec<u8>> {
    resize_window::<CH>(src, src_dims, (dw, dh), kernel, (0, 0, dw, dh))
}

/// Source and target sizes: one pixel either way, upscales,
/// downscales, an unchanged size, exact halvings and
/// doublings (whose sums land on half values), a long thin
/// frame, and the sizes the callers reach.
const SIZES: [((u32, u32), (u32, u32)); 17] = [
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
            let filter = crate_filter(kernel);
            let rgb = noise((sw * sh * 3) as usize, sw * 31 + sh);
            let img = image::RgbImage::from_raw(sw, sh, rgb.clone()).unwrap();
            assert_eq!(
                resize::<3>(&rgb, (sw, sh), (dw, dh), kernel),
                Some(image::imageops::resize(&img, dw, dh, filter).into_raw()),
                "RGB {sw}x{sh} to {dw}x{dh} with {kernel:?}"
            );
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

#[test]
fn a_window_is_the_crop_of_the_full_resize() {
    // The 224-crop's windows (wide, tall, and at the long
    // side's cap), a window in a corner, one against the far
    // edges, a single sample, and a window of an unchanged
    // size.
    let cases: [((u32, u32), (u32, u32), (u32, u32, u32, u32)); 8] = [
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

/// The crate's conversion of a sum to a `u8`: its own clamp,
/// `round`, and the checked cast, which refuses a NaN (the
/// crate unwraps it; a bounded weighted sum of 8-bit samples
/// is never one).
fn crate_to_u8(t: f32) -> Option<u8> {
    let x = if t < 0.0 {
        0.0
    } else if t > 255.0 {
        255.0
    } else {
        t
    };
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
