use super::*;
use crate::url::FormatSpec;
use image::Rgb;

fn alpha_at(img: &DynamicImage, x: u32, y: u32) -> u8 {
    img.to_rgba8().get_pixel(x, y)[3]
}

fn opaque(w: u32, h: u32, color: [u8; 3]) -> DynamicImage {
    let mut img = DynamicImage::new_rgb8(w, h);
    for p in img.as_mut_rgb8().unwrap().pixels_mut() {
        *p = Rgb(color);
    }
    img
}

#[test]
fn blur_at_the_sigma_cap_still_renders() {
    let img = opaque(16, 16, [120, 30, 200]);
    let (out, _) = apply_filters(img, &[Filter::Blur(MAX_BLUR_SIGMA)]);
    assert_eq!((out.width(), out.height()), (16, 16));
}

#[test]
fn blur_sigma_guard_tames_out_of_contract_values() {
    assert_eq!(blur_sigma(2.5), 2.5);
    assert_eq!(blur_sigma(1e9), MAX_BLUR_SIGMA);
    assert_eq!(blur_sigma(0.0), f32::MIN_POSITIVE);
    assert_eq!(blur_sigma(f32::NAN), f32::MIN_POSITIVE);
    // A Filter built without the parser must not panic the
    // image crate (subnormal sigma) or size a kernel by an
    // unbounded one.
    let img = opaque(8, 8, [10, 20, 30]);
    let (out, _) = apply_filters(img.clone(), &[Filter::Blur(1e-45)]);
    assert_eq!((out.width(), out.height()), (8, 8));
    let (out, _) = apply_filters(img, &[Filter::Blur(1e9)]);
    assert_eq!((out.width(), out.height()), (8, 8));
}

#[test]
fn border_of_zero_width_is_the_identity() {
    let img = opaque(10, 8, [10, 20, 30]);
    let out = draw_border(&img, 0, [255, 0, 0]);
    assert_eq!(out.to_rgba8(), img.to_rgba8());
}

#[test]
fn radius_cuts_corners_and_antialiases() {
    let img = rounded_corners(&opaque(100, 100, [255, 0, 0]), [20, 20, 20, 20]);
    assert_eq!(alpha_at(&img, 0, 0), 0);
    assert_eq!(alpha_at(&img, 50, 50), 255);
    let edge = alpha_at(&img, 5, 6);
    assert!(edge > 0 && edge < 255, "antialiased edge, got {edge}");
    // Color channels are untouched.
    assert_eq!(img.into_rgba8().get_pixel(50, 50).0, [255, 0, 0, 255]);
}

#[test]
fn radius_multiplies_existing_alpha() {
    let mut img = DynamicImage::new_rgba8(40, 40);
    for p in img.as_mut_rgba8().unwrap().pixels_mut() {
        *p = image::Rgba([10, 20, 30, 100]);
    }
    let img = rounded_corners(&img, [10, 10, 10, 10]);
    assert_eq!(alpha_at(&img, 20, 20), 100);
    assert_eq!(alpha_at(&img, 0, 0), 0);
}

#[test]
fn radius_is_per_corner() {
    let img = rounded_corners(&opaque(100, 100, [0, 0, 255]), [20, 0, 0, 0]);
    assert_eq!(alpha_at(&img, 0, 0), 0);
    assert_eq!(alpha_at(&img, 99, 0), 255);
    assert_eq!(alpha_at(&img, 99, 99), 255);
    assert_eq!(alpha_at(&img, 0, 99), 255);
}

#[test]
fn radius_scales_to_the_frame() {
    // 80px radii on a 100px frame shrink to 50 (CSS rule).
    let img = rounded_corners(&opaque(100, 100, [9, 9, 9]), [80, 80, 80, 80]);
    assert_eq!(alpha_at(&img, 0, 0), 0);
    // (0, 50) sits on the scaled arc itself: antialiased
    // to nearly full, not exactly full.
    assert!(alpha_at(&img, 0, 50) >= 250);
    assert_eq!(alpha_at(&img, 50, 50), 255);
}

#[test]
fn mask_circle_and_ellipse_geometry() {
    let circle = mask_shape(&opaque(100, 60, [255, 0, 0]), crate::url::MaskShape::Circle);
    assert_eq!(alpha_at(&circle, 50, 30), 255);
    assert_eq!(alpha_at(&circle, 0, 0), 0);
    assert_eq!(alpha_at(&circle, 99, 30), 0);
    // The circle's diameter is the height, so it touches
    // the top and bottom edges at the horizontal center.
    assert!(alpha_at(&circle, 50, 59) >= 250);
    assert!(alpha_at(&circle, 50, 0) >= 250);
    let ellipse = mask_shape(
        &opaque(100, 60, [255, 0, 0]),
        crate::url::MaskShape::Ellipse,
    );
    assert_eq!(alpha_at(&ellipse, 50, 30), 255);
    assert!(alpha_at(&ellipse, 50, 0) >= 250);
    // The ellipse touches the left edge at the vertical
    // center: inside the feather there, fully outside
    // higher up.
    assert!(alpha_at(&ellipse, 0, 30) >= 150);
    assert_eq!(alpha_at(&ellipse, 0, 15), 0);
    assert_eq!(alpha_at(&ellipse, 0, 0), 0);
}

#[test]
fn border_paints_inside_the_frame() {
    let img = draw_border(&opaque(10, 10, [255, 0, 0]), 2, [0, 255, 0]);
    assert_eq!((img.width(), img.height()), (10, 10));
    let rgba = img.into_rgba8();
    for (x, y) in [(0, 0), (1, 5), (9, 9), (5, 0), (8, 8)] {
        assert_eq!(
            rgba.get_pixel(x, y).0,
            [0, 255, 0, 255],
            "band pixel {x},{y}"
        );
    }
    for (x, y) in [(2, 2), (5, 5), (7, 7)] {
        assert_eq!(
            rgba.get_pixel(x, y).0,
            [255, 0, 0, 255],
            "interior pixel {x},{y}"
        );
    }
}

#[test]
fn border_wider_than_half_paints_everything() {
    let img = draw_border(&opaque(10, 10, [255, 0, 0]), 5, [0, 0, 255]);
    assert_eq!(img.into_rgba8().get_pixel(5, 5).0, [0, 0, 255, 255]);
}

#[test]
fn sepia_tones_with_the_standard_matrix() {
    let (img, _) = apply_filters(solid(128, 128, 128), &[Filter::Sepia]);
    assert_eq!(first_pixel(&img), [173, 154, 120]);
    let (img, _) = apply_filters(solid(0, 0, 0), &[Filter::Sepia]);
    assert_eq!(first_pixel(&img), [0, 0, 0]);
}

#[test]
fn sepia_keeps_alpha() {
    let mut img = DynamicImage::new_rgba8(2, 2);
    for p in img.as_mut_rgba8().unwrap().pixels_mut() {
        *p = image::Rgba([100, 150, 200, 77]);
    }
    let out = sepia(img);
    assert_eq!(out.to_rgba8().get_pixel(0, 0)[3], 77);
}

fn pixel_row(pixels: &[[u8; 3]]) -> DynamicImage {
    let mut img = DynamicImage::new_rgb8(pixels.len() as u32, 1);
    for (x, _, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
        *p = Rgb(pixels[x as usize]);
    }
    img
}

fn row_rgb(img: &DynamicImage) -> Vec<[u8; 3]> {
    img.to_rgb8().pixels().map(|p| [p[0], p[1], p[2]]).collect()
}

#[test]
fn curve_identity_is_a_no_op() {
    let img = striped();
    let before = img.to_rgb8().into_raw();
    for channel in [
        CurveChannel::Rgb,
        CurveChannel::R,
        CurveChannel::G,
        CurveChannel::B,
    ] {
        let (out, _) = apply_filters(
            img.clone(),
            &[Filter::Curve(channel, vec![(0, 0), (255, 255)])],
        );
        assert_eq!(out.to_rgb8().into_raw(), before, "channel {channel:?}");
    }
}

#[test]
fn curve_two_point_lift_matches_hand_computed_values() {
    // curve(r,0,64,255,255): black lifts to 64, white
    // stays, and 128 lands at 64 + round(191 * 128 / 255)
    // = 64 + 96 = 160.
    let img = pixel_row(&[[0, 10, 20], [128, 10, 20], [255, 10, 20]]);
    let (out, _) = apply_filters(
        img,
        &[Filter::Curve(CurveChannel::R, vec![(0, 64), (255, 255)])],
    );
    assert_eq!(
        row_rgb(&out),
        vec![[64, 10, 20], [160, 10, 20], [255, 10, 20]]
    );
}

#[test]
fn curve_interpolation_rounds_half_away_from_zero() {
    // Points (0,0) and (2,1): input 1 sits exactly on the
    // midpoint 0.5 and rounds away from zero, to 1. The
    // descending mirror (0,1),(2,0) sends 1 to 0 by the
    // same rule.
    let img = pixel_row(&[[0, 0, 0], [1, 0, 0], [2, 0, 0]]);
    let (out, _) = apply_filters(
        img.clone(),
        &[Filter::Curve(CurveChannel::R, vec![(0, 0), (2, 1)])],
    );
    assert_eq!(row_rgb(&out), vec![[0, 0, 0], [1, 0, 0], [1, 0, 0]]);
    let (out, _) = apply_filters(img, &[Filter::Curve(CurveChannel::R, vec![(0, 1), (2, 0)])]);
    assert_eq!(row_rgb(&out), vec![[1, 0, 0], [0, 0, 0], [0, 0, 0]]);
}

#[test]
fn curve_channels_are_independent_and_alpha_survives() {
    let mut img = DynamicImage::new_rgba8(1, 1);
    img.as_mut_rgba8().unwrap().get_pixel_mut(0, 0).0 = [100, 150, 200, 77];
    let (out, _) = apply_filters(
        img,
        &[Filter::Curve(CurveChannel::R, vec![(0, 64), (255, 255)])],
    );
    // Red: 64 + round(191 * 100 / 255) = 64 + 75 = 139.
    // Green, blue, and alpha are not the curve's business.
    assert_eq!(out.to_rgba8().get_pixel(0, 0).0, [139, 150, 200, 77]);
}

#[test]
fn curve_clamps_outside_the_point_range() {
    // curve(rgb,32,10,224,245): everything at or below 32
    // is 10, everything at or above 224 is 245, and the
    // middle interpolates (100 -> 93, and 128 is the
    // segment midpoint: 10 + round(235 * 96 / 192) = 128).
    let img = pixel_row(&[
        [0, 0, 0],
        [31, 31, 31],
        [32, 32, 32],
        [100, 100, 100],
        [128, 128, 128],
        [224, 224, 224],
        [255, 255, 255],
    ]);
    let (out, _) = apply_filters(
        img,
        &[Filter::Curve(CurveChannel::Rgb, vec![(32, 10), (224, 245)])],
    );
    let mapped: Vec<u8> = row_rgb(&out).iter().map(|p| p[0]).collect();
    assert_eq!(mapped, vec![10, 10, 10, 93, 128, 245, 245]);
}

#[test]
fn curve_uses_the_segment_around_the_input() {
    // Three points: a steep rise to white at 128, then
    // flat. 64 is the first segment's exact midpoint
    // (255 * 64 / 128 = 127.5, rounding to 128); 200 sits
    // in the flat segment and stays 255.
    let img = pixel_row(&[[64, 64, 64], [128, 128, 128], [200, 200, 200]]);
    let (out, _) = apply_filters(
        img,
        &[Filter::Curve(
            CurveChannel::Rgb,
            vec![(0, 0), (128, 255), (255, 255)],
        )],
    );
    assert_eq!(
        row_rgb(&out),
        vec![[128, 128, 128], [255, 255, 255], [255, 255, 255]]
    );
}

#[test]
fn curves_compose_in_chain_order() {
    // An rgb curve followed by an r curve: the second
    // sees the first's output (in the other order the red
    // would end at 128, not 255).
    let img = pixel_row(&[[255, 255, 255]]);
    let (out, _) = apply_filters(
        img,
        &[
            Filter::Curve(CurveChannel::Rgb, vec![(0, 0), (255, 128)]),
            Filter::Curve(CurveChannel::R, vec![(0, 0), (128, 255)]),
        ],
    );
    assert_eq!(row_rgb(&out), vec![[255, 128, 128]]);
}

fn gray_grid(rows: &[&[u8]]) -> DynamicImage {
    let mut img = DynamicImage::new_rgb8(rows[0].len() as u32, rows.len() as u32);
    for (y, row) in rows.iter().enumerate() {
        for (x, v) in row.iter().enumerate() {
            img.as_mut_rgb8()
                .unwrap()
                .get_pixel_mut(x as u32, y as u32)
                .0 = [*v, *v, *v];
        }
    }
    img
}

fn gray_values(img: &DynamicImage) -> Vec<Vec<u8>> {
    let rgb = img.to_rgb8();
    (0..rgb.height())
        .map(|y| (0..rgb.width()).map(|x| rgb.get_pixel(x, y)[0]).collect())
        .collect()
}

/// The gradient fixture several convolution tests share.
fn gradient() -> DynamicImage {
    gray_grid(&[&[0, 10, 20], &[30, 40, 50], &[60, 70, 80]])
}

#[test]
fn convolution_identity_is_a_no_op() {
    let img = striped();
    let before = img.to_rgb8().into_raw();
    let mut identity5 = vec![0.0; 25];
    identity5[12] = 1.0;
    for kernel in [
        (vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], false),
        (vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], true),
        (identity5, false),
    ] {
        let (out, _) = apply_filters(img.clone(), &[Filter::Convolution(kernel.0, kernel.1)]);
        assert_eq!(out.to_rgb8().into_raw(), before);
    }
}

#[test]
fn convolution_sharpen_matches_hand_computed_pixels() {
    // The classic sharpen kernel on a lone bright pixel:
    // the center is 5 * 50 - (10 + 10 + 10 + 10) = 210,
    // and every other window's sum goes negative (the
    // bright pixel is at most one cross tap) and clamps
    // to 0.
    let img = gray_grid(&[&[0, 10, 0], &[10, 50, 10], &[0, 10, 0]]);
    let (out, _) = apply_filters(
        img,
        &[Filter::Convolution(
            vec![0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0],
            false,
        )],
    );
    assert_eq!(
        gray_values(&out),
        vec![vec![0, 0, 0], vec![0, 210, 0], vec![0, 0, 0]]
    );
}

#[test]
fn convolution_normalized_box_averages_the_window() {
    // An all-ones kernel with normalize on is the box
    // average over the edge-replicated window: the center
    // is 360 / 9 = 40, the corner counts its replicated
    // taps (120 / 9, rounding to 13), and so on.
    let (out, _) = apply_filters(gradient(), &[Filter::Convolution(vec![1.0; 9], true)]);
    assert_eq!(
        gray_values(&out),
        vec![vec![13, 20, 27], vec![33, 40, 47], vec![53, 60, 67]]
    );
    // Rounding is to nearest: a lone 5 in a 3x3 of zeros
    // averages to 5 / 9 = 0.55..., which rounds up to 1
    // in every window (each contains the 5 exactly once).
    let lone = gray_grid(&[&[0, 0, 0], &[0, 5, 0], &[0, 0, 0]]);
    let (out, _) = apply_filters(lone, &[Filter::Convolution(vec![1.0; 9], true)]);
    assert_eq!(
        gray_values(&out),
        vec![vec![1, 1, 1], vec![1, 1, 1], vec![1, 1, 1]]
    );
}

#[test]
fn convolution_unnormalized_box_sums_and_clamps() {
    // The same kernel without normalize sums the window:
    // the top row stays under the ceiling, everything
    // from the middle row down clamps at 255.
    let (out, _) = apply_filters(gradient(), &[Filter::Convolution(vec![1.0; 9], false)]);
    assert_eq!(
        gray_values(&out),
        vec![
            vec![120, 180, 240],
            vec![255, 255, 255],
            vec![255, 255, 255]
        ]
    );
}

#[test]
fn convolution_zero_sum_kernel_ignores_normalize() {
    // An edge kernel sums to zero, so normalize has no
    // scale to apply: both spellings must produce the
    // kernel exactly as written (center: the cross taps
    // sum to 10 + 30 + 50 + 70 = 160, minus 4 * 40, is 0).
    let kernel = vec![0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0];
    let (plain, _) = apply_filters(gradient(), &[Filter::Convolution(kernel.clone(), false)]);
    let (normalized, _) = apply_filters(gradient(), &[Filter::Convolution(kernel, true)]);
    assert_eq!(plain.to_rgb8().into_raw(), normalized.to_rgb8().into_raw());
    assert_eq!(gray_values(&normalized)[1][1], 0);
}

#[test]
fn convolution_five_by_five_applies_its_taps() {
    // A single tap in the kernel's top-left corner: every
    // output reads the source two pixels up and left,
    // clamped to the frame (the edge replication rule).
    let img = gray_grid(&[
        &[0, 1, 2, 3, 4],
        &[5, 6, 7, 8, 9],
        &[10, 11, 12, 13, 14],
        &[15, 16, 17, 18, 19],
        &[20, 21, 22, 23, 24],
    ]);
    let mut kernel = vec![0.0; 25];
    kernel[0] = 1.0;
    let (out, _) = apply_filters(img, &[Filter::Convolution(kernel, false)]);
    assert_eq!(
        gray_values(&out),
        vec![
            vec![0, 0, 0, 1, 2],
            vec![0, 0, 0, 1, 2],
            vec![0, 0, 0, 1, 2],
            vec![5, 5, 5, 6, 7],
            vec![10, 10, 10, 11, 12],
        ]
    );
}

#[test]
fn convolution_edges_replicate_at_the_corners() {
    // A kernel with only its right tap set: each output
    // is the pixel to its right, and the right column
    // reads itself (the edge, replicated).
    let img = gray_grid(&[&[10, 20], &[30, 40]]);
    let (out, _) = apply_filters(
        img,
        &[Filter::Convolution(
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            false,
        )],
    );
    assert_eq!(gray_values(&out), vec![vec![20, 20], vec![40, 40]]);
}

#[test]
fn convolution_decimal_kernel_matches_hand_computed_values() {
    // The 3x3 gaussian (weights sum to 1) on the linear
    // gradient: the center is the corners at 1/16
    // (160 / 16 = 10), the edges at 1/8 (160 / 8 = 20),
    // and the center at 1/4 (40 / 4 = 10), totaling 40.
    let gaussian = vec![
        0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625,
    ];
    let (out, _) = apply_filters(gradient(), &[Filter::Convolution(gaussian.clone(), false)]);
    assert_eq!(gray_values(&out)[1][1], 40);
    // On a flat field the weights summing to 1 leave the
    // value alone, exactly (every weight is dyadic).
    let flat = gray_grid(&[&[80, 80, 80], &[80, 80, 80], &[80, 80, 80]]);
    let (out, _) = apply_filters(flat, &[Filter::Convolution(gaussian, false)]);
    assert_eq!(
        gray_values(&out),
        vec![vec![80, 80, 80], vec![80, 80, 80], vec![80, 80, 80]]
    );
}

#[test]
fn convolution_never_touches_alpha() {
    let mut img = DynamicImage::new_rgba8(2, 2);
    let px = [
        [10, 20, 30, 77],
        [40, 50, 60, 77],
        [70, 80, 90, 77],
        [100, 110, 120, 77],
    ];
    for (i, p) in img.as_mut_rgba8().unwrap().pixels_mut().enumerate() {
        *p = image::Rgba(px[i]);
    }
    let (out, _) = apply_filters(img, &[Filter::Convolution(vec![1.0; 9], true)]);
    let rgba = out.to_rgba8();
    // The box average at (0, 0) weights the corner pixel
    // four times, its edge neighbors twice, and the far
    // corner once: (360, 450, 540) / 9 = (40, 50, 60).
    assert_eq!(rgba.get_pixel(0, 0).0, [40, 50, 60, 77]);
    assert!(rgba.pixels().all(|p| p[3] == 77));
}

#[test]
fn noise_zero_is_a_byte_identical_no_op() {
    // On an RGBA image (variant and alpha included) and
    // an RGB one: amount 0 returns the image untouched.
    let mut img = DynamicImage::new_rgba8(3, 2);
    for (i, p) in img.as_mut_rgba8().unwrap().pixels_mut().enumerate() {
        *p = image::Rgba([10 + 30 * i as u8, 20 + 20 * i as u8, 30 + 10 * i as u8, 77]);
    }
    let before = img.to_rgba8().into_raw();
    let (out, _) = apply_filters(img, &[Filter::Noise(0)]);
    assert!(matches!(out, DynamicImage::ImageRgba8(_)));
    assert_eq!(out.to_rgba8().into_raw(), before);
    let img = striped();
    let before = img.to_rgb8().into_raw();
    let (out, _) = apply_filters(img, &[Filter::Noise(0)]);
    assert_eq!(out.to_rgb8().into_raw(), before);
}

#[test]
fn noise_matches_the_documented_generator() {
    // The generator, pinned end to end: the seed is the
    // FNV-1a hash of the RGBA buffer folded with the
    // amount, the stream is SplitMix64, one draw per
    // pixel in row-major order, and the delta is the
    // draw modulo 2A+1 minus A, with A the rounded
    // amplitude (255 at amount 100, 102 at 40). For
    // this row the amount-100 draws give +220, +131,
    // and -238 (the last pixel clamps at 0), and the
    // amount-40 draws give -61, +20, and +22.
    let img = pixel_row(&[[10, 20, 30], [100, 110, 120], [200, 210, 220]]);
    let (out, _) = apply_filters(img.clone(), &[Filter::Noise(100)]);
    assert_eq!(
        row_rgb(&out),
        vec![[230, 240, 250], [231, 241, 251], [0, 0, 0]]
    );
    let (out, _) = apply_filters(img, &[Filter::Noise(40)]);
    assert_eq!(
        row_rgb(&out),
        vec![[0, 0, 0], [120, 130, 140], [222, 232, 242]]
    );
}

#[test]
fn noise_is_deterministic_and_depends_on_the_amount() {
    // The same input and amount render the same bytes
    // every run; a different amount is a different
    // field (and a different image).
    let img = striped();
    let (a, _) = apply_filters(img.clone(), &[Filter::Noise(60)]);
    let (b, _) = apply_filters(img.clone(), &[Filter::Noise(60)]);
    assert_eq!(a.to_rgb8().into_raw(), b.to_rgb8().into_raw());
    let (c, _) = apply_filters(img, &[Filter::Noise(61)]);
    assert_ne!(a.to_rgb8().into_raw(), c.to_rgb8().into_raw());
}

#[test]
fn noise_full_strength_moves_a_flat_field_both_ways() {
    // On a flat mid-gray field at 100, some pixels land
    // above 128 and some below: the field swings both
    // ways and stays inside the byte range by clamping.
    let (out, _) = apply_filters(opaque(64, 64, [128, 128, 128]), &[Filter::Noise(100)]);
    let raw = out.to_rgb8().into_raw();
    assert!(raw.iter().any(|&v| v > 128));
    assert!(raw.iter().any(|&v| v < 128));
    // No channel of any pixel may differ from another:
    // even on the way through the clamp the value is
    // one per pixel, shared by all three channels.
    for px in out.to_rgb8().pixels() {
        assert_eq!(px[0], px[1]);
        assert_eq!(px[1], px[2]);
    }
}

#[test]
fn noise_is_monochrome_and_preserves_alpha() {
    // Mid-range pixels and a modest amount (amplitude
    // 51 at 20), so nothing clamps and the per-pixel
    // delta must be identical on all three channels.
    let mut img = DynamicImage::new_rgba8(4, 1);
    let src = [
        [110, 120, 130, 77],
        [125, 135, 145, 77],
        [100, 140, 120, 77],
        [130, 115, 105, 77],
    ];
    for (i, p) in img.as_mut_rgba8().unwrap().pixels_mut().enumerate() {
        *p = image::Rgba(src[i]);
    }
    let (out, _) = apply_filters(img, &[Filter::Noise(20)]);
    let rgba = out.to_rgba8();
    for (x, s) in src.iter().enumerate() {
        let o = rgba.get_pixel(x as u32, 0).0;
        let delta = i32::from(o[0]) - i32::from(s[0]);
        assert_eq!(i32::from(o[1]) - i32::from(s[1]), delta, "pixel {x}");
        assert_eq!(i32::from(o[2]) - i32::from(s[2]), delta, "pixel {x}");
        assert_eq!(o[3], 77, "alpha survives at pixel {x}");
    }
}

#[test]
fn duotone_maps_shadows_and_highlights() {
    let bw = |px: [u8; 3]| {
        let mut img = DynamicImage::new_rgb8(1, 1);
        img.as_mut_rgb8().unwrap().get_pixel_mut(0, 0).0 = px;
        duotone(img.clone(), [255, 0, 0], [0, 0, 255])
            .to_rgb8()
            .get_pixel(0, 0)
            .0
    };
    assert_eq!(bw([0, 0, 0]), [255, 0, 0]);
    assert_eq!(bw([255, 255, 255]), [0, 0, 255]);
    // Pure red's luminance is 0.299 of the way from red to blue.
    assert_eq!(bw([255, 0, 0]), [179, 0, 76]);
}

#[test]
fn monochrome_scales_luminance_by_the_color() {
    let one = |px: [u8; 3]| {
        let mut img = DynamicImage::new_rgb8(1, 1);
        img.as_mut_rgb8().unwrap().get_pixel_mut(0, 0).0 = px;
        monochrome(img.clone(), [255, 128, 0])
            .to_rgb8()
            .get_pixel(0, 0)
            .0
    };
    assert_eq!(one([255, 255, 255]), [255, 128, 0]);
    assert_eq!(one([0, 0, 0]), [0, 0, 0]);
    assert_eq!(one([128, 128, 128]), [128, 64, 0]);
}

#[test]
fn pixelate_averages_blocks() {
    // 4x4, four solid quadrants: 2px blocks stay pure.
    let mut img = DynamicImage::new_rgb8(4, 4);
    for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
        *p = match (x < 2, y < 2) {
            (true, true) => Rgb([255, 0, 0]),
            (false, true) => Rgb([0, 0, 255]),
            (true, false) => Rgb([0, 255, 0]),
            (false, false) => Rgb([255, 255, 255]),
        };
    }
    let out = pixelate(&img, 2);
    let rgb = out.to_rgb8();
    assert_eq!(rgb.get_pixel(0, 0).0, [255, 0, 0]);
    assert_eq!(rgb.get_pixel(3, 0).0, [0, 0, 255]);
    assert_eq!(rgb.get_pixel(0, 3).0, [0, 255, 0]);
    assert_eq!(rgb.get_pixel(3, 3).0, [255, 255, 255]);
    // One 4x4 block over a half-red half-blue image averages.
    let mut img = DynamicImage::new_rgb8(4, 2);
    for (x, _y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
        *p = if x < 2 {
            Rgb([255, 0, 0])
        } else {
            Rgb([0, 0, 255])
        };
    }
    let out = pixelate(&img, 4);
    assert_eq!(out.to_rgb8().get_pixel(0, 0).0, [128, 0, 128]);
    assert_eq!(out.to_rgb8().get_pixel(3, 1).0, [128, 0, 128]);
    // Block 1 is the identity.
    assert_eq!(pixelate(&img, 1).to_rgb8(), img.into_rgb8());
}

#[test]
fn flip_mirrors_both_ways() {
    let mut img = DynamicImage::new_rgb8(2, 1);
    img.as_mut_rgb8().unwrap().get_pixel_mut(0, 0).0 = [255, 0, 0];
    img.as_mut_rgb8().unwrap().get_pixel_mut(1, 0).0 = [0, 0, 255];
    let (out, _) = apply_filters(
        img.clone(),
        &[Filter::Flip(crate::url::FlipDirection::Horizontal)],
    );
    assert_eq!(out.to_rgb8().get_pixel(0, 0).0, [0, 0, 255]);
    assert_eq!(out.to_rgb8().get_pixel(1, 0).0, [255, 0, 0]);
    let mut img = DynamicImage::new_rgb8(1, 2);
    img.as_mut_rgb8().unwrap().get_pixel_mut(0, 0).0 = [255, 0, 0];
    img.as_mut_rgb8().unwrap().get_pixel_mut(0, 1).0 = [0, 0, 255];
    let (out, _) = apply_filters(img, &[Filter::Flip(crate::url::FlipDirection::Vertical)]);
    assert_eq!(out.to_rgb8().get_pixel(0, 0).0, [0, 0, 255]);
    assert_eq!(out.to_rgb8().get_pixel(0, 1).0, [255, 0, 0]);
}

#[test]
fn blur_region_smooths_only_the_region() {
    // 1px checkerboard: maximal local contrast.
    let mut img = DynamicImage::new_rgb8(40, 40);
    for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
        let v = if (x + y) % 2 == 0 { 0 } else { 255 };
        *p = Rgb([v, v, v]);
    }
    let out = blur_region(&img, 10, 10, 20, 20, None).unwrap();
    let rgb = out.to_rgb8();
    // Outside the region: pixel-exact.
    assert_eq!(rgb.get_pixel(0, 0).0, [0, 0, 0]);
    assert_eq!(rgb.get_pixel(39, 39).0, [0, 0, 0]);
    assert_eq!(rgb.get_pixel(8, 15).0, [255, 255, 255]);
    // Inside: the checkerboard is smoothed toward mid gray,
    // so neighbors differ by far less than the original 255.
    let a = rgb.get_pixel(20, 20)[0] as i32;
    let b = rgb.get_pixel(21, 20)[0] as i32;
    assert!((a - b).abs() < 60, "neighbors {a} vs {b}");
    assert!(a > 60 && a < 195, "smoothed toward gray, got {a}");
}

#[test]
fn pixelate_region_averages_inside_only() {
    // Left half red, right half blue, edge at x = 20.
    let mut img = DynamicImage::new_rgb8(40, 40);
    for (x, _y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
        *p = if x < 20 {
            Rgb([255, 0, 0])
        } else {
            Rgb([0, 0, 255])
        };
    }
    // Region starts at x = 11, so its 2px blocks straddle
    // the color edge at block [19, 20].
    let out = pixelate_region(&img, 11, 10, 20, 20);
    let rgb = out.to_rgb8();
    assert_eq!(rgb.get_pixel(19, 15).0, [128, 0, 128]);
    assert_eq!(rgb.get_pixel(20, 15).0, [128, 0, 128]);
    // Pure blocks inside stay pure.
    assert_eq!(rgb.get_pixel(12, 15).0, [255, 0, 0]);
    assert_eq!(rgb.get_pixel(29, 15).0, [0, 0, 255]);
    // Outside the region nothing changes.
    assert_eq!(rgb.get_pixel(5, 15).0, [255, 0, 0]);
    assert_eq!(rgb.get_pixel(35, 15).0, [0, 0, 255]);
    assert_eq!(rgb.get_pixel(19, 5).0, [255, 0, 0]);
}

fn face_box(x: u32, y: u32, w: u32, h: u32) -> crate::ai::FaceBox {
    crate::ai::FaceBox {
        x,
        y,
        w,
        h,
        score: 0.9,
    }
}

/// A mid-gray 64x64 frame with a one-pixel black and white
/// checkerboard inside the box: the finest detail a face
/// can carry.
fn checker_face_scene(fx: u32, fy: u32, fw: u32, fh: u32) -> DynamicImage {
    let mut img = image::RgbImage::from_pixel(64, 64, Rgb([90, 90, 90]));
    for (x, y, p) in img.enumerate_pixels_mut() {
        if (fx..fx + fw).contains(&x) && (fy..fy + fh).contains(&y) {
            *p = if (x + y) % 2 == 0 {
                Rgb([0, 0, 0])
            } else {
                Rgb([255, 255, 255])
            };
        }
    }
    DynamicImage::ImageRgb8(img)
}

/// Asserts the checkerboard in the face box is gone: no
/// pixel inside keeps its checker value (the averages of
/// cells at least 2x2 px are never pure black or white),
/// and every pixel shares its value with a horizontal and a
/// vertical neighbor inside the box, so no per-pixel detail
/// survives anywhere, edges included. Pixels outside the
/// box must be untouched.
fn assert_checker_face_redacted(
    src: &DynamicImage,
    out: &DynamicImage,
    (fx, fy, fw, fh): (u32, u32, u32, u32),
) {
    let (src, out) = (src.to_rgb8(), out.to_rgb8());
    assert_eq!(out.dimensions(), src.dimensions());
    let inside = |x: u32, y: u32| (fx..fx + fw).contains(&x) && (fy..fy + fh).contains(&y);
    for (x, y, p) in out.enumerate_pixels() {
        if !inside(x, y) {
            assert_eq!(p, src.get_pixel(x, y), "outside pixel ({x}, {y}) changed");
            continue;
        }
        assert_ne!(p, src.get_pixel(x, y), "face pixel ({x}, {y}) survived");
        let same = |nx: u32, ny: u32| inside(nx, ny) && out.get_pixel(nx, ny) == p;
        assert!(
            (x > 0 && same(x - 1, y)) || same(x + 1, y),
            "face pixel ({x}, {y}) differs from both horizontal neighbors"
        );
        assert!(
            (y > 0 && same(x, y - 1)) || same(x, y + 1),
            "face pixel ({x}, {y}) differs from both vertical neighbors"
        );
    }
}

#[test]
fn pixelate_faces_redacts_small_faces() {
    // Issue #401: under 20 px the old block was 1 px and the
    // face came back bit-identical; at 20-29 px nearest
    // sampling kept a quarter of the pixels. The odd sizes
    // would leave a 1 px strip under top-left-anchored
    // 2 px blocks.
    for (fw, fh) in [(12, 12), (24, 24), (8, 6), (13, 17), (19, 25), (29, 29)] {
        let src = checker_face_scene(10, 11, fw, fh);
        let out = pixelate_faces(&src, &[face_box(10, 11, fw, fh)]);
        assert_checker_face_redacted(&src, &out, (10, 11, fw, fh));
    }
}

#[test]
fn pixelate_faces_redacts_a_face_cut_by_the_frame() {
    // The box hangs 20 px past the right edge: only the
    // visible 6 px are redacted, and the cell size still
    // comes from the whole face, so the strip is a single
    // averaged cell wide.
    let src = checker_face_scene(58, 20, 6, 26);
    let out = pixelate_faces(&src, &[face_box(58, 20, 26, 26)]);
    assert_checker_face_redacted(&src, &out, (58, 20, 6, 26));
    let rgb = out.to_rgb8();
    assert!((58..64).all(|x| rgb.get_pixel(x, 22) == rgb.get_pixel(58, 22)));
}

#[test]
fn pixelate_faces_caps_the_cells_across_a_face() {
    // A 48 px face is 8 cells of 6 px across. A horizontal
    // ramp gives every cell column its own average.
    let mut img = image::RgbImage::from_pixel(64, 64, Rgb([90, 90, 90]));
    for (x, y, p) in img.enumerate_pixels_mut() {
        if (8..56).contains(&x) && (8..56).contains(&y) {
            let v = ((x - 8) * 5) as u8;
            *p = Rgb([v, v, v]);
        }
    }
    let out = pixelate_faces(&DynamicImage::ImageRgb8(img), &[face_box(8, 8, 48, 48)]);
    let rgb = out.to_rgb8();
    let runs = (9..56)
        .filter(|&x| rgb.get_pixel(x, 30) != rgb.get_pixel(x - 1, 30))
        .count()
        + 1;
    assert_eq!(runs, 8);
    assert_eq!(rgb.get_pixel(8, 30).0, [13, 13, 13]);
    assert_eq!(rgb.get_pixel(7, 30).0, [90, 90, 90]);
}

#[test]
fn mosaic_averages_alpha_too() {
    let mut img = image::RgbaImage::new(4, 4);
    for (x, y, p) in img.enumerate_pixels_mut() {
        let a = if (x + y) % 2 == 0 { 0 } else { 255 };
        *p = image::Rgba([200, 10, 10, a]);
    }
    let out = mosaic(&DynamicImage::ImageRgba8(img), 2).to_rgba8();
    assert!(out.pixels().all(|p| p.0 == [200, 10, 10, 128]));
}

#[test]
fn pixelate_faces_survives_degenerate_boxes() {
    // Boxes come from a model: empty, off-frame, enormous
    // and one-pixel boxes must neither panic nor touch
    // pixels outside the frame's part of the box.
    let src = checker_face_scene(0, 0, 64, 64);
    let boxes = [
        face_box(0, 0, 0, 0),
        face_box(70, 70, 10, 10),
        face_box(5, 5, 1, 1),
        face_box(0, 0, u32::MAX, u32::MAX),
    ];
    for b in boxes {
        let out = pixelate_faces(&src, &[b]);
        assert_eq!((out.width(), out.height()), (64, 64));
    }
    let out = pixelate_faces(&src, &[face_box(0, 0, u32::MAX, u32::MAX)]);
    let rgb = out.to_rgb8();
    // The whole frame is one cell of a huge face.
    assert!(rgb.pixels().all(|p| p == rgb.get_pixel(0, 0)));
}

#[test]
fn mosaic_cell_edges_do_not_overflow_on_long_sides() {
    // 50,000 cells along 100,000 px: i * side passes
    // u32::MAX, so the edges must be computed in u64.
    let mut img = image::RgbaImage::new(100_000, 1);
    img.put_pixel(99_999, 0, image::Rgba([255, 255, 255, 255]));
    let out = mosaic(&DynamicImage::ImageRgba8(img), 2).to_rgba8();
    assert_eq!(out.dimensions(), (100_000, 1));
    assert_eq!(out.get_pixel(99_998, 0).0, [128, 128, 128, 128]);
    assert_eq!(out.get_pixel(99_997, 0).0, [0, 0, 0, 0]);
}

/// `checker_face_scene` stored as RGBA, the face box at
/// `alpha` and the rest opaque (issue #444): a face a PNG
/// viewer sees faintly or not at all, but that the face
/// detector (RGB only) finds and a JPEG output (alpha
/// dropped) shows.
fn checker_face_scene_at_alpha(bx: (u32, u32, u32, u32), alpha: u8) -> image::RgbaImage {
    let (fx, fy, fw, fh) = bx;
    let mut rgba = checker_face_scene(fx, fy, fw, fh).to_rgba8();
    for (x, y, p) in rgba.enumerate_pixels_mut() {
        if (fx..fx + fw).contains(&x) && (fy..fy + fh).contains(&y) {
            p[3] = alpha;
        }
    }
    rgba
}

/// The least-squares slope of the output's red channel
/// against the source's inside the box, the measure issue
/// #444 reported: 1 when the original passes through, 0
/// when none of it survives.
fn leak_slope(
    src: &image::RgbaImage,
    out: &image::RgbaImage,
    (fx, fy, fw, fh): (u32, u32, u32, u32),
) -> f64 {
    let pairs: Vec<(f64, f64)> = (fy..fy + fh)
        .flat_map(|y| (fx..fx + fw).map(move |x| (x, y)))
        .map(|(x, y)| (src.get_pixel(x, y)[0] as f64, out.get_pixel(x, y)[0] as f64))
        .collect();
    let n = pairs.len() as f64;
    let ms = pairs.iter().map(|p| p.0).sum::<f64>() / n;
    let mo = pairs.iter().map(|p| p.1).sum::<f64>() / n;
    let cov: f64 = pairs.iter().map(|(s, o)| (s - ms) * (o - mo)).sum();
    let var: f64 = pairs.iter().map(|(s, _)| (s - ms).powi(2)).sum();
    cov / var
}

/// Redact the checker face at alpha 0, 128, and 255, and
/// require that none of its original color survive under
/// any of them (issue #444). The stored RGB, all a JPEG
/// output writes, must be exactly what the opaque face
/// redacts to and must not keep the source's value; the
/// face keeps its own alpha, and nothing outside the box
/// changes.
fn assert_redacted_under_any_alpha(
    name: &str,
    bx: (u32, u32, u32, u32),
    redact: impl Fn(&DynamicImage) -> DynamicImage,
) {
    let (fx, fy, fw, fh) = bx;
    let inside = |x: u32, y: u32| (fx..fx + fw).contains(&x) && (fy..fy + fh).contains(&y);
    let opaque = DynamicImage::ImageRgba8(checker_face_scene_at_alpha(bx, 255));
    let opaque = redact(&opaque).to_rgba8();
    for alpha in [0u8, 128, 255] {
        let src = checker_face_scene_at_alpha(bx, alpha);
        let out = redact(&DynamicImage::ImageRgba8(src.clone())).to_rgba8();
        let slope = leak_slope(&src, &out, bx);
        assert!(
            slope.abs() < 0.05,
            "{name} at alpha {alpha}: slope {slope:.3}"
        );
        for (x, y, p) in out.enumerate_pixels() {
            let s = src.get_pixel(x, y);
            if !inside(x, y) {
                assert_eq!(p, s, "{name} at alpha {alpha}: outside ({x}, {y}) changed");
                continue;
            }
            let at = format!("{name} at alpha {alpha}, ({x}, {y})");
            assert_eq!(
                p.0[..3],
                opaque.get_pixel(x, y).0[..3],
                "{at}: not the opaque redaction"
            );
            assert_ne!(p.0[..3], s.0[..3], "{at}: kept its color");
            assert_eq!(p[3], alpha, "{at}: alpha changed");
        }
    }
}

#[test]
fn face_redaction_replaces_the_face_under_any_alpha() {
    // Issue #444: the redacted face was composited over the
    // original, so at alpha 0 the original passed through
    // whole and at alpha 128 a third of it did.
    let bx = (8, 8, 40, 40);
    let faces = [face_box(8, 8, 40, 40)];
    assert_redacted_under_any_alpha("blur_faces", bx, |img| {
        blur_faces(img, &faces, None).unwrap()
    });
    assert_redacted_under_any_alpha("pixelate_faces", bx, |img| pixelate_faces(img, &faces));
}

#[test]
fn region_redaction_replaces_the_region_under_any_alpha() {
    // The model-free forms share the leak and need no model
    // to reach it. A 40 px region pixelates in 4 px blocks,
    // so no 1 px strip is left at the edges.
    let bx = (8, 8, 40, 40);
    assert_redacted_under_any_alpha("blur_region", bx, |img| {
        blur_region(img, 8, 8, 40, 40, None).unwrap()
    });
    assert_redacted_under_any_alpha("pixelate_region", bx, |img| {
        pixelate_region(img, 8, 8, 40, 40)
    });
}

#[test]
fn class_redaction_replaces_the_object_under_any_alpha() {
    // Rectangle and silhouette forms: the silhouette form
    // takes the rectangle form's pixels inside the mask.
    let bx = (8, 8, 40, 40);
    let boxes = [detection(8, 8, 40, 40)];
    let masked = || {
        vec![MaskedDetection {
            detection: detection(8, 8, 40, 40),
            mask: image::GrayImage::from_pixel(40, 40, image::Luma([255])),
        }]
    };
    assert_redacted_under_any_alpha("blur_detections", bx, |img| {
        blur_detections(img, &boxes, None).unwrap()
    });
    assert_redacted_under_any_alpha("pixelate_detections", bx, |img| {
        pixelate_detections(img, &boxes)
    });
    assert_redacted_under_any_alpha("blur_detections_masked", bx, |img| {
        blur_detections_masked(img, &masked(), None).unwrap()
    });
    assert_redacted_under_any_alpha("pixelate_detections_masked", bx, |img| {
        pixelate_detections_masked(img, &masked())
    });
}

#[test]
fn region_redaction_redacts_detail_carried_in_alpha() {
    // Text or a logo is often drawn as alpha over one flat
    // color, so its shape lives in the alpha channel alone.
    // An alpha that varies is redacted like the color,
    // never kept or composited back.
    let src = DynamicImage::ImageRgba8(image::RgbaImage::from_fn(64, 64, |x, y| {
        image::Rgba([20, 20, 20, if (x + y) % 2 == 0 { 0 } else { 255 }])
    }));
    for (name, out) in [
        (
            "blur_region",
            blur_region(&src, 8, 8, 40, 40, None).unwrap(),
        ),
        ("pixelate_region", pixelate_region(&src, 8, 8, 40, 40)),
    ] {
        let out = out.to_rgba8();
        for (x, y, p) in out.enumerate_pixels() {
            if (8..48).contains(&x) && (8..48).contains(&y) {
                assert!(
                    p[3] > 60 && p[3] < 195,
                    "{name}: alpha {} at ({x}, {y})",
                    p[3]
                );
            } else {
                assert_eq!(p, src.to_rgba8().get_pixel(x, y), "{name}: ({x}, {y})");
            }
        }
    }
}

#[test]
fn redaction_keeps_an_opaque_rgba_image_opaque() {
    // An opaque image stored with an alpha channel. A
    // 240 px box blurs at sigma 30 as a region and 40 as a
    // face, where the blur alone takes a constant 255 alpha
    // below 255: the box's uniform alpha is kept, or the
    // redacted box of an opaque image would come back
    // translucent.
    let src = DynamicImage::ImageRgba8(image::RgbaImage::from_fn(260, 260, |x, y| {
        let v = ((x * 7 + y * 13) % 256) as u8;
        image::Rgba([v, 255 - v, v / 2, 255])
    }));
    let region = src.crop_imm(10, 10, 240, 240);
    let blurred = blur_fast(&region, 30.0).to_rgba8();
    assert!(blurred.get_pixel(120, 120)[3] < 255, "the premise moved");
    let out = blur_region(&src, 10, 10, 240, 240, None)
        .unwrap()
        .to_rgba8();
    assert!(out.pixels().all(|p| p[3] == 255));
    // Inside the box the color is the blur's, with nothing
    // of the original blended in.
    for (x, y, p) in blurred.enumerate_pixels() {
        assert_eq!(out.get_pixel(x + 10, y + 10).0[..3], p.0[..3], "({x}, {y})");
    }
    let out = blur_faces(&src, &[face_box(10, 10, 240, 240)], None)
        .unwrap()
        .to_rgba8();
    assert!(out.pixels().all(|p| p[3] == 255));
}

#[test]
fn put_redacted_clips_to_the_image() {
    // The composite it replaced clipped silently, so an
    // empty or out-of-frame rectangle must stay a no-op
    // rather than panic on the alpha scan.
    let mut base = image::RgbaImage::from_pixel(8, 8, image::Rgba([1, 2, 3, 255]));
    let done = image::RgbaImage::from_pixel(4, 4, image::Rgba([9, 9, 9, 9]));
    put_redacted(&mut base, done.clone(), 8, 0);
    put_redacted(&mut base, done.clone(), 0, u32::MAX);
    put_redacted(&mut base, image::RgbaImage::new(0, 0), 0, 0);
    assert!(base.pixels().all(|p| p.0 == [1, 2, 3, 255]));
    // Hanging over the corner: the 2x2 in frame is
    // written, and its uniform alpha is kept.
    put_redacted(&mut base, done, 6, 6);
    assert_eq!(base.get_pixel(7, 7).0, [9, 9, 9, 255]);
    assert_eq!(base.get_pixel(6, 6).0, [9, 9, 9, 255]);
    assert_eq!(base.get_pixel(5, 7).0, [1, 2, 3, 255]);
}

fn striped() -> DynamicImage {
    let mut img = DynamicImage::new_rgb8(64, 64);
    for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
        let v = ((x + y) % 256) as u8;
        *p = Rgb([v, 255 - v, (x * y % 256) as u8]);
    }
    img
}

fn solid(r: u8, g: u8, b: u8) -> DynamicImage {
    let mut img = DynamicImage::new_rgb8(4, 4);
    for p in img.as_mut_rgb8().unwrap().pixels_mut() {
        *p = Rgb([r, g, b]);
    }
    img
}

fn first_pixel(img: &DynamicImage) -> [u8; 3] {
    let rgb = img.to_rgb8();
    let p = rgb.pixels().next().unwrap();
    [p[0], p[1], p[2]]
}

#[test]
fn later_directives_win() {
    let (img, opts) = apply_filters(
        striped(),
        &[
            Filter::Quality(Quality::Value(10)),
            Filter::Quality(Quality::Value(90)),
            Filter::Format(FormatSpec::plain(ImageFormat::Png)),
            Filter::Format(FormatSpec::plain(ImageFormat::Jpeg)),
        ],
    );
    assert_eq!((img.width(), img.height()), (64, 64));
    assert_eq!(
        opts,
        EncodeOptions {
            format: Some(ImageFormat::Jpeg),
            format_options: FormatOptions::default(),
            quality: Some(Quality::Value(90)),
            palette: None,
            no_upscale: false,
            fill: None,
            max_bytes: None,
            keep_profile: false,
        }
    );
}

#[test]
fn no_upscale_is_collected_as_a_directive() {
    let img = DynamicImage::new_rgb8(8, 8);
    let (_, opts) = apply_filters(img, &[Filter::NoUpscale]);
    assert!(opts.no_upscale);
}

#[test]
fn sharpen_increases_edge_contrast() {
    // Vertical step: left half 100, right half 200.
    let mut img = DynamicImage::new_rgb8(16, 16);
    for (x, _, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
        let v = if x < 8 { 100 } else { 200 };
        *p = Rgb([v, v, v]);
    }
    let (img, _) = apply_filters(img, &[Filter::Sharpen(1.0)]);
    let rgb = img.into_rgb8();
    let left = rgb.get_pixel(7, 8)[0] as i32;
    let right = rgb.get_pixel(8, 8)[0] as i32;
    assert!(right - left > 100, "left {left} right {right}");
}

#[test]
fn saturation_scales_color() {
    let (img, _) = apply_filters(solid(200, 100, 50), &[Filter::Saturation(0.0)]);
    let p = first_pixel(&img);
    assert_eq!(p[0], p[1]);
    assert_eq!(p[1], p[2]);
    let (img, _) = apply_filters(solid(200, 100, 50), &[Filter::Saturation(1.0)]);
    assert_eq!(first_pixel(&img), [200, 100, 50]);
    let (img, _) = apply_filters(solid(200, 100, 50), &[Filter::Saturation(2.0)]);
    let p = first_pixel(&img);
    assert!(p[0] as i32 - p[2] as i32 > 150, "pixel {p:?}");
}

#[test]
fn rgb_offsets_channels() {
    let (img, _) = apply_filters(solid(100, 100, 100), &[Filter::Rgb(50, -30, 0)]);
    assert_eq!(first_pixel(&img), [150, 70, 100]);
}

#[test]
fn equalize_spreads_the_range() {
    // All pixels in a narrow band, 100..=107.
    let mut img = DynamicImage::new_rgb8(16, 16);
    for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
        let v = 100 + ((x + y) % 8) as u8;
        *p = Rgb([v, v, v]);
    }
    let (img, _) = apply_filters(img, &[Filter::Equalize]);
    let rgb = img.into_rgb8();
    let min = rgb.pixels().map(|p| p[0]).min().unwrap();
    let max = rgb.pixels().map(|p| p[0]).max().unwrap();
    assert_eq!(min, 0);
    assert_eq!(max, 255);
}

#[test]
fn rotate_45_expands_the_canvas_and_fills_corners() {
    let img = DynamicImage::ImageRgb8(solid(255, 0, 0).to_rgb8());
    let img = img.resize_exact(4, 2, image::imageops::FilterType::Nearest);
    let (out, _) = apply_filters(img, &[Filter::Rotate(45.0)]);
    // ceil(4*cos45 + 2*sin45) = ceil(4.243) = 5 on both axes.
    assert_eq!((out.width(), out.height()), (5, 5));
    let rgb = out.to_rgb8();
    // The uncovered corner takes the default background.
    assert_eq!(rgb.get_pixel(0, 0).0, [255, 255, 255]);
    // The middle is still the source color.
    assert_eq!(rgb.get_pixel(2, 2).0, [255, 0, 0]);
}

#[test]
fn rotate_45_uses_the_request_fill_for_corners() {
    let img = DynamicImage::ImageRgb8(solid(255, 0, 0).to_rgb8()).resize_exact(
        4,
        2,
        image::imageops::FilterType::Nearest,
    );
    let (out, _) = apply_filters_filled(img, &[Filter::Rotate(45.0)], Some([10, 20, 30]));
    assert_eq!(out.to_rgb8().get_pixel(0, 0).0, [10, 20, 30]);
}

#[test]
fn rotate_45_keeps_alpha_corners_transparent() {
    let mut rgba = image::RgbaImage::new(4, 2);
    for p in rgba.pixels_mut() {
        *p = image::Rgba([255, 0, 0, 255]);
    }
    let (out, _) = apply_filters(DynamicImage::ImageRgba8(rgba), &[Filter::Rotate(45.0)]);
    let rgba = out.to_rgba8();
    assert_eq!(rgba.get_pixel(0, 0).0[3], 0);
    assert_eq!(rgba.get_pixel(2, 2).0, [255, 0, 0, 255]);
}

#[test]
fn rotate_positive_angles_turn_clockwise() {
    // A marker above center moves to the upper right under
    // a positive (clockwise) rotation, matching rotate90.
    let mut rgb = image::RgbImage::new(20, 20);
    rgb.put_pixel(10, 2, image::Rgb([255, 0, 0]));
    let (out, _) = apply_filters(DynamicImage::ImageRgb8(rgb), &[Filter::Rotate(45.0)]);
    let rgb = out.to_rgb8();
    // Track the reddest pixel (the white corners have no
    // red-over-green excess).
    let (mut bx, mut by, mut best) = (0u32, 0u32, 0i16);
    for (x, y, p) in rgb.enumerate_pixels() {
        let excess = p.0[0] as i16 - p.0[1] as i16;
        if excess > best {
            best = excess;
            bx = x;
            by = y;
        }
    }
    assert!(bx > out.width() / 2, "marker x {bx} of {}", out.width());
    assert!(by < out.height() / 2, "marker y {by} of {}", out.height());
}

#[test]
fn rotate_full_turns_are_identity() {
    let (out, _) = apply_filters(DynamicImage::new_rgb8(4, 2), &[Filter::Rotate(360.0)]);
    assert_eq!((out.width(), out.height()), (4, 2));
}

#[test]
fn rotations_land_on_the_predicted_canvas() {
    // The engine checks `rotated_canvas` against the pixel
    // budget before rotating, so the rotation must
    // allocate exactly that canvas. The negative angles
    // here fold to a value that a second fold moves, the
    // case where the two used to disagree by a pixel
    // (issue #415): -113.0128 on 640x480 predicted 692x777
    // and allocated 693x777.
    for (w, h, degrees) in [
        (640u32, 480u32, -113.0128f32),
        (1000, 1, -105.01277),
        (1000, 1, -108.00192),
        (24, 18, 91.0),
        (24, 18, 45.0),
        (24, 18, -30.25),
        (7, 3, 359.9),
        (7, 3, 0.0001),
    ] {
        let (out, _) = apply_filters(DynamicImage::new_rgb8(w, h), &[Filter::Rotate(degrees)]);
        assert_eq!(
            (out.width(), out.height()),
            rotated_canvas(w, h, degrees),
            "rotate({degrees}) of {w}x{h}"
        );
    }
    assert_eq!(rotated_canvas(640, 480, -113.0128), (692, 777));
    assert_eq!(rotated_canvas(24, 18, 91.0), (19, 25));
}

#[test]
fn rotate_swaps_dimensions() {
    let (img, _) = apply_filters(DynamicImage::new_rgb8(4, 2), &[Filter::Rotate(90.0)]);
    assert_eq!((img.width(), img.height()), (2, 4));
    let (img, _) = apply_filters(DynamicImage::new_rgb8(4, 2), &[Filter::Rotate(-90.0)]);
    assert_eq!((img.width(), img.height()), (2, 4));
    let (img, _) = apply_filters(DynamicImage::new_rgb8(4, 2), &[Filter::Rotate(180.0)]);
    assert_eq!((img.width(), img.height()), (4, 2));
}

#[test]
fn fill_and_max_bytes_are_collected() {
    let (_, opts) = apply_filters(
        solid(1, 2, 3),
        &[Filter::Fill([255, 136, 0]), Filter::MaxBytes(5000)],
    );
    assert_eq!(opts.fill, Some([255, 136, 0]));
    assert_eq!(opts.max_bytes, Some(5000));
}

#[test]
fn absorb_merges_directives() {
    let mut base = EncodeOptions::default();
    base.absorb(EncodeOptions {
        format: Some(ImageFormat::Png),
        format_options: FormatOptions::default(),
        quality: None,
        palette: Some(64),
        no_upscale: true,
        fill: None,
        max_bytes: Some(100),
        keep_profile: true,
    });
    base.absorb(EncodeOptions {
        format: None,
        format_options: FormatOptions::default(),
        quality: Some(Quality::Value(50)),
        palette: None,
        no_upscale: false,
        fill: None,
        max_bytes: None,
        keep_profile: false,
    });
    assert_eq!(
        base,
        EncodeOptions {
            keep_profile: true,
            format: Some(ImageFormat::Png),
            format_options: FormatOptions::default(),
            quality: Some(Quality::Value(50)),
            palette: Some(64),
            no_upscale: true,
            fill: None,
            max_bytes: Some(100),
        }
    );
}

#[test]
fn empty_filters_are_noop() {
    let (img, opts) = apply_filters(striped(), &[]);
    assert_eq!((img.width(), img.height()), (64, 64));
    assert_eq!(opts, EncodeOptions::default());
}

#[test]
fn grayscale_desaturates() {
    let (img, _) = apply_filters(striped(), &[Filter::Grayscale]);
    for p in img.into_rgb8().pixels() {
        assert_eq!(p[0], p[1]);
        assert_eq!(p[1], p[2]);
    }
}

fn rgba_mixed() -> DynamicImage {
    let mut img = DynamicImage::new_rgba8(4, 4);
    for (x, y, p) in img.as_mut_rgba8().unwrap().enumerate_pixels_mut() {
        let a = if x == 0 && y == 0 { 0 } else { 200 };
        *p = image::Rgba([180, 110, 50, a]);
    }
    img
}

#[test]
fn color_filters_preserve_alpha() {
    let cases: [(&str, Filter); 6] = [
        ("grayscale", Filter::Grayscale),
        ("contrast", Filter::Contrast(2.0)),
        ("sharpen", Filter::Sharpen(1.0)),
        ("saturation", Filter::Saturation(0.5)),
        ("rgb", Filter::Rgb(10, -10, 20)),
        ("equalize", Filter::Equalize),
    ];
    for (name, filter) in cases {
        let (out, _) = apply_filters(rgba_mixed(), std::slice::from_ref(&filter));
        assert!(out.color().has_alpha(), "{name} dropped the alpha channel");
        assert_eq!(
            alpha_at(&out, 0, 0),
            0,
            "{name} changed a transparent pixel"
        );
        assert_eq!(alpha_at(&out, 3, 3), 200, "{name} changed an opaque pixel");
    }
}

#[test]
fn grayscale_then_radius_keeps_alpha_and_gray() {
    let mut img = DynamicImage::new_rgba8(20, 20);
    for p in img.as_mut_rgba8().unwrap().pixels_mut() {
        *p = image::Rgba([200, 120, 60, 255]);
    }
    let (out, _) = apply_filters(img, &[Filter::Grayscale, Filter::Radius([8, 8, 8, 8])]);
    assert!(out.color().has_alpha());
    assert_eq!(alpha_at(&out, 0, 0), 0);
    let rgba = out.to_rgba8();
    let c = *rgba.get_pixel(10, 10);
    assert_eq!(c[3], 255);
    assert_eq!(c[0], c[1]);
    assert_eq!(c[1], c[2]);
}

#[test]
fn invert_flips_channels() {
    let (img, _) = apply_filters(solid(10, 128, 250), &[Filter::Invert]);
    assert_eq!(first_pixel(&img), [245, 127, 5]);
}

#[test]
fn brighten_shifts_and_clamps() {
    let (img, _) = apply_filters(solid(100, 150, 200), &[Filter::Brighten(10)]);
    assert_eq!(first_pixel(&img), [110, 160, 210]);

    let (img, _) = apply_filters(solid(200, 200, 200), &[Filter::Brighten(100)]);
    assert_eq!(first_pixel(&img), [255, 255, 255]);

    let (img, _) = apply_filters(solid(100, 150, 200), &[Filter::Brighten(-300)]);
    assert_eq!(first_pixel(&img), [0, 0, 0]);
}

#[test]
fn blur_smooths_a_hard_edge() {
    let mut img = DynamicImage::new_rgb8(32, 32);
    for (x, _y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
        let v = if x < 16 { 0 } else { 255 };
        *p = Rgb([v, v, v]);
    }
    let (img, _) = apply_filters(img, &[Filter::Blur(2.0)]);
    let rgb = img.into_rgb8();
    // Pixels hugging the edge are now a blend of black and white.
    for (x, y, p) in rgb.enumerate_pixels() {
        if x == 15 || x == 16 {
            assert!(p[0] > 0 && p[0] < 255, "pixel ({x}, {y}) = {p:?}");
        }
    }
}

#[test]
fn huerotate_rotates_and_comes_back() {
    let (img, _) = apply_filters(solid(255, 0, 0), &[Filter::HueRotate(90)]);
    let px = first_pixel(&img);
    // The hue moved: pure red is gone.
    assert_ne!(px, [255, 0, 0]);

    // A full turn is (approximately) the identity.
    let (img, _) = apply_filters(solid(200, 100, 50), &[Filter::HueRotate(360)]);
    let px = first_pixel(&img);
    for (got, want) in px.iter().zip([200, 100, 50]) {
        assert!((*got as i16 - want as i16).abs() <= 1, "pixel = {px:?}");
    }
}

#[test]
fn contrast_pushes_values_from_mid_gray() {
    let (img, _) = apply_filters(solid(100, 100, 100), &[Filter::Contrast(2.0)]);
    let px = first_pixel(&img);
    assert!(px[0] < 100, "pixel = {px:?}");

    let (img, _) = apply_filters(solid(200, 200, 200), &[Filter::Contrast(2.0)]);
    let px = first_pixel(&img);
    assert!(px[0] > 200, "pixel = {px:?}");
}

#[test]
fn gamma_one_is_the_identity() {
    let lut = gamma_lut(1.0);
    for (i, entry) in lut.iter().enumerate() {
        assert_eq!(*entry, i as u8);
    }
    let (img, _) = apply_filters(solid(37, 128, 250), &[Filter::Gamma(1.0)]);
    assert_eq!(first_pixel(&img), [37, 128, 250]);
}

#[test]
fn gamma_moves_midtones_and_keeps_the_endpoints() {
    // 255 * (64 / 255)^(1/2) rounds to 128; squared
    // (gamma 0.5) it rounds to 16.
    let (img, _) = apply_filters(solid(64, 64, 64), &[Filter::Gamma(2.0)]);
    assert_eq!(first_pixel(&img), [128, 128, 128]);
    let (img, _) = apply_filters(solid(64, 64, 64), &[Filter::Gamma(0.5)]);
    assert_eq!(first_pixel(&img), [16, 16, 16]);
    for gamma in [0.45, 2.2, 10.0] {
        let (img, _) = apply_filters(solid(0, 255, 0), &[Filter::Gamma(gamma)]);
        assert_eq!(first_pixel(&img), [0, 255, 0], "gamma {gamma}");
    }
    // The table is monotone at the extremes of the range.
    for gamma in [0.1, 10.0] {
        let lut = gamma_lut(gamma);
        assert!(lut.windows(2).all(|w| w[0] <= w[1]), "gamma {gamma}");
    }
}

#[test]
fn gamma_preserves_alpha() {
    let img = DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
        2,
        2,
        image::Rgba([64, 32, 16, 128]),
    ));
    let (out, _) = apply_filters(img, &[Filter::Gamma(2.0)]);
    let px = out.to_rgba8().get_pixel(0, 0).0;
    assert_eq!(px[3], 128);
    assert_eq!(px[0], 128);
}

#[test]
fn gamma_lut_sanitizes_out_of_contract_values() {
    // The parser never produces these; the table builder
    // still refuses to reach powf with them.
    for bad in [f32::NAN, f32::INFINITY, 0.0, -2.0] {
        let lut = gamma_lut(bad);
        assert_eq!(lut[64], 64, "gamma {bad}");
        assert_eq!(lut[200], 200, "gamma {bad}");
    }
    assert_eq!(gamma_lut(100.0), gamma_lut(MAX_GAMMA));
}

#[test]
fn exposure_zero_stops_is_the_identity() {
    let lut = exposure_lut(0.0);
    for (i, entry) in lut.iter().enumerate() {
        assert_eq!(*entry, i as u8);
    }
    let (img, _) = apply_filters(solid(37, 128, 250), &[Filter::Exposure(0.0)]);
    assert_eq!(first_pixel(&img), [37, 128, 250]);
}

#[test]
fn exposure_multiplies_in_linear_light() {
    // One stop doubles the light: a mid gray at 128 has
    // linear value about 0.216, doubled to 0.432, which
    // encodes back to 176. Minus one stop halves it and
    // lands on 92. Two stops on 64 (linear about 0.051)
    // land on 125, and minus two stops on 200 land on
    // 106: the same stop count moves shadows less than
    // highlights in encoded values, which is the point
    // of working in linear light.
    let (img, _) = apply_filters(solid(128, 128, 128), &[Filter::Exposure(1.0)]);
    assert_eq!(first_pixel(&img), [176, 176, 176]);
    let (img, _) = apply_filters(solid(128, 128, 128), &[Filter::Exposure(-1.0)]);
    assert_eq!(first_pixel(&img), [92, 92, 92]);
    let (img, _) = apply_filters(solid(64, 64, 64), &[Filter::Exposure(2.0)]);
    assert_eq!(first_pixel(&img), [125, 125, 125]);
    let (img, _) = apply_filters(solid(200, 200, 200), &[Filter::Exposure(-2.0)]);
    assert_eq!(first_pixel(&img), [106, 106, 106]);
    // Black stays black at any setting; white stays
    // white under brightening and gives ground only
    // under darkening (255 at minus one stop is 188).
    for stops in [-10.0, -1.0, 0.5, 3.0, 10.0] {
        let (img, _) = apply_filters(solid(0, 0, 0), &[Filter::Exposure(stops)]);
        assert_eq!(first_pixel(&img), [0, 0, 0], "stops {stops}");
    }
    let (img, _) = apply_filters(solid(255, 255, 255), &[Filter::Exposure(1.0)]);
    assert_eq!(first_pixel(&img), [255, 255, 255]);
    let (img, _) = apply_filters(solid(255, 255, 255), &[Filter::Exposure(-1.0)]);
    assert_eq!(first_pixel(&img), [188, 188, 188]);
    // The table is monotone across the range.
    for stops in [-10.0, -0.5, 0.5, 10.0] {
        let lut = exposure_lut(stops);
        assert!(lut.windows(2).all(|w| w[0] <= w[1]), "stops {stops}");
    }
}

#[test]
fn exposure_preserves_alpha() {
    let img = DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
        2,
        2,
        image::Rgba([128, 64, 32, 128]),
    ));
    let (out, _) = apply_filters(img, &[Filter::Exposure(1.0)]);
    let px = out.to_rgba8().get_pixel(0, 0).0;
    assert_eq!(px[3], 128);
    assert_eq!(px[0], 176);
}

#[test]
fn exposure_lut_sanitizes_out_of_contract_values() {
    // The parser never produces these; the table builder
    // still refuses to reach powf with them.
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let lut = exposure_lut(bad);
        assert_eq!(lut[64], 64, "stops {bad}");
        assert_eq!(lut[200], 200, "stops {bad}");
    }
    assert_eq!(exposure_lut(100.0), exposure_lut(MAX_EXPOSURE));
    assert_eq!(exposure_lut(-100.0), exposure_lut(-MAX_EXPOSURE));
}

#[test]
fn highlights_shadows_zero_zero_is_the_identity() {
    let (img, _) = apply_filters(solid(200, 100, 50), &[Filter::HighlightsShadows(0.0, 0.0)]);
    assert_eq!(first_pixel(&img), [200, 100, 50]);
}

#[test]
fn highlights_shadows_move_their_own_ends() {
    // Pure black under full shadows lifts by the full
    // quarter scale, 64 levels; under full highlights
    // it does not move at all.
    let (img, _) = apply_filters(solid(0, 0, 0), &[Filter::HighlightsShadows(0.0, 100.0)]);
    assert_eq!(first_pixel(&img), [64, 64, 64]);
    let (img, _) = apply_filters(solid(0, 0, 0), &[Filter::HighlightsShadows(100.0, 0.0)]);
    assert_eq!(first_pixel(&img), [0, 0, 0]);
    // Pure white under full negative highlights gives
    // back the same 64 levels; under full shadows it
    // does not move.
    let (img, _) = apply_filters(
        solid(255, 255, 255),
        &[Filter::HighlightsShadows(-100.0, 0.0)],
    );
    assert_eq!(first_pixel(&img), [191, 191, 191]);
    let (img, _) = apply_filters(
        solid(255, 255, 255),
        &[Filter::HighlightsShadows(0.0, 100.0)],
    );
    assert_eq!(first_pixel(&img), [255, 255, 255]);
    // A mid gray feels both ends at about a quarter of
    // their full strength (its weights are near 0.25):
    // full shadows move 128 to 144, full negative
    // highlights to 112.
    let (img, _) = apply_filters(
        solid(128, 128, 128),
        &[Filter::HighlightsShadows(0.0, 100.0)],
    );
    assert_eq!(first_pixel(&img), [144, 144, 144]);
    let (img, _) = apply_filters(
        solid(128, 128, 128),
        &[Filter::HighlightsShadows(-100.0, 0.0)],
    );
    assert_eq!(first_pixel(&img), [112, 112, 112]);
}

#[test]
fn highlights_shadows_weight_by_luminance() {
    // A dark pixel (32) is a shadow pixel: full shadows
    // move it 49 levels, full highlights only 1. A
    // bright pixel (224) is the mirror image.
    let (img, _) = apply_filters(solid(32, 32, 32), &[Filter::HighlightsShadows(0.0, 100.0)]);
    assert_eq!(first_pixel(&img), [81, 81, 81]);
    let (img, _) = apply_filters(solid(32, 32, 32), &[Filter::HighlightsShadows(100.0, 0.0)]);
    assert_eq!(first_pixel(&img), [33, 33, 33]);
    let (img, _) = apply_filters(
        solid(224, 224, 224),
        &[Filter::HighlightsShadows(0.0, 100.0)],
    );
    assert_eq!(first_pixel(&img), [225, 225, 225]);
    let (img, _) = apply_filters(
        solid(224, 224, 224),
        &[Filter::HighlightsShadows(-100.0, 0.0)],
    );
    assert_eq!(first_pixel(&img), [175, 175, 175]);
    // The offset is shared by the channels: a color
    // pixel keeps its channel order and gaps up to
    // clamping (luma of (200, 100, 50) is about 124, so
    // full shadows shift by 17 and full highlights
    // by 15).
    let (img, _) = apply_filters(
        solid(200, 100, 50),
        &[Filter::HighlightsShadows(0.0, 100.0)],
    );
    assert_eq!(first_pixel(&img), [217, 117, 67]);
    let (img, _) = apply_filters(
        solid(200, 100, 50),
        &[Filter::HighlightsShadows(100.0, 0.0)],
    );
    assert_eq!(first_pixel(&img), [215, 115, 65]);
}

#[test]
fn highlights_shadows_preserves_alpha() {
    let img = DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
        2,
        2,
        image::Rgba([0, 0, 0, 128]),
    ));
    let (out, _) = apply_filters(img, &[Filter::HighlightsShadows(0.0, 100.0)]);
    let px = out.to_rgba8().get_pixel(0, 0).0;
    assert_eq!(px[3], 128);
    assert_eq!(px[0], 64);
}

#[test]
fn highlights_shadows_sanitizes_out_of_contract_values() {
    // The parser never produces these; the engine still
    // clamps instead of shifting by a non-finite or
    // out-of-range amount.
    let (img, _) = apply_filters(
        solid(128, 128, 128),
        &[Filter::HighlightsShadows(f32::NAN, f32::INFINITY)],
    );
    assert_eq!(first_pixel(&img), [128, 128, 128]);
    let (clamped, _) = apply_filters(solid(0, 0, 0), &[Filter::HighlightsShadows(0.0, 250.0)]);
    let (at_cap, _) = apply_filters(solid(0, 0, 0), &[Filter::HighlightsShadows(0.0, 100.0)]);
    assert_eq!(first_pixel(&clamped), first_pixel(&at_cap));
}

#[test]
fn vibrance_zero_is_the_identity() {
    let (img, _) = apply_filters(solid(140, 120, 110), &[Filter::Vibrance(0.0)]);
    assert_eq!(first_pixel(&img), [140, 120, 110]);
}

#[test]
fn vibrance_weights_muted_colors_most() {
    // (140, 120, 110) has HSV saturation about 0.214,
    // so at +100 its factor is about 1.786 and at -100
    // about 0.214: the full swing, in both directions.
    let (img, _) = apply_filters(solid(140, 120, 110), &[Filter::Vibrance(100.0)]);
    assert_eq!(first_pixel(&img), [151, 116, 98]);
    let (img, _) = apply_filters(solid(140, 120, 110), &[Filter::Vibrance(-100.0)]);
    assert_eq!(first_pixel(&img), [128, 123, 121]);
    // A fully saturated pixel does not move at either
    // extreme: its weight is zero.
    for amount in [-100.0, 100.0] {
        let (img, _) = apply_filters(solid(255, 0, 0), &[Filter::Vibrance(amount)]);
        assert_eq!(first_pixel(&img), [255, 0, 0], "amount {amount}");
    }
    // A gray pixel has no chroma to scale and does not
    // move either.
    let (img, _) = apply_filters(solid(100, 100, 100), &[Filter::Vibrance(100.0)]);
    assert_eq!(first_pixel(&img), [100, 100, 100]);
    // Between the extremes, the more saturated pixel
    // moves proportionally less: (200, 80, 40) has
    // saturation 0.8, so its factor is only 1.2, where
    // the muted pixel above took 1.786.
    let (img, _) = apply_filters(solid(200, 80, 40), &[Filter::Vibrance(100.0)]);
    assert_eq!(first_pixel(&img), [217, 73, 25]);
}

#[test]
fn vibrance_preserves_alpha() {
    let img = DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(
        2,
        2,
        image::Rgba([140, 120, 110, 128]),
    ));
    let (out, _) = apply_filters(img, &[Filter::Vibrance(100.0)]);
    let px = out.to_rgba8().get_pixel(0, 0).0;
    assert_eq!(px[3], 128);
    assert_eq!(&px[..3], &[151, 116, 98]);
}

#[test]
fn vibrance_sanitizes_out_of_contract_values() {
    let (img, _) = apply_filters(solid(140, 120, 110), &[Filter::Vibrance(f32::NAN)]);
    assert_eq!(first_pixel(&img), [140, 120, 110]);
    let (clamped, _) = apply_filters(solid(140, 120, 110), &[Filter::Vibrance(250.0)]);
    let (at_cap, _) = apply_filters(solid(140, 120, 110), &[Filter::Vibrance(100.0)]);
    assert_eq!(first_pixel(&clamped), first_pixel(&at_cap));
}

/// A gray image from a value census: `counts` pairs of
/// (value, pixels), laid out in order.
fn census_image(counts: &[(u8, usize)], w: u32) -> DynamicImage {
    let total: usize = counts.iter().map(|c| c.1).sum();
    let h = (total as u32).div_ceil(w);
    let mut img = DynamicImage::new_rgb8(w, h);
    let rgb = img.as_mut_rgb8().unwrap();
    let mut pixels = rgb.pixels_mut();
    for &(value, n) in counts {
        for _ in 0..n {
            if let Some(p) = pixels.next() {
                *p = image::Rgb([value, value, value]);
            }
        }
    }
    img
}

#[test]
fn enhance_stretches_a_compressed_range() {
    // 1% of pixels at 30, 98% at 100, 1% at 220: the
    // 0.5th percentile is 30 and the 99.5th is 220, so
    // 30 maps to 0, 220 to 255, and 100 lands on
    // ((100 - 30) * 255 + 95) / 190 = 94.
    let img = census_image(&[(30, 100), (100, 9800), (220, 100)], 100);
    let (out, _) = apply_filters(img, &[Filter::Enhance]);
    let rgb = out.to_rgb8();
    assert_eq!(rgb.get_pixel(0, 0)[0], 0);
    assert_eq!(rgb.get_pixel(50, 50)[0], 94);
    assert_eq!(rgb.get_pixel(99, 99)[0], 255);
}

#[test]
fn enhance_leaves_flat_and_full_range_images_alone() {
    // A flat channel has no span to stretch.
    let (out, _) = apply_filters(solid(100, 100, 100), &[Filter::Enhance]);
    assert_eq!(first_pixel(&out), [100, 100, 100]);
    // A near-flat channel (span 10, under the floor of
    // 16) is left unmapped too.
    let img = census_image(&[(100, 5000), (110, 5000)], 100);
    let (out, _) = apply_filters(img, &[Filter::Enhance]);
    let rgb = out.to_rgb8();
    assert_eq!(rgb.get_pixel(0, 0)[0], 100);
    assert_eq!(rgb.get_pixel(99, 99)[0], 110);
    // An image already spanning the range maps its
    // percentiles to the same ends and stays put.
    let img = census_image(&[(0, 5000), (255, 5000)], 100);
    let (out, _) = apply_filters(img, &[Filter::Enhance]);
    let rgb = out.to_rgb8();
    assert_eq!(rgb.get_pixel(0, 0)[0], 0);
    assert_eq!(rgb.get_pixel(99, 99)[0], 255);
}

#[test]
fn enhance_is_gentler_than_equalize() {
    // A quarter black, half mid, a quarter white: the
    // percentiles already sit at the ends, so enhance
    // leaves the mid value at 100, while equalize
    // redistributes it to 170.
    let img = census_image(&[(0, 2500), (100, 5000), (255, 2500)], 100);
    let (out, _) = apply_filters(img.clone(), &[Filter::Enhance]);
    assert_eq!(out.to_rgb8().get_pixel(50, 50)[0], 100);
    let (out, _) = apply_filters(img, &[Filter::Equalize]);
    assert_eq!(out.to_rgb8().get_pixel(50, 50)[0], 170);
}

#[test]
fn enhance_stretches_channels_independently() {
    // The red channel carries the compressed census;
    // green and blue are flat and must not move.
    let mut img = DynamicImage::new_rgb8(100, 100);
    {
        let rgb = img.as_mut_rgb8().unwrap();
        for (i, p) in rgb.pixels_mut().enumerate() {
            let r = if i < 100 {
                30
            } else if i >= 9900 {
                220
            } else {
                100
            };
            *p = image::Rgb([r, 90, 200]);
        }
    }
    let (out, _) = apply_filters(img, &[Filter::Enhance]);
    let rgb = out.to_rgb8();
    assert_eq!(rgb.get_pixel(0, 0).0, [0, 90, 200]);
    assert_eq!(rgb.get_pixel(50, 50).0, [94, 90, 200]);
    assert_eq!(rgb.get_pixel(99, 99).0, [255, 90, 200]);
}

#[test]
fn enhance_preserves_alpha() {
    let mut img = DynamicImage::new_rgba8(100, 100);
    {
        let rgba = img.as_mut_rgba8().unwrap();
        for (i, p) in rgba.pixels_mut().enumerate() {
            let v = if i < 100 {
                30
            } else if i >= 9900 {
                220
            } else {
                100
            };
            *p = image::Rgba([v, v, v, 128]);
        }
    }
    let (out, _) = apply_filters(img, &[Filter::Enhance]);
    let px = out.to_rgba8().get_pixel(50, 50).0;
    assert_eq!(px[3], 128);
    assert_eq!(px[0], 94);
}

#[test]
fn palette_filter_sets_the_encode_directive() {
    let (_, opts) = apply_filters(solid(10, 20, 30), &[Filter::Palette(64)]);
    assert_eq!(opts.palette, Some(64));
    // The last one in the chain wins, like quality.
    let (_, opts) = apply_filters(
        solid(10, 20, 30),
        &[Filter::Palette(64), Filter::Palette(16)],
    );
    assert_eq!(opts.palette, Some(16));
    // And the pixels pass through untouched: the
    // directive only steers the encoder.
    let (img, _) = apply_filters(solid(10, 20, 30), &[Filter::Palette(2)]);
    assert_eq!(first_pixel(&img), [10, 20, 30]);
}

#[test]
fn new_filters_compose_in_order() {
    // invert then brighten(10): (10 -> 245 -> 255).
    let (img, _) = apply_filters(solid(10, 10, 10), &[Filter::Invert, Filter::Brighten(10)]);
    assert_eq!(first_pixel(&img), [255, 255, 255]);
}

/// 40x30 `bg` with an `fg` rectangle at (10..30, 8..22).
fn bordered(bg: [u8; 3], fg: [u8; 3]) -> DynamicImage {
    let mut img = DynamicImage::new_rgb8(40, 30);
    for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
        *p = if (10..30).contains(&x) && (8..22).contains(&y) {
            Rgb(fg)
        } else {
            Rgb(bg)
        };
    }
    img
}

#[test]
fn trim_crops_to_the_inner_box() {
    let (img, _) = apply_filters(bordered([255, 255, 255], [255, 0, 0]), &[Filter::Trim(0)]);
    assert_eq!((img.width(), img.height()), (20, 14));
    assert_eq!(first_pixel(&img), [255, 0, 0]);
}

#[test]
fn trim_tolerance_decides_what_counts_as_border() {
    // Corner 255, an inner ring of 245, black center 10..20.
    let build = || {
        let mut img = DynamicImage::new_rgb8(30, 30);
        for (x, y, p) in img.as_mut_rgb8().unwrap().enumerate_pixels_mut() {
            *p = if (10..20).contains(&x) && (10..20).contains(&y) {
                Rgb([0, 0, 0])
            } else if (5..25).contains(&x) && (5..25).contains(&y) {
                Rgb([245, 245, 245])
            } else {
                Rgb([255, 255, 255])
            };
        }
        img
    };
    // Tolerance 10: the ring (10 away) is still border.
    let (img, _) = apply_filters(build(), &[Filter::Trim(10)]);
    assert_eq!((img.width(), img.height()), (10, 10));
    // Tolerance 5: the ring is content.
    let (img, _) = apply_filters(build(), &[Filter::Trim(5)]);
    assert_eq!((img.width(), img.height()), (20, 20));
}

#[test]
fn trim_uniform_image_is_a_noop() {
    let (img, _) = apply_filters(solid(77, 77, 77), &[Filter::Trim(10)]);
    assert_eq!((img.width(), img.height()), (4, 4));
    assert_eq!(first_pixel(&img), [77, 77, 77]);
}

#[test]
fn trim_sides_shaves_after_the_tolerance_pass() {
    // bordered() trims to the 20x14 inner box; the sides
    // then come off that box, not the original frame
    // (sides first would still end at 20x14 after the
    // tolerance pass found the same inner box).
    let (img, _) = apply_filters(
        bordered([255, 255, 255], [255, 0, 0]),
        &[Filter::TrimSides(0, [5, 2, 5, 2])],
    );
    assert_eq!((img.width(), img.height()), (10, 10));
    assert_eq!(first_pixel(&img), [255, 0, 0]);
    // Zero sides are exactly the tolerance-only form.
    let (img, _) = apply_filters(
        bordered([255, 255, 255], [255, 0, 0]),
        &[Filter::TrimSides(0, [0, 0, 0, 0])],
    );
    assert_eq!((img.width(), img.height()), (20, 14));
    // On a uniform image the tolerance pass is a no-op
    // and the sides still apply.
    let (img, _) = apply_filters(
        DynamicImage::new_rgb8(10, 6),
        &[Filter::TrimSides(10, [2, 1, 2, 1])],
    );
    assert_eq!((img.width(), img.height()), (6, 4));
}

#[test]
fn trim_sides_clamps_to_one_surviving_pixel() {
    // Amounts past the frame clamp instead of
    // underflowing: the near side takes from the
    // dimension first, the far side from what is left.
    let (img, _) = apply_filters(
        bordered([255, 255, 255], [255, 0, 0]),
        &[Filter::TrimSides(0, [100, 100, 100, 100])],
    );
    assert_eq!((img.width(), img.height()), (1, 1));
    assert_eq!(first_pixel(&img), [255, 0, 0]);
    // One axis only: 19 from the left of the 20-wide
    // box leaves one column; the right amount gets 0.
    let (img, _) = apply_filters(
        bordered([255, 255, 255], [255, 0, 0]),
        &[Filter::TrimSides(0, [19, 0, 19, 0])],
    );
    assert_eq!((img.width(), img.height()), (1, 14));
}

#[test]
fn trim_window_shaves_the_sides_off_the_found_box() {
    // The window the engine maps the focal forms through
    // and shares across animation frames (issue #465) is
    // the window the filter itself cuts.
    let img = bordered([255, 255, 255], [255, 0, 0]);
    let found = trim_box(&img, 0);
    let window = trim_window((img.width(), img.height()), found, [5, 2, 5, 2]);
    let (cut, _) = apply_filters(img.clone(), &[Filter::TrimSides(0, [5, 2, 5, 2])]);
    assert_eq!((window.2, window.3), (cut.width(), cut.height()));
    let (x, y, w, h) = window;
    assert_eq!(
        img.crop_imm(x, y, w, h).to_rgba8().as_raw(),
        cut.to_rgba8().as_raw()
    );
    // No box: the sides come off the whole frame, and a
    // frame with nothing to cut comes back untouched.
    assert_eq!(trim_window((10, 8), None, [1, 1, 1, 1]), (1, 1, 8, 6));
    assert_eq!(trim_window((10, 8), None, [0; 4]), (0, 0, 10, 8));
}

#[test]
fn trim_survives_degenerate_frames() {
    // A zero-sized frame has no corner to compare with:
    // no box, and the cut leaves it as it is instead of
    // indexing a pixel that is not there.
    let empty = DynamicImage::new_rgba8(0, 0);
    assert_eq!(trim_box(&empty, 10), None);
    let out = trim_to(empty, None, [3, 3, 3, 3]);
    assert_eq!((out.width(), out.height()), (0, 0));
    // A one-pixel frame keeps its pixel whatever the sides.
    let dot = DynamicImage::new_rgba8(1, 1);
    let out = trim_to(dot, None, [u32::MAX; 4]);
    assert_eq!((out.width(), out.height()), (1, 1));
}

fn detection(x: u32, y: u32, w: u32, h: u32) -> crate::ai::Detection {
    crate::ai::Detection {
        label: "thing".to_string(),
        score: 0.9,
        x,
        y,
        w,
        h,
    }
}

#[test]
fn recolor_detections_paints_the_box_and_keeps_alpha() {
    // A translucent red image: the paint replaces RGB
    // inside the box and leaves alpha (and everything
    // outside the box) alone.
    let mut rgba = image::RgbaImage::new(20, 10);
    for p in rgba.pixels_mut() {
        *p = image::Rgba([200, 30, 30, 128]);
    }
    let img = DynamicImage::ImageRgba8(rgba);
    let out = recolor_detections(&img, &[detection(5, 2, 10, 6)], [0, 0, 255]);
    let out = out.to_rgba8();
    assert_eq!(out.get_pixel(10, 5), &image::Rgba([0, 0, 255, 128]));
    assert_eq!(out.get_pixel(0, 0), &image::Rgba([200, 30, 30, 128]));
}

#[test]
fn recolor_masked_paints_only_the_silhouette() {
    // Mask: left half of the box at full strength, one
    // column at half strength (the antialiased edge),
    // the rest background.
    let img = opaque(20, 10, [200, 200, 200]);
    let mask = image::GrayImage::from_fn(10, 6, |lx, _| {
        if lx < 4 {
            image::Luma([255])
        } else if lx == 4 {
            image::Luma([128])
        } else {
            image::Luma([0])
        }
    });
    let masked = vec![MaskedDetection {
        detection: detection(5, 2, 10, 6),
        mask,
    }];
    let out = recolor_detections_masked(&img, &masked, [0, 0, 255]).to_rgb8();
    // Full strength: the paint color.
    assert_eq!(out.get_pixel(6, 4), &Rgb([0, 0, 255]));
    // Half strength: halfway between the source and
    // the paint.
    let edge = out.get_pixel(9, 4);
    assert!(edge[0] > 90 && edge[0] < 110, "{edge:?}");
    assert!(edge[2] > 220 && edge[2] < 235, "{edge:?}");
    // Background inside the box: the source, untouched.
    assert_eq!(out.get_pixel(12, 4), &Rgb([200, 200, 200]));
}

#[test]
fn blur_masked_leaves_the_unmasked_box_pixels_alone() {
    // Checkerboard box: the masked (left) half blurs,
    // the unmasked half keeps its exact pixels, unlike
    // the rectangle form which blurs the whole box.
    let img = DynamicImage::ImageRgb8(image::RgbImage::from_fn(20, 10, |x, y| {
        if (x + y) % 2 == 0 {
            Rgb([0, 0, 0])
        } else {
            Rgb([255, 255, 255])
        }
    }));
    let mask = image::GrayImage::from_fn(10, 6, |lx, _| {
        if lx < 5 {
            image::Luma([255])
        } else {
            image::Luma([0])
        }
    });
    let masked = vec![MaskedDetection {
        detection: detection(5, 2, 10, 6),
        mask,
    }];
    let out = blur_detections_masked(&img, &masked, None)
        .unwrap()
        .to_rgb8();
    // (11, 4): box-local (6, 2), unmasked: (11+4) is
    // odd, so the source pixel is white, exactly.
    assert_eq!(out.get_pixel(11, 4), &Rgb([255, 255, 255]));
    // (7, 4): masked: the blur mixes the checkerboard,
    // so the pixel is neither source extreme.
    let blurred = out.get_pixel(7, 4);
    assert!(blurred[0] > 40 && blurred[0] < 215, "{blurred:?}");
}

#[test]
fn blur_stops_at_the_deadline() {
    // Issue #614: blur(512) on a large canvas ran for minutes
    // in one call, past the request's deadline. Both banded
    // paths now check the deadline before each row: one
    // already passed blurs no row, and one that passes part
    // way stops the pass there.
    let img = DynamicImage::ImageRgb8(image::RgbImage::from_fn(1200, 800, |x, y| {
        Rgb([(x % 256) as u8, (y % 256) as u8, 9])
    }));
    let past = Some(std::time::Instant::now());
    // Sigma 2 takes the ring path, 8.5 and 512 the transient.
    for sigma in [2.0, 8.5, 512.0] {
        assert_eq!(
            blur_fast_within(&img, sigma, past).map(|_| ()),
            Err(DeadlinePassed),
            "sigma {sigma}"
        );
    }
    // A 16-bit or float image takes the same banded passes,
    // and so does the studio shadow's gray blur, which runs
    // in float.
    for deep in [
        DynamicImage::ImageRgb16(img.to_rgb16()),
        DynamicImage::ImageRgba32F(img.to_rgba32f()),
    ] {
        for sigma in [2.0, 8.5, 512.0] {
            assert_eq!(
                blur_fast_within(&deep, sigma, past).map(|_| ()),
                Err(DeadlinePassed),
                "{:?} sigma {sigma}",
                deep.color()
            );
        }
    }
    for sigma in [2.0, 8.5, 512.0] {
        assert_eq!(
            blur_gray_within(&img.to_luma8(), sigma, past).map(|_| ()),
            Err(DeadlinePassed),
            "gray sigma {sigma}"
        );
    }
    // Part way: sigma 512 here takes far longer than 20 ms,
    // and the pass stops within a row of the deadline.
    let started = std::time::Instant::now();
    let deadline = started + std::time::Duration::from_millis(20);
    assert_eq!(
        blur_fast_within(&img, 512.0, Some(deadline)).map(|_| ()),
        Err(DeadlinePassed)
    );
    let took = started.elapsed();
    assert!(took < std::time::Duration::from_secs(2), "{took:?}");
    // The same for a 16-bit image, which before ran as one
    // call to the image crate on one core.
    let deep = DynamicImage::ImageRgb16(img.to_rgb16());
    let started = std::time::Instant::now();
    let deadline = started + std::time::Duration::from_millis(20);
    assert_eq!(
        blur_fast_within(&deep, 512.0, Some(deadline)).map(|_| ()),
        Err(DeadlinePassed)
    );
    let took = started.elapsed();
    assert!(took < std::time::Duration::from_secs(2), "{took:?}");
    // The plain filter path passes the deadline down.
    assert_eq!(
        apply_filters_within(img, &[Filter::Blur(512.0)], None, past).map(|_| ()),
        Err(DeadlinePassed)
    );
}

#[test]
fn redaction_blurs_stop_at_the_deadline() {
    // Issue #614: the face, region, and class blurs run the
    // same banded passes; the face blur's sigma follows the
    // face and has no cap. Each stops at the deadline, and a
    // deadline still ahead redacts exactly as none does.
    let img = DynamicImage::ImageRgb8(image::RgbImage::from_fn(200, 160, |x, y| {
        Rgb([(x * 3 % 256) as u8, (y * 5 % 256) as u8, 40])
    }));
    let faces = [face_box(10, 10, 120, 100)];
    let boxes = [detection(10, 10, 120, 100)];
    let masked = || {
        vec![MaskedDetection {
            detection: detection(10, 10, 120, 100),
            mask: image::GrayImage::from_fn(120, 100, |x, _| image::Luma([(x * 2) as u8])),
        }]
    };
    let past = Some(std::time::Instant::now());
    let ahead = Some(std::time::Instant::now() + std::time::Duration::from_secs(600));
    type Redact<'a> =
        Box<dyn Fn(Option<std::time::Instant>) -> Result<DynamicImage, DeadlinePassed> + 'a>;
    let cases: [(&str, Redact); 4] = [
        (
            "blur_region",
            Box::new(|deadline| blur_region(&img, 10, 10, 120, 100, deadline)),
        ),
        (
            "blur_faces",
            Box::new(|deadline| blur_faces(&img, &faces, deadline)),
        ),
        (
            "blur_detections",
            Box::new(|deadline| blur_detections(&img, &boxes, deadline)),
        ),
        (
            "blur_detections_masked",
            Box::new(|deadline| blur_detections_masked(&img, &masked(), deadline)),
        ),
    ];
    for (name, redact) in cases {
        assert_eq!(redact(past).map(|_| ()), Err(DeadlinePassed), "{name}");
        assert_eq!(
            redact(ahead).unwrap().to_rgba8().into_raw(),
            redact(None).unwrap().to_rgba8().into_raw(),
            "{name}"
        );
    }
    // Nothing to blur is nothing to stop.
    assert!(blur_faces(&img, &[], past).is_ok());
    assert!(blur_detections(&img, &[], past).is_ok());
}
