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

/// Grayscale as it stood before issue #659: a pixel at a
/// time, rounded by `f32::round`.
fn old_grayscale(img: DynamicImage) -> DynamicImage {
    map_rgb(img, |r, g, b| {
        let y = luma(r, g, b).round().clamp(0.0, 255.0) as u8;
        [y, y, y]
    })
}

fn new_grayscale(img: DynamicImage) -> DynamicImage {
    apply_filters(img, &[Filter::Grayscale]).0
}

/// Saturation as it stood before issue #659.
fn old_saturate(mut out: image::RgbImage, amount: f32) -> image::RgbImage {
    let w = out.width() as usize;
    map_rgb_bands::<3>(out.as_mut(), w, &|r, g, b| {
        let luma = 0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32;
        [
            (luma + (r as f32 - luma) * amount).clamp(0.0, 255.0) as u8,
            (luma + (g as f32 - luma) * amount).clamp(0.0, 255.0) as u8,
            (luma + (b as f32 - luma) * amount).clamp(0.0, 255.0) as u8,
        ]
    });
    out
}

#[test]
fn lane_rounding_matches_round_then_clamp() {
    // Issue #659: grayscale rounds without `f32::round`. All
    // 2^32 bit patterns agree (checked outside the suite);
    // here, a pattern in every 4099, which crosses every
    // sign, exponent and NaN payload, then the 64 floats on
    // each side of every whole and every half from -2 to
    // 258, where a rounding rule or a clamp bound could
    // differ, then the values with a rule of their own.
    let old = |x: f32| x.round().clamp(0.0, 255.0) as u8;
    let mut cases = 0u32;
    let mut check = |bits: u32| {
        let x = f32::from_bits(bits);
        assert_eq!(round_lane(x) as u8, old(x), "{x:e} ({bits:#010x})");
        cases += 1;
    };
    for bits in (0..=u32::MAX).step_by(4099) {
        check(bits);
    }
    for half in -4..=516 {
        let center = (half as f32 * 0.5).to_bits();
        for step in 0..=64 {
            check(center.wrapping_add(step));
            check(center.wrapping_sub(step));
        }
    }
    for x in [
        f32::NAN,
        -f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        f32::MIN,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::EPSILON,
        1e-45,
    ] {
        check(x.to_bits());
    }
    assert_eq!(cases, 1_047_809 + 521 * 130 + 10);
    // The halves round away from zero and the result stays
    // inside a byte.
    assert_eq!(round_lane(0.5), 1.0);
    assert_eq!(round_lane(1.5), 2.0);
    assert_eq!(round_lane(2.5), 3.0);
    assert_eq!(round_lane(254.5), 255.0);
    assert_eq!(round_lane(0.49999997), 0.0);
    assert_eq!(round_lane(1e9), 255.0);
    assert_eq!(round_lane(-1e9), 0.0);
    assert_eq!(round_lane(f32::NAN), 0.0);
}

/// Colors a rewrite of the grayscale sum would move. For the
/// first six, the rounded luma depends on the order the
/// three products are added in; for the last six the luma
/// is an exact half, where the rounding rule shows.
const GRAY_WITNESSES: [u32; 12] = [
    0x013fe6, 0x513b20, 0x87e97e, 0xb0f420, 0xd73f51, 0xeadc15, 0x0000fa, 0x2a2c01, 0x540a65,
    0x7e2035, 0xa89625, 0xd2c04f,
];

/// Colors that `saturation(1.2)` written as a blend of the
/// sample and the luma (`s * a + luma * (1 - a)`) would move.
const SATURATION_WITNESSES: [u32; 6] = [0x007828, 0x298949, 0x4f4174, 0x811734, 0xaff148, 0xdd7390];

fn channels(color: u32) -> [u8; 3] {
    let [_, r, g, b] = color.to_be_bytes();
    [r, g, b]
}

/// A lattice of the RGB cube that keeps both ends and the
/// middle of every channel, then the witnesses, as one image
/// of an odd width.
fn sampled_colors() -> image::RgbImage {
    let levels = [
        0u8, 1, 2, 3, 7, 15, 16, 17, 31, 63, 64, 65, 85, 100, 127, 128, 129, 150, 191, 200, 223,
        253, 254, 255,
    ];
    let mut colors = Vec::new();
    for r in levels {
        for g in levels {
            for b in levels {
                colors.push([r, g, b]);
            }
        }
    }
    colors.extend(GRAY_WITNESSES.map(channels));
    colors.extend(SATURATION_WITNESSES.map(channels));
    let width = 173;
    let height = colors.len().div_ceil(width);
    image::RgbImage::from_fn(width as u32, height as u32, |x, y| {
        image::Rgb(colors[(y as usize * width + x as usize) % colors.len()])
    })
}

#[test]
fn the_witness_colors_tell_a_rewrite_apart() {
    // What makes the sample of the next test worth running:
    // each witness is a color on which a plausible other
    // spelling of the arithmetic gives another byte.
    for color in &GRAY_WITNESSES[..6] {
        let [r, g, b] = channels(*color).map(f32::from);
        let regrouped = 0.299 * r + (0.587 * g + 0.114 * b);
        assert_ne!(luma_of(r, g, b).round(), regrouped.round(), "{color:#08x}");
    }
    for color in &GRAY_WITNESSES[6..] {
        let [r, g, b] = channels(*color).map(f32::from);
        let luma = luma_of(r, g, b);
        assert_eq!(luma.fract(), 0.5, "{color:#08x}");
        assert_ne!(luma.round(), luma.round_ties_even(), "{color:#08x}");
    }
    let amount = 1.2f32;
    for color in SATURATION_WITNESSES {
        let rgb = channels(color).map(f32::from);
        let luma = luma_of(rgb[0], rgb[1], rgb[2]);
        let ours = rgb.map(|s| (luma + (s - luma) * amount).clamp(0.0, 255.0) as u8);
        let blended = rgb.map(|s| (s * amount + luma * (1.0 - amount)).clamp(0.0, 255.0) as u8);
        assert_ne!(ours, blended, "{color:#08x}");
    }
}

#[test]
fn grayscale_and_saturation_lanes_match_the_per_pixel_maps() {
    // Issue #659. A sample of the RGB cube with the colors
    // above in it (all 16.7 million agree, checked outside
    // the suite); saturation at amounts across its range and
    // at the values no request can spell.
    let colors = sampled_colors();
    let dynamic = DynamicImage::ImageRgb8(colors.clone());
    assert_eq!(
        old_grayscale(dynamic.clone()).as_bytes(),
        new_grayscale(dynamic).as_bytes()
    );
    for amount in [
        0.0f32,
        0.3,
        0.5,
        1.0,
        1.1,
        1.2,
        1.5,
        2.0,
        7.3,
        255.0,
        -1.0,
        f32::INFINITY,
        f32::NAN,
    ] {
        assert_eq!(
            old_saturate(colors.clone(), amount).as_raw(),
            saturate(colors.clone(), amount).as_raw(),
            "saturation {amount}"
        );
    }
    // Every remainder a row can leave against the lanes, with
    // and without alpha, and a gray source, which is widened
    // to RGB first. Alpha comes through untouched.
    for w in 1..=33 {
        let h = 1 + w % 3;
        let rgb = DynamicImage::ImageRgb8(rgb(w, h, harsh));
        let mut rgba = rgb.to_rgba8();
        for (x, y, p) in rgba.enumerate_pixels_mut() {
            p[3] = smooth(y, x, 3);
        }
        let rgba = DynamicImage::ImageRgba8(rgba);
        let luma = DynamicImage::ImageLuma8(rgb.to_luma8());
        for img in [&rgb, &rgba, &luma] {
            let ours = new_grayscale(img.clone());
            let old = old_grayscale(img.clone());
            assert_eq!(old.color(), ours.color(), "{:?} {w}x{h}", img.color());
            assert_eq!(old.as_bytes(), ours.as_bytes(), "{:?} {w}x{h}", img.color());
        }
        let gray = new_grayscale(rgba.clone()).to_rgba8();
        assert!(
            gray.pixels()
                .zip(rgba.to_rgba8().pixels())
                .all(|(a, b)| a[3] == b[3])
        );
        let raw = rgb.to_rgb8();
        for amount in [0.3f32, 1.2] {
            assert_eq!(
                old_saturate(raw.clone(), amount).as_raw(),
                saturate(raw.clone(), amount).as_raw(),
                "saturation {amount} {w}x{h}"
            );
        }
    }
}

#[test]
fn lanes_split_into_bands_where_the_per_pixel_maps_do() {
    // Issue #659: the lane mapping bands a frame under the
    // rule the per-pixel maps follow, read from one place:
    // the calling thread under 3,000,000 subpixels or 256
    // rows, else one band per core, four at most.
    let threads = band_threads();
    assert!((1..=4).contains(&threads));
    assert_eq!(pointwise_band_rows(2_999_997, 2_997), None);
    assert_eq!(pointwise_band_rows(255 * 30_000, 30_000), None);
    assert_eq!(pointwise_band_rows(0, 0), None);
    let banded = (threads > 1).then(|| 1001usize.div_ceil(threads));
    assert_eq!(pointwise_band_rows(1001 * 3_009, 3_009), banded);
    // Over the threshold the bytes are still the per-pixel
    // maps', bands or not, with a last band that is short
    // and rows that do not fill a block.
    let big = rgb(1003, 1001, harsh);
    assert!(big.as_raw().len() >= 3_000_000);
    let dynamic = DynamicImage::ImageRgb8(big.clone());
    assert_eq!(
        old_grayscale(dynamic.clone()).as_bytes(),
        new_grayscale(dynamic).as_bytes()
    );
    assert_eq!(
        old_saturate(big.clone(), 1.2).as_raw(),
        saturate(big, 1.2).as_raw()
    );
    let mut rgba = DynamicImage::ImageRgb8(rgb(867, 869, smooth)).to_rgba8();
    for (x, y, p) in rgba.enumerate_pixels_mut() {
        p[3] = harsh(x, y, 3);
    }
    assert!(rgba.as_raw().len() >= 3_000_000);
    let rgba = DynamicImage::ImageRgba8(rgba);
    assert_eq!(
        old_grayscale(rgba.clone()).as_bytes(),
        new_grayscale(rgba).as_bytes()
    );
}

/// An 8-bit sample with the arithmetic of `u8` and the blur
/// passes as they stood before issue #660: the trait's own
/// loops, a sample at a time over the kernel.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
struct PlainU8(u8);

impl BlurSample for PlainU8 {
    type Acc = u32;
    fn weight(w: f32) -> u32 {
        blur_q15(w)
    }
    fn widen(self) -> u32 {
        self.0 as u32
    }
    fn mac(acc: u32, a: u32, b: u32) -> u32 {
        acc + a * b
    }
    fn store(acc: u32) -> PlainU8 {
        PlainU8(blur_q15_out(acc))
    }
}

/// `w` by `h` pixels of `N` channels, as bytes and as the
/// plain samples.
fn blur_input<const N: usize>(
    w: usize,
    h: usize,
    content: fn(u32, u32, u32) -> u8,
) -> (Vec<u8>, Vec<PlainU8>) {
    let mut raw = Vec::with_capacity(w * h * N);
    for y in 0..h {
        for x in 0..w {
            for c in 0..N {
                raw.push(content(x as u32, y as u32, c as u32));
            }
        }
    }
    let plain = raw.iter().map(|&v| PlainU8(v)).collect();
    (raw, plain)
}

/// Both passes of one blur, chunked and plain, on `channels`
/// channels.
fn blur_both_ways(w: usize, h: usize, channels: usize, sigma: f32) {
    fn on<const N: usize>(w: usize, h: usize, sigma: f32) {
        for content in [smooth as fn(u32, u32, u32) -> u8, harsh] {
            let (raw, plain) = blur_input::<N>(w, h, content);
            let old = blur_samples::<PlainU8, N>(&plain, w, h, sigma, None).expect("no deadline");
            let new = blur_samples::<u8, N>(&raw, w, h, sigma, None).expect("no deadline");
            assert!(
                old.iter().map(|sample| sample.0).eq(new.iter().copied()),
                "{w}x{h} of {N} channels at sigma {sigma}"
            );
        }
    }
    match channels {
        1 => on::<1>(w, h, sigma),
        2 => on::<2>(w, h, sigma),
        3 => on::<3>(w, h, sigma),
        _ => on::<4>(w, h, sigma),
    }
}

#[test]
fn chunked_blur_sums_match_the_per_sample_sums() {
    // Issue #660. Sigmas on both sides of the kernel size
    // where the ring gives way to the transient (55 taps,
    // between 8 and 8.7), up to kernels far wider than the
    // frame; one to four channels; one row and one column;
    // rows shorter than a chunk, exactly one chunk (64 pixels
    // of four channels), a chunk and one sample (257 of
    // one), and several chunks. The widest kernels run on
    // the smallest frames, for the time a debug build takes.
    let every = [0.3f32, 0.8, 1.0, 2.5, 5.0, 8.0, 8.7, 9.0, 20.0, 75.0, 300.0];
    let mut cases = 0;
    for (w, h, channels, sigmas) in [
        (1, 1, &[1, 2, 3, 4][..], &every[..]),
        (1, 9, &[1, 2, 3, 4], &every[..10]),
        (9, 1, &[1, 2, 3, 4], &every[..10]),
        (2, 2, &[1, 2, 3, 4], &every[..9]),
        (5, 3, &[1, 2, 3, 4], &every),
        (17, 4, &[3, 4], &every[..10]),
        (64, 5, &[1, 4], &every[..9]),
        (257, 3, &[1, 3], &[1.0f32, 2.5, 8.0, 8.7]),
        (513, 2, &[4], &[2.5f32, 9.0]),
    ] {
        for &channels in channels {
            for &sigma in sigmas {
                blur_both_ways(w, h, channels, sigma);
                cases += 1;
            }
        }
    }
    assert_eq!(
        cases,
        4 * (11 + 10 + 10 + 9 + 11) + 2 * 10 + 2 * 9 + 2 * 4 + 2
    );
    // The chunked passes sit behind the same row gate: a
    // deadline that has passed stops them before a row.
    let (raw, _) = blur_input::<3>(64, 5, harsh);
    let past = Some(std::time::Instant::now());
    for sigma in [2.5f32, 20.0] {
        assert_eq!(
            blur_samples::<u8, 3>(&raw, 64, 5, sigma, past),
            Err(DeadlinePassed),
            "sigma {sigma}"
        );
    }
}

#[test]
fn chunked_blur_totals_stay_far_inside_u32() {
    // Issue #660: a chunk's totals are `u32`, and its
    // multiply takes the weights as 16-bit values. A total
    // is at most 255 times the sum of the kernel's weights
    // (255 for the middle sample, 510 for each pair), so for
    // every sigma a request can name, from the smallest to
    // `MAX_BLUR_SIGMA`, and for the pad fill's own range,
    // the weights must fit 15 bits and sum to about 2^15.
    let mut sigmas: Vec<f32> = (0..=640).map(|step| 0.05 + step as f32 * 0.8).collect();
    sigmas.extend([
        f32::MIN_POSITIVE,
        0.3,
        0.8,
        2.0,
        80.0,
        crate::url::MAX_BLUR_SIGMA,
    ]);
    let mut widest = 0;
    for sigma in sigmas {
        let kernel = blur_kernel_for::<u8>(blur_sigma(sigma));
        assert!(
            kernel.iter().all(|&weight| weight < 1 << 15),
            "sigma {sigma}"
        );
        let sum: u64 = kernel.iter().map(|&weight| u64::from(weight)).sum();
        assert!(
            (1 << 14..=(1 << 15) + 64).contains(&sum),
            "sigma {sigma}: the weights sum to {sum}"
        );
        assert!(255 * sum < u64::from(u32::MAX) / 256, "sigma {sigma}");
        widest = widest.max(kernel.len());
    }
    // The widest kernel of the sweep is the grammar's widest.
    assert_eq!(
        widest,
        blur_kernel_for::<u8>(crate::url::MAX_BLUR_SIGMA).len()
    );
    assert!(widest > 2000, "{widest}");
}
