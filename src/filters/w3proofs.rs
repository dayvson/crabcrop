use super::*;

fn photo(w: u32, h: u32) -> DynamicImage {
    let mut buf = vec![0u8; (w * h * 3) as usize];
    let mut state: u64 = 0x1234_5678_9abc_def0;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u8
    };
    for y in 0..h {
        for x in 0..w {
            let i = ((y * w + x) * 3) as usize;
            buf[i] = ((x * 255) / w.max(1)) as u8 ^ (next() >> 3);
            buf[i + 1] = ((y * 255) / h.max(1)) as u8 ^ (next() >> 3);
            buf[i + 2] = (((x + y) * 255) / (w + h).max(1)) as u8 ^ (next() >> 3);
        }
    }
    DynamicImage::ImageRgb8(image::RgbImage::from_raw(w, h, buf).unwrap())
}

fn checker(w: u32, h: u32) -> DynamicImage {
    let mut img = image::RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let v = if (x / 7 + y / 7) % 2 == 0 { 255 } else { 0 };
            img.put_pixel(x, y, image::Rgb([v, 255 - v, (x % 251) as u8]));
        }
    }
    DynamicImage::ImageRgb8(img)
}

fn bytes(img: &DynamicImage) -> (std::mem::Discriminant<DynamicImage>, (u32, u32), Vec<u8>) {
    let raw: Vec<u8> = match img {
        DynamicImage::ImageLuma8(i) => i.as_raw().clone(),
        DynamicImage::ImageLumaA8(i) => i.as_raw().clone(),
        DynamicImage::ImageRgb8(i) => i.as_raw().clone(),
        DynamicImage::ImageRgba8(i) => i.as_raw().clone(),
        other => other.to_rgba8().as_raw().clone(),
    };
    (
        std::mem::discriminant(img),
        (img.width(), img.height()),
        raw,
    )
}

fn variants(img: &DynamicImage) -> Vec<DynamicImage> {
    vec![
        DynamicImage::ImageLuma8(img.to_luma8()),
        DynamicImage::ImageLumaA8(img.to_luma_alpha8()),
        img.to_rgb8().into(),
        img.to_rgba8().into(),
    ]
}

#[test]
fn w3_blur_fast_matches_crate() {
    // The sigma set straddles the crate's ring/transient
    // engine split (kernel 55 at sigma ~8.45).
    for make in [photo as fn(u32, u32) -> DynamicImage, checker] {
        for (w, h) in [(333u32, 257u32), (64, 48), (7, 5), (3, 2), (1, 1), (31, 17)] {
            let base = make(w, h);
            for v in variants(&base) {
                for sigma in [0.3f32, 0.5, 1.0, 2.0, 4.0, 8.0, 8.4, 8.5, 12.0, 16.0, 40.0] {
                    assert_eq!(
                        bytes(&blur_fast(&v, sigma)),
                        bytes(&v.blur(sigma)),
                        "blur sigma {sigma} at {w}x{h}"
                    );
                }
            }
        }
    }
    let big = photo(1200, 900);
    for sigma in [2.0f32, 8.5, 16.0] {
        assert_eq!(
            bytes(&blur_fast(&big, sigma)),
            bytes(&big.blur(sigma)),
            "big blur {sigma}"
        );
    }
    // sigma == 0 follows the crate's 0.8 substitution.
    assert_eq!(bytes(&blur_fast(&big, 0.0)), bytes(&big.blur(0.0)));
}

#[test]
fn w3_blur_under_a_deadline_matches_crate() {
    // Issue #614: with a deadline the passes check the clock
    // before each row. A deadline still ahead changes no
    // byte: the same sweep as above, through the checks.
    let ahead = Some(std::time::Instant::now() + std::time::Duration::from_secs(3600));
    for make in [photo as fn(u32, u32) -> DynamicImage, checker] {
        for (w, h) in [(333u32, 257u32), (64, 48), (7, 5), (3, 2), (1, 1), (31, 17)] {
            let base = make(w, h);
            for v in variants(&base) {
                for sigma in [0.3f32, 0.5, 1.0, 2.0, 4.0, 8.0, 8.4, 8.5, 12.0, 16.0, 40.0] {
                    assert_eq!(
                        bytes(&blur_fast_within(&v, sigma, ahead).unwrap()),
                        bytes(&v.blur(sigma)),
                        "blur sigma {sigma} at {w}x{h}"
                    );
                }
            }
        }
    }
    let big = photo(1200, 900);
    for sigma in [2.0f32, 8.5, 16.0, 0.0] {
        assert_eq!(
            bytes(&blur_fast_within(&big, sigma, ahead).unwrap()),
            bytes(&big.blur(sigma)),
            "big blur {sigma}"
        );
    }
    // A 16-bit image, through the same checks (the full
    // sweep of the 16-bit and float types is below).
    let deep = DynamicImage::ImageRgb16(photo(64, 48).to_rgb16());
    assert_eq!(
        bytes(&blur_fast_within(&deep, 4.0, ahead).unwrap()),
        bytes(&deep.blur(4.0))
    );
}

/// The 16-bit and float types the crate blurs, made from
/// `img`. The 16-bit samples get low bytes of their own and
/// the alpha varies, so the rounding sees every bit. The
/// float samples run from -0.25 to 1.25, as an HDR source's
/// can.
fn deep_variants(img: &DynamicImage) -> Vec<DynamicImage> {
    let rgba = img.to_rgba8();
    let (w, h) = (img.width(), img.height());
    let mut state: u64 = 0x0bad_5eed_1234_5678;
    let mut low = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 40) & 0xff) as u16
    };
    let mut deep: image::ImageBuffer<image::Rgba<u16>, Vec<u16>> = image::ImageBuffer::new(w, h);
    for (o, p) in deep.pixels_mut().zip(rgba.pixels()) {
        for c in 0..3 {
            o[c] = ((p[c] as u16) << 8) | low();
        }
        o[3] = (((p[0] ^ p[2]) as u16) << 8) | low();
    }
    let float = |v: u16| v as f32 / 65535.0 * 1.5 - 0.25;
    let at = |x: u32, y: u32| deep.get_pixel(x, y).0;
    vec![
        DynamicImage::ImageLuma16(image::ImageBuffer::from_fn(w, h, |x, y| {
            image::Luma([at(x, y)[1]])
        })),
        DynamicImage::ImageLumaA16(image::ImageBuffer::from_fn(w, h, |x, y| {
            let p = at(x, y);
            image::LumaA([p[1], p[3]])
        })),
        DynamicImage::ImageRgb16(image::ImageBuffer::from_fn(w, h, |x, y| {
            let p = at(x, y);
            image::Rgb([p[0], p[1], p[2]])
        })),
        DynamicImage::ImageRgb32F(image::ImageBuffer::from_fn(w, h, |x, y| {
            let p = at(x, y);
            image::Rgb([float(p[0]), float(p[1]), float(p[2])])
        })),
        DynamicImage::ImageRgba32F(image::ImageBuffer::from_fn(w, h, |x, y| {
            image::Rgba(at(x, y).map(float))
        })),
        DynamicImage::ImageRgba16(deep.clone()),
    ]
}

/// Every byte of an image of any type, its type, size and
/// color space.
fn raw(
    img: &DynamicImage,
) -> (
    std::mem::Discriminant<DynamicImage>,
    (u32, u32),
    image::metadata::Cicp,
    Vec<u8>,
) {
    (
        std::mem::discriminant(img),
        (img.width(), img.height()),
        img.color_space(),
        img.as_bytes().to_vec(),
    )
}

#[test]
fn w3_deep_blur_matches_crate() {
    // Issue #614: a 16-bit or float image blurred in one
    // call to the image crate, which no deadline could stop.
    // It now takes the banded passes, with the crate's Q0.15
    // integer arithmetic for 16-bit samples and its float
    // arithmetic for float ones. The same sweep as the 8-bit
    // proofs: the photo with no deadline, the checkerboard
    // and the large image with one far ahead.
    let ahead = Some(std::time::Instant::now() + std::time::Duration::from_secs(3600));
    for (make, deadline) in [
        (photo as fn(u32, u32) -> DynamicImage, None),
        (checker, ahead),
    ] {
        for (w, h) in [(333u32, 257u32), (64, 48), (7, 5), (3, 2), (1, 1), (31, 17)] {
            let base = make(w, h);
            for v in deep_variants(&base) {
                for sigma in [0.3f32, 0.5, 1.0, 2.0, 4.0, 8.0, 8.4, 8.5, 12.0, 16.0, 40.0] {
                    assert_eq!(
                        raw(&blur_fast_within(&v, sigma, deadline).unwrap()),
                        raw(&v.blur(sigma)),
                        "{:?} blur sigma {sigma} at {w}x{h}, deadline {deadline:?}",
                        v.color()
                    );
                }
            }
        }
    }
    // The large image in two of the types, one of each kind:
    // the crate blurs these on one core, slowly in a debug
    // build.
    let big = photo(1200, 900);
    for v in deep_variants(&big).into_iter().filter(|v| {
        matches!(
            v,
            DynamicImage::ImageRgba16(_) | DynamicImage::ImageRgb32F(_)
        )
    }) {
        for sigma in [2.0f32, 8.5, 16.0, 0.0] {
            assert_eq!(
                raw(&blur_fast_within(&v, sigma, ahead).unwrap()),
                raw(&v.blur(sigma)),
                "{:?} big blur {sigma}",
                v.color()
            );
        }
    }
}

#[test]
fn w3_gray_blur_matches_crate() {
    // Issue #614: the studio shadow blurs the subject's alpha
    // with `imageops::blur`, which runs a gray image through
    // the crate's float blur and rounds back. Its sigma grows
    // with the canvas, so it now runs through the banded
    // passes as `blur_gray_within`, the same bytes.
    let ahead = Some(std::time::Instant::now() + std::time::Duration::from_secs(3600));
    for make in [photo as fn(u32, u32) -> DynamicImage, checker] {
        for (w, h) in [(333u32, 257u32), (64, 48), (7, 5), (3, 2), (1, 1), (31, 17)] {
            let gray = make(w, h).to_luma8();
            for sigma in [
                0.0f32, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0, 8.4, 8.5, 12.0, 16.0, 40.0,
            ] {
                let theirs = image::imageops::blur(&gray, sigma);
                assert_eq!(
                    blur_gray_within(&gray, sigma, None).unwrap().as_raw(),
                    theirs.as_raw(),
                    "gray blur sigma {sigma} at {w}x{h}"
                );
                assert_eq!(
                    blur_gray_within(&gray, sigma, ahead).unwrap().as_raw(),
                    theirs.as_raw(),
                    "gray blur sigma {sigma} at {w}x{h} under a deadline"
                );
            }
        }
    }
    // A shadow's own input: a soft-edged disc of alpha on a
    // 600x400 canvas, at the strong and soft shadow sigmas
    // for that canvas (0.6 and 2 percent of its short side)
    // and past the ring's cutoff.
    let disc = image::GrayImage::from_fn(600, 400, |x, y| {
        let d = ((x as f32 - 300.0).hypot(y as f32 - 200.0) - 150.0).clamp(0.0, 10.0);
        image::Luma([(255.0 - d * 25.5) as u8])
    });
    for sigma in [2.4f32, 8.0, 12.0, 40.0] {
        assert_eq!(
            blur_gray_within(&disc, sigma, ahead).unwrap().as_raw(),
            image::imageops::blur(&disc, sigma).as_raw(),
            "disc blur sigma {sigma}"
        );
    }
}

#[test]
fn w3_perms_match_crate() {
    // 1601x901 RGB is above the fast-path size gate, so
    // these exercise the tiled/banded code, not the
    // fallback; the small size exercises the fallback.
    for (w, h) in [(1601u32, 901u32), (333, 257), (64, 48)] {
        let base = photo(w, h);
        for v in variants(&base) {
            assert_eq!(
                bytes(&rotate90_fast(&v)),
                bytes(&v.rotate90()),
                "r90 {w}x{h}"
            );
            assert_eq!(
                bytes(&rotate180_fast(&v)),
                bytes(&v.rotate180()),
                "r180 {w}x{h}"
            );
            assert_eq!(
                bytes(&rotate270_fast(&v)),
                bytes(&v.rotate270()),
                "r270 {w}x{h}"
            );
            assert_eq!(bytes(&fliph_fast(&v)), bytes(&v.fliph()), "fliph {w}x{h}");
            assert_eq!(bytes(&flipv_fast(&v)), bytes(&v.flipv()), "flipv {w}x{h}");
        }
    }
    // Single-channel frames need 2000x2000 to clear the
    // size gate; check the fast path engages correctly
    // there too.
    let big = photo(2000, 2000);
    for v in [
        DynamicImage::ImageLuma8(big.to_luma8()),
        DynamicImage::ImageLumaA8(big.to_luma_alpha8()),
    ] {
        assert_eq!(
            bytes(&rotate90_fast(&v)),
            bytes(&v.rotate90()),
            "r90 luma big"
        );
        assert_eq!(
            bytes(&rotate180_fast(&v)),
            bytes(&v.rotate180()),
            "r180 luma big"
        );
        assert_eq!(
            bytes(&rotate270_fast(&v)),
            bytes(&v.rotate270()),
            "r270 luma big"
        );
        assert_eq!(bytes(&fliph_fast(&v)), bytes(&v.fliph()), "fliph luma big");
        assert_eq!(bytes(&flipv_fast(&v)), bytes(&v.flipv()), "flipv luma big");
    }
}

/// The block rotation of issue #648 is the crate's `rotate90`
/// for every pixel layout the fast path takes, at any size:
/// under one block of output rows, on both sides of a block
/// boundary, sides that are no multiple of it, and one-pixel
/// lines. The kernel is called directly, so these small frames
/// run it, where `rotate90_fast` would hand them to the crate.
#[test]
fn rotate90_blocks_match_crate_at_any_size() {
    let block = ROT90_BLOCK_ROWS as u32;
    let mut sizes = vec![(1, 1), (1, 2), (2, 1), (1, 37), (37, 1), (3, 100), (100, 3)];
    for side in [
        block - 1,
        block,
        block + 1,
        2 * block - 1,
        2 * block,
        3 * block + 5,
    ] {
        sizes.extend([(side, side), (side, 7), (7, side), (side, 1), (1, side)]);
    }
    sizes.extend([(257, 129), (129, 257)]);
    let mut compared = 0;
    for (w, h) in sizes {
        let base = photo(w, h);
        let (uw, uh) = (w as usize, h as usize);
        for v in variants(&base) {
            let got = match &v {
                DynamicImage::ImageLuma8(i) => run_perm::<1>(i.as_raw(), uw, uh, rot90_px),
                DynamicImage::ImageLumaA8(i) => run_perm::<2>(i.as_raw(), uw, uh, rot90_px),
                DynamicImage::ImageRgb8(i) => run_perm::<3>(i.as_raw(), uw, uh, rot90_px),
                DynamicImage::ImageRgba8(i) => run_perm::<4>(i.as_raw(), uw, uh, rot90_px),
                other => panic!("an 8-bit layout was expected, not {:?}", other.color()),
            };
            let (_, size, want) = bytes(&v.rotate90());
            assert_eq!(size, (h, w), "{:?} at {w}x{h}", v.color());
            assert_eq!(got, want, "{:?} at {w}x{h}", v.color());
            compared += 1;
        }
    }
    assert_eq!(compared, 39 * 4);
}

/// `rotate90_fast` on both sides of its size gate, at sides
/// that are no multiple of the block, and on one-pixel lines
/// long enough to pass the gate. 16-bit frames keep the
/// crate's own loop, as they do for the other permutations.
/// (`w3_perms_match_crate` has the landscape frame in every
/// layout and the gray frames past the gate.)
#[test]
fn rotate90_fast_matches_crate_around_the_size_gate() {
    let same = |v: &DynamicImage| {
        assert_eq!(
            bytes(&rotate90_fast(v)),
            bytes(&v.rotate90()),
            "{:?} at {}x{}",
            v.color(),
            v.width(),
            v.height()
        );
    };
    // 1155x1155 RGB is 4,002,075 subpixels, just past the gate;
    // 1154x1155 is 3,998,610, just under it.
    let square = photo(1155, 1155);
    assert!(square.as_bytes().len() >= PERM_FAST_MIN_SUBPIXELS);
    same(&square);
    same(&DynamicImage::ImageRgba8(square.to_rgba8()));
    let under = photo(1154, 1155);
    assert!(under.as_bytes().len() < PERM_FAST_MIN_SUBPIXELS);
    same(&under);
    // A portrait frame, as a phone stores a landscape photo.
    same(&photo(901, 1601));
    // One-pixel lines of 1,000,003 RGBA pixels, down and across.
    for (w, h) in [(1u32, 1_000_003u32), (1_000_003, 1)] {
        let line = image::RgbaImage::from_fn(w, h, |x, y| {
            let v = ((x + y) % 251) as u8;
            image::Rgba([v, v ^ 0x55, 255 - v, v.wrapping_mul(7)])
        });
        same(&DynamicImage::ImageRgba8(line));
    }
    // 901x741 RGB16 is 4,005,846 bytes: past the gate, and the
    // crate's loop all the same.
    let deep = DynamicImage::ImageRgb16(photo(901, 741).to_rgb16());
    assert!(deep.as_bytes().len() >= PERM_FAST_MIN_SUBPIXELS);
    let turned = rotate90_fast(&deep);
    assert_eq!(turned.color(), image::ColorType::Rgb16);
    assert_eq!(turned.as_bytes(), deep.rotate90().as_bytes());
}

/// `filters:rotate(90)` takes the block quarter turn of issue
/// #648 (issue #661), as `rotate(180)` and `rotate(270)` take
/// theirs: the image crate's bytes on every 8-bit layout, past
/// the size gate and under it, at odd sizes, and however the
/// angle is spelled. A frame under the gate and a 16-bit frame
/// keep the crate's loop. The two write the same bytes, so the
/// count of block turns says which one ran.
#[test]
fn rotate_90_filter_takes_the_block_turn_and_matches_the_crate() {
    let turn = |img: &DynamicImage, degrees: f32| {
        let before = ROT90_BLOCK_TURNS.with(|turns| turns.get());
        let (out, _) = apply_filters(img.clone(), &[Filter::Rotate(degrees)]);
        (out, ROT90_BLOCK_TURNS.with(|turns| turns.get()) - before)
    };
    let (mut past_gate, mut under_gate) = (0, 0);
    // A landscape frame, its portrait twin, an odd square and
    // small odd frames. At 1601x901 the gray layouts are under
    // the gate and the color layouts past it.
    for (w, h) in [
        (1601u32, 901u32),
        (901, 1601),
        (1155, 1155),
        (333, 257),
        (7, 5),
        (1, 1),
    ] {
        let base = photo(w, h);
        for v in variants(&base) {
            let past = v.as_bytes().len() >= PERM_FAST_MIN_SUBPIXELS;
            for degrees in [90.0, -270.0, 450.0] {
                let (out, block_turns) = turn(&v, degrees);
                let what = format!("rotate({degrees}) of {:?} at {w}x{h}", v.color());
                assert_eq!(bytes(&out), bytes(&v.rotate90()), "{what}");
                assert_eq!(block_turns, usize::from(past), "{what}");
            }
            if past {
                past_gate += 1;
            } else {
                under_gate += 1;
            }
        }
    }
    assert_eq!((past_gate, under_gate), (6, 18));

    // 901x741 RGB16 is 4,005,846 bytes: past the gate, and the
    // crate's loop all the same.
    let deep = DynamicImage::ImageRgb16(photo(901, 741).to_rgb16());
    assert!(deep.as_bytes().len() >= PERM_FAST_MIN_SUBPIXELS);
    let (out, block_turns) = turn(&deep, 90.0);
    assert_eq!(out.color(), image::ColorType::Rgb16);
    assert_eq!(out.as_bytes(), deep.rotate90().as_bytes());
    assert_eq!(block_turns, 0);

    // The other turns do not come this way.
    let frame = photo(1601, 901);
    for degrees in [0.0, 180.0, 270.0, 33.0] {
        assert_eq!(turn(&frame, degrees).1, 0, "rotate({degrees})");
    }
}

#[test]
fn w3_huerotate_and_brighten_match_crate() {
    for (w, h) in [(1600u32, 900u32), (333, 257)] {
        let base = photo(w, h);
        for v in variants(&base) {
            for deg in [0i32, 30, 90, 137, 180, 359, -45, 720] {
                assert_eq!(
                    bytes(&huerotate_fast(v.clone(), deg)),
                    bytes(&v.huerotate(deg)),
                    "hue {deg} {w}x{h}"
                );
            }
            for val in [-200i32, -40, -1, 0, 1, 40, 200] {
                assert_eq!(
                    bytes(&brighten_fast(v.clone(), val)),
                    bytes(&v.brighten(val)),
                    "brighten {val} {w}x{h}"
                );
            }
        }
    }
}

/// The pre-wave-3 rotate_arbitrary body, kept as the
/// reference the banded version must match byte for byte.
fn seq_rotate_reference(img: DynamicImage, degrees: f32, fill: Option<[u8; 3]>) -> DynamicImage {
    let has_alpha = img.color().has_alpha();
    let background: [u8; 4] = if has_alpha {
        [0, 0, 0, 0]
    } else {
        let [r, g, b] = fill.unwrap_or([255, 255, 255]);
        [r, g, b, 255]
    };
    let src = img.into_rgba8();
    let (w, h) = (src.width() as f32, src.height() as f32);
    let theta = degrees.to_radians();
    let (sin, cos) = theta.sin_cos();
    let (new_w, new_h) = folded_canvas(src.width(), src.height(), degrees);
    let (scx, scy) = ((w - 1.0) / 2.0, (h - 1.0) / 2.0);
    let (dcx, dcy) = ((new_w as f32 - 1.0) / 2.0, (new_h as f32 - 1.0) / 2.0);
    let mut out = image::RgbaImage::from_pixel(new_w, new_h, image::Rgba(background));
    let sample = |x: i32, y: i32| -> [f32; 4] {
        if x < 0 || y < 0 || x >= src.width() as i32 || y >= src.height() as i32 {
            [
                background[0] as f32,
                background[1] as f32,
                background[2] as f32,
                background[3] as f32,
            ]
        } else {
            let p = src.get_pixel(x as u32, y as u32).0;
            [p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32]
        }
    };
    for y in 0..new_h {
        for x in 0..new_w {
            let dx = x as f32 - dcx;
            let dy = y as f32 - dcy;
            let sx = cos * dx + sin * dy + scx;
            let sy = -sin * dx + cos * dy + scy;
            let x0 = sx.floor();
            let y0 = sy.floor();
            let fx = sx - x0;
            let fy = sy - y0;
            let (x0, y0) = (x0 as i32, y0 as i32);
            let mut px = [0f32; 4];
            for (wx, xx) in [(1.0 - fx, x0), (fx, x0 + 1)] {
                for (wy, yy) in [(1.0 - fy, y0), (fy, y0 + 1)] {
                    let tap = sample(xx, yy);
                    let weight = wx * wy;
                    for (i, channel) in px.iter_mut().enumerate() {
                        *channel += tap[i] * weight;
                    }
                }
            }
            out.put_pixel(
                x,
                y,
                image::Rgba([
                    px[0].round() as u8,
                    px[1].round() as u8,
                    px[2].round() as u8,
                    px[3].round() as u8,
                ]),
            );
        }
    }
    if has_alpha {
        DynamicImage::ImageRgba8(out)
    } else {
        DynamicImage::ImageRgb8(DynamicImage::ImageRgba8(out).to_rgb8())
    }
}

#[test]
fn w3_rotate_arbitrary_banded_matches_sequential() {
    for (w, h) in [(800u32, 533u32), (333, 257)] {
        let rgb = photo(w, h);
        let rgba = DynamicImage::ImageRgba8(rgb.to_rgba8());
        for deg in [30.0f32, 45.0, 12.5, 200.0] {
            assert_eq!(
                bytes(&rotate_arbitrary(rgb.clone(), deg, None)),
                bytes(&seq_rotate_reference(rgb.clone(), deg, None)),
                "rotate {deg} rgb {w}x{h}"
            );
            assert_eq!(
                bytes(&rotate_arbitrary(rgba.clone(), deg, Some([9, 99, 199]))),
                bytes(&seq_rotate_reference(rgba.clone(), deg, Some([9, 99, 199]))),
                "rotate {deg} rgba {w}x{h}"
            );
        }
    }
}
