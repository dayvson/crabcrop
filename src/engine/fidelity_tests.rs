//! Issue #713, through the engine: an aspect crop and a
//! composite that lands nowhere keep the frame's layout and
//! depth; `quality(auto)` scores a JPEG with a fill against
//! the flattened image it encodes; `sharpen` and `huerotate`
//! URLs past their bounds still answer; a video's frames share
//! one smart crop window; class redaction holds its
//! segmentation decodes to a cap and checks the deadline
//! between them; and the frame copies the animated paths no
//! longer make change nothing they render.

use super::*;
use std::io::Cursor;

/// A `w` by `h` frame with detail everywhere in each layout a
/// decode can hand the chain: 8-bit RGB and RGBA, gray, 16-bit
/// RGB and RGBA, and float RGB and RGBA. The 16-bit and float
/// samples use their low bits, so a trip through 8 bits shows.
fn layouts(w: u32, h: u32) -> Vec<(&'static str, DynamicImage)> {
    let level = |x: u32, y: u32, c: u32| ((x * 31 + y * 17 + c * 59 + (x * y) % 11) % 256) as u8;
    let deep =
        |x: u32, y: u32, c: u32| u16::from(level(x, y, c)) << 8 | u16::from(level(y, x, c + 7));
    let alpha = |x: u32, y: u32| [255u8, 128, 0][((x / 4 + y / 3) % 3) as usize];
    let float = |v: u16| f32::from(v) / 65535.0;
    vec![
        (
            "rgb8",
            DynamicImage::ImageRgb8(image::RgbImage::from_fn(w, h, |x, y| {
                image::Rgb([level(x, y, 0), level(x, y, 1), level(x, y, 2)])
            })),
        ),
        (
            "rgba8",
            DynamicImage::ImageRgba8(image::RgbaImage::from_fn(w, h, |x, y| {
                image::Rgba([level(x, y, 0), level(x, y, 1), level(x, y, 2), alpha(x, y)])
            })),
        ),
        (
            "luma8",
            DynamicImage::ImageLuma8(image::GrayImage::from_fn(w, h, |x, y| {
                image::Luma([level(x, y, 0)])
            })),
        ),
        (
            "rgb16",
            DynamicImage::ImageRgb16(image::ImageBuffer::from_fn(w, h, |x, y| {
                image::Rgb([deep(x, y, 0), deep(x, y, 1), deep(x, y, 2)])
            })),
        ),
        (
            "rgba16",
            DynamicImage::ImageRgba16(image::ImageBuffer::from_fn(w, h, |x, y| {
                image::Rgba([
                    deep(x, y, 0),
                    deep(x, y, 1),
                    deep(x, y, 2),
                    u16::from(alpha(x, y)) * 257,
                ])
            })),
        ),
        (
            "rgb32f",
            DynamicImage::ImageRgb32F(image::ImageBuffer::from_fn(w, h, |x, y| {
                image::Rgb([
                    float(deep(x, y, 0)),
                    float(deep(x, y, 1)),
                    float(deep(x, y, 2)),
                ])
            })),
        ),
        (
            "rgba32f",
            DynamicImage::ImageRgba32F(image::ImageBuffer::from_fn(w, h, |x, y| {
                image::Rgba([
                    float(deep(x, y, 0)),
                    float(deep(x, y, 1)),
                    float(deep(x, y, 2)),
                    f32::from(alpha(x, y)) / 255.0,
                ])
            })),
        ),
    ]
}

fn ops_of(remainder: &str) -> Vec<Operation> {
    let path = if remainder.is_empty() {
        "example.com/a.png".to_string()
    } else {
        format!("{remainder}/example.com/a.png")
    };
    crate::url::parse_operations(&path)
        .unwrap_or_else(|e| panic!("{remainder}: {e:?}"))
        .0
}

/// One still through the chain, as a render runs it.
fn chain(img: &DynamicImage, remainder: &str) -> DynamicImage {
    let ops = ops_of(remainder);
    let directives = crate::url::RequestDirectives::analyze(&ops);
    let ctx = ProcessContext::default();
    apply_operations(img.clone(), &ops, None, 1.0, &ctx, &directives, None)
        .unwrap_or_else(|e| panic!("{remainder}: {e:?}"))
        .img
}

fn encoded(img: &DynamicImage, format: image::ImageFormat) -> Vec<u8> {
    let mut out = Vec::new();
    img.write_to(&mut Cursor::new(&mut out), format).unwrap();
    out
}

fn render(bytes: &[u8], remainder: &str) -> ProcessedImage {
    Engine
        .process(bytes, &ops_of(remainder))
        .unwrap_or_else(|e| panic!("{remainder}: {e:?}"))
}

#[test]
fn aspect_crops_keep_the_layout_and_depth_of_the_frame() {
    // Issue #713: an aspect crop cut every layout to 8-bit
    // RGBA, so an opaque frame gained an alpha channel and a
    // 16-bit or float frame came out with 8 bits, where a
    // plain resize of the same frame kept both. Every
    // alignment, a window that needs a resample down and up,
    // one that is cut and kept at its own size, and an aspect
    // that already matches.
    let mut cases = 0;
    for (name, img) in layouts(96, 60) {
        for remainder in [
            "crop:40x40",
            "crop:40x40:smart",
            "crop:30x50:top-left",
            "crop:120x40:bottom",
            "crop:96x30",
            "crop:48x30",
        ] {
            let out = chain(&img, remainder);
            let at = format!("{remainder} of {name}");
            assert_eq!(out.color(), img.color(), "{at}");
            cases += 1;
        }
        // A window the crop cuts and does not resample is the
        // frame's own pixels: the middle 96x30 of the frame.
        let cut = chain(&img, "crop:96x30");
        assert_eq!(
            cut.as_bytes(),
            img.crop_imm(0, 15, 96, 30).as_bytes(),
            "{name}"
        );
        // A resampled window is the cut, resized in the
        // frame's layout.
        let resized = chain(&img, "crop:40x40");
        let expected = resize_lanczos3(&img.crop_imm(18, 0, 60, 60), 40, 40);
        assert_eq!(resized.as_bytes(), expected.as_bytes(), "{name}");
    }
    assert_eq!(cases, 7 * 6);
}

#[test]
fn an_aspect_crop_keeps_its_source_s_color_type_in_the_file() {
    // The run that found issue #713: `crop:400x200` of an
    // opaque JPEG came out an 8-bit RGBA PNG where `400x250`
    // came out RGB, and `crop:32x32` of a 16-bit PNG came out
    // 8-bit RGBA where a resize kept 16 bits.
    let photo = DynamicImage::ImageRgb8(image::RgbImage::from_fn(800, 500, |x, y| {
        image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
    }));
    let jpeg = encoded(&photo, image::ImageFormat::Jpeg);
    for remainder in ["crop:400x200", "400x250", "crop:400x200:smart"] {
        let out = render(&jpeg, &format!("{remainder}/filters:format(png)"));
        let png = image::load_from_memory(&out.bytes).unwrap();
        assert_eq!(png.color(), image::ColorType::Rgb8, "{remainder}");
    }
    let deep = DynamicImage::ImageRgb16(image::ImageBuffer::from_fn(64, 48, |x, y| {
        image::Rgb([(x * 1000) as u16, (y * 1300) as u16, 40_001])
    }));
    let png16 = encoded(&deep, image::ImageFormat::Png);
    for remainder in ["crop:32x32", "32x24", "crop:20x40:right"] {
        let out = render(&png16, &format!("{remainder}/filters:format(png)"));
        let png = image::load_from_memory(&out.bytes).unwrap();
        assert_eq!(png.color(), image::ColorType::Rgb16, "{remainder}");
    }
    // An image with alpha keeps it, and an RGBA8 crop's file
    // is the file it always was.
    let translucent = DynamicImage::ImageRgba8(image::RgbaImage::from_fn(64, 48, |x, y| {
        image::Rgba([(x * 4) as u8, (y * 5) as u8, 90, (x * y % 256) as u8])
    }));
    let png = encoded(&translucent, image::ImageFormat::Png);
    let out = render(&png, "crop:32x32/filters:format(png)");
    assert_eq!(
        image::load_from_memory(&out.bytes).unwrap().color(),
        image::ColorType::Rgba8
    );
}

#[test]
fn composites_that_land_nowhere_leave_the_base_as_it_was() {
    // Issue #713: a watermark or an overlay placed off the
    // canvas drew nothing, yet handed the base back as 8-bit
    // RGBA. It now hands back the base itself, whatever its
    // layout. One that lands still composites in 8-bit RGBA,
    // as it always has.
    let mut map = crate::filters::WatermarkImages::new();
    map.insert(
        "wm".to_string(),
        crate::filters::WatermarkImage::new(DynamicImage::ImageRgba8(
            image::RgbaImage::from_pixel(6, 4, image::Rgba([200, 10, 10, 255])),
        )),
    );
    let ctx = ProcessContext {
        watermarks: Some(&map),
        ..Default::default()
    };
    let off = [(40, 0), (0, 30), (-40, 3), (5, -30), (i32::MAX, i32::MIN)];
    for (name, base) in layouts(24, 16) {
        for (x, y) in off {
            let filters = [
                Filter::Watermark {
                    url: "wm".to_string(),
                    x,
                    y,
                    alpha: 80,
                },
                Filter::Overlay {
                    url: "wm".to_string(),
                    x,
                    y,
                    alpha: 80,
                    blend: crate::url::BlendMode::Multiply,
                    width: None,
                },
                Filter::Overlay {
                    url: "wm".to_string(),
                    x,
                    y,
                    alpha: 80,
                    blend: crate::url::BlendMode::Screen,
                    width: Some(50),
                },
            ];
            let ops = [Operation::Filters(filters.to_vec())];
            let directives = crate::url::RequestDirectives::analyze(&ops);
            let mut state = ChainState::new(&ops, None, 1.0, &ctx, &directives, None);
            for filter in &filters {
                let out = state.apply_filter(base.clone(), filter).unwrap();
                assert_eq!(out, base, "{filter:?} on {name}");
            }
        }
        let landed = overlay_watermark(&ctx, &base, &map["wm"], 2, 2, 80).unwrap();
        assert!(matches!(landed, DynamicImage::ImageRgba8(_)), "{name}");
    }
}

#[test]
fn quality_auto_scores_a_filled_jpeg_against_what_it_encodes() {
    // Issue #713: a JPEG with a fill encodes the image
    // flattened over the fill, but `quality(auto)` scored its
    // candidates against the image with alpha dropped, which
    // shows the colors stored under transparent pixels: here
    // noise, which the file never shows. The score it reports
    // is now the one of the bytes it serves against the
    // flattened image.
    let src = DynamicImage::ImageRgba8(image::RgbaImage::from_fn(96, 64, |x, y| {
        if x < 48 {
            let n = (x * 197 + y * 89 + x * y * 7) % 256;
            image::Rgba([n as u8, (255 - n) as u8, (n * 3 % 256) as u8, 0])
        } else {
            image::Rgba([(x * 2) as u8, (y * 3) as u8, 120, 255])
        }
    }));
    let png = encoded(&src, image::ImageFormat::Png);
    let out = render(&png, "filters:format(jpeg):fill(ffffff):quality(auto)");
    let decoded = decode_image(&out.bytes).unwrap();
    let flattened = DynamicImage::ImageRgb8(flatten(&src, [255, 255, 255]));
    let against_encoded = crate::quality::ssimulacra2(&flattened, &decoded).unwrap();
    let against_unflattened =
        crate::quality::ssimulacra2(&DynamicImage::ImageRgb8(src.to_rgb8()), &decoded).unwrap();
    assert_eq!(out.ssimulacra2, Some(against_encoded));
    assert!(
        against_unflattened < against_encoded - 10.0,
        "{against_unflattened} against {against_encoded}"
    );
    // A fill with a fixed quality writes the bytes it wrote:
    // the image is flattened once now instead of at every
    // encode, to the same pixels.
    let fixed = render(&png, "filters:format(jpeg):fill(ffffff):quality(70)");
    let flattened_jpeg = render(
        &encoded(&flattened, image::ImageFormat::Png),
        "filters:format(jpeg):quality(70)",
    );
    assert_eq!(fixed.bytes, flattened_jpeg.bytes);
}

#[test]
fn sharpen_and_huerotate_past_their_bounds_still_answer() {
    // Issue #713: the bounds hold the working values, not the
    // URL: every amount and every degree the grammar takes
    // renders, as the value it is held or reduced to.
    let img = DynamicImage::ImageRgb8(image::RgbImage::from_fn(32, 24, |x, y| {
        image::Rgb([(x * 8) as u8, (y * 10) as u8, ((x * y) % 256) as u8])
    }));
    let png = encoded(&img, image::ImageFormat::Png);
    let bytes = |remainder: &str| render(&png, &format!("filters:{remainder}:format(png)")).bytes;
    let bound = bytes("sharpen(255)");
    for amount in [
        "300",
        "100000000",
        "300000000000000000000000000000000000000",
    ] {
        assert_eq!(
            bytes(&format!("sharpen({amount})")),
            bound,
            "sharpen({amount})"
        );
    }
    assert_eq!(bytes("huerotate(2147483647)"), bytes("huerotate(127)"));
    assert_eq!(bytes("huerotate(-2147483648)"), bytes("huerotate(-128)"));
    assert_eq!(bytes("huerotate(450)"), bytes("huerotate(90)"));
}

/// An animated GIF of three 60x20 frames, flat gray but for
/// an 8x8 checkerboard that moves from the left of the frame
/// to its middle and its right: each frame's own smart crop to
/// 20x20 would follow it.
fn moving_detail_gif() -> Vec<u8> {
    let delay = image::Delay::from_numer_denom_ms(100, 1);
    let frames: Vec<image::Frame> = (0..3)
        .map(|k| {
            let left = 6 + 20 * k;
            let buffer = image::RgbaImage::from_fn(60, 20, |x, y| {
                let detail = (left..left + 8).contains(&x) && (6..14).contains(&y);
                let v = if !detail {
                    128
                } else if (x + y) % 2 == 0 {
                    0
                } else {
                    255
                };
                image::Rgba([v, v, v, 255])
            });
            image::Frame::from_parts(buffer, 0, 0, delay)
        })
        .collect();
    let mut out = Vec::new();
    image::codecs::gif::GifEncoder::new(&mut out)
        .encode_frames(frames)
        .unwrap();
    out
}

#[test]
fn a_video_s_frames_share_one_smart_crop_window() {
    // Issue #713: the GIF and WebP outputs of an animation
    // decide a smart crop on the first frame and replay its
    // window on the rest (issue #376), but the video path let
    // each frame decide for itself, so the crop followed the
    // detail across the frame and jittered. The frames the
    // encoder receives are cut from one window now: the
    // first, which holds the detail, and two flat ones.
    let dir = crate::test_support::scratch_dir("fidelity-video");
    let frames_path = dir.join("frames.rgba");
    let script = dir.join("encoder.sh");
    crate::test_support::write_script(
        &script,
        &format!(
            "#!/bin/sh\ncat > '{}'\nprintf 'ENGINE-VIDEO'\n",
            frames_path.display()
        ),
    );
    let encoder =
        crate::video::VideoEncoder::new(script, std::time::Duration::from_secs(30), 1 << 20)
            .unwrap();
    let ops = ops_of("crop:20x20:smart/filters:format(mp4)");
    let out = Engine
        .process_preferred(
            &moving_detail_gif(),
            &ops,
            ProcessContext {
                video_encoder: Some(&encoder),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(out.bytes, b"ENGINE-VIDEO");
    let raw = std::fs::read(&frames_path).unwrap();
    let frame_bytes = 20 * 20 * 4;
    assert_eq!(raw.len() % frame_bytes, 0);
    let samples: Vec<&[u8]> = raw.chunks_exact(frame_bytes).collect();
    assert_eq!(samples.len(), 3, "one sample a frame at 10 fps");
    let spread = |frame: &[u8]| {
        let (min, max) = frame
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| p[0])
            .fold((255, 0), |(lo, hi), v| (v.min(lo), v.max(hi)));
        max - min
    };
    assert!(spread(samples[0]) > 200, "the first frame holds the detail");
    for (k, frame) in samples.iter().enumerate().skip(1) {
        assert!(
            spread(frame) <= 2,
            "frame {k} was cut from another window: spread {}",
            spread(frame)
        );
    }
}

/// Detections of "person" in a grid: `n` boxes of 10x10, ten to
/// a row, 20 px apart across and 14 down.
#[derive(Debug)]
struct Crowd(usize);

fn crowd_box(i: usize) -> (u32, u32) {
    (20 * (i % 10) as u32, 14 * (i / 10) as u32)
}

impl crate::ai::DetectionProvider for Crowd {
    fn knows_class(&self, name: &str) -> bool {
        name.eq_ignore_ascii_case("person")
    }
    fn describe(&self) -> Vec<String> {
        vec!["person".to_string()]
    }
    fn detect(&self, _img: &DynamicImage) -> Result<Vec<crate::ai::Detection>, crate::ai::AiError> {
        Ok((0..self.0)
            .map(|i| {
                let (x, y) = crowd_box(i);
                crate::ai::Detection {
                    label: "person".to_string(),
                    score: 0.9 - i as f32 * 0.001,
                    x,
                    y,
                    w: 10,
                    h: 10,
                }
            })
            .collect())
    }
}

/// A segment provider whose silhouette is the left half of
/// every box, counting its decodes, each taking `pause`.
#[derive(Debug, Default)]
struct LeftHalves {
    decodes: std::sync::atomic::AtomicUsize,
    pause: std::time::Duration,
}

impl crate::ai::SegmentProvider for LeftHalves {
    fn encode(
        &self,
        _img: &DynamicImage,
    ) -> Result<crate::ai::SegmentEmbedding, crate::ai::AiError> {
        Ok(crate::ai::SegmentEmbedding { data: vec![0.0] })
    }

    fn decode(
        &self,
        _embedding: &crate::ai::SegmentEmbedding,
        _img_w: u32,
        _img_h: u32,
        (_, _, w, h): (u32, u32, u32, u32),
    ) -> Result<image::GrayImage, crate::ai::AiError> {
        self.decodes
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        std::thread::sleep(self.pause);
        Ok(image::GrayImage::from_fn(w, h, |x, _| {
            image::Luma([if x < w / 2 { 255 } else { 0 }])
        }))
    }
}

fn checkerboard(w: u32, h: u32) -> DynamicImage {
    DynamicImage::ImageRgb8(image::RgbImage::from_fn(w, h, |x, y| {
        let v = if (x + y) % 2 == 0 { 0 } else { 255 };
        image::Rgb([v, v, v])
    }))
}

#[test]
fn class_redaction_segments_at_most_its_cap_and_boxes_the_rest() {
    // Issue #713: a class filter ran one segmentation decode
    // for every detection, however many: a crowd held an AI
    // permit for a decode per person. A request runs at most
    // MAX_SEGMENT_DECODES, to the detections in the
    // detector's order, and redacts the rest by their boxes,
    // which hold the whole object.
    assert_eq!(MAX_SEGMENT_DECODES, 64);
    let img = checkerboard(200, 100);
    let crowd = Crowd(70);
    let segment = LeftHalves::default();
    let ctx = ProcessContext {
        detect_provider: Some(&crowd),
        segment_provider: Some(&segment),
        ..Default::default()
    };
    let ops = ops_of("filters:blurclass(person)");
    let directives = crate::url::RequestDirectives::analyze(&ops);
    let outcome = apply_operations(img.clone(), &ops, None, 1.0, &ctx, &directives, None).unwrap();
    assert_eq!(
        segment.decodes.load(std::sync::atomic::Ordering::SeqCst),
        MAX_SEGMENT_DECODES
    );
    assert_eq!(
        outcome.segments.expect("a segment record").objects.len(),
        MAX_SEGMENT_DECODES
    );
    let (out, src) = (outcome.img.to_rgb8(), img.to_rgb8());
    for i in 0..70 {
        let (x, y) = crowd_box(i);
        // The left half of every box is blurred; the right
        // half only where the box stood in for a silhouette.
        assert_ne!(
            out.get_pixel(x + 2, y + 4),
            src.get_pixel(x + 2, y + 4),
            "box {i} left"
        );
        let right = (x + 8, y + 4);
        let kept = out.get_pixel(right.0, right.1) == src.get_pixel(right.0, right.1);
        assert_eq!(kept, i < MAX_SEGMENT_DECODES, "box {i} right");
    }
    // The cap is the request's: a second class filter on the
    // same request decodes nothing more, and boxes all.
    let segment = LeftHalves::default();
    let ctx = ProcessContext {
        segment_provider: Some(&segment),
        ..ctx
    };
    let ops = ops_of("filters:blurclass(person):pixelateclass(person)");
    let directives = crate::url::RequestDirectives::analyze(&ops);
    apply_operations(img, &ops, None, 1.0, &ctx, &directives, None).unwrap();
    assert_eq!(
        segment.decodes.load(std::sync::atomic::Ordering::SeqCst),
        MAX_SEGMENT_DECODES
    );
}

#[test]
fn class_redaction_checks_the_deadline_between_decodes() {
    // Issue #713: nothing checked the deadline between one
    // box's decode and the next, so a crowd decoded on past
    // the request's limit. Each decode here takes 50 ms
    // against a limit of 300 ms: the request stops at its
    // deadline after a few, not after all 40.
    let crowd = Crowd(40);
    let segment = LeftHalves {
        pause: std::time::Duration::from_millis(50),
        ..Default::default()
    };
    let ctx = ProcessContext {
        detect_provider: Some(&crowd),
        segment_provider: Some(&segment),
        deadline: Some(std::time::Instant::now() + std::time::Duration::from_millis(300)),
        ..Default::default()
    };
    let ops = ops_of("filters:pixelateclass(person)");
    let directives = crate::url::RequestDirectives::analyze(&ops);
    let err = apply_operations(
        checkerboard(200, 100),
        &ops,
        None,
        1.0,
        &ctx,
        &directives,
        None,
    )
    .unwrap_err();
    assert!(matches!(err, EngineError::Timeout), "{err:?}");
    let decodes = segment.decodes.load(std::sync::atomic::Ordering::SeqCst);
    assert!(decodes < 10, "{decodes} decodes");
}

#[test]
fn the_first_frame_is_lent_to_its_looks_and_put_back() {
    // Issue #713: the quality pick and the placeholder each
    // read the first rendered frame through a copy of it. The
    // frame's own buffer is lent now; they see the same
    // pixels, and the frame comes back as it was.
    let buffer = image::RgbaImage::from_fn(48, 32, |x, y| {
        image::Rgba([(x * 5) as u8, (y * 7) as u8, ((x * y) % 256) as u8, 200])
    });
    let delay = image::Delay::from_numer_denom_ms(70, 1);
    let mut frames = Frames {
        first: image::Frame::from_parts(buffer.clone(), 3, 4, delay),
        rest: vec![image::Frame::new(image::RgbaImage::new(48, 32))],
    };
    let copy = DynamicImage::ImageRgba8(buffer.clone());
    assert_eq!(frames.with_first(auto_quality), auto_quality(&copy));
    assert_eq!(
        frames.with_first(crate::placeholder::compute),
        crate::placeholder::compute(&copy)
    );
    assert_eq!(frames.first.buffer(), &buffer);
    assert_eq!(
        (
            frames.first.left(),
            frames.first.top(),
            frames.first.delay()
        ),
        (3, 4, delay)
    );
    assert_eq!(frames.len(), 2);
}

/// An animated GIF of three 40x30 frames on a white border,
/// each with its content in another part of the frame.
fn bordered_frames_gif() -> Vec<u8> {
    let delay = image::Delay::from_numer_denom_ms(100, 1);
    let frames: Vec<image::Frame> = [(4, 5), (15, 9), (9, 18)]
        .into_iter()
        .map(|(left, top)| {
            let buffer = image::RgbaImage::from_fn(40, 30, |x, y| {
                let inside = (left..left + 12).contains(&x) && (top..top + 8).contains(&y);
                if inside {
                    image::Rgba([(x * 6) as u8, 40, (y * 8) as u8, 255])
                } else {
                    image::Rgba([255, 255, 255, 255])
                }
            });
            image::Frame::from_parts(buffer, 0, 0, delay)
        })
        .collect();
    let mut out = Vec::new();
    image::codecs::gif::GifEncoder::new(&mut out)
        .encode_frames(frames)
        .unwrap();
    out
}

#[test]
fn a_trim_nothing_runs_before_surveys_the_frames_where_they_are() {
    // Issue #713: the trim survey copied every frame once per
    // trim to run the chain up to it. A trim that is the
    // chain's first step has nothing to run, and reads the
    // frames where they are. It settles the window the copy
    // settled: the same window as a chain whose first step
    // changes no pixel (a fill), and the same render.
    let gif = bordered_frames_gif();
    let decoded = decode_gif_frames(&gif, None).unwrap();
    let ctx = ProcessContext::default();
    let window_of = |remainder: &str, key: (usize, usize)| {
        let ops = ops_of(remainder);
        let directives = crate::url::RequestDirectives::analyze(&ops);
        let mut share = FrameShare::default();
        survey_trims(&decoded, &ops, None, 1.0, &ctx, &directives, &mut share).unwrap();
        share.trims.get(key, (40, 30)).expect("a settled window")
    };
    let first = window_of("filters:trim(10)", (0, 0));
    assert_eq!(first, window_of("filters:fill(000000):trim(10)", (0, 1)));
    assert_eq!(first, Some((4, 5, 23, 21)));
    assert_eq!(
        render(&gif, "filters:trim(10)").bytes,
        render(&gif, "filters:fill(000000):trim(10)").bytes
    );
}
