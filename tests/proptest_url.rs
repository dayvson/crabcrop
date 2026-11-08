//! Property tests for the URL grammar (issue #478): operation
//! chains generated per grammar family, each operation in a
//! spelling the parser accepts (the canonical one or an
//! equivalent: the `jpg` alias, leading zeros, hex case,
//! trailing fraction zeros, optional arguments, split filter
//! segments, interleaved `exp:` and `maxage:` metadata),
//! must
//!
//! - parse back to the operations that were generated;
//! - canonicalize (`canonical_form`) to text that re-parses
//!   (`parse_request`) to those same operations, exactly;
//! - re-canonicalize to the same text, which is also what
//!   `canonical_key_form` keys the original spelling under.
//!
//! The float arguments compare exactly too: every canonical
//! float is the shortest text that reads back as the same
//! value, so no rounding is involved anywhere. Generated
//! chains stay under the per-request operation cap
//! ([`OPERATION_CAP`]), so they stay valid requests when the
//! cap moves to parse time (issue #469).
//!
//! The cases per property default to [`DEFAULT_CASES`];
//! `PROPTEST_CASES` raises it (the nightly job does). A
//! failure is saved next to this file in
//! `proptest_url.proptest-regressions`, which is checked in so
//! the seed replays first on every later run.

use crabcrop::url::{
    self, Align, Anchor, BlendMode, CurveChannel, Dimension, Filter, FlipDirection, FormatOptions,
    FormatSpec, FrameSelection, ImageFormat, ImageSize, MaskShape, Operation, PadFill, Quality,
    QualityValue, Signature, StudioBg, StudioMode, StudioShadow, StudioSpec, Subsampling,
};
use proptest::prelude::*;
use proptest::sample::select;
use proptest::test_runner::FileFailurePersistence;

/// Cases per property when `PROPTEST_CASES` is unset.
const DEFAULT_CASES: u32 = 64;

/// The most operations one request may execute (each filter
/// counts once), the engine's `MAX_OPERATIONS`, which issue
/// #469 moves to `url::MAX_OPERATIONS` and checks at parse
/// time.
const OPERATION_CAP: usize = 100;

const MAX_DIMENSION: u32 = url::MAX_DIMENSION;

fn config() -> ProptestConfig {
    let mut config = ProptestConfig {
        failure_persistence: Some(Box::new(FileFailurePersistence::WithSource(
            "proptest-regressions",
        ))),
        ..ProptestConfig::default()
    };
    if std::env::var_os("PROPTEST_CASES").is_none() {
        config.cases = DEFAULT_CASES;
    }
    config
}

/// A generated value and one spelling of it.
type Spelled<T> = (T, String);

/// Pair each generated value with one of its spellings.
fn spelled<T, S, F>(values: S, spell: F) -> BoxedStrategy<Spelled<T>>
where
    T: Clone + std::fmt::Debug + 'static,
    S: Strategy<Value = T> + 'static,
    F: Fn(&T) -> Vec<String> + 'static,
{
    values
        .prop_flat_map(move |v| {
            let spellings = spell(&v);
            (Just(v), select(spellings))
        })
        .boxed()
}

// ---- Spellings of the argument kinds ----

/// An unsigned integer, bare or with a leading zero.
fn uint(n: impl Into<u64>) -> Vec<String> {
    let n = n.into();
    vec![n.to_string(), format!("0{n}")]
}

/// A signed integer: a leading zero only when non-negative.
fn int(n: impl Into<i64>) -> Vec<String> {
    let n = n.into();
    if n >= 0 {
        uint(n as u64)
    } else {
        vec![n.to_string()]
    }
}

/// A decimal: the shortest form, and the same value with a
/// trailing fraction zero (`1.5` and `1.50`, `2` and `2.0`).
fn decimal(text: String) -> Vec<String> {
    let longer = if text.contains('.') {
        format!("{text}0")
    } else {
        format!("{text}.0")
    };
    vec![text, longer]
}

fn decimal32(v: f32) -> Vec<String> {
    decimal(format!("{v}"))
}

fn hex(bytes: &[u8]) -> Vec<String> {
    let lower: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    vec![lower.clone(), lower.to_ascii_uppercase()]
}

/// Percent-encode everything outside the unreserved set (the
/// canonical encoding, upper or lower hex), or leave the
/// sub-delimiters, `:` and `@` raw as well: the argument
/// parsers split from the right or inside parentheses, so
/// those survive unencoded.
fn encoded(text: &str, keep: &str) -> Vec<String> {
    let encode = |upper: bool, keep: &str| {
        let mut out = String::new();
        for &b in text.as_bytes() {
            if b.is_ascii_alphanumeric() || b"-._~".contains(&b) || keep.as_bytes().contains(&b) {
                out.push(b as char);
            } else if upper {
                out.push_str(&format!("%{b:02X}"));
            } else {
                out.push_str(&format!("%{b:02x}"));
            }
        }
        out
    };
    let mut all = vec![encode(true, ""), encode(false, "")];
    if !keep.is_empty() {
        all.push(encode(true, keep));
    }
    all
}

/// Every combination of one spelling per argument, joined
/// with commas.
fn args(parts: &[Vec<String>]) -> Vec<String> {
    let mut out = vec![String::new()];
    for (i, choices) in parts.iter().enumerate() {
        out = out
            .iter()
            .flat_map(|prefix| {
                choices.iter().map(move |c| {
                    if i == 0 {
                        c.clone()
                    } else {
                        format!("{prefix},{c}")
                    }
                })
            })
            .collect();
    }
    out
}

/// `name(args...)` for every spelling combination.
fn call(name: &str, parts: &[Vec<String>]) -> Vec<String> {
    args(parts)
        .into_iter()
        .map(|a| format!("{name}({a})"))
        .collect()
}

fn format_name(format: ImageFormat) -> &'static str {
    match format {
        ImageFormat::Jpeg => "jpeg",
        ImageFormat::Png => "png",
        ImageFormat::WebP => "webp",
        ImageFormat::Jxl => "jxl",
        ImageFormat::Avif => "avif",
        ImageFormat::Gif => "gif",
        ImageFormat::Mp4 => "mp4",
        ImageFormat::WebM => "webm",
    }
}

/// A format name, with the documented `jpg` alias.
fn format_spellings(format: ImageFormat) -> Vec<String> {
    match format {
        ImageFormat::Jpeg => vec!["jpeg".into(), "jpg".into()],
        other => vec![format_name(other).into()],
    }
}

// ---- Value strategies ----

fn rgb() -> impl Strategy<Value = [u8; 3]> {
    any::<[u8; 3]>()
}

/// A detection class name: the grammar's charset (letters
/// and digits in any script, space, `-`, `_`, `.`, `/`),
/// trimmed, at most 64 bytes.
fn class_name() -> impl Strategy<Value = String> {
    "[A-Za-z0-9éß日]([A-Za-z0-9éß日 ._/-]{0,18}[A-Za-z0-9éß日])?"
}

/// Free text for labels and overlay URLs: any characters.
fn free_text() -> impl Strategy<Value = String> {
    prop::collection::vec(any::<char>(), 1..=16).prop_map(String::from_iter)
}

fn image_format() -> impl Strategy<Value = ImageFormat> {
    select(vec![
        ImageFormat::Jpeg,
        ImageFormat::Png,
        ImageFormat::WebP,
        ImageFormat::Jxl,
        ImageFormat::Avif,
        ImageFormat::Gif,
        ImageFormat::Mp4,
        ImageFormat::WebM,
    ])
}

fn quality_value() -> impl Strategy<Value = QualityValue> {
    prop_oneof![
        (1u8..=100).prop_map(QualityValue::Value),
        Just(QualityValue::Auto)
    ]
}

fn quality_value_spellings(value: QualityValue) -> Vec<String> {
    match value {
        QualityValue::Auto => vec!["auto".into()],
        QualityValue::Value(v) => uint(v),
    }
}

// ---- Filters, by family ----

/// A filter that takes no arguments, bare or with empty
/// parentheses.
fn bare(filter: Filter, name: &'static str) -> BoxedStrategy<Spelled<Filter>> {
    spelled(Just(filter), move |_| {
        vec![name.to_string(), format!("{name}()")]
    })
}

/// Tone and adjustment filters.
fn tone_filter() -> BoxedStrategy<Spelled<Filter>> {
    let percent = || prop_oneof![-100.0f32..=100.0, Just(-100.0), Just(100.0), Just(-0.0)];
    prop_oneof![
        bare(Filter::Grayscale, "grayscale"),
        bare(Filter::Invert, "invert"),
        bare(Filter::Sepia, "sepia"),
        bare(Filter::Equalize, "equalize"),
        bare(Filter::Enhance, "enhance"),
        spelled(
            prop_oneof![
                f32::MIN_POSITIVE..=url::MAX_BLUR_SIGMA,
                Just(url::MAX_BLUR_SIGMA)
            ],
            |v| call("blur", &[decimal32(*v)])
        )
        .prop_map(|(v, s)| (Filter::Blur(v), s)),
        spelled(-255i32..=255, |v| call("brighten", &[int(*v)]))
            .prop_map(|(v, s)| (Filter::Brighten(v), s)),
        spelled(percent(), |v| call("contrast", &[decimal32(*v)]))
            .prop_map(|(v, s)| (Filter::Contrast(v), s)),
        spelled(
            prop_oneof![
                (0.0f32..=url::MAX_GAMMA).prop_filter("positive", |v| *v > 0.0),
                Just(f32::from_bits(1)),
                Just(url::MAX_GAMMA)
            ],
            |v| call("gamma", &[decimal32(*v)])
        )
        .prop_map(|(v, s)| (Filter::Gamma(v), s)),
        spelled(-url::MAX_EXPOSURE..=url::MAX_EXPOSURE, |v| call(
            "exposure",
            &[decimal32(*v)]
        ))
        .prop_map(|(v, s)| (Filter::Exposure(v), s)),
        spelled((percent(), percent()), |(h, s)| call(
            "highlights_shadows",
            &[decimal32(*h), decimal32(*s)]
        ))
        .prop_map(|((h, sh), s)| (Filter::HighlightsShadows(h, sh), s)),
        spelled(percent(), |v| call("vibrance", &[decimal32(*v)]))
            .prop_map(|(v, s)| (Filter::Vibrance(v), s)),
        spelled(any::<i32>(), |v| call("huerotate", &[int(*v)]))
            .prop_map(|(v, s)| (Filter::HueRotate(v), s)),
        spelled(
            prop_oneof![1e-3f32..=100.0, Just(f32::MAX), Just(f32::from_bits(1))],
            |v| call("sharpen", &[decimal32(*v)])
        )
        .prop_map(|(v, s)| (Filter::Sharpen(v), s)),
        spelled(prop_oneof![0.0f32..=255.0, Just(255.0)], |v| call(
            "saturation",
            &[decimal32(*v)]
        ))
        .prop_map(|(v, s)| (Filter::Saturation(v), s)),
        spelled(
            (-255i32..=255, -255i32..=255, -255i32..=255),
            |(r, g, b)| call("rgb", &[int(*r), int(*g), int(*b)])
        )
        .prop_map(|((r, g, b), s)| (Filter::Rgb(r, g, b), s)),
        spelled(0u8..=100, |v| call("noise", &[uint(*v)])).prop_map(|(v, s)| (Filter::Noise(v), s)),
    ]
    .boxed()
}

/// Color, curve, and kernel filters.
fn color_filter() -> BoxedStrategy<Spelled<Filter>> {
    let channel = select(vec![
        (CurveChannel::Rgb, "rgb"),
        (CurveChannel::R, "r"),
        (CurveChannel::G, "g"),
        (CurveChannel::B, "b"),
    ]);
    let points = (
        prop::collection::btree_set(any::<u8>(), 2..=16),
        prop::collection::vec(any::<u8>(), 16),
    )
        .prop_map(|(xs, ys)| xs.into_iter().zip(ys).collect::<Vec<(u8, u8)>>());
    let kernel_value = prop_oneof![
        (-1024i32..=1024).prop_map(f64::from),
        (-2048i32..=2048).prop_map(|v| f64::from(v) / 2.0),
        -url::MAX_KERNEL_VALUE..=url::MAX_KERNEL_VALUE,
    ];
    let kernel = (
        prop_oneof![Just(9usize), Just(25usize)]
            .prop_flat_map(move |n| prop::collection::vec(kernel_value.clone(), n)),
        any::<bool>(),
    );
    prop_oneof![
        spelled(rgb(), |c| call("fill", &[hex(c)])).prop_map(|(c, s)| (Filter::Fill(c), s)),
        spelled(rgb(), |c| call("monochrome", &[hex(c)]))
            .prop_map(|(c, s)| (Filter::Monochrome(c), s)),
        spelled((rgb(), rgb()), |(a, b)| call("duotone", &[hex(a), hex(b)]))
            .prop_map(|((a, b), s)| (Filter::Duotone(a, b), s)),
        spelled((channel, points), |((_, name), points)| {
            // Two spellings: every coordinate bare, or every
            // coordinate with a leading zero.
            (0..2)
                .map(|variant| {
                    let mut text = format!("curve({name}");
                    for (x, y) in points {
                        text.push_str(&format!(",{},{}", uint(*x)[variant], uint(*y)[variant]));
                    }
                    text + ")"
                })
                .collect()
        })
        .prop_map(|((channel, points), s)| (Filter::Curve(channel.0, points), s)),
        spelled(kernel, |(values, normalize)| {
            let flag = if *normalize { "1" } else { "0" };
            (0..2)
                .map(|variant| {
                    let mut parts: Vec<String> = values
                        .iter()
                        .map(|v| {
                            let text = if *v == v.trunc() {
                                format!("{}", *v as i64)
                            } else {
                                format!("{v}")
                            };
                            decimal(text)[variant].clone()
                        })
                        .collect();
                    parts.push(flag.to_string());
                    format!("convolution({})", parts.join(","))
                })
                .collect()
        })
        .prop_map(|((values, normalize), s)| (Filter::Convolution(values, normalize), s)),
    ]
    .boxed()
}

/// Geometry and shape filters.
fn geometry_filter() -> BoxedStrategy<Spelled<Filter>> {
    let rect = || {
        (
            0..=MAX_DIMENSION,
            0..=MAX_DIMENSION,
            1..=MAX_DIMENSION,
            1..=MAX_DIMENSION,
        )
    };
    let region = |name: &'static str| {
        spelled(rect(), move |(x, y, w, h)| {
            call(name, &[uint(*x), uint(*y), uint(*w), uint(*h)])
        })
    };
    prop_oneof![
        spelled(
            prop_oneof![
                (-36000i32..=36000).prop_map(|v| v as f32),
                (-400i32..=400).prop_map(|q| q as f32 * 90.0),
                -36000.0f32..=36000.0,
            ],
            |v| {
                let text = if *v == v.trunc() {
                    format!("{}", *v as i64)
                } else {
                    format!("{v}")
                };
                call("rotate", &[decimal(text)])
            }
        )
        .prop_map(|(v, s)| (Filter::Rotate(v), s)),
        spelled(any::<u8>(), |v| {
            let mut all = call("trim", &[uint(*v)]);
            if *v == 10 {
                all.extend(["trim".to_string(), "trim()".to_string()]);
            }
            all
        })
        .prop_map(|(v, s)| (Filter::Trim(v), s)),
        spelled((any::<u8>(), any::<[u32; 4]>()), |(t, [l, tp, r, b])| {
            call("trim", &[uint(*t), uint(*l), uint(*tp), uint(*r), uint(*b)])
        })
        .prop_map(|((t, sides), s)| (Filter::TrimSides(t, sides), s)),
        spelled(
            select(vec![
                (FlipDirection::Horizontal, "h"),
                (FlipDirection::Vertical, "v")
            ]),
            |(_, name)| call("flip", &[vec![name.to_string()]])
        )
        .prop_map(|((d, _), s)| (Filter::Flip(d), s)),
        spelled(
            prop_oneof![
                (0..=MAX_DIMENSION).prop_map(|r| [r; 4]),
                prop::array::uniform4(0..=MAX_DIMENSION),
            ],
            |r| {
                let mut all = call("radius", &[uint(r[0]), uint(r[1]), uint(r[2]), uint(r[3])]);
                if r.iter().all(|v| *v == r[0]) {
                    all.extend(call("radius", &[uint(r[0])]));
                }
                all
            }
        )
        .prop_map(|(r, s)| (Filter::Radius(r), s)),
        spelled((0..=MAX_DIMENSION, rgb()), |(w, c)| call(
            "border",
            &[uint(*w), hex(c)]
        ))
        .prop_map(|((w, c), s)| (Filter::Border(w, c), s)),
        spelled(
            select(vec![
                (MaskShape::Circle, "circle"),
                (MaskShape::Ellipse, "ellipse")
            ]),
            |(_, name)| call("mask", &[vec![name.to_string()]])
        )
        .prop_map(|((m, _), s)| (Filter::Mask(m), s)),
        spelled(1..=MAX_DIMENSION, |v| call("pixelate", &[uint(*v)]))
            .prop_map(|(v, s)| (Filter::Pixelate(v), s)),
        region("blur_region").prop_map(|((x, y, w, h), s)| (Filter::BlurRegion { x, y, w, h }, s)),
        region("pixelate_region")
            .prop_map(|((x, y, w, h), s)| (Filter::PixelateRegion { x, y, w, h }, s)),
        region("remove_region")
            .prop_map(|((x, y, w, h), s)| (Filter::RemoveRegion { x, y, w, h }, s)),
    ]
    .boxed()
}

/// A per-format quality list, in the parser's canonical
/// entry order (the fallback first, then by format name).
/// It names at least one format: a lone fallback is the
/// single form, `Quality::Value` or `Quality::Auto`.
fn per_format_quality() -> impl Strategy<Value = Vec<(Option<ImageFormat>, QualityValue)>> {
    let still = vec![
        ImageFormat::Jpeg,
        ImageFormat::Png,
        ImageFormat::WebP,
        ImageFormat::Jxl,
        ImageFormat::Avif,
        ImageFormat::Gif,
    ];
    (
        proptest::option::of(quality_value()),
        prop::sample::subsequence(still, 1..=6).prop_flat_map(|formats| {
            let n = formats.len();
            (Just(formats), prop::collection::vec(quality_value(), n))
        }),
    )
        .prop_map(|(fallback, (formats, values))| {
            let mut entries: Vec<(Option<ImageFormat>, QualityValue)> =
                fallback.map(|v| (None, v)).into_iter().collect();
            entries.extend(formats.into_iter().map(Some).zip(values));
            sort_quality(&mut entries);
            entries
        })
}

fn sort_quality(entries: &mut [(Option<ImageFormat>, QualityValue)]) {
    entries.sort_by_key(|(format, _)| format.map_or("", format_name));
}

fn quality_spellings(quality: &Quality) -> Vec<String> {
    match quality {
        Quality::Auto => vec!["quality(auto)".into()],
        Quality::Value(v) => call("quality", &[uint(*v)]),
        Quality::PerFormat(entries) => {
            let parts: Vec<Vec<String>> = entries
                .iter()
                .map(|(format, value)| match format {
                    None => quality_value_spellings(*value),
                    Some(f) => format_spellings(*f)
                        .iter()
                        .flat_map(|name| {
                            quality_value_spellings(*value)
                                .into_iter()
                                .map(move |v| format!("{name}:{v}"))
                        })
                        .collect(),
                })
                .collect();
            // The canonical order, and the same entries
            // reversed: entry order is spelling, not meaning.
            let mut reversed = parts.clone();
            reversed.reverse();
            let mut all = call("quality", &parts);
            all.extend(call("quality", &reversed));
            all
        }
    }
}

fn format_spec() -> impl Strategy<Value = FormatSpec> {
    let subsampling = select(vec![
        Subsampling::S444,
        Subsampling::S422,
        Subsampling::S420,
    ]);
    image_format().prop_flat_map(move |format| {
        let options = match format {
            ImageFormat::Jpeg => (any::<bool>(), proptest::option::of(subsampling.clone()))
                .prop_map(|(progressive, sub)| FormatOptions {
                    jpeg_progressive: progressive,
                    jpeg_subsampling: sub,
                    ..FormatOptions::default()
                })
                .boxed(),
            ImageFormat::Png => any::<bool>()
                .prop_map(|interlace| FormatOptions {
                    png_interlace: interlace,
                    ..FormatOptions::default()
                })
                .boxed(),
            ImageFormat::WebP => (
                proptest::option::of(0u8..=6),
                proptest::option::of(0u8..=100),
            )
                .prop_map(|(method, sns)| FormatOptions {
                    webp_method: method,
                    webp_sns: sns,
                    ..FormatOptions::default()
                })
                .boxed(),
            _ => Just(FormatOptions::default()).boxed(),
        };
        options.prop_map(move |options| FormatSpec { format, options })
    })
}

fn format_spec_spellings(spec: &FormatSpec) -> Vec<String> {
    let o = &spec.options;
    let mut options: Vec<Vec<String>> = Vec::new();
    if o.jpeg_progressive {
        options.push(vec!["progressive".into()]);
    }
    if let Some(sub) = o.jpeg_subsampling {
        options.push(vec![format!("subsampling:{}", sub.name())]);
    }
    if o.png_interlace {
        options.push(vec!["interlace".into()]);
    }
    if let Some(m) = o.webp_method {
        options.push(uint(m).into_iter().map(|v| format!("method:{v}")).collect());
    }
    if let Some(s) = o.webp_sns {
        options.push(uint(s).into_iter().map(|v| format!("sns:{v}")).collect());
    }
    // Options in canonical order and reversed: their order
    // is spelling, not meaning.
    let mut reversed = options.clone();
    reversed.reverse();
    let mut all = Vec::new();
    for order in [options, reversed] {
        let mut parts = vec![format_spellings(spec.format)];
        parts.extend(order);
        all.extend(call("format", &parts));
    }
    all
}

/// Encoding directives and request-wide switches.
fn encoding_filter() -> BoxedStrategy<Spelled<Filter>> {
    let quality = prop_oneof![
        (1u8..=100).prop_map(Quality::Value),
        Just(Quality::Auto),
        per_format_quality().prop_map(Quality::PerFormat),
    ];
    prop_oneof![
        spelled(quality, quality_spellings).prop_map(|(q, s)| (Filter::Quality(q), s)),
        spelled(format_spec(), format_spec_spellings).prop_map(|(f, s)| (Filter::Format(f), s)),
        spelled(2u16..=256, |v| call("palette", &[uint(*v)]))
            .prop_map(|(v, s)| (Filter::Palette(v), s)),
        spelled(1u64..=u64::MAX, |v| call("max_bytes", &[uint(*v)]))
            .prop_map(|(v, s)| (Filter::MaxBytes(v), s)),
        spelled(select(vec![1.0f32, 1.5, 2.0, 3.0]), |v| call(
            "dpr",
            &[decimal32(*v)]
        ))
        .prop_map(|(v, s)| (Filter::Dpr(v), s)),
        bare(Filter::NoUpscale, "no_upscale"),
        bare(Filter::KeepProfile, "keep_profile"),
    ]
    .boxed()
}

fn studio_spec() -> impl Strategy<Value = StudioSpec> {
    let bg = prop_oneof![
        Just(StudioBg::Default),
        Just(StudioBg::White),
        Just(StudioBg::Brand),
        Just(StudioBg::None),
        rgb().prop_map(StudioBg::Color),
    ];
    let shadow = select(vec![
        StudioShadow::None,
        StudioShadow::Soft,
        StudioShadow::Strong,
    ]);
    prop_oneof![
        (bg, shadow, 0u8..=40).prop_map(|(bg, shadow, pad)| StudioSpec {
            mode: StudioMode::Product,
            bg,
            shadow,
            pad,
        }),
        (0u8..=40).prop_map(|pad| StudioSpec {
            mode: StudioMode::Scene,
            pad,
            ..StudioSpec::default()
        }),
    ]
}

fn studio_spellings(spec: &StudioSpec) -> Vec<String> {
    // Each named argument's spellings, and whether it holds
    // the default (so it may also be left out).
    let mut named: Vec<(Vec<String>, bool)> = Vec::new();
    if spec.mode == StudioMode::Product {
        let bg = match spec.bg {
            StudioBg::Default => Vec::new(),
            StudioBg::White => vec!["bg:white".to_string()],
            StudioBg::Brand => vec!["bg:brand".to_string()],
            StudioBg::None => vec!["bg:none".to_string()],
            StudioBg::Color(c) => hex(&c).into_iter().map(|h| format!("bg:{h}")).collect(),
        };
        if !bg.is_empty() {
            named.push((bg, false));
        }
        named.push((
            vec![format!("shadow:{}", spec.shadow.name())],
            spec.shadow == StudioShadow::Soft,
        ));
    }
    named.push((
        uint(spec.pad)
            .into_iter()
            .map(|p| format!("pad:{p}"))
            .collect(),
        spec.pad == 10,
    ));
    let mut all = Vec::new();
    for spell_defaults in [true, false] {
        let mut parts: Vec<Vec<String>> = named
            .iter()
            .filter(|(_, default)| spell_defaults || !default)
            .map(|(spellings, _)| spellings.clone())
            .collect();
        // Named arguments in either order.
        for _ in 0..2 {
            let mut full = vec![vec![spec.mode.name().to_string()]];
            full.extend(parts.iter().cloned());
            all.extend(call("studio", &full));
            parts.reverse();
        }
    }
    all
}

/// The filters backed by AI models.
fn ai_filter() -> BoxedStrategy<Spelled<Filter>> {
    let class = |name: &'static str| spelled(class_name(), move |c| call(name, &[encoded(c, "")]));
    prop_oneof![
        bare(Filter::RemoveBg, "removebg"),
        bare(Filter::BlurFaces, "blurfaces"),
        bare(Filter::PixelateFaces, "pixelatefaces"),
        class("blurclass").prop_map(|(c, s)| (Filter::BlurClass(c), s)),
        class("pixelateclass").prop_map(|(c, s)| (Filter::PixelateClass(c), s)),
        spelled((class_name(), rgb()), |(name, c)| call(
            "recolorclass",
            &[encoded(name, ""), hex(c)]
        ))
        .prop_map(|((name, c), s)| (Filter::RecolorClass(name, c), s)),
        spelled(prop_oneof![0.0f32..=3.0, Just(3.0)], |v| call(
            "facepad",
            &[decimal32(*v)]
        ))
        .prop_map(|(v, s)| (Filter::FacePad(v), s)),
        spelled(select(vec![2u32, 4]), |v| call("upscale", &[uint(*v)]))
            .prop_map(|(v, s)| (Filter::Upscale(v), s)),
        spelled(studio_spec(), studio_spellings).prop_map(|(spec, s)| (Filter::Studio(spec), s)),
    ]
    .boxed()
}

fn anchor() -> impl Strategy<Value = (Anchor, &'static str)> {
    select(vec![
        (Anchor::TopLeft, "top-left"),
        (Anchor::TopCenter, "top-center"),
        (Anchor::TopRight, "top-right"),
        (Anchor::CenterLeft, "center-left"),
        (Anchor::Center, "center"),
        (Anchor::CenterRight, "center-right"),
        (Anchor::BottomLeft, "bottom-left"),
        (Anchor::BottomCenter, "bottom-center"),
        (Anchor::BottomRight, "bottom-right"),
    ])
}

/// Text, watermark, and overlay filters: free-text arguments.
/// Text labels and watermark URLs may also carry raw commas
/// (their fixed fields are split off from the right); an
/// overlay URL may not, because its optional trailing width
/// makes a comma in the URL ambiguous: with six or more
/// fields the last one is read as the width (the playground
/// and the TypeScript explainer mirror that rule).
fn overlay_filter() -> BoxedStrategy<Spelled<Filter>> {
    let keep = "!$&'*+,;=:@";
    let overlay_keep = "!$&'*+;=:@";
    let blend = select(vec![
        (BlendMode::Normal, "normal"),
        (BlendMode::Multiply, "multiply"),
        (BlendMode::Screen, "screen"),
        (BlendMode::Overlay, "overlay"),
    ]);
    prop_oneof![
        spelled(
            (free_text(), 1u32..=1000, any::<[u8; 4]>(), anchor()),
            move |(text, size, color, (_, anchor))| {
                let mut colors = hex(color);
                if color[3] == 255 {
                    colors.extend(hex(&color[..3]));
                }
                call(
                    "text",
                    &[
                        encoded(text, keep),
                        uint(*size),
                        colors,
                        vec![anchor.to_string()],
                    ],
                )
            }
        )
        .prop_map(|((text, size, color, (anchor, _)), s)| (
            Filter::Text {
                text,
                size,
                color,
                anchor
            },
            s
        )),
        spelled(
            (free_text(), any::<i32>(), any::<i32>(), 0u8..=100),
            move |(url, x, y, alpha)| call(
                "watermark",
                &[encoded(url, keep), int(*x), int(*y), uint(*alpha)]
            )
        )
        .prop_map(|((url, x, y, alpha), s)| (Filter::Watermark { url, x, y, alpha }, s)),
        spelled(
            (
                free_text(),
                any::<i32>(),
                any::<i32>(),
                0u8..=100,
                blend,
                proptest::option::of(1u32..=400)
            ),
            move |(url, x, y, alpha, (_, blend), width)| {
                let mut parts = vec![
                    encoded(url, overlay_keep),
                    int(*x),
                    int(*y),
                    uint(*alpha),
                    vec![blend.to_string()],
                ];
                if let Some(w) = width {
                    parts.push(uint(*w));
                }
                call("overlay", &parts)
            }
        )
        .prop_map(|((url, x, y, alpha, (blend, _), width), s)| (
            Filter::Overlay {
                url,
                x,
                y,
                alpha,
                blend,
                width
            },
            s
        )),
    ]
    .boxed()
}

fn any_filter() -> BoxedStrategy<Spelled<Filter>> {
    prop_oneof![
        tone_filter(),
        color_filter(),
        geometry_filter(),
        encoding_filter(),
        ai_filter(),
        overlay_filter(),
    ]
    .boxed()
}

/// One `filters:` segment of one to six filters.
fn filters_op(filter: BoxedStrategy<Spelled<Filter>>) -> BoxedStrategy<Spelled<Operation>> {
    prop::collection::vec(filter, 1..=6)
        .prop_map(|list| {
            let (filters, texts): (Vec<Filter>, Vec<String>) = list.into_iter().unzip();
            (
                Operation::Filters(filters),
                format!("filters:{}", texts.join(":")),
            )
        })
        .boxed()
}

// ---- Operations, by family ----

fn dimension() -> impl Strategy<Value = Dimension> {
    prop_oneof![
        4 => (0..=MAX_DIMENSION).prop_map(Dimension::Pixels),
        1 => Just(Dimension::Pixels(0)),
        1 => Just(Dimension::Pixels(MAX_DIMENSION)),
        1 => Just(Dimension::Proportional),
        1 => Just(Dimension::Original),
    ]
}

fn dimension_spellings(d: Dimension, flip: bool) -> Vec<String> {
    let sign = if flip { "-" } else { "" };
    let bare = match d {
        // "0" is proportional; only a longer run of zeros
        // reads back as an exact zero.
        Dimension::Pixels(0) => vec!["00".to_string(), "000".to_string()],
        Dimension::Pixels(n) => uint(n),
        Dimension::Proportional => vec![String::new(), "0".to_string()],
        Dimension::Original => vec!["orig".to_string()],
    };
    bare.into_iter().map(|b| format!("{sign}{b}")).collect()
}

fn size_spellings(size: &ImageSize) -> Vec<String> {
    let mut all = Vec::new();
    for w in dimension_spellings(size.width, size.flip_horizontal) {
        for h in dimension_spellings(size.height, size.flip_vertical) {
            all.push(format!("{w}x{h}"));
        }
    }
    all
}

fn image_size(flips: bool) -> impl Strategy<Value = ImageSize> {
    let flip = if flips {
        any::<bool>().boxed()
    } else {
        Just(false).boxed()
    };
    (dimension(), dimension(), flip.clone(), flip).prop_map(|(width, height, fh, fv)| ImageSize {
        width,
        height,
        flip_horizontal: fh,
        flip_vertical: fv,
    })
}

/// The exact box `crop:` and `pad:` take: pixels on both
/// axes, at least 1, no flips.
fn exact_box() -> impl Strategy<Value = ImageSize> {
    let side = prop_oneof![1..=MAX_DIMENSION, Just(1), Just(MAX_DIMENSION)];
    (side.clone(), side).prop_map(|(w, h)| ImageSize {
        width: Dimension::Pixels(w),
        height: Dimension::Pixels(h),
        flip_horizontal: false,
        flip_vertical: false,
    })
}

fn size_op() -> BoxedStrategy<Spelled<Operation>> {
    spelled(image_size(true), size_spellings)
        .prop_map(|(size, s)| (Operation::Size(size), s))
        .boxed()
}

fn fit_in_and_pad_op() -> BoxedStrategy<Spelled<Operation>> {
    let fill = prop_oneof![
        Just(PadFill::Directive),
        Just(PadFill::Blur),
        rgb().prop_map(PadFill::Color),
    ];
    prop_oneof![
        spelled((image_size(false), any::<bool>()), |(size, adaptive)| {
            size_spellings(size)
                .into_iter()
                .map(|s| {
                    if *adaptive {
                        format!("fit-in:{s}:adaptive")
                    } else {
                        format!("fit-in:{s}")
                    }
                })
                .collect()
        })
        .prop_map(|((size, adaptive), s)| (Operation::FitIn { size, adaptive }, s)),
        spelled((exact_box(), fill), |(size, fill)| {
            let fills = match fill {
                PadFill::Directive => vec![String::new()],
                PadFill::Blur => vec![":blur".to_string()],
                PadFill::Color(c) => hex(c).into_iter().map(|h| format!(":{h}")).collect(),
            };
            size_spellings(size)
                .into_iter()
                .flat_map(|s| fills.iter().map(move |f| format!("pad:{s}{f}")))
                .collect()
        })
        .prop_map(|((size, fill), s)| (Operation::Pad { size, fill }, s)),
    ]
    .boxed()
}

/// Distinct class names (the parser keeps the first
/// spelling of a name repeated in any ASCII case).
fn classes() -> impl Strategy<Value = Vec<String>> {
    prop::collection::vec(class_name(), 1..=4).prop_map(|names| {
        let mut kept: Vec<String> = Vec::new();
        for name in names {
            if !kept.iter().any(|k| k.eq_ignore_ascii_case(&name)) {
                kept.push(name);
            }
        }
        kept
    })
}

fn align() -> impl Strategy<Value = (Align, &'static str)> {
    prop_oneof![
        8 => select(vec![
            (Align::Center, "center"),
            (Align::Smart, "smart"),
            (Align::Face, "face"),
            (Align::Top, "top"),
            (Align::Bottom, "bottom"),
            (Align::Left, "left"),
            (Align::Right, "right"),
            (Align::TopLeft, "top-left"),
            (Align::TopRight, "top-right"),
            (Align::BottomLeft, "bottom-left"),
            (Align::BottomRight, "bottom-right"),
        ]),
        2 => (
            classes(),
            prop_oneof![
                Just(1.0f32),
                (0.0f32..=url::MAX_OBJECTS_SCALE).prop_filter("positive", |v| *v > 0.0),
                Just(url::MAX_OBJECTS_SCALE),
            ],
        )
            .prop_map(|(classes, scale)| (Align::Objects { classes, scale }, "objects")),
    ]
}

fn align_spellings(align: &Align, name: &str) -> Vec<String> {
    match align {
        Align::Center => vec![String::new(), ":center".to_string()],
        Align::Objects { classes, scale } => {
            let lists: Vec<Vec<String>> = classes.iter().map(|c| encoded(c, "")).collect();
            // A repeat of the first name in another ASCII case
            // names the same class and is dropped.
            let first = &classes[0];
            let swapped: String = first
                .chars()
                .map(|c| {
                    if c.is_ascii_lowercase() {
                        c.to_ascii_uppercase()
                    } else {
                        c.to_ascii_lowercase()
                    }
                })
                .collect();
            let with_repeat = {
                let mut l = lists.clone();
                l.push(encoded(&swapped, ""));
                l
            };
            let scales: Vec<String> = if *scale == 1.0 {
                vec![String::new(), ":1".to_string(), ":1.0".to_string()]
            } else {
                decimal32(*scale)
                    .into_iter()
                    .map(|s| format!(":{s}"))
                    .collect()
            };
            let mut all = Vec::new();
            for list in [args(&lists), args(&with_repeat)] {
                for l in list {
                    for s in &scales {
                        all.push(format!(":objects({l}){s}"));
                    }
                }
            }
            all
        }
        _ => vec![format!(":{name}")],
    }
}

fn crop_op() -> BoxedStrategy<Spelled<Operation>> {
    spelled((exact_box(), align()), |(size, (align, name))| {
        let aligns = align_spellings(align, name);
        size_spellings(size)
            .into_iter()
            .flat_map(|s| aligns.iter().map(move |a| format!("crop:{s}{a}")))
            .collect()
    })
    .prop_map(|((size, (align, _)), s)| (Operation::Crop { size, align }, s))
    .boxed()
}

/// An ordered span `start < end` over the whole `u32` range:
/// rectangle corners are bounded by the image at render
/// time, not by the grammar.
fn span() -> impl Strategy<Value = (u32, u32)> {
    prop_oneof![
        (0u32..u32::MAX).prop_flat_map(|start| (Just(start), start + 1..=u32::MAX)),
        (0u32..=MAX_DIMENSION).prop_flat_map(|start| (Just(start), start + 1..=MAX_DIMENSION + 1)),
    ]
}

fn rect_spellings(
    prefix: &str,
    (left, right): (u32, u32),
    (top, bottom): (u32, u32),
) -> Vec<String> {
    (0..2)
        .map(|v| {
            format!(
                "{prefix}:{}x{}:{}x{}",
                uint(left)[v],
                uint(top)[v],
                uint(right)[v],
                uint(bottom)[v]
            )
        })
        .collect()
}

fn rectangle_and_point_op() -> BoxedStrategy<Spelled<Operation>> {
    let fraction = || prop_oneof![0.0f32..=1.0, Just(0.0), Just(1.0)];
    let zoom = prop_oneof![
        Just(1.0f32),
        (0.0f32..=url::MAX_FOCAL_ZOOM).prop_filter("positive", |v| *v > 0.0),
        Just(url::MAX_FOCAL_ZOOM),
    ];
    prop_oneof![
        spelled((span(), span()), |(x, y)| rect_spellings(
            "crop-rect",
            *x,
            *y
        ))
        .prop_map(|(((left, right), (top, bottom)), s)| (
            Operation::CropRect {
                left,
                top,
                right,
                bottom
            },
            s
        )),
        spelled((span(), span()), |(x, y)| rect_spellings("focal", *x, *y)).prop_map(
            |(((left, right), (top, bottom)), s)| (
                Operation::FocalRect {
                    left,
                    top,
                    right,
                    bottom
                },
                s
            )
        ),
        spelled((fraction(), fraction(), zoom), |(x, y, zoom)| {
            let zooms: Vec<String> = if *zoom == 1.0 {
                vec![String::new(), ":1".to_string()]
            } else {
                decimal32(*zoom)
                    .into_iter()
                    .map(|z| format!(":{z}"))
                    .collect()
            };
            let mut all = Vec::new();
            for xs in decimal32(*x) {
                for ys in decimal32(*y) {
                    for z in &zooms {
                        all.push(format!("focalpoint:{xs}x{ys}{z}"));
                    }
                }
            }
            all
        })
        .prop_map(|((x, y, zoom), s)| (Operation::FocalPoint { x, y, zoom }, s)),
    ]
    .boxed()
}

fn frame_page_and_debug_op() -> BoxedStrategy<Spelled<Operation>> {
    let seconds = prop_oneof![
        0.0f64..=url::MAX_FRAME_SECONDS,
        (0u32..=86_400).prop_map(f64::from),
        Just(url::MAX_FRAME_SECONDS),
    ];
    let percent = prop_oneof![
        0.0f64..=url::MAX_FRAME_PERCENT,
        (0u32..=100).prop_map(f64::from),
    ];
    let whole_or_shortest = |v: f64| {
        if v == v.trunc() {
            format!("{}", v as i64)
        } else {
            format!("{v}")
        }
    };
    prop_oneof![
        Just((
            Operation::Frame(FrameSelection::Poster),
            "frame:poster".to_string()
        )),
        spelled(seconds, move |v| decimal(whole_or_shortest(*v))
            .into_iter()
            .map(|t| format!("frame:{t}"))
            .collect())
        .prop_map(|(v, s)| (Operation::Frame(FrameSelection::Seconds(v)), s)),
        spelled(percent, move |v| decimal(whole_or_shortest(*v))
            .into_iter()
            .map(|t| format!("frame:{t}%"))
            .collect())
        .prop_map(|(v, s)| (Operation::Frame(FrameSelection::Percent(v)), s)),
        spelled(1u32..=url::MAX_PAGE_NUMBER, |v| {
            let mut all: Vec<String> = uint(*v).into_iter().map(|p| format!("page:{p}")).collect();
            all.push(format!("page:00{v}"));
            all
        })
        .prop_map(|(v, s)| (Operation::Page(v), s)),
        Just((Operation::DebugFocal, "debug:focal".to_string())),
    ]
    .boxed()
}

fn any_op() -> BoxedStrategy<Spelled<Operation>> {
    prop_oneof![
        1 => size_op(),
        1 => crop_op(),
        1 => fit_in_and_pad_op(),
        1 => rectangle_and_point_op(),
        1 => frame_page_and_debug_op(),
        2 => filters_op(any_filter()),
    ]
    .boxed()
}

// ---- Whole requests ----

/// The request metadata a remainder may carry among its
/// operations: not operations, and not in the canonical form.
#[derive(Debug, Clone)]
enum Meta {
    Expiry(u64),
    MaxAge(u32),
}

#[derive(Debug, Clone)]
enum Segment {
    Op(Operation),
    Meta(Meta),
}

#[derive(Debug, Clone)]
struct Case {
    signature_text: String,
    signature: Signature,
    key_id: Option<String>,
    segments: Vec<Spelled<Segment>>,
    source: String,
}

fn meta() -> BoxedStrategy<Spelled<Segment>> {
    prop_oneof![
        spelled(any::<u64>(), |v| uint(*v)
            .into_iter()
            .map(|t| format!("exp:{t}"))
            .collect())
        .prop_map(|(v, s)| (Segment::Meta(Meta::Expiry(v)), s)),
        spelled(0..=url::MAX_CACHE_AGE, |v| uint(*v)
            .into_iter()
            .map(|t| format!("maxage:{t}"))
            .collect())
        .prop_map(|(v, s)| (Segment::Meta(Meta::MaxAge(v)), s)),
    ]
    .boxed()
}

/// A source whose first segment is not an operation: a host,
/// a scheme, a bucket, a root reference, or a file name that
/// looks like a size but carries a dot.
fn source() -> impl Strategy<Value = String> {
    (
        select(vec![
            "example.com",
            "https://cdn.example.com",
            "http://img.example.org:8080",
            "s3://media",
            "root:photos",
            "100x200.jpg",
        ]),
        prop::collection::vec("[A-Za-z0-9._~-]{1,12}", 0..=3),
    )
        .prop_map(|(head, rest)| {
            let mut s = head.to_string();
            for seg in rest {
                s.push('/');
                s.push_str(&seg);
            }
            s
        })
}

fn signature() -> impl Strategy<Value = (String, Signature, Option<String>)> {
    prop_oneof![
        Just(("unsafe".to_string(), Signature::Unsafe, None)),
        "[0-9a-f]{64}".prop_map(|hex| (hex.clone(), Signature::Hmac(hex), None)),
        ("[A-Za-z0-9_-]{1,32}", "[0-9a-f]{64}").prop_map(|(kid, hex)| (
            format!("{kid}:{hex}"),
            Signature::Hmac(hex),
            Some(kid)
        )),
    ]
}

/// A whole request: zero to eight operations of one family,
/// with metadata segments mixed in.
fn case(op: BoxedStrategy<Spelled<Operation>>) -> impl Strategy<Value = Case> {
    let segment = prop_oneof![
        6 => op.prop_map(|(o, s)| (Segment::Op(o), s)),
        1 => meta(),
    ];
    (signature(), prop::collection::vec(segment, 0..=8), source()).prop_map(
        |((signature_text, signature, key_id), segments, source)| Case {
            signature_text,
            signature,
            key_id,
            segments,
            source,
        },
    )
}

/// Operations as the canonical form reads them back:
/// adjacent `filters:` segments merged, and per-format
/// quality lists in canonical entry order.
fn normalize(ops: &[Operation]) -> Vec<Operation> {
    let mut out: Vec<Operation> = Vec::new();
    for op in ops {
        let op = match op {
            Operation::Filters(list) => Operation::Filters(
                list.iter()
                    .map(|f| match f {
                        Filter::Quality(Quality::PerFormat(entries)) => {
                            let mut entries = entries.clone();
                            sort_quality(&mut entries);
                            Filter::Quality(Quality::PerFormat(entries))
                        }
                        other => other.clone(),
                    })
                    .collect(),
            ),
            other => other.clone(),
        };
        match (out.last_mut(), op) {
            (Some(Operation::Filters(prev)), Operation::Filters(next)) => prev.extend(next),
            (_, op) => out.push(op),
        }
    }
    out
}

fn operation_count(ops: &[Operation]) -> usize {
    ops.iter()
        .map(|op| match op {
            Operation::Filters(list) => list.len(),
            _ => 1,
        })
        .sum()
}

fn check(case: &Case) -> Result<(), TestCaseError> {
    let mut remainder = String::new();
    let mut generated = Vec::new();
    let (mut expiry, mut max_age) = (None, None);
    for (segment, text) in &case.segments {
        remainder.push_str(text);
        remainder.push('/');
        match segment {
            Segment::Op(op) => generated.push(op.clone()),
            Segment::Meta(Meta::Expiry(v)) => expiry = Some(*v),
            Segment::Meta(Meta::MaxAge(v)) => max_age = Some(*v),
        }
    }
    remainder.push_str(&case.source);
    let expected = normalize(&generated);
    prop_assert!(operation_count(&expected) <= OPERATION_CAP);

    let path = format!("/{}/{remainder}", case.signature_text);
    let (signature, key_id, request) = url::parse_request(&path)
        .map_err(|e| TestCaseError::fail(format!("{path} does not parse: {e}")))?;
    prop_assert_eq!(&signature, &case.signature);
    prop_assert_eq!(&key_id, &case.key_id);
    prop_assert_eq!(request.source_url, case.source.as_str());
    prop_assert_eq!(normalize(&request.operations), expected.clone(), "{}", path);

    let (_, _, exp, age) = url::parse_operations_with_expiry(&remainder)
        .map_err(|e| TestCaseError::fail(e.to_string()))?;
    prop_assert_eq!(
        (exp, age),
        (expiry, max_age),
        "the last metadata of each kind wins"
    );

    let canonical = url::canonical_form(&request.operations, request.source_url);
    let reparsed = format!("/{}/{canonical}", case.signature_text);
    let (signature2, key_id2, again) = url::parse_request(&reparsed)
        .map_err(|e| TestCaseError::fail(format!("canonical {canonical} does not parse: {e}")))?;
    prop_assert_eq!(&signature2, &case.signature);
    prop_assert_eq!(&key_id2, &case.key_id);
    prop_assert_eq!(&again.operations, &expected, "canonical {}", canonical);
    prop_assert_eq!(
        url::canonical_form(&again.operations, again.source_url),
        canonical.clone()
    );
    prop_assert_eq!(url::canonical_key_form(&remainder), canonical.clone());
    prop_assert_eq!(url::canonical_key_form(&canonical), canonical);
    Ok(())
}

proptest! {
    #![proptest_config(config())]

    #[test]
    fn size_chains_round_trip(case in case(size_op())) {
        check(&case)?;
    }

    #[test]
    fn crop_chains_round_trip(case in case(crop_op())) {
        check(&case)?;
    }

    #[test]
    fn fit_in_and_pad_chains_round_trip(case in case(fit_in_and_pad_op())) {
        check(&case)?;
    }

    #[test]
    fn rectangle_and_point_chains_round_trip(case in case(rectangle_and_point_op())) {
        check(&case)?;
    }

    #[test]
    fn frame_page_and_debug_chains_round_trip(case in case(frame_page_and_debug_op())) {
        check(&case)?;
    }

    #[test]
    fn tone_filter_chains_round_trip(case in case(filters_op(tone_filter()))) {
        check(&case)?;
    }

    #[test]
    fn color_filter_chains_round_trip(case in case(filters_op(color_filter()))) {
        check(&case)?;
    }

    #[test]
    fn geometry_filter_chains_round_trip(case in case(filters_op(geometry_filter()))) {
        check(&case)?;
    }

    #[test]
    fn encoding_filter_chains_round_trip(case in case(filters_op(encoding_filter()))) {
        check(&case)?;
    }

    #[test]
    fn ai_filter_chains_round_trip(case in case(filters_op(ai_filter()))) {
        check(&case)?;
    }

    #[test]
    fn overlay_filter_chains_round_trip(case in case(filters_op(overlay_filter()))) {
        check(&case)?;
    }

    #[test]
    fn mixed_chains_round_trip(case in case(any_op())) {
        check(&case)?;
    }
}
