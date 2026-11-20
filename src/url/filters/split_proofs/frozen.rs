//! Main's filter grammar as it stood before issue #741
//! (step 4) moved the plain filters into their own enum:
//! the oracle `split_proofs` holds the split to. It is
//! main's code at 6b30e25, copied verbatim and run through
//! rustfmt, except that
//!
//! - the `Filter` and `Operation` enums lost their doc
//!   comments, and are `pub(super)`;
//! - `filter_specs!` lost the list of rows the table's own
//!   tests walk (`FILTER_SPECS`), which neither the parser
//!   nor the speller reads;
//! - `RequestDirectives::analyze` is the free function
//!   [`analyze`], which `merge_default_ops` calls in its
//!   place;
//! - `parse_operations_with_expiry` returns the
//!   [`ParsedRemainder`] of the frozen `Operation`;
//! - [`parse_operation`] and [`from_current`] are new glue:
//!   a `filters:` segment parses through the frozen table,
//!   and every other segment through the current parser,
//!   which the split leaves alone.
//!
//! The argument kinds, their bounds and the argument
//! spellers (`curve_args` and the rest) are the current
//! table's: the split does not touch them.
//!
//! When the grammar changes on purpose, delete this module
//! and `split_proofs.rs` with it: they pin the split, not
//! the grammar.

// The current table's kinds, bounds and spellers, and the
// url helpers main's code calls. Main's own definitions
// below shadow the current ones they replace (`Filter`,
// `Operation`, the reader, the speller, the parse).
use crate::url::filters::*;
use crate::url::*;

/// What [`parse_operations_with_expiry`] yields, in the
/// frozen `Operation`.
pub(super) type ParsedRemainder<'a> = (Vec<Operation>, &'a str, Option<u64>, Option<u32>);

/// Main's `parse_operation` as far as the split could reach
/// it: a `filters:` segment parses through main's table,
/// every other segment through the current parser. None of
/// the prefixes the parser tries first (`debug:`, `frame:`,
/// `page:`, `crop-rect:`, `focalpoint:`, `focal:`, `crop:`,
/// `pad:`, `fit-in:`) can start a `filters:` segment, so the
/// order of the tries is main's.
fn parse_operation(seg: &str) -> Option<Result<Operation, ParseError>> {
    if let Some(rest) = seg.strip_prefix("filters:") {
        return Some(parse_filters(rest, seg));
    }
    crate::url::parse_operation(seg).map(|parsed| parsed.map(from_current))
}

/// A current operation other than `filters:`, which is the
/// same value in main's `Operation`.
fn from_current(op: crate::url::Operation) -> Operation {
    use crate::url::Operation as Current;
    match op {
        Current::Size(size) => Operation::Size(size),
        Current::Crop { size, align } => Operation::Crop { size, align },
        Current::FitIn { size, adaptive } => Operation::FitIn { size, adaptive },
        Current::Pad { size, fill } => Operation::Pad { size, fill },
        Current::CropRect {
            left,
            top,
            right,
            bottom,
        } => Operation::CropRect {
            left,
            top,
            right,
            bottom,
        },
        Current::FocalRect {
            left,
            top,
            right,
            bottom,
        } => Operation::FocalRect {
            left,
            top,
            right,
            bottom,
        },
        Current::FocalPoint { x, y, zoom } => Operation::FocalPoint { x, y, zoom },
        Current::DebugFocal => Operation::DebugFocal,
        Current::Frame(selection) => Operation::Frame(selection),
        Current::Page(page) => Operation::Page(page),
        Current::Filters(_) => unreachable!("a filters: segment parses through main's table"),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum Filter {
    Grayscale,
    Invert,
    Blur(f32),
    Brighten(i32),
    Contrast(f32),
    Gamma(f32),
    Exposure(f32),
    HighlightsShadows(f32, f32),
    Vibrance(f32),
    HueRotate(i32),
    Sharpen(f32),
    Saturation(f32),
    Rgb(i32, i32, i32),
    Curve(CurveChannel, Vec<(u8, u8)>),
    Convolution(Vec<f64>, bool),
    Noise(u8),
    Equalize,
    Enhance,
    Rotate(f32),
    Trim(u8),
    TrimSides(u8, [u32; 4]),
    Watermark {
        url: String,
        x: i32,
        y: i32,
        alpha: u8,
    },
    Overlay {
        url: String,
        x: i32,
        y: i32,
        alpha: u8,
        blend: BlendMode,
        width: Option<u32>,
    },
    RemoveBg,
    Studio(StudioSpec),
    KeepProfile,
    BlurFaces,
    PixelateFaces,
    BlurClass(String),
    PixelateClass(String),
    RecolorClass(String, [u8; 3]),
    FacePad(f32),
    Upscale(u32),
    Dpr(f32),
    Text {
        text: String,
        size: u32,
        color: [u8; 4],
        anchor: Anchor,
    },
    Fill([u8; 3]),
    Radius([u32; 4]),
    Border(u32, [u8; 3]),
    Mask(MaskShape),
    Sepia,
    Duotone([u8; 3], [u8; 3]),
    Monochrome([u8; 3]),
    Pixelate(u32),
    Flip(FlipDirection),
    BlurRegion {
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    },
    PixelateRegion {
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    },
    RemoveRegion {
        x: u32,
        y: u32,
        w: u32,
        h: u32,
    },
    MaxBytes(u64),
    Quality(Quality),
    Palette(u16),
    Format(FormatSpec),
    NoUpscale,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum Operation {
    Size(ImageSize),
    Crop {
        size: ImageSize,
        align: Align,
    },
    FitIn {
        size: ImageSize,
        adaptive: bool,
    },
    Pad {
        size: ImageSize,
        fill: PadFill,
    },
    CropRect {
        left: u32,
        top: u32,
        right: u32,
        bottom: u32,
    },
    FocalRect {
        left: u32,
        top: u32,
        right: u32,
        bottom: u32,
    },
    FocalPoint {
        x: f32,
        y: f32,
        zoom: f32,
    },
    DebugFocal,
    Frame(FrameSelection),
    Page(u32),
    Filters(Vec<Filter>),
}

/// Generate the reader, the speller, and the argument error
/// codes from the rows (see the module docs). A recursive
/// macro reads one row at a time, appending to three lists:
/// the reader's match arms, the speller's match arms, and
/// the error codes.
macro_rules! filter_specs {
    (@rows [$($read:tt)*] [$($spell:tt)*] [$($error:tt)*]) => {
        /// The filter `name` builds from its arguments, or
        /// None when no row accepts them. `params` is the text
        /// between the parentheses (None when there are none,
        /// or they are empty) and `args` that text split on
        /// commas.
        fn build(name: &str, params: Option<&str>, args: &[&str]) -> Option<Filter> {
            match (name, params, args) {
                $($read)*
                _ => None,
            }
        }

        /// The one canonical spelling of a filter: the grammar
        /// text the parser produces this exact value from.
        pub(super) fn filter_text(filter: &Filter) -> String {
            match filter {
                $($spell)*
            }
        }

        /// The error a bad argument list for `name` answers.
        fn argument_error(name: &str) -> fn(String) -> ParseError {
            match name {
                $($error)*
                _ => ParseError::InvalidFilters,
            }
        }
    };

    // `else Code`: the row's argument errors carry this code.
    (@rows [$($read:tt)*] [$($spell:tt)*] [$($error:tt)*]
        $name:literal $args:tt else $code:ident => $($rest:tt)*) => {
        filter_specs!(@rows
            [$($read)*]
            [$($spell)*]
            [$($error)* $name => ParseError::$code,]
            $name $args => $($rest)*);
    };

    // A bare shorthand: read, never written.
    (@rows [$($read:tt)*] [$($spell:tt)*] [$($error:tt)*]
        shorthand $name:literal => $variant:ident $fields:tt; $($rest:tt)*) => {
        filter_specs!(@rows
            [$($read)* ($name, None, _) => Some(Filter::$variant $fields),]
            [$($spell)*]
            [$($error)*]
            $($rest)*);
    };

    // A shorthand with arguments: read, never written.
    (@rows [$($read:tt)*] [$($spell:tt)*] [$($error:tt)*]
        shorthand $name:literal ($($arg:ident: $kind:ident),+) => $variant:ident $fields:tt;
        $($rest:tt)*) => {
        filter_specs!(@rows
            [$($read)* ($name, Some(_), [$($arg),+]) => {
                $(let $arg = $kind.read($arg)?;)+
                Some(Filter::$variant $fields)
            }]
            [$($spell)*]
            [$($error)*]
            $($rest)*);
    };

    // A sub-grammar of its own.
    (@rows [$($read:tt)*] [$($spell:tt)*] [$($error:tt)*]
        $name:literal (custom $parse:expr) => $variant:ident $fields:tt spelled $text:expr;
        $($rest:tt)*) => {
        filter_specs!(@rows
            [$($read)* ($name, Some(params), _) => {
                let read: fn(&str) -> Option<Filter> = $parse;
                read(params)
            }]
            [$($spell)* Filter::$variant $fields => format!(concat!($name, "({})"), $text),]
            [$($error)*]
            $($rest)*);
    };

    // No arguments.
    (@rows [$($read:tt)*] [$($spell:tt)*] [$($error:tt)*]
        $name:literal => $variant:ident; $($rest:tt)*) => {
        filter_specs!(@rows
            [$($read)* ($name, None, _) => Some(Filter::$variant),]
            [$($spell)* Filter::$variant => String::from($name),]
            [$($error)*]
            $($rest)*);
    };

    // One kind per argument.
    (@rows [$($read:tt)*] [$($spell:tt)*] [$($error:tt)*]
        $name:literal ($($arg:ident: $kind:ident),+) => $variant:ident $fields:tt;
        $($rest:tt)*) => {
        filter_specs!(@rows
            [$($read)* ($name, Some(_), [$($arg),+]) => {
                $(let $arg = $kind.read($arg)?;)+
                Some(Filter::$variant $fields)
            }]
            [$($spell)* Filter::$variant $fields => format!(
                concat!($name, "({})"),
                [$($kind.spell($arg)),+].join(",")
            ),]
            [$($error)*]
            $($rest)*);
    };

    ($($rows:tt)*) => {
        filter_specs!(@rows [] [] [] $($rows)*);
    };
}

filter_specs! {
    // Pixel filters.
    "grayscale" => Grayscale;
    "invert" => Invert;
    "blur"(sigma: BLUR_SIGMA) => Blur(sigma);
    "brighten"(amount: CHANNEL_OFFSET) => Brighten(amount);
    "contrast"(amount: PERCENT) => Contrast(amount);
    "gamma"(divisor: GAMMA) => Gamma(divisor);
    "exposure"(stops: EXPOSURE_STOPS) => Exposure(stops);
    "highlights_shadows"(highlights: PERCENT, shadows: PERCENT)
        => HighlightsShadows(highlights, shadows);
    "huerotate"(degrees: HUE_DEGREES) => HueRotate(degrees);
    "sharpen"(amount: SHARPEN_AMOUNT) => Sharpen(amount);
    "saturation"(factor: SATURATION_FACTOR) => Saturation(factor);
    "vibrance"(amount: PERCENT) => Vibrance(amount);
    "rgb"(red: CHANNEL_OFFSET, green: CHANNEL_OFFSET, blue: CHANNEL_OFFSET)
        => Rgb(red, green, blue);
    "curve"(custom read_curve) else InvalidCurve
        => Curve(channel, points) spelled curve_args(*channel, points);
    "convolution"(custom read_convolution) else InvalidConvolution
        => Convolution(values, normalize) spelled convolution_args(values, *normalize);
    "noise"(amount: NOISE_PERCENT) else InvalidNoise => Noise(amount);
    "equalize" => Equalize;
    "enhance" => Enhance;
    "rotate"(degrees: ROTATE_ANGLE) => Rotate(degrees);
    shorthand "trim" => Trim(DEFAULT_TRIM_TOLERANCE);
    "trim"(tolerance: TRIM_TOLERANCE) => Trim(tolerance);
    "trim"(
        tolerance: TRIM_TOLERANCE,
        left: TRIM_PIXELS,
        top: TRIM_PIXELS,
        right: TRIM_PIXELS,
        bottom: TRIM_PIXELS
    ) => TrimSides(tolerance, [left, top, right, bottom]);
    "sepia" => Sepia;
    "duotone"(shadows: COLOR, highlights: COLOR) => Duotone(shadows, highlights);
    "monochrome"(color: COLOR) => Monochrome(color);
    "pixelate"(block: PIXELATE_BLOCK) => Pixelate(block);
    "flip"(direction: FLIP_DIRECTIONS) => Flip(direction);
    "blur_region"(x: REGION_ORIGIN, y: REGION_ORIGIN, w: REGION_EXTENT, h: REGION_EXTENT)
        => BlurRegion { x, y, w, h };
    "pixelate_region"(x: REGION_ORIGIN, y: REGION_ORIGIN, w: REGION_EXTENT, h: REGION_EXTENT)
        => PixelateRegion { x, y, w, h };
    "remove_region"(x: REGION_ORIGIN, y: REGION_ORIGIN, w: REGION_EXTENT, h: REGION_EXTENT)
        => RemoveRegion { x, y, w, h };
    shorthand "radius"(radius: CORNER_RADIUS) => Radius([radius; 4]);
    "radius"(
        top_left: CORNER_RADIUS,
        top_right: CORNER_RADIUS,
        bottom_right: CORNER_RADIUS,
        bottom_left: CORNER_RADIUS
    ) => Radius([top_left, top_right, bottom_right, bottom_left]);
    "border"(width: BORDER_WIDTH, color: COLOR) => Border(width, color);
    "mask"(shape: MASK_SHAPES) => Mask(shape);

    // Encoding and request directives.
    "format"(custom |p| parse_format_spec(p).map(Filter::Format))
        => Format(spec) spelled format_spec_args(spec);
    "quality"(custom |p| parse_quality(p).map(Filter::Quality))
        => Quality(quality) spelled quality_args(quality);
    "palette"(colors: PALETTE_COLORS) => Palette(colors);
    "max_bytes"(budget: BYTE_BUDGET) => MaxBytes(budget);
    "fill"(color: COLOR) => Fill(color);
    "no_upscale" => NoUpscale;
    "dpr"(factor: DPR_FACTOR) => Dpr(factor);

    // Compositing.
    "watermark"(custom read_watermark)
        => Watermark { url, x, y, alpha } spelled watermark_args(url, *x, *y, *alpha);
    "overlay"(custom read_overlay)
        => Overlay { url, x, y, alpha, blend, width }
        spelled overlay_args(url, *x, *y, *alpha, *blend, *width);
    "text"(custom read_text)
        => Text { text, size, color, anchor } spelled text_args(text, *size, color, *anchor);

    // AI filters.
    "removebg" => RemoveBg;
    "studio"(custom |p| parse_studio(p).map(Filter::Studio))
        => Studio(spec) spelled studio_args(spec);
    "keep_profile" => KeepProfile;
    "blurfaces" => BlurFaces;
    "pixelatefaces" => PixelateFaces;
    "blurclass"(class: CLASS_NAME) => BlurClass(class);
    "pixelateclass"(class: CLASS_NAME) => PixelateClass(class);
    "recolorclass"(class: CLASS_NAME, color: COLOR) => RecolorClass(class, color);
    "facepad"(pad: FACE_PAD) => FacePad(pad);
    "upscale"(factor: UPSCALE_FACTOR) => Upscale(factor);
}

/// Read one filter of a `filters:` segment: its `name`, and
/// the text between its parentheses (`params`, None when
/// there are none or they are empty). Errors carry the whole
/// segment, `seg`.
pub(super) fn read_filter(
    name: &str,
    params: Option<&str>,
    seg: &str,
) -> Result<Filter, ParseError> {
    let args: Vec<&str> = params.map_or_else(Vec::new, |p| p.split(',').collect());
    build(name, params, &args).ok_or_else(|| {
        let error = match params {
            Some(_) => argument_error(name),
            None => ParseError::InvalidFilters,
        };
        error(seg.to_string())
    })
}

/// `curve(<channel>,<x>,<y>,...)`: the channel, then a flat
/// list of control points, every coordinate 0-255 and the x
/// values strictly increasing.
fn read_curve(p: &str) -> Option<Filter> {
    let parts: Vec<&str> = p.split(',').collect();
    let channel = CurveChannel::parse(parts[0])?;
    let coords = &parts[1..];
    if !coords.len().is_multiple_of(2) || !CURVE_POINTS.contains(&(coords.len() / 2)) {
        return None;
    }
    let mut points: Vec<(u8, u8)> = Vec::with_capacity(coords.len() / 2);
    for pair in coords.as_chunks::<2>().0 {
        let x: u8 = pair[0].parse().ok()?;
        let y: u8 = pair[1].parse().ok()?;
        if points.last().is_some_and(|last| x <= last.0) {
            return None;
        }
        points.push((x, y));
    }
    Some(Filter::Curve(channel, points))
}

/// `convolution(<v1>,...,<vN>,<normalize>)`: the kernel
/// values in row-major order, then the normalize flag, 0 or 1.
fn read_convolution(p: &str) -> Option<Filter> {
    let parts: Vec<&str> = p.split(',').collect();
    let (normalize, values) = parts.split_last()?;
    if !KERNEL_SIZES.contains(&values.len()) {
        return None;
    }
    let normalize = match *normalize {
        "0" => false,
        "1" => true,
        _ => return None,
    };
    let values = values
        .iter()
        .map(|v| parse_kernel_value(v).ok())
        .collect::<Option<Vec<f64>>>()?;
    Some(Filter::Convolution(values, normalize))
}

/// `watermark(<url>,<x>,<y>,<alpha>)`. The URL lives inside a
/// path segment, so it arrives percent-encoded; it may itself
/// contain commas, so the split runs from the right, where
/// the numeric fields are.
fn read_watermark(p: &str) -> Option<Filter> {
    let parts: Vec<&str> = p.rsplitn(4, ',').collect();
    let [alpha, y, x, url] = parts[..] else {
        return None;
    };
    let alpha = OPACITY.read(alpha)?;
    let y = OFFSET.read(y)?;
    let x = OFFSET.read(x)?;
    let url = decode_arg(url)?;
    if url.is_empty() {
        return None;
    }
    Some(Filter::Watermark { url, x, y, alpha })
}

/// `overlay(<url>,<x>,<y>,<alpha>,<blend>[,<width>])`. Like
/// watermark, the split runs from the right: the URL may
/// contain commas, the trailing fields may not.
fn read_overlay(p: &str) -> Option<Filter> {
    let parts: Vec<&str> = p.rsplitn(6, ',').collect();
    let (width, rest) = match parts.len() {
        6 => (Some(OVERLAY_WIDTH.read(parts[0])?), &parts[1..]),
        5 => (None, &parts[..]),
        _ => return None,
    };
    let [blend, alpha, y, x, url] = rest[..] else {
        return None;
    };
    let blend = BLEND_MODES.read(blend)?;
    let alpha = OPACITY.read(alpha)?;
    let y = OFFSET.read(y)?;
    let x = OFFSET.read(x)?;
    let url = decode_arg(url)?;
    if url.is_empty() {
        return None;
    }
    Some(Filter::Overlay {
        url,
        x,
        y,
        alpha,
        blend,
        width,
    })
}

/// `text(<text>,<size>,<color>,<anchor>)`. The text is
/// percent-encoded and may contain commas, so the split runs
/// from the right, where the fixed fields are. The decoded
/// label holds at most [`MAX_TEXT_GLYPHS`] characters, and
/// their count times the size is at most [`MAX_TEXT_EXTENT`]
/// (issue #606).
fn read_text(p: &str) -> Option<Filter> {
    let parts: Vec<&str> = p.rsplitn(4, ',').collect();
    let [anchor, color, size, text] = parts[..] else {
        return None;
    };
    let anchor = Anchor::parse(anchor)?;
    let color = parse_text_color(color)?;
    let size = TEXT_SIZE.read(size)?;
    let text = decode_arg(text)?;
    let glyphs = text.chars().count();
    if glyphs == 0 || glyphs > MAX_TEXT_GLYPHS || glyphs as u64 * size as u64 > MAX_TEXT_EXTENT {
        return None;
    }
    Some(Filter::Text {
        text,
        size,
        color,
        anchor,
    })
}

/// Parse the filter list after `filters:`: split it into
/// filters, then read each through the filter table in
/// `url/filters.rs`.
fn parse_filters(rest: &str, seg: &str) -> Result<Operation, ParseError> {
    let invalid = || ParseError::InvalidFilters(seg.to_string());
    if rest.is_empty() {
        return Err(invalid());
    }
    let mut filters = Vec::new();
    for part in split_filter_chain(rest) {
        let (name, params) = match part.find('(') {
            Some(i) => {
                let (n, p) = part.split_at(i);
                let inner = p
                    .strip_prefix('(')
                    .and_then(|s| s.strip_suffix(')'))
                    .ok_or_else(invalid)?;
                // Empty parens mean no parameters: `grayscale()`.
                (n, if inner.is_empty() { None } else { Some(inner) })
            }
            None => (part, None),
        };
        filters.push(read_filter(name, params, seg)?);
    }
    Ok(Operation::Filters(filters))
}

/// Parse the operation segments off the front of the remainder,
/// also extracting the request's metadata segments: an
/// `exp:<unix-seconds>` expiry and a `maxage:<seconds>` cache
/// lifetime may appear anywhere among the operations. Both are
/// metadata, not operations, and both sit inside the signed
/// remainder, so the HMAC covers them. The last one of each
/// kind wins.
pub fn parse_operations_with_expiry(input: &str) -> Result<ParsedRemainder<'_>, ParseError> {
    let mut operations = Vec::new();
    let mut expiry = None;
    let mut max_age = None;
    let mut rest = input;
    loop {
        let (seg, remaining) = match rest.split_once('/') {
            Some((s, r)) => (s, r),
            None => (rest, ""),
        };
        if let Some(ts) = seg.strip_prefix("exp:") {
            expiry = Some(parse_expiry(ts, seg)?);
            rest = remaining;
            continue;
        }
        if let Some(secs) = seg.strip_prefix("maxage:") {
            if secs.is_empty() || !secs.bytes().all(|b| b.is_ascii_digit()) {
                return Err(ParseError::InvalidMaxAge(seg.to_string()));
            }
            let value = secs
                .parse::<u64>()
                .map_err(|_| ParseError::InvalidMaxAge(seg.to_string()))?;
            if value > u64::from(MAX_CACHE_AGE) {
                return Err(ParseError::InvalidMaxAge(seg.to_string()));
            }
            max_age = Some(value as u32);
            rest = remaining;
            continue;
        }
        match parse_operation(seg) {
            Some(Ok(op)) => {
                operations.push(op);
                rest = remaining;
            }
            Some(Err(e)) => return Err(e),
            None => {
                if seg.is_empty() {
                    return Err(ParseError::MissingSourceUrl);
                }
                return Ok((operations, rest, expiry, max_age));
            }
        }
    }
}

fn operation_text(op: &Operation) -> String {
    match op {
        Operation::Size(size) => size_text(size),
        Operation::Crop { size, align } => match align {
            Align::Objects { classes, scale } => format!(
                "crop:{}:objects({}):{scale}",
                size_text(size),
                classes
                    .iter()
                    .map(|c| encode_arg(c))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            _ => match align_text(align) {
                Some(name) => format!("crop:{}:{name}", size_text(size)),
                None => format!("crop:{}", size_text(size)),
            },
        },
        Operation::FitIn { size, adaptive } => {
            if *adaptive {
                format!("fit-in:{}:adaptive", size_text(size))
            } else {
                format!("fit-in:{}", size_text(size))
            }
        }
        Operation::Pad { size, fill } => match fill {
            PadFill::Directive => format!("pad:{}", size_text(size)),
            PadFill::Blur => format!("pad:{}:blur", size_text(size)),
            PadFill::Color(c) => format!("pad:{}:{}", size_text(size), hex3(*c)),
        },
        Operation::CropRect {
            left,
            top,
            right,
            bottom,
        } => format!("crop-rect:{left}x{top}:{right}x{bottom}"),
        Operation::FocalRect {
            left,
            top,
            right,
            bottom,
        } => format!("focal:{left}x{top}:{right}x{bottom}"),
        Operation::FocalPoint { x, y, zoom } => format!("focalpoint:{x}x{y}:{zoom}"),
        Operation::DebugFocal => "debug:focal".to_string(),
        Operation::Frame(selection) => frame_text(selection),
        Operation::Page(page) => format!("page:{page}"),
        Operation::Filters(_) => unreachable!("filters are merged by canonical_form"),
    }
}

/// The canonical form of a parsed request: every operation
/// re-serialized in its one canonical spelling, adjacent
/// `filters:` segments merged into one, plus the source.
/// Requests that differ only in spelling (format aliases,
/// hex case, leading zeros, size forms, split filter
/// segments) share this form, and therefore a cache entry
/// (review Q1). Operation and filter order are part of
/// the form: order changes pixels, so it is meaning, not
/// spelling.
pub fn canonical_form(ops: &[Operation], source: &str) -> String {
    // Written into one buffer as it goes (issue #640): the
    // form is built for every request's cache key, hits
    // included.
    let mut out = String::with_capacity(source.len() + 24 * ops.len() + 16);
    // Whether a segment has been written, and whether the
    // last one is a `filters:` segment, which the next
    // filter joins.
    let mut any = false;
    let mut in_filters = false;
    for op in ops {
        match op {
            Operation::Filters(list) => {
                for filter in list {
                    if in_filters {
                        out.push(':');
                    } else {
                        if any {
                            out.push('/');
                        }
                        out.push_str("filters:");
                        any = true;
                        in_filters = true;
                    }
                    out.push_str(&filter_text(filter));
                }
            }
            other => {
                if any {
                    out.push('/');
                }
                out.push_str(&operation_text(other));
                any = true;
                in_filters = false;
            }
        }
    }
    if any {
        out.push('/');
    }
    out.push_str(source);
    out
}

/// The cache key for a remainder: its canonical form, which
/// drops every `exp:` and `maxage:` segment (audit P-2). Both
/// are per-request metadata enforced or applied outside the
/// stored entry (expiry before lookup, the cache lifetime on
/// the response), so neither may fragment the cache or
/// shield entries from purges. A remainder that does not
/// parse passes through unchanged: the parse is what finds
/// where the source starts, so without it no segment can be
/// told apart from source text, and no stored entry is keyed
/// on an unparseable remainder for the text to reach.
pub fn canonical_key_form(remainder: &str) -> String {
    match parse_operations_with_expiry(remainder) {
        Ok((ops, source, _, _)) => canonical_form(&ops, source),
        Err(_) => remainder.to_string(),
    }
}

/// Merge server-wide defaults under a request's operations
/// (issue #145): the defaults come first, the request's own
/// operations follow, and for the singleton directives
/// (format, quality, fill, dpr, max_bytes) the request wins
/// by the default being dropped, not merely overridden
/// downstream. Dropping is what makes a bare request under
/// defaults and the same request spelled out in full parse
/// to the same operations, and so share a cache entry.
/// Everything else (sizes, filters, watermarks) applies from
/// both, in order.
pub fn merge_default_ops(defaults: &[Operation], url_ops: Vec<Operation>) -> Vec<Operation> {
    if defaults.is_empty() {
        return url_ops;
    }
    let directives = analyze(&url_ops);
    let url_has_max_bytes = url_ops.iter().any(|op| {
        matches!(
            op,
            Operation::Filters(filters)
                if filters.iter().any(|f| matches!(f, Filter::MaxBytes(_)))
        )
    });
    let keep_default_filter = |f: &Filter| -> bool {
        match f {
            Filter::Format(_) => directives.explicit_format.is_none(),
            Filter::Quality(_) => !directives.quality_directive,
            Filter::Fill(_) => directives.fill.is_none(),
            Filter::Dpr(_) => directives.dpr.is_none(),
            Filter::MaxBytes(_) => !url_has_max_bytes,
            _ => true,
        }
    };
    let mut merged: Vec<Operation> = Vec::with_capacity(defaults.len() + url_ops.len());
    for op in defaults {
        match op {
            Operation::Filters(filters) => {
                let kept: Vec<Filter> = filters
                    .iter()
                    .filter(|f| keep_default_filter(f))
                    .cloned()
                    .collect();
                if !kept.is_empty() {
                    merged.push(Operation::Filters(kept));
                }
            }
            other => merged.push(other.clone()),
        }
    }
    merged.extend(url_ops);
    merged
}

/// Analyze a parsed request's operations in a single walk.
pub(super) fn analyze(ops: &[Operation]) -> RequestDirectives {
    let mut d = RequestDirectives {
        explicit_format: None,
        format_options: FormatOptions::default(),
        no_upscale: false,
        fill: None,
        removebg: false,
        studio: false,
        remove_region: false,
        shape_filter: None,
        upscale: false,
        smart_crop: false,
        dpr: None,
        has_size_operation: false,
        keep_profile: false,
        quality_directive: false,
        text: false,
        needs_face_model: false,
        classes: Vec::new(),
        objects_crop: false,
        object_classes: Vec::new(),
        face_padding: 0.5,
        watermark_urls: Vec::new(),
        debug_focal: false,
        frame: None,
        page: None,
    };
    for op in ops {
        match op {
            Operation::Size(_)
            | Operation::Crop { .. }
            | Operation::FitIn { .. }
            | Operation::Pad { .. }
            | Operation::CropRect { .. } => d.has_size_operation = true,
            Operation::DebugFocal => d.debug_focal = true,
            Operation::Frame(selection) => d.frame = Some(*selection),
            Operation::Page(page) => d.page = Some(*page),
            _ => {}
        }
        if let Operation::Crop { align, .. } = op {
            match align {
                Align::Smart => d.smart_crop = true,
                Align::Face => d.needs_face_model = true,
                Align::Objects { classes, .. } => {
                    d.objects_crop = true;
                    d.object_classes.extend(classes.iter().cloned());
                }
                _ => {}
            }
        }
        if let Operation::Filters(filters) = op {
            for f in filters {
                match f {
                    Filter::Format(spec) => {
                        d.explicit_format = Some(spec.format);
                        d.format_options = spec.options;
                    }
                    Filter::NoUpscale => d.no_upscale = true,
                    Filter::Fill(color) => d.fill = Some(*color),
                    Filter::RemoveBg => d.removebg = true,
                    Filter::Studio(_) => d.studio = true,
                    Filter::RemoveRegion { .. } => d.remove_region = true,
                    Filter::Radius(_) if d.shape_filter.is_none() => {
                        d.shape_filter = Some("radius")
                    }
                    Filter::Mask(_) if d.shape_filter.is_none() => d.shape_filter = Some("mask"),
                    Filter::Upscale(_) => d.upscale = true,
                    Filter::Dpr(factor) => d.dpr = Some(*factor),
                    Filter::KeepProfile => d.keep_profile = true,
                    Filter::Quality(_) => d.quality_directive = true,
                    Filter::Text { .. } => d.text = true,
                    Filter::BlurFaces | Filter::PixelateFaces => d.needs_face_model = true,
                    Filter::BlurClass(name)
                    | Filter::PixelateClass(name)
                    | Filter::RecolorClass(name, _) => d.classes.push(name.clone()),
                    Filter::FacePad(p) => d.face_padding = *p,
                    Filter::Watermark { url, .. } | Filter::Overlay { url, .. }
                        if !d.watermark_urls.contains(url) =>
                    {
                        d.watermark_urls.push(url.clone());
                    }
                    _ => {}
                }
            }
        }
    }
    d
}
