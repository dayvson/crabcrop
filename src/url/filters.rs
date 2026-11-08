//! The filter table (issue #483): every filter a `filters:`
//! segment accepts, declared once. A row names the filter,
//! the kind of each argument (which carries its range), and
//! the [`Filter`] the arguments build. The parser
//! ([`read_filter`]) and the canonical spelling
//! ([`filter_text`]) are both generated from the rows, so
//! they cannot drift apart, and adding a filter is adding a
//! row: a [`Filter`] variant without one fails to compile.
//!
//! A row takes one of four shapes:
//!
//! - `"grayscale" => Grayscale;` takes no arguments.
//! - `"blur"(sigma: BLUR_SIGMA) => Blur(sigma);` takes a
//!   comma-separated list, one kind per argument. The kind
//!   reads the argument and writes it back canonically; the
//!   names bind the variant's fields, which the right-hand
//!   side both builds and matches.
//! - `"curve"(custom read_curve) => Curve(channel, points)
//!   spelled curve_args(*channel, points);` has a
//!   sub-grammar of its own: the function reads the whole
//!   argument list, and the expression after `spelled`
//!   writes it.
//! - `shorthand "trim" => Trim(DEFAULT_TRIM_TOLERANCE);` is
//!   another spelling of a filter that a full row writes:
//!   it is read, never written.
//!
//! One name may have several rows, told apart by argument
//! count (`trim`, `trim(20)`, `trim(20,5,0,5,0)`). A bad
//! argument list answers `invalid_filters`, unless the row
//! names its own code with `else`; an unknown name, or a
//! bare name whose filter needs arguments, always answers
//! `invalid_filters`.

use std::fmt::Display;
use std::ops::RangeInclusive;
use std::str::FromStr;

use super::{
    Anchor, BlendMode, CurveChannel, Filter, FlipDirection, FormatSpec, ImageFormat,
    MAX_BLUR_SIGMA, MAX_DIMENSION, MAX_EXPOSURE, MAX_GAMMA, MaskShape, ParseError, Quality,
    QualityValue, StudioMode, StudioSpec, anchor_text, encode_arg, format_angle,
    format_kernel_value, format_text, hex3, parse_angle, parse_class_name, parse_format_spec,
    parse_hex6, parse_kernel_value, parse_quality, parse_studio, parse_text_color, percent_decode,
    quality_value_text,
};

/// How one argument is read from the URL and written in the
/// canonical form.
trait Kind {
    type Value;
    /// The value `text` spells, or None when it is not one
    /// this kind accepts.
    fn read(&self, text: &str) -> Option<Self::Value>;
    /// The one canonical spelling of `value`.
    fn spell(&self, value: &Self::Value) -> String;
}

/// A number, read by its own type's `FromStr` and accepted
/// inside `min..=max` (with `min` itself refused when the
/// bound is open), written by its `Display`, the shortest
/// spelling that reads back to the same value. Reading in
/// the field's own type keeps each filter's spellings
/// exact: an unsigned argument refuses a minus sign, `-0`
/// included, and a float bound refuses NaN and the
/// infinities.
struct Number<T> {
    min: T,
    max: T,
    open: bool,
}

impl<T> Number<T> {
    /// `min` to `max`, both included.
    const fn closed(min: T, max: T) -> Self {
        Number {
            min,
            max,
            open: false,
        }
    }

    /// Above `min`, up to and including `max`.
    const fn above(min: T, max: T) -> Self {
        Number {
            min,
            max,
            open: true,
        }
    }
}

impl<T: FromStr + PartialOrd + Display> Kind for Number<T> {
    type Value = T;

    fn read(&self, text: &str) -> Option<T> {
        let value: T = text.parse().ok()?;
        let above_min = if self.open {
            value > self.min
        } else {
            value >= self.min
        };
        (above_min && value <= self.max).then_some(value)
    }

    fn spell(&self, value: &T) -> String {
        value.to_string()
    }
}

/// A number that must be one of a fixed set, read and
/// written like [`Number`].
struct OneOf<T: 'static>(&'static [T]);

impl<T: FromStr + PartialEq + Display> Kind for OneOf<T> {
    type Value = T;

    fn read(&self, text: &str) -> Option<T> {
        let value: T = text.parse().ok()?;
        self.0.contains(&value).then_some(value)
    }

    fn spell(&self, value: &T) -> String {
        value.to_string()
    }
}

/// A keyword naming one value of an enum.
struct Keyword<T: 'static>(&'static [(&'static str, T)]);

impl<T: Copy + PartialEq> Kind for Keyword<T> {
    type Value = T;

    fn read(&self, text: &str) -> Option<T> {
        self.0
            .iter()
            .find(|(name, _)| *name == text)
            .map(|&(_, value)| value)
    }

    fn spell(&self, value: &T) -> String {
        // Only `read` builds these values, so every one has
        // its keyword in the table.
        let (name, _) = self
            .0
            .iter()
            .find(|(_, v)| v == value)
            .expect("a keyword value comes from its own table");
        name.to_string()
    }
}

/// Six hex digits, RGB, in either case; written lowercase.
struct Hex;

impl Kind for Hex {
    type Value = [u8; 3];

    fn read(&self, text: &str) -> Option<[u8; 3]> {
        parse_hex6(text).ok()
    }

    fn spell(&self, value: &[u8; 3]) -> String {
        hex3(*value)
    }
}

/// A detection class name, percent-decoded on the way in and
/// encoded on the way out (see `parse_class_name`).
struct ClassName;

impl Kind for ClassName {
    type Value = String;

    fn read(&self, text: &str) -> Option<String> {
        parse_class_name(&percent_decode(text)?)
    }

    fn spell(&self, value: &String) -> String {
        encode_arg(value)
    }
}

/// A rotate angle, in degrees (see `parse_angle`).
struct Angle;

impl Kind for Angle {
    type Value = f32;

    fn read(&self, text: &str) -> Option<f32> {
        parse_angle(text).ok()
    }

    fn spell(&self, value: &f32) -> String {
        format_angle(*value)
    }
}

/// `blur` sigma, positive and normal, up to
/// [`MAX_BLUR_SIGMA`]: the smallest positive normal float is
/// the lower bound, which refuses zero and the subnormals
/// that panic the image crate's Gaussian setup.
const BLUR_SIGMA: Number<f32> = Number::closed(f32::MIN_POSITIVE, MAX_BLUR_SIGMA);
/// A level offset added to a channel (`brighten`, `rgb`).
const CHANNEL_OFFSET: Number<i32> = Number::closed(-255, 255);
/// The adjustment family's percent (`contrast`, `vibrance`,
/// `highlights_shadows`): 0 is the identity.
const PERCENT: Number<f32> = Number::closed(-100.0, 100.0);
/// `gamma` divisor.
const GAMMA: Number<f32> = Number::above(0.0, MAX_GAMMA);
/// `exposure`, in stops.
const EXPOSURE_STOPS: Number<f32> = Number::closed(-MAX_EXPOSURE, MAX_EXPOSURE);
/// `huerotate` degrees: any integer.
const HUE_DEGREES: Number<i32> = Number::closed(i32::MIN, i32::MAX);
/// `sharpen` strength: positive and finite.
const SHARPEN_AMOUNT: Number<f32> = Number::above(0.0, f32::MAX);
/// `saturation` factor: 0 is grayscale, 1 the identity.
const SATURATION_FACTOR: Number<f32> = Number::closed(0.0, 255.0);
/// The control points of one `curve`.
const CURVE_POINTS: RangeInclusive<usize> = 2..=16;
/// The entries of a `convolution` kernel: 3x3 or 5x5.
const KERNEL_SIZES: [usize; 2] = [9, 25];
/// `noise` strength, percent of full scale.
const NOISE_PERCENT: Number<u8> = Number::closed(0, 100);
/// `rotate` angle.
const ROTATE_ANGLE: Angle = Angle;
/// `trim` tolerance: any channel difference.
const TRIM_TOLERANCE: Number<u8> = Number::closed(0, u8::MAX);
/// The tolerance a bare `trim` uses.
const DEFAULT_TRIM_TOLERANCE: u8 = 10;
/// Pixels `trim` removes from one side; the engine clamps
/// them so a pixel survives on each axis.
const TRIM_PIXELS: Number<u32> = Number::closed(0, u32::MAX);
/// An RGB color.
const COLOR: Hex = Hex;
/// `pixelate` block size.
const PIXELATE_BLOCK: Number<u32> = Number::closed(1, MAX_DIMENSION);
/// `flip` directions.
const FLIP_DIRECTIONS: Keyword<FlipDirection> = Keyword(&[
    ("h", FlipDirection::Horizontal),
    ("v", FlipDirection::Vertical),
]);
/// The corner of a redaction region.
const REGION_ORIGIN: Number<u32> = Number::closed(0, MAX_DIMENSION);
/// The width or height of a redaction region.
const REGION_EXTENT: Number<u32> = Number::closed(1, MAX_DIMENSION);
/// One `radius` corner.
const CORNER_RADIUS: Number<u32> = Number::closed(0, MAX_DIMENSION);
/// `border` width.
const BORDER_WIDTH: Number<u32> = Number::closed(0, MAX_DIMENSION);
/// `mask` shapes.
const MASK_SHAPES: Keyword<MaskShape> = Keyword(&[
    ("circle", MaskShape::Circle),
    ("ellipse", MaskShape::Ellipse),
]);
/// `palette` colors.
const PALETTE_COLORS: Number<u16> = Number::closed(2, 256);
/// `max_bytes` budget.
const BYTE_BUDGET: Number<u64> = Number::closed(1, u64::MAX);
/// `dpr` factors.
const DPR_FACTOR: OneOf<f32> = OneOf(&[1.0, 1.5, 2.0, 3.0]);
/// A watermark or overlay position; negative values anchor
/// from the right and bottom edges.
const OFFSET: Number<i32> = Number::closed(i32::MIN, i32::MAX);
/// A watermark or overlay opacity.
const OPACITY: Number<u8> = Number::closed(0, 100);
/// `overlay` blend modes.
const BLEND_MODES: Keyword<BlendMode> = Keyword(&[
    ("normal", BlendMode::Normal),
    ("multiply", BlendMode::Multiply),
    ("screen", BlendMode::Screen),
    ("overlay", BlendMode::Overlay),
]);
/// `overlay` width, percent of the base image's width.
const OVERLAY_WIDTH: Number<u32> = Number::closed(1, 400);
/// `text` size, in pixels.
const TEXT_SIZE: Number<u32> = Number::closed(1, 1000);
/// A detection class (`blurclass`, `pixelateclass`,
/// `recolorclass`).
const CLASS_NAME: ClassName = ClassName;
/// `facepad` fraction of the face size.
const FACE_PAD: Number<f32> = Number::closed(0.0, 3.0);
/// `upscale` factors.
const UPSCALE_FACTOR: OneOf<u32> = OneOf(&[2, 4]);

/// Generate the reader, the speller, and the argument error
/// codes from the rows (see the module docs). A recursive
/// macro reads one row at a time, appending to four lists:
/// the reader's match arms, the speller's match arms, the
/// error codes, and the rows the tests walk.
macro_rules! filter_specs {
    (@rows [$($read:tt)*] [$($spell:tt)*] [$($error:tt)*] [$($spec:tt)*]) => {
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

        /// The rows, for the tests that walk the table.
        #[cfg(test)]
        const FILTER_SPECS: &[tests::Spec] = &[$($spec)*];
    };

    // `else Code`: the row's argument errors carry this code.
    (@rows [$($read:tt)*] [$($spell:tt)*] [$($error:tt)*] [$($spec:tt)*]
        $name:literal $args:tt else $code:ident => $($rest:tt)*) => {
        filter_specs!(@rows
            [$($read)*]
            [$($spell)*]
            [$($error)* $name => ParseError::$code,]
            [$($spec)*]
            $name $args => $($rest)*);
    };

    // A bare shorthand: read, never written.
    (@rows [$($read:tt)*] [$($spell:tt)*] [$($error:tt)*] [$($spec:tt)*]
        shorthand $name:literal => $variant:ident $fields:tt; $($rest:tt)*) => {
        filter_specs!(@rows
            [$($read)* ($name, None, _) => Some(Filter::$variant $fields),]
            [$($spell)*]
            [$($error)*]
            [$($spec)* tests::Spec { name: $name, args: Some(&[]), shorthand: true },]
            $($rest)*);
    };

    // A shorthand with arguments: read, never written.
    (@rows [$($read:tt)*] [$($spell:tt)*] [$($error:tt)*] [$($spec:tt)*]
        shorthand $name:literal ($($arg:ident: $kind:ident),+) => $variant:ident $fields:tt;
        $($rest:tt)*) => {
        filter_specs!(@rows
            [$($read)* ($name, Some(_), [$($arg),+]) => {
                $(let $arg = $kind.read($arg)?;)+
                Some(Filter::$variant $fields)
            }]
            [$($spell)*]
            [$($error)*]
            [$($spec)* tests::Spec { name: $name, args: Some(&[$(&$kind),+]), shorthand: true },]
            $($rest)*);
    };

    // A sub-grammar of its own.
    (@rows [$($read:tt)*] [$($spell:tt)*] [$($error:tt)*] [$($spec:tt)*]
        $name:literal (custom $parse:expr) => $variant:ident $fields:tt spelled $text:expr;
        $($rest:tt)*) => {
        filter_specs!(@rows
            [$($read)* ($name, Some(params), _) => {
                let read: fn(&str) -> Option<Filter> = $parse;
                read(params)
            }]
            [$($spell)* Filter::$variant $fields => format!(concat!($name, "({})"), $text),]
            [$($error)*]
            [$($spec)* tests::Spec { name: $name, args: None, shorthand: false },]
            $($rest)*);
    };

    // No arguments.
    (@rows [$($read:tt)*] [$($spell:tt)*] [$($error:tt)*] [$($spec:tt)*]
        $name:literal => $variant:ident; $($rest:tt)*) => {
        filter_specs!(@rows
            [$($read)* ($name, None, _) => Some(Filter::$variant),]
            [$($spell)* Filter::$variant => String::from($name),]
            [$($error)*]
            [$($spec)* tests::Spec { name: $name, args: Some(&[]), shorthand: false },]
            $($rest)*);
    };

    // One kind per argument.
    (@rows [$($read:tt)*] [$($spell:tt)*] [$($error:tt)*] [$($spec:tt)*]
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
            [$($spec)* tests::Spec { name: $name, args: Some(&[$(&$kind),+]), shorthand: false },]
            $($rest)*);
    };

    ($($rows:tt)*) => {
        filter_specs!(@rows [] [] [] [] $($rows)*);
    };
}

// Grouped as docs/filters.md lists them.
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

fn curve_args(channel: CurveChannel, points: &[(u8, u8)]) -> String {
    let mut text = channel.name().to_string();
    for (x, y) in points {
        text.push_str(&format!(",{x},{y}"));
    }
    text
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

fn convolution_args(values: &[f64], normalize: bool) -> String {
    let mut text = values
        .iter()
        .map(|v| format_kernel_value(*v))
        .collect::<Vec<_>>()
        .join(",");
    text.push_str(if normalize { ",1" } else { ",0" });
    text
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
    let url = percent_decode(url)?;
    if url.is_empty() {
        return None;
    }
    Some(Filter::Watermark { url, x, y, alpha })
}

fn watermark_args(url: &str, x: i32, y: i32, alpha: u8) -> String {
    format!("{},{x},{y},{alpha}", encode_arg(url))
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
    let url = percent_decode(url)?;
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

fn overlay_args(
    url: &str,
    x: i32,
    y: i32,
    alpha: u8,
    blend: BlendMode,
    width: Option<u32>,
) -> String {
    let blend = BLEND_MODES.spell(&blend);
    let width = width.map(|w| format!(",{w}")).unwrap_or_default();
    format!("{},{x},{y},{alpha},{blend}{width}", encode_arg(url))
}

/// `text(<text>,<size>,<color>,<anchor>)`. The text is
/// percent-encoded and may contain commas, so the split runs
/// from the right, where the fixed fields are.
fn read_text(p: &str) -> Option<Filter> {
    let parts: Vec<&str> = p.rsplitn(4, ',').collect();
    let [anchor, color, size, text] = parts[..] else {
        return None;
    };
    let anchor = Anchor::parse(anchor)?;
    let color = parse_text_color(color)?;
    let size = TEXT_SIZE.read(size)?;
    let text = percent_decode(text)?;
    if text.is_empty() {
        return None;
    }
    Some(Filter::Text {
        text,
        size,
        color,
        anchor,
    })
}

fn text_args(text: &str, size: u32, color: &[u8; 4], anchor: Anchor) -> String {
    format!(
        "{},{size},{:02x}{:02x}{:02x}{:02x},{}",
        encode_arg(text),
        color[0],
        color[1],
        color[2],
        color[3],
        anchor_text(anchor)
    )
}

/// Only the arguments that differ from the defaults are
/// spelled out, in grammar order (bg, shadow, pad), so
/// `studio(product)` and `studio(product,pad:10)` share one
/// canonical form. Scene spells only its pad, the one
/// argument it takes.
fn studio_args(spec: &StudioSpec) -> String {
    let default = StudioSpec::default();
    let mut text = spec.mode.name().to_string();
    if spec.mode == StudioMode::Product {
        if let Some(bg) = spec.bg.text() {
            text.push_str(&format!(",bg:{bg}"));
        }
        if spec.shadow != default.shadow {
            text.push_str(&format!(",shadow:{}", spec.shadow.name()));
        }
    }
    if spec.pad != default.pad {
        text.push_str(&format!(",pad:{}", spec.pad));
    }
    text
}

/// The fallback first, then one entry per format in
/// format-name order, so the same set spells the same way
/// however it was written.
fn quality_args(quality: &Quality) -> String {
    let entries = match quality {
        Quality::Auto => return "auto".to_string(),
        Quality::Value(v) => return v.to_string(),
        Quality::PerFormat(entries) => entries,
    };
    let mut parts: Vec<String> = Vec::with_capacity(entries.len());
    for (key, value) in entries {
        if key.is_none() {
            parts.push(quality_value_text(*value));
        }
    }
    let mut named: Vec<(ImageFormat, QualityValue)> = entries
        .iter()
        .filter_map(|(k, v)| k.map(|f| (f, *v)))
        .collect();
    named.sort_by_key(|(f, _)| format_text(*f));
    for (format, value) in named {
        parts.push(format!(
            "{}:{}",
            format_text(format),
            quality_value_text(value)
        ));
    }
    parts.join(",")
}

/// The format, then its options in a fixed order per format,
/// whatever order the arguments were written in.
fn format_spec_args(spec: &FormatSpec) -> String {
    let mut out = format_text(spec.format).to_string();
    let o = &spec.options;
    if o.jpeg_progressive {
        out.push_str(",progressive");
    }
    if let Some(sub) = o.jpeg_subsampling {
        out.push_str(&format!(",subsampling:{}", sub.name()));
    }
    if o.png_interlace {
        out.push_str(",interlace");
    }
    if let Some(method) = o.webp_method {
        out.push_str(&format!(",method:{method}"));
    }
    if let Some(sns) = o.webp_sns {
        out.push_str(&format!(",sns:{sns}"));
    }
    out
}

#[cfg(test)]
mod tests;
