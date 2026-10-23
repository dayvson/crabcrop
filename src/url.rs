//! Parsing of crabcrop's image URLs.
//!
//! ```text
//! /<signature>/<operations...>/<image-url>
//! ```
//!
//! [`parse_request`] parses a full path into its [`Signature`] and a
//! [`Request`]: the [`Operation`]s in path order (size, crop, filters)
//! plus the source image URL with its slashes intact. [`parse_signature`]
//! and [`parse_operations`] are the two halves, usable on their own.

/// How the request proves it was minted by someone holding the server secret.
///
/// Either an HMAC-SHA256 (hex) of the rest of the URL, or the literal `unsafe`
/// when the operator allows unsigned URLs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Signature {
    /// Hex-encoded HMAC-SHA256 of the remainder of the path.
    Hmac(String),
    /// Unsigned request; honored only when the server enables it.
    Unsafe,
}

/// What went wrong while splitting the signature off a request path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// The path was empty or just `/`.
    EmptyPath,
    /// There was no `/` after the first segment, so no operations or
    /// image URL could follow.
    MissingSignature,
    /// The signature segment was neither `unsafe` nor lowercase
    /// hex (audit PL-7: one canonical spelling).
    MalformedSignature,
    /// A size segment did not match `<width>x<height>`. Carries the
    /// offending segment.
    InvalidSize(String),
    /// A `crop:...` segment did not match `crop:<WxH>[:<align>]`.
    /// Carries the offending segment.
    InvalidCrop(String),
    /// A `crop-rect:...` segment did not match
    /// `crop-rect:<left>x<top>:<right>x<bottom>` with right > left
    /// and bottom > top. Carries the offending segment.
    InvalidCropRect(String),
    /// A `pad:...` segment did not match `pad:<WxH>[:<fill>]`.
    /// Carries the offending segment.
    InvalidPad(String),
    /// An `exp:...` segment did not carry a valid unix
    /// timestamp. Carries the offending segment.
    InvalidExpiry(String),
    /// A `preset:<name>` segment named a preset the server has
    /// not configured. Carries the name.
    UnknownPreset(String),
    /// A preset definition (or a `preset:` segment itself) is
    /// malformed. Carries the offending text.
    InvalidPreset(String),
    /// Free-form operations were used while the server runs in
    /// presets-only mode. Carries the offending segment.
    PresetsOnly(String),
    /// A `filters:...` segment did not parse. Carries the offending
    /// segment.
    InvalidFilters(String),
    /// No source URL followed the operations.
    MissingSourceUrl,
}

impl ParseError {
    /// The variant's stable machine-readable code, for the
    /// v2 error contract (`{"error":{"code":...}}`). These
    /// strings are frozen once published: a variant may be
    /// renamed in code, but its code never changes.
    pub fn code(&self) -> &'static str {
        match self {
            ParseError::EmptyPath => "empty_path",
            ParseError::MissingSignature => "missing_signature",
            ParseError::MalformedSignature => "malformed_signature",
            ParseError::InvalidSize(_) => "invalid_size",
            ParseError::InvalidCrop(_) => "invalid_crop",
            ParseError::InvalidCropRect(_) => "invalid_crop_rect",
            ParseError::InvalidPad(_) => "invalid_pad",
            ParseError::InvalidExpiry(_) => "invalid_expiry",
            ParseError::UnknownPreset(_) => "unknown_preset",
            ParseError::InvalidPreset(_) => "invalid_preset",
            ParseError::PresetsOnly(_) => "presets_only",
            ParseError::InvalidFilters(_) => "invalid_filters",
            ParseError::MissingSourceUrl => "missing_source_url",
        }
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::EmptyPath => write!(f, "empty path"),
            ParseError::MissingSignature => write!(f, "missing signature segment"),
            ParseError::MalformedSignature => write!(f, "malformed signature segment"),
            ParseError::InvalidSize(seg) => write!(f, "invalid size segment: {seg}"),
            ParseError::InvalidCrop(seg) => write!(f, "invalid crop segment: {seg}"),
            ParseError::InvalidCropRect(seg) => {
                write!(f, "invalid crop-rect segment: {seg}")
            }
            ParseError::InvalidPad(seg) => write!(f, "invalid pad segment: {seg}"),
            ParseError::InvalidExpiry(seg) => write!(f, "invalid expiry segment: {seg}"),
            ParseError::UnknownPreset(name) => write!(f, "unknown preset: {name}"),
            ParseError::InvalidPreset(text) => write!(f, "invalid preset: {text}"),
            ParseError::PresetsOnly(seg) => {
                write!(f, "free-form operations are disabled (presets only): {seg}")
            }
            ParseError::InvalidFilters(seg) => {
                write!(f, "invalid filters segment: {seg}")
            }
            ParseError::MissingSourceUrl => write!(f, "missing source URL"),
        }
    }
}

impl std::error::Error for ParseError {}

/// One dimension of a requested image size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dimension {
    /// An exact pixel count, e.g. the `300` in `300x200`.
    Pixels(u32),
    /// `0` or omitted: scale proportionally to the other dimension.
    /// `0x0` therefore keeps the original size.
    Proportional,
    /// `orig`: keep the source image's dimension on this axis.
    Original,
}

/// A parsed size segment such as `300x200`, `-300x-200`, or `origx100`.
///
/// A leading `-` on a dimension flips the image along that axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSize {
    pub width: Dimension,
    pub height: Dimension,
    pub flip_horizontal: bool,
    pub flip_vertical: bool,
}

/// How `pad` fills the canvas around the fitted image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadFill {
    /// A solid RGB color.
    Color([u8; 3]),
    /// A blurred, cover-scaled copy of the image itself
    /// (ambient bars).
    Blur,
    /// The request's `fill(...)` directive color, or black
    /// when the request names none.
    Directive,
}

/// Where the crop keeps the image when aspect ratios differ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Align {
    /// Center the crop (default).
    Center,
    /// Keep the top.
    Top,
    /// Keep the bottom.
    Bottom,
    /// Keep the left.
    Left,
    /// Keep the right.
    Right,
    /// Position on the detected focal point.
    Smart,
    /// Crop around the largest detected face.
    Face,
}

/// Where a text overlay anchors: one of the nine grid points
/// of the image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Anchor {
    TopLeft,
    TopCenter,
    TopRight,
    CenterLeft,
    Center,
    CenterRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
}

impl Anchor {
    fn parse(s: &str) -> Option<Anchor> {
        Some(match s {
            "top-left" => Anchor::TopLeft,
            "top-center" => Anchor::TopCenter,
            "top-right" => Anchor::TopRight,
            "center-left" => Anchor::CenterLeft,
            "center" => Anchor::Center,
            "center-right" => Anchor::CenterRight,
            "bottom-left" => Anchor::BottomLeft,
            "bottom-center" => Anchor::BottomCenter,
            "bottom-right" => Anchor::BottomRight,
            _ => return None,
        })
    }

    /// (is-left, is-center-x, is-right) style placement axes:
    /// returns (horizontal, vertical) each in 0..=2 (start,
    /// middle, end).
    pub fn axes(self) -> (u8, u8) {
        match self {
            Anchor::TopLeft => (0, 0),
            Anchor::TopCenter => (1, 0),
            Anchor::TopRight => (2, 0),
            Anchor::CenterLeft => (0, 1),
            Anchor::Center => (1, 1),
            Anchor::CenterRight => (2, 1),
            Anchor::BottomLeft => (0, 2),
            Anchor::BottomCenter => (1, 2),
            Anchor::BottomRight => (2, 2),
        }
    }
}

/// Output image format requested by a `format(...)` filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Jpeg,
    Png,
    WebP,
    Jxl,
    Avif,
    Gif,
}

/// Encoding quality from a `quality(...)` filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quality {
    /// An explicit value, 0-100.
    Value(u8),
    /// Chosen per image by content analysis at encode time.
    Auto,
}

/// One filter from a `filters:...` segment.
///
/// `f32` parameters mean `Eq` cannot be derived; `PartialEq` is enough
/// for the tests and the pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum Filter {
    /// Desaturate the image.
    Grayscale,
    /// Invert every channel.
    Invert,
    /// Gaussian blur; sigma must be positive.
    Blur(f32),
    /// Add to every channel, clamped to 0-255.
    Brighten(i32),
    /// Adjust contrast.
    Contrast(f32),
    /// Rotate the hue, in degrees.
    HueRotate(i32),
    /// Sharpen by this amount (unsharp mask strength, > 0).
    Sharpen(f32),
    /// Scale color saturation: 0 is grayscale, 1 is identity.
    Saturation(f32),
    /// Per-channel level offsets, each -255..=255.
    Rgb(i32, i32, i32),
    /// Histogram equalization per channel.
    Equalize,
    /// Rotate by a multiple of 90 degrees.
    Rotate(i32),
    /// Crop away the uniform border, keeping pixels that differ
    /// from the corner color by more than this tolerance (0-255).
    Trim(u8),
    /// Overlay another image at (x, y) with an opacity (0-100).
    /// The URL is the percent-decoded watermark location; the
    /// server fetches it before the engine runs. Negative
    /// coordinates anchor from the right/bottom edges.
    Watermark {
        url: String,
        x: i32,
        y: i32,
        alpha: u8,
    },
    /// Composite another image over the result with a blend
    /// mode, at (x, y) with an opacity (0-100), optionally
    /// scaled to `width` percent of the base image's width.
    /// The URL is the percent-decoded overlay location; the
    /// server fetches it before the engine runs, through the
    /// same prefetch as watermarks. Negative coordinates
    /// anchor from the right/bottom edges.
    Overlay {
        url: String,
        x: i32,
        y: i32,
        alpha: u8,
        blend: BlendMode,
        width: Option<u32>,
    },
    /// Remove the background with the server's segmentation
    /// model, keeping the subject over transparency. The model
    /// is operator-configured; the engine applies its mask.
    RemoveBg,
    /// Keep the source ICC profile instead of converting
    /// to sRGB (`filters:keep_profile`).
    KeepProfile,
    /// Blur every detected face (privacy redaction), backed by
    /// the server's face model.
    BlurFaces,
    /// Pixelate every detected face (privacy redaction),
    /// backed by the server's face model.
    PixelateFaces,
    /// Blur every detected object of the named class
    /// (privacy redaction), backed by the server's
    /// detection model.
    BlurClass(String),
    /// Pixelate every detected object of the named class
    /// (privacy redaction), backed by the server's
    /// detection model.
    PixelateClass(String),
    /// Padding for `crop:*:face`, as a fraction of the face
    /// size added on every side. A request-wide directive,
    /// like `no_upscale`.
    FacePad(f32),
    /// AI super-resolution by the given factor (2 or 4),
    /// backed by the server's upscale model.
    Upscale(u32),
    /// Device-pixel-ratio directive: scale size and crop
    /// targets by the factor (1, 1.5, 2, or 3) before the
    /// no-upscale cap sees them.
    Dpr(f32),
    /// Rendered text overlaid at one of the nine grid anchors.
    /// The text is the percent-decoded label; color is RGBA.
    Text {
        text: String,
        size: u32,
        color: [u8; 4],
        anchor: Anchor,
    },
    /// Background color for flattening transparency, as RGB.
    Fill([u8; 3]),
    /// Rounded corners: radii in pixels, CSS order (top-left,
    /// top-right, bottom-right, bottom-left), applied to the
    /// alpha channel with antialiased edges.
    Radius([u32; 4]),
    /// An opaque band of this width and color (RGB) painted
    /// inside the frame; dimensions do not change.
    Border(u32, [u8; 3]),
    /// Keep only the inscribed shape; the rest of the frame
    /// becomes transparent.
    Mask(MaskShape),
    /// The classic warm sepia tone.
    Sepia,
    /// Map shadows to the first color and highlights to the
    /// second, by luminance.
    Duotone([u8; 3], [u8; 3]),
    /// Single-hue rendering in the given color: full
    /// luminance renders exactly that color.
    Monochrome([u8; 3]),
    /// Mosaic the image into blocks of this many pixels,
    /// each block the average of its pixels.
    Pixelate(u32),
    /// Mirror the image, horizontally or vertically.
    Flip(FlipDirection),
    /// Gaussian-blur the rectangle (x, y, w, h), in the
    /// pixels of the image at the point the filter runs.
    /// Redaction of plates, screens, documents.
    BlurRegion { x: u32, y: u32, w: u32, h: u32 },
    /// Pixelate the rectangle (x, y, w, h), same coordinate
    /// convention as `BlurRegion`.
    PixelateRegion { x: u32, y: u32, w: u32, h: u32 },
    /// Encoding directive: re-encode at lower quality until the
    /// output fits this many bytes, quality permitting.
    MaxBytes(u64),
    /// Encoding quality.
    Quality(Quality),
    /// Output container.
    Format(ImageFormat),
    /// Cap every resize at the source dimensions.
    NoUpscale,
}

/// A blend mode for the `overlay` filter: the CSS formulas,
/// applied per RGB channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    /// The overlay color, composited by alpha (watermark's
    /// behavior).
    Normal,
    /// base x overlay: darkens.
    Multiply,
    /// Inverted multiply: lightens.
    Screen,
    /// Multiply below mid-gray, screen above it.
    Overlay,
}

/// A direction for the `flip` filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlipDirection {
    /// Mirror left-to-right.
    Horizontal,
    /// Mirror top-to-bottom.
    Vertical,
}

/// A shape for the `mask` filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskShape {
    /// The inscribed circle: diameter is the smaller
    /// dimension, centered.
    Circle,
    /// The inscribed ellipse, touching all four edges.
    Ellipse,
}

/// One operation from the request path, in path order.
#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    /// Resize to a size segment (`300x200`, ...).
    Size(ImageSize),
    /// Crop to a size with an alignment, before resize.
    Crop { size: ImageSize, align: Align },
    /// Scale proportionally to fit inside a box (`fit-in:300x200`).
    FitIn(ImageSize),
    /// Fit inside a box, then extend the canvas to exactly the
    /// box, filling the remainder (`pad:300x200[:<fill>]`).
    Pad { size: ImageSize, fill: PadFill },
    /// Cut an exact pixel rectangle
    /// (`crop-rect:<left>x<top>:<right>x<bottom>`).
    CropRect {
        left: u32,
        top: u32,
        right: u32,
        bottom: u32,
    },
    /// Filters applied in order after resize.
    Filters(Vec<Filter>),
}

/// A fully parsed request path: operations plus the source image URL.
///
/// The composed parse result. The server does not use it: HMAC
/// verification needs the raw remainder of the path, so the server
/// works with [`parse_signature`] and [`parse_operations`]
/// separately. This type and [`parse_request`] are the convenient
/// whole-path form, kept for tests and future library use.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub struct Request<'a> {
    pub operations: Vec<Operation>,
    pub source_url: &'a str,
}

/// The largest dimension a request may name, per axis.
/// Anything larger is a parse error, not an allocation
/// (security audit E-4): the value sits one below the decode
/// ceiling and matches the strictest encoder limit in the
/// pipeline (WebP's 16383).
pub const MAX_DIMENSION: u32 = 16_383;

fn parse_dimension(part: &str) -> Result<(Dimension, bool), ()> {
    let (flip, digits) = match part.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, part),
    };
    let dimension = match digits {
        "" | "0" => Dimension::Proportional,
        "orig" => Dimension::Original,
        d => {
            let n = d.parse::<u32>().map_err(|_| ())?;
            if n > MAX_DIMENSION {
                return Err(());
            }
            Dimension::Pixels(n)
        }
    };
    Ok((dimension, flip))
}

/// Parse a size segment (`300x200`, `-300x0`, `origx100`, ...) into an
/// [`ImageSize`].
pub fn parse_size(segment: &str) -> Result<ImageSize, ParseError> {
    let invalid = || ParseError::InvalidSize(segment.to_string());
    let (w, h) = segment.split_once('x').ok_or_else(invalid)?;
    let (width, flip_horizontal) = parse_dimension(w).map_err(|_| invalid())?;
    let (height, flip_vertical) = parse_dimension(h).map_err(|_| invalid())?;
    Ok(ImageSize {
        width,
        height,
        flip_horizontal,
        flip_vertical,
    })
}

/// Split the signature off the front of a request path.
///
/// Returns the signature and the remainder of the path (operations and
/// image URL, still slash-separated).
pub fn parse_signature(path: &str) -> Result<(Signature, &str), ParseError> {
    let path = path.strip_prefix('/').unwrap_or(path);
    if path.is_empty() {
        return Err(ParseError::EmptyPath);
    }
    let (first, rest) = path.split_once('/').ok_or(ParseError::MissingSignature)?;
    if first.is_empty() {
        return Err(ParseError::EmptyPath);
    }
    let signature = if first == "unsafe" {
        Signature::Unsafe
    } else if !first.is_empty()
        && first
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        // One canonical spelling (audit PL-7): builders emit
        // lowercase hex, and other spellings (uppercase, non-hex
        // tags) are parse errors rather than alternate forms of
        // the same URL.
        Signature::Hmac(first.to_string())
    } else {
        return Err(ParseError::MalformedSignature);
    };
    Ok((signature, rest))
}

/// Parse one path segment as an operation.
///
/// Returns `None` when the segment is not an operation at all, which
/// means it starts the source URL. A segment that claims to be an
/// operation (`crop:` / `filters:` prefix) but is malformed is an error.
fn parse_operation(seg: &str) -> Option<Result<Operation, ParseError>> {
    if let Some(rest) = seg.strip_prefix("crop-rect:") {
        return Some(parse_crop_rect(rest, seg));
    }
    if let Some(rest) = seg.strip_prefix("crop:") {
        return Some(parse_crop(rest, seg));
    }
    if let Some(rest) = seg.strip_prefix("pad:") {
        return Some(parse_pad(rest, seg));
    }
    if let Some(rest) = seg.strip_prefix("fit-in:") {
        return Some(parse_fit_in(rest, seg));
    }
    if let Some(rest) = seg.strip_prefix("filters:") {
        return Some(parse_filters(rest, seg));
    }
    match parse_size(seg) {
        Ok(size) => Some(Ok(Operation::Size(size))),
        Err(_) => None,
    }
}

/// Parse the box after `fit-in:`. Flips make no sense when fitting,
/// so a flipped size is invalid here.
fn parse_fit_in(rest: &str, seg: &str) -> Result<Operation, ParseError> {
    let invalid = || ParseError::InvalidSize(seg.to_string());
    let size = parse_size(rest).map_err(|_| invalid())?;
    if size.flip_horizontal || size.flip_vertical {
        return Err(invalid());
    }
    Ok(Operation::FitIn(size))
}

/// Parse `<left>x<top>:<right>x<bottom>` after `crop-rect:`.
/// Ordering is validated here; bounds against the actual image are
/// the engine's job, since they depend on the source.
fn parse_crop_rect(rest: &str, seg: &str) -> Result<Operation, ParseError> {
    let invalid = || ParseError::InvalidCropRect(seg.to_string());
    let (lt, rb) = rest.split_once(':').ok_or_else(invalid)?;
    let (left, top) = parse_point(lt).ok_or_else(invalid)?;
    let (right, bottom) = parse_point(rb).ok_or_else(invalid)?;
    if right <= left || bottom <= top {
        return Err(invalid());
    }
    Ok(Operation::CropRect {
        left,
        top,
        right,
        bottom,
    })
}

/// Parse an `<x>x<y>` point with unsigned coordinates.
fn parse_point(s: &str) -> Option<(u32, u32)> {
    let (x, y) = s.split_once('x')?;
    Some((x.parse().ok()?, y.parse().ok()?))
}

fn parse_align(s: &str) -> Option<Align> {
    match s {
        "center" => Some(Align::Center),
        "smart" => Some(Align::Smart),
        "face" => Some(Align::Face),
        "top" => Some(Align::Top),
        "bottom" => Some(Align::Bottom),
        "left" => Some(Align::Left),
        "right" => Some(Align::Right),
        _ => None,
    }
}

/// The size contract for `crop:` and `pad:`: an exact box,
/// pixels on both axes and no flips. The general size
/// grammar also spells proportional and original sizes,
/// which mean something for resizes and fit-in but not for
/// an operation whose whole job is one exact box, and a
/// flip belongs to a resize or `filters:flip`. Rejecting
/// the spellings at parse keeps them out of URLs the
/// engine could only refuse after fetching the source.
fn exact_box(size: &ImageSize) -> bool {
    let pixels = |d: &Dimension| matches!(d, Dimension::Pixels(n) if *n >= 1);
    pixels(&size.width) && pixels(&size.height) && !size.flip_horizontal && !size.flip_vertical
}

/// Parse the dimensions (and optional alignment) after `crop:`.
fn parse_crop(rest: &str, seg: &str) -> Result<Operation, ParseError> {
    let invalid = || ParseError::InvalidCrop(seg.to_string());
    let (dims, align) = match rest.split_once(':') {
        Some((d, a)) => (d, parse_align(a).ok_or_else(invalid)?),
        None => (rest, Align::Center),
    };
    let size = parse_size(dims).map_err(|_| invalid())?;
    if !exact_box(&size) {
        return Err(invalid());
    }
    Ok(Operation::Crop { size, align })
}

/// Parse the dimensions (and optional fill) after `pad:`.
/// The fill is `blur`, or a 6-digit hex color with the same
/// semantics as the `fill` filter; absent means the request's
/// `fill(...)` directive (or black).
fn parse_pad(rest: &str, seg: &str) -> Result<Operation, ParseError> {
    let invalid = || ParseError::InvalidPad(seg.to_string());
    let (dims, fill) = match rest.split_once(':') {
        Some((d, f)) => (d, Some(f)),
        None => (rest, None),
    };
    let fill = match fill {
        None => PadFill::Directive,
        Some("blur") => PadFill::Blur,
        Some(p) => {
            if p.len() != 6 || !p.chars().all(|c| c.is_ascii_hexdigit()) {
                return Err(invalid());
            }
            let byte = |i: usize| u8::from_str_radix(&p[i..i + 2], 16).map_err(|_| invalid());
            PadFill::Color([byte(0)?, byte(2)?, byte(4)?])
        }
    };
    let size = parse_size(dims).map_err(|_| invalid())?;
    if !exact_box(&size) {
        return Err(invalid());
    }
    Ok(Operation::Pad { size, fill })
}

fn parse_format(s: &str) -> Option<ImageFormat> {
    match s {
        "jpeg" | "jpg" => Some(ImageFormat::Jpeg),
        "png" => Some(ImageFormat::Png),
        "webp" => Some(ImageFormat::WebP),
        "jxl" => Some(ImageFormat::Jxl),
        "avif" => Some(ImageFormat::Avif),
        "gif" => Some(ImageFormat::Gif),
        _ => None,
    }
}

/// Parse the filter list after `filters:`.
/// A detection class name for `blurclass` / `pixelateclass`:
/// the vocabulary's own spellings (COCO names like `traffic
/// light` and `cell phone` included), so letters, digits,
/// spaces, and a few joiners, trimmed, non-empty, bounded.
fn parse_class_name(raw: &str) -> Option<String> {
    let name = raw.trim();
    if name.is_empty() || name.len() > 64 {
        return None;
    }
    if !name
        .chars()
        .all(|c| c.is_alphanumeric() || matches!(c, ' ' | '-' | '_' | '.' | '/'))
    {
        return None;
    }
    Some(name.to_string())
}

fn parse_filters(rest: &str, seg: &str) -> Result<Operation, ParseError> {
    let invalid = || ParseError::InvalidFilters(seg.to_string());
    if rest.is_empty() {
        return Err(invalid());
    }
    let mut filters = Vec::new();
    for part in rest.split(':') {
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
        let filter = match (name, params) {
            ("grayscale", None) => Filter::Grayscale,
            ("invert", None) => Filter::Invert,
            ("no_upscale", None) => Filter::NoUpscale,
            ("blur", Some(p)) => {
                let sigma: f32 = p.parse().map_err(|_| invalid())?;
                if !sigma.is_finite() || sigma <= 0.0 {
                    return Err(invalid());
                }
                Filter::Blur(sigma)
            }
            ("brighten", Some(p)) => {
                let amount: i32 = p.parse().map_err(|_| invalid())?;
                Filter::Brighten(amount)
            }
            ("contrast", Some(p)) => {
                let amount: f32 = p.parse().map_err(|_| invalid())?;
                if !amount.is_finite() {
                    return Err(invalid());
                }
                Filter::Contrast(amount)
            }
            ("huerotate", Some(p)) => {
                let degrees: i32 = p.parse().map_err(|_| invalid())?;
                Filter::HueRotate(degrees)
            }
            ("quality", Some(p)) => {
                if p == "auto" {
                    Filter::Quality(Quality::Auto)
                } else {
                    let q: u8 = p.parse().map_err(|_| invalid())?;
                    if q > 100 {
                        return Err(invalid());
                    }
                    Filter::Quality(Quality::Value(q))
                }
            }
            ("format", Some(p)) => Filter::Format(parse_format(p).ok_or_else(invalid)?),
            ("sharpen", Some(p)) => {
                let amount: f32 = p.parse().map_err(|_| invalid())?;
                if !amount.is_finite() || amount <= 0.0 {
                    return Err(invalid());
                }
                Filter::Sharpen(amount)
            }
            ("saturation", Some(p)) => {
                let amount: f32 = p.parse().map_err(|_| invalid())?;
                if !amount.is_finite() || amount < 0.0 {
                    return Err(invalid());
                }
                Filter::Saturation(amount)
            }
            ("rgb", Some(p)) => {
                let parts: Vec<&str> = p.split(',').collect();
                if parts.len() != 3 {
                    return Err(invalid());
                }
                let mut offsets = [0i32; 3];
                for (i, part) in parts.iter().enumerate() {
                    let v: i32 = part.parse().map_err(|_| invalid())?;
                    if !(-255..=255).contains(&v) {
                        return Err(invalid());
                    }
                    offsets[i] = v;
                }
                Filter::Rgb(offsets[0], offsets[1], offsets[2])
            }
            ("equalize", None) => Filter::Equalize,
            ("removebg", None) => Filter::RemoveBg,
            ("keep_profile", None) => Filter::KeepProfile,
            ("blurfaces", None) => Filter::BlurFaces,
            ("pixelatefaces", None) => Filter::PixelateFaces,
            ("blurclass", Some(p)) => Filter::BlurClass(
                parse_class_name(&percent_decode(p).ok_or_else(invalid)?).ok_or_else(invalid)?,
            ),
            ("pixelateclass", Some(p)) => Filter::PixelateClass(
                parse_class_name(&percent_decode(p).ok_or_else(invalid)?).ok_or_else(invalid)?,
            ),
            ("facepad", Some(p)) => {
                let pad: f32 = p.parse().map_err(|_| invalid())?;
                if !pad.is_finite() || !(0.0..=3.0).contains(&pad) {
                    return Err(invalid());
                }
                Filter::FacePad(pad)
            }
            ("upscale", Some(p)) => {
                let factor: u32 = p.parse().map_err(|_| invalid())?;
                if !matches!(factor, 2 | 4) {
                    return Err(invalid());
                }
                Filter::Upscale(factor)
            }
            ("dpr", Some(p)) => {
                let factor: f32 = p.parse().map_err(|_| invalid())?;
                if ![1.0, 1.5, 2.0, 3.0].contains(&factor) {
                    return Err(invalid());
                }
                Filter::Dpr(factor)
            }
            ("text", Some(p)) => {
                // <text>,<size>,<color>,<anchor>. The text is
                // percent-encoded and may contain commas, so
                // split from the right where the fixed fields
                // are.
                let parts: Vec<&str> = p.rsplitn(4, ',').collect();
                if parts.len() != 4 {
                    return Err(invalid());
                }
                let anchor = Anchor::parse(parts[0]).ok_or_else(invalid)?;
                let color = parse_text_color(parts[1]).ok_or_else(invalid)?;
                let size: u32 = parts[2].parse().map_err(|_| invalid())?;
                if !(1..=1000).contains(&size) {
                    return Err(invalid());
                }
                let text = percent_decode(parts[3]).ok_or_else(invalid)?;
                if text.is_empty() {
                    return Err(invalid());
                }
                Filter::Text {
                    text,
                    size,
                    color,
                    anchor,
                }
            }
            ("trim", None) => Filter::Trim(10),
            ("trim", Some(p)) => {
                let tolerance: u8 = p.parse().map_err(|_| invalid())?;
                Filter::Trim(tolerance)
            }
            ("watermark", Some(p)) => {
                // <url>,<x>,<y>,<alpha>. The URL lives inside a
                // path segment, so it arrives percent-encoded; it
                // may itself contain commas, so split from the
                // right where the numeric fields are.
                let parts: Vec<&str> = p.rsplitn(4, ',').collect();
                if parts.len() != 4 {
                    return Err(invalid());
                }
                let alpha: u32 = parts[0].parse().map_err(|_| invalid())?;
                if alpha > 100 {
                    return Err(invalid());
                }
                let y: i32 = parts[1].parse().map_err(|_| invalid())?;
                let x: i32 = parts[2].parse().map_err(|_| invalid())?;
                let url = percent_decode(parts[3]).ok_or_else(invalid)?;
                if url.is_empty() {
                    return Err(invalid());
                }
                Filter::Watermark {
                    url,
                    x,
                    y,
                    alpha: alpha as u8,
                }
            }
            ("rotate", Some(p)) => {
                let degrees: i32 = p.parse().map_err(|_| invalid())?;
                if degrees % 90 != 0 {
                    return Err(invalid());
                }
                Filter::Rotate(degrees)
            }
            ("overlay", Some(p)) => {
                // <url>,<x>,<y>,<alpha>,<blend>[,<width>]. Like
                // watermark, split from the right: the URL may
                // contain commas, the trailing fields may not.
                let parts: Vec<&str> = p.rsplitn(6, ',').collect();
                if parts.len() != 5 && parts.len() != 6 {
                    return Err(invalid());
                }
                let (width, rest) = if parts.len() == 6 {
                    let w: u32 = parts[0].parse().map_err(|_| invalid())?;
                    if w == 0 || w > 400 {
                        return Err(invalid());
                    }
                    (Some(w), &parts[1..])
                } else {
                    (None, &parts[..])
                };
                let blend = match rest[0] {
                    "normal" => BlendMode::Normal,
                    "multiply" => BlendMode::Multiply,
                    "screen" => BlendMode::Screen,
                    "overlay" => BlendMode::Overlay,
                    _ => return Err(invalid()),
                };
                let alpha: u32 = rest[1].parse().map_err(|_| invalid())?;
                if alpha > 100 {
                    return Err(invalid());
                }
                let y: i32 = rest[2].parse().map_err(|_| invalid())?;
                let x: i32 = rest[3].parse().map_err(|_| invalid())?;
                let url = percent_decode(rest[4]).ok_or_else(invalid)?;
                if url.is_empty() {
                    return Err(invalid());
                }
                Filter::Overlay {
                    url,
                    x,
                    y,
                    alpha: alpha as u8,
                    blend,
                    width,
                }
            }
            ("radius", Some(p)) => {
                let parts: Vec<&str> = p.split(',').collect();
                if parts.len() != 1 && parts.len() != 4 {
                    return Err(invalid());
                }
                let mut radii = [0u32; 4];
                for (i, part) in parts.iter().enumerate() {
                    let v: u32 = part.parse().map_err(|_| invalid())?;
                    if v > MAX_DIMENSION {
                        return Err(invalid());
                    }
                    radii[i] = v;
                }
                if parts.len() == 1 {
                    radii = [radii[0]; 4];
                }
                Filter::Radius(radii)
            }
            ("border", Some(p)) => {
                // <width>,<color>, the color spelled like fill's.
                let parts: Vec<&str> = p.split(',').collect();
                if parts.len() != 2 {
                    return Err(invalid());
                }
                let width: u32 = parts[0].parse().map_err(|_| invalid())?;
                if width == 0 || width > MAX_DIMENSION {
                    return Err(invalid());
                }
                let hex = parts[1];
                if hex.len() != 6 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
                    return Err(invalid());
                }
                let byte = |i: usize| u8::from_str_radix(&hex[i..i + 2], 16).map_err(|_| invalid());
                Filter::Border(width, [byte(0)?, byte(2)?, byte(4)?])
            }
            ("mask", Some(p)) => match p {
                "circle" => Filter::Mask(MaskShape::Circle),
                "ellipse" => Filter::Mask(MaskShape::Ellipse),
                _ => return Err(invalid()),
            },
            ("sepia", None) => Filter::Sepia,
            ("duotone", Some(p)) => {
                let parts: Vec<&str> = p.split(',').collect();
                if parts.len() != 2 {
                    return Err(invalid());
                }
                let color = |hex: &str| -> Result<[u8; 3], ParseError> {
                    if hex.len() != 6 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
                        return Err(invalid());
                    }
                    let byte =
                        |i: usize| u8::from_str_radix(&hex[i..i + 2], 16).map_err(|_| invalid());
                    Ok([byte(0)?, byte(2)?, byte(4)?])
                };
                Filter::Duotone(color(parts[0])?, color(parts[1])?)
            }
            ("monochrome", Some(p)) => {
                if p.len() != 6 || !p.chars().all(|c| c.is_ascii_hexdigit()) {
                    return Err(invalid());
                }
                let byte = |i: usize| u8::from_str_radix(&p[i..i + 2], 16).map_err(|_| invalid());
                Filter::Monochrome([byte(0)?, byte(2)?, byte(4)?])
            }
            ("pixelate", Some(p)) => {
                let block: u32 = p.parse().map_err(|_| invalid())?;
                if block == 0 || block > MAX_DIMENSION {
                    return Err(invalid());
                }
                Filter::Pixelate(block)
            }
            ("blur_region", Some(p)) | ("pixelate_region", Some(p)) => {
                let parts: Vec<&str> = p.split(',').collect();
                if parts.len() != 4 {
                    return Err(invalid());
                }
                let mut v = [0u32; 4];
                for (i, part) in parts.iter().enumerate() {
                    v[i] = part.parse().map_err(|_| invalid())?;
                    if v[i] > MAX_DIMENSION {
                        return Err(invalid());
                    }
                }
                if v[2] == 0 || v[3] == 0 {
                    return Err(invalid());
                }
                let (x, y, w, h) = (v[0], v[1], v[2], v[3]);
                if name == "blur_region" {
                    Filter::BlurRegion { x, y, w, h }
                } else {
                    Filter::PixelateRegion { x, y, w, h }
                }
            }
            ("flip", Some(p)) => match p {
                "h" => Filter::Flip(FlipDirection::Horizontal),
                "v" => Filter::Flip(FlipDirection::Vertical),
                _ => return Err(invalid()),
            },
            ("fill", Some(p)) => {
                if p.len() != 6 || !p.chars().all(|c| c.is_ascii_hexdigit()) {
                    return Err(invalid());
                }
                let byte = |i: usize| u8::from_str_radix(&p[i..i + 2], 16).map_err(|_| invalid());
                Filter::Fill([byte(0)?, byte(2)?, byte(4)?])
            }
            ("max_bytes", Some(p)) => {
                let n: u64 = p.parse().map_err(|_| invalid())?;
                if n == 0 {
                    return Err(invalid());
                }
                Filter::MaxBytes(n)
            }
            _ => return Err(invalid()),
        };
        filters.push(filter);
    }
    Ok(Operation::Filters(filters))
}

/// Split operations off the front of the path remainder.
///
/// Returns the operations in path order and the source URL (with its
/// slashes intact). The first segment that is not an operation starts
/// the source URL.
pub fn parse_operations(input: &str) -> Result<(Vec<Operation>, &str), ParseError> {
    let (operations, source_url, _) = parse_operations_with_expiry(input)?;
    Ok((operations, source_url))
}

/// Parse the operation segments off the front of the remainder,
/// also extracting the request's expiry: an `exp:<unix-seconds>`
/// segment may appear anywhere among the operations. It is
/// metadata, not an operation, and it sits inside the signed
/// remainder, so the HMAC covers it. The last `exp` wins.
pub fn parse_operations_with_expiry(
    input: &str,
) -> Result<(Vec<Operation>, &str, Option<u64>), ParseError> {
    let mut operations = Vec::new();
    let mut expiry = None;
    let mut rest = input;
    loop {
        let (seg, remaining) = match rest.split_once('/') {
            Some((s, r)) => (s, r),
            None => (rest, ""),
        };
        if let Some(ts) = seg.strip_prefix("exp:") {
            if ts.is_empty() || !ts.bytes().all(|b| b.is_ascii_digit()) {
                return Err(ParseError::InvalidExpiry(seg.to_string()));
            }
            expiry = Some(
                ts.parse::<u64>()
                    .map_err(|_| ParseError::InvalidExpiry(seg.to_string()))?,
            );
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
                return Ok((operations, rest, expiry));
            }
        }
    }
}

/// Parse an operations-only string, the shape of a preset
/// definition: every segment must be an operation, and neither
/// `exp:` metadata nor nested `preset:` references are allowed.
/// Used to validate preset definitions at server boot; request
/// expansion itself is textual (see [`expand_presets`]).
pub fn parse_preset_operations(input: &str) -> Result<Vec<Operation>, ParseError> {
    if input.is_empty() {
        return Err(ParseError::InvalidPreset(input.to_string()));
    }
    let mut operations = Vec::new();
    for seg in input.split('/') {
        if seg.starts_with("preset:") || seg.starts_with("exp:") {
            return Err(ParseError::InvalidPreset(input.to_string()));
        }
        match parse_operation(seg) {
            Some(Ok(op)) => operations.push(op),
            Some(Err(e)) => return Err(e),
            None => return Err(ParseError::InvalidPreset(input.to_string())),
        }
    }
    Ok(operations)
}

/// Expand `preset:<name>` segments in a request remainder,
/// splicing each preset's operation string in its place.
///
/// The signature covers the unexpanded remainder (the preset
/// reference is what gets signed); expansion is deterministic
/// and server-side, and everything downstream of it (parsing,
/// cache keys) sees the expanded string, so editing a preset
/// definition produces fresh cache entries naturally.
///
/// The walk stops expanding at the first segment that starts
/// the source URL, using the same dispatch as
/// [`parse_operations_with_expiry`]: a `preset:` segment inside
/// the source path is URL text, not a reference. Expansion is
/// single-pass: spliced text is not rescanned (and configured
/// presets cannot contain `preset:` segments anyway; see
/// [`parse_preset_operations`]).
///
/// With `only_presets`, any real operation segment in the
/// request is an error: the operations must all arrive via
/// presets. `exp:` metadata stays allowed; it is not an
/// operation.
/// The preset names a remainder references, in order,
/// deduplicated: the cache groups (RFC 9875) the response
/// belongs to. Scans the same leading segment run
/// [`expand_presets`] expands and stops where it stops (the
/// source, or a segment that is not an operation).
pub fn preset_names(rest: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut tail = rest;
    loop {
        let (seg, remaining) = match tail.split_once('/') {
            Some((s, r)) => (s, r),
            None => (tail, ""),
        };
        if let Some(name) = seg.strip_prefix("preset:") {
            if !name.is_empty() && !out.iter().any(|n| n == name) {
                out.push(name.to_string());
            }
        } else if !(seg.starts_with("exp:") || parse_operation(seg).is_some_and(|r| r.is_ok())) {
            break;
        }
        if remaining.is_empty() {
            break;
        }
        tail = remaining;
    }
    out
}

pub fn expand_presets(
    rest: &str,
    presets: &std::collections::HashMap<String, String>,
    only_presets: bool,
) -> Result<String, ParseError> {
    let mut out = String::with_capacity(rest.len());
    let mut tail = rest;
    loop {
        let (seg, remaining) = match tail.split_once('/') {
            Some((s, r)) => (s, r),
            None => (tail, ""),
        };
        if let Some(name) = seg.strip_prefix("preset:") {
            let definition = presets
                .get(name)
                .ok_or_else(|| ParseError::UnknownPreset(name.to_string()))?;
            out.push_str(definition);
        } else if seg.starts_with("exp:") || parse_operation(seg).is_some_and(|r| r.is_ok()) {
            if only_presets && !seg.starts_with("exp:") {
                return Err(ParseError::PresetsOnly(seg.to_string()));
            }
            out.push_str(seg);
        } else {
            // Source URL (or a malformed operation, whose error
            // the real parse will produce): copy the rest of
            // the path verbatim and stop.
            out.push_str(tail);
            return Ok(out);
        }
        if remaining.is_empty() {
            return Ok(out);
        }
        out.push('/');
        tail = remaining;
    }
}

/// The output format a request names explicitly, if any: the
/// last `format(...)` filter wins, matching how the engine reads
/// directives. Shared by the engine and the server (whose cache
/// key must know whether negotiation can change the answer).
pub fn explicit_format(ops: &[Operation]) -> Option<ImageFormat> {
    ops.iter().rev().find_map(|op| match op {
        Operation::Filters(filters) => filters.iter().rev().find_map(|f| match f {
            Filter::Format(fmt) => Some(*fmt),
            _ => None,
        }),
        _ => None,
    })
}

/// Whether any filters segment in `ops` requests `no_upscale`.
pub fn has_no_upscale(ops: &[Operation]) -> bool {
    ops.iter().any(|op| match op {
        Operation::Filters(filters) => filters.iter().any(|f| matches!(f, Filter::NoUpscale)),
        _ => false,
    })
}

/// The request's `fill(...)` directive color: the last one in
/// the request wins.
pub fn explicit_fill(ops: &[Operation]) -> Option<[u8; 3]> {
    ops.iter().rev().find_map(|op| match op {
        Operation::Filters(filters) => filters.iter().rev().find_map(|f| match f {
            Filter::Fill(color) => Some(*color),
            _ => None,
        }),
        _ => None,
    })
}

/// Whether any filters segment in `ops` requests `removebg`.
pub fn has_removebg(ops: &[Operation]) -> bool {
    ops.iter().any(|op| match op {
        Operation::Filters(filters) => filters.iter().any(|f| matches!(f, Filter::RemoveBg)),
        _ => false,
    })
}

/// The first shape filter in `ops` that produces transparency
/// (`radius` or `mask`), named for error messages.
pub fn shape_filter(ops: &[Operation]) -> Option<&'static str> {
    ops.iter().find_map(|op| match op {
        Operation::Filters(filters) => filters.iter().find_map(|f| match f {
            Filter::Radius(_) => Some("radius"),
            Filter::Mask(_) => Some("mask"),
            _ => None,
        }),
        _ => None,
    })
}

/// Whether any filters segment in `ops` requests `upscale`.
pub fn has_upscale(ops: &[Operation]) -> bool {
    ops.iter().any(|op| match op {
        Operation::Filters(filters) => filters.iter().any(|f| matches!(f, Filter::Upscale(_))),
        _ => false,
    })
}

/// Whether any operation in `ops` is a smart-aligned crop
/// (which may run face detection when one is configured).
pub fn has_smart_crop(ops: &[Operation]) -> bool {
    ops.iter().any(|op| {
        matches!(
            op,
            Operation::Crop {
                align: Align::Smart,
                ..
            }
        )
    })
}

/// The `text(...)` color field: `rrggbb` (opaque) or
/// `rrggbbaa`, as RGBA bytes.
fn parse_text_color(s: &str) -> Option<[u8; 4]> {
    if !matches!(s.len(), 6 | 8) {
        return None;
    }
    let v = u32::from_str_radix(s, 16).ok()?;
    if s.len() == 6 {
        Some([(v >> 16) as u8, (v >> 8) as u8, v as u8, 255])
    } else {
        Some([(v >> 24) as u8, (v >> 16) as u8, (v >> 8) as u8, v as u8])
    }
}

/// The request's explicit DPR factor, if a `dpr` directive is
/// present. Like format and quality it is request-wide: the
/// last segment that sets it wins.
pub fn explicit_dpr(ops: &[Operation]) -> Option<f32> {
    let mut found = None;
    for op in ops {
        if let Operation::Filters(filters) = op {
            for f in filters {
                if let Filter::Dpr(factor) = f {
                    found = Some(*factor);
                }
            }
        }
    }
    found
}

/// Whether `ops` fix the output dimensions themselves (a
/// size, crop, fit-in, pad, or crop-rect operation). A
/// width client hint only applies when this is false.
pub fn has_size_operation(ops: &[Operation]) -> bool {
    ops.iter().any(|op| {
        matches!(
            op,
            Operation::Size(_)
                | Operation::Crop { .. }
                | Operation::FitIn(_)
                | Operation::Pad { .. }
                | Operation::CropRect { .. }
        )
    })
}

/// Whether `ops` ask to keep the source color profile.
pub fn has_keep_profile(ops: &[Operation]) -> bool {
    ops.iter().any(|op| match op {
        Operation::Filters(filters) => filters.iter().any(|f| matches!(f, Filter::KeepProfile)),
        _ => false,
    })
}

/// Whether `ops` carry any `quality(...)` directive, fixed
/// or auto. Save-Data only steps quality when this is false.
pub fn has_quality_directive(ops: &[Operation]) -> bool {
    ops.iter().any(|op| match op {
        Operation::Filters(filters) => filters.iter().any(|f| matches!(f, Filter::Quality(_))),
        _ => false,
    })
}

/// Whether any filters segment in `ops` requests `text`.
pub fn has_text(ops: &[Operation]) -> bool {
    ops.iter().any(|op| match op {
        Operation::Filters(filters) => filters.iter().any(|f| matches!(f, Filter::Text { .. })),
        _ => false,
    })
}

/// Whether the request uses an operation that requires the
/// face model: a face-aligned crop, or a face redaction
/// filter. (Smart alignment does not count: it falls back to
/// saliency without a model.)
pub fn needs_face_model(ops: &[Operation]) -> bool {
    ops.iter().any(|op| match op {
        Operation::Crop { align, .. } => *align == Align::Face,
        Operation::Filters(filters) => filters
            .iter()
            .any(|f| matches!(f, Filter::BlurFaces | Filter::PixelateFaces)),
        _ => false,
    })
}

/// Whether any filters segment in `ops` requests a
/// class redaction (`blurclass` / `pixelateclass`), which
/// needs the detection model and the AI gate.
pub fn has_class_redaction(ops: &[Operation]) -> bool {
    !class_filters(ops).is_empty()
}

/// The class names requested by `ops`' class redaction
/// filters, in request order.
pub fn class_filters(ops: &[Operation]) -> Vec<&str> {
    let mut out = Vec::new();
    for op in ops {
        if let Operation::Filters(filters) = op {
            for f in filters {
                match f {
                    Filter::BlurClass(name) | Filter::PixelateClass(name) => {
                        out.push(name.as_str())
                    }
                    _ => {}
                }
            }
        }
    }
    out
}

/// Padding for a face-aligned crop: the last `facepad(...)`
/// directive in the request, as a fraction of the face size
/// added on every side. Defaults to 0.5.
pub fn face_padding(ops: &[Operation]) -> f32 {
    ops.iter()
        .rev()
        .find_map(|op| match op {
            Operation::Filters(filters) => filters.iter().rev().find_map(|f| match f {
                Filter::FacePad(p) => Some(*p),
                _ => None,
            }),
            _ => None,
        })
        .unwrap_or(0.5)
}

/// The distinct watermark and overlay URLs a request
/// references, in order of first appearance. The server
/// fetches each one before the engine runs.
pub fn watermark_urls(ops: &[Operation]) -> Vec<&str> {
    let mut urls: Vec<&str> = Vec::new();
    for op in ops {
        if let Operation::Filters(filters) = op {
            for filter in filters {
                let url = match filter {
                    Filter::Watermark { url, .. } => url,
                    Filter::Overlay { url, .. } => url,
                    _ => continue,
                };
                if !urls.contains(&url.as_str()) {
                    urls.push(url.as_str());
                }
            }
        }
    }
    urls
}

/// Decode `%XX` escapes; other characters pass through. A
/// malformed escape or non-UTF-8 result is an error.
fn percent_decode(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = s.get(i + 1..i + 3)?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

fn hex3(c: [u8; 3]) -> String {
    format!("{:02x}{:02x}{:02x}", c[0], c[1], c[2])
}

/// Percent-encode a decoded filter argument (a watermark
/// or overlay URL, a class name, a text label) for the
/// canonical form: everything outside the unreserved set
/// is encoded, so the encoding is deterministic and
/// `percent_decode` restores the exact string.
fn encode_arg(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for &b in text.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

fn dim_text(d: Dimension, flip: bool) -> String {
    let sign = if flip { "-" } else { "" };
    match d {
        // "0" itself parses as Proportional; a multi-digit
        // zero is the only spelling that parses back to
        // Pixels(0), which the engine treats as an exact
        // (if degenerate) size, not as proportional.
        Dimension::Pixels(0) => format!("{sign}00"),
        Dimension::Pixels(n) => format!("{sign}{n}"),
        Dimension::Proportional => format!("{sign}0"),
        Dimension::Original => format!("{sign}orig"),
    }
}

fn size_text(size: &ImageSize) -> String {
    format!(
        "{}x{}",
        dim_text(size.width, size.flip_horizontal),
        dim_text(size.height, size.flip_vertical)
    )
}

fn align_text(align: Align) -> Option<&'static str> {
    match align {
        Align::Center => None,
        Align::Smart => Some("smart"),
        Align::Face => Some("face"),
        Align::Top => Some("top"),
        Align::Bottom => Some("bottom"),
        Align::Left => Some("left"),
        Align::Right => Some("right"),
    }
}

fn anchor_text(anchor: Anchor) -> &'static str {
    match anchor {
        Anchor::TopLeft => "top-left",
        Anchor::TopCenter => "top-center",
        Anchor::TopRight => "top-right",
        Anchor::CenterLeft => "center-left",
        Anchor::Center => "center",
        Anchor::CenterRight => "center-right",
        Anchor::BottomLeft => "bottom-left",
        Anchor::BottomCenter => "bottom-center",
        Anchor::BottomRight => "bottom-right",
    }
}

fn format_text(format: ImageFormat) -> &'static str {
    match format {
        ImageFormat::Jpeg => "jpeg",
        ImageFormat::Png => "png",
        ImageFormat::WebP => "webp",
        ImageFormat::Gif => "gif",
        ImageFormat::Avif => "avif",
        ImageFormat::Jxl => "jxl",
    }
}

/// The one canonical spelling of a filter: the grammar
/// text the parser produces this exact value from.
fn filter_text(filter: &Filter) -> String {
    match filter {
        Filter::Grayscale => "grayscale".to_string(),
        Filter::Invert => "invert".to_string(),
        Filter::Blur(v) => format!("blur({v})"),
        Filter::Brighten(v) => format!("brighten({v})"),
        Filter::Contrast(v) => format!("contrast({v})"),
        Filter::HueRotate(v) => format!("huerotate({v})"),
        Filter::Sharpen(v) => format!("sharpen({v})"),
        Filter::Saturation(v) => format!("saturation({v})"),
        Filter::Rgb(r, g, b) => format!("rgb({r},{g},{b})"),
        Filter::Equalize => "equalize".to_string(),
        Filter::Rotate(v) => format!("rotate({v})"),
        Filter::Trim(v) => format!("trim({v})"),
        Filter::Watermark { url, x, y, alpha } => {
            format!("watermark({},{x},{y},{alpha})", encode_arg(url))
        }
        Filter::Overlay {
            url,
            x,
            y,
            alpha,
            blend,
            width,
        } => {
            let blend = match blend {
                BlendMode::Normal => "normal",
                BlendMode::Multiply => "multiply",
                BlendMode::Screen => "screen",
                BlendMode::Overlay => "overlay",
            };
            let width = width.map(|w| format!(",{w}")).unwrap_or_default();
            format!(
                "overlay({},{x},{y},{alpha},{blend}{width})",
                encode_arg(url)
            )
        }
        Filter::RemoveBg => "removebg".to_string(),
        Filter::KeepProfile => "keep_profile".to_string(),
        Filter::BlurFaces => "blurfaces".to_string(),
        Filter::PixelateFaces => "pixelatefaces".to_string(),
        Filter::BlurClass(name) => format!("blurclass({})", encode_arg(name)),
        Filter::PixelateClass(name) => format!("pixelateclass({})", encode_arg(name)),
        Filter::FacePad(v) => format!("facepad({v})"),
        Filter::Upscale(v) => format!("upscale({v})"),
        Filter::Dpr(v) => format!("dpr({v})"),
        Filter::Text {
            text,
            size,
            color,
            anchor,
        } => format!(
            "text({},{size},{:02x}{:02x}{:02x}{:02x},{})",
            encode_arg(text),
            color[0],
            color[1],
            color[2],
            color[3],
            anchor_text(*anchor)
        ),
        Filter::Fill(c) => format!("fill({})", hex3(*c)),
        Filter::Radius(r) => format!("radius({},{},{},{})", r[0], r[1], r[2], r[3]),
        Filter::Border(w, c) => format!("border({w},{})", hex3(*c)),
        Filter::Mask(shape) => format!(
            "mask({})",
            match shape {
                MaskShape::Circle => "circle",
                MaskShape::Ellipse => "ellipse",
            }
        ),
        Filter::Sepia => "sepia".to_string(),
        Filter::Duotone(a, b) => format!("duotone({},{})", hex3(*a), hex3(*b)),
        Filter::Monochrome(c) => format!("monochrome({})", hex3(*c)),
        Filter::Pixelate(v) => format!("pixelate({v})"),
        Filter::Flip(d) => format!(
            "flip({})",
            match d {
                FlipDirection::Horizontal => "h",
                FlipDirection::Vertical => "v",
            }
        ),
        Filter::BlurRegion { x, y, w, h } => format!("blur_region({x},{y},{w},{h})"),
        Filter::PixelateRegion { x, y, w, h } => {
            format!("pixelate_region({x},{y},{w},{h})")
        }
        Filter::MaxBytes(v) => format!("max_bytes({v})"),
        Filter::Quality(q) => match q {
            Quality::Auto => "quality(auto)".to_string(),
            Quality::Value(v) => format!("quality({v})"),
        },
        Filter::Format(f) => format!("format({})", format_text(*f)),
        Filter::NoUpscale => "no_upscale".to_string(),
    }
}

fn operation_text(op: &Operation) -> String {
    match op {
        Operation::Size(size) => size_text(size),
        Operation::Crop { size, align } => match align_text(*align) {
            Some(name) => format!("crop:{}:{name}", size_text(size)),
            None => format!("crop:{}", size_text(size)),
        },
        Operation::FitIn(size) => format!("fit-in:{}", size_text(size)),
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
    let mut segments: Vec<String> = Vec::new();
    let mut filters: Vec<String> = Vec::new();
    for op in ops {
        match op {
            Operation::Filters(list) => {
                filters.extend(list.iter().map(filter_text));
            }
            other => {
                if !filters.is_empty() {
                    segments.push(format!("filters:{}", filters.join(":")));
                    filters.clear();
                }
                segments.push(operation_text(other));
            }
        }
    }
    if !filters.is_empty() {
        segments.push(format!("filters:{}", filters.join(":")));
    }
    if segments.is_empty() {
        source.to_string()
    } else {
        format!("{}/{source}", segments.join("/"))
    }
}

/// The cache key for a remainder: its canonical form.
/// Remainders that do not parse (purge targets naming
/// presets are expanded by the purge handler's own path;
/// opaque strings) fall back to the expiry-stripped
/// textual form.
pub fn canonical_key_form(remainder: &str) -> String {
    match parse_operations_with_expiry(remainder) {
        Ok((ops, source, _)) => canonical_form(&ops, source),
        Err(_) => cache_key_form(remainder),
    }
}

/// The cache-key form of an expanded remainder: identical,
/// except every `exp:` segment is removed (audit P-2).
/// Expiry is enforced before lookup, so it must not fragment
/// the cache or shield entries from purges. Unparseable
/// remainders pass through unchanged.
pub fn cache_key_form(remainder: &str) -> String {
    let Ok((_, source)) = parse_operations(remainder) else {
        return remainder.to_string();
    };
    let Some(ops_part) = remainder.strip_suffix(source) else {
        return remainder.to_string();
    };
    let kept: Vec<&str> = ops_part
        .split('/')
        .filter(|seg| !seg.is_empty() && !seg.starts_with("exp:"))
        .collect();
    if kept.is_empty() {
        source.to_string()
    } else {
        format!("{}/{source}", kept.join("/"))
    }
}

/// Parse a full request path into its signature and [`Request`].
#[allow(dead_code)]
pub fn parse_request(path: &str) -> Result<(Signature, Request<'_>), ParseError> {
    let (signature, rest) = parse_signature(path)?;
    let (operations, source_url) = parse_operations(rest)?;
    Ok((
        signature,
        Request {
            operations,
            source_url,
        },
    ))
}

/// Verify a request signature against the server secret.
///
/// `remainder` is the path after the signature segment (operations and
/// image URL, still slash-separated). The HMAC is computed over it with
/// the server secret and compared to the presented signature in
/// constant time. A signature that is not valid hex, or not the 64
/// characters a SHA-256 tag needs, fails without panicking.
///
/// [`Signature::Unsafe`] is honored only when `allow_unsafe` is true.
/// The server decides that from its own configuration; the secret
/// itself always comes from config or environment, never from code.
pub fn verify(secret: &[u8], signature: &Signature, remainder: &str, allow_unsafe: bool) -> bool {
    match signature {
        Signature::Unsafe => allow_unsafe,
        // The HMAC check itself lives in the `crabcrop-url`
        // builder crate, so the server verifies with exactly
        // the code clients sign with.
        Signature::Hmac(hex_digest) => crabcrop_url::verify(secret, hex_digest, remainder),
    }
}

/// Derive the purge signing key from the server secret.
///
/// Purge signatures live in their own domain (security audit
/// P-1): they are HMACs under this derived key, never under the
/// secret itself, so a signature minted for an image URL can
/// never be replayed against the purge endpoint, even when the
/// image remainder begins with `purge/`. Before this change the
/// purge scheme signed the literal `purge/<target>` with the
/// secret, which is exactly an image signature over a remainder
/// of that shape.
pub fn purge_key(secret: &[u8]) -> Vec<u8> {
    crabcrop_url::purge_key(secret)
}

/// The purge signature for a target path: HMAC-SHA256 of
/// the path alone, under the derived purge key. Delegates
/// to the `crabcrop-url` crate, the one implementation
/// clients and server share (Q6). The server only
/// verifies; this mints, for operators and tests.
#[allow(dead_code)]
pub fn purge_signature(secret: &[u8], target: &str) -> String {
    crabcrop_url::purge_signature(secret, target)
}

/// The purge-group signature for a cache group, over the
/// literal `purge-group/<group>` string. Delegates to the
/// crate helper so nobody hand-constructs the message.
#[allow(dead_code)]
pub fn purge_group_signature(secret: &[u8], group: &str) -> String {
    crabcrop_url::purge_group_signature(secret, group)
}

/// Verify a purge signature. Purges are the server's only
/// destructive operation, so they never honor `unsafe`
/// (review Q3): an HMAC under the purge subkey is always
/// required, whatever the image-read mode. A read-mode
/// flag must not silently grant them.
pub fn verify_purge(secret: &[u8], signature: &Signature, target: &str) -> bool {
    match signature {
        Signature::Unsafe => false,
        Signature::Hmac(hex_digest) => crabcrop_url::verify(&purge_key(secret), hex_digest, target),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_unsafe_signature() {
        let (sig, rest) = parse_signature("/unsafe/300x200/example.com/img.jpg").unwrap();
        assert_eq!(sig, Signature::Unsafe);
        assert_eq!(rest, "300x200/example.com/img.jpg");
    }

    #[test]
    fn parses_hmac_signature() {
        let (sig, rest) = parse_signature("/a1b2c3d4/300x200/example.com/img.jpg").unwrap();
        assert_eq!(sig, Signature::Hmac("a1b2c3d4".to_string()));
        assert_eq!(rest, "300x200/example.com/img.jpg");
    }

    #[test]
    fn signatures_have_one_canonical_spelling() {
        // Uppercase and non-hex spellings are parse errors
        // (audit PL-7); builders only ever emit lowercase hex.
        assert!(parse_signature("/A1B2C3D4/300x200/example.com/img.jpg").is_err());
        assert!(parse_signature("/not-a-sig/300x200/example.com/img.jpg").is_err());
        assert!(parse_signature("/a1b2c3d4/300x200/example.com/img.jpg").is_ok());
    }

    #[test]
    fn signature_is_case_sensitive_but_unsafe_is_lowercase_only() {
        // "Unsafe" is not the magic word, and under the
        // canonical-spelling rule (PL-7) it is not lowercase
        // hex either: a parse error, not an HMAC signature.
        assert!(parse_signature("/Unsafe/300x200/example.com/img.jpg").is_err());
        let (sig, _) = parse_signature("/unsafe/300x200/example.com/img.jpg").unwrap();
        assert_eq!(sig, Signature::Unsafe);
    }

    #[test]
    fn works_without_leading_slash() {
        let (sig, rest) = parse_signature("unsafe/300x200/example.com/img.jpg").unwrap();
        assert_eq!(sig, Signature::Unsafe);
        assert_eq!(rest, "300x200/example.com/img.jpg");
    }

    #[test]
    fn rejects_empty_paths() {
        for path in ["", "/"] {
            assert_eq!(
                parse_signature(path),
                Err(ParseError::EmptyPath),
                "path: {path:?}"
            );
        }
    }

    #[test]
    fn rejects_lone_segment() {
        // A signature with nothing after it is not a valid image request.
        for path in ["/unsafe", "unsafe", "/a1b2c3"] {
            assert_eq!(
                parse_signature(path),
                Err(ParseError::MissingSignature),
                "path: {path:?}"
            );
        }
    }

    #[test]
    fn rejects_empty_first_segment() {
        assert_eq!(
            parse_signature("//300x200/x.jpg"),
            Err(ParseError::EmptyPath)
        );
    }

    #[test]
    fn keeps_image_url_with_slashes_intact() {
        let (sig, rest) =
            parse_signature("/unsafe/filters:grayscale()/https://example.com/a/b/c.jpg").unwrap();
        assert_eq!(sig, Signature::Unsafe);
        assert_eq!(rest, "filters:grayscale()/https://example.com/a/b/c.jpg");
    }

    #[test]
    fn cache_key_form_strips_expiry_segments() {
        let src = "https://example.com/a.jpg";
        assert_eq!(
            cache_key_form(&format!("exp:1893456000/300x200/{src}")),
            format!("300x200/{src}")
        );
        assert_eq!(
            cache_key_form(&format!("300x200/exp:1893456000/{src}")),
            format!("300x200/{src}")
        );
        assert_eq!(cache_key_form(&format!("exp:1893456000/{src}")), src);
        assert_eq!(
            cache_key_form(&format!("300x200/{src}")),
            format!("300x200/{src}")
        );
        // An exp: segment inside the source is URL text, kept.
        let tricky = "https://example.com/exp:123/a.jpg";
        assert_eq!(cache_key_form(tricky), tricky);
    }

    #[test]
    fn parse_error_codes_are_the_frozen_v2_set() {
        let cases: Vec<(ParseError, &str)> = vec![
            (ParseError::EmptyPath, "empty_path"),
            (ParseError::MissingSignature, "missing_signature"),
            (ParseError::MalformedSignature, "malformed_signature"),
            (ParseError::InvalidSize("x".into()), "invalid_size"),
            (ParseError::InvalidCrop("x".into()), "invalid_crop"),
            (ParseError::InvalidCropRect("x".into()), "invalid_crop_rect"),
            (ParseError::InvalidPad("x".into()), "invalid_pad"),
            (ParseError::InvalidExpiry("x".into()), "invalid_expiry"),
            (ParseError::UnknownPreset("x".into()), "unknown_preset"),
            (ParseError::InvalidPreset("x".into()), "invalid_preset"),
            (ParseError::PresetsOnly("x".into()), "presets_only"),
            (ParseError::InvalidFilters("x".into()), "invalid_filters"),
            (ParseError::MissingSourceUrl, "missing_source_url"),
        ];
        let mut seen = std::collections::HashSet::new();
        for (err, expected) in &cases {
            assert_eq!(err.code(), *expected);
            assert!(seen.insert(err.code()), "codes must be unique");
        }
        assert_eq!(seen.len(), cases.len());
    }

    #[test]
    fn canonical_key_collapses_spelling_variants() {
        let src = "https://example.com/a.jpg";
        let pairs = [
            // Format alias.
            ("filters:format(jpg)", "filters:format(jpeg)"),
            // Leading zeros.
            ("0300x0200", "300x200"),
            // Omitted axis vs explicit 0.
            ("300x", "300x0"),
            // Hex case in colors.
            ("filters:fill(FFAA00)", "filters:fill(ffaa00)"),
            // Split filter segments merge.
            (
                "filters:grayscale/filters:blur(2)",
                "filters:grayscale:blur(2)",
            ),
            // The default crop alignment is no alignment.
            ("crop:300x200:center", "crop:300x200"),
            // Bare trim is trim(10).
            ("filters:trim", "filters:trim(10)"),
            // One radius is four.
            ("filters:radius(10)", "filters:radius(10,10,10,10)"),
            // Float spellings.
            ("filters:contrast(1.50)", "filters:contrast(1.5)"),
            // Six-digit text colors gain an opaque alpha.
            (
                "filters:text(Hi,20,ffaa00,center)",
                "filters:text(Hi,20,ffaa00ff,center)",
            ),
        ];
        for (a, b) in pairs {
            assert_eq!(
                canonical_key_form(&format!("{a}/{src}")),
                canonical_key_form(&format!("{b}/{src}")),
                "{a} and {b} must share a key"
            );
        }
        // Expiry never enters the key.
        assert_eq!(
            canonical_key_form(&format!("exp:1893456000/300x200/{src}")),
            canonical_key_form(&format!("300x200/{src}"))
        );
        // Order is meaning, not spelling: it stays in the key.
        assert_ne!(
            canonical_key_form(&format!("filters:blur(2):grayscale/{src}")),
            canonical_key_form(&format!("filters:grayscale:blur(2)/{src}"))
        );
        // Pixels(0) is an exact size, not proportional.
        assert_ne!(
            canonical_key_form(&format!("00x200/{src}")),
            canonical_key_form(&format!("0x200/{src}"))
        );
    }

    #[test]
    fn canonical_form_is_a_fixed_point_and_round_trips() {
        let src = "https://example.com/a.jpg";
        let corpus = [
            "300x200",
            "-300xorig",
            "crop:100x100:smart",
            "crop-rect:1x2:30x40",
            "fit-in:300x200",
            "pad:300x200",
            "pad:300x200:blur",
            "pad:300x200:A0B0C0",
            "filters:grayscale:invert:blur(1.5):brighten(-20):contrast(1.25)",
            "filters:huerotate(90):sharpen(0.5):saturation(2):rgb(1,-2,3):equalize",
            "filters:rotate(-90):trim(25):format(avif):quality(auto):quality(80)",
            "filters:watermark(https%3A%2F%2Fcdn.example.com%2Flogo.png,10,-10,80)",
            "filters:overlay(https%3A%2F%2Fe.com%2Fo.png,0,0,50,multiply,25)",
            "filters:removebg:keep_profile:blurfaces:pixelatefaces:no_upscale",
            "filters:blurclass(person):pixelateclass(license%20plate):facepad(0.5)",
            "filters:upscale(2):dpr(1.5):max_bytes(50000)",
            "filters:text(Hello%2C%20world,24,FFAA00CC,bottom-right)",
            "filters:fill(00ff00):radius(1,2,3,4):border(3,102030):mask(ellipse)",
            "filters:sepia:duotone(101010,f0e0c0):monochrome(336699):pixelate(8)",
            "filters:flip(h):blur_region(1,2,30,40):pixelate_region(5,6,70,80)",
            "300x200/filters:grayscale",
            "filters:grayscale/300x200/filters:blur(2)",
        ];
        for text in corpus {
            let remainder = format!("{text}/{src}");
            let canonical = canonical_key_form(&remainder);
            assert_eq!(
                canonical_key_form(&canonical),
                canonical,
                "canonical form of {text} is not a fixed point"
            );
            let (ops_before, _, _) = parse_operations_with_expiry(&remainder).unwrap();
            let (ops_after, _, _) = parse_operations_with_expiry(&canonical).unwrap();
            assert_eq!(ops_before, ops_after, "canonical form changed {text}");
        }
    }

    #[test]
    fn purge_signatures_are_domain_separated() {
        let secret = b"test-secret";
        let target = "300x200/https://example.com/a.jpg";
        let purge_sig = purge_signature(secret, target);
        // The replay from audit P-1: an image signature over
        // "purge/<target>" must not verify as a purge signature.
        let image_sig = crabcrop_url::sign(secret, &format!("purge/{target}"));
        assert_ne!(purge_sig, image_sig);
        assert!(verify_purge(secret, &Signature::Hmac(purge_sig), target));
        assert!(!verify_purge(secret, &Signature::Hmac(image_sig), target));
        // Purges never honor unsafe (Q3), in any mode.
        assert!(!verify_purge(secret, &Signature::Unsafe, target));
    }

    #[test]
    fn dimensions_are_capped_at_parse_time() {
        assert!(parse_size("16383x100").is_ok());
        assert!(parse_size("16384x100").is_err());
        assert!(parse_size("100x16384").is_err());
        // The cap applies wherever a size parses.
        assert!(crate::url::parse_operations("fit-in:16384x10").is_err());
        assert!(crate::url::parse_operations("pad:20000x20000").is_err());
        assert!(crate::url::parse_operations("crop:16384x10").is_err());
    }

    #[test]
    fn parses_sizes() {
        use Dimension::{Original, Pixels, Proportional};

        let size = |width, height, flip_horizontal, flip_vertical| ImageSize {
            width,
            height,
            flip_horizontal,
            flip_vertical,
        };
        let plain = |width, height| size(width, height, false, false);

        let cases = [
            ("300x200", plain(Pixels(300), Pixels(200))),
            ("-300x200", size(Pixels(300), Pixels(200), true, false)),
            ("300x-200", size(Pixels(300), Pixels(200), false, true)),
            ("-300x-200", size(Pixels(300), Pixels(200), true, true)),
            ("300x0", plain(Pixels(300), Proportional)),
            ("300x", plain(Pixels(300), Proportional)),
            ("x200", plain(Proportional, Pixels(200))),
            ("0x200", plain(Proportional, Pixels(200))),
            ("0x0", plain(Proportional, Proportional)),
            ("origx100", plain(Original, Pixels(100))),
            ("100xorig", plain(Pixels(100), Original)),
            ("origxorig", plain(Original, Original)),
            ("-0x-0", size(Proportional, Proportional, true, true)),
        ];
        for (input, expected) in cases {
            assert_eq!(parse_size(input), Ok(expected), "input: {input:?}");
        }
    }

    #[test]
    fn rejects_invalid_sizes() {
        for input in ["", "300", "300xx200", "abcx200", "300xabc", "12.5x100"] {
            assert_eq!(
                parse_size(input),
                Err(ParseError::InvalidSize(input.to_string())),
                "input: {input:?}"
            );
        }
    }

    #[test]
    fn signature_then_size_compose() {
        // The two primitives chain: signature off the front, size next.
        let (sig, rest) = parse_signature("/unsafe/300x200/example.com/img.jpg").unwrap();
        assert_eq!(sig, Signature::Unsafe);
        let (size_seg, image) = rest.split_once('/').unwrap();
        let size = parse_size(size_seg).unwrap();
        assert_eq!(size.width, Dimension::Pixels(300));
        assert_eq!(size.height, Dimension::Pixels(200));
        assert_eq!(image, "example.com/img.jpg");
    }

    /// HMAC-SHA256 with secret `Jefe` over `KNOWN_REMAINDER`.
    /// Cross-checked against Python's hmac module and OpenSSL before
    /// it was written into this test.
    const KNOWN_HMAC: &str = "ba5396e10039ec002bf1fde117eee48df8c2f06bc47b5f2d3272ea77f6862da9";
    const KNOWN_REMAINDER: &str = "what do ye want for nothing?";

    fn known_signature() -> Signature {
        Signature::Hmac(KNOWN_HMAC.to_string())
    }

    #[test]
    fn verify_accepts_valid_hmac_sha256() {
        assert!(verify(b"Jefe", &known_signature(), KNOWN_REMAINDER, false));
    }

    #[test]
    fn verify_rejects_tampered_remainder() {
        assert!(!verify(
            b"Jefe",
            &known_signature(),
            "what do ye want for something?",
            false
        ));
    }

    #[test]
    fn verify_rejects_wrong_secret() {
        assert!(!verify(
            b"NotJefe",
            &known_signature(),
            KNOWN_REMAINDER,
            false
        ));
    }

    #[test]
    fn verify_unsafe_honored_only_when_enabled() {
        assert!(verify(b"secret", &Signature::Unsafe, "anything", true));
        assert!(!verify(b"secret", &Signature::Unsafe, "anything", false));
    }

    #[test]
    fn verify_rejects_malformed_signatures_without_panicking() {
        for bad in [
            "",
            "abc",      // odd length, not decodable as hex bytes
            "zz",       // not hex at all
            "ba5396e1", // valid hex, wrong length
            "ba5396e10039ec002bf1fde117eee48df8c2f06bc47b5f2d3272ea77f6862da", // 63 chars, odd
        ] {
            let sig = Signature::Hmac(bad.to_string());
            assert!(
                !verify(b"Jefe", &sig, KNOWN_REMAINDER, false),
                "signature: {bad:?}"
            );
        }
    }

    #[test]
    fn parsed_signature_verifies_against_remainder() {
        let path = format!("/{KNOWN_HMAC}/{KNOWN_REMAINDER}");
        let (sig, rest) = parse_signature(&path).unwrap();
        assert!(verify(b"Jefe", &sig, rest, false));
    }

    fn size_op(width: u32, height: u32) -> Operation {
        Operation::Size(ImageSize {
            width: Dimension::Pixels(width),
            height: Dimension::Pixels(height),
            flip_horizontal: false,
            flip_vertical: false,
        })
    }

    #[test]
    fn parses_size_operation() {
        let (ops, url) = parse_operations("300x200/example.com/img.jpg").unwrap();
        assert_eq!(ops, vec![size_op(300, 200)]);
        assert_eq!(url, "example.com/img.jpg");
    }

    #[test]
    fn parses_crop_with_default_center() {
        let (ops, _) = parse_operations("crop:300x200/example.com/img.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::Crop {
                size: ImageSize {
                    width: Dimension::Pixels(300),
                    height: Dimension::Pixels(200),
                    flip_horizontal: false,
                    flip_vertical: false,
                },
                align: Align::Center,
            }]
        );
    }

    #[test]
    fn parses_crop_with_alignment() {
        let (ops, _) = parse_operations("crop:300x200:top/example.com/img.jpg").unwrap();
        assert!(matches!(
            ops[0],
            Operation::Crop {
                align: Align::Top,
                ..
            }
        ));
    }

    #[test]
    fn rejects_bad_crops() {
        for seg in [
            "crop:",
            "crop:300",
            "crop:300x200:sideways",
            "crop:300x200:center:extra",
            // The exact-box contract: pixels on both axes,
            // no flips (fit-in owns proportional boxes, and
            // a flip is filters:flip).
            "crop:300x0",
            "crop:0x200",
            "crop:origx100",
            "crop:300xorig",
            "crop:-300x200",
            "crop:300x-200",
            "crop:00x200",
        ] {
            assert_eq!(
                parse_operations(&format!("{seg}/example.com/img.jpg")),
                Err(ParseError::InvalidCrop(seg.to_string())),
                "segment: {seg:?}"
            );
        }
    }

    #[test]
    fn parses_filters() {
        let (ops, _) =
            parse_operations("filters:grayscale():quality(80):format(webp)/example.com/img.jpg")
                .unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![
                Filter::Grayscale,
                Filter::Quality(Quality::Value(80)),
                Filter::Format(ImageFormat::WebP),
            ])]
        );
    }

    #[test]
    fn parses_shape_filters() {
        let (ops, _) = parse_operations(
            "filters:radius(12):border(2,00ff00):mask(ellipse)/example.com/img.jpg",
        )
        .unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![
                Filter::Radius([12, 12, 12, 12]),
                Filter::Border(2, [0, 255, 0]),
                Filter::Mask(MaskShape::Ellipse),
            ])]
        );
        let (ops, _) =
            parse_operations("filters:radius(1,2,3,4):mask(circle)/example.com/img.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![
                Filter::Radius([1, 2, 3, 4]),
                Filter::Mask(MaskShape::Circle),
            ])]
        );
    }

    #[test]
    fn parses_finishing_filters() {
        let (ops, _) = parse_operations(
            "filters:sepia:duotone(000000,ffffff):monochrome(ff8000):pixelate(8):flip(h)/example.com/img.jpg",
        )
        .unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![
                Filter::Sepia,
                Filter::Duotone([0, 0, 0], [255, 255, 255]),
                Filter::Monochrome([255, 128, 0]),
                Filter::Pixelate(8),
                Filter::Flip(FlipDirection::Horizontal),
            ])]
        );
    }

    #[test]
    fn parses_region_filters() {
        let (ops, _) = parse_operations(
            "filters:blur_region(1,2,30,40):pixelate_region(0,0,5,5)/example.com/img.jpg",
        )
        .unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![
                Filter::BlurRegion {
                    x: 1,
                    y: 2,
                    w: 30,
                    h: 40
                },
                Filter::PixelateRegion {
                    x: 0,
                    y: 0,
                    w: 5,
                    h: 5
                },
            ])]
        );
    }

    #[test]
    fn parses_overlay() {
        let (ops, _) = parse_operations(
            "filters:overlay(https%3A%2F%2Fcdn.example.com%2Fov.png,10,-5,80,multiply)/example.com/img.jpg",
        )
        .unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![Filter::Overlay {
                url: "https://cdn.example.com/ov.png".to_string(),
                x: 10,
                y: -5,
                alpha: 80,
                blend: BlendMode::Multiply,
                width: None,
            }])]
        );
        // Width form, and a comma inside the URL itself.
        let (ops, _) = parse_operations(
            "filters:overlay(https%3A%2F%2Fx%2Fa,b.png,0,0,100,screen,25)/example.com/img.jpg",
        )
        .unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![Filter::Overlay {
                url: "https://x/a,b.png".to_string(),
                x: 0,
                y: 0,
                alpha: 100,
                blend: BlendMode::Screen,
                width: Some(25),
            }])]
        );
        // Overlay URLs join the prefetch set.
        let (ops, _) = parse_operations(
            "filters:overlay(https%3A%2F%2Fx%2Fov.png,0,0,100,normal)/example.com/img.jpg",
        )
        .unwrap();
        assert_eq!(watermark_urls(&ops), vec!["https://x/ov.png"]);
    }

    #[test]
    fn rejects_bad_filters() {
        for seg in [
            "filters:",
            "filters:blur()",
            "filters:quality(200)",
            "filters:quality()",
            "filters:quality(autoo)",
            "filters:grayscale(1)",
            "filters:format(bmp)",
            "filters:quality(80",
            "filters:radius()",
            "filters:radius(1,2)",
            "filters:radius(1,2,3)",
            "filters:radius(16384)",
            "filters:radius(-1)",
            "filters:border(0,FF0000)",
            "filters:border(2,FFF)",
            "filters:border(2)",
            "filters:border(2,FF0000,1)",
            "filters:mask(square)",
            "filters:mask()",
            "filters:sepia(1)",
            "filters:duotone(FF0000)",
            "filters:duotone(FF0000,FFF)",
            "filters:monochrome(FFF)",
            "filters:pixelate(0)",
            "filters:pixelate(16384)",
            "filters:flip(x)",
            "filters:flip()",
            "filters:blur_region(1,2,0,40)",
            "filters:blur_region(1,2,3)",
            "filters:blur_region(1,2,3,4,5)",
            "filters:blur_region(-1,0,10,10)",
            "filters:pixelate_region(0,0,10,16384)",
            "filters:overlay(https%3A%2F%2Fx%2Fov.png,0,0,101,normal)",
            "filters:overlay(https%3A%2F%2Fx%2Fov.png,0,0,100,darken)",
            "filters:overlay(https%3A%2F%2Fx%2Fov.png,0,0,100,normal,0)",
            "filters:overlay(https%3A%2F%2Fx%2Fov.png,0,0,100,normal,401)",
            "filters:overlay(https%3A%2F%2Fx%2Fov.png,0,0,100)",
            "filters:overlay(,0,0,100,normal)",
        ] {
            assert_eq!(
                parse_operations(&format!("{seg}/example.com/img.jpg")),
                Err(ParseError::InvalidFilters(seg.to_string())),
                "segment: {seg:?}"
            );
        }
    }

    #[test]
    fn parses_dpr_directive() {
        let (ops, _) = parse_operations("filters:dpr(2)/example.com/i.png").unwrap();
        assert_eq!(ops, vec![Operation::Filters(vec![Filter::Dpr(2.0)])]);
        assert_eq!(explicit_dpr(&ops), Some(2.0));
        for bad in [
            "filters:dpr(1.25)", // not one of the settled factors
            "filters:dpr(0)",
            "filters:dpr(4)",
            "filters:dpr(abc)",
            "filters:dpr", // the factor is required
        ] {
            let path = format!("{bad}/example.com/i.png");
            assert!(
                matches!(parse_operations(&path), Err(ParseError::InvalidFilters(_))),
                "input: {bad}"
            );
        }
    }

    #[test]
    fn parses_text_with_encoded_unicode_and_commas() {
        let (ops, _) = parse_operations(
            "filters:text(Hello%2C%20World!%20%E2%9C%93,32,ff8800,bottom-right)/example.com/i.png",
        )
        .unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![Filter::Text {
                text: "Hello, World! ✓".to_string(),
                size: 32,
                color: [255, 136, 0, 255],
                anchor: Anchor::BottomRight,
            }])]
        );
        // An 8-digit color carries its own alpha.
        let (ops, _) =
            parse_operations("filters:text(X,10,ff880080,center)/example.com/i.png").unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![Filter::Text {
                text: "X".to_string(),
                size: 10,
                color: [255, 136, 0, 128],
                anchor: Anchor::Center,
            }])]
        );
    }

    #[test]
    fn rejects_bad_text_params() {
        for bad in [
            "filters:text(X,0,ff8800,center)",    // size 0
            "filters:text(X,1001,ff8800,center)", // size too big
            "filters:text(X,10,ff88,center)",     // short color
            "filters:text(X,10,zzzzzz,center)",   // non-hex color
            "filters:text(X,10,ff8800,middle)",   // unknown anchor
            "filters:text(X,10,ff8800)",          // missing field
            "filters:text(,10,ff8800,center)",    // empty text
            "filters:text(%zz,10,ff8800,center)", // bad encoding
        ] {
            let path = format!("{bad}/example.com/i.png");
            assert!(
                matches!(parse_operations(&path), Err(ParseError::InvalidFilters(_))),
                "input: {bad}"
            );
        }
    }

    #[test]
    fn parses_jxl_format() {
        let (ops, _) = parse_operations("filters:format(jxl)/example.com/img.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![Filter::Format(ImageFormat::Jxl)])]
        );
        // `removebg` parses, and takes no parameters.
        let (ops, _) = parse_operations("filters:removebg/example.com/img.png").unwrap();
        assert_eq!(ops, vec![Operation::Filters(vec![Filter::RemoveBg])]);
        // `upscale` takes a factor of 2 or 4, nothing else.
        let (ops, _) = parse_operations("filters:upscale(2)/example.com/img.png").unwrap();
        assert_eq!(ops, vec![Operation::Filters(vec![Filter::Upscale(2)])]);
        let (ops, _) = parse_operations("filters:upscale(4)/example.com/img.png").unwrap();
        assert_eq!(ops, vec![Operation::Filters(vec![Filter::Upscale(4)])]);
        assert!(parse_operations("filters:upscale(3)/example.com/img.png").is_err());
        assert!(parse_operations("filters:upscale(0)/example.com/img.png").is_err());
        assert!(parse_operations("filters:upscale/example.com/img.png").is_err());
    }

    #[test]
    fn parses_auto_quality() {
        let (ops, _) = parse_operations("filters:quality(auto)/example.com/img.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![Filter::Quality(Quality::Auto)])]
        );
    }

    #[test]
    fn parses_avif_format() {
        let (ops, _) = parse_operations("filters:format(avif)/example.com/img.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![Filter::Format(ImageFormat::Avif)])]
        );
    }

    #[test]
    fn parses_fit_in() {
        let (ops, _) = parse_operations("fit-in:300x200/example.com/img.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::FitIn(ImageSize {
                width: Dimension::Pixels(300),
                height: Dimension::Pixels(200),
                flip_horizontal: false,
                flip_vertical: false,
            })]
        );
        let (ops, _) = parse_operations("fit-in:300x0/example.com/img.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::FitIn(ImageSize {
                width: Dimension::Pixels(300),
                height: Dimension::Proportional,
                flip_horizontal: false,
                flip_vertical: false,
            })]
        );
    }

    #[test]
    fn rejects_bad_fit_in() {
        for seg in ["fit-in:-300x200", "fit-in:wide", "fit-in:"] {
            let path = format!("{seg}/example.com/img.jpg");
            assert!(parse_operations(&path).is_err(), "{seg}");
        }
    }

    #[test]
    fn parses_crop_rect() {
        let (ops, _) = parse_operations("crop-rect:10x20:110x220/example.com/img.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::CropRect {
                left: 10,
                top: 20,
                right: 110,
                bottom: 220,
            }]
        );
    }

    #[test]
    fn rejects_bad_crop_rect() {
        for seg in [
            "crop-rect:10x20",
            "crop-rect:110x220:10x20",
            "crop-rect:10x20:10x220",
            "crop-rect:10x20:110x20",
            "crop-rect:-1x0:10x10",
            "crop-rect:axbx:cxdx",
        ] {
            let path = format!("{seg}/example.com/img.jpg");
            assert!(
                matches!(parse_operations(&path), Err(ParseError::InvalidCropRect(_))),
                "{seg}"
            );
        }
    }

    #[test]
    fn parses_no_upscale() {
        let (ops, _) = parse_operations("filters:no_upscale/example.com/img.jpg").unwrap();
        assert_eq!(ops, vec![Operation::Filters(vec![Filter::NoUpscale])]);
    }

    #[test]
    fn parses_watermark() {
        let (ops, _) = parse_operations(
            "filters:watermark(https%3A%2F%2Fcdn.example.com%2Fwm.png,10,-5,80)/example.com/img.jpg",
        )
        .unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![Filter::Watermark {
                url: "https://cdn.example.com/wm.png".to_string(),
                x: 10,
                y: -5,
                alpha: 80,
            }])]
        );
    }

    #[test]
    fn rejects_bad_watermarks() {
        for seg in [
            "watermark(http%3A%2F%2Fx%2Fwm.png,0,0,101)",
            "watermark(http%3A%2F%2Fx%2Fwm.png,0,0)",
            "watermark(%zz,0,0,50)",
            "watermark(,0,0,50)",
        ] {
            let path = format!("filters:{seg}/example.com/img.jpg");
            assert!(parse_operations(&path).is_err(), "{seg}");
        }
        // `removebg` accepts no parameters.
        assert!(parse_operations("filters:removebg(1)/example.com/img.png").is_err());
    }

    #[test]
    fn watermark_urls_are_distinct_and_ordered() {
        let (ops, _) = parse_operations(
            "filters:watermark(a%2F1.png,0,0,50):watermark(a%2F1.png,5,5,50):watermark(b%2F2.png,0,0,50)/example.com/img.jpg",
        )
        .unwrap();
        assert_eq!(watermark_urls(&ops), vec!["a/1.png", "b/2.png"]);
    }

    #[test]
    fn parses_trim() {
        let (ops, _) = parse_operations("filters:trim/example.com/img.jpg").unwrap();
        assert_eq!(ops, vec![Operation::Filters(vec![Filter::Trim(10)])]);
        let (ops, _) = parse_operations("filters:trim(0)/example.com/img.jpg").unwrap();
        assert_eq!(ops, vec![Operation::Filters(vec![Filter::Trim(0)])]);
        assert!(parse_operations("filters:trim(256)/example.com/img.jpg").is_err());
        assert!(parse_operations("filters:trim(-1)/example.com/img.jpg").is_err());
    }

    #[test]
    fn parses_gif_format() {
        let (ops, _) = parse_operations("filters:format(gif)/example.com/img.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![Filter::Format(ImageFormat::Gif)])]
        );
    }

    #[test]
    fn parses_crop_with_smart_alignment() {
        let (ops, _) = parse_operations("crop:300x200:smart/example.com/img.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::Crop {
                size: ImageSize {
                    width: Dimension::Pixels(300),
                    height: Dimension::Pixels(200),
                    flip_horizontal: false,
                    flip_vertical: false,
                },
                align: Align::Smart,
            }]
        );
    }

    #[test]
    fn parses_face_operations() {
        let (ops, _) = parse_operations("crop:300x300:face/example.com/img.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::Crop {
                size: ImageSize {
                    width: Dimension::Pixels(300),
                    height: Dimension::Pixels(300),
                    flip_horizontal: false,
                    flip_vertical: false,
                },
                align: Align::Face,
            }]
        );
        assert!(needs_face_model(&ops));
        assert_eq!(face_padding(&ops), 0.5);

        let (ops, _) =
            parse_operations("filters:blurfaces:pixelatefaces:facepad(1.5)/example.com/img.jpg")
                .unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![
                Filter::BlurFaces,
                Filter::PixelateFaces,
                Filter::FacePad(1.5),
            ])]
        );
        assert!(needs_face_model(&ops));
        assert_eq!(face_padding(&ops), 1.5);

        // Smart crops and plain requests do not require a model.
        let (ops, _) = parse_operations("crop:300x300:smart/example.com/img.jpg").unwrap();
        assert!(!needs_face_model(&ops));

        // Class redaction (#62): names parse as written,
        // spaces included; the class list is in order.
        let (ops, _) = parse_operations(
            "filters:blurclass(person):pixelateclass(traffic%20light)/example.com/img.jpg",
        )
        .unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![
                Filter::BlurClass("person".to_string()),
                Filter::PixelateClass("traffic light".to_string()),
            ])]
        );
        assert!(has_class_redaction(&ops));
        assert_eq!(class_filters(&ops), ["person", "traffic light"]);
        for bad in [
            "filters:blurclass()/example.com/img.jpg",
            "filters:blurclass( )/example.com/img.jpg",
            "filters:blurclass(per%24on)/example.com/img.jpg",
            "filters:pixelateclass/example.com/img.jpg",
        ] {
            assert!(parse_operations(bad).is_err(), "{bad}");
        }

        for bad in [
            "filters:facepad(9)/example.com/img.jpg",
            "filters:facepad(-1)/example.com/img.jpg",
            "filters:facepad(x)/example.com/img.jpg",
            "filters:blurfaces(1)/example.com/img.jpg",
        ] {
            assert!(parse_operations(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn parses_pad() {
        let size = ImageSize {
            width: Dimension::Pixels(200),
            height: Dimension::Pixels(100),
            flip_horizontal: false,
            flip_vertical: false,
        };
        let (ops, _) = parse_operations("pad:200x100/example.com/img.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::Pad {
                size,
                fill: PadFill::Directive
            }]
        );
        let (ops, _) = parse_operations("pad:200x100:blur/example.com/img.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::Pad {
                size,
                fill: PadFill::Blur
            }]
        );
        let (ops, _) = parse_operations("pad:200x100:ff8800/example.com/img.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::Pad {
                size,
                fill: PadFill::Color([255, 136, 0])
            }]
        );
        for bad in [
            "pad:",
            "pad:200",
            "pad:200x100:red",
            "pad:200x100:ff88",
            "pad:200x100:blur:extra",
            // The exact-box contract, same as crop:.
            "pad:200x0",
            "pad:origx100",
            "pad:-200x100",
        ] {
            assert!(parse_operations(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn parses_expiry_segments() {
        let (ops, src, exp) =
            parse_operations_with_expiry("exp:1893456000/300x200/example.com/img.jpg").unwrap();
        assert_eq!(exp, Some(1893456000));
        assert_eq!(ops.len(), 1);
        assert_eq!(src, "example.com/img.jpg");
        // Position-independent among the operations; last wins.
        let (_, _, exp) =
            parse_operations_with_expiry("300x200/exp:100/exp:200/example.com/img.jpg").unwrap();
        assert_eq!(exp, Some(200));
        // Once the source URL starts, `exp:` is just URL text.
        let (ops, src, exp) = parse_operations_with_expiry("example.com/exp:123").unwrap();
        assert!(ops.is_empty() && exp.is_none());
        assert_eq!(src, "example.com/exp:123");
        // Absent means no expiry.
        let (_, _, exp) = parse_operations_with_expiry("300x200/example.com/img.jpg").unwrap();
        assert_eq!(exp, None);
        // Malformed values are parse errors.
        for bad in [
            "exp:/example.com/i.jpg",
            "exp:abc/example.com/i.jpg",
            "exp:-1/example.com/i.jpg",
            "exp:12.5/example.com/i.jpg",
            "exp:99999999999999999999999999/example.com/i.jpg",
        ] {
            assert!(parse_operations_with_expiry(bad).is_err(), "{bad}");
        }
    }

    fn preset_map() -> std::collections::HashMap<String, String> {
        [
            ("thumb".to_string(), "300x200".to_string()),
            (
                "og".to_string(),
                "crop:1200x630:smart/filters:format(webp)".to_string(),
            ),
        ]
        .into_iter()
        .collect()
    }

    #[test]
    fn preset_definitions_validate() {
        assert_eq!(
            parse_preset_operations("crop:1200x630:smart/filters:format(webp):quality(80)")
                .unwrap()
                .len(),
            2
        );
        // Source-like segments, expiry, nested presets, and the
        // empty string are all invalid definitions.
        for bad in [
            "",
            "example.com/img.jpg",
            "exp:123/300x200",
            "preset:other/300x200",
            "300x200/",
        ] {
            assert!(parse_preset_operations(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn presets_expand_in_the_operations_region() {
        let map = preset_map();
        assert_eq!(
            expand_presets("preset:thumb/example.com/img.jpg", &map, false).unwrap(),
            "300x200/example.com/img.jpg"
        );
        assert_eq!(
            expand_presets("preset:og/exp:999/example.com/img.jpg", &map, false).unwrap(),
            "crop:1200x630:smart/filters:format(webp)/exp:999/example.com/img.jpg"
        );
        // Mixed with free-form operations.
        assert_eq!(
            expand_presets(
                "preset:thumb/filters:grayscale/example.com/i.jpg",
                &map,
                false
            )
            .unwrap(),
            "300x200/filters:grayscale/example.com/i.jpg"
        );
        // A preset: segment inside the source path is URL text.
        assert_eq!(
            expand_presets("example.com/preset:thumb/img.jpg", &map, false).unwrap(),
            "example.com/preset:thumb/img.jpg"
        );
        // Unknown names are errors.
        assert_eq!(
            expand_presets("preset:nope/example.com/img.jpg", &map, false),
            Err(ParseError::UnknownPreset("nope".to_string()))
        );
    }

    #[test]
    fn presets_only_mode_rejects_free_form_operations() {
        let map = preset_map();
        assert!(expand_presets("preset:thumb/example.com/img.jpg", &map, true).is_ok());
        assert!(expand_presets("exp:999/preset:thumb/example.com/img.jpg", &map, true).is_ok());
        assert!(expand_presets("example.com/img.jpg", &map, true).is_ok());
        assert_eq!(
            expand_presets("300x200/example.com/img.jpg", &map, true),
            Err(ParseError::PresetsOnly("300x200".to_string()))
        );
        assert_eq!(
            expand_presets(
                "preset:thumb/filters:grayscale/example.com/i.jpg",
                &map,
                true
            ),
            Err(ParseError::PresetsOnly("filters:grayscale".to_string()))
        );
    }

    #[test]
    fn parses_filter_batch_two() {
        let cases: Vec<(&str, Filter)> = vec![
            ("sharpen(1.5)", Filter::Sharpen(1.5)),
            ("saturation(0)", Filter::Saturation(0.0)),
            ("saturation(2.5)", Filter::Saturation(2.5)),
            ("rgb(255,-255,0)", Filter::Rgb(255, -255, 0)),
            ("equalize", Filter::Equalize),
            ("rotate(90)", Filter::Rotate(90)),
            ("rotate(-90)", Filter::Rotate(-90)),
            ("fill(ff8800)", Filter::Fill([255, 136, 0])),
            ("max_bytes(5000)", Filter::MaxBytes(5000)),
        ];
        for (seg, want) in cases {
            let path = format!("filters:{seg}/example.com/img.jpg");
            let (ops, _) = parse_operations(&path).unwrap();
            assert_eq!(ops, vec![Operation::Filters(vec![want])], "{seg}");
        }
    }

    #[test]
    fn rejects_bad_filter_batch_two() {
        for seg in [
            "sharpen(0)",
            "sharpen(-1)",
            "saturation(-0.5)",
            "rgb(256,0,0)",
            "rgb(1,2)",
            "equalize(1)",
            "rotate(45)",
            "fill(ff88)",
            "fill(gggggg)",
            "max_bytes(0)",
            "max_bytes(-5)",
        ] {
            let path = format!("filters:{seg}/example.com/img.jpg");
            assert!(parse_operations(&path).is_err(), "{seg}");
        }
    }

    #[test]
    fn parses_new_filters() {
        let (ops, _) = parse_operations(
            "filters:invert():blur(2.5):brighten(10):contrast(1.2):huerotate(90)/example.com/img.jpg",
        )
        .unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![
                Filter::Invert,
                Filter::Blur(2.5),
                Filter::Brighten(10),
                Filter::Contrast(1.2),
                Filter::HueRotate(90),
            ])]
        );
    }

    #[test]
    fn rejects_bad_new_filter_params() {
        for seg in [
            "filters:blur()",
            "filters:blur(0)",
            "filters:blur(-1.5)",
            "filters:blur(lots)",
            "filters:brighten()",
            "filters:brighten(lots)",
            "filters:contrast()",
            "filters:contrast(lots)",
            "filters:huerotate()",
            "filters:huerotate(lots)",
            "filters:invert(1)",
        ] {
            assert_eq!(
                parse_operations(&format!("{seg}/example.com/img.jpg")),
                Err(ParseError::InvalidFilters(seg.to_string())),
                "segment: {seg:?}"
            );
        }
    }

    #[test]
    fn keeps_operations_in_path_order() {
        let (ops, url) =
            parse_operations("crop:100x100/300x200/filters:grayscale()/example.com/x.jpg").unwrap();
        assert_eq!(ops.len(), 3);
        assert!(matches!(ops[0], Operation::Crop { .. }));
        assert!(matches!(ops[1], Operation::Size(_)));
        assert!(matches!(ops[2], Operation::Filters(_)));
        assert_eq!(url, "example.com/x.jpg");
    }

    #[test]
    fn keeps_source_url_with_slashes_intact() {
        let (ops, url) = parse_operations("300x200/https://example.com/a/b/c.jpg").unwrap();
        assert_eq!(ops, vec![size_op(300, 200)]);
        assert_eq!(url, "https://example.com/a/b/c.jpg");
    }

    #[test]
    fn rejects_missing_source_url() {
        for input in ["", "300x200", "crop:100x100/filters:grayscale()"] {
            assert_eq!(
                parse_operations(input),
                Err(ParseError::MissingSourceUrl),
                "input: {input:?}"
            );
        }
    }

    #[test]
    fn non_operation_segment_starts_source_url() {
        // A segment is an operation only if it parses as one or carries a
        // known prefix; otherwise it starts the source URL.
        let (ops, url) = parse_operations("300xabc/example.com/img.jpg").unwrap();
        assert!(ops.is_empty());
        assert_eq!(url, "300xabc/example.com/img.jpg");
    }

    #[test]
    fn parse_request_composes_signature_and_operations() {
        let (sig, req) =
            parse_request("/unsafe/300x200/filters:grayscale()/example.com/img.jpg").unwrap();
        assert_eq!(sig, Signature::Unsafe);
        assert_eq!(req.operations.len(), 2);
        assert!(matches!(req.operations[0], Operation::Size(_)));
        assert!(matches!(req.operations[1], Operation::Filters(_)));
        assert_eq!(req.source_url, "example.com/img.jpg");
    }

    #[test]
    fn parse_errors_render() {
        assert_eq!(
            ParseError::InvalidSize("x".to_string()).to_string(),
            "invalid size segment: x"
        );
        assert_eq!(
            ParseError::MissingSourceUrl.to_string(),
            "missing source URL"
        );
    }

    #[test]
    fn operation_queries_find_directives() {
        let (ops, _) =
            parse_operations("100x100/filters:format(png):no_upscale/example.com/i.jpg").unwrap();
        assert_eq!(explicit_format(&ops), Some(ImageFormat::Png));
        assert!(has_no_upscale(&ops));
        let (ops, _) = parse_operations("100x100/example.com/i.jpg").unwrap();
        assert_eq!(explicit_format(&ops), None);
        assert!(!has_no_upscale(&ops));
        // A later format filter wins over an earlier one.
        let (ops, _) =
            parse_operations("filters:format(jpeg):format(webp)/example.com/i.jpg").unwrap();
        assert_eq!(explicit_format(&ops), Some(ImageFormat::WebP));
    }
}
