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
    /// A `focal:...` segment did not match
    /// `focal:<left>x<top>:<right>x<bottom>` with right > left
    /// and bottom > top. Carries the offending segment.
    InvalidFocalRect(String),
    /// A `pad:...` segment did not match `pad:<WxH>[:<fill>]`.
    /// Carries the offending segment.
    InvalidPad(String),
    /// An `exp:...` segment did not carry a valid unix
    /// timestamp. Carries the offending segment.
    InvalidExpiry(String),
    /// A `maxage:...` segment did not carry a valid cache
    /// lifetime: whole seconds from 0 to [`MAX_CACHE_AGE`].
    /// Carries the offending segment.
    InvalidMaxAge(String),
    /// A `debug:...` segment named a debug view the grammar
    /// does not have: the only one is `debug:focal`. Carries
    /// the offending segment.
    InvalidDebug(String),
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
    /// A `curve(...)` filter inside a `filters:...` segment did
    /// not parse: unknown channel, a coordinate count that is
    /// not 2 to 16 `x,y` points, a coordinate outside 0-255, or
    /// x values not strictly increasing. Carries the offending
    /// segment.
    InvalidCurve(String),
    /// A `convolution(...)` filter inside a `filters:...`
    /// segment did not parse: a value count other than 9
    /// kernel values plus the flag (3x3) or 25 plus the flag
    /// (5x5), a value outside the kernel grammar or the
    /// [`MAX_KERNEL_VALUE`] bound, or a normalize flag other
    /// than 0 or 1. Carries the offending segment.
    InvalidConvolution(String),
    /// A `noise(...)` filter inside a `filters:...` segment
    /// did not parse: the amount is not an integer in
    /// 0..=100. Carries the offending segment.
    InvalidNoise(String),
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
            ParseError::InvalidFocalRect(_) => "invalid_focal_rect",
            ParseError::InvalidPad(_) => "invalid_pad",
            ParseError::InvalidExpiry(_) => "invalid_expiry",
            ParseError::InvalidMaxAge(_) => "invalid_maxage",
            ParseError::InvalidDebug(_) => "invalid_debug",
            ParseError::UnknownPreset(_) => "unknown_preset",
            ParseError::InvalidPreset(_) => "invalid_preset",
            ParseError::PresetsOnly(_) => "presets_only",
            ParseError::InvalidFilters(_) => "invalid_filters",
            ParseError::InvalidCurve(_) => "invalid_curve",
            ParseError::InvalidConvolution(_) => "invalid_convolution",
            ParseError::InvalidNoise(_) => "invalid_noise",
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
            ParseError::InvalidFocalRect(seg) => {
                write!(f, "invalid focal segment: {seg}")
            }
            ParseError::InvalidPad(seg) => write!(f, "invalid pad segment: {seg}"),
            ParseError::InvalidExpiry(seg) => write!(f, "invalid expiry segment: {seg}"),
            ParseError::InvalidMaxAge(seg) => write!(f, "invalid maxage segment: {seg}"),
            ParseError::InvalidDebug(seg) => write!(f, "invalid debug segment: {seg}"),
            ParseError::UnknownPreset(name) => write!(f, "unknown preset: {name}"),
            ParseError::InvalidPreset(text) => write!(f, "invalid preset: {text}"),
            ParseError::PresetsOnly(seg) => {
                write!(f, "free-form operations are disabled (presets only): {seg}")
            }
            ParseError::InvalidFilters(seg) => {
                write!(f, "invalid filters segment: {seg}")
            }
            ParseError::InvalidCurve(seg) => {
                write!(f, "invalid curve filter in segment: {seg}")
            }
            ParseError::InvalidConvolution(seg) => {
                write!(f, "invalid convolution filter in segment: {seg}")
            }
            ParseError::InvalidNoise(seg) => {
                write!(f, "invalid noise filter in segment: {seg}")
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
    /// Keep the top-left corner: the left when the width is
    /// cropped, the top when the height is.
    TopLeft,
    /// Keep the top-right corner.
    TopRight,
    /// Keep the bottom-left corner.
    BottomLeft,
    /// Keep the bottom-right corner.
    BottomRight,
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

/// Output format requested by a `format(...)` filter.
/// Mostly image formats; `Mp4` and `WebM` are video outputs
/// (issue #162), produced from animated sources by the
/// operator-configured external encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Jpeg,
    Png,
    WebP,
    Jxl,
    Avif,
    Gif,
    Mp4,
    WebM,
}

impl ImageFormat {
    /// Whether this format is a video output (`mp4`,
    /// `webm`): animated sources only, an external encoder,
    /// and no `Accept` negotiation.
    pub fn is_video(self) -> bool {
        matches!(self, ImageFormat::Mp4 | ImageFormat::WebM)
    }
}

/// Encoding quality from a `quality(...)` filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quality {
    /// An explicit value, 0-100.
    Value(u8),
    /// Chosen per image by content analysis at encode time.
    Auto,
}

/// Which channels a `curve(...)` filter remaps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurveChannel {
    /// All three color channels through the one curve (`rgb`).
    Rgb,
    /// The red channel only (`r`).
    R,
    /// The green channel only (`g`).
    G,
    /// The blue channel only (`b`).
    B,
}

impl CurveChannel {
    /// The canonical spelling of the channel, as written in
    /// the `curve(...)` argument list.
    pub fn name(self) -> &'static str {
        match self {
            CurveChannel::Rgb => "rgb",
            CurveChannel::R => "r",
            CurveChannel::G => "g",
            CurveChannel::B => "b",
        }
    }

    fn parse(s: &str) -> Option<CurveChannel> {
        Some(match s {
            "rgb" => CurveChannel::Rgb,
            "r" => CurveChannel::R,
            "g" => CurveChannel::G,
            "b" => CurveChannel::B,
            _ => return None,
        })
    }
}

/// The backdrop a `studio(...)` composite is built on,
/// resolved against operator configuration when the filter
/// runs (issue #197).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StudioBg {
    /// Not named in the URL: the operator brand color
    /// (`CRABCROP_BRAND_COLOR`) when one is configured,
    /// else white.
    Default,
    /// Plain white, named in the URL.
    White,
    /// The named color, as RGB.
    Color([u8; 3]),
    /// The operator brand color, named in the URL. With no
    /// brand color configured it resolves to white, and the
    /// resolution is recorded as a fallback in meta.
    Brand,
    /// No backdrop: the subject keeps its transparency and
    /// no shadow is synthesized.
    None,
}

impl StudioBg {
    /// The canonical spelling inside `studio(...)`, or
    /// `None` when the value is the default and the
    /// argument is omitted from the canonical form.
    fn text(self) -> Option<String> {
        match self {
            StudioBg::Default => None,
            StudioBg::White => Some("white".to_string()),
            StudioBg::Color(c) => Some(hex3(c)),
            StudioBg::Brand => Some("brand".to_string()),
            StudioBg::None => Some("none".to_string()),
        }
    }

    fn parse(s: &str) -> Option<StudioBg> {
        Some(match s {
            "white" => StudioBg::White,
            "brand" => StudioBg::Brand,
            "none" => StudioBg::None,
            _ => StudioBg::Color(parse_hex6(s).ok()?),
        })
    }
}

fn parse_hex6(s: &str) -> Result<[u8; 3], ()> {
    if s.len() != 6 || !s.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(());
    }
    let byte = |i: usize| u8::from_str_radix(&s[i..i + 2], 16).map_err(|_| ());
    Ok([byte(0)?, byte(2)?, byte(4)?])
}

/// The contact shadow a `studio(...)` composite
/// synthesizes under the subject (issue #197).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StudioShadow {
    /// No shadow.
    None,
    /// A soft, wide, light shadow (the default).
    Soft,
    /// A tighter, darker shadow.
    Strong,
}

impl StudioShadow {
    /// The canonical spelling inside `studio(...)`.
    pub fn name(self) -> &'static str {
        match self {
            StudioShadow::None => "none",
            StudioShadow::Soft => "soft",
            StudioShadow::Strong => "strong",
        }
    }

    fn parse(s: &str) -> Option<StudioShadow> {
        Some(match s {
            "none" => StudioShadow::None,
            "soft" => StudioShadow::Soft,
            "strong" => StudioShadow::Strong,
            _ => return None,
        })
    }
}

/// The `studio(product,...)` argument set, with the
/// defaults already applied: an unnamed backdrop
/// (`StudioBg::Default`), a soft shadow, and a pad of 10
/// percent (issue #197).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StudioSpec {
    pub bg: StudioBg,
    pub shadow: StudioShadow,
    pub pad: u8,
}

impl Default for StudioSpec {
    fn default() -> Self {
        StudioSpec {
            bg: StudioBg::Default,
            shadow: StudioShadow::Soft,
            pad: 10,
        }
    }
}

/// Parse the `studio(...)` argument list: the mode
/// (`product`, the only one) followed by named `key:value`
/// arguments. Bad arguments fail as the segment-wide
/// invalid filter error, like every other filter: the
/// frozen ParseError set does not grow for studio.
fn parse_studio(p: &str) -> Option<StudioSpec> {
    let mut spec = StudioSpec::default();
    let mut seen: Vec<&str> = Vec::new();
    let mut parts = p.split(',');
    if parts.next() != Some("product") {
        return None;
    }
    for part in parts {
        let (key, value) = part.split_once(':')?;
        if seen.contains(&key) {
            return None;
        }
        seen.push(key);
        match key {
            "bg" => spec.bg = StudioBg::parse(value)?,
            "shadow" => spec.shadow = StudioShadow::parse(value)?,
            "pad" => {
                let pad: u8 = value.parse().ok()?;
                if pad > 40 {
                    return None;
                }
                spec.pad = pad;
            }
            _ => return None,
        }
    }
    Some(spec)
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
    /// Gaussian blur; sigma is a positive, normal f32 no
    /// larger than [`MAX_BLUR_SIGMA`].
    Blur(f32),
    /// Add to every channel (-255..=255), clamped to 0-255.
    Brighten(i32),
    /// Adjust contrast, percent-style (-100..=100): 0 is
    /// identity, -100 is flat gray.
    Contrast(f32),
    /// Rotate the hue, in degrees.
    HueRotate(i32),
    /// Sharpen by this amount (unsharp mask strength, > 0).
    Sharpen(f32),
    /// Scale color saturation (0..=255): 0 is grayscale, 1
    /// is identity.
    Saturation(f32),
    /// Per-channel level offsets, each -255..=255.
    Rgb(i32, i32, i32),
    /// Tone curve: remap the named channel(s) through the
    /// control points (x, y), each 0-255 with x strictly
    /// increasing, 2 to 16 points. Between points the mapping
    /// is linear; inputs outside the points' x range clamp
    /// to the nearest end point's y.
    Curve(CurveChannel, Vec<(u8, u8)>),
    /// A caller-supplied convolution kernel: 9 values (3x3)
    /// or 25 (5x5) in row-major order, each within
    /// [`MAX_KERNEL_VALUE`] in absolute value. With the
    /// normalize flag set, the kernel is scaled by the sum
    /// of its entries before it applies (a zero sum applies
    /// the kernel as written).
    Convolution(Vec<f64>, bool),
    /// Monochrome film grain: one noise value per pixel,
    /// added equally to the red, green, and blue channels
    /// (alpha untouched), the value uniform in plus or
    /// minus this percent of full scale. The field is
    /// deterministic, seeded from the image content and
    /// the amount, so a request always renders the same
    /// bytes. 0 is a no-op.
    Noise(u8),
    /// Histogram equalization per channel.
    Equalize,
    /// Rotate by the given angle in degrees (issue #144):
    /// multiples of 90 take the exact quarter-turn path, any
    /// other angle rotates geometrically onto an expanded
    /// canvas.
    Rotate(f32),
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
    /// Studio mode (issue #197): composite the subject the
    /// segmentation model finds onto a backdrop with a
    /// synthesized contact shadow, scaled and centered to
    /// a pad margin. A deterministic expansion into mask,
    /// defringe, backdrop, shadow, and pad steps, reported
    /// in meta.
    Studio(StudioSpec),
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
    /// With `adaptive` (`fit-in:300x200:adaptive`), the box axes
    /// swap when the source orientation and the box orientation
    /// disagree, so a portrait source fits the landscape box's
    /// transpose instead of letterboxing into the tighter axis.
    FitIn { size: ImageSize, adaptive: bool },
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
    /// Name the rectangle the next crop must keep
    /// (`focal:<left>x<top>:<right>x<bottom>`). Produces no
    /// pixels of its own; the next `Crop` consumes it.
    FocalRect {
        left: u32,
        top: u32,
        right: u32,
        bottom: u32,
    },
    /// Render the smart-crop debug view (`debug:focal`): the
    /// response is a diagnostic image of the focal decision
    /// the next `Crop` makes (or of the final image, when no
    /// crop follows), never the transform result. Unlike
    /// `exp:`/`maxage:` this is an operation, not metadata:
    /// it changes the response, so it stays in the canonical
    /// form (and therefore the cache key form) in place.
    DebugFocal,
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

/// The largest cache lifetime a `maxage:` directive may name,
/// in seconds: one year, the server's own default. A request
/// may shorten its response's lifetime, never extend it past
/// the default the immutable policy already grants.
pub const MAX_CACHE_AGE: u32 = 31_536_000;

/// The largest absolute value a `convolution(...)` kernel
/// entry may carry. Kernel entries are weights, not pixel
/// values; real kernels (sharpen, blur, edge detection) stay
/// in single digits, so the bound is generous headroom that
/// still keeps a hostile kernel's accumulated sums far from
/// any overflow concern.
pub const MAX_KERNEL_VALUE: f64 = 1024.0;

/// The largest `blur(...)` sigma the grammar accepts. The
/// Gaussian kernel grows at about 6.67 entries per unit of
/// sigma and is allocated per axis before any pixel is
/// processed, so an unbounded sigma is an allocation dial:
/// sigma 1e8 asks for gigabytes of kernel. 512 is already a
/// full wash on any canvas the output budget allows, so the
/// cap costs no real image. Non-normal values are rejected
/// with the cap: a subnormal sigma panics the image crate's
/// Gaussian setup, which asserts `sigma.is_normal()`.
pub const MAX_BLUR_SIGMA: f32 = 512.0;

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
/// Whether `id` is a well-formed key ID (issue #143): 1 to 32
/// characters of `[A-Za-z0-9_-]`. Key IDs are routing metadata
/// in the signature segment, never secret material.
pub fn is_valid_key_id(id: &str) -> bool {
    crabcrop_url::is_valid_key_id(id)
}

/// Split the signature segment off a request path, returning
/// the signature, the key ID it names (issue #143: an optional
/// `<key-id>:` prefix on the signature segment, how rotation
/// works without invalidating every outstanding URL), and
/// the remainder.
pub fn parse_signature(path: &str) -> Result<(Signature, Option<String>, &str), ParseError> {
    let path = path.strip_prefix('/').unwrap_or(path);
    if path.is_empty() {
        return Err(ParseError::EmptyPath);
    }
    let (first, rest) = path.split_once('/').ok_or(ParseError::MissingSignature)?;
    if first.is_empty() {
        return Err(ParseError::EmptyPath);
    }
    if first == "unsafe" {
        return Ok((Signature::Unsafe, None, rest));
    }
    let (key_id, hex) = match first.split_once(':') {
        Some((kid, hex)) if is_valid_key_id(kid) => (Some(kid.to_string()), hex),
        Some(_) => return Err(ParseError::MalformedSignature),
        None => (None, first),
    };
    if !hex.is_empty()
        && hex
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        // One canonical spelling (audit PL-7): builders emit
        // lowercase hex, and other spellings (uppercase, non-hex
        // tags) are parse errors rather than alternate forms of
        // the same URL.
        Ok((Signature::Hmac(hex.to_string()), key_id, rest))
    } else {
        Err(ParseError::MalformedSignature)
    }
}

/// Parse one path segment as an operation.
///
/// Returns `None` when the segment is not an operation at all, which
/// means it starts the source URL. A segment that claims to be an
/// operation (`crop:` / `filters:` prefix) but is malformed is an error.
fn parse_operation(seg: &str) -> Option<Result<Operation, ParseError>> {
    if let Some(rest) = seg.strip_prefix("debug:") {
        return Some(match rest {
            "focal" => Ok(Operation::DebugFocal),
            _ => Err(ParseError::InvalidDebug(seg.to_string())),
        });
    }
    if let Some(rest) = seg.strip_prefix("crop-rect:") {
        return Some(parse_crop_rect(rest, seg));
    }
    if let Some(rest) = seg.strip_prefix("focal:") {
        return Some(parse_focal_rect(rest, seg));
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
        Err(_) => {
            // A segment that fails the size grammar but is
            // shaped like a size (an `x` joining two parts, at
            // least one of them a dimension spelling, and no
            // URL punctuation) is a malformed size, not the
            // start of a source URL: without this, an over-cap
            // or garbled size silently became a source path and
            // the request failed downstream as a 502 fetch
            // error instead of the documented 400 (issue #228).
            if size_shaped(seg) {
                Some(Err(ParseError::InvalidSize(seg.to_string())))
            } else {
                None
            }
        }
    }
}

/// Whether a segment that failed [`parse_size`] still reads as
/// a size attempt. At least one side of the `x` must carry a
/// dimension spelling (digits, `orig`, or empty, with an
/// optional flip sign), and neither side may carry the `.` or
/// `:` that mark URL text (a host, a file name, a scheme, a
/// port), so genuine source segments like `example.com` or
/// `100x200.jpg` never trip it.
fn size_shaped(seg: &str) -> bool {
    let Some((left, right)) = seg.split_once('x') else {
        return false;
    };
    if left.contains('.') || right.contains('.') || seg.contains(':') {
        return false;
    }
    let dimension_spelling = |part: &str| {
        let digits = part.strip_prefix('-').unwrap_or(part);
        digits.is_empty() || digits == "orig" || digits.bytes().all(|b| b.is_ascii_digit())
    };
    dimension_spelling(left) || dimension_spelling(right)
}

/// Parse the box after `fit-in:`. Flips make no sense when fitting,
/// so a flipped size is invalid here. The one modifier is
/// `:adaptive` after the box; anything else in that slot (an
/// unknown modifier, a second modifier, `adaptive` standing in
/// for the box) is the same `invalid_size` error as a bad box.
fn parse_fit_in(rest: &str, seg: &str) -> Result<Operation, ParseError> {
    let invalid = || ParseError::InvalidSize(seg.to_string());
    let (dims, adaptive) = match rest.split_once(':') {
        Some((dims, "adaptive")) => (dims, true),
        Some(_) => return Err(invalid()),
        None => (rest, false),
    };
    let size = parse_size(dims).map_err(|_| invalid())?;
    if size.flip_horizontal || size.flip_vertical {
        return Err(invalid());
    }
    Ok(Operation::FitIn { size, adaptive })
}

/// Parse `<left>x<top>:<right>x<bottom>` with ordered corners
/// (right > left, bottom > top). Bounds against the actual image
/// are the engine's job, since they depend on the source.
fn parse_rect_points(rest: &str) -> Option<(u32, u32, u32, u32)> {
    let (lt, rb) = rest.split_once(':')?;
    let (left, top) = parse_point(lt)?;
    let (right, bottom) = parse_point(rb)?;
    if right <= left || bottom <= top {
        return None;
    }
    Some((left, top, right, bottom))
}

/// Parse `<left>x<top>:<right>x<bottom>` after `crop-rect:`.
fn parse_crop_rect(rest: &str, seg: &str) -> Result<Operation, ParseError> {
    let (left, top, right, bottom) =
        parse_rect_points(rest).ok_or_else(|| ParseError::InvalidCropRect(seg.to_string()))?;
    Ok(Operation::CropRect {
        left,
        top,
        right,
        bottom,
    })
}

/// Parse `<left>x<top>:<right>x<bottom>` after `focal:`.
fn parse_focal_rect(rest: &str, seg: &str) -> Result<Operation, ParseError> {
    let (left, top, right, bottom) =
        parse_rect_points(rest).ok_or_else(|| ParseError::InvalidFocalRect(seg.to_string()))?;
    Ok(Operation::FocalRect {
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
        "top-left" => Some(Align::TopLeft),
        "top-right" => Some(Align::TopRight),
        "bottom-left" => Some(Align::BottomLeft),
        "bottom-right" => Some(Align::BottomRight),
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
        Some(p) => PadFill::Color(parse_hex6(p).map_err(|_| invalid())?),
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
        "mp4" => Some(ImageFormat::Mp4),
        "webm" => Some(ImageFormat::WebM),
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

/// Parse a `rotate` angle (issue #144): an optional minus,
/// digits, and an optional fraction, with no exponent and no
/// leading dot, so every implementation of the grammar reads
/// the same strings. Bounded to 100 full turns; the value is
/// stored as written and reduced at apply time.
fn parse_angle(text: &str) -> Result<f32, ()> {
    let digits = text.strip_prefix('-').unwrap_or(text);
    let (whole, fraction) = match digits.split_once('.') {
        Some((w, f)) => (w, Some(f)),
        None => (digits, None),
    };
    if whole.is_empty() || !whole.bytes().all(|b| b.is_ascii_digit()) {
        return Err(());
    }
    if let Some(f) = fraction
        && (f.is_empty() || !f.bytes().all(|b| b.is_ascii_digit()))
    {
        return Err(());
    }
    let value: f32 = text.parse().map_err(|_| ())?;
    if !value.is_finite() || value.abs() > 36000.0 {
        return Err(());
    }
    Ok(value)
}

/// The canonical spelling of a rotate angle: whole values
/// print as integers (so `rotate(90)` keeps the spelling it
/// has always had, and with it its cache key), fractions use
/// the shortest round-trip form.
fn format_angle(value: f32) -> String {
    if value == value.trunc() {
        format!("{}", value as i64)
    } else {
        format!("{value}")
    }
}

/// Parse a `convolution` kernel value: the rotate angle's
/// strict decimal grammar (an optional minus, digits, and an
/// optional fraction, with no exponent and no leading dot),
/// so every implementation of the grammar reads the same
/// strings. Bounded to [`MAX_KERNEL_VALUE`] in absolute
/// value; decimals are expected here (a gaussian kernel is
/// fractions), unlike the integer-only filter arguments.
fn parse_kernel_value(text: &str) -> Result<f64, ()> {
    let digits = text.strip_prefix('-').unwrap_or(text);
    let (whole, fraction) = match digits.split_once('.') {
        Some((w, f)) => (w, Some(f)),
        None => (digits, None),
    };
    if whole.is_empty() || !whole.bytes().all(|b| b.is_ascii_digit()) {
        return Err(());
    }
    if let Some(f) = fraction
        && (f.is_empty() || !f.bytes().all(|b| b.is_ascii_digit()))
    {
        return Err(());
    }
    let value: f64 = text.parse().map_err(|_| ())?;
    if !value.is_finite() || value.abs() > MAX_KERNEL_VALUE {
        return Err(());
    }
    Ok(value)
}

/// The canonical spelling of a kernel value, matching the
/// rotate angle's: whole values print as integers, fractions
/// use the shortest round-trip form, so spellings like
/// `0.50` and `-1.0` collapse to `0.5` and `-1` in the
/// canonical form and share a cache key.
pub(crate) fn format_kernel_value(value: f64) -> String {
    if value == value.trunc() {
        format!("{}", value as i64)
    } else {
        format!("{value}")
    }
}

/// Split a filters segment into its filters on the `:`
/// separators that sit outside parentheses: an argument
/// list may itself contain `:` (the `studio(...)` named
/// arguments are `key:value`), so a naive split would cut
/// those filters apart. Unbalanced parentheses leave the
/// tail in one part, which then fails its own parse, as
/// before.
fn split_filter_chain(rest: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0;
    for (i, c) in rest.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            ':' if depth == 0 => {
                parts.push(&rest[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&rest[start..]);
    parts
}

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
        let filter = match (name, params) {
            ("grayscale", None) => Filter::Grayscale,
            ("invert", None) => Filter::Invert,
            ("no_upscale", None) => Filter::NoUpscale,
            ("blur", Some(p)) => {
                let sigma: f32 = p.parse().map_err(|_| invalid())?;
                if !sigma.is_normal() || sigma <= 0.0 || sigma > MAX_BLUR_SIGMA {
                    return Err(invalid());
                }
                Filter::Blur(sigma)
            }
            ("brighten", Some(p)) => {
                let amount: i32 = p.parse().map_err(|_| invalid())?;
                if !(-255..=255).contains(&amount) {
                    return Err(invalid());
                }
                Filter::Brighten(amount)
            }
            ("contrast", Some(p)) => {
                let amount: f32 = p.parse().map_err(|_| invalid())?;
                if !amount.is_finite() || !(-100.0..=100.0).contains(&amount) {
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
                    if !(1..=100).contains(&q) {
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
                if !amount.is_finite() || !(0.0..=255.0).contains(&amount) {
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
            ("curve", Some(p)) => {
                // curve(<channel>,<x>,<y>,...): the channel,
                // then a flat list of control points, 2 to 16
                // of them, every coordinate 0-255 and the x
                // values strictly increasing. Curve failures
                // carry their own error code (invalid_curve),
                // not the segment-wide invalid_filters.
                let bad_curve = || ParseError::InvalidCurve(seg.to_string());
                let parts: Vec<&str> = p.split(',').collect();
                let channel = CurveChannel::parse(parts[0]).ok_or_else(bad_curve)?;
                let coords = &parts[1..];
                if coords.len() < 4 || coords.len() > 32 || !coords.len().is_multiple_of(2) {
                    return Err(bad_curve());
                }
                let mut points: Vec<(u8, u8)> = Vec::with_capacity(coords.len() / 2);
                for pair in coords.as_chunks::<2>().0 {
                    let x: u8 = pair[0].parse().map_err(|_| bad_curve())?;
                    let y: u8 = pair[1].parse().map_err(|_| bad_curve())?;
                    if points.last().is_some_and(|last| x <= last.0) {
                        return Err(bad_curve());
                    }
                    points.push((x, y));
                }
                Filter::Curve(channel, points)
            }
            ("convolution", Some(p)) => {
                // convolution(<v1>,...,<vN>,<normalize>): the
                // kernel values in row-major order, 9 of them
                // (3x3) or 25 (5x5), then the normalize flag,
                // 0 or 1. Convolution failures carry their own
                // error code (invalid_convolution), not the
                // segment-wide invalid_filters.
                let bad_conv = || ParseError::InvalidConvolution(seg.to_string());
                let parts: Vec<&str> = p.split(',').collect();
                if parts.len() != 10 && parts.len() != 26 {
                    return Err(bad_conv());
                }
                let normalize = match parts[parts.len() - 1] {
                    "0" => false,
                    "1" => true,
                    _ => return Err(bad_conv()),
                };
                let mut values = Vec::with_capacity(parts.len() - 1);
                for part in &parts[..parts.len() - 1] {
                    values.push(parse_kernel_value(part).map_err(|_| bad_conv())?);
                }
                Filter::Convolution(values, normalize)
            }
            ("noise", Some(p)) => {
                // noise(<amount>): film grain strength, an
                // integer 0 to 100 percent of full scale, in
                // the numeric filter-table style. Noise
                // failures carry their own error code
                // (invalid_noise), not the segment-wide
                // invalid_filters.
                let bad_noise = || ParseError::InvalidNoise(seg.to_string());
                let amount: u8 = p.parse().map_err(|_| bad_noise())?;
                if amount > 100 {
                    return Err(bad_noise());
                }
                Filter::Noise(amount)
            }
            ("equalize", None) => Filter::Equalize,
            ("removebg", None) => Filter::RemoveBg,
            ("studio", Some(p)) => Filter::Studio(parse_studio(p).ok_or_else(invalid)?),
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
            ("rotate", Some(p)) => Filter::Rotate(parse_angle(p).map_err(|_| invalid())?),
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
                if width > MAX_DIMENSION {
                    return Err(invalid());
                }
                let hex = parts[1];
                Filter::Border(width, parse_hex6(hex).map_err(|_| invalid())?)
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
                Filter::Duotone(
                    parse_hex6(parts[0]).map_err(|_| invalid())?,
                    parse_hex6(parts[1]).map_err(|_| invalid())?,
                )
            }
            ("monochrome", Some(p)) => Filter::Monochrome(parse_hex6(p).map_err(|_| invalid())?),
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
            ("fill", Some(p)) => Filter::Fill(parse_hex6(p).map_err(|_| invalid())?),
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
    let (operations, source_url, _, _) = parse_operations_with_expiry(input)?;
    Ok((operations, source_url))
}

/// What parsing a remainder yields: the operations, the
/// source URL, and the per-request metadata that is not
/// operations (the `exp:` expiry, then the `maxage:` cache
/// lifetime).
pub type ParsedRemainder<'a> = (Vec<Operation>, &'a str, Option<u64>, Option<u32>);

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

/// Parse an operations-only string, the shape of a preset
/// definition: every segment must be an operation, and neither
/// `exp:` or `maxage:` metadata nor nested `preset:` references
/// are allowed: both metadata kinds are per-request, and a
/// preset is a stable template. Used to validate preset
/// definitions at server boot; request expansion itself is
/// textual (see [`expand_presets`]).
pub fn parse_preset_operations(input: &str) -> Result<Vec<Operation>, ParseError> {
    if input.is_empty() {
        return Err(ParseError::InvalidPreset(input.to_string()));
    }
    let mut operations = Vec::new();
    for seg in input.split('/') {
        if seg.starts_with("preset:") || seg.starts_with("exp:") || seg.starts_with("maxage:") {
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
/// presets. `exp:` and `maxage:` metadata stay allowed; neither
/// is an operation.
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
        } else if !(seg.starts_with("exp:")
            || seg.starts_with("maxage:")
            || parse_operation(seg).is_some_and(|r| r.is_ok()))
        {
            break;
        }
        if remaining.is_empty() {
            break;
        }
        tail = remaining;
    }
    out
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
    let directives = RequestDirectives::analyze(&url_ops);
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

/// Parse a server-wide default operations string (issue
/// #145): preset references expand against the boot preset
/// table, and the rest must be operations only, with no
/// source. The operations are returned ready to prepend to a
/// request's own. An empty string yields no operations.
pub fn parse_default_ops(
    text: &str,
    presets: &std::collections::HashMap<String, String>,
) -> Result<Vec<Operation>, ParseError> {
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }
    let expanded = expand_presets(text, presets, false)?;
    // The operations parser expects a source after the
    // operations; a synthetic one terminates the list. The
    // parse must land exactly on it: anything else means
    // part of the string read as a source, which in a
    // defaults string is an operator error, not an
    // operation.
    let terminated = format!("{expanded}/example.com/defaults");
    let (ops, source, _, _) = parse_operations_with_expiry(&terminated)?;
    if source != "example.com/defaults" {
        return Err(ParseError::InvalidPreset(text.to_string()));
    }
    Ok(ops)
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
        } else if seg.starts_with("exp:")
            || seg.starts_with("maxage:")
            || parse_operation(seg).is_some_and(|r| r.is_ok())
        {
            if only_presets && !seg.starts_with("exp:") && !seg.starts_with("maxage:") {
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

/// Everything the server and the engine need to know about a
/// parsed request's directives, computed in ONE pass over the
/// operations. This replaces the former family of per-question
/// walks (each helper re-scanned the whole list, and the
/// server's AI-gate predicate had drifted from the engine's
/// preflight): there is now exactly one place that reads the
/// ops list for directives, and both sides consume it.
///
/// Field semantics match the historical helpers: `format`,
/// `fill`, `dpr`, and `facepad` are request-wide with the LAST
/// segment winning; the shape filter is the FIRST one (it names
/// error messages); class names keep request order; watermark
/// URLs are distinct, in order of first appearance.
#[derive(Debug, Clone, PartialEq)]
pub struct RequestDirectives {
    /// The explicitly named output format, if any.
    pub explicit_format: Option<ImageFormat>,
    /// Any `no_upscale` directive.
    pub no_upscale: bool,
    /// The `fill(...)` directive color, if any.
    pub fill: Option<[u8; 3]>,
    /// Any `removebg` filter.
    pub removebg: bool,
    /// Any `studio(...)` filter (issue #197): it runs the
    /// mask model, so it takes an AI permit and needs the
    /// provider preflight, like `removebg`.
    pub studio: bool,
    /// The first shape filter (`radius` or `mask`), named for
    /// error messages.
    pub shape_filter: Option<&'static str>,
    /// Any `upscale` filter.
    pub upscale: bool,
    /// Any smart-aligned crop.
    pub smart_crop: bool,
    /// The explicit DPR factor, if a `dpr` directive is present.
    pub dpr: Option<f32>,
    /// The ops fix the output dimensions themselves (a size,
    /// crop, fit-in, pad, or crop-rect operation), so a width
    /// client hint does not apply.
    pub has_size_operation: bool,
    /// Any `keep_profile` directive.
    pub keep_profile: bool,
    /// Any `quality(...)` directive, fixed or auto.
    pub quality_directive: bool,
    /// Any `text` filter.
    pub text: bool,
    /// The request requires the face model: a face-aligned
    /// crop, or a face redaction filter. (Smart alignment does
    /// not count here: it falls back to saliency.)
    pub needs_face_model: bool,
    /// Class names requested by class redaction filters, in
    /// request order.
    pub classes: Vec<String>,
    /// Padding for a face-aligned crop, as a fraction of the
    /// face size added on every side. Defaults to 0.5.
    pub face_padding: f32,
    /// The distinct watermark and overlay URLs referenced, in
    /// order of first appearance. The server fetches each one
    /// before the engine runs.
    pub watermark_urls: Vec<String>,
    /// Any `debug:focal` segment: the response is the focal
    /// debug view, never a cached transform result.
    pub debug_focal: bool,
}

impl RequestDirectives {
    /// Analyze a parsed request's operations in a single walk.
    pub fn analyze(ops: &[Operation]) -> Self {
        let mut d = RequestDirectives {
            explicit_format: None,
            no_upscale: false,
            fill: None,
            removebg: false,
            studio: false,
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
            face_padding: 0.5,
            watermark_urls: Vec::new(),
            debug_focal: false,
        };
        for op in ops {
            match op {
                Operation::Size(_)
                | Operation::Crop { .. }
                | Operation::FitIn { .. }
                | Operation::Pad { .. }
                | Operation::CropRect { .. } => d.has_size_operation = true,
                Operation::DebugFocal => d.debug_focal = true,
                _ => {}
            }
            if let Operation::Crop { align, .. } = op {
                match align {
                    Align::Smart => d.smart_crop = true,
                    Align::Face => d.needs_face_model = true,
                    _ => {}
                }
            }
            if let Operation::Filters(filters) = op {
                for f in filters {
                    match f {
                        Filter::Format(fmt) => d.explicit_format = Some(*fmt),
                        Filter::NoUpscale => d.no_upscale = true,
                        Filter::Fill(color) => d.fill = Some(*color),
                        Filter::RemoveBg => d.removebg = true,
                        Filter::Studio(_) => d.studio = true,
                        Filter::Radius(_) if d.shape_filter.is_none() => {
                            d.shape_filter = Some("radius")
                        }
                        Filter::Mask(_) if d.shape_filter.is_none() => {
                            d.shape_filter = Some("mask")
                        }
                        Filter::Upscale(_) => d.upscale = true,
                        Filter::Dpr(factor) => d.dpr = Some(*factor),
                        Filter::KeepProfile => d.keep_profile = true,
                        Filter::Quality(_) => d.quality_directive = true,
                        Filter::Text { .. } => d.text = true,
                        Filter::BlurFaces | Filter::PixelateFaces => d.needs_face_model = true,
                        Filter::BlurClass(name) | Filter::PixelateClass(name) => {
                            d.classes.push(name.clone())
                        }
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

    /// Whether any class redaction filter is present (which
    /// needs the detection model and the AI gate).
    pub fn has_class_redaction(&self) -> bool {
        !self.classes.is_empty()
    }

    /// Whether the request can run AI inference in the engine.
    /// This is the ONE predicate behind both the server's AI
    /// gate and the engine's expectations: background removal,
    /// studio mode, upscale, face operations, and class
    /// redaction always infer; smart crops infer when a face
    /// provider is configured, so they are gated too.
    pub fn uses_ai(&self) -> bool {
        self.removebg
            || self.studio
            || self.upscale
            || self.needs_face_model
            || self.smart_crop
            || self.has_class_redaction()
            // `debug:focal` runs the face and focal providers
            // and bypasses the cache in both directions, so
            // it takes an AI permit like the work it runs
            // (audit R2-17).
            || self.debug_focal
    }
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
        Align::TopLeft => Some("top-left"),
        Align::TopRight => Some("top-right"),
        Align::BottomLeft => Some("bottom-left"),
        Align::BottomRight => Some("bottom-right"),
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

pub(crate) fn format_text(format: ImageFormat) -> &'static str {
    match format {
        ImageFormat::Jpeg => "jpeg",
        ImageFormat::Png => "png",
        ImageFormat::WebP => "webp",
        ImageFormat::Gif => "gif",
        ImageFormat::Avif => "avif",
        ImageFormat::Jxl => "jxl",
        ImageFormat::Mp4 => "mp4",
        ImageFormat::WebM => "webm",
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
        Filter::Curve(channel, points) => {
            let mut text = format!("curve({}", channel.name());
            for (x, y) in points {
                text.push_str(&format!(",{x},{y}"));
            }
            text.push(')');
            text
        }
        Filter::Convolution(values, normalize) => {
            let mut text = "convolution(".to_string();
            for (i, v) in values.iter().enumerate() {
                if i > 0 {
                    text.push(',');
                }
                text.push_str(&format_kernel_value(*v));
            }
            text.push_str(if *normalize { ",1)" } else { ",0)" });
            text
        }
        Filter::Noise(v) => format!("noise({v})"),
        Filter::Equalize => "equalize".to_string(),
        Filter::Rotate(v) => format!("rotate({})", format_angle(*v)),
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
        Filter::Studio(spec) => {
            // Only the arguments that differ from the
            // defaults are spelled out, in grammar order
            // (bg, shadow, pad), so `studio(product)` and
            // `studio(product,pad:10)` (the default pad)
            // share one canonical form.
            let mut text = "studio(product".to_string();
            if let Some(bg) = spec.bg.text() {
                text.push_str(&format!(",bg:{bg}"));
            }
            if spec.shadow != StudioShadow::Soft {
                text.push_str(&format!(",shadow:{}", spec.shadow.name()));
            }
            if spec.pad != 10 {
                text.push_str(&format!(",pad:{}", spec.pad));
            }
            text.push(')');
            text
        }
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
        Operation::DebugFocal => "debug:focal".to_string(),
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
        Ok((ops, source, _, _)) => canonical_form(&ops, source),
        Err(_) => cache_key_form(remainder),
    }
}

/// The cache-key form of an expanded remainder: identical,
/// except every `exp:` and `maxage:` segment is removed (audit
/// P-2). Both are per-request metadata enforced or applied
/// outside the stored entry (expiry before lookup, the cache
/// lifetime on the response), so neither may fragment the
/// cache or shield entries from purges. Unparseable remainders
/// pass through unchanged.
pub fn cache_key_form(remainder: &str) -> String {
    let Ok((_, source)) = parse_operations(remainder) else {
        return remainder.to_string();
    };
    let Some(ops_part) = remainder.strip_suffix(source) else {
        return remainder.to_string();
    };
    let kept: Vec<&str> = ops_part
        .split('/')
        .filter(|seg| !seg.is_empty() && !seg.starts_with("exp:") && !seg.starts_with("maxage:"))
        .collect();
    if kept.is_empty() {
        source.to_string()
    } else {
        format!("{}/{source}", kept.join("/"))
    }
}

/// Parse a full request path into its signature and [`Request`].
#[allow(dead_code)]
pub fn parse_request(path: &str) -> Result<(Signature, Option<String>, Request<'_>), ParseError> {
    let (signature, key_id, rest) = parse_signature(path)?;
    let (operations, source_url) = parse_operations(rest)?;
    Ok((
        signature,
        key_id,
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

/// Derive the cache entry MAC key from the server secret
/// (issue #190). Persistent cache entries are MACed under
/// this derived key, beside the purge and upload domains:
/// an entry planted in the disk cache or the shared
/// bucket by anyone without the secret fails verification
/// and reads as a miss. The key is server-internal: it
/// signs stored entries, never URLs.
pub fn entry_key(secret: &[u8]) -> Vec<u8> {
    crabcrop_url::entry_key(secret)
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

/// Derive the purge-group signing key from the server
/// secret. Purge-group signatures live in their own domain
/// beside the purge domain (audit R2-14), under this
/// derived key, so a per-target purge signature can never
/// be transplanted to the purge-group endpoint.
pub fn purge_group_key(secret: &[u8]) -> Vec<u8> {
    crabcrop_url::purge_group_key(secret)
}

/// The purge-group signature for a cache group: HMAC of
/// the group name under the derived purge-group key.
/// Delegates to the crate helper so nobody
/// hand-constructs the message.
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

/// Verify a purge-group signature: an HMAC of the group
/// name under the purge-group subkey, a separate domain
/// from [`verify_purge`] (audit R2-14). Like purges, group
/// purges never honor `unsafe`.
pub fn verify_purge_group(secret: &[u8], signature: &Signature, group: &str) -> bool {
    match signature {
        Signature::Unsafe => false,
        Signature::Hmac(hex_digest) => {
            crabcrop_url::verify(&purge_group_key(secret), hex_digest, group)
        }
    }
}

/// Derive the upload signing key from the server secret
/// (issue #168). Upload signatures live in their own
/// domain beside the purge domain, under this derived
/// key, so neither an image signature nor a purge
/// signature can be replayed against the upload endpoint.
pub fn upload_key(secret: &[u8]) -> Vec<u8> {
    crabcrop_url::upload_key(secret)
}

/// The upload signature for an object name: HMAC-SHA256
/// of the name alone, under the derived upload key.
/// Delegates to the `crabcrop-url` crate. The server only
/// verifies; this mints, for operators and tests.
#[allow(dead_code)]
pub fn upload_signature(secret: &[u8], name: &str) -> String {
    crabcrop_url::upload_signature(secret, name)
}

/// The expiring upload signature for an object name
/// (issue #192): HMAC-SHA256 of `exp:<expires>/<name>`
/// under the derived upload key. Delegates to the
/// `crabcrop-url` crate; this mints, for operators and
/// tests.
#[allow(dead_code)]
pub fn upload_signature_with_expiry(secret: &[u8], name: &str, expires: u64) -> String {
    crabcrop_url::upload_signature_with_expiry(secret, name, expires)
}

/// Split an upload request remainder (what follows the
/// signature segment) into its optional expiry and the
/// object name. The plain form is the name alone; the
/// expiring form leads with an `exp:<unix-seconds>`
/// segment, the image grammar's spelling, which is
/// unambiguous here because an upload name can never
/// contain a `:` (see `crate::upload::validate_upload_name`).
/// A leading segment that starts with `exp:` but does not
/// parse is a grammar error, not a name. The signature
/// covers the remainder exactly as presented, expiry
/// included, so verification runs on the unsplit string
/// before this is called.
pub fn split_upload_remainder(rest: &str) -> Result<(Option<u64>, &str), ParseError> {
    let Some((first, name)) = rest.split_once('/') else {
        // No slash: the remainder is one segment. It is a
        // name, unless it spells an expiry with no name
        // after it, which is malformed.
        if rest.starts_with("exp:") {
            return Err(ParseError::InvalidExpiry(rest.to_string()));
        }
        return Ok((None, rest));
    };
    let Some(ts) = first.strip_prefix("exp:") else {
        return Ok((None, rest));
    };
    let exp: u64 = ts
        .parse()
        .map_err(|_| ParseError::InvalidExpiry(first.to_string()))?;
    Ok((Some(exp), name))
}

/// Verify an upload signature. Like purges, uploads are
/// a write operation and never honor `unsafe`: an HMAC
/// under the upload subkey is always required, whatever
/// the image-read mode.
pub fn verify_upload(secret: &[u8], signature: &Signature, name: &str) -> bool {
    match signature {
        Signature::Unsafe => false,
        Signature::Hmac(hex_digest) => crabcrop_url::verify(&upload_key(secret), hex_digest, name),
    }
}

/// The server's signing keys (issue #143): one current key,
/// and during a rotation window one previous key. A URL may
/// name its key with the `<key-id>:` signature prefix; the
/// verification then consults exactly that key. A URL without
/// a prefix is checked against every key in the ring, so URLs
/// minted before key IDs (or by clients that do not send one)
/// stay valid through the rotation, and age out through the
/// caches like any other entry. An unknown key ID consults
/// nothing and fails as a bad signature, indistinguishable
/// from any other forgery attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Keyring {
    pub current_id: String,
    pub current: Vec<u8>,
    pub previous: Option<(String, Vec<u8>)>,
}

impl Keyring {
    /// A ring with a single key: today's deployment shape.
    pub fn single(secret: Vec<u8>) -> Self {
        Self {
            current_id: "current".to_string(),
            current: secret,
            previous: None,
        }
    }

    fn candidates(&self, key_id: Option<&str>) -> Vec<&[u8]> {
        match key_id {
            Some(id) if id == self.current_id => vec![self.current.as_slice()],
            Some(id) => match &self.previous {
                Some((pid, secret)) if pid == id => vec![secret.as_slice()],
                _ => Vec::new(),
            },
            None => {
                let mut out = vec![self.current.as_slice()];
                if let Some((_, secret)) = &self.previous {
                    out.push(secret.as_slice());
                }
                out
            }
        }
    }

    /// Verify an image signature against the ring (see
    /// [`verify`] for the single-key semantics).
    pub fn verify(
        &self,
        key_id: Option<&str>,
        signature: &Signature,
        remainder: &str,
        allow_unsafe: bool,
    ) -> bool {
        match signature {
            Signature::Unsafe => allow_unsafe,
            Signature::Hmac(_) => self
                .candidates(key_id)
                .iter()
                .any(|secret| verify(secret, signature, remainder, false)),
        }
    }

    /// Verify a purge signature against the ring (see
    /// [`verify_purge`]).
    pub fn verify_purge(&self, key_id: Option<&str>, signature: &Signature, target: &str) -> bool {
        self.candidates(key_id)
            .iter()
            .any(|secret| verify_purge(secret, signature, target))
    }

    /// Verify a purge-group signature against the ring (see
    /// [`verify_purge_group`]).
    pub fn verify_purge_group(
        &self,
        key_id: Option<&str>,
        signature: &Signature,
        group: &str,
    ) -> bool {
        self.candidates(key_id)
            .iter()
            .any(|secret| verify_purge_group(secret, signature, group))
    }

    /// Verify an upload signature against the ring (see
    /// [`verify_upload`]).
    pub fn verify_upload(&self, key_id: Option<&str>, signature: &Signature, name: &str) -> bool {
        self.candidates(key_id)
            .iter()
            .any(|secret| verify_upload(secret, signature, name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_keyed_signature() {
        let (sig, kid, rest) = parse_signature("/k2:a1b2c3d4/300x200/example.com/img.jpg").unwrap();
        assert_eq!(sig, Signature::Hmac("a1b2c3d4".to_string()));
        assert_eq!(kid.as_deref(), Some("k2"));
        assert_eq!(rest, "300x200/example.com/img.jpg");
    }

    #[test]
    fn rejects_malformed_key_ids() {
        for path in [
            "/:a1b2c3d4/300x200/x.jpg",
            "/k ey:a1b2c3d4/300x200/x.jpg",
            "/k2:/300x200/x.jpg",
            "/k2:nothex!/300x200/x.jpg",
            "/k2:unsafe/300x200/x.jpg",
        ] {
            assert!(
                matches!(parse_signature(path), Err(ParseError::MalformedSignature)),
                "{path}"
            );
        }
        let long = format!("/{}/a1b2/300x200/x.jpg", "k".repeat(32));
        assert!(parse_signature(&long.replace("/a1b2/", ":a1b2/")).is_ok());
    }

    #[test]
    fn keyring_verifies_across_a_rotation_window() {
        let ring = Keyring {
            current_id: "k-new".to_string(),
            current: b"new-secret-value".to_vec(),
            previous: Some(("k-old".to_string(), b"old-secret-value".to_vec())),
        };
        let remainder = "300x200/example.com/img.jpg";
        let new_sig = Signature::Hmac(crabcrop_url::sign(b"new-secret-value", remainder));
        let old_sig = Signature::Hmac(crabcrop_url::sign(b"old-secret-value", remainder));
        // Keyed URLs consult exactly the named key.
        assert!(ring.verify(Some("k-new"), &new_sig, remainder, false));
        assert!(ring.verify(Some("k-old"), &old_sig, remainder, false));
        assert!(!ring.verify(Some("k-new"), &old_sig, remainder, false));
        assert!(!ring.verify(Some("k-old"), &new_sig, remainder, false));
        // Unkeyed URLs (everything minted before key IDs)
        // consult the whole ring.
        assert!(ring.verify(None, &new_sig, remainder, false));
        assert!(ring.verify(None, &old_sig, remainder, false));
        // An unknown key ID fails like any bad signature.
        assert!(!ring.verify(Some("k-who"), &new_sig, remainder, false));
        // Unsafe still follows the server flag only.
        assert!(ring.verify(None, &Signature::Unsafe, remainder, true));
        assert!(!ring.verify(Some("k-new"), &Signature::Unsafe, remainder, false));
    }

    #[test]
    fn keyring_purge_verification_keeps_domain_separation() {
        let ring = Keyring {
            current_id: "k-new".to_string(),
            current: b"new-secret-value".to_vec(),
            previous: Some(("k-old".to_string(), b"old-secret-value".to_vec())),
        };
        let target = "300x200/example.com/img.jpg";
        let old_purge = Signature::Hmac(crabcrop_url::purge_signature(b"old-secret-value", target));
        assert!(ring.verify_purge(Some("k-old"), &old_purge, target));
        assert!(ring.verify_purge(None, &old_purge, target));
        assert!(!ring.verify_purge(Some("k-new"), &old_purge, target));
        // An image signature under the old key is not a purge
        // signature under any routing.
        let image_sig = Signature::Hmac(crabcrop_url::sign(b"old-secret-value", target));
        assert!(!ring.verify_purge(Some("k-old"), &image_sig, target));
    }

    #[test]
    fn parses_unsafe_signature() {
        let (sig, _kid, rest) = parse_signature("/unsafe/300x200/example.com/img.jpg").unwrap();
        assert_eq!(sig, Signature::Unsafe);
        assert_eq!(rest, "300x200/example.com/img.jpg");
    }

    #[test]
    fn parses_hmac_signature() {
        let (sig, _kid, rest) = parse_signature("/a1b2c3d4/300x200/example.com/img.jpg").unwrap();
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
        let (sig, _kid, _) = parse_signature("/unsafe/300x200/example.com/img.jpg").unwrap();
        assert_eq!(sig, Signature::Unsafe);
    }

    #[test]
    fn works_without_leading_slash() {
        let (sig, _kid, rest) = parse_signature("unsafe/300x200/example.com/img.jpg").unwrap();
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
        let (sig, _kid, rest) =
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
        // maxage: is per-request metadata too: stripped wherever
        // it sits among the operations, kept inside the source.
        assert_eq!(
            cache_key_form(&format!("maxage:60/300x200/{src}")),
            format!("300x200/{src}")
        );
        assert_eq!(
            cache_key_form(&format!("300x200/maxage:60/{src}")),
            format!("300x200/{src}")
        );
        assert_eq!(cache_key_form(&format!("maxage:60/{src}")), src);
        let tricky = "https://example.com/maxage:60/a.jpg";
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
            (
                ParseError::InvalidFocalRect("x".into()),
                "invalid_focal_rect",
            ),
            (ParseError::InvalidPad("x".into()), "invalid_pad"),
            (ParseError::InvalidExpiry("x".into()), "invalid_expiry"),
            (ParseError::InvalidMaxAge("x".into()), "invalid_maxage"),
            (ParseError::InvalidDebug("x".into()), "invalid_debug"),
            (ParseError::UnknownPreset("x".into()), "unknown_preset"),
            (ParseError::InvalidPreset("x".into()), "invalid_preset"),
            (ParseError::PresetsOnly("x".into()), "presets_only"),
            (ParseError::InvalidFilters("x".into()), "invalid_filters"),
            (ParseError::InvalidCurve("x".into()), "invalid_curve"),
            (
                ParseError::InvalidConvolution("x".into()),
                "invalid_convolution",
            ),
            (ParseError::InvalidNoise("x".into()), "invalid_noise"),
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
        // Neither does the cache lifetime: two URLs that differ
        // only in maxage share one cache entry.
        assert_eq!(
            canonical_key_form(&format!("maxage:60/300x200/{src}")),
            canonical_key_form(&format!("maxage:31536000/300x200/{src}"))
        );
        assert_eq!(
            canonical_key_form(&format!("maxage:60/300x200/{src}")),
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
            "crop:100x100:bottom-right",
            "crop-rect:1x2:30x40",
            "focal:1x2:30x40",
            "focal:1x2:30x40/crop:100x100:smart",
            "debug:focal",
            "debug:focal/crop:100x100:smart",
            "maxage:60/300x200",
            "fit-in:300x200",
            "fit-in:300x200:adaptive",
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
            "filters:curve(rgb,0,0,255,255):curve(b,32,10,224,245)",
            "filters:convolution(0,-1,0,-1,5,-1,0,-1,0,0)",
            "filters:convolution(0.0625,0.125,0.0625,0.125,0.25,0.125,0.0625,0.125,0.0625,1)",
            "filters:noise(0)",
            "filters:noise(40)",
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
            let (ops_before, _, _, _) = parse_operations_with_expiry(&remainder).unwrap();
            let (ops_after, _, _, _) = parse_operations_with_expiry(&canonical).unwrap();
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
    fn upload_signatures_are_domain_separated() {
        let secret = b"test-secret";
        let name = "heroes/cat.jpg";
        let upload_sig = upload_signature(secret, name);
        assert_eq!(
            upload_sig,
            "81d87b588203fdad32f5431c0a893bf7ddad8188f5cca123a1b057a439b34b84"
        );
        assert!(verify_upload(
            secret,
            &Signature::Hmac(upload_sig.clone()),
            name
        ));
        // Neither an image signature nor a purge signature
        // over the same string verifies as an upload
        // signature, and the upload signature verifies in
        // neither other domain.
        let image_sig = crabcrop_url::sign(secret, name);
        let purge_sig = purge_signature(secret, name);
        assert!(!verify_upload(
            secret,
            &Signature::Hmac(image_sig.clone()),
            name
        ));
        assert!(!verify_upload(secret, &Signature::Hmac(purge_sig), name));
        assert!(!verify_purge(
            secret,
            &Signature::Hmac(upload_sig.clone()),
            name
        ));
        assert!(!verify(
            secret,
            &Signature::Hmac(upload_sig.clone()),
            name,
            false
        ));
        // Uploads never honor unsafe, in any mode.
        assert!(!verify_upload(secret, &Signature::Unsafe, name));
        // A signature for one name authorizes no other.
        assert!(!verify_upload(
            secret,
            &Signature::Hmac(upload_sig),
            "heroes/dog.jpg"
        ));
    }

    #[test]
    fn upload_remainders_split_an_optional_expiry() {
        // The plain form is the name alone.
        assert_eq!(
            split_upload_remainder("heroes/cat.jpg").unwrap(),
            (None, "heroes/cat.jpg")
        );
        assert_eq!(
            split_upload_remainder("cat.jpg").unwrap(),
            (None, "cat.jpg")
        );
        // The expiring form leads with the exp: segment.
        assert_eq!(
            split_upload_remainder("exp:1893456000/heroes/cat.jpg").unwrap(),
            (Some(1893456000), "heroes/cat.jpg")
        );
        // A segment that opens with exp: but does not parse
        // is a grammar error with the frozen code, never a
        // name (names cannot contain ':' anyway).
        for bad in [
            "exp:soon/cat.jpg",
            "exp:/cat.jpg",
            "exp:-5/cat.jpg",
            "exp:99999999999999999999999999/cat.jpg",
            "exp:1893456000",
        ] {
            let err = split_upload_remainder(bad).unwrap_err();
            assert_eq!(err.code(), "invalid_expiry", "{bad}");
        }
        // A name segment merely resembling the prefix
        // without the colon is an ordinary name.
        assert_eq!(
            split_upload_remainder("expedition/cat.jpg").unwrap(),
            (None, "expedition/cat.jpg")
        );
    }

    #[test]
    fn expiring_upload_signatures_verify_over_the_full_remainder() {
        let secret = b"test-secret";
        let name = "heroes/cat.jpg";
        let exp = 1893456000u64;
        let sig = upload_signature_with_expiry(secret, name, exp);
        let remainder = format!("exp:{exp}/{name}");
        // Verification runs over the remainder exactly as
        // presented, expiry segment included.
        assert!(verify_upload(
            secret,
            &Signature::Hmac(sig.clone()),
            &remainder
        ));
        // The same token authorizes no other name and no
        // other expiry: both are inside the signed string.
        assert!(!verify_upload(
            secret,
            &Signature::Hmac(sig.clone()),
            "exp:1893456000/heroes/dog.jpg"
        ));
        assert!(!verify_upload(
            secret,
            &Signature::Hmac(sig.clone()),
            "exp:1893456001/heroes/cat.jpg"
        ));
        // Stripping the expiry to replay the token as a
        // plain one fails, and the plain token gains no
        // expiry form by presentation.
        assert!(!verify_upload(secret, &Signature::Hmac(sig), name));
        let plain = upload_signature(secret, name);
        assert!(!verify_upload(secret, &Signature::Hmac(plain), &remainder));
    }

    #[test]
    fn keyring_upload_verification_keeps_domain_separation() {
        let ring = Keyring {
            current_id: "k-new".to_string(),
            current: b"new-secret-value".to_vec(),
            previous: Some(("k-old".to_string(), b"old-secret-value".to_vec())),
        };
        let name = "heroes/cat.jpg";
        let old_upload = Signature::Hmac(crabcrop_url::upload_signature(b"old-secret-value", name));
        assert!(ring.verify_upload(Some("k-old"), &old_upload, name));
        assert!(ring.verify_upload(None, &old_upload, name));
        assert!(!ring.verify_upload(Some("k-new"), &old_upload, name));
        // An image signature under the old key is not an
        // upload signature under any routing.
        let image_sig = Signature::Hmac(crabcrop_url::sign(b"old-secret-value", name));
        assert!(!ring.verify_upload(Some("k-old"), &image_sig, name));
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
    fn malformed_size_segments_are_parse_errors_not_sources() {
        // Issue #228: an over-cap or garbled size segment used
        // to fall out of the parse and be treated as the start
        // of a source URL, surfacing as a 502 fetch failure.
        for input in [
            "16400x100/example.com/img.jpg",
            "16384x100/example.com/img.jpg",
            "99999x100/example.com/img.jpg",
            "abcx100/example.com/img.jpg",
            "300xabc/example.com/img.jpg",
        ] {
            assert_eq!(
                parse_operations(input),
                Err(ParseError::InvalidSize(
                    input.split('/').next().unwrap().to_string()
                )),
                "input: {input:?}"
            );
        }
        // Source segments that merely contain an `x` still
        // start the source: hosts, file names, and schemes are
        // never size attempts.
        for (input, source) in [
            ("example.com/img.jpg", "example.com/img.jpg"),
            ("300x200/example.com/img.jpg", "example.com/img.jpg"),
            ("300x200/100x200.jpg", "100x200.jpg"),
        ] {
            let (_, parsed_source) = parse_operations(input).unwrap();
            assert_eq!(parsed_source, source, "input: {input:?}");
        }
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
        let (sig, _kid, rest) = parse_signature("/unsafe/300x200/example.com/img.jpg").unwrap();
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
        let (sig, _kid, rest) = parse_signature(&path).unwrap();
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
    fn parses_crop_with_corner_alignment() {
        for (spelling, want) in [
            ("top-left", Align::TopLeft),
            ("top-right", Align::TopRight),
            ("bottom-left", Align::BottomLeft),
            ("bottom-right", Align::BottomRight),
        ] {
            let path = format!("crop:300x200:{spelling}/example.com/img.jpg");
            let (ops, _) = parse_operations(&path).unwrap();
            assert!(
                matches!(
                    ops[0],
                    Operation::Crop { align, .. } if align == want
                ),
                "spelling: {spelling}"
            );
        }
    }

    #[test]
    fn rejects_bad_crops() {
        for seg in [
            "crop:",
            "crop:300",
            "crop:300x200:sideways",
            "crop:300x200:center:extra",
            // Corner spellings are vertical-first, hyphenated:
            // near-misses do not parse.
            "crop:300x200:left-top",
            "crop:300x200:top_left",
            "crop:300x200:topleft",
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
        assert_eq!(
            RequestDirectives::analyze(&ops).watermark_urls,
            ["https://x/ov.png"]
        );
    }

    #[test]
    fn numeric_filter_bounds_accept_the_endpoints() {
        // The AP-6 validation table: the endpoints are in,
        // one step past them is a parse error (pinned in
        // rejects_bad_filters).
        for seg in [
            "filters:brighten(255)",
            "filters:brighten(-255)",
            "filters:contrast(100)",
            "filters:contrast(-100)",
            "filters:saturation(0)",
            "filters:saturation(255)",
            "filters:quality(1)",
            "filters:quality(100)",
        ] {
            assert!(
                parse_operations(&format!("{seg}/example.com/img.jpg")).is_ok(),
                "{seg}"
            );
        }
    }

    #[test]
    fn parses_border_filter() {
        // The ends of the range parse: 0 (a no-op, like
        // radius) and the dimension cap; one past the cap is
        // pinned in rejects_bad_filters.
        let (ops, _) = parse_operations("filters:border(0,FF0000)/example.com/img.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![Filter::Border(0, [255, 0, 0])])]
        );
        assert!(parse_operations("filters:border(16383,FF0000)/example.com/img.jpg").is_ok());
    }

    #[test]
    fn rejects_bad_filters() {
        for seg in [
            "filters:",
            "filters:blur()",
            "filters:quality(200)",
            "filters:quality(0)",
            "filters:brighten(256)",
            "filters:brighten(-256)",
            "filters:contrast(101)",
            "filters:contrast(-100.5)",
            "filters:saturation(256)",
            "filters:saturation(-1)",
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
            "filters:border(16384,FF0000)",
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
    fn parses_curve_filter() {
        let (ops, _) = parse_operations(
            "filters:curve(rgb,0,0,255,255):curve(r,0,64,255,255)/example.com/img.jpg",
        )
        .unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![
                Filter::Curve(CurveChannel::Rgb, vec![(0, 0), (255, 255)]),
                Filter::Curve(CurveChannel::R, vec![(0, 64), (255, 255)]),
            ])]
        );
        // The point-count endpoints: 2 and 16 points parse.
        assert!(parse_operations("filters:curve(g,0,0,255,255)/example.com/img.jpg").is_ok());
        let sixteen: Vec<String> = (0..16).map(|i| format!("{},{}", i * 17, i * 17)).collect();
        assert!(
            parse_operations(&format!(
                "filters:curve(b,{})/example.com/img.jpg",
                sixteen.join(",")
            ))
            .is_ok()
        );
        // Leading zeros and a `+` sign are spelling, not
        // meaning: the canonical form prints the plain values.
        assert_eq!(
            canonical_key_form("filters:curve(rgb,007,0,+255,255)/example.com/img.jpg"),
            "filters:curve(rgb,7,0,255,255)/example.com/img.jpg"
        );
    }

    #[test]
    fn rejects_bad_curves() {
        // Every curve failure is invalid_curve, not the
        // segment-wide invalid_filters.
        for seg in [
            // Unknown channel.
            "filters:curve(x,0,0,255,255)",
            "filters:curve(RGB,0,0,255,255)",
            "filters:curve(l,0,0,255,255)",
            // Fewer than 2 points, and an odd coordinate count.
            "filters:curve(rgb)",
            "filters:curve(rgb,0,0)",
            "filters:curve(rgb,0,0,255)",
            "filters:curve(rgb,0,0,255,255,128)",
            // Coordinates outside 0-255.
            "filters:curve(rgb,0,0,256,255)",
            "filters:curve(rgb,0,-1,255,255)",
            "filters:curve(rgb,0,0,255,1.5)",
            // x not strictly increasing.
            "filters:curve(rgb,0,0,0,255)",
            "filters:curve(rgb,128,0,64,255)",
            "filters:curve(rgb,0,0,128,10,128,20)",
        ] {
            assert_eq!(
                parse_operations(&format!("{seg}/example.com/img.jpg")),
                Err(ParseError::InvalidCurve(seg.to_string())),
                "segment: {seg:?}"
            );
        }
        // 17 points is one past the cap.
        let seventeen: Vec<String> = (0..17).map(|i| format!("{},{}", i * 15, i * 15)).collect();
        let seg = format!("filters:curve(rgb,{})", seventeen.join(","));
        assert_eq!(
            parse_operations(&format!("{seg}/example.com/img.jpg")),
            Err(ParseError::InvalidCurve(seg.clone()))
        );
        // A bare `curve` with no argument list is a plain
        // invalid filters segment, like every other filter.
        assert_eq!(
            parse_operations("filters:curve/example.com/img.jpg"),
            Err(ParseError::InvalidFilters("filters:curve".to_string()))
        );
    }

    #[test]
    fn parses_convolution_filter() {
        let (ops, _) =
            parse_operations("filters:convolution(0,-1,0,-1,5,-1,0,-1,0,0)/example.com/img.jpg")
                .unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![Filter::Convolution(
                vec![0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0],
                false,
            )])]
        );
        // Decimal values (a gaussian kernel) and the flag.
        let (ops, _) = parse_operations(
            "filters:convolution(0.0625,0.125,0.0625,0.125,0.25,0.125,0.0625,0.125,0.0625,1)/example.com/img.jpg",
        )
        .unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![Filter::Convolution(
                vec![
                    0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625
                ],
                true,
            )])]
        );
        // A 5x5 kernel parses: 25 values plus the flag.
        let taps: Vec<String> = (0..25)
            .map(|i| {
                if i == 12 {
                    "1".to_string()
                } else {
                    "0".to_string()
                }
            })
            .collect();
        assert!(
            parse_operations(&format!(
                "filters:convolution({},0)/example.com/img.jpg",
                taps.join(",")
            ))
            .is_ok()
        );
        // The bound itself is allowed, on either sign.
        assert!(
            parse_operations("filters:convolution(1024,-1024,0,0,1,0,0,0,0,0)/example.com/img.jpg")
                .is_ok()
        );
        // Spelling variants collapse in the canonical form:
        // trailing fraction zeros and a `.0` are the value,
        // not the spelling.
        assert_eq!(
            canonical_key_form(
                "filters:convolution(0.50,-1.0,0,0,1,0,0,0,0,0)/example.com/img.jpg"
            ),
            "filters:convolution(0.5,-1,0,0,1,0,0,0,0,0)/example.com/img.jpg"
        );
    }

    #[test]
    fn rejects_bad_convolutions() {
        // Every convolution failure is invalid_convolution,
        // not the segment-wide invalid_filters.
        for seg in [
            // Wrong counts: 8 values, 10 values, 24 values,
            // 26 values (each plus the flag).
            "filters:convolution(0,0,0,0,1,0,0,0,0)",
            "filters:convolution(0,0,0,0,1,0,0,0,0,0,0)",
            "filters:convolution(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)",
            "filters:convolution(0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0)",
            // The flag is exactly 0 or 1.
            "filters:convolution(0,-1,0,-1,5,-1,0,-1,0,2)",
            "filters:convolution(0,-1,0,-1,5,-1,0,-1,0,-1)",
            "filters:convolution(0,-1,0,-1,5,-1,0,-1,0,true)",
            "filters:convolution(0,-1,0,-1,5,-1,0,-1,0,01)",
            // Values outside the strict decimal grammar: no
            // exponent, no leading dot, no plus, no bare dot.
            "filters:convolution(0,-1,0,-1,1e2,-1,0,-1,0,0)",
            "filters:convolution(0,-1,0,-1,.5,-1,0,-1,0,0)",
            "filters:convolution(0,-1,0,-1,+5,-1,0,-1,0,0)",
            "filters:convolution(0,-1,0,-1,5.,-1,0,-1,0,0)",
            "filters:convolution(0,-1,0,-1,x,-1,0,-1,0,0)",
            "filters:convolution(0,-1,0,-1,,-1,0,-1,0,0)",
            // Values past the kernel bound.
            "filters:convolution(0,-1,0,-1,1025,-1,0,-1,0,0)",
            "filters:convolution(0,-1,0,-1,-1024.5,-1,0,-1,0,0)",
            "filters:convolution(0,-1,0,-1,999999999999999999999,-1,0,-1,0,0)",
        ] {
            assert_eq!(
                parse_operations(&format!("{seg}/example.com/img.jpg")),
                Err(ParseError::InvalidConvolution(seg.to_string())),
                "segment: {seg:?}"
            );
        }
        // A bare `convolution` with no argument list is a
        // plain invalid filters segment, like every other
        // filter.
        assert_eq!(
            parse_operations("filters:convolution/example.com/img.jpg"),
            Err(ParseError::InvalidFilters(
                "filters:convolution".to_string()
            ))
        );
    }

    #[test]
    fn parses_noise_filter() {
        let (ops, _) = parse_operations("filters:noise(40)/example.com/img.jpg").unwrap();
        assert_eq!(ops, vec![Operation::Filters(vec![Filter::Noise(40)])]);
        // The ends of the range parse: 0 (a no-op) and 100
        // (full scale).
        assert!(parse_operations("filters:noise(0)/example.com/img.jpg").is_ok());
        assert!(parse_operations("filters:noise(100)/example.com/img.jpg").is_ok());
        // Leading zeros and a `+` sign are spelling, not
        // meaning: the canonical form prints the plain value.
        assert_eq!(
            canonical_key_form("filters:noise(007)/example.com/img.jpg"),
            "filters:noise(7)/example.com/img.jpg"
        );
        assert_eq!(
            canonical_key_form("filters:noise(+40)/example.com/img.jpg"),
            "filters:noise(40)/example.com/img.jpg"
        );
    }

    #[test]
    fn rejects_bad_noise() {
        // Every noise failure is invalid_noise, not the
        // segment-wide invalid_filters.
        for seg in [
            // Past the 0..=100 range (256 does not even
            // fit the u8 the amount parses as).
            "filters:noise(101)",
            "filters:noise(256)",
            // Not an integer in the grammar.
            "filters:noise(-1)",
            "filters:noise(1.5)",
            "filters:noise(abc)",
            "filters:noise(1,2)",
        ] {
            assert_eq!(
                parse_operations(&format!("{seg}/example.com/img.jpg")),
                Err(ParseError::InvalidNoise(seg.to_string())),
                "segment: {seg:?}"
            );
        }
        // A bare `noise` with no argument list is a plain
        // invalid filters segment, like every other filter.
        assert_eq!(
            parse_operations("filters:noise/example.com/img.jpg"),
            Err(ParseError::InvalidFilters("filters:noise".to_string()))
        );
    }

    #[test]
    fn parses_dpr_directive() {
        let (ops, _) = parse_operations("filters:dpr(2)/example.com/i.png").unwrap();
        assert_eq!(ops, vec![Operation::Filters(vec![Filter::Dpr(2.0)])]);
        assert_eq!(RequestDirectives::analyze(&ops).dpr, Some(2.0));
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
    fn parses_studio_defaults_and_arguments() {
        // Bare `studio(product)` takes every default.
        let (ops, _) = parse_operations("filters:studio(product)/example.com/img.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![Filter::Studio(StudioSpec {
                bg: StudioBg::Default,
                shadow: StudioShadow::Soft,
                pad: 10,
            })])]
        );
        assert_eq!(
            canonical_form(&ops, "example.com/img.jpg"),
            "filters:studio(product)/example.com/img.jpg"
        );
        // The full argument set, spelled in any order, parses
        // to the same spec and canonicalizes in grammar
        // order (bg, shadow, pad).
        let (ops, _) = parse_operations(
            "filters:studio(product,pad:25,shadow:strong,bg:f5f0e8)/example.com/img.jpg",
        )
        .unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![Filter::Studio(StudioSpec {
                bg: StudioBg::Color([0xf5, 0xf0, 0xe8]),
                shadow: StudioShadow::Strong,
                pad: 25,
            })])]
        );
        assert_eq!(
            canonical_form(&ops, "example.com/img.jpg"),
            "filters:studio(product,bg:f5f0e8,shadow:strong,pad:25)/example.com/img.jpg"
        );
        // Every backdrop spelling.
        for (spelling, bg) in [
            ("white", StudioBg::White),
            ("brand", StudioBg::Brand),
            ("none", StudioBg::None),
        ] {
            let (ops, _) = parse_operations(&format!(
                "filters:studio(product,bg:{spelling})/example.com/img.jpg"
            ))
            .unwrap();
            assert_eq!(
                ops,
                vec![Operation::Filters(vec![Filter::Studio(StudioSpec {
                    bg,
                    ..StudioSpec::default()
                })])]
            );
            assert_eq!(
                canonical_form(&ops, "example.com/img.jpg"),
                format!("filters:studio(product,bg:{spelling})/example.com/img.jpg")
            );
        }
        // Arguments spelled at their defaults collapse into
        // the bare form, so the two spellings share a cache
        // entry (the values ARE the defaults).
        let (ops, _) =
            parse_operations("filters:studio(product,shadow:soft,pad:10)/example.com/img.jpg")
                .unwrap();
        assert_eq!(
            canonical_form(&ops, "example.com/img.jpg"),
            "filters:studio(product)/example.com/img.jpg"
        );
        // But `bg:white` is not the default: with a brand
        // color configured they resolve differently, so
        // they key apart.
        assert_ne!(
            canonical_key_form("filters:studio(product)/example.com/img.jpg"),
            canonical_key_form("filters:studio(product,bg:white)/example.com/img.jpg")
        );
    }

    #[test]
    fn rejects_bad_studio_arguments() {
        for seg in [
            // The mode is required, and `product` is the
            // only one (studio(scene) is a separate design).
            "filters:studio",
            "filters:studio()",
            "filters:studio(scene)",
            "filters:studio(product,bg:blur)",
            "filters:studio(product,bg:fffff)",
            "filters:studio(product,bg:fffffff)",
            "filters:studio(product,shadow:hard)",
            "filters:studio(product,pad:41)",
            "filters:studio(product,pad:-1)",
            "filters:studio(product,pad:1.5)",
            "filters:studio(product,zoom:2)",
            // A key may not repeat.
            "filters:studio(product,pad:5,pad:6)",
            "filters:studio(product,bg:white,bg:none)",
        ] {
            let path = format!("{seg}/example.com/img.jpg");
            let result = parse_operations(&path);
            assert!(
                matches!(result, Err(ParseError::InvalidFilters(_))),
                "{seg} should fail as invalid filters, got {result:?}"
            );
        }
    }

    #[test]
    fn studio_marks_the_request_as_ai() {
        let (ops, _) = parse_operations("filters:studio(product)/example.com/img.jpg").unwrap();
        let d = RequestDirectives::analyze(&ops);
        assert!(d.studio);
        assert!(
            d.uses_ai(),
            "studio runs the mask model, so it takes an AI permit"
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
    fn parses_video_formats() {
        for (spelling, format) in [("mp4", ImageFormat::Mp4), ("webm", ImageFormat::WebM)] {
            let (ops, _) =
                parse_operations(&format!("filters:format({spelling})/example.com/img.gif"))
                    .unwrap();
            assert_eq!(ops, vec![Operation::Filters(vec![Filter::Format(format)])]);
            assert!(format.is_video());
            // The canonical form spells them back unchanged.
            assert_eq!(
                canonical_form(&ops, "example.com/img.gif"),
                format!("filters:format({spelling})/example.com/img.gif")
            );
        }
        assert!(!ImageFormat::Gif.is_video());
    }

    #[test]
    fn parses_fit_in() {
        let (ops, _) = parse_operations("fit-in:300x200/example.com/img.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::FitIn {
                size: ImageSize {
                    width: Dimension::Pixels(300),
                    height: Dimension::Pixels(200),
                    flip_horizontal: false,
                    flip_vertical: false,
                },
                adaptive: false,
            }]
        );
        let (ops, _) = parse_operations("fit-in:300x0/example.com/img.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::FitIn {
                size: ImageSize {
                    width: Dimension::Pixels(300),
                    height: Dimension::Proportional,
                    flip_horizontal: false,
                    flip_vertical: false,
                },
                adaptive: false,
            }]
        );
    }

    #[test]
    fn parses_adaptive_fit_in() {
        let (ops, _) = parse_operations("fit-in:300x200:adaptive/example.com/img.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::FitIn {
                size: ImageSize {
                    width: Dimension::Pixels(300),
                    height: Dimension::Pixels(200),
                    flip_horizontal: false,
                    flip_vertical: false,
                },
                adaptive: true,
            }]
        );
        // The modifier is part of the operation: it stays in the
        // canonical form, so an adaptive URL keys apart from the
        // plain fit-in of the same box.
        assert_eq!(
            canonical_form(&ops, "example.com/img.jpg"),
            "fit-in:300x200:adaptive/example.com/img.jpg"
        );
        assert_ne!(
            canonical_key_form("fit-in:300x200:adaptive/example.com/img.jpg"),
            canonical_key_form("fit-in:300x200/example.com/img.jpg")
        );
    }

    #[test]
    fn rejects_bad_fit_in() {
        for seg in [
            "fit-in:-300x200",
            "fit-in:wide",
            "fit-in:",
            // The modifier slot takes exactly `adaptive`, once.
            "fit-in:300x200:sideways",
            "fit-in:300x200:adaptive:extra",
            "fit-in:300x200:",
            "fit-in:300x200:Adaptive",
            // `adaptive` is a modifier, never the box itself.
            "fit-in:adaptive",
            "fit-in:adaptive:300x200",
            // A flipped box stays invalid with the modifier.
            "fit-in:-300x200:adaptive",
        ] {
            let path = format!("{seg}/example.com/img.jpg");
            assert_eq!(
                parse_operations(&path),
                Err(ParseError::InvalidSize(seg.to_string())),
                "{seg}"
            );
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
    fn parses_focal_rect() {
        let (ops, _) = parse_operations("focal:10x20:110x220/example.com/img.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::FocalRect {
                left: 10,
                top: 20,
                right: 110,
                bottom: 220,
            }]
        );
    }

    #[test]
    fn rejects_bad_focal_rect() {
        for seg in [
            "focal:10x20",
            "focal:110x220:10x20",
            "focal:10x20:10x220",
            "focal:10x20:110x20",
            "focal:-1x0:10x10",
            "focal:axbx:cxdx",
        ] {
            let path = format!("{seg}/example.com/img.jpg");
            assert!(
                matches!(
                    parse_operations(&path),
                    Err(ParseError::InvalidFocalRect(_))
                ),
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
        assert_eq!(
            RequestDirectives::analyze(&ops).watermark_urls,
            ["a/1.png", "b/2.png"]
        );
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
        assert!(RequestDirectives::analyze(&ops).needs_face_model);
        assert_eq!(RequestDirectives::analyze(&ops).face_padding, 0.5);

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
        assert!(RequestDirectives::analyze(&ops).needs_face_model);
        assert_eq!(RequestDirectives::analyze(&ops).face_padding, 1.5);

        // Smart crops and plain requests do not require a model.
        let (ops, _) = parse_operations("crop:300x300:smart/example.com/img.jpg").unwrap();
        assert!(!RequestDirectives::analyze(&ops).needs_face_model);

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
        assert!(RequestDirectives::analyze(&ops).has_class_redaction());
        assert_eq!(
            RequestDirectives::analyze(&ops).classes,
            ["person", "traffic light"]
        );
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
            // `adaptive` belongs to fit-in: pad's canvas is
            // explicit, so the modifier is not a pad fill.
            "pad:200x100:adaptive",
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
        let (ops, src, exp, _) =
            parse_operations_with_expiry("exp:1893456000/300x200/example.com/img.jpg").unwrap();
        assert_eq!(exp, Some(1893456000));
        assert_eq!(ops.len(), 1);
        assert_eq!(src, "example.com/img.jpg");
        // Position-independent among the operations; last wins.
        let (_, _, exp, _) =
            parse_operations_with_expiry("300x200/exp:100/exp:200/example.com/img.jpg").unwrap();
        assert_eq!(exp, Some(200));
        // Once the source URL starts, `exp:` is just URL text.
        let (ops, src, exp, _) = parse_operations_with_expiry("example.com/exp:123").unwrap();
        assert!(ops.is_empty() && exp.is_none());
        assert_eq!(src, "example.com/exp:123");
        // Absent means no expiry.
        let (_, _, exp, _) = parse_operations_with_expiry("300x200/example.com/img.jpg").unwrap();
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

    #[test]
    fn parses_maxage_segments() {
        let (ops, src, _, max_age) =
            parse_operations_with_expiry("maxage:60/300x200/example.com/img.jpg").unwrap();
        assert_eq!(max_age, Some(60));
        assert_eq!(ops.len(), 1);
        assert_eq!(src, "example.com/img.jpg");
        // Position-independent among the operations; last wins.
        let (_, _, _, max_age) =
            parse_operations_with_expiry("300x200/maxage:100/maxage:200/example.com/img.jpg")
                .unwrap();
        assert_eq!(max_age, Some(200));
        // Zero is valid (revalidate every time), and so is the
        // one-year cap itself.
        let (_, _, _, max_age) =
            parse_operations_with_expiry("maxage:0/example.com/img.jpg").unwrap();
        assert_eq!(max_age, Some(0));
        let (_, _, _, max_age) =
            parse_operations_with_expiry("maxage:31536000/example.com/img.jpg").unwrap();
        assert_eq!(max_age, Some(MAX_CACHE_AGE));
        // Once the source URL starts, `maxage:` is just URL text.
        let (ops, src, _, max_age) = parse_operations_with_expiry("example.com/maxage:60").unwrap();
        assert!(ops.is_empty() && max_age.is_none());
        assert_eq!(src, "example.com/maxage:60");
        // Absent means the server default.
        let (_, _, _, max_age) =
            parse_operations_with_expiry("300x200/example.com/img.jpg").unwrap();
        assert_eq!(max_age, None);
        // Malformed and out-of-range values are parse errors of
        // the dedicated kind.
        for bad in [
            "maxage:/example.com/i.jpg",
            "maxage:abc/example.com/i.jpg",
            "maxage:-1/example.com/i.jpg",
            "maxage:12.5/example.com/i.jpg",
            "maxage:31536001/example.com/i.jpg",
            "maxage:99999999999999999999999999/example.com/i.jpg",
        ] {
            assert!(
                matches!(
                    parse_operations_with_expiry(bad),
                    Err(ParseError::InvalidMaxAge(_))
                ),
                "{bad}"
            );
        }
        // The two metadata kinds extract side by side.
        let (_, _, exp, max_age) =
            parse_operations_with_expiry("exp:1893456000/maxage:60/300x200/example.com/img.jpg")
                .unwrap();
        assert_eq!(exp, Some(1893456000));
        assert_eq!(max_age, Some(60));
    }

    #[test]
    fn parses_debug_focal_segments() {
        let (ops, src, _, _) =
            parse_operations_with_expiry("debug:focal/crop:100x100:smart/example.com/img.jpg")
                .unwrap();
        assert_eq!(
            ops,
            vec![
                Operation::DebugFocal,
                Operation::Crop {
                    size: ImageSize {
                        width: Dimension::Pixels(100),
                        height: Dimension::Pixels(100),
                        flip_horizontal: false,
                        flip_vertical: false,
                    },
                    align: Align::Smart,
                },
            ]
        );
        assert_eq!(src, "example.com/img.jpg");
        // The directive analysis sees it.
        assert!(RequestDirectives::analyze(&ops).debug_focal);
        // It is an operation, not metadata: it stays in the
        // canonical form in place, so a debug URL keys apart
        // from the same URL without it (and the server bypasses
        // the cache for it regardless).
        let with = canonical_key_form("debug:focal/crop:100x100:smart/example.com/img.jpg");
        let without = canonical_key_form("crop:100x100:smart/example.com/img.jpg");
        assert_eq!(with, "debug:focal/crop:100x100:smart/example.com/img.jpg");
        assert_ne!(with, without);
        // Once the source URL starts, `debug:` is just URL text.
        let (ops, src, _, _) = parse_operations_with_expiry("example.com/debug:focal").unwrap();
        assert!(ops.is_empty());
        assert_eq!(src, "example.com/debug:focal");
        // Any other debug view is a parse error of the
        // dedicated kind.
        for bad in [
            "debug:/example.com/i.jpg",
            "debug:banana/example.com/i.jpg",
            "debug:focalx/example.com/i.jpg",
            "debug:smart/example.com/i.jpg",
        ] {
            let err = parse_operations_with_expiry(bad).unwrap_err();
            assert!(matches!(err, ParseError::InvalidDebug(_)), "{bad}");
            assert_eq!(err.code(), "invalid_debug");
        }
    }

    #[test]
    fn debug_focal_requests_count_as_ai_use() {
        // `debug:focal` runs the face and focal providers
        // and is never cached, so the directive analysis
        // classes it as AI use: the server holds an AI-gate
        // permit for it (audit R2-17).
        let (ops, _, _, _) = parse_operations_with_expiry("debug:focal/example.com/i.jpg").unwrap();
        assert!(RequestDirectives::analyze(&ops).uses_ai());
        // A plain transform is not AI use.
        let (ops, _, _, _) = parse_operations_with_expiry("100x100/example.com/i.jpg").unwrap();
        assert!(!RequestDirectives::analyze(&ops).uses_ai());
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
        // Source-like segments, expiry, cache-lifetime metadata,
        // nested presets, and the empty string are all invalid
        // definitions.
        for bad in [
            "",
            "example.com/img.jpg",
            "exp:123/300x200",
            "maxage:60/300x200",
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
        // maxage: metadata walks like exp: it neither stops the
        // expansion nor is expanded itself, on either side of a
        // preset reference.
        assert_eq!(
            expand_presets("maxage:60/preset:thumb/example.com/img.jpg", &map, false).unwrap(),
            "maxage:60/300x200/example.com/img.jpg"
        );
        assert_eq!(
            expand_presets("preset:thumb/maxage:60/example.com/img.jpg", &map, false).unwrap(),
            "300x200/maxage:60/example.com/img.jpg"
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
        assert!(expand_presets("maxage:60/preset:thumb/example.com/img.jpg", &map, true).is_ok());
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
            ("rotate(90)", Filter::Rotate(90.0)),
            ("rotate(-90)", Filter::Rotate(-90.0)),
            ("rotate(45)", Filter::Rotate(45.0)),
            ("rotate(45.5)", Filter::Rotate(45.5)),
            ("rotate(-30.25)", Filter::Rotate(-30.25)),
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
    fn default_ops_parse_and_expand() {
        let mut presets = std::collections::HashMap::new();
        presets.insert(
            "thumb".to_string(),
            "200x200/filters:format(webp)".to_string(),
        );
        // Plain operations.
        let ops = parse_default_ops("filters:quality(80)", &presets).unwrap();
        assert_eq!(ops.len(), 1);
        // Preset references expand at parse (boot) time.
        let ops = parse_default_ops("preset:thumb", &presets).unwrap();
        assert_eq!(ops.len(), 2);
        // Empty means none.
        assert!(parse_default_ops("", &presets).unwrap().is_empty());
        assert!(parse_default_ops("  ", &presets).unwrap().is_empty());
        // Unknown presets and bad operations are errors.
        assert!(parse_default_ops("preset:missing", &presets).is_err());
        assert!(parse_default_ops("filters:blur(0)", &presets).is_err());
        assert!(parse_default_ops("nonsense!!", &presets).is_err());
    }

    #[test]
    fn default_ops_merge_drops_overridden_singletons() {
        let presets = std::collections::HashMap::new();
        let defaults =
            parse_default_ops("filters:format(webp):quality(80):blur(2)", &presets).unwrap();
        // A bare request keeps every default.
        let (bare, _) = parse_operations("example.com/i.jpg").unwrap();
        let merged = merge_default_ops(&defaults, bare);
        assert_eq!(
            canonical_form(&merged, "example.com/i.jpg"),
            "filters:format(webp):quality(80):blur(2)/example.com/i.jpg"
        );
        // A request that spells out format and quality drops
        // those defaults; the surviving default (blur) runs
        // first.
        let (explicit, _) =
            parse_operations("filters:format(png):quality(60)/example.com/i.jpg").unwrap();
        let merged = merge_default_ops(&defaults, explicit);
        assert_eq!(
            canonical_form(&merged, "example.com/i.jpg"),
            "filters:blur(2):format(png):quality(60)/example.com/i.jpg"
        );
        // Non-singleton defaults stack with the request's own
        // filters instead of being dropped.
        let (blurred, _) = parse_operations("filters:blur(3)/example.com/i.jpg").unwrap();
        let merged = merge_default_ops(&defaults, blurred);
        assert_eq!(
            canonical_form(&merged, "example.com/i.jpg"),
            "filters:format(webp):quality(80):blur(2):blur(3)/example.com/i.jpg"
        );
    }

    #[test]
    fn rotate_angles_canonicalize() {
        // Whole spellings keep the historic integer form (and
        // its cache key); fractions use the shortest form.
        assert_eq!(
            canonical_key_form("filters:rotate(90.0)/example.com/img.jpg"),
            "filters:rotate(90)/example.com/img.jpg"
        );
        assert_eq!(
            canonical_key_form("filters:rotate(45.50)/example.com/img.jpg"),
            "filters:rotate(45.5)/example.com/img.jpg"
        );
        assert_eq!(
            canonical_key_form("filters:rotate(-90)/example.com/img.jpg"),
            "filters:rotate(-90)/example.com/img.jpg"
        );
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
            "rotate(1e2)",
            "rotate(.5)",
            "rotate(45.)",
            "rotate(nan)",
            "rotate(36001)",
            "rotate(-)",
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
            "filters:blur(513)",
            "filters:blur(512.5)",
            "filters:blur(1000000000)",
            "filters:blur(1e-45)",
            "filters:blur(1e-40)",
            "filters:blur(inf)",
            "filters:blur(NaN)",
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
    fn blur_sigma_cap_boundary() {
        // The cap itself, and a small normal sigma, are accepted.
        let (ops, _) = parse_operations("filters:blur(512)/example.com/img.jpg").unwrap();
        assert_eq!(ops, vec![Operation::Filters(vec![Filter::Blur(512.0)])]);
        let (ops, _) = parse_operations("filters:blur(0.001)/example.com/img.jpg").unwrap();
        assert_eq!(ops, vec![Operation::Filters(vec![Filter::Blur(0.001)])]);
        // Just over the cap is invalid_filters, like any other
        // out-of-range filter parameter.
        assert_eq!(
            parse_operations("filters:blur(512.5)/example.com/img.jpg"),
            Err(ParseError::InvalidFilters(
                "filters:blur(512.5)".to_string()
            ))
        );
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
        // known prefix; otherwise it starts the source URL. (A segment
        // shaped like a size is the one exception: it errors instead,
        // see malformed_size_segments_are_parse_errors_not_sources.)
        let (ops, url) = parse_operations("photos/img.jpg").unwrap();
        assert!(ops.is_empty());
        assert_eq!(url, "photos/img.jpg");
    }

    #[test]
    fn parse_request_composes_signature_and_operations() {
        let (sig, _kid, req) =
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
        assert_eq!(
            RequestDirectives::analyze(&ops).explicit_format,
            Some(ImageFormat::Png)
        );
        assert!(RequestDirectives::analyze(&ops).no_upscale);
        let (ops, _) = parse_operations("100x100/example.com/i.jpg").unwrap();
        assert_eq!(RequestDirectives::analyze(&ops).explicit_format, None);
        assert!(!RequestDirectives::analyze(&ops).no_upscale);
        // A later format filter wins over an earlier one.
        let (ops, _) =
            parse_operations("filters:format(jpeg):format(webp)/example.com/i.jpg").unwrap();
        assert_eq!(
            RequestDirectives::analyze(&ops).explicit_format,
            Some(ImageFormat::WebP)
        );
    }
}
