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

mod filters;

use filters::{filter_text, read_filter};

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

/// Why a request path failed to parse: its signature segment,
/// an operation, a preset or source-root reference, or the
/// source.
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
    /// A `focalpoint:...` segment did not match
    /// `focalpoint:<x>x<y>[:<zoom>]` with both fractions in
    /// 0..=1 and a zoom in (0, [`MAX_FOCAL_ZOOM`]]. Carries
    /// the offending segment.
    InvalidFocalPoint(String),
    /// A `pad:...` segment did not match `pad:<WxH>[:<fill>]`.
    /// Carries the offending segment.
    InvalidPad(String),
    /// An `exp:...` segment did not carry a valid unix
    /// timestamp, or sat in a server-wide default operations
    /// string, where per-request metadata is refused (issue
    /// #469). Carries the offending segment.
    InvalidExpiry(String),
    /// A `maxage:...` segment did not carry a valid cache
    /// lifetime: whole seconds from 0 to [`MAX_CACHE_AGE`].
    /// Also refused in a server-wide default operations
    /// string, like `exp:` (issue #469). Carries the
    /// offending segment.
    InvalidMaxAge(String),
    /// A `debug:...` segment named a debug view the grammar
    /// does not have: the only one is `debug:focal`. Carries
    /// the offending segment.
    InvalidDebug(String),
    /// A `frame:...` segment did not match the frame
    /// selector grammar: `frame:poster`, `frame:<seconds>`,
    /// or `frame:<percent>%`. Carries the offending segment.
    InvalidFrame(String),
    /// A `page:...` segment did not match the page
    /// selector grammar: `page:<n>` with `n` a 1-based
    /// page number of ASCII digits within the grammar
    /// bound. Carries the offending segment.
    InvalidPage(String),
    /// A `preset:<name>` segment named a preset the server has
    /// not configured. Carries the name.
    UnknownPreset(String),
    /// A `root:<name>` source named a source root the server
    /// has not configured. Carries the name.
    UnknownRoot(String),
    /// A `root:<name>/<path>` source is malformed: no path,
    /// a path that normalizes to nothing, or a path that
    /// climbs above the root (including by percent-encoded
    /// dot segments or encoded separators). Carries the
    /// offending source.
    InvalidRoot(String),
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
    /// The operation list a request would execute, after
    /// preset expansion and with the server-wide defaults
    /// merged in, runs past [`MAX_OPERATIONS`] (see
    /// [`check_operation_count`]). Carries the count.
    TooManyOperations(usize),
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
            ParseError::InvalidFocalPoint(_) => "invalid_focal_point",
            ParseError::InvalidPad(_) => "invalid_pad",
            ParseError::InvalidExpiry(_) => "invalid_expiry",
            ParseError::InvalidMaxAge(_) => "invalid_maxage",
            ParseError::InvalidDebug(_) => "invalid_debug",
            ParseError::InvalidFrame(_) => "invalid_frame",
            ParseError::InvalidPage(_) => "invalid_page",
            ParseError::UnknownPreset(_) => "unknown_preset",
            ParseError::UnknownRoot(_) => "unknown_root",
            ParseError::InvalidRoot(_) => "invalid_root",
            ParseError::InvalidPreset(_) => "invalid_preset",
            ParseError::PresetsOnly(_) => "presets_only",
            ParseError::InvalidFilters(_) => "invalid_filters",
            ParseError::InvalidCurve(_) => "invalid_curve",
            ParseError::InvalidConvolution(_) => "invalid_convolution",
            ParseError::InvalidNoise(_) => "invalid_noise",
            ParseError::MissingSourceUrl => "missing_source_url",
            // The engine's code for the same refusal, which it
            // answered alone until the check moved to parse
            // time (issue #469): moving it earlier changed
            // when the 400 answers, not what it says.
            ParseError::TooManyOperations(_) => "unsupported_operation",
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
            ParseError::InvalidFocalPoint(seg) => {
                write!(f, "invalid focalpoint segment: {seg}")
            }
            ParseError::InvalidPad(seg) => write!(f, "invalid pad segment: {seg}"),
            ParseError::InvalidExpiry(seg) => write!(f, "invalid expiry segment: {seg}"),
            ParseError::InvalidMaxAge(seg) => write!(f, "invalid maxage segment: {seg}"),
            ParseError::InvalidDebug(seg) => write!(f, "invalid debug segment: {seg}"),
            ParseError::InvalidFrame(seg) => write!(f, "invalid frame segment: {seg}"),
            ParseError::InvalidPage(seg) => write!(f, "invalid page segment: {seg}"),
            ParseError::UnknownPreset(name) => write!(f, "unknown preset: {name}"),
            ParseError::UnknownRoot(name) => write!(f, "unknown source root: {name}"),
            ParseError::InvalidRoot(source) => {
                write!(f, "invalid root source: {source}")
            }
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
            ParseError::TooManyOperations(count) => write!(
                f,
                "request has {count} operations, over the {MAX_OPERATIONS} operation limit"
            ),
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
#[derive(Debug, Clone, PartialEq)]
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
    /// Crop to the union of the detected objects of the
    /// named classes (issue #318): the crop window frames
    /// the union box, scaled by `scale` around the box's
    /// center (1 is the tight union; larger values add
    /// context around it), and resolves to the target
    /// aspect the way the face crop resolves around its
    /// subject. When detection finds none of the classes
    /// the crop falls back to the plain center crop of the
    /// same size.
    Objects { classes: Vec<String>, scale: f32 },
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Quality {
    /// An explicit value, 0-100.
    Value(u8),
    /// Chosen per image by content analysis at encode time.
    Auto,
    /// Per-format values from one `quality(...)` filter
    /// (issue #319): `quality(80,jpeg:90,webp:75)` carries a
    /// value per output format so one signed URL serves every
    /// negotiated format at its intended quality. Entries are
    /// `(format, value)` pairs; the `None` key is the fallback
    /// for formats with no entry of their own. Resolution
    /// happens after format negotiation, in
    /// [`Quality::resolved_for`].
    PerFormat(Vec<(Option<ImageFormat>, QualityValue)>),
}

/// One value inside a per-format `quality(...)` list: the
/// same two forms a standalone quality takes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityValue {
    /// An explicit value, 1-100.
    Value(u8),
    /// Chosen per image by content analysis at encode time.
    Auto,
}

impl QualityValue {
    fn to_quality(self) -> Quality {
        match self {
            QualityValue::Value(v) => Quality::Value(v),
            QualityValue::Auto => Quality::Auto,
        }
    }
}

impl Quality {
    /// The quality in force once the output format is known:
    /// a plain value or `auto` resolves to itself, a per-format
    /// list resolves to the entry for `format`, else its
    /// fallback, else `None` (no quality directive in force
    /// for this format, so encoder defaults and the Save-Data
    /// step-down apply as if quality were absent).
    pub fn resolved_for(&self, format: ImageFormat) -> Option<Quality> {
        match self {
            Quality::Value(v) => Some(Quality::Value(*v)),
            Quality::Auto => Some(Quality::Auto),
            Quality::PerFormat(entries) => {
                let mut fallback = None;
                for (key, value) in entries {
                    match key {
                        Some(f) if *f == format => return Some(value.to_quality()),
                        None => fallback = Some(value.to_quality()),
                        _ => {}
                    }
                }
                fallback
            }
        }
    }
}

/// Chroma subsampling as spelled in `format(...)` options
/// (`subsampling:444` and friends), the ratio notation the
/// JPEG world uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subsampling {
    /// 4:4:4, no subsampling.
    S444,
    /// 4:2:2, chroma halved horizontally.
    S422,
    /// 4:2:0, chroma halved in both directions.
    S420,
}

impl Subsampling {
    /// The canonical spelling, as written in the option.
    pub fn name(self) -> &'static str {
        match self {
            Subsampling::S444 => "444",
            Subsampling::S422 => "422",
            Subsampling::S420 => "420",
        }
    }

    fn parse(s: &str) -> Option<Subsampling> {
        Some(match s {
            "444" => Subsampling::S444,
            "422" => Subsampling::S422,
            "420" => Subsampling::S420,
            _ => return None,
        })
    }
}

/// Per-format encoder options carried as arguments on a
/// `format(...)` filter (issue #319): `format(jpeg,progressive)`,
/// `format(webp,method:6)`. Every field defaults to the
/// encoder's standing behavior, so a format filter with no
/// options renders byte for byte as it did before the options
/// existed. Only the fields for the named format are ever
/// set; the parser rejects an option its format cannot
/// honour rather than letting it be silently ignored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FormatOptions {
    /// JPEG: progressive (multi-scan) encoding.
    pub jpeg_progressive: bool,
    /// JPEG: chroma subsampling. `None` keeps the encoder's
    /// standing 4:2:0.
    pub jpeg_subsampling: Option<Subsampling>,
    /// PNG: Adam7 interlacing.
    pub png_interlace: bool,
    /// WebP: libwebp compression method, 0-6. `None` keeps
    /// the standing method 4.
    pub webp_method: Option<u8>,
    /// WebP: libwebp smart subsampling (spatial noise
    /// shaping) strength, 0-100. `None` keeps the libwebp
    /// default.
    pub webp_sns: Option<u8>,
}

impl FormatOptions {
    /// Whether no option is set: the filter renders exactly
    /// as a bare `format(...)` always has.
    pub fn is_empty(&self) -> bool {
        *self == FormatOptions::default()
    }
}

/// A `format(...)` filter: the output format plus its
/// per-format encoder options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatSpec {
    /// The output format.
    pub format: ImageFormat,
    /// The format's encoder options (all default when the
    /// filter named no options).
    pub options: FormatOptions,
}

impl FormatSpec {
    /// A format with no options: every pre-options URL.
    pub fn plain(format: ImageFormat) -> FormatSpec {
        FormatSpec {
            format,
            options: FormatOptions::default(),
        }
    }
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

/// The studio mode a `studio(...)` filter runs (issues
/// #197 and #198): `product` cuts the subject out onto a
/// backdrop; `scene` keeps the photograph's scene and
/// reframes it on the subject instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StudioMode {
    Product,
    Scene,
}

impl StudioMode {
    /// The canonical spelling inside `studio(...)`.
    pub fn name(self) -> &'static str {
        match self {
            StudioMode::Product => "product",
            StudioMode::Scene => "scene",
        }
    }
}

/// The `studio(...)` argument set, with the defaults
/// already applied: the `product` mode (issue #197), an
/// unnamed backdrop (`StudioBg::Default`), a soft shadow,
/// and a pad of 10 percent. Under `scene` (issue #198)
/// only `pad` applies: it is the margin the reframed
/// subject keeps, with the same range and default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StudioSpec {
    pub mode: StudioMode,
    pub bg: StudioBg,
    pub shadow: StudioShadow,
    pub pad: u8,
}

impl Default for StudioSpec {
    fn default() -> Self {
        StudioSpec {
            mode: StudioMode::Product,
            bg: StudioBg::Default,
            shadow: StudioShadow::Soft,
            pad: 10,
        }
    }
}

/// Parse the `studio(...)` argument list: the mode
/// (`product` or `scene`) followed by named `key:value`
/// arguments. The product keys are `bg`, `shadow`, and
/// `pad`; scene takes `pad` only, because a backdrop and
/// a shadow are product concepts (naming one under scene
/// is a bad argument, not a silent no-op). Bad arguments
/// fail as the segment-wide invalid filter error, like
/// every other filter: the frozen ParseError set does not
/// grow for studio.
fn parse_studio(p: &str) -> Option<StudioSpec> {
    let mut spec = StudioSpec::default();
    let mut seen: Vec<&str> = Vec::new();
    let mut parts = p.split(',');
    match parts.next() {
        Some("product") => spec.mode = StudioMode::Product,
        Some("scene") => spec.mode = StudioMode::Scene,
        _ => return None,
    }
    for part in parts {
        let (key, value) = part.split_once(':')?;
        if seen.contains(&key) {
            return None;
        }
        seen.push(key);
        match (spec.mode, key) {
            (StudioMode::Product, "bg") => spec.bg = StudioBg::parse(value)?,
            (StudioMode::Product, "shadow") => spec.shadow = StudioShadow::parse(value)?,
            (_, "pad") => {
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
    /// Gamma correction with the given divisor: each
    /// channel value `v` becomes `255 * (v / 255)^(1 / g)`,
    /// the levels-style midtone gamma. 1 is identity,
    /// values above 1 brighten the midtones, values below
    /// 1 darken them. The grammar accepts 0 < g <= 10
    /// (issue #313); the scale is the exponent itself, the
    /// one scale comparable products agree on for gamma.
    Gamma(f32),
    /// Exposure adjustment in stops: each channel value is
    /// converted to linear light, multiplied by `2^stops`,
    /// and converted back. 0 is identity, positive stops
    /// brighten, negative stops darken. The grammar accepts
    /// -10 to 10 stops ([`MAX_EXPOSURE`]). The scale is the
    /// photographic stop itself, stated explicitly because
    /// it is not a percent (issue #317); unlike `brighten`,
    /// whose shift is additive, a stop multiplies, so it
    /// behaves the same at every brightness.
    Exposure(f32),
    /// Highlights and shadows: one tonal filter with a
    /// percent argument per end of the tonal scale, each
    /// -100..=100 in the adjustment family's percent
    /// convention (issue #317). The first argument moves
    /// bright pixels most, the second dark pixels most,
    /// each weighted by the pixel's luminance, so the two
    /// ranges lift or compress independently; `contrast`
    /// cannot express this, because it moves both ends
    /// together. 0 in both is the identity.
    HighlightsShadows(f32, f32),
    /// Vibrance: saturation weighted inversely to the
    /// saturation a pixel already has, in the adjustment
    /// family's percent convention, -100..=100 (issue
    /// #317). Muted colors gain (or lose) the most and
    /// already-vivid colors barely move, the gentler,
    /// skin-tone-friendly sibling of `saturation`, which
    /// scales every pixel by the same factor. 0 is the
    /// identity.
    Vibrance(f32),
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
    /// Gentle per-channel auto-levels: each channel's
    /// 0.5th percentile maps to 0 and its 99.5th to 255
    /// (issue #317), the conservative clipping the studio
    /// scene correction uses, with a channel whose
    /// percentile span is too small left unmapped.
    /// Deliberately weaker than `equalize`, which
    /// redistributes the whole histogram: enhance
    /// stretches a compressed range toward the ends and
    /// leaves a well-exposed image nearly alone.
    Enhance,
    /// Rotate by the given angle in degrees (issue #144):
    /// multiples of 90 take the exact quarter-turn path, any
    /// other angle rotates geometrically onto an expanded
    /// canvas.
    Rotate(f32),
    /// Crop away the uniform border, keeping pixels that differ
    /// from the corner color by more than this tolerance (0-255).
    Trim(u8),
    /// Trim at this tolerance, then remove the named pixel
    /// amounts from the left, top, right, and bottom of what
    /// remains (issue #313). The amounts are plain pixel
    /// counts, clamped by the engine so at least one pixel
    /// survives on each axis.
    TrimSides(u8, [u32; 4]),
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
    /// Studio mode (issues #197 and #198): one filter
    /// naming an outcome, expanded deterministically in the
    /// engine and reported in meta. `product` composites
    /// the subject the segmentation model finds onto a
    /// backdrop with a synthesized contact shadow, scaled
    /// and centered to a pad margin; `scene` keeps the
    /// photograph's scene, reframing it on the detected
    /// subject and applying a per-image light and color
    /// correction.
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
    /// Recolor every detected object of the named class to
    /// the given RGB color (issue #175), backed by the
    /// server's detection model. With a segment provider
    /// configured the paint follows the object's
    /// silhouette; without one it fills the box.
    RecolorClass(String, [u8; 3]),
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
    /// Erase the rectangle (x, y, w, h) and inpaint it from
    /// the surrounding pixels, same coordinate convention
    /// and bounds discipline as `BlurRegion`. Backed by the
    /// server's inpainting model (issue #176).
    RemoveRegion { x: u32, y: u32, w: u32, h: u32 },
    /// Encoding directive: re-encode at lower quality until the
    /// output fits this many bytes, quality permitting.
    MaxBytes(u64),
    /// Encoding quality.
    Quality(Quality),
    /// Encoding directive: write PNG output as an indexed
    /// (palette) image with at most this many colors, from
    /// 2 to 256 (issue #313). The other formats ignore it,
    /// the same stance `quality` takes on PNG.
    Palette(u16),
    /// Output container.
    Format(FormatSpec),
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

/// Which frame of a video source a `frame:` segment
/// selects (issue #214). The selection is resolved against
/// the probed duration at decode time: seconds and poster
/// need no duration, a percentage does. Out-of-range
/// selections clamp to the last frame there, never here:
/// the grammar accepts any value inside its own bounds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FrameSelection {
    /// `frame:poster`: the poster frame, the video's first
    /// frame.
    Poster,
    /// `frame:<seconds>`: the frame at this timestamp, in
    /// seconds from the start.
    Seconds(f64),
    /// `frame:<percent>%`: the frame at this percentage of
    /// the duration, 0 to 100.
    Percent(f64),
}

/// The largest timestamp a `frame:` segment may name, in
/// seconds: 24 hours. Real sources the loader will fetch
/// (10 MiB by default, 1 GiB at most) almost always run
/// shorter, and a percentage reaches any point of a longer
/// one; the cap keeps the grammar's numbers finite and
/// small, and anything past a real duration clamps to the
/// last frame anyway.
pub const MAX_FRAME_SECONDS: f64 = 86_400.0;

/// The largest percentage a `frame:` segment may name.
pub const MAX_FRAME_PERCENT: f64 = 100.0;

/// The largest page number a `page:` segment may name
/// (issue #215). Pages are 1-based ordinals, so zero is a
/// grammar error, not a clamp to the first page. The bound
/// keeps the grammar's numbers finite and small in the
/// `MAX_FRAME_SECONDS` spirit; the probed page count of a
/// real document enforces the rest, and documents cap at
/// far fewer pages than this (see the PDF extractor).
pub const MAX_PAGE_NUMBER: u32 = 999_999;

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
    /// Name the point the next crop centers on, as fractions
    /// of the image dimensions, with a zoom factor
    /// (`focalpoint:<x>x<y>[:<zoom>]`; issue #313). Zoom 1 is
    /// the ordinary fill crop positioned on the point; above
    /// 1 the crop window tightens around the point, below 1
    /// it loosens toward the fill window. Produces no pixels
    /// of its own; the next `Crop` consumes it, like
    /// [`Operation::FocalRect`].
    FocalPoint { x: f32, y: f32, zoom: f32 },
    /// Render the smart-crop debug view (`debug:focal`): the
    /// response is a diagnostic image of the focal decision
    /// the next `Crop` makes (or of the final image, when no
    /// crop follows), never the transform result. Unlike
    /// `exp:`/`maxage:` this is an operation, not metadata:
    /// it changes the response, so it stays in the canonical
    /// form (and therefore the cache key form) in place.
    DebugFocal,
    /// Select one frame of a video source (`frame:poster`,
    /// `frame:<seconds>`, `frame:<percent>%`; issue #214).
    /// The selection is applied at decode, before every
    /// other operation, wherever the segment sits in the
    /// path: the extracted frame becomes the source image
    /// the rest of the chain runs on. It is an operation,
    /// not metadata, because it changes the response, so
    /// it stays in the canonical form in place. When a
    /// request carries several, the last one wins (the
    /// directives analysis settles it, like `exp:`).
    Frame(FrameSelection),
    /// Select one page of a PDF source (`page:<n>`, 1-based;
    /// issue #215). The selection is applied at source
    /// preparation, before every other operation, wherever
    /// the segment sits in the path: the rendered page
    /// becomes the source image the rest of the chain runs
    /// on. It is an operation, not metadata, because it
    /// changes the response, so it stays in the canonical
    /// form in place. When a request carries several, the
    /// last one wins (the directives analysis settles it,
    /// like `frame:`). Unlike a frame timestamp, an
    /// out-of-range page never clamps: a page number names
    /// a discrete page, and serving a different one under
    /// this URL would be a wrong answer, so the engine
    /// refuses it against the probed page count.
    Page(u32),
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

/// The largest `gamma` divisor the grammar accepts (issue
/// #313). 10 is the top of the range the levels tools this
/// filter maps from use, and at the extremes past it the
/// curve crushes all but a handful of input values to the
/// endpoints, so the cap costs no real image.
pub const MAX_GAMMA: f32 = 10.0;

/// The largest `exposure` adjustment the grammar accepts,
/// in stops (issue #317). Ten stops is a factor of 1024 in
/// linear light, far past the point where every channel
/// but the darkest has clamped to an endpoint, so the cap
/// costs no real image; the bound exists so a hostile
/// value cannot size the multiply into the non-finite
/// range.
pub const MAX_EXPOSURE: f32 = 10.0;

/// The largest `focalpoint:` zoom the grammar accepts
/// (issue #313). At 100 the crop window is one hundredth of
/// the fill window on each axis, already a handful of
/// pixels on any canvas the output budget allows.
pub const MAX_FOCAL_ZOOM: f32 = 100.0;

/// The largest `objects` crop scale the grammar accepts
/// (issue #318): the same bound as the focal zoom, and
/// equally theoretical, since a scale past the point where
/// the scaled union box covers the frame changes nothing.
/// The bound exists so every implementation of the grammar
/// accepts the same spellings.
pub const MAX_OBJECTS_SCALE: f32 = 100.0;

/// The most classes one `objects` crop may name (issue
/// #318). Detection vocabularies run to tens of classes;
/// 32 names in one crop is past any real framing decision,
/// and the bound keeps a single segment from carrying an
/// unbounded list into every implementation.
pub const MAX_OBJECTS_CLASSES: usize = 32;

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

/// Whether `id` is a well-formed key ID (issue #143): 1 to 32
/// characters of `[A-Za-z0-9_-]`. Key IDs are routing metadata
/// in the signature segment, never secret material.
pub fn is_valid_key_id(id: &str) -> bool {
    crabcrop_url::is_valid_key_id(id)
}

/// Whether a signature segment presents as a sealed
/// spelling (issue #255): a real token (see
/// [`split_sealed_segment`]), or a marker-shaped imitation,
/// `s<digits>.` under an optional valid key-ID prefix. The
/// imitation matters for error uniformity: an unknown
/// version marker must be indistinguishable from a failed
/// open, and answering it with the plain grammar's 400
/// would announce exactly that. No valid plain segment has
/// this shape (hex, `unsafe`, and key IDs contain no dot),
/// so the test is complete the same way the prefix test
/// is.
pub fn looks_sealed(segment: &str) -> bool {
    let body = match segment.split_once(':') {
        Some((id, rest)) if is_valid_key_id(id) => rest,
        _ => segment,
    };
    if body.starts_with(crabcrop_url::SEALED_PREFIX) {
        return true;
    }
    let Some(rest) = body.strip_prefix('s') else {
        return false;
    };
    let digits = rest.bytes().take_while(|b| b.is_ascii_digit()).count();
    digits > 0 && rest.as_bytes().get(digits) == Some(&b'.')
}

/// Split a sealed token off the front of a signature
/// segment (issue #255): `Some((key_id, token))` when the
/// segment is a sealed spelling, `[<key-id>:]s1.<body>`,
/// `None` when it is a plain signature segment. Detection is
/// a prefix test, complete by construction: a plain
/// signature is 64 lowercase hex characters or `unsafe`, and
/// a key ID is `[A-Za-z0-9_-]`, so no valid plain segment
/// can start with `s1.`. The token keeps its `s1.` marker;
/// the key ID routes the open (see [`Keyring::open_sealed`])
/// and is not part of the ciphertext or the AAD.
pub fn split_sealed_segment(segment: &str) -> Option<(Option<&str>, &str)> {
    if segment.starts_with(crabcrop_url::SEALED_PREFIX) {
        return Some((None, segment));
    }
    let (id, rest) = segment.split_once(':')?;
    if is_valid_key_id(id) && rest.starts_with(crabcrop_url::SEALED_PREFIX) {
        return Some((Some(id), rest));
    }
    None
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
    if let Some(rest) = seg.strip_prefix("frame:") {
        return Some(parse_frame(rest, seg));
    }
    if let Some(rest) = seg.strip_prefix("page:") {
        return Some(parse_page(rest, seg));
    }
    if let Some(rest) = seg.strip_prefix("crop-rect:") {
        return Some(parse_crop_rect(rest, seg));
    }
    if let Some(rest) = seg.strip_prefix("focalpoint:") {
        return Some(parse_focal_point(rest, seg));
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

/// Parse a strict decimal for the `frame:` grammar: an
/// unsigned whole part, an optional fraction, no sign, no
/// exponent, no leading dot, the same discipline as the
/// `rotate` angle and the `convolution` kernel values, so
/// every implementation of the grammar reads the same
/// strings. Bounded by `max` (inclusive).
fn parse_frame_decimal(text: &str, max: f64) -> Result<f64, ()> {
    let (whole, fraction) = match text.split_once('.') {
        Some((w, f)) => (w, Some(f)),
        None => (text, None),
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
    if !value.is_finite() || value < 0.0 || value > max {
        return Err(());
    }
    Ok(value)
}

/// Parse the selector after `frame:`: `poster`, a seconds
/// timestamp, or a percentage with its `%` suffix.
fn parse_frame(rest: &str, seg: &str) -> Result<Operation, ParseError> {
    let invalid = || ParseError::InvalidFrame(seg.to_string());
    if rest == "poster" {
        return Ok(Operation::Frame(FrameSelection::Poster));
    }
    if let Some(pct) = rest.strip_suffix('%') {
        let value = parse_frame_decimal(pct, MAX_FRAME_PERCENT).map_err(|_| invalid())?;
        return Ok(Operation::Frame(FrameSelection::Percent(value)));
    }
    let value = parse_frame_decimal(rest, MAX_FRAME_SECONDS).map_err(|_| invalid())?;
    Ok(Operation::Frame(FrameSelection::Seconds(value)))
}

/// Parse the number after `page:`: ASCII digits only, no
/// sign, no fraction, no exponent, the strictness the
/// `frame:` grammar applies to its own numbers. Leading
/// zeros are accepted and canonicalized away (`page:007`
/// is `page:7`), so the two spellings are one request with
/// one cache entry, the `frame:1.50` rule. Zero is outside
/// the grammar: pages count from 1.
fn parse_page(rest: &str, seg: &str) -> Result<Operation, ParseError> {
    let invalid = || ParseError::InvalidPage(seg.to_string());
    if rest.is_empty() || !rest.bytes().all(|b| b.is_ascii_digit()) {
        return Err(invalid());
    }
    // Strip leading zeros before parsing, so a long run of
    // them cannot overflow the integer parse on its way to
    // the bound check; what remains is the number itself.
    let digits = rest.trim_start_matches('0');
    if digits.is_empty() || digits.len() > 6 {
        return Err(invalid());
    }
    let page: u32 = digits.parse().map_err(|_| invalid())?;
    if page == 0 || page > MAX_PAGE_NUMBER {
        return Err(invalid());
    }
    Ok(Operation::Page(page))
}

/// The canonical spelling of a frame timestamp or
/// percentage: whole values print as integers, fractions
/// use the shortest round-trip form, matching the rotate
/// angle's canonicalization, so `frame:1.50` and
/// `frame:1.5` are one request with one cache entry.
pub(crate) fn format_frame_value(value: f64) -> String {
    if value == value.trunc() {
        format!("{}", value as i64)
    } else {
        format!("{value}")
    }
}

/// The canonical segment text of a frame selection.
fn frame_text(selection: &FrameSelection) -> String {
    match selection {
        FrameSelection::Poster => "frame:poster".to_string(),
        FrameSelection::Seconds(s) => format!("frame:{}", format_frame_value(*s)),
        FrameSelection::Percent(p) => format!("frame:{}%", format_frame_value(*p)),
    }
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

/// Parse `<x>x<y>[:<zoom>]` after `focalpoint:` (issue
/// #313): the point is a pair of fractions of the image
/// dimensions and the zoom, when spelled, is a positive
/// factor no larger than [`MAX_FOCAL_ZOOM`]. The zoom is
/// optional on input and always spelled in the canonical
/// form, the way bare `trim` canonicalizes to `trim(10)`.
fn parse_focal_point(rest: &str, seg: &str) -> Result<Operation, ParseError> {
    let invalid = || ParseError::InvalidFocalPoint(seg.to_string());
    let (point, zoom_text) = match rest.split_once(':') {
        Some((point, zoom)) => (point, Some(zoom)),
        None => (rest, None),
    };
    let (x_text, y_text) = point.split_once('x').ok_or_else(invalid)?;
    let x: f32 = x_text.parse().map_err(|_| invalid())?;
    let y: f32 = y_text.parse().map_err(|_| invalid())?;
    if !x.is_finite() || !y.is_finite() || !(0.0..=1.0).contains(&x) || !(0.0..=1.0).contains(&y) {
        return Err(invalid());
    }
    let zoom: f32 = match zoom_text {
        Some(text) => text.parse().map_err(|_| invalid())?,
        None => 1.0,
    };
    if !zoom.is_finite() || zoom <= 0.0 || zoom > MAX_FOCAL_ZOOM {
        return Err(invalid());
    }
    Ok(Operation::FocalPoint { x, y, zoom })
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

/// The alignment after `crop:<WxH>:`. Every other
/// alignment is a bare keyword; the `objects` mode (issue
/// #318) is the one parameterized form,
/// `objects(<class>[,<class>...])[:<scale>]`: the classes
/// spell exactly like the class filters' argument (the
/// charset excludes commas, so the comma is a safe list
/// separator, and percent-decoding runs per class, as
/// there). The scale is a positive multiplier of the union
/// box, bounded by [`MAX_OBJECTS_SCALE`]; it is optional on
/// input and always spelled in the canonical form, the
/// `focalpoint` zoom precedent. A class named twice (in
/// any case: vocabulary matching is case-insensitive)
/// names one class, so the canonical form keeps the first
/// spelling of each, and the [`MAX_OBJECTS_CLASSES`] cap
/// counts distinct names: a repeat costs nothing against
/// it, and a list naming more distinct classes than the cap
/// is rejected, never truncated.
fn parse_crop_align(s: &str) -> Option<Align> {
    let Some(args) = s.strip_prefix("objects") else {
        return parse_align(s);
    };
    let inner = args.strip_prefix('(')?;
    let close = inner.find(')')?;
    let (list, tail) = (&inner[..close], &inner[close + 1..]);
    let scale = match tail {
        "" => 1.0,
        _ => {
            let value: f32 = tail.strip_prefix(':')?.parse().ok()?;
            if !value.is_finite() || value <= 0.0 || value > MAX_OBJECTS_SCALE {
                return None;
            }
            value
        }
    };
    let mut classes: Vec<String> = Vec::new();
    for raw in list.split(',') {
        let name = parse_class_name(&percent_decode(raw)?)?;
        if classes.iter().any(|c| c.eq_ignore_ascii_case(&name)) {
            continue;
        }
        // The cap applies as the list is built (issue
        // #391), so the kept set never grows past it and
        // each name costs at most a cap-sized scan instead
        // of a quadratic one. A distinct name past the cap
        // rejects the segment (issue #408): truncating
        // would crop to a set the caller never asked for,
        // and center-crop when the only class in the image
        // is one of the dropped names.
        if classes.len() == MAX_OBJECTS_CLASSES {
            return None;
        }
        classes.push(name);
    }
    if classes.is_empty() {
        return None;
    }
    Some(Align::Objects { classes, scale })
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
        Some((d, a)) => (d, parse_crop_align(a).ok_or_else(invalid)?),
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

/// Parse one quality value inside a `quality(...)` filter:
/// `auto` or a number in 1..=100, the standing range.
fn parse_quality_value(s: &str) -> Option<QualityValue> {
    if s == "auto" {
        return Some(QualityValue::Auto);
    }
    let q: u8 = s.parse().ok()?;
    if !(1..=100).contains(&q) {
        return None;
    }
    Some(QualityValue::Value(q))
}

/// Parse a `quality(...)` argument (issue #319): either the
/// standing single form (`80`, `auto`) or a per-format list
/// (`80,jpeg:90,webp:75`) with at most one bare fallback
/// entry and at most one entry per format. Formats a quality
/// cannot reach (the video outputs, whose encoding is an
/// external tool's) are rejected, as are unknown format
/// names: an entry that could never take effect is an error,
/// never silently ignored.
fn parse_quality(p: &str) -> Option<Quality> {
    if !p.contains(',') && !p.contains(':') {
        return match parse_quality_value(p)? {
            QualityValue::Value(v) => Some(Quality::Value(v)),
            QualityValue::Auto => Some(Quality::Auto),
        };
    }
    let mut entries: Vec<(Option<ImageFormat>, QualityValue)> = Vec::new();
    for part in p.split(',') {
        if part.is_empty() {
            return None;
        }
        match part.split_once(':') {
            Some((name, value)) => {
                let format = parse_format(name)?;
                if format.is_video() {
                    return None;
                }
                if entries.iter().any(|(k, _)| *k == Some(format)) {
                    return None;
                }
                entries.push((Some(format), parse_quality_value(value)?));
            }
            None => {
                if entries.iter().any(|(k, _)| k.is_none()) {
                    return None;
                }
                entries.push((None, parse_quality_value(part)?));
            }
        }
    }
    if entries.is_empty() {
        return None;
    }
    Some(Quality::PerFormat(entries))
}

/// Parse a `format(...)` argument (issue #319): a format
/// name plus its per-format encoder options as
/// comma-separated arguments. An option the named format's
/// encoder cannot honour is rejected here, at parse time:
/// AVIF takes none (the ravif encoder writes 4:4:4 only and
/// exposes no subsampling control, so a subsampling option
/// would be a lie), and neither do JPEG XL, GIF, or the
/// video formats.
fn parse_format_spec(p: &str) -> Option<FormatSpec> {
    let mut parts = p.split(',');
    let format = parse_format(parts.next()?)?;
    let mut spec = FormatSpec::plain(format);
    let options = &mut spec.options;
    for part in parts {
        match format {
            ImageFormat::Jpeg => match part {
                "progressive" => {
                    if options.jpeg_progressive {
                        return None;
                    }
                    options.jpeg_progressive = true;
                }
                _ => {
                    let value = part.strip_prefix("subsampling:")?;
                    if options.jpeg_subsampling.is_some() {
                        return None;
                    }
                    options.jpeg_subsampling = Some(Subsampling::parse(value)?);
                }
            },
            ImageFormat::Png => {
                if part != "interlace" || options.png_interlace {
                    return None;
                }
                options.png_interlace = true;
            }
            ImageFormat::WebP => {
                let (value, max, slot) = match part.strip_prefix("method:") {
                    Some(v) => (v, 6, &mut options.webp_method),
                    None => (part.strip_prefix("sns:")?, 100, &mut options.webp_sns),
                };
                if slot.is_some() {
                    return None;
                }
                let parsed: u8 = value.parse().ok()?;
                if parsed > max {
                    return None;
                }
                *slot = Some(parsed);
            }
            _ => return None,
        }
    }
    Some(spec)
}

/// A detection class name for the class filters
/// (`blurclass` / `pixelateclass` / `recolorclass`):
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

/// The preset names a remainder references, in order,
/// deduplicated: the cache groups (RFC 9875) the response
/// belongs to. Scans the same leading segment run
/// [`expand_presets`] expands and stops where it stops (the
/// source, or a segment that is not an operation).
///
/// The scan runs on the request path before expansion
/// rejects unknown names, so it sees untrusted input (issue
/// #409). A hash set tracks the names already kept: n
/// distinct names cost linear work, where scanning the kept
/// list was quadratic. The output keeps first-seen order.
pub fn preset_names(rest: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut tail = rest;
    loop {
        let (seg, remaining) = match tail.split_once('/') {
            Some((s, r)) => (s, r),
            None => (tail, ""),
        };
        if let Some(name) = seg.strip_prefix("preset:") {
            if !name.is_empty() && seen.insert(name) {
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

/// The most operations one request may execute (issue #182,
/// audit R2-12), counted by [`operation_count`] on the list
/// the engine runs: after preset expansion and with the
/// server-wide defaults merged in, so neither a short URL
/// that repeats a preset nor the defaults can multiply work
/// past it. Real chains (a size, a crop, a dozen filters)
/// sit far below it; the cap exists so thousands of
/// repeated cheap operations cannot burn a process-gate
/// slot on CPU alone.
pub const MAX_OPERATIONS: usize = 100;

/// How many operations a list executes: each operation
/// once, and each filter inside a `filters:` segment once,
/// because each is a pass over the pixels.
pub fn operation_count(ops: &[Operation]) -> usize {
    ops.iter()
        .map(|op| match op {
            Operation::Filters(filters) => filters.len(),
            _ => 1,
        })
        .sum()
}

/// Refuse an operation list over [`MAX_OPERATIONS`] (issue
/// #469). The server runs it on the merged list right after
/// parsing, on the image and explain routes alike, so an
/// over-cap request answers its `400` before the cache, the
/// gates, or any source fetch; the engine keeps the same
/// check as a backstop.
pub fn check_operation_count(ops: &[Operation]) -> Result<(), ParseError> {
    let count = operation_count(ops);
    if count > MAX_OPERATIONS {
        return Err(ParseError::TooManyOperations(count));
    }
    Ok(())
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
///
/// `exp:` and `maxage:` are refused, as preset definitions
/// refuse them (issue #469): both are per-request metadata,
/// which the operations parse returns apart from the
/// operations, so a default lifetime used to boot cleanly
/// and never apply. A string that alone runs past
/// [`MAX_OPERATIONS`] is refused too: every request under it
/// would answer `400`.
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
    // The parse succeeded up to the synthetic source, so every
    // segment of `expanded` is an operation or metadata, and a
    // metadata prefix is always read as metadata.
    if let Some(seg) = expanded
        .split('/')
        .find(|seg| seg.starts_with("exp:") || seg.starts_with("maxage:"))
    {
        return Err(if seg.starts_with("exp:") {
            ParseError::InvalidExpiry(seg.to_string())
        } else {
            ParseError::InvalidMaxAge(seg.to_string())
        });
    }
    check_operation_count(&ops)?;
    Ok(ops)
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

/// A source root reference as written in a URL: the root
/// name and the path after it, exactly as presented. The
/// expansion a reference resolves to is server state; the
/// reference itself is all a report may ever show (see
/// [`expand_source_root`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootRef {
    pub name: String,
    pub path: String,
}

/// Expand a `root:<name>/<path>` short source against the
/// server's named source roots (issue #248), returning the
/// expanded source and the reference as written. A source
/// that is not a root reference passes through untouched
/// with a `None` reference.
///
/// The signature covers the short form exactly as written;
/// expansion is deterministic and server-side, and it runs
/// after verification and before cache keying, so a short
/// source and the full source it expands to are one render:
/// one cache entry, one purge target. The expanded URL goes
/// through every loader check unchanged, the allowlist
/// included: the allowlist is evaluated against the
/// expansion, never against the short form.
///
/// The path must stay under the root. Dot segments are
/// normalized (`.` drops, `..` pops one segment); a `..`
/// that would climb above the root is rejected. Segments
/// are percent-decoded for this check, so `%2e%2e` is a
/// `..`; a segment whose decoded form carries a `/` or `\`
/// separator is refused outright, as an encoded separator
/// is never a legitimate path segment here.
pub fn expand_source_root(
    source: &str,
    roots: &std::collections::HashMap<String, String>,
) -> Result<(String, Option<RootRef>), ParseError> {
    let Some(body) = source.strip_prefix("root:") else {
        return Ok((source.to_string(), None));
    };
    let invalid = || ParseError::InvalidRoot(source.to_string());
    let Some((name, path)) = body.split_once('/') else {
        return Err(invalid());
    };
    if name.is_empty() || path.is_empty() {
        return Err(invalid());
    }
    let base = roots
        .get(name)
        .ok_or_else(|| ParseError::UnknownRoot(name.to_string()))?;
    let normalized = normalize_root_path(path).ok_or_else(invalid)?;
    Ok((
        format!("{base}/{normalized}"),
        Some(RootRef {
            name: name.to_string(),
            path: path.to_string(),
        }),
    ))
}

/// Expand the root source inside a whole remainder, for the
/// paths that work on remainders rather than parsed sources
/// (purge): the remainder's source is located with the same
/// parse the serve path uses, and the returned remainder
/// carries the expansion in its place, so it canonicalizes
/// exactly like the entry the serve path stored. A remainder
/// that does not parse, or whose source is not a root
/// reference, comes back unchanged.
pub fn expand_source_root_in_remainder(
    rest: &str,
    roots: &std::collections::HashMap<String, String>,
) -> Result<String, ParseError> {
    let Ok((_, source, _, _)) = parse_operations_with_expiry(rest) else {
        return Ok(rest.to_string());
    };
    let (expanded, reference) = expand_source_root(source, roots)?;
    if reference.is_none() {
        return Ok(rest.to_string());
    }
    Ok(format!("{}{expanded}", &rest[..rest.len() - source.len()]))
}

/// Normalize a root-relative path for joining under a root:
/// returns the path with dot segments resolved, or `None`
/// when the path is not usable under the root (it climbs
/// above the root, carries an encoded separator, or
/// resolves to no object at all). Non-dot segments keep
/// their exact written bytes; only the dot check decodes.
fn normalize_root_path(path: &str) -> Option<String> {
    let mut out: Vec<&str> = Vec::new();
    for seg in path.split('/') {
        let decoded = percent_decode(seg)?;
        if decoded.contains('/') || decoded.contains('\\') {
            return None;
        }
        match decoded.as_str() {
            "." => {}
            ".." => {
                out.pop()?;
            }
            _ => out.push(seg),
        }
    }
    if !out.iter().any(|seg| !seg.is_empty()) {
        return None;
    }
    Some(out.join("/"))
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
/// operations. This is the one place that reads the ops list
/// for directives, and both the server and the engine consume
/// it, so the server's AI-gate predicate cannot drift from the
/// engine's preflight.
///
/// Field semantics: `format`,
/// `fill`, `dpr`, and `facepad` are request-wide with the LAST
/// segment winning; the shape filter is the FIRST one (it names
/// error messages); class names keep request order; watermark
/// URLs are distinct, in order of first appearance.
#[derive(Debug, Clone, PartialEq)]
pub struct RequestDirectives {
    /// The explicitly named output format, if any.
    pub explicit_format: Option<ImageFormat>,
    /// The encoder options on the explicit format filter
    /// (issue #319), all default when it named none.
    pub format_options: FormatOptions,
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
    /// Any `remove_region(...)` filter (issue #176): it
    /// runs the inpainting model, so it takes an AI permit
    /// and needs the provider preflight, like `upscale`.
    pub remove_region: bool,
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
    /// Class names requested by class operations (the
    /// redaction filters and `recolorclass`), in request
    /// order.
    pub classes: Vec<String>,
    /// Any `objects` crop (issue #318): it runs object
    /// detection, so it takes an AI permit and needs the
    /// provider preflight, like the class operations.
    pub objects_crop: bool,
    /// Class names requested by `objects` crops, in
    /// request order.
    pub object_classes: Vec<String>,
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
    /// The frame selection (issue #214) when the request
    /// carries a `frame:` segment: the last one in path
    /// order wins, the same rule as the `exp:` expiry.
    pub frame: Option<FrameSelection>,
    /// The page selection (issue #215) when the request
    /// carries a `page:` segment: the last one in path
    /// order wins, the same rule as the frame selection.
    pub page: Option<u32>,
}

impl RequestDirectives {
    /// Analyze a parsed request's operations in a single walk.
    pub fn analyze(ops: &[Operation]) -> Self {
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
                        Filter::Mask(_) if d.shape_filter.is_none() => {
                            d.shape_filter = Some("mask")
                        }
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

    /// Whether any class operation is present (which
    /// needs the detection model and the AI gate).
    pub fn has_class_redaction(&self) -> bool {
        !self.classes.is_empty()
    }

    /// Whether the request names any AI directive: background
    /// removal, studio mode, region removal, upscale, face
    /// operations, class redaction, object crops,
    /// `debug:focal`, or a smart crop. A smart crop infers
    /// only when a face or focal provider is configured, so
    /// the server's AI gate uses `uses_ai_beyond_smart_crop`
    /// plus that provider check (issue #379). Only the tests
    /// call this, to pin the directive classification.
    #[allow(dead_code)]
    pub fn uses_ai(&self) -> bool {
        self.uses_ai_beyond_smart_crop() || self.smart_crop
    }

    /// Every `uses_ai` term except the smart crop (issue
    /// #379): the directives alone cannot say whether a
    /// smart crop infers, since that depends on the
    /// deployment's providers, so the server's gate
    /// predicate combines this with the provider check.
    pub fn uses_ai_beyond_smart_crop(&self) -> bool {
        self.removebg
            || self.studio
            || self.remove_region
            || self.upscale
            || self.needs_face_model
            || self.has_class_redaction()
            || self.objects_crop
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

fn align_text(align: &Align) -> Option<&'static str> {
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
        // The objects alignment carries its classes and
        // scale; `operation_text` spells it in full.
        Align::Objects { .. } => None,
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

/// The canonical spelling of one quality value: `auto` or
/// the number.
fn quality_value_text(value: QualityValue) -> String {
    match value {
        QualityValue::Auto => "auto".to_string(),
        QualityValue::Value(v) => v.to_string(),
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
/// image remainder begins with `purge/`. Signing the literal
/// `purge/<target>` with the secret itself would be exactly an
/// image signature over a remainder of that shape.
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
///
/// Its `Debug` prints the key IDs and redacts the keys
/// (issue #455). It is not `PartialEq`: nothing compares
/// rings, and a derived comparison of key bytes would not be
/// constant-time.
#[derive(Clone)]
pub struct Keyring {
    pub current_id: String,
    pub current: Vec<u8>,
    pub previous: Option<(String, Vec<u8>)>,
}

impl std::fmt::Debug for Keyring {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Keyring")
            .field("current_id", &self.current_id)
            .field("current", &crate::redact::Redacted)
            .field(
                "previous",
                &self
                    .previous
                    .as_ref()
                    .map(|(id, _)| (id, crate::redact::Redacted)),
            )
            .finish()
    }
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

    /// Open a sealed token under the ring (issue #255): a
    /// keyed request opens under the named key only (see
    /// [`Self::candidates`]); an unkeyed one tries current,
    /// then previous, which is the rotation window sealed
    /// URLs inherit from the keyring with no second secret.
    /// Returns the decrypted remainder, or `None` when no
    /// candidate opens it: the caller maps every `None` to
    /// the one uniform refusal and never learns, or
    /// reveals, which stage failed.
    pub fn open_sealed(&self, key_id: Option<&str>, token: &str) -> Option<String> {
        self.candidates(key_id)
            .iter()
            .find_map(|secret| crabcrop_url::open_token(secret, token).ok())
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
    fn canonical_key_form_strips_expiry_segments() {
        let src = "https://example.com/a.jpg";
        assert_eq!(
            canonical_key_form(&format!("exp:1893456000/300x200/{src}")),
            format!("300x200/{src}")
        );
        assert_eq!(
            canonical_key_form(&format!("300x200/exp:1893456000/{src}")),
            format!("300x200/{src}")
        );
        assert_eq!(canonical_key_form(&format!("exp:1893456000/{src}")), src);
        assert_eq!(
            canonical_key_form(&format!("300x200/{src}")),
            format!("300x200/{src}")
        );
        // An exp: segment inside the source is URL text, kept.
        let tricky = "https://example.com/exp:123/a.jpg";
        assert_eq!(canonical_key_form(tricky), tricky);
        // maxage: is per-request metadata too: stripped wherever
        // it sits among the operations, kept inside the source.
        assert_eq!(
            canonical_key_form(&format!("maxage:60/300x200/{src}")),
            format!("300x200/{src}")
        );
        assert_eq!(
            canonical_key_form(&format!("300x200/maxage:60/{src}")),
            format!("300x200/{src}")
        );
        assert_eq!(canonical_key_form(&format!("maxage:60/{src}")), src);
        let tricky = "https://example.com/maxage:60/a.jpg";
        assert_eq!(canonical_key_form(tricky), tricky);
    }

    #[test]
    fn canonical_key_form_passes_unparseable_remainders_through() {
        // A remainder that does not parse keys as itself,
        // byte for byte, and stays a fixed point: no source
        // boundary is known, so no segment is stripped, not
        // even a well-formed `exp:` or `maxage:` ahead of the
        // failing one. Purge targets are caller text, so the
        // set is hostile: malformed metadata, a bad operation
        // argument, a missing source, an empty remainder, a
        // long run of segments.
        let long = format!("{}filters:blur(x)/a.jpg", "exp:1/".repeat(10_000));
        let cases = [
            "".to_string(),
            "/".to_string(),
            "exp:/300x200/https://example.com/a.jpg".to_string(),
            "exp:12x/300x200/https://example.com/a.jpg".to_string(),
            "maxage:99999999999/https://example.com/a.jpg".to_string(),
            "exp:1893456000/maxage:60/maxage:/https://example.com/a.jpg".to_string(),
            "exp:1893456000/300x200/".to_string(),
            "maxage:60/300x200/filters:blur(nope)/https://example.com/a.jpg".to_string(),
            "exp:99999999999999999999999/https://example.com/a.jpg".to_string(),
            long,
        ];
        for remainder in &cases {
            assert!(
                parse_operations_with_expiry(remainder).is_err(),
                "the case must not parse: {remainder:.80}"
            );
            let key = canonical_key_form(remainder);
            assert_eq!(&key, remainder, "an unparseable remainder keys as itself");
            assert_eq!(canonical_key_form(&key), key, "the key is a fixed point");
        }
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
            (
                ParseError::InvalidFocalPoint("x".into()),
                "invalid_focal_point",
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
            (ParseError::InvalidFrame("x".into()), "invalid_frame"),
            (ParseError::InvalidPage("x".into()), "invalid_page"),
            (ParseError::MissingSourceUrl, "missing_source_url"),
            // The engine's code, kept when the cap check moved
            // to parse time (issue #469).
            (ParseError::TooManyOperations(101), "unsupported_operation"),
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
            // The focal point zoom defaults to 1 and is
            // spelled out in the canonical form.
            ("focalpoint:0.75x0.4", "focalpoint:0.75x0.4:1"),
            // One radius is four.
            ("filters:radius(10)", "filters:radius(10,10,10,10)"),
            // Float spellings.
            ("filters:contrast(1.50)", "filters:contrast(1.5)"),
            ("filters:gamma(2.20)", "filters:gamma(2.2)"),
            ("filters:exposure(1.50)", "filters:exposure(1.5)"),
            ("filters:exposure(-2.0)", "filters:exposure(-2)"),
            (
                "filters:highlights_shadows(30.0,-20.50)",
                "filters:highlights_shadows(30,-20.5)",
            ),
            ("filters:vibrance(40.0)", "filters:vibrance(40)"),
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
    fn sealed_segment_splitting() {
        assert_eq!(split_sealed_segment("s1.AAECAw"), Some((None, "s1.AAECAw")));
        assert_eq!(
            split_sealed_segment("k2:s1.AAECAw"),
            Some((Some("k2"), "s1.AAECAw"))
        );
        // Plain spellings never split as sealed: a hex
        // signature, `unsafe`, a keyed signature, and
        // near-miss markers.
        for plain in [
            "0123456789abcdef",
            "unsafe",
            "k2:0123456789abcdef",
            "s1",
            "s12.AAAA",
            "k ey:s1.AAAA",
            ":s1.AAAA",
        ] {
            assert_eq!(split_sealed_segment(plain), None, "{plain}");
        }
        // Marker-shaped imitations still present as sealed
        // (they earn the uniform refusal, not the grammar's
        // 400), while ordinary segments do not.
        for imitation in ["s2.AAAA", "s12.x", "k2:s2.AAAA", "s1.anything"] {
            assert!(looks_sealed(imitation), "{imitation}");
        }
        for plain in ["unsafe", "0123456789abcdef", "sx.AAAA", "s.AAAA", "k2:a1b2"] {
            assert!(!looks_sealed(plain), "{plain}");
        }
    }

    #[test]
    fn keyring_opens_sealed_tokens_across_rotation() {
        let remainder = "300x200/https://example.com/photo.jpg";
        let ring = Keyring {
            current_id: "k-new".to_string(),
            current: b"new-secret-value".to_vec(),
            previous: Some(("k-old".to_string(), b"old-secret-value".to_vec())),
        };
        // A token minted under the previous secret opens
        // during the rotation window: unkeyed (current is
        // tried, then previous) and keyed to the old ID.
        let old_token = crabcrop_url::seal(b"old-secret-value", remainder).unwrap();
        assert_eq!(
            ring.open_sealed(None, &old_token).as_deref(),
            Some(remainder)
        );
        assert_eq!(
            ring.open_sealed(Some("k-old"), &old_token).as_deref(),
            Some(remainder)
        );
        // Keyed routing is exact: naming the current ID does
        // not open a previous-key token, and an unknown ID
        // opens nothing.
        assert_eq!(ring.open_sealed(Some("k-new"), &old_token), None);
        assert_eq!(ring.open_sealed(Some("k-nope"), &old_token), None);
        // A token under a foreign secret opens under no
        // routing.
        let foreign = crabcrop_url::seal(b"someone-elses-secret", remainder).unwrap();
        assert_eq!(ring.open_sealed(None, &foreign), None);
        // Once the previous key leaves the ring, its tokens
        // stop opening: the window has closed.
        let rotated = Keyring {
            current_id: "k-new".to_string(),
            current: b"new-secret-value".to_vec(),
            previous: None,
        };
        assert_eq!(rotated.open_sealed(None, &old_token), None);
        assert_eq!(rotated.open_sealed(Some("k-old"), &old_token), None);
        // The current key's tokens open throughout.
        let new_token = crabcrop_url::seal(b"new-secret-value", remainder).unwrap();
        assert_eq!(
            rotated.open_sealed(None, &new_token).as_deref(),
            Some(remainder)
        );
        // A tampered token opens nowhere.
        let mut bytes = old_token.into_bytes();
        let last = bytes.len() - 1;
        bytes[last] = if bytes[last] == b'A' { b'B' } else { b'A' };
        let tampered = String::from_utf8(bytes).unwrap();
        assert_eq!(ring.open_sealed(None, &tampered), None);
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
                    &ops[0],
                    Operation::Crop { align, .. } if align == &want
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
                Filter::Format(FormatSpec::plain(ImageFormat::WebP)),
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
            "filters:gamma(10)",
            "filters:gamma(0.1)",
            "filters:exposure(10)",
            "filters:exposure(-10)",
            "filters:exposure(0)",
            "filters:highlights_shadows(100,100)",
            "filters:highlights_shadows(-100,-100)",
            "filters:vibrance(100)",
            "filters:vibrance(-100)",
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
    fn parses_gamma_filter() {
        let (ops, _) = parse_operations("filters:gamma(2.2)/example.com/img.jpg").unwrap();
        assert_eq!(ops, vec![Operation::Filters(vec![Filter::Gamma(2.2)])]);
    }

    #[test]
    fn parses_exposure_filter() {
        let (ops, _) = parse_operations("filters:exposure(1.5)/example.com/img.jpg").unwrap();
        assert_eq!(ops, vec![Operation::Filters(vec![Filter::Exposure(1.5)])]);
        // Zero stops is the identity and parses, like
        // contrast at 0; the endpoints of the stops range
        // parse too (one step past them is pinned in
        // rejects_bad_filters).
        let (ops, _) = parse_operations("filters:exposure(-2)/example.com/img.jpg").unwrap();
        assert_eq!(ops, vec![Operation::Filters(vec![Filter::Exposure(-2.0)])]);
        assert!(parse_operations("filters:exposure(0)/example.com/img.jpg").is_ok());
    }

    #[test]
    fn parses_highlights_shadows_filter() {
        let (ops, _) =
            parse_operations("filters:highlights_shadows(30,-20)/example.com/img.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![Filter::HighlightsShadows(
                30.0, -20.0
            )])]
        );
        // Both at 0 are the identity and parse, like
        // contrast at 0.
        let (ops, _) =
            parse_operations("filters:highlights_shadows(0,0)/example.com/img.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![Filter::HighlightsShadows(
                0.0, 0.0
            )])]
        );
    }

    #[test]
    fn parses_vibrance_filter() {
        let (ops, _) = parse_operations("filters:vibrance(40)/example.com/img.jpg").unwrap();
        assert_eq!(ops, vec![Operation::Filters(vec![Filter::Vibrance(40.0)])]);
        // Zero is the identity and parses, like contrast
        // at 0.
        let (ops, _) = parse_operations("filters:vibrance(0)/example.com/img.jpg").unwrap();
        assert_eq!(ops, vec![Operation::Filters(vec![Filter::Vibrance(0.0)])]);
    }

    #[test]
    fn parses_enhance_filter() {
        let (ops, _) = parse_operations("filters:enhance/example.com/img.jpg").unwrap();
        assert_eq!(ops, vec![Operation::Filters(vec![Filter::Enhance])]);
        // Empty parentheses spell the same filter, like
        // every no-argument filter.
        let (ops, _) = parse_operations("filters:enhance()/example.com/img.jpg").unwrap();
        assert_eq!(ops, vec![Operation::Filters(vec![Filter::Enhance])]);
    }

    #[test]
    fn parses_palette_filter() {
        let (ops, _) = parse_operations("filters:palette(64)/example.com/img.jpg").unwrap();
        assert_eq!(ops, vec![Operation::Filters(vec![Filter::Palette(64)])]);
        // The endpoints of the color budget parse.
        assert!(parse_operations("filters:palette(2)/example.com/img.jpg").is_ok());
        assert!(parse_operations("filters:palette(256)/example.com/img.jpg").is_ok());
        for seg in [
            "filters:palette(0)",
            "filters:palette(1)",
            "filters:palette(257)",
            "filters:palette(-8)",
            "filters:palette()",
            "filters:palette(64,64)",
        ] {
            assert!(
                parse_operations(&format!("{seg}/example.com/img.jpg")).is_err(),
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
            "filters:gamma(0)",
            "filters:gamma(-1)",
            "filters:gamma(10.5)",
            "filters:gamma(nan)",
            "filters:gamma(inf)",
            "filters:gamma()",
            "filters:exposure(10.5)",
            "filters:exposure(-10.5)",
            "filters:exposure(nan)",
            "filters:exposure(inf)",
            "filters:exposure()",
            "filters:highlights_shadows(101,0)",
            "filters:highlights_shadows(0,-101)",
            "filters:highlights_shadows(30)",
            "filters:highlights_shadows(1,2,3)",
            "filters:highlights_shadows(nan,0)",
            "filters:highlights_shadows()",
            "filters:vibrance(101)",
            "filters:vibrance(-100.5)",
            "filters:vibrance(nan)",
            "filters:vibrance(inf)",
            "filters:vibrance()",
            "filters:saturation(256)",
            "filters:saturation(-1)",
            "filters:quality()",
            "filters:quality(autoo)",
            "filters:grayscale(1)",
            "filters:enhance(1)",
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
            vec![Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::Jxl
            ))])]
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
    fn parses_format_options() {
        let (ops, _) =
            parse_operations("filters:format(jpeg,progressive,subsampling:444)/example.com/i.jpg")
                .unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![Filter::Format(FormatSpec {
                format: ImageFormat::Jpeg,
                options: FormatOptions {
                    jpeg_progressive: true,
                    jpeg_subsampling: Some(Subsampling::S444),
                    ..FormatOptions::default()
                },
            })])]
        );
        let (ops, _) = parse_operations("filters:format(png,interlace)/example.com/i.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![Filter::Format(FormatSpec {
                format: ImageFormat::Png,
                options: FormatOptions {
                    png_interlace: true,
                    ..FormatOptions::default()
                },
            })])]
        );
        let (ops, _) =
            parse_operations("filters:format(webp,method:6,sns:20)/example.com/i.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![Filter::Format(FormatSpec {
                format: ImageFormat::WebP,
                options: FormatOptions {
                    webp_method: Some(6),
                    webp_sns: Some(20),
                    ..FormatOptions::default()
                },
            })])]
        );
        // A bare format is the plain spec every pre-options
        // URL already means.
        let (ops, _) = parse_operations("filters:format(webp)/example.com/i.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::WebP
            ))])]
        );
    }

    #[test]
    fn format_options_canonicalize_in_fixed_order() {
        for (given, canonical) in [
            (
                "filters:format(webp,sns:20,method:6)",
                "filters:format(webp,method:6,sns:20)",
            ),
            (
                "filters:format(jpeg,subsampling:422,progressive)",
                "filters:format(jpeg,progressive,subsampling:422)",
            ),
            (
                "filters:format(png,interlace)",
                "filters:format(png,interlace)",
            ),
            ("filters:format(jpeg)", "filters:format(jpeg)"),
        ] {
            let url = format!("{given}/example.com/i.jpg");
            let (ops, source) = parse_operations(&url).unwrap();
            assert_eq!(
                canonical_form(&ops, source),
                format!("{canonical}/example.com/i.jpg"),
                "{given}"
            );
        }
    }

    #[test]
    fn format_options_reject_what_encoders_cannot_honour() {
        for bad in [
            // Options on formats whose encoders have none.
            "filters:format(avif,subsampling:420)",
            "filters:format(avif,progressive)",
            "filters:format(jxl,method:4)",
            "filters:format(gif,interlace)",
            "filters:format(mp4,method:4)",
            // Another format's options.
            "filters:format(jpeg,interlace)",
            "filters:format(jpeg,method:4)",
            "filters:format(png,progressive)",
            "filters:format(png,method:4)",
            "filters:format(webp,progressive)",
            "filters:format(webp,subsampling:420)",
            // Unknown, duplicated, or out-of-range options.
            "filters:format(jpeg,fast)",
            "filters:format(jpeg,progressive,progressive)",
            "filters:format(jpeg,subsampling:421)",
            "filters:format(jpeg,subsampling:)",
            "filters:format(webp,method:7)",
            "filters:format(webp,method:-1)",
            "filters:format(webp,method:99999999999999999999)",
            "filters:format(webp,method:4,method:5)",
            "filters:format(webp,sns:101)",
            "filters:format(webp,sns:4.5)",
            "filters:format(webp,)",
            "filters:format(,method:4)",
            "filters:format(webp,m\u{e9}thod:4)",
        ] {
            assert!(
                parse_operations(&format!("{bad}/example.com/i.jpg")).is_err(),
                "{bad} must not parse"
            );
        }
    }

    #[test]
    fn parses_per_format_quality() {
        let (ops, _) =
            parse_operations("filters:quality(jpeg:80,webp:75)/example.com/i.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![Filter::Quality(
                Quality::PerFormat(vec![
                    (Some(ImageFormat::Jpeg), QualityValue::Value(80)),
                    (Some(ImageFormat::WebP), QualityValue::Value(75)),
                ],)
            )])]
        );
        // Fallback plus entries, with an auto entry.
        let (ops, _) =
            parse_operations("filters:quality(70,jpeg:auto,avif:60)/example.com/i.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![Filter::Quality(
                Quality::PerFormat(vec![
                    (None, QualityValue::Value(70)),
                    (Some(ImageFormat::Jpeg), QualityValue::Auto),
                    (Some(ImageFormat::Avif), QualityValue::Value(60)),
                ],)
            )])]
        );
        // A single named entry is a list of one.
        let (ops, _) = parse_operations("filters:quality(webp:75)/example.com/i.jpg").unwrap();
        assert!(matches!(
            ops[0],
            Operation::Filters(ref f) if matches!(f[0], Filter::Quality(Quality::PerFormat(_)))
        ));
    }

    #[test]
    fn per_format_quality_canonicalizes_sorted_with_fallback_first() {
        for (given, canonical) in [
            (
                "filters:quality(webp:75,jpeg:80)",
                "filters:quality(jpeg:80,webp:75)",
            ),
            (
                "filters:quality(jpeg:90,70,avif:60)",
                "filters:quality(70,avif:60,jpeg:90)",
            ),
            ("filters:quality(jpg:80)", "filters:quality(jpeg:80)"),
        ] {
            let url = format!("{given}/example.com/i.jpg");
            let (ops, source) = parse_operations(&url).unwrap();
            assert_eq!(
                canonical_form(&ops, source),
                format!("{canonical}/example.com/i.jpg"),
                "{given}"
            );
        }
    }

    #[test]
    fn per_format_quality_rejects_bad_lists() {
        for bad in [
            "filters:quality(jpeg:80,jpeg:90)",
            "filters:quality(70,80)",
            "filters:quality(70,auto)",
            "filters:quality(unknown:80)",
            "filters:quality(mp4:80)",
            "filters:quality(webm:80)",
            "filters:quality(jpeg:0)",
            "filters:quality(jpeg:101)",
            "filters:quality(jpeg:)",
            "filters:quality(:80)",
            "filters:quality(jpeg:80,)",
            "filters:quality(,jpeg:80)",
            "filters:quality(jpeg:80,,webp:75)",
            "filters:quality(jpeg:80:90)",
            "filters:quality(jpeg:99999999999999999999)",
        ] {
            assert!(
                parse_operations(&format!("{bad}/example.com/i.jpg")).is_err(),
                "{bad} must not parse"
            );
        }
    }

    #[test]
    fn per_format_quality_resolves_for_the_negotiated_format() {
        let q = Quality::PerFormat(vec![
            (None, QualityValue::Value(70)),
            (Some(ImageFormat::Jpeg), QualityValue::Value(90)),
            (Some(ImageFormat::WebP), QualityValue::Auto),
        ]);
        assert_eq!(q.resolved_for(ImageFormat::Jpeg), Some(Quality::Value(90)));
        assert_eq!(q.resolved_for(ImageFormat::WebP), Some(Quality::Auto));
        // Unlisted formats fall back.
        assert_eq!(q.resolved_for(ImageFormat::Avif), Some(Quality::Value(70)));
        // No entry and no fallback: no directive in force.
        let q = Quality::PerFormat(vec![(Some(ImageFormat::Jpeg), QualityValue::Value(90))]);
        assert_eq!(q.resolved_for(ImageFormat::WebP), None);
        // Plain forms resolve to themselves for every format.
        assert_eq!(
            Quality::Value(55).resolved_for(ImageFormat::Png),
            Some(Quality::Value(55))
        );
        assert_eq!(
            Quality::Auto.resolved_for(ImageFormat::Gif),
            Some(Quality::Auto)
        );
    }

    #[test]
    fn parses_studio_defaults_and_arguments() {
        // Bare `studio(product)` takes every default.
        let (ops, _) = parse_operations("filters:studio(product)/example.com/img.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![Filter::Studio(StudioSpec {
                mode: StudioMode::Product,
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
                mode: StudioMode::Product,
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
    fn parses_studio_scene_mode() {
        // Bare `studio(scene)` (issue #198): the scene
        // mode at the default pad, keyed apart from every
        // product spelling.
        let (ops, _) = parse_operations("filters:studio(scene)/example.com/img.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![Filter::Studio(StudioSpec {
                mode: StudioMode::Scene,
                ..StudioSpec::default()
            })])]
        );
        assert_eq!(
            canonical_form(&ops, "example.com/img.jpg"),
            "filters:studio(scene)/example.com/img.jpg"
        );
        assert_ne!(
            canonical_key_form("filters:studio(scene)/example.com/img.jpg"),
            canonical_key_form("filters:studio(product)/example.com/img.jpg")
        );
        // `pad` is the one argument scene takes, with the
        // product range and default: spelled at the
        // default it collapses into the bare form.
        let (ops, _) =
            parse_operations("filters:studio(scene,pad:25)/example.com/img.jpg").unwrap();
        assert_eq!(
            canonical_form(&ops, "example.com/img.jpg"),
            "filters:studio(scene,pad:25)/example.com/img.jpg"
        );
        let (ops, _) =
            parse_operations("filters:studio(scene,pad:10)/example.com/img.jpg").unwrap();
        assert_eq!(
            canonical_form(&ops, "example.com/img.jpg"),
            "filters:studio(scene)/example.com/img.jpg"
        );
        // A scene request takes the studio preflight and
        // the AI gate, like product.
        let d = RequestDirectives::analyze(&ops);
        assert!(d.studio);
        assert!(d.uses_ai());
        // The product-only arguments are bad arguments
        // under scene, an unknown mode is not a mode, and
        // the pad bounds hold in both modes.
        for bad in [
            "filters:studio(scene,bg:white)/example.com/img.jpg",
            "filters:studio(scene,shadow:soft)/example.com/img.jpg",
            "filters:studio(scene,pad:41)/example.com/img.jpg",
            "filters:studio(scene,pad:-1)/example.com/img.jpg",
            "filters:studio(product,pad:41)/example.com/img.jpg",
            "filters:studio(portrait)/example.com/img.jpg",
            "filters:studio()/example.com/img.jpg",
            "filters:studio(scene,pad:10,pad:20)/example.com/img.jpg",
        ] {
            assert!(parse_operations(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn rejects_bad_studio_arguments() {
        for seg in [
            // The mode is required, and only `product`
            // and `scene` are modes.
            "filters:studio",
            "filters:studio()",
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
    fn remove_region_parses_like_the_other_region_filters() {
        // The grammar mirrors `blur_region`/`pixelate_region`
        // exactly (issue #176): four pixel values, w and h
        // at least 1, in a `filters:` list, canonicalized
        // back to the same spelling.
        let (ops, _) = parse_operations(
            "filters:blur_region(1,2,30,40):remove_region(5,6,70,80)/example.com/img.jpg",
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
                Filter::RemoveRegion {
                    x: 5,
                    y: 6,
                    w: 70,
                    h: 80
                },
            ])]
        );
        assert_eq!(
            canonical_key_form("filters:remove_region(5,6,70,80)/example.com/img.jpg"),
            "filters:remove_region(5,6,70,80)/example.com/img.jpg"
        );
        let d = RequestDirectives::analyze(&ops);
        assert!(d.remove_region);
        assert!(
            d.uses_ai(),
            "remove_region runs the inpainting model, so it takes an AI permit"
        );
    }

    #[test]
    fn remove_region_rejects_malformed_rectangles() {
        for seg in [
            "filters:remove_region(1,2,0,40)",
            "filters:remove_region(1,2,30,0)",
            "filters:remove_region(1,2,3)",
            "filters:remove_region(1,2,3,4,5)",
            "filters:remove_region(-1,0,10,10)",
            "filters:remove_region(0,0,10,16384)",
            "filters:remove_region(0,0,10)",
            "filters:remove_region()",
            "filters:remove_region(1.5,2,3,4)",
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
    fn parses_avif_format() {
        let (ops, _) = parse_operations("filters:format(avif)/example.com/img.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::Avif
            ))])]
        );
    }

    #[test]
    fn parses_video_formats() {
        for (spelling, format) in [("mp4", ImageFormat::Mp4), ("webm", ImageFormat::WebM)] {
            let (ops, _) =
                parse_operations(&format!("filters:format({spelling})/example.com/img.gif"))
                    .unwrap();
            assert_eq!(
                ops,
                vec![Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                    format
                ))])]
            );
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
    fn parses_focal_point() {
        // The zoom is optional on input and defaults to 1.
        let (ops, _) = parse_operations("focalpoint:0.75x0.4/example.com/img.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::FocalPoint {
                x: 0.75,
                y: 0.4,
                zoom: 1.0,
            }]
        );
        let (ops, _) = parse_operations("focalpoint:0.75x0.4:2/example.com/img.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::FocalPoint {
                x: 0.75,
                y: 0.4,
                zoom: 2.0,
            }]
        );
        // The corners of the frame are valid points, and the
        // zoom endpoints are in.
        assert!(parse_operations("focalpoint:0x0:100/example.com/img.jpg").is_ok());
        assert!(parse_operations("focalpoint:1x1:0.01/example.com/img.jpg").is_ok());
    }

    #[test]
    fn rejects_bad_focal_point() {
        for seg in [
            "focalpoint:",
            "focalpoint:0.75",
            "focalpoint:0.75x",
            "focalpoint:x0.4",
            "focalpoint:0.75x0.4x0.2",
            "focalpoint:1.5x0.4",
            "focalpoint:0.75x-0.1",
            "focalpoint:nanx0.4",
            "focalpoint:0.75x0.4:0",
            "focalpoint:0.75x0.4:-2",
            "focalpoint:0.75x0.4:101",
            "focalpoint:0.75x0.4:inf",
            "focalpoint:0.75x0.4:2:3",
            "focalpoint:0.75x0.4:",
        ] {
            let path = format!("{seg}/example.com/img.jpg");
            assert!(
                matches!(
                    parse_operations(&path),
                    Err(ParseError::InvalidFocalPoint(_))
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
    fn parses_trim_with_per_side_amounts() {
        let (ops, _) = parse_operations("filters:trim(20,5,0,5,0)/example.com/img.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![Filter::TrimSides(
                20,
                [5, 0, 5, 0]
            )])]
        );
        // The largest side the grammar takes.
        assert!(parse_operations("filters:trim(0,4294967295,0,0,0)/example.com/img.jpg").is_ok());
        // Partial forms are malformed, not shorthand.
        for seg in [
            "filters:trim(20,5)",
            "filters:trim(20,5,0)",
            "filters:trim(20,5,0,5)",
            "filters:trim(20,5,0,5,0,1)",
            "filters:trim(20,)",
            "filters:trim(,5,0,5,0)",
        ] {
            assert!(
                parse_operations(&format!("{seg}/example.com/img.jpg")).is_err(),
                "{seg}"
            );
        }
        // The tolerance keeps its own bound inside the
        // five-argument form, and the sides are unsigned.
        for seg in [
            "filters:trim(256,0,0,0,0)",
            "filters:trim(20,-1,0,0,0)",
            "filters:trim(20,4294967296,0,0,0)",
            "filters:trim(20,1.5,0,0,0)",
        ] {
            assert!(
                parse_operations(&format!("{seg}/example.com/img.jpg")).is_err(),
                "{seg}"
            );
        }
    }

    #[test]
    fn parses_gif_format() {
        let (ops, _) = parse_operations("filters:format(gif)/example.com/img.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![Filter::Format(FormatSpec::plain(
                ImageFormat::Gif
            ))])]
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
    fn parses_objects_crop_alignment() {
        // Issue #318: one class, the scale defaulted.
        let (ops, _) =
            parse_operations("crop:300x200:objects(person)/example.com/img.jpg").unwrap();
        assert_eq!(
            ops,
            vec![Operation::Crop {
                size: ImageSize {
                    width: Dimension::Pixels(300),
                    height: Dimension::Pixels(200),
                    flip_horizontal: false,
                    flip_vertical: false,
                },
                align: Align::Objects {
                    classes: vec!["person".to_string()],
                    scale: 1.0,
                },
            }]
        );
        // The mode takes the detection preflight and the AI
        // gate, through its own directive fields: it is not
        // class redaction and not a face operation.
        let d = RequestDirectives::analyze(&ops);
        assert!(d.objects_crop);
        assert_eq!(d.object_classes, ["person"]);
        assert!(d.uses_ai());
        assert!(!d.has_class_redaction());
        assert!(!d.needs_face_model);

        // Several classes and an explicit scale; class names
        // percent-decode like the class filters' argument.
        let (ops, _) = parse_operations(
            "crop:300x200:objects(person,traffic%20light,car):1.5/example.com/img.jpg",
        )
        .unwrap();
        assert!(matches!(
            &ops[0],
            Operation::Crop {
                align: Align::Objects { classes, scale },
                ..
            } if classes == &["person", "traffic light", "car"] && *scale == 1.5
        ));
        let d = RequestDirectives::analyze(&ops);
        assert_eq!(d.object_classes, ["person", "traffic light", "car"]);

        // A class named twice names one class: the first
        // spelling stands (matching itself is
        // case-insensitive, like the vocabulary checks).
        let (ops, _) =
            parse_operations("crop:300x200:objects(Person,person,car)/example.com/img.jpg")
                .unwrap();
        assert!(matches!(
            &ops[0],
            Operation::Crop {
                align: Align::Objects { classes, .. },
                ..
            } if classes == &["Person", "car"]
        ));
    }

    #[test]
    fn objects_crop_canonical_form() {
        // The scale is optional on input and always spelled
        // in the canonical form, the focalpoint zoom rule;
        // class names re-encode deterministically.
        assert_eq!(
            canonical_key_form("crop:300x200:objects(person)/example.com/img.jpg"),
            "crop:300x200:objects(person):1/example.com/img.jpg"
        );
        assert_eq!(
            canonical_key_form("crop:300x200:objects(person):1.50/example.com/img.jpg"),
            "crop:300x200:objects(person):1.5/example.com/img.jpg"
        );
        assert_eq!(
            canonical_key_form(
                "crop:300x200:objects(traffic%20light,person):2/example.com/img.jpg"
            ),
            "crop:300x200:objects(traffic%20light,person):2/example.com/img.jpg"
        );
        // The canonical form parses back to the same ops.
        let (ops, _) =
            parse_operations("crop:300x200:objects(traffic%20light,person):2/example.com/img.jpg")
                .unwrap();
        assert!(matches!(
            &ops[0],
            Operation::Crop {
                align: Align::Objects { classes, scale },
                ..
            } if classes == &["traffic light", "person"] && *scale == 2.0
        ));
    }

    #[test]
    fn rejects_bad_objects_crops() {
        for bad in [
            // No classes, or an empty one.
            "crop:300x200:objects()/example.com/img.jpg",
            "crop:300x200:objects( )/example.com/img.jpg",
            "crop:300x200:objects(person,)/example.com/img.jpg",
            "crop:300x200:objects(,person)/example.com/img.jpg",
            // The class charset is the class filters' own.
            "crop:300x200:objects(per%24on)/example.com/img.jpg",
            "crop:300x200:objects(per%2Con)/example.com/img.jpg",
            // The parameterized form must be complete.
            "crop:300x200:objects/example.com/img.jpg",
            "crop:300x200:objects(person/example.com/img.jpg",
            "crop:300x200:objects(person)junk/example.com/img.jpg",
            "crop:300x200:objects(person):/example.com/img.jpg",
            "crop:300x200:objects(person):1.5:extra/example.com/img.jpg",
            // The scale is a positive multiplier up to the cap.
            "crop:300x200:objects(person):0/example.com/img.jpg",
            "crop:300x200:objects(person):-1/example.com/img.jpg",
            "crop:300x200:objects(person):100.5/example.com/img.jpg",
            "crop:300x200:objects(person):NaN/example.com/img.jpg",
            "crop:300x200:objects(person):inf/example.com/img.jpg",
            "crop:300x200:objects(person):wide/example.com/img.jpg",
            // The exact-box contract still applies.
            "crop:300:objects(person)/example.com/img.jpg",
            "crop:0x200:objects(person)/example.com/img.jpg",
            "crop:-300x200:objects(person)/example.com/img.jpg",
        ] {
            assert!(parse_operations(bad).is_err(), "{bad}");
        }
        // The class list is bounded: 32 names parse in
        // full.
        let names = |n: usize| {
            (0..n)
                .map(|i| format!("c{i}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        let ok = format!("crop:300x200:objects({})/example.com/img.jpg", names(32));
        let (ops, _) = parse_operations(&ok).unwrap();
        assert!(matches!(
            &ops[0],
            Operation::Crop {
                align: Align::Objects { classes, .. },
                ..
            } if classes.len() == MAX_OBJECTS_CLASSES
                && classes.first().map(String::as_str) == Some("c0")
                && classes.last().map(String::as_str) == Some("c31")
        ));
        // A 33rd distinct class is a bad crop (issue #408),
        // as it was at 0.9.10, not a silent truncation to
        // the first 32 (what the #391 fix briefly shipped).
        for n in [33, 40] {
            let over = format!("crop:300x200:objects({})/example.com/img.jpg", names(n));
            assert!(
                matches!(parse_operations(&over), Err(ParseError::InvalidCrop(_))),
                "{n} classes"
            );
        }
        // The cap counts distinct names: repeats in any case
        // fold before they count, so 32 classes spelled 64
        // times parse to the 32.
        let repeated = format!(
            "crop:300x200:objects({},{})/example.com/img.jpg",
            names(32),
            names(32).to_uppercase()
        );
        let (ops, _) = parse_operations(&repeated).unwrap();
        assert!(matches!(
            &ops[0],
            Operation::Crop {
                align: Align::Objects { classes, .. },
                ..
            } if classes.len() == MAX_OBJECTS_CLASSES
        ));
        // An invalid name past a full set is still a bad crop.
        let bad_tail = format!(
            "crop:300x200:objects({},per%24on)/example.com/img.jpg",
            names(32)
        );
        assert!(parse_operations(&bad_tail).is_err());
    }

    #[test]
    fn a_long_objects_list_is_rejected_at_the_cap() {
        // Hostile input (issues #391, #408): a list of
        // thousands of distinct names is rejected when the
        // 33rd arrives, so the parse never scans past a
        // cap-sized set; one of thousands of repeats of a
        // single name parses to that one class. Both stay
        // linear in the segment, which the URI length
        // already bounds.
        let distinct: Vec<String> = (0..20_000).map(|i| format!("c{i}")).collect();
        let over = format!(
            "crop:300x200:objects({})/example.com/img.jpg",
            distinct.join(",")
        );
        assert!(matches!(
            parse_operations(&over),
            Err(ParseError::InvalidCrop(_))
        ));
        let same = vec!["person"; 20_000].join(",");
        let (ops, _) =
            parse_operations(&format!("crop:300x200:objects({same})/example.com/img.jpg")).unwrap();
        assert!(matches!(
            &ops[0],
            Operation::Crop {
                align: Align::Objects { classes, .. },
                ..
            } if classes.len() == 1
        ));
    }

    #[test]
    fn recolorclass_parses_like_the_other_class_filters() {
        // Issue #175: <class>,<hex>, the class percent-
        // decoded like the other class filters, the hex in
        // the color parameters' spelling; the canonical
        // form lowercases the hex and re-encodes the name.
        let (ops, _) =
            parse_operations("filters:recolorclass(traffic%20light,FF8800)/example.com/img.jpg")
                .unwrap();
        assert_eq!(
            ops,
            vec![Operation::Filters(vec![Filter::RecolorClass(
                "traffic light".to_string(),
                [0xff, 0x88, 0x00],
            )])]
        );
        assert_eq!(
            canonical_key_form("filters:recolorclass(traffic%20light,FF8800)/example.com/img.jpg"),
            "filters:recolorclass(traffic%20light,ff8800)/example.com/img.jpg"
        );
        // A recolor names a class like the redaction
        // filters: it joins the class list and takes the
        // AI gate.
        let d = RequestDirectives::analyze(&ops);
        assert_eq!(d.classes, ["traffic light"]);
        assert!(d.has_class_redaction());
        assert!(d.uses_ai());
        for bad in [
            "filters:recolorclass(person)/example.com/img.jpg",
            "filters:recolorclass(person,)/example.com/img.jpg",
            "filters:recolorclass(,ff0000)/example.com/img.jpg",
            "filters:recolorclass(person,ff00)/example.com/img.jpg",
            "filters:recolorclass(person,ff00000)/example.com/img.jpg",
            "filters:recolorclass(person,gg0000)/example.com/img.jpg",
            "filters:recolorclass(person,#ff0000)/example.com/img.jpg",
            "filters:recolorclass(per%24on,ff0000)/example.com/img.jpg",
            "filters:recolorclass/example.com/img.jpg",
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
    fn frame_segments_parse_and_canonicalize() {
        // The three selector forms (issue #214).
        let (ops, src, _, _) =
            parse_operations_with_expiry("frame:poster/example.com/v.mp4").unwrap();
        assert_eq!(ops, vec![Operation::Frame(FrameSelection::Poster)]);
        assert_eq!(src, "example.com/v.mp4");
        let (ops, ..) = parse_operations_with_expiry("frame:12.5/example.com/v.mp4").unwrap();
        assert_eq!(ops, vec![Operation::Frame(FrameSelection::Seconds(12.5))]);
        let (ops, ..) = parse_operations_with_expiry("frame:50%/example.com/v.mp4").unwrap();
        assert_eq!(ops, vec![Operation::Frame(FrameSelection::Percent(50.0))]);
        // Bounds are inclusive: zero, the full duration, and
        // the 24 h cap itself all parse.
        for (seg, want) in [
            ("frame:0", FrameSelection::Seconds(0.0)),
            ("frame:100%", FrameSelection::Percent(100.0)),
            ("frame:0%", FrameSelection::Percent(0.0)),
            ("frame:86400", FrameSelection::Seconds(86_400.0)),
            ("frame:33.333%", FrameSelection::Percent(33.333)),
        ] {
            let (ops, ..) = parse_operations_with_expiry(&format!("{seg}/example.com/v.mp4"))
                .unwrap_or_else(|e| panic!("{seg}: {e}"));
            assert_eq!(ops, vec![Operation::Frame(want)], "{seg}");
        }
        // Canonical spelling: whole values print as integers,
        // fractions shorten, so spelling variants are one
        // request with one cache entry.
        assert_eq!(
            canonical_key_form("frame:1.50/300x200/example.com/v.mp4"),
            "frame:1.5/300x200/example.com/v.mp4"
        );
        assert_eq!(
            canonical_key_form("frame:2.0/example.com/v.mp4"),
            "frame:2/example.com/v.mp4"
        );
        assert_eq!(
            canonical_key_form("frame:050%/example.com/v.mp4"),
            "frame:50%/example.com/v.mp4"
        );
        // The directive analysis sees the selection, and the
        // last segment wins when a request carries several.
        let (ops, ..) =
            parse_operations_with_expiry("frame:1/frame:poster/example.com/v.mp4").unwrap();
        assert_eq!(
            RequestDirectives::analyze(&ops).frame,
            Some(FrameSelection::Poster)
        );
        let (ops, ..) = parse_operations_with_expiry("300x200/example.com/v.mp4").unwrap();
        assert_eq!(RequestDirectives::analyze(&ops).frame, None);
        // A frame segment inside the source is URL text.
        let (ops, src, _, _) = parse_operations_with_expiry("example.com/frame:1/v.mp4").unwrap();
        assert!(ops.is_empty());
        assert_eq!(src, "example.com/frame:1/v.mp4");
    }

    #[test]
    fn frame_segments_refuse_malformed_selectors() {
        for bad in [
            "frame:/example.com/v.mp4",
            "frame:banana/example.com/v.mp4",
            "frame:-1/example.com/v.mp4",
            "frame:+1/example.com/v.mp4",
            "frame:.5/example.com/v.mp4",
            "frame:1./example.com/v.mp4",
            "frame:1.2.3/example.com/v.mp4",
            "frame:1e3/example.com/v.mp4",
            "frame:86401/example.com/v.mp4",
            "frame:86400.5/example.com/v.mp4",
            "frame:101%/example.com/v.mp4",
            "frame:100.1%/example.com/v.mp4",
            "frame:%/example.com/v.mp4",
            "frame:50%%/example.com/v.mp4",
            "frame:poster%/example.com/v.mp4",
            "frame:Poster/example.com/v.mp4",
        ] {
            let err = parse_operations_with_expiry(bad).unwrap_err();
            assert!(matches!(err, ParseError::InvalidFrame(_)), "{bad}: {err:?}");
            assert_eq!(err.code(), "invalid_frame", "{bad}");
        }
    }

    #[test]
    fn page_segments_parse_and_canonicalize() {
        // The selector form (issue #215): 1-based digits.
        let (ops, src, _, _) = parse_operations_with_expiry("page:1/example.com/doc.pdf").unwrap();
        assert_eq!(ops, vec![Operation::Page(1)]);
        assert_eq!(src, "example.com/doc.pdf");
        // Bounds are inclusive: the first page and the
        // grammar bound itself both parse.
        for (seg, want) in [("page:1", 1), ("page:999999", 999_999), ("page:42", 42)] {
            let (ops, ..) = parse_operations_with_expiry(&format!("{seg}/example.com/doc.pdf"))
                .unwrap_or_else(|e| panic!("{seg}: {e}"));
            assert_eq!(ops, vec![Operation::Page(want)], "{seg}");
        }
        // Canonical spelling: leading zeros are spelling,
        // not meaning, so page:007 and page:7 are one
        // request with one cache entry.
        assert_eq!(
            canonical_key_form("page:007/300x200/example.com/doc.pdf"),
            "page:7/300x200/example.com/doc.pdf"
        );
        assert_eq!(
            canonical_key_form("page:000001/example.com/doc.pdf"),
            "page:1/example.com/doc.pdf"
        );
        // The directive analysis sees the selection, and the
        // last segment wins when a request carries several.
        let (ops, ..) = parse_operations_with_expiry("page:2/page:5/example.com/doc.pdf").unwrap();
        assert_eq!(RequestDirectives::analyze(&ops).page, Some(5));
        let (ops, ..) = parse_operations_with_expiry("300x200/example.com/doc.pdf").unwrap();
        assert_eq!(RequestDirectives::analyze(&ops).page, None);
        // A page segment inside the source is URL text.
        let (ops, src, _, _) = parse_operations_with_expiry("example.com/page:2/doc.pdf").unwrap();
        assert!(ops.is_empty());
        assert_eq!(src, "example.com/page:2/doc.pdf");
    }

    #[test]
    fn page_segments_refuse_malformed_selectors() {
        for bad in [
            "page:/example.com/doc.pdf",
            "page:0/example.com/doc.pdf",
            "page:00/example.com/doc.pdf",
            "page:-1/example.com/doc.pdf",
            "page:+1/example.com/doc.pdf",
            "page:1.5/example.com/doc.pdf",
            "page:1./example.com/doc.pdf",
            "page:.5/example.com/doc.pdf",
            "page:1e3/example.com/doc.pdf",
            "page:abc/example.com/doc.pdf",
            "page:1000000/example.com/doc.pdf",
            "page:9999999/example.com/doc.pdf",
            "page: 1/example.com/doc.pdf",
            "page:1 /example.com/doc.pdf",
        ] {
            let err = parse_operations_with_expiry(bad).unwrap_err();
            assert!(matches!(err, ParseError::InvalidPage(_)), "{bad}: {err:?}");
            assert_eq!(err.code(), "invalid_page", "{bad}");
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

    fn root_map() -> std::collections::HashMap<String, String> {
        [
            (
                "photos".to_string(),
                "https://cdn.example.com/photos".to_string(),
            ),
            (
                "media".to_string(),
                "https://assets.example.com".to_string(),
            ),
        ]
        .into_iter()
        .collect()
    }

    #[test]
    fn root_sources_expand_against_the_named_root() {
        let map = root_map();
        let (expanded, reference) = expand_source_root("root:photos/cat.jpg", &map).unwrap();
        assert_eq!(expanded, "https://cdn.example.com/photos/cat.jpg");
        assert_eq!(
            reference,
            Some(RootRef {
                name: "photos".to_string(),
                path: "cat.jpg".to_string(),
            })
        );
        // A root with no path prefix joins at the host root.
        let (expanded, _) = expand_source_root("root:media/a/b.png", &map).unwrap();
        assert_eq!(expanded, "https://assets.example.com/a/b.png");
        // Non-root sources pass through untouched.
        for source in [
            "https://example.com/cat.jpg",
            "s3://media/cat.jpg",
            "file:///srv/cat.jpg",
            "example.com/cat.jpg",
            "https://example.com/root:photos/cat.jpg",
        ] {
            assert_eq!(
                expand_source_root(source, &map).unwrap(),
                (source.to_string(), None),
                "{source}"
            );
        }
    }

    #[test]
    fn root_sources_answer_unknown_root_for_any_unconfigured_name() {
        let map = root_map();
        assert_eq!(
            expand_source_root("root:nope/cat.jpg", &map),
            Err(ParseError::UnknownRoot("nope".to_string()))
        );
        // With no roots configured every name is unknown, the
        // same rule presets follow.
        let empty = std::collections::HashMap::new();
        assert_eq!(
            expand_source_root("root:photos/cat.jpg", &empty),
            Err(ParseError::UnknownRoot("photos".to_string()))
        );
        assert_eq!(
            ParseError::UnknownRoot("x".to_string()).code(),
            "unknown_root"
        );
        assert_eq!(
            ParseError::InvalidRoot("x".to_string()).code(),
            "invalid_root"
        );
    }

    #[test]
    fn root_paths_normalize_dots_but_never_escape_the_root() {
        let map = root_map();
        for (path, joined) in [
            ("a/./cat.jpg", "a/cat.jpg"),
            ("a/b/../cat.jpg", "a/cat.jpg"),
            ("./cat.jpg", "cat.jpg"),
            ("a//cat.jpg", "a//cat.jpg"),
        ] {
            let (expanded, _) = expand_source_root(&format!("root:photos/{path}"), &map).unwrap();
            assert_eq!(expanded, format!("https://cdn.example.com/photos/{joined}"));
        }
        for bad in [
            "root:photos",
            "root:photos/",
            "root:/cat.jpg",
            "root:photos/../secret.jpg",
            "root:photos/a/../../secret.jpg",
            "root:photos/%2e%2e/secret.jpg",
            "root:photos/%2E./secret.jpg",
            "root:photos/a/%2e%2e/%2e%2e/secret.jpg",
            "root:photos/..%2fsecret.jpg",
            "root:photos/a%2fb.jpg",
            "root:photos/..\\secret.jpg",
            "root:photos/a/..",
        ] {
            assert_eq!(
                expand_source_root(bad, &map),
                Err(ParseError::InvalidRoot(bad.to_string())),
                "{bad}"
            );
        }
    }

    #[test]
    fn root_expansion_inside_a_remainder_matches_the_serve_parse() {
        let map = root_map();
        assert_eq!(
            expand_source_root_in_remainder("300x200/root:photos/cat.jpg", &map).unwrap(),
            "300x200/https://cdn.example.com/photos/cat.jpg"
        );
        // exp: metadata in front of the source does not
        // disturb the rewrite, and the source's own text
        // after the root segment stays intact.
        assert_eq!(
            expand_source_root_in_remainder("exp:999/root:media/a/root:b.jpg", &map).unwrap(),
            "exp:999/https://assets.example.com/a/root:b.jpg"
        );
        // Full sources and unparseable remainders pass
        // through unchanged.
        assert_eq!(
            expand_source_root_in_remainder("300x200/https://example.com/cat.jpg", &map).unwrap(),
            "300x200/https://example.com/cat.jpg"
        );
        assert_eq!(
            expand_source_root_in_remainder("root:nope/cat.jpg", &map),
            Err(ParseError::UnknownRoot("nope".to_string()))
        );
    }

    /// Issue #250: any name that is not exactly a configured
    /// key is `unknown_root`, however it is spelled. Names
    /// are never decoded, normalized, or case-folded before
    /// the lookup, so lookalikes cannot reach a configured
    /// root.
    #[test]
    fn root_names_that_are_not_the_configured_key_are_unknown_root() {
        let map = root_map();
        for (source, name) in [
            ("root:Photos/cat.jpg", "Photos"),
            ("root:PHOTOS/cat.jpg", "PHOTOS"),
            ("root:photo/cat.jpg", "photo"),
            ("root:photoss/cat.jpg", "photoss"),
            ("root:photos./cat.jpg", "photos."),
            ("root:photos name/cat.jpg", "photos name"),
            ("root: photos/cat.jpg", " photos"),
            ("root:photos%2f/cat.jpg", "photos%2f"),
            ("root:photos:80/cat.jpg", "photos:80"),
            ("root:root:photos/cat.jpg", "root:photos"),
            ("root:fötös/cat.jpg", "fötös"),
            ("root:🦀/cat.jpg", "🦀"),
        ] {
            let err = expand_source_root(source, &map).unwrap_err();
            assert_eq!(err, ParseError::UnknownRoot(name.to_string()), "{source}");
            assert_eq!(err.code(), "unknown_root", "{source}");
        }
        // A name of arbitrary length is a lookup miss, not a
        // failure mode of its own.
        let long_name = "a".repeat(10_000);
        let source = format!("root:{long_name}/cat.jpg");
        assert_eq!(
            expand_source_root(&source, &map),
            Err(ParseError::UnknownRoot(long_name))
        );
    }

    /// Issue #250: short forms that cannot name a root plus
    /// a path under it are `invalid_root`: the shape check
    /// runs before the name lookup, so even a configured
    /// name cannot rescue a malformed spelling.
    #[test]
    fn root_short_forms_without_a_usable_path_are_invalid_root() {
        let map = root_map();
        for source in [
            "root:",                   // nothing after the prefix
            "root:/",                  // a slash and nothing else
            "root://cat.jpg",          // empty name
            "root:photos//",           // path is one empty segment
            "root:photos///",          // path is only empty segments
            "root:photos/.",           // path normalizes to nothing
            "root:photos/%2e",         // an encoded dot, the same
            "root:photos/%2e/",        // dot and an empty segment
            "root:photos/%",           // malformed percent escape
            "root:photos/%zz/cat.jpg", // malformed percent escape
            "root:photos/cat.jpg%",    // truncated escape at the end
            "root:photos/%ff/cat.jpg", // escape to invalid UTF-8
        ] {
            let err = expand_source_root(source, &map).unwrap_err();
            assert_eq!(err, ParseError::InvalidRoot(source.to_string()), "{source}");
            assert_eq!(err.code(), "invalid_root", "{source}");
        }
    }

    /// Issue #250: traversal and host-change tricks. What
    /// can be refused is refused; what cannot be refused
    /// (it is a legal path under the root) still expands
    /// to a URL under the root's base, byte for byte.
    #[test]
    fn root_paths_cannot_escape_by_encoding_or_host_tricks() {
        let map = root_map();
        for bad in [
            "root:photos/..%5csecret.jpg",      // encoded backslash climb
            "root:photos/%2e%2e%5csecret.jpg",  // encoded dots, encoded backslash
            "root:photos/a/%5c..%5csecret.jpg", // separators inside one segment
            "root:photos/.%2e/secret.jpg",      // mixed-encoding dotdot
        ] {
            assert_eq!(
                expand_source_root(bad, &map),
                Err(ParseError::InvalidRoot(bad.to_string())),
                "{bad}"
            );
        }
        for (source, expanded) in [
            // Decoding happens exactly once, like every URL
            // layer: a double-encoded dot segment expands
            // to a literal `%2e%2e` file name under the
            // root, not to a dot segment anywhere.
            (
                "root:photos/%252e%252e/secret.jpg",
                "https://cdn.example.com/photos/%252e%252e/secret.jpg",
            ),
            // Three dots are a file name, not a dot segment.
            (
                "root:photos/%2e%2e./secret.jpg",
                "https://cdn.example.com/photos/%2e%2e./secret.jpg",
            ),
            // An `@`, `?`, or `#` inside the path is path
            // text: the base already fixed the scheme, host,
            // and prefix, so none of them can re-target the
            // fetch.
            (
                "root:photos/@evil.example/cat.jpg",
                "https://cdn.example.com/photos/@evil.example/cat.jpg",
            ),
            (
                "root:photos/cat.jpg?x=1",
                "https://cdn.example.com/photos/cat.jpg?x=1",
            ),
            (
                "root:photos/cat.jpg#frag",
                "https://cdn.example.com/photos/cat.jpg#frag",
            ),
            (
                "root:photos//evil.example/cat.jpg",
                "https://cdn.example.com/photos//evil.example/cat.jpg",
            ),
            // Trailing-dot segments are file names, not
            // dot segments.
            (
                "root:photos/cat.jpg.",
                "https://cdn.example.com/photos/cat.jpg.",
            ),
            (
                "root:photos/a./cat.jpg",
                "https://cdn.example.com/photos/a./cat.jpg",
            ),
            // A `..` that stays inside the root is a normal
            // path, however it is spelled.
            (
                "root:photos/a/b/../../cat.jpg",
                "https://cdn.example.com/photos/cat.jpg",
            ),
        ] {
            let (got, _) = expand_source_root(source, &map).unwrap();
            assert_eq!(got, expanded, "{source}");
            assert!(
                got.starts_with("https://cdn.example.com/photos/"),
                "{source}"
            );
        }
    }

    /// Issue #250: depth does not change the rules, and no
    /// input in the adversarial space expands outside its
    /// root's base or fails with anything but the two
    /// frozen root codes.
    #[test]
    fn root_expansion_holds_its_invariants_at_any_depth() {
        let map = root_map();
        // Ten thousand segments deep, served fine...
        let deep = format!("root:photos/{}/cat.jpg", "a/".repeat(9_999));
        let (expanded, _) = expand_source_root(&deep, &map).unwrap();
        assert!(expanded.starts_with("https://cdn.example.com/photos/a/"));
        assert!(expanded.ends_with("/cat.jpg"));
        // ...and ten thousand climbs are refused by the
        // second one already.
        let climbs = format!("root:photos/a/{}/cat.jpg", "../".repeat(9_999));
        assert_eq!(
            expand_source_root(&climbs, &map),
            Err(ParseError::InvalidRoot(climbs.clone()))
        );
        // The sweep: every combination of an adversarial
        // name and an adversarial path either refuses with
        // a frozen root code or expands under the base of
        // the root it named.
        let names = ["photos", "media", "nope", "Photos", "", "🦀", "a b"];
        let paths = [
            "cat.jpg",
            "../cat.jpg",
            "%2e%2e/cat.jpg",
            "..%5ccat.jpg",
            "%252e%252e/cat.jpg",
            "@evil.example/cat.jpg",
            "a//b.jpg",
            ".",
            "..",
            "%",
            "a/./b/../cat.jpg",
            "cat.jpg?x",
            "cat.jpg#x",
        ];
        for name in names {
            for path in paths {
                let source = format!("root:{name}/{path}");
                match expand_source_root(&source, &map) {
                    Ok((expanded, reference)) => {
                        let reference = reference.expect("a root source names its root");
                        let base = &map[&reference.name];
                        assert!(
                            expanded.starts_with(&format!("{base}/")),
                            "{source} expanded to {expanded}"
                        );
                    }
                    Err(e) => assert!(
                        matches!(e.code(), "unknown_root" | "invalid_root"),
                        "{source}: {e}"
                    ),
                }
            }
        }
    }

    /// Issue #250: `root:` only has meaning as the very
    /// start of the source. Inside a full source it is path
    /// text, and a remainder whose source position holds a
    /// malformed short form fails the same way the serve
    /// path's own expansion fails.
    #[test]
    fn root_prefix_only_counts_at_the_source_start() {
        let map = root_map();
        assert_eq!(
            expand_source_root_in_remainder(
                "300x200/https://example.com/root:photos/cat.jpg",
                &map
            )
            .unwrap(),
            "300x200/https://example.com/root:photos/cat.jpg"
        );
        assert_eq!(
            expand_source_root_in_remainder("300x200/root:photos", &map),
            Err(ParseError::InvalidRoot("root:photos".to_string()))
        );
        assert_eq!(
            expand_source_root_in_remainder("300x200/root:/cat.jpg", &map),
            Err(ParseError::InvalidRoot("root:/cat.jpg".to_string()))
        );
        assert_eq!(
            expand_source_root_in_remainder("300x200/root:photos/%2e%2e/secret.jpg", &map),
            Err(ParseError::InvalidRoot(
                "root:photos/%2e%2e/secret.jpg".to_string()
            ))
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
    fn preset_names_dedupe_in_first_seen_order() {
        assert_eq!(
            preset_names("preset:a/preset:b/preset:a/exp:999/preset:c/preset:b/example.com/i.jpg"),
            vec!["a", "b", "c"]
        );
        // The scan stops where expansion stops: a `preset:`
        // segment in the source path is URL text.
        assert_eq!(
            preset_names("preset:a/example.com/preset:b/i.jpg"),
            vec!["a"]
        );
        assert!(preset_names("example.com/i.jpg").is_empty());
        assert!(preset_names("preset:/example.com/i.jpg").is_empty());
    }

    #[test]
    fn preset_names_stay_linear_on_long_distinct_lists() {
        // Hostile input (issue #409): the scan runs before
        // expansion rejects unknown names, so a remainder of
        // thousands of distinct names must cost linear work.
        let remainder = |names: &[String]| {
            let mut text: String = names.iter().map(|n| format!("preset:{n}/")).collect();
            text.push_str("example.com/img.jpg");
            text
        };
        // The report's shape: about 6,000 distinct short
        // names (a 65 KB remainder), then the same names
        // again, which fold into the first-seen order.
        let names: Vec<String> = (0..6_000).map(|i| format!("{i:x}")).collect();
        let doubled: Vec<String> = names.iter().chain(names.iter()).cloned().collect();
        assert_eq!(preset_names(&remainder(&names)), names);
        assert_eq!(preset_names(&remainder(&doubled)), names);
        // Linear, not quadratic: 50,000 distinct names take
        // milliseconds with the hash set, where the old scan
        // of the kept list made about 1.25 billion string
        // comparisons (seconds even in a release build).
        let many: Vec<String> = (0..50_000).map(|i| format!("{i:x}")).collect();
        let text = remainder(&many);
        let started = std::time::Instant::now();
        assert_eq!(preset_names(&text).len(), many.len());
        assert!(
            started.elapsed() < std::time::Duration::from_secs(2),
            "50,000 names took {:?}",
            started.elapsed()
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
    fn default_ops_refuse_request_metadata() {
        // Issue #469: the parse used to accept `exp:` and
        // `maxage:` and drop them, so a default lifetime
        // booted cleanly and never applied. Both are refused
        // now, naming the segment, wherever they sit.
        let mut presets = std::collections::HashMap::new();
        presets.insert("thumb".to_string(), "100x0".to_string());
        for (text, want) in [
            (
                "filters:format(webp)/maxage:60",
                ParseError::InvalidMaxAge("maxage:60".to_string()),
            ),
            (
                "maxage:0",
                ParseError::InvalidMaxAge("maxage:0".to_string()),
            ),
            (
                "exp:1893456000/filters:format(webp)",
                ParseError::InvalidExpiry("exp:1893456000".to_string()),
            ),
            (
                "preset:thumb/exp:1/maxage:5",
                ParseError::InvalidExpiry("exp:1".to_string()),
            ),
            (
                "maxage:60/preset:thumb",
                ParseError::InvalidMaxAge("maxage:60".to_string()),
            ),
        ] {
            assert_eq!(parse_default_ops(text, &presets), Err(want), "{text}");
        }
        // Malformed metadata still fails as the parser's own
        // error, and metadata-free strings still parse.
        assert_eq!(
            parse_default_ops("maxage:soon", &presets),
            Err(ParseError::InvalidMaxAge("maxage:soon".to_string()))
        );
        assert_eq!(
            parse_default_ops("preset:thumb/filters:format(webp)", &presets)
                .unwrap()
                .len(),
            2
        );
    }

    #[test]
    fn operation_count_counts_each_filter_and_every_other_operation_once() {
        let (ops, _) = parse_operations(
            "crop:1200x630:smart/filters:grayscale:quality(80)/exp:9/maxage:5/300x200/example.com/i.jpg",
        )
        .unwrap();
        // crop (1) + two filters (2) + size (1); exp and
        // maxage are not operations.
        assert_eq!(operation_count(&ops), 4);
        assert_eq!(operation_count(&[]), 0);
        assert!(check_operation_count(&[]).is_ok());
        // Exactly at the cap passes, either way it is spelled.
        let one_segment = vec![Operation::Filters(vec![Filter::Invert; MAX_OPERATIONS])];
        assert!(check_operation_count(&one_segment).is_ok());
        let segments = vec![Operation::Filters(vec![Filter::Invert]); MAX_OPERATIONS];
        assert!(check_operation_count(&segments).is_ok());
        // One more fails with the count in the message and the
        // engine's code, so the HTTP answer did not change when
        // the check moved to parse time.
        let mut over = segments.clone();
        over.push(Operation::Filters(vec![Filter::Invert]));
        let err = check_operation_count(&over).unwrap_err();
        assert_eq!(err, ParseError::TooManyOperations(MAX_OPERATIONS + 1));
        assert_eq!(err.code(), "unsupported_operation");
        assert_eq!(
            err.to_string(),
            "request has 101 operations, over the 100 operation limit"
        );
        // An empty filters list contributes nothing, and a
        // chain about as long as a 64 KiB URL can spell in one
        // segment is refused with its full count.
        assert_eq!(operation_count(&[Operation::Filters(Vec::new())]), 0);
        let (hostile, _) = parse_operations(&format!(
            "filters:{}/example.com/i.jpg",
            vec!["invert"; 9_000].join(":")
        ))
        .unwrap();
        assert_eq!(
            check_operation_count(&hostile),
            Err(ParseError::TooManyOperations(9_000))
        );
    }

    #[test]
    fn default_ops_over_the_cap_alone_are_refused() {
        // Issue #469: defaults that alone exceed the cap would
        // make every request answer 400, so the boot refuses
        // them. The count is the expanded one: a preset counts
        // as the operations it expands to.
        let mut presets = std::collections::HashMap::new();
        presets.insert(
            "ten".to_string(),
            format!("filters:{}", ["invert"; 10].join(":")),
        );
        let at_cap = format!("filters:{}", vec!["invert"; MAX_OPERATIONS].join(":"));
        assert_eq!(
            operation_count(&parse_default_ops(&at_cap, &presets).unwrap()),
            MAX_OPERATIONS
        );
        let over = format!("{at_cap}:invert");
        assert_eq!(
            parse_default_ops(&over, &presets),
            Err(ParseError::TooManyOperations(MAX_OPERATIONS + 1))
        );
        let tens = ["preset:ten"; 10].join("/");
        assert!(parse_default_ops(&tens, &presets).is_ok());
        assert_eq!(
            parse_default_ops(&format!("{tens}/300x200"), &presets),
            Err(ParseError::TooManyOperations(MAX_OPERATIONS + 1))
        );
    }

    #[test]
    fn merged_count_drops_overridden_default_singletons() {
        // The cap counts the merged list: a default singleton
        // the URL overrides is dropped before counting, the
        // rest add to the URL's own operations.
        let presets = std::collections::HashMap::new();
        let defaults = parse_default_ops("filters:format(webp):blur(2)", &presets).unwrap();
        let ninety_eight = vec!["filters:invert"; MAX_OPERATIONS - 2].join("/");
        let (ops, _) = parse_operations(&format!("{ninety_eight}/example.com/i.jpg")).unwrap();
        assert!(check_operation_count(&merge_default_ops(&defaults, ops)).is_ok());
        let (ops, _) =
            parse_operations(&format!("{ninety_eight}/filters:invert/example.com/i.jpg")).unwrap();
        assert_eq!(
            check_operation_count(&merge_default_ops(&defaults, ops)),
            Err(ParseError::TooManyOperations(MAX_OPERATIONS + 1))
        );
        // The URL's own format drops the default one, so 98
        // inverts plus a URL format still fit: the surviving
        // default blur, 98 inverts, and the format make 100,
        // where counting the unmerged lists would make 101.
        let (ops, _) = parse_operations(&format!(
            "{ninety_eight}/filters:format(png)/example.com/i.jpg"
        ))
        .unwrap();
        assert!(check_operation_count(&merge_default_ops(&defaults, ops)).is_ok());
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
