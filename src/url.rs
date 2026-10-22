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

use hmac::{Hmac, Mac};
use sha2::Sha256;

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
    /// A size segment did not match `<width>x<height>`. Carries the
    /// offending segment.
    InvalidSize(String),
    /// A `crop:...` segment did not match `crop:<WxH>[:<align>]`.
    /// Carries the offending segment.
    InvalidCrop(String),
    /// A `filters:...` segment did not parse. Carries the offending
    /// segment.
    InvalidFilters(String),
    /// No source URL followed the operations.
    MissingSourceUrl,
}

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
}

/// Output image format requested by a `format(...)` filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Jpeg,
    Png,
    WebP,
    Jxl,
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
    /// Encoding quality, 0-100.
    Quality(u8),
    /// Output container.
    Format(ImageFormat),
}

/// One operation from the request path, in path order.
#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    /// Resize to a size segment (`300x200`, ...).
    Size(ImageSize),
    /// Crop to a size with an alignment, before resize.
    Crop { size: ImageSize, align: Align },
    /// Filters applied in order after resize.
    Filters(Vec<Filter>),
}

/// A fully parsed request path: operations plus the source image URL.
#[derive(Debug, Clone, PartialEq)]
pub struct Request<'a> {
    pub operations: Vec<Operation>,
    pub source_url: &'a str,
}

fn parse_dimension(part: &str) -> Result<(Dimension, bool), ()> {
    let (flip, digits) = match part.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, part),
    };
    let dimension = match digits {
        "" | "0" => Dimension::Proportional,
        "orig" => Dimension::Original,
        d => d.parse::<u32>().map(Dimension::Pixels).map_err(|_| ())?,
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
    } else {
        Signature::Hmac(first.to_string())
    };
    Ok((signature, rest))
}

/// Parse one path segment as an operation.
///
/// Returns `None` when the segment is not an operation at all, which
/// means it starts the source URL. A segment that claims to be an
/// operation (`crop:` / `filters:` prefix) but is malformed is an error.
fn parse_operation(seg: &str) -> Option<Result<Operation, ParseError>> {
    if let Some(rest) = seg.strip_prefix("crop:") {
        return Some(parse_crop(rest, seg));
    }
    if let Some(rest) = seg.strip_prefix("filters:") {
        return Some(parse_filters(rest, seg));
    }
    match parse_size(seg) {
        Ok(size) => Some(Ok(Operation::Size(size))),
        Err(_) => None,
    }
}

fn parse_align(s: &str) -> Option<Align> {
    match s {
        "center" => Some(Align::Center),
        "top" => Some(Align::Top),
        "bottom" => Some(Align::Bottom),
        "left" => Some(Align::Left),
        "right" => Some(Align::Right),
        _ => None,
    }
}

/// Parse the dimensions (and optional alignment) after `crop:`.
fn parse_crop(rest: &str, seg: &str) -> Result<Operation, ParseError> {
    let invalid = || ParseError::InvalidCrop(seg.to_string());
    let (dims, align) = match rest.split_once(':') {
        Some((d, a)) => (d, parse_align(a).ok_or_else(invalid)?),
        None => (rest, Align::Center),
    };
    let size = parse_size(dims).map_err(|_| invalid())?;
    Ok(Operation::Crop { size, align })
}

fn parse_format(s: &str) -> Option<ImageFormat> {
    match s {
        "jpeg" | "jpg" => Some(ImageFormat::Jpeg),
        "png" => Some(ImageFormat::Png),
        "webp" => Some(ImageFormat::WebP),
        "jxl" => Some(ImageFormat::Jxl),
        _ => None,
    }
}

/// Parse the filter list after `filters:`.
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
                let q: u8 = p.parse().map_err(|_| invalid())?;
                if q > 100 {
                    return Err(invalid());
                }
                Filter::Quality(q)
            }
            ("format", Some(p)) => Filter::Format(parse_format(p).ok_or_else(invalid)?),
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
    let mut operations = Vec::new();
    let mut rest = input;
    loop {
        let (seg, remaining) = match rest.split_once('/') {
            Some((s, r)) => (s, r),
            None => (rest, ""),
        };
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
                return Ok((operations, rest));
            }
        }
    }
}

/// Parse a full request path into its signature and [`Request`].
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
        Signature::Hmac(hex_digest) => {
            let Ok(expected) = hex::decode(hex_digest) else {
                return false;
            };
            let Ok(mut mac) = Hmac::<Sha256>::new_from_slice(secret) else {
                return false;
            };
            mac.update(remainder.as_bytes());
            mac.verify_slice(&expected).is_ok()
        }
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
    fn signature_is_case_sensitive_but_unsafe_is_lowercase_only() {
        // "Unsafe" is not the magic word; it becomes an HMAC-looking signature.
        let (sig, _) = parse_signature("/Unsafe/300x200/example.com/img.jpg").unwrap();
        assert_eq!(sig, Signature::Hmac("Unsafe".to_string()));
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
                Filter::Quality(80),
                Filter::Format(ImageFormat::WebP),
            ])]
        );
    }

    #[test]
    fn rejects_bad_filters() {
        for seg in [
            "filters:",
            "filters:blur()",
            "filters:quality(200)",
            "filters:quality()",
            "filters:grayscale(1)",
            "filters:format(bmp)",
            "filters:quality(80",
        ] {
            assert_eq!(
                parse_operations(&format!("{seg}/example.com/img.jpg")),
                Err(ParseError::InvalidFilters(seg.to_string())),
                "segment: {seg:?}"
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
}
