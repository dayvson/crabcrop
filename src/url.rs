//! Parsing of crabcrop's image URLs.
//!
//! ```text
//! /<signature>/<operations...>/<image-url>
//! ```
//!
//! This module starts with the smallest meaningful piece: the [`Signature`],
//! the first path segment. Everything else (crop, size, filters, source URL)
//! builds on top of [`parse_signature`], which splits the signature off and
//! hands back the remainder of the path.

/// How the request proves it was minted by someone holding the server secret.
///
/// Either an HMAC-SHA1 (hex) of the rest of the URL, or the literal `unsafe`
/// when the operator allows unsigned URLs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Signature {
    /// Hex-encoded HMAC-SHA1 of the remainder of the path.
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
}
