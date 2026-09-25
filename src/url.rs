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
        let (sig, rest) =
            parse_signature("/a1b2c3d4/300x200/example.com/img.jpg").unwrap();
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
            assert_eq!(parse_signature(path), Err(ParseError::EmptyPath), "path: {path:?}");
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
        assert_eq!(parse_signature("//300x200/x.jpg"), Err(ParseError::EmptyPath));
    }

    #[test]
    fn keeps_image_url_with_slashes_intact() {
        let (sig, rest) =
            parse_signature("/unsafe/filters:grayscale()/https://example.com/a/b/c.jpg")
                .unwrap();
        assert_eq!(sig, Signature::Unsafe);
        assert_eq!(rest, "filters:grayscale()/https://example.com/a/b/c.jpg");
    }
}
