<?php
/**
 * A crabcrop URL builder for PHP, implementing spec/url-signing.md.
 *
 * This file has no WordPress dependencies: it is loaded by the plugin,
 * and directly by the test suite in tests/. Keep it that way.
 */

if (!class_exists('CrabcropUrl')) {

    final class CrabcropUrl
    {
        /**
         * The first remainder segment reserved for key derivation.
         * Every derived subkey (purge, purge-group, upload, seal,
         * cache entry) is the HMAC of a label such as
         * crabcrop/upload-key under the secret, which is the image
         * signature of that label. The server refuses an image URL
         * that begins with this segment, and no signer signs one.
         */
        const RESERVED_SEGMENT = 'crabcrop';

        /**
         * The prefixes of the segments the server reads ahead of the
         * source (issue #782): the exp: and maxage: metadata, a
         * preset: reference, and every operation that opens with a
         * name. A size has none (see size_shaped). The list is the
         * crabcrop-url crate's, which the server's tests hold to its
         * parser.
         */
        const LEADING_PREFIXES = array(
            'exp:',
            'maxage:',
            'preset:',
            'debug:',
            'frame:',
            'page:',
            'crop-rect:',
            'focalpoint:',
            'focal:',
            'crop:',
            'pad:',
            'fit-in:',
            'filters:',
        );

        /**
         * Whether $remainder is the reserved segment or starts with
         * crabcrop/.
         */
        public static function is_reserved_remainder($remainder)
        {
            $head = explode('/', $remainder, 2);
            return $head[0] === self::RESERVED_SEGMENT;
        }

        /**
         * The signature for a remainder: lowercase hex HMAC-SHA256,
         * keyed with the secret, over the remainder exactly as it
         * appears in the URL. Throws InvalidArgumentException for a
         * remainder that begins with the reserved segment: its
         * signature could be a derived subkey.
         */
        public static function sign($secret, $remainder)
        {
            self::check_not_reserved($remainder);
            return hash_hmac('sha256', $remainder, $secret);
        }

        private static function check_not_reserved($remainder)
        {
            if (self::is_reserved_remainder($remainder)) {
                throw new InvalidArgumentException(
                    'remainder begins with the reserved segment "' . self::RESERVED_SEGMENT
                        . '": it is kept for key derivation, and no image URL may begin with it'
                );
            }
        }

        /**
         * Build a full URL: {base}/{signature}/{operations}/{source}.
         *
         * $secret is the deployment's CRABCROP_SECRET, or null for the
         * `unsafe` form used by unsafe and presets-only deployments.
         * Only null means unsafe: an empty string is a real key and
         * signs. $operations is the caller's operation string verbatim
         * (it may hold several segments and preset: references, or be
         * empty). $expiry, when given, is a Unix timestamp emitted as a
         * leading exp: segment. $key_id, when given, stamps a
         * <key-id>: prefix on the signature segment for rotation
         * windows.
         *
         * Throws InvalidArgumentException for inputs that can never
         * verify: a negative or non-integer expiry, a malformed key ID,
         * or operations or a source containing ?, #, whitespace,
         * control, or non-ASCII characters (the remainder travels as a
         * raw path, so those bytes cannot be represented in it), or one
         * of " < > \ ^ ` { | }, which a browser rewrites before the
         * request leaves the page (a URL signed over them answers 403),
         * and a remainder that begins with the reserved segment, signed
         * or not (the server refuses it whatever the signature). With an
         * expiry, it also throws for operations that carry an exp: or
         * maxage: segment, and for a source whose first segment the
         * server reads as an operation, exp:, maxage: or preset: (issue
         * #782): the server keeps the last exp: of a URL, so either
         * would decide the URL in the expiry's place.
         */
        public static function build($base, $secret, $operations, $source, $expiry = null, $key_id = null)
        {
            self::check_remainder_part($operations, 'operations');
            self::check_remainder_part($source, 'source');
            if ($source === '') {
                throw new InvalidArgumentException('source must not be empty');
            }
            if ($key_id !== null && !preg_match('/^[A-Za-z0-9_-]{1,32}$/', $key_id)) {
                throw new InvalidArgumentException('key id must be 1 to 32 characters of [A-Za-z0-9_-]');
            }

            $parts = array();
            if ($expiry !== null) {
                if (!is_int($expiry) || $expiry < 0) {
                    throw new InvalidArgumentException('expiry must be a non-negative integer');
                }
                self::check_no_expiry_override($operations, $source);
                $parts[] = 'exp:' . $expiry;
            }
            if ($operations !== '') {
                $parts[] = $operations;
            }
            $parts[] = $source;
            $remainder = implode('/', $parts);
            self::check_not_reserved($remainder);

            if ($secret === null) {
                $signature = 'unsafe';
            } else {
                $signature = self::sign($secret, $remainder);
                if ($key_id !== null) {
                    $signature = $key_id . ':' . $signature;
                }
            }

            return rtrim($base, '/') . '/' . $signature . '/' . $remainder;
        }

        /**
         * With an expiry, refuse the text that would decide the URL in
         * its place (issue #782): an exp: or maxage: segment anywhere
         * among the operations, or a first source segment the server
         * reads ahead of the source. The messages are the ones every
         * other builder gives.
         */
        private static function check_no_expiry_override($operations, $source)
        {
            foreach (explode('/', $operations) as $segment) {
                if (strpos($segment, 'exp:') === 0 || strpos($segment, 'maxage:') === 0) {
                    throw new InvalidArgumentException(
                        'operations carries the segment "' . $segment . '": with an expiry, the operations'
                            . ' may carry no exp: or maxage: segment, since the server keeps the last of each'
                    );
                }
            }
            $head = explode('/', $source, 2);
            if (self::read_before_source($head[0])) {
                throw new InvalidArgumentException(
                    'source begins with the segment "' . $head[0] . '", which the server reads ahead of'
                        . ' the source: with an expiry, the source may not begin with an operation, exp:,'
                        . ' maxage: or preset: segment'
                );
            }
        }

        /**
         * Whether the server reads $segment ahead of the source: as
         * metadata, a preset reference or an operation, valid or not.
         * The first segment that is none of these starts the source.
         */
        private static function read_before_source($segment)
        {
            foreach (self::LEADING_PREFIXES as $prefix) {
                if (strpos($segment, $prefix) === 0) {
                    return true;
                }
            }
            return self::size_shaped($segment);
        }

        /**
         * The server's size shape: an x joining two parts, at least one
         * of them a dimension spelling (digits, orig, or nothing, after
         * an optional -), and no . or : in the segment. Every size the
         * server parses has it, and the server reads a segment with it
         * that does not parse as a malformed size, never as source text.
         */
        private static function size_shaped($segment)
        {
            $x = strpos($segment, 'x');
            if ($x === false || strpbrk($segment, '.:') !== false) {
                return false;
            }
            foreach (array(substr($segment, 0, $x), (string) substr($segment, $x + 1)) as $side) {
                if (preg_match('/^-?(orig|[0-9]*)$/D', $side)) {
                    return true;
                }
            }
            return false;
        }

        private static function check_remainder_part($part, $what)
        {
            if (!is_string($part)) {
                throw new InvalidArgumentException($what . ' must be a string');
            }
            // ?, #, whitespace, control characters, DEL, and every
            // non-ASCII byte: none of them can appear in a raw path
            // remainder that still verifies.
            if (preg_match('/[\x00-\x20\x7F-\xFF?#]/', $part)) {
                throw new InvalidArgumentException(
                    $what . ' contains a character that cannot appear in a signed URL'
                );
            }
            // A browser turns \ into / and percent-encodes the rest of
            // these in a path, so the server would verify different
            // bytes than were signed (issue #472). Percent-encoded,
            // they travel unchanged.
            if (preg_match('/["<>\\\\^`{|}]/', $part, $match)) {
                throw new InvalidArgumentException(sprintf(
                    '%s contains a character that cannot appear in a signed URL: %s'
                        . ' (browsers rewrite it; percent-encode it as %%%02X)',
                    $what,
                    $match[0],
                    ord($match[0])
                ));
            }
        }
    }
}
