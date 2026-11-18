In the `crabcrop-url` crate, `BuildError` and `SealError` are
`#[non_exhaustive]`, `SealError` gains `RandomnessFailed` and `EncryptFailed`
for the failures `seal` used to report as `OpenFailed`, `build_url_with_key_id`
takes the secret and its key ID as `(&[u8], &str)` instead of an `Option` whose
`None` was always an `InvalidKeyId("")` error, and `verify` refuses a tag that
is not 64 lowercase hex characters, as the server's parser does, where it
accepted uppercase hex (issue #712). A `match` on either error type needs a
wildcard arm, and a call to `build_url_with_key_id` drops its `Some(...)`.
`unseal`'s documentation no longer claims that `verify` reads keyed segments
the same way: `verify` takes the bare tag. This changes the crate's public
API, so it ships in a minor release (0.14.0), not a patch release.
