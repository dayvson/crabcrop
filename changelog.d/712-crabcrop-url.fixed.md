The `crabcrop-url` crate's tests run from a copy of `spec/test-vectors.json`
that the package ships, so `cargo test` passes on the published crate, which
has no `spec/` directory and failed to compile its tests (issue #712). In the
repository a test holds the copy to the spec's file.
