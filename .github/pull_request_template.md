## Summary

<!-- What does this change, and why? -->

## Related issue

<!-- e.g. Closes #123 -->

## Checklist

- [ ] `cargo fmt --all` and `cargo clippy --all-targets -- -D warnings` are clean
- [ ] Tests pass (`cargo test`, and `scripts/test_all.sh` if the change spans languages)
- [ ] Added or updated tests for the change
- [ ] Updated the README or other docs if behavior changed
- [ ] Added a `CHANGELOG.md` entry under `[Unreleased]` if the change is user-facing
- [ ] If the URL grammar or signing changed: updated `spec/` and kept the SDKs and playground in sync
