# Releasing crabcrop

One version number covers every artifact: the server binaries and container images, the `crabcrop-url` crate (crates.io), `@crabcrop/url` (npm), `crabcrop_url` (PyPI), the MCP server package, the framework loaders, and the Helm chart. Releases are driven by git tags (`vX.Y.Z`).

## The pipeline

Pushing a tag like `v0.2.0` runs `.github/workflows/release.yml`. Its first job, **preflight** (`scripts/release_preflight.py`), stops the run before anything is built unless both of these hold:

- The `release` environment has a **Required reviewers** rule (Settings, Environments, `release`, Deployment protection rules). That rule is what makes a human approve the registry jobs in step 3; without it they would publish unattended.
- The tagged commit's latest push run of `rust.yml` on main passed. A run that is still going does not count.

The job's error says which check failed. Nothing was built or published, so restore the rule, or wait for main CI to pass, and re-run the failed jobs. If main CI failed, fix main and tag a green commit instead. Once the preflight passes:

1. **cargo-dist** builds the server for Linux and macOS on both architectures, plus the shell installer and SHA-256 checksums, and attaches them to a GitHub Release whose notes come from CHANGELOG.md. The `host` job creates the release with `scripts/publish_github_release.sh`. If a draft for the tag already exists, made by hand or left by an earlier attempt, the script uploads the assets to it (replacing any of the same name), sets the title, notes, and prerelease flag, and publishes it, so the job passes on its first attempt either way. A release that is already published stops the job instead (see **Publishing** below).
2. **Container images**: the base image packages the step 1 binaries (`Dockerfile.release`, staged checksum-verified from the release tarballs by `scripts/stage_image_binaries.sh`, so an image ships the exact binary its release published), and the AI variant layers on it from `Dockerfile.ai`. Both are pushed to GHCR as `ghcr.io/dayvson/crabcrop` and `ghcr.io/dayvson/crabcrop-ai`, tagged with the version, and with `latest` only for a plain release: a prerelease such as `v0.10.0-rc.1` pushes its version tag alone. The repo `Dockerfile` remains the from-source build for operators building their own.
3. **Registry publishing** (`cargo publish` for the crate, `npm publish` for the SDK, `twine upload` for the Python package) runs as separate jobs gated on the `release` GitHub Environment, whose Required reviewers rule makes a human approve them in the Actions UI after inspecting the built artifacts (the preflight fails the run when that rule is gone). The jobs need the `CARGO_REGISTRY_TOKEN`, `NPM_TOKEN`, and `PYPI_TOKEN` repository secrets. The two npm packages (`@crabcrop/url` and `@crabcrop/mcp`) publish with an explicit dist-tag from `scripts/npm_dist_tag.sh`: `latest` for a plain release, and `next` for a prerelease, so a release candidate never becomes what `npm install @crabcrop/url` or `npx -y @crabcrop/mcp` resolves (npm itself would put every publish on `latest`). Install a prerelease by its tag or version (`npm install @crabcrop/url@next`), and after publishing one, check that `npm dist-tag ls @crabcrop/url` and `npm dist-tag ls @crabcrop/mcp` show it under `next` with `latest` unchanged. crates.io and PyPI need no equivalent: their clients skip prereleases unless asked for one.
4. **The Homebrew formula** is generated from the release checksums and uploaded to the release as `crabcrop.rb`:

   ```bash
   gh release download v0.2.0 --pattern sha256.sum
   python3 scripts/release_formula.py 0.2.0 sha256.sum > crabcrop.rb
   gh release upload v0.2.0 crabcrop.rb
   ```

   Moving it into a `homebrew-tap` repository is a separate, later step.

**The release gate**: once the preflight passes, the same run calls `.github/workflows/ai-models.yml`, which downloads the pinned ONNX Runtime and models and runs the real-model AI tests (`cargo test -- --ignored`) alongside the builds. The `host` job, which creates the GitHub Release, waits for it, and the image and registry jobs wait for `host`, so a tag whose AI code fails against the real models publishes nothing. Nothing was published, so delete the tag (`git push origin :refs/tags/vX.Y.Z`), land the fix on main, and tag the fixed commit with the same version; for a transient download failure, re-run the failed jobs, which re-runs the gate and everything waiting on it.

**Publishing**: the release workflow is the only way to publish a release. Never create or publish the GitHub Release by hand. Publishing one in the web UI also creates the tag, so the page goes public before the gates have run, with none of the assets: v0.10.0 was published that way, and for 37 minutes `releases/latest/download/crabcrop-installer.sh` returned 404 (issue #552). A draft made by hand is fine, since `host` publishes it. If `host` fails with "Release vX.Y.Z is already published", turn the release back into a draft (`gh release edit vX.Y.Z --draft=true`) and re-run the failed jobs. If an earlier attempt of the workflow published it, `host` already passed, so re-run only the failed jobs, not all of them: a full re-run would replace assets people may already have downloaded. To keep a published release as it is, set the repository variable `ADOPT_PUBLISHED_RELEASE` to `1` (Settings, Secrets and variables, Actions, Variables) for one re-run, and delete it afterwards. The job then adopts the release with a warning.

The workflow file is generated by cargo-dist (`cargo dist generate`) plus the image and publishing jobs maintained in this repo. Because the automation credential cannot push workflow files, the complete workflow lives at `docs/release-workflow.yml`; land it at `.github/workflows/release.yml` through the GitHub web UI (the same route as the rust.yml gates in #36) before the first tag.

## Cutting a release

1. Read the release's CHANGELOG section before you write it. It holds the entries under `## [Unreleased]` in CHANGELOG.md plus every fragment in `changelog.d/` (CONTRIBUTING.md, Changelog). Print it as the release will write it, and fix any wording in the fragment or the line itself:

   ```bash
   python3 scripts/assemble_changelog.py --check
   ```

   The bump in step 2 writes it: `scripts/bump_version.py` runs `scripts/assemble_changelog.py X.Y.Z`, which moves the Unreleased entries and the fragments into a new `## [X.Y.Z] - YYYY-MM-DD` section dated today (`--date YYYY-MM-DD` sets another day), each section's hand-written entries first and then its fragments in issue order, leaves an empty `## [Unreleased]` heading above it for the next change, and deletes the fragments. The bump does this, rather than a command of its own, so that the one command that sets the version also writes the section cargo-dist takes the release notes from: a release cannot carry its version without its notes, or leave fragments behind for the next release. A malformed fragment stops the bump before it writes anything. Edit the assembled section by hand afterwards if it needs it. `scripts/check_docs.py` (part of `scripts/test_all.sh`) fails when the CHANGELOG does not open with that heading, or when a fragment is malformed. The release commit touches `sdks/`, `frameworks/`, and `plugins/` and deletes fragments without adding an entry, so if it goes through a pull request, that pull request needs the `no-changelog` label.
2. Bump every version site and verify they agree:

   ```bash
   python3 scripts/bump_version.py X.Y.Z
   python3 scripts/check_versions.py
   export PATH="$HOME/.cargo/bin:$PATH"
   cargo test --locked && (cd sdks/typescript && npm test) && (cd sdks/python && python3 -m unittest discover -s tests)
   ```

   The bump writes the CHANGELOG section (step 1) and edits every site listed in `scripts/version_sites.py`, then runs the check over the same list; `--dry-run` lists what it would change. The sites are the crates, the npm packages and the playground, every `package-lock.json` (its own version fields and, where it links the TypeScript SDK, the linked `@crabcrop/url` entry), the `crabcrop` and `crabcrop-url` entries in `Cargo.lock` and `fuzz/Cargo.lock`, the Python package, the MCP server info and registry manifest, the WordPress plugin header, the Helm chart and both values files, the kustomize images, and the image tags in the README and the platform guides, the Helm README's `--set image.tag=` line included. A minor bump also moves the server's `crabcrop-url` requirement in `Cargo.toml` (`0.9` to `0.10`). The lockfiles are part of the bump, so no `cargo update` follows; the `--locked` test run proves the Cargo lockfiles agree. A site that is not where the list says, such as a third `tag:` line in the Helm values, stops the bump before it writes anything.

   Afterwards `git grep -wF <previous version>` finds only CHANGELOG.md, third-party lockfile entries, prereleases of that version (`-rc.1`, which are other versions), the rendering version's history above `RENDERING_VERSION` in `src/main.rs` (step 4), and the test inputs and examples listed in `scripts/test_version_tools.py`. Any other hit is a site the list misses: add it to `scripts/version_sites.py` instead of editing it by hand. `scripts/test_version_tools.py`, which the gate runs, bumps a copy of the tree and fails on such a leftover.

3. Run the upgrade test against the release candidate build before tagging:

   ```bash
   cargo build --release --locked
   python3 tests/upgrade/run_upgrade.py --bin target/release/crabcrop
   ```

   The corpus in `tests/upgrade/corpus.json` was minted and rendered under the previous release, so a failure means the candidate broke a URL that is already in the wild. Fix the incompatibility, or, when the byte change is deliberate, re-baseline the entry with the reason recorded in `corpus.json` and the CHANGELOG (see `tests/upgrade/README.md`) and raise the rendering version (step 4). Step 2 has already written `## [X.Y.Z]`, so record the reason there by hand: a fragment added now would wait for the next release. Do not tag over a failing upgrade test.

4. Check the rendering version. `RENDERING_VERSION` in `src/main.rs` is part of the configuration fingerprint every cache key carries, and the only part that moves with the code rather than the configuration: a release that changes rendered output for existing URLs must raise it, or upgraded servers keep serving the previous release's renders from the disk and shared tiers, and replicas on different versions serve different bytes for the same URL during a rollout. A corpus entry re-baselined since the last tag is that kind of change, and the gate refuses it without a bump:

   ```bash
   python3 scripts/check_rendering_version.py
   ```

   The check compares the working tree with the last release tag, so run it from a full clone. `scripts/test_all.sh` runs it too, and so does CI's suites job, which checks out the full history and fails if the check cannot reach a release tag. Raise the version by one, note the reason next to the constant as `N: the first release after X.Y.Z, which changes output ...` (X.Y.Z is the release before the change; the leftover check in step 2 skips a version only in that form, in that comment), and say in the release's CHANGELOG section that the release starts a new cache epoch: a note that opens "This release starts a new cache epoch" and leads Changed. The pull request that raised the version usually wrote it, as a line or as a `changelog.d/<issue>.changed.md` fragment, which the assembly puts first under Changed. When you raise it during the release, step 2 has already written `## [X.Y.Z]`: write the note there by hand, first under Changed, since a fragment added now would wait for the next release. Raise it as well for a deliberate output change the corpus does not cover; the check only sees the corpus.

5. Commit ("Release vX.Y.Z"), push, and tag: `git tag vX.Y.Z` on the release commit (the local master and the remote main carry the same tree, as usual). Create the tag with git, never by publishing a release in the GitHub UI (see **Publishing** above).
6. Land `docs/release-workflow.yml` at `.github/workflows/release.yml` if it is not already there. Once the release commit's `rust.yml` run on main has passed, push the tag and watch the run; a failed preflight names its reason (see the pipeline above). Approve the `release` environment jobs when the artifacts look right.
7. Smoke test the published artifacts in a clean environment: `cargo add crabcrop-url`, `npm install @crabcrop/url`, `pip install crabcrop-url`, `docker pull ghcr.io/dayvson/crabcrop:X.Y.Z`, and the shell installer from the GitHub Release.

## Version policy

Semantic Versioning, applied to the public contract: the URL grammar and signing scheme in `spec/url-signing.md`, the configuration surface, and the client libraries' APIs. A breaking change to the URL grammar would arrive as the spec describes (a new leading segment), never as a silent reinterpretation of existing URLs.
