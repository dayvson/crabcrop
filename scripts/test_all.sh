#!/usr/bin/env bash
# Run every test suite in the repository: Rust (binary and
# crate), the TypeScript SDK, the Python SDK, the MCP server,
# the framework modules (Nuxt, Next.js, Hydrogen, Astro,
# SvelteKit) and browser builds of their client code, the
# WordPress plugin, and the playground (unit and integration).
#
# The integration suites need live servers; this script starts
# and stops them:
#   - a static server for test-fixtures/ on 127.0.0.1:8909
#     (playground integration; the MCP suite starts its own on
#     8912 against the same directory, and the browser-bundles
#     suite serves its Nuxt build on a free port);
#   - crabcrop on 127.0.0.1:3321 with the playground's secret.
#
# Usage: scripts/test_all.sh [--skip-rust]
#   --skip-rust  Skip the cargo test runs (the debug binary is
#                still built: the MCP suite spawns it). CI's
#                dedicated Rust job covers them there.
set -euo pipefail
cd "$(dirname "$0")/.."

# Local test traffic must never leave through a proxy.
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"

SKIP_RUST=0
if [ "${1:-}" = "--skip-rust" ]; then
  SKIP_RUST=1
fi

step() { printf '\n=== %s ===\n' "$1"; }

step "Workflow mirrors (docs copies match the live workflows)"
# Every live workflow has a mirror, docs/<name>-workflow.yml, and the
# two are identical: the automation credential cannot push workflow
# files, so changes are made in the mirror and land through the web UI.
# ai-models.yml had none, so its unpinned cache action went unseen
# (issue #593).
for live in .github/workflows/*.yml; do
  mirror="docs/$(basename "$live" .yml)-workflow.yml"
  if [ ! -f "$mirror" ]; then
    echo "workflow mirror missing: $live has no $mirror; add one identical to the live workflow" >&2
    exit 1
  fi
  if ! cmp -s "$live" "$mirror"; then
    echo "workflow mirror drift: $mirror differs from $live; sync the mirror to the live workflow" >&2
    exit 1
  fi
done
for mirror in docs/*-workflow.yml; do
  live=".github/workflows/$(basename "$mirror" -workflow.yml).yml"
  if [ ! -f "$live" ]; then
    echo "workflow mirror without a live workflow: $mirror; land it at $live or delete it" >&2
    exit 1
  fi
done

step "Workflows (actions pinned by SHA, no download piped into a shell, hashed pip installs)"
python3 scripts/check_workflows.py
python3 scripts/test_check_workflows.py

step "Functional tier (a pull request's diff picks smoke or full)"
python3 scripts/test_functional_tier.py

step "npm dist-tag (a prerelease publishes under next, never latest)"
for pair in "0.10.0 latest" "0.10.0-rc.1 next" "1.2.3-beta.2+sha.5 next" "1.2.3+build-7 latest"; do
  set -- $pair
  got=$(bash scripts/npm_dist_tag.sh "$1")
  if [ "$got" != "$2" ]; then
    echo "npm_dist_tag.sh $1 printed '$got', expected '$2'" >&2
    exit 1
  fi
done
for bad in "v0.10.0" "crabcrop-url/0.10.0" "0.10" "0.10.0-"; do
  if bash scripts/npm_dist_tag.sh "$bad" >/dev/null 2>&1; then
    echo "npm_dist_tag.sh accepted '$bad', which is not a version" >&2
    exit 1
  fi
done
# Every npm publish command in any workflow passes the tag, whatever
# its other flags: the two known steps in release.yml (@crabcrop/url
# and @crabcrop/mcp) and any publish added later, one-off workflows
# included (issue #457). A `name:` line, a workflow's, a job's or a
# step's, is a label, not a command.
publishes=$(grep -hE 'npm publish( |$)' .github/workflows/*.yml \
  | grep -vE '^[[:space:]]*(-[[:space:]]+)?(name:|#)' || true)
untagged=$(printf '%s\n' "$publishes" | grep -v -- '--tag "\$NPM_TAG"$' || true)
if [ "$(printf '%s\n' "$publishes" | grep -c .)" -lt 2 ] || [ -n "$untagged" ]; then
  echo "every npm publish in .github/workflows must pass --tag \"\$NPM_TAG\" from scripts/npm_dist_tag.sh:" >&2
  printf '%s\n' "${untagged:-(no npm publish commands found)}" >&2
  exit 1
fi

step "Operation vectors (the generated corpus is current)"
python3 spec/generate_operation_vectors.py --check

step "Version consistency (every artifact and deploy pin agrees, and a bump moves them all)"
python3 scripts/check_versions.py
python3 scripts/test_version_tools.py

step "Lockfiles (one version of each decoder, and the fuzz lock matches the root lock)"
python3 scripts/check_locks.py
python3 scripts/test_check_locks.py

step "Rendering version (a corpus re-baseline since the last release tag raises it)"
# CI checks out the full history, so there a skip means the checkout
# lost the tags: --strict fails it instead.
python3 scripts/check_rendering_version.py ${GITHUB_ACTIONS:+--strict}
python3 scripts/test_check_rendering_version.py

step "Release scripts (the preflight checks, and creating or updating the GitHub Release)"
python3 scripts/test_release_preflight.py
python3 scripts/test_publish_github_release.py

step "Docs sync (env vars, error codes, filters and the CHANGELOG heading match the code)"
python3 scripts/check_docs.py
python3 scripts/test_check_docs.py

step "Build (debug binary: the MCP suite spawns it)"
cargo build --locked

if [ "$SKIP_RUST" -eq 0 ]; then
  step "Rust: binary tests"
  cargo test --locked
  step "Rust: crabcrop-url crate tests"
  cargo test --locked -p crabcrop-url
fi

step "Upgrade test (previous-release corpus)"
python3 tests/upgrade/run_upgrade.py --bin target/debug/crabcrop

step "TypeScript SDK"
(cd sdks/typescript && npm ci --no-audit --no-fund && npm run build && npm test)

step "Python SDK"
(cd sdks/python && python3 -m unittest discover -s tests)

step "MCP server"
(cd sdks/mcp && npm ci --no-audit --no-fund && npm test)

step "Nuxt module"
(cd frameworks/nuxt && npm ci --no-audit --no-fund && npm test)

step "Next.js module"
(cd frameworks/nextjs && npm ci --no-audit --no-fund && npm test)

step "Hydrogen module"
(cd frameworks/hydrogen && npm ci --no-audit --no-fund && npm test)

step "Astro module"
(cd frameworks/astro && npm ci --no-audit --no-fund && npm test)

step "SvelteKit module"
(cd frameworks/sveltekit && npm ci --no-audit --no-fund && npm test)

step "Browser bundles (client builds of the integrations: Vite, and a full Nuxt build)"
(cd tests/browser-bundles && npm ci --no-audit --no-fund && npm test)

step "WordPress plugin"
(cd plugins/wordpress && php tests/run.php)

step "One JavaScript grammar (the playground serves the SDK's explainer build, and no second copy exists)"
python3 scripts/sync_playground_sdk.py --check
python3 scripts/test_sync_playground_sdk.py

step "Playground (unit)"
(cd playground && npm test)

step "Playground (integration)"
FIXTURE_PID=""
SERVER_PID=""
cleanup() {
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null || true
  [ -n "$FIXTURE_PID" ] && kill "$FIXTURE_PID" 2>/dev/null || true
}
trap cleanup EXIT

(cd test-fixtures && python3 -m http.server 8909 >/dev/null 2>&1) &
FIXTURE_PID=$!
PLAYGROUND_SECRET=$(printf 'playground-%s' secret)
CRABCROP_SECRET="$PLAYGROUND_SECRET" CRABCROP_ALLOWED_SOURCES=127.0.0.1 PORT=3321 ./target/debug/crabcrop >/dev/null 2>&1 &
SERVER_PID=$!
for url in "http://127.0.0.1:8909/src00.jpg" "http://127.0.0.1:3321/health"; do
  ok=0
  for _ in $(seq 1 50); do
    if curl -sf -o /dev/null "$url"; then ok=1; break; fi
    sleep 0.2
  done
  if [ "$ok" -ne 1 ]; then
    echo "server never became ready: $url" >&2
    exit 1
  fi
done
(cd playground && npm run test:integration)

step "Fuzz smoke (parser invariants, corpus + mutations)"
# Without the fuzz crate's default avif-decode feature, so the
# smoke needs no dav1d; the nightly fuzz workflow runs it with.
cargo test --locked --manifest-path fuzz/Cargo.toml --no-default-features --lib --test smoke

step "All suites passed"
