#!/usr/bin/env bash
# Run every test suite in the repository: Rust (binary and
# crate), the TypeScript SDK, the Python SDK, the MCP server,
# the framework modules (Nuxt, Next.js, Hydrogen), and the
# playground (unit and integration).
#
# The integration suites need live servers; this script starts
# and stops them:
#   - a static server for test-fixtures/ on 127.0.0.1:8909
#     (playground integration; the MCP suite starts its own on
#     8912 against the same directory);
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

step "Build (debug binary: the MCP suite spawns it)"
cargo build

if [ "$SKIP_RUST" -eq 0 ]; then
  step "Rust: binary tests"
  cargo test
  step "Rust: crabcrop-url crate tests"
  cargo test -p crabcrop-url
fi

step "TypeScript SDK"
(cd sdks/typescript && { [ -d node_modules ] || npm install --no-audit --no-fund; } && npm run build && npm test)

step "Python SDK"
(cd sdks/python && python3 -m unittest discover -s tests)

step "MCP server"
(cd sdks/mcp && { [ -d node_modules ] || npm install --no-audit --no-fund; } && npm test)

step "Nuxt module"
(cd frameworks/nuxt && { [ -d node_modules ] || npm install --no-audit --no-fund; } && npm test)

step "Next.js module"
(cd frameworks/nextjs && { [ -d node_modules ] || npm install --no-audit --no-fund; } && npm test)

step "Hydrogen module"
(cd frameworks/hydrogen && { [ -d node_modules ] || npm install --no-audit --no-fund; } && npm test)

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
CRABCROP_SECRET="$PLAYGROUND_SECRET" PORT=3321 ./target/debug/crabcrop >/dev/null 2>&1 &
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
cargo test --manifest-path fuzz/Cargo.toml --lib --test smoke

step "All suites passed"
