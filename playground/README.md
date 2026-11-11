# crabcrop playground

A static page for building, signing, previewing, and debugging crabcrop URLs. No build step, no server state: serve this directory from any static host and point it at your crabcrop endpoint. Browsers do not load module scripts from `file://`, so opening `index.html` from disk does not work.

```bash
python3 -m http.server --directory playground 8910
# then open http://localhost:8910
```

## What it does

- **Build**: enter the endpoint, secret, source, and operations (quick-add chips included), optionally a preset name and an expiry. The URL is assembled and signed locally, with a live preview and the byte savings versus the original. Sizes are read by fetching both images; if the endpoint or source does not send CORS headers, the preview still works and the page says the sizes are unavailable instead of guessing.
- **Explain**: paste any crabcrop URL and get its anatomy back: the signature kind, each operation segment with a plain-language explanation, `exp:` rendered as a UTC instant with an expired/valid verdict, and the source. Malformed segments and unknown filters are highlighted as errors, mirroring the server's own parsing. Paste the secret in the builder and the debugger also verifies the signature locally. Text longer than 64 KiB is not explained or verified.

## The secret stays in the browser

Signing uses WebCrypto (`crypto.subtle`) in the page. The secret is never stored, never leaves the page, and is only ever used as the HMAC key; the only network requests are to the endpoint and sources you enter. This is stated on the page itself, above the form.

## Implementation notes

`lib.mjs` is the page's engine. The signer and URL assembly implement [`spec/url-signing.md`](../spec/url-signing.md) with the platform's WebCrypto. The explainer is the TypeScript SDK's own: `sdk/` holds the SDK's build of [`explain.ts`](../sdks/typescript/src/explain.ts) and the one module it imports, so the page stays a static directory with no build step, and the URL grammar has one JavaScript implementation. Do not edit `sdk/`. Change the SDK, then run `python3 scripts/sync_playground_sdk.py` from the repository root: it builds the SDK (run `npm ci` in `sdks/typescript` once first) and copies the result here. `scripts/test_all.sh` runs it with `--check`, which fails when `sdk/` differs from a fresh build or when a second copy of the grammar appears in the JavaScript sources.

`npm test` pins the signer to the canonical vectors in `spec/test-vectors.json`, the assembly byte-for-byte against the TypeScript SDK's `buildURL`, and the explainer to the SDK's source on every remainder of the shared corpora. It imports the SDK, so build it first (`npm run build` in `sdks/typescript`). `npm run test:integration` additionally drives the page's own `app.mjs` (against a minimal DOM stub) with a crabcrop running on `127.0.0.1:3321` under the secret `playground-secret` and any image source on `127.0.0.1:8909`: it asserts the built URL is accepted, the savings line is real, the explainer verifies the signature, and a tampered URL is caught.
