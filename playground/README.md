# crabcrop playground

A static page for building, signing, previewing, and debugging crabcrop URLs. No build step, no server state: serve this directory from any static host (or open `index.html` from disk) and point it at your crabcrop endpoint.

```bash
python3 -m http.server --directory playground 8910
# then open http://localhost:8910
```

## What it does

- **Build**: enter the endpoint, secret, source, and operations (quick-add chips included), optionally a preset name and an expiry. The URL is assembled and signed locally, with a live preview and the byte savings versus the original. Sizes are read by fetching both images; if the endpoint or source does not send CORS headers, the preview still works and the page says the sizes are unavailable instead of guessing.
- **Explain**: paste any crabcrop URL and get its anatomy back: the signature kind, each operation segment with a plain-language explanation, `exp:` rendered as a UTC instant with an expired/valid verdict, and the source. Malformed segments and unknown filters are highlighted as errors, mirroring the server's own parsing. Paste the secret in the builder and the debugger also verifies the signature locally.

## The secret stays in the browser

Signing uses WebCrypto (`crypto.subtle`) in the page. The secret is never stored, never leaves the page, and is only ever used as the HMAC key; the only network requests are to the endpoint and sources you enter. This is stated on the page itself, above the form.

## Implementation notes

`lib.mjs` is the whole engine: the signer and URL assembly implement [`spec/url-signing.md`](../spec/url-signing.md) (the SDK's `sign` is Node-only, so the browser uses the platform WebCrypto implementation of the same scheme), and the explainer mirrors the server's segment dispatch. `npm test` pins the signer to the canonical vectors in `spec/test-vectors.json` and the assembly byte-for-byte against the TypeScript SDK's `buildURL`. `npm run test:integration` additionally drives the page's own `app.mjs` (against a minimal DOM stub) with a crabcrop running on `127.0.0.1:3321` under the secret `playground-secret` and any image source on `127.0.0.1:8909`: it asserts the built URL is accepted, the savings line is real, the explainer verifies the signature, and a tampered URL is caught.
