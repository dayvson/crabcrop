# crabcrop playground

A static page for building, signing, previewing, and debugging crabcrop URLs. No build step, no server state: serve this directory from a static host and point it at your crabcrop endpoint. The host must not add scripts to the page; see [Content-Security-Policy](#content-security-policy). Browsers do not load module scripts from `file://`, so opening `index.html` from disk does not work.

```bash
python3 -m http.server --directory playground 8910
# then open http://localhost:8910
```

## What it does

- **Build**: enter the endpoint, secret, source, and operations (quick-add chips included), optionally a preset name and an expiry. The URL is assembled and signed locally, with a live preview and the byte savings versus the original. Sizes are read by fetching both images; if the endpoint or source does not send CORS headers, the preview still works and the page says the sizes are unavailable instead of guessing. On load the page shows the preview of the prefilled example but does not measure it: the size check waits until you change a field or click a chip.
- **Explain**: paste any crabcrop URL and get its anatomy back: the signature kind, each operation segment with a plain-language explanation, `exp:` rendered as a UTC instant with an expired/valid verdict, and the source. Malformed segments and unknown filters are highlighted as errors, mirroring the server's own parsing. Paste the secret in the builder and the debugger also verifies the signature locally.

## The secret stays in the browser

Signing uses WebCrypto (`crypto.subtle`) in the page. The secret is never stored, never leaves the page, and is only ever used as the HMAC key. Apart from its own files, the page sends requests only to the endpoint and the source in the form. On load it requests only the preview, from the endpoint. The size check also fetches the source, and it waits for your first edit, so a visit contacts no source host until you change a field. This is stated on the page itself, above the form.

## Content-Security-Policy

`index.html` carries this policy in a `<meta>` tag, the first tag after `<meta charset>`:

```
script-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'
```

Only scripts from the page's own origin run: `app.mjs` and the modules it imports from this directory. An inline script, or a script from another origin, does not run, so it cannot read the secret field. A meta policy covers only the markup after it, so keep it first in `<head>`. `img-src`, `connect-src` and `style-src` stay open, because the endpoint and the sources are whatever you type, and the page uses inline styles. So the policy cannot stop a script that already runs from sending the secret out; its job is to keep other scripts from running at all. It does not help if someone edits the page's own files on the host.

Scripts that a host adds to the page are blocked on purpose when they are inline or come from another origin, such as the analytics tags of a site template. A CDN feature can also serve its script from a path on the page's own origin, and `'self'` lets that run. So turn off the features that inject scripts for the playground's host, such as Cloudflare Rocket Loader, Zaraz and Web Analytics, instead of relying on the policy. Where the host can set response headers, send the same policy as a `Content-Security-Policy` header too.

`npm test` fails when the policy is missing, loosened or moved down the page, and when the page gains an inline script, an inline event handler, a `javascript:` URL, a script from another origin, or a use of `eval`.

## Implementation notes

`lib.mjs` is the page's engine. The signer and URL assembly implement [`spec/url-signing.md`](../spec/url-signing.md) with the platform's WebCrypto. The explainer is the TypeScript SDK's own: `sdk/` holds the SDK's build of [`explain.ts`](../sdks/typescript/src/explain.ts) and the one module it imports, so the page stays a static directory with no build step, and the URL grammar has one JavaScript implementation. Do not edit `sdk/`. Change the SDK, then run `python3 scripts/sync_playground_sdk.py` from the repository root: it builds the SDK (run `npm ci` in `sdks/typescript` once first) and copies the result here. `scripts/test_all.sh` runs it with `--check`, which fails when `sdk/` differs from a fresh build or when a second copy of the grammar appears in the JavaScript sources.

`npm test` pins the signer to the canonical vectors in `spec/test-vectors.json`, the assembly byte-for-byte against the TypeScript SDK's `buildURL`, and the explainer to the SDK's source on every remainder of the shared corpora. It imports the SDK, so build it first (`npm run build` in `sdks/typescript`). `npm test` also drives the page's own `app.mjs` against a minimal DOM stub (`test/dom-stub.mjs`) and a recording `fetch`, and checks that a visit fetches nothing until the first edit. `npm run test:integration` additionally drives `app.mjs` with a crabcrop running on `127.0.0.1:3321` under the secret `playground-secret` and any image source on `127.0.0.1:8909`: it asserts the built URL is accepted, the savings line is real, the explainer verifies the signature, and a tampered URL is caught.
