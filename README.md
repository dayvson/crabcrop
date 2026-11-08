# crabcrop

<p align="center">
  <img src="assets/crabcrop-mascot.webp" alt="The crabcrop mascot: a crochet crab waving hello and holding a photo" width="260">
</p>

<p align="center">
  <a href="https://crabcrop.org/">Website</a> · <a href="https://crabcrop.org/docs">Documentation</a> · <a href="https://play.crabcrop.org">Playground</a>
</p>

**Blazing fast, on-demand images.**

crabcrop is a Rust-built, lightning-fast on-demand image API supporting all major formats in and out, with over 50 filters and AI features like background removal. It resizes, crops, converts, and filters your images straight from a signed URL. One binary, no pre-generated thumbnails, no build step: the URL is the API.

Everything an image pipeline does, in one signed URL: resize, crop, adjust, shape, overlay, and encode, chained in the order written, plus background removal, studio mode, smart face-safe crops, redaction, and content provenance. Ask for JSON instead of an image and the meta endpoint describes the picture: dimensions, detections, OCR text, and more.

crabcrop is for sites and apps that serve the same images at many sizes and want signed, cacheable image URLs in front of an origin image store.

## Install

**Shell installer** (macOS and Linux):

```bash
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/dayvson/crabcrop/releases/latest/download/crabcrop-installer.sh | sh
```

**Docker:** the base image, and the AI variant with background removal and face detection on board:

```bash
docker pull ghcr.io/dayvson/crabcrop:0.10.0
docker pull ghcr.io/dayvson/crabcrop-ai:0.10.0
docker run -e CRABCROP_SECRET=your-secret-change-me -p 3000:3000 ghcr.io/dayvson/crabcrop:0.10.0
```

A secret of at least 16 bytes is the only required configuration; the server will not start without one. By default the loader fetches public sources and refuses sources that resolve to private addresses, and an allowlist names any exceptions. See [Configuration](docs/configuration.md).

**Release binaries:** prebuilt tarballs for macOS and Linux, on x86_64 and arm64, are on the [releases page](https://github.com/dayvson/crabcrop/releases).

**crates.io:** `cargo install crabcrop` builds the current release from source; the prerequisites are in [Deployment](docs/deployment.md).

## Quickstart

**1. Start the server.** For local development, also allow unsigned URLs:

```bash
CRABCROP_SECRET=local-dev-secret CRABCROP_ALLOW_UNSAFE=1 cargo run --release
# crabcrop listening on port 3000
```

**2. Request an image.** This resizes a source photo to 300x200:

```bash
curl "http://localhost:3000/unsafe/300x200/https://example.com/photo.jpg" --output photo.jpg
```

Replace `example.com/photo.jpg` with the URL of an image you control. You get back a 300x200 JPEG.

**3. Sign your URLs in production.** Drop `CRABCROP_ALLOW_UNSAFE` and put an HMAC signature in the first path segment:

```bash
curl "http://localhost:3000/9e24ae8bf61c0d40cb6f1ca9f9e1d3e771f6c8bc8b8ab7270e6d15458d4c14ab/300x200/https://example.com/photo.jpg" --output photo.jpg
```

That signature is HMAC-SHA256 of `300x200/https://example.com/photo.jpg` under the secret from step 1. The SDKs and the `crabcrop sign` command compute it for you; the full signing rules are in [URL grammar and signing](docs/url-grammar.md).

## How requests work

Every image request has three parts:

```text
GET /<signature>/<operations>/<source-url>
```

The signature is an HMAC-SHA256 (hex) of everything after it, or the literal `unsafe` on a server that allows it. Operations are zero or more segments, applied in path order. The source is a full URL, a bucket or file locator, or a short `root:` reference the server expands against origins it keeps to itself.

A signed URL names a transformation of a source locator, not the source's contents: the cache key is built from the operations and the locator, so if the image at a source changes, a purged entry re-renders the new image under the same URL. Because successful responses carry a one year immutable cache lifetime, a source that must stay fixed under its URL should be versioned or immutable itself (a content hash in the source path is the usual way). A render is cached under the canonical form of its operations, so equivalent spellings share one entry. A `preset:` segment names an operation string held on the server, so templates stay short and the transformation menu stays operator controlled.

## One URL, decoded

```text
https://img.example.com/<signature>/crop:1200x630:smart/filters:format(webp):quality(80)/https://example.com/photo.jpg
```

Smart-crops the photo to the 1200x630 social card size, keeping the subject and any detected faces whole, then encodes it as WebP at quality 80. Sign it with the TypeScript or Python SDK, the crabcrop-url Rust crate, the CLI, or plain curl: same inputs, same URL.

Send `Accept: application/json` with the same URL and the answer is the render's meta document instead of its bytes: dimensions, format, byte size, and provenance, with richer annotations as opt-ins. Meta mode is off by default, because every meta request renders afresh; the operator turns it on with `CRABCROP_META_MODE=1`.

## Features

- **All major formats, in and out.** JPEG, PNG, WebP, GIF, AVIF, and JPEG XL, in and out. HEIC, HEIF, TIFF, BMP, and ICO decode only; SVG decodes, sanitized and rasterized. Name no format and the Accept header decides: AVIF first, then JPEG XL, then WebP. An explicit `format(...)` always wins.
- **More than photos.** The source can be one frame of a video, one page of a PDF, an object in S3 or GCS, a local file, or a short `root:` reference. Originals can also arrive through the signed upload endpoint, off by default.
- **51 filters, one chain.** Resize, crop, fit-in, and pad are their own path segments. The 51 filters chain inside one `filters:` segment, everything running in the order written.
- **Smart crop, face safe.** The crop window finds the subject, and keeps detected faces whole instead of clipping them at the edge.
- **Studio mode.** The listing shot in one filter: cut out, backdrop, contact shadow, margin. Or `studio(scene)` reframes the photo on its subject and corrects the light and color.
- **Background removal.** Your ONNX model, in process. The subject stays, the background becomes real alpha.
- **AI upscale.** Super-resolution from a model you supply rebuilds detail instead of stretching pixels, and it never falls back to a plain resize.
- **Object removal.** Name a rectangle and an inpainting model fills it in from the surrounding pixels. Everything outside the rectangle comes back bit-identical.
- **Face detection and redaction.** Blur or pixelate faces, or every detected object of a class, following the object's silhouette when a segmentation model is configured. Redaction never guesses: a detection failure fails the request. It replaces the pixels it covers, so no original color survives under transparency.
- **Metadata that describes the picture.** JSON meta for every render: dimensions, format, byte size, provenance. Richer answers are opt-ins with their own model sets: CLIP tags and alt text, detections, moderation verdicts, embeddings, an aesthetic score.
- **Reads the text in the picture.** Opt a meta request in and the answer reads the picture: the text lines in reading order, each with its box and a confidence score. The annotation never touches the image bytes.
- **Provenance, checked and signed.** Meta reports the C2PA Content Credentials a source carries: manifest present, whether it validates, who signed it. With your own certificate, outputs are signed as served, the source recorded as an ingredient, so the chain survives the transform.
- **Sealed URLs.** For links that stay public but unreadable: the whole recipe, operations and source, encrypted into one opaque token in the signature slot. Only the server opens it. The readable form stays the default.
- **Presets and purge.** Server-side preset names keep templates short. Purge one result, or a whole preset group, with signatures from their own subkey.
- **URLs that explain themselves.** Put `explain/` in front of any signed URL and the server reports on it instead of serving the image.
- **Sources on your terms.** The loader fetches only what your policy allows: host allowlists, private address blocking, the source policy re-checked on every redirect hop, and fixed fetch and decode bounds, so a small file cannot expand into gigabytes.

## Documentation

This README is the front door. The full reference lives on the website and in this repo's docs folder:

- [Full documentation](https://crabcrop.org/docs): every operation with examples and interactive demos, section by section.
- [Playground](https://play.crabcrop.org): build, sign, preview, and debug URLs in the browser. The secret never leaves the page.
- The reference, in this repo:
  - [URL grammar and signing](docs/url-grammar.md): request anatomy, signing, sealed and expiring URLs, source images and source roots.
  - [Operations](docs/operations.md): resize, fit-in, pad, the crop family, focal forms, debug view, presets.
  - [Filters](docs/filters.md): the pixel filters, encoding directives, watermark, overlay, text, the AI filters, and studio mode.
  - [Media sources](docs/media.md): format negotiation, video output, video frames, PDF pages.
  - [Meta and provenance](docs/meta.md): meta responses and C2PA verification and signing.
  - [Responses and errors](docs/responses.md): response headers, the cache tiers, color management, the fallback image, and the error table.
  - [Endpoints and CLI](docs/endpoints.md): health, metrics, purge, uploads, self-explaining URLs, cache groups, and the command line verbs.
  - [Configuration](docs/configuration.md): every environment variable, server-wide default operations, rotating the secret.
  - [AI features](docs/ai-features.md): the model-backed features and declared detectors.
  - [Clients and SDKs](docs/clients.md): the Rust crate, the TypeScript and Python SDKs, framework loaders, the WordPress plugin, and the MCP server.
  - [Deployment](docs/deployment.md): platform recipes, CDN guides, and production notes.

## SDKs and integrations

Build and sign URLs from code: `@crabcrop/url` for TypeScript, `crabcrop_url` for Python, and the `crabcrop-url` Rust crate, all running the same spec test vectors, so client and server cannot drift. Loaders and integrations cover Next.js, Nuxt, Hydrogen, Astro, and SvelteKit, plus a WordPress plugin. For coding agents, a first-party MCP server exposes signing, explaining, meta, and purge over the same surface. Details in [Clients and SDKs](docs/clients.md).

The server binary doubles as its own operator CLI: `crabcrop sign`, `seal`, `unseal`, `explain`, `meta`, `purge`, `purge-group`, `upload`, and `convert-thumbor` run against a server (or offline, for the grammar-only verbs) with the same configuration conventions. Only a bare `crabcrop` starts the server; any argument runs the CLI, and one it does not know is an error. Every verb and flag is in [Command line](docs/endpoints.md#command-line).

Migrating from another service: guides for [Thumbor](docs/migrating-from-thumbor.md), [imgproxy](docs/migrating-from-imgproxy.md), [Cloudinary](docs/migrating-from-cloudinary.md), [imgix](docs/migrating-from-imgix.md), [Cloudflare Images](docs/migrating-from-cloudflare-images.md), [ImageKit](docs/migrating-from-imagekit.md), and [Gumlet](docs/migrating-from-gumlet.md).

## Deployment

Every recipe runs the same server, configured entirely through the environment. Release images are published to GHCR as `ghcr.io/dayvson/crabcrop`, with the AI variant at `ghcr.io/dayvson/crabcrop-ai`. Recipes ship for Kubernetes (Helm and Kustomize), ECS Fargate, Cloud Run, Container Apps, Docker Compose, and a binary on a VM, plus CDN guides for Cloudflare, CloudFront, Fastly, BunnyCDN, and Akamai. See [Deployment](docs/deployment.md).

## Development

```bash
cargo build --locked
bash scripts/test_all.sh
```

## Security

Found a security vulnerability? Please report it privately: see [SECURITY.md](SECURITY.md). Please do not open a public issue, pull request, or discussion for security problems.

## License

MIT. See [LICENSE](LICENSE).
