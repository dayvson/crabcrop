# crabcrop behind Cloudflare

A Cloudflare zone in front of a crabcrop origin: the cache rules, the purge mapping, and the two settings that quietly break image services when they are wrong.

**Status: doc-verified.** Every Cloudflare behavior below was checked against Cloudflare's documentation on 2026-10-25. No live zone was used; the checklist at the end is the live test, run it against yours.

**The shape.** Give crabcrop its own hostname (`images.example.com`), proxied through Cloudflare, with SSL mode Full (Strict). A dedicated hostname keeps every rule below scoped to image traffic and nothing else.

Why the pairing works at all: crabcrop puts the signature and the operations in the URL path, and Cloudflare's default cache key is the full URL, so every signed variant is its own cache entry with no key configuration. crabcrop also sends `Cache-Control: public, max-age=31536000, immutable` on image responses, so an edge that respects origin headers caches for a year and never revalidates, which matches how crabcrop treats its own tiers.

## 1. Make image responses eligible, and respect the origin

By default Cloudflare only caches a fixed list of file extensions, and a crabcrop URL does not guarantee one (a source can be extensionless). One Cache Rule fixes eligibility and pins the TTL behavior:

- Match: `http.host eq "images.example.com"`
- Cache eligibility: **Eligible for cache**
- Edge TTL: **Respect origin**
- Browser TTL: **Respect origin**

**Never use an Edge TTL override on the crabcrop hostname.** An override that ignores origin cache headers caches even `no-store` responses, so crabcrop's meta JSON (`private, no-store`) and its error responses would be pinned at the edge. Respect origin also preserves the per-URL `maxage:` directive, which an override would flatten to a single value.

## 2. Turn Polish off for this hostname

Polish (Pro plans and up) recompresses images at the edge. On a crabcrop hostname that means every image is optimized twice, and double lossy compression costs more quality than it saves bytes. Polish's optional WebP conversion also fights crabcrop's own format negotiation (step 3).

- Configuration Rule, match `http.host eq "images.example.com"`: **Polish: Off**

Image Transformations need no opt-out: they only run on `/cdn-cgi/image/` URLs or from a Worker, so crabcrop paths are untouched. Do not add a Transform Rule that rewrites the whole zone (or this hostname) into `/cdn-cgi/image/`; that rule would put a second transformer in front of the first.

## 3. Format negotiation: pin it, or split the cache

Without an explicit `filters:format(...)`, crabcrop negotiates the output format from the request's `Accept` header (AVIF, then JPEG XL, then WebP) and can vary dimensions by client hints (DPR), all under one URL. Cloudflare's default cache key does not include `Accept`, so the first variant cached is the variant every later visitor gets, whatever their browser sent. Three ways to handle it, in order:

1. **Pin the format in the URL** (`filters:format(avif)`). Each format becomes a different signed URL and its own cache entry. Works on every plan, no edge features involved. This is the recommended default behind any CDN, not just Cloudflare.
2. **Vary for Images** (Pro plans and up), Cloudflare's mechanism for origin-negotiated variants. It keys off a configured extension list (avif, webp, jpg, png, gif, and a few more), so it covers AVIF and WebP negotiation, but not JPEG XL, and not DPR variants. It is configured through the API, not a Cache Rule, and it is incompatible with Cache Reserve (step 6).
3. **A custom cache key including `Accept` values** (Enterprise only) splits entries by negotiated format without URL changes, at Enterprise prices.

DPR variants share an entry under options 2 and 3 as well: build width variants as distinct operations (the SDKs' `build_srcset` does exactly this) instead of relying on client hints behind a CDN.

## 4. Expiring URLs: the edge does not read them

`exp:` expiry lives inside the signed path, and Cloudflare never looks inside the URL. A response cached before expiry keeps being served until its Edge TTL runs out or it is purged: with the respect-origin rule above, that can be a year. What expiry still guarantees is that the origin refuses the URL, so nothing new gets rendered or re-cached after it passes.

If expiry has to bite at the edge, pick one:

- Add a second Cache Rule for the expiring paths with an Edge TTL override no longer than the signed lifetime. The override replaces crabcrop's TTL for those paths only, including any `maxage:` on them.
- Purge the URL when it expires, from the same job that tracks the expiry.
- Accept it, when expiry is about minting (no new renders) rather than access control. Most signed-URL deployments mean exactly that.

Cloudflare can validate its own HMAC tokens per request in the WAF (`is_timed_hmac_valid_v0()`), but those are Cloudflare-format tokens, not crabcrop signatures. Adopting it means a second token layer in front of the first, which this recipe does not recommend.

## 5. Purge mapping

Purge both layers, crabcrop first and Cloudflare second: purge the edge first and a concurrent request re-fills it from crabcrop's still-stale tiers, pinning the old bytes for another year. Every Cloudflare purge method below is available on every plan (Cloudflare opened tag, hostname, and prefix purge to all plans in April 2025). The limits are rate limits: flexible purges allow 100 operations per request, at 5 per minute on Free, 5 per second on Pro, and 10 per second on Business.

| You changed | crabcrop | Cloudflare |
| --- | --- | --- |
| One rendered URL | `POST /purge/<signature>/<url>`, or `crabcrop purge` | Purge by URL (`files`), the same URL |
| A preset (a cache group) | `POST /purge-group/<signature>/<name>`, or `crabcrop purge-group` | Purge by tag (`tags`), tag = preset name, tags attached per below |
| Everything on the hostname | Per URL or per group, as above | Purge by hostname (`hosts`) |

The API endpoint for all three is `POST /zones/{zone_id}/purge_cache`, with a `files`, `tags`, or `hosts` array in the body.

Attaching the tags: purge by tag only works if responses were cached carrying a `Cache-Tag` header. crabcrop sends its group names in the `Cache-Group` header (RFC 9875) on grouped responses, and Cloudflare wants `Cache-Tag`. Two bridges:

- **A Cache Response Rule (no code):** one rule per preset. Match `http.request.uri.path contains "/preset:<name>/"` and set Cache-Tag to `<name>`. Preset URLs carry `preset:<name>` as a path segment, so the match is exact. Rule budgets are small on lower plans, so this suits a handful of presets.
- **The Worker bridge (many or dynamic groups):** `cache-tag-worker.js` in this directory copies whatever `Cache-Group` values a response carries into `Cache-Tag`. Preset names must be valid tag characters (printable ASCII, no spaces), which crabcrop preset names are in practice. Cloudflare strips `Cache-Tag` before delivery, so visitors never see it. Step 4 of the checklist proves the tags landed on your zone.

One membership subtlety, mirrored on both sides: crabcrop groups accumulate (an entry joins a group the first time any grouped spelling of its URL is served), and Cloudflare tags only attach to responses cached while the bridge was live. Right after enabling a bridge, purge the hostname once to re-seed the edge cache with tagged copies.

## 6. Tiered Cache and Cache Reserve (optional)

- **Tiered Cache (Smart topology)** is available on all plans and worth turning on: upper-tier data centers are the only ones that talk to the origin, so crabcrop sees one fill per region instead of one per data center, on top of its own two tiers.
- **Cache Reserve** (a paid, R2-backed store) keeps long-tail variants from being evicted. crabcrop's one-year TTL clears its 10-hour admission bar. Skip it if you enabled Vary for Images; the two are incompatible.

## Checklist

Run against your zone after setup:

1. Request a signed image URL twice. The second response shows `cf-cache-status: HIT`, and the first carries crabcrop's own `cache-status` header naming the tier that served the origin fill.
2. Request the same URL as meta (`Accept: application/json`). It is never a HIT on repeat: `private, no-store` is respected.
3. Purge one URL in both layers, then request it: `cf-cache-status: MISS`, then HIT again on the next request.
4. If you use the tag bridge: request a `preset:` URL, purge by its tag, and request it again. A MISS proves the tag landed.
5. If you left formats unpinned, request one URL with an AVIF `Accept` header and then with a JPEG-only one. Identical bytes mean your variant split from step 3 is not in effect.
