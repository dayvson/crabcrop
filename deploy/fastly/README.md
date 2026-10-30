# crabcrop behind Fastly

A Fastly CDN service in front of a crabcrop origin: origin cache headers left in charge, cache groups bridged to surrogate keys for group purges, and the one service-wide switch that must stay off.

**Status: doc-verified.** Every Fastly behavior below was checked against Fastly's documentation on 2026-10-25, and the snippet logic was tested against the server's negotiation and header rules. No live service was used; the checklist at the end is the live test, run it against yours.

**The shape.** Give crabcrop its own Fastly service, with `images.example.com` as a domain on it and the crabcrop origin as its host over HTTPS. A dedicated service keeps every setting below scoped to image traffic, and it matters twice over below: Image Optimizer is a per-service switch (step 2), and purges execute per service, so purge all (step 5) can only ever mean this hostname's content.

Why the pairing works at all: crabcrop puts the signature and the operations in the URL path, and Fastly's default cache key (the standard VCL boilerplate's `vcl_hash`) is the request URL plus the Host header, so every signed variant is its own cache object with no key configuration. crabcrop also sends `Cache-Control: public, max-age=31536000, immutable` on image responses, and Fastly honors `Cache-Control: max-age` as the TTL whenever no `Surrogate-Control` is present, so an image caches for a year and is not revalidated, which matches how crabcrop treats its own tiers.

Do not add an edge token-authentication layer in front of the service. That is a second token checked against Fastly-side secrets; crabcrop's in-path signatures already do the job, and stacking the two means every client integration carries both.

## Prerequisites

- A running crabcrop origin, reachable over HTTPS. Every recipe in this family configures a CDN in front of an origin that already works; deploy one from the platform table in the main README first.
- A Fastly service for the crabcrop domain (the recipe gives crabcrop its own service; see The shape above), and an API token: step 5's purges go through the API.

## 1. Let origin headers govern

Fastly needs no TTL configuration for this origin, provided you do not add any:

- **Create no cache settings (and no request settings) that force a TTL or a cache action for this service.** A TTL set in the control panel can override the TTL the origin sent. Left alone, the origin's `max-age` is the TTL.
- **Nothing has to be done about `private` and `no-store`.** The standard boilerplate every generated service runs passes any response whose `Cache-Control` contains `private` or `no-store`, so crabcrop's meta responses (`private, no-store`) and its error responses (`no-store`) are never stored, and no pass rules are needed for them or for anything else: signed image paths are plain GETs, which the boilerplate looks up by default.
- **The fallback TTL never fires here.** A response carrying no `max-age` at all would get the boilerplate's 3600-second fallback. crabcrop sends explicit cache headers on every response class, so leave the fallback setting alone.

One header-order note: Fastly prefers `Surrogate-Control: max-age` over `Cache-Control: max-age` when both are present. crabcrop does not send `Surrogate-Control`, so the one `Cache-Control` header governs Fastly and browsers alike. Do not synthesize a `Surrogate-Control` at the edge; splitting the two audiences is a problem this origin does not have.

Optional hardening: a crabcrop URL carries everything in its path, but the default cache key includes the query string, so a junk parameter on a shared link mints a second cache object for the same image. The `crabcrop-strip-query-string` snippet in `fastly-snippets.vcl` (type recv) drops the query string before the lookup; the origin never reads one.

## 2. Keep Image Optimizer off

Fastly Image Optimizer is the second-optimizer trap on this CDN, and it is blunter than most: it is a per-service purchase and switch, turning it on takes effect immediately on all service versions including the active one, and once on, its default settings transform every image the service serves. Those defaults recompress JPEG and WebP output to quality 85, convert to WebP in browsers that support it (its own negotiation, fighting crabcrop's), and strip metadata. By default it routes by file extension (PNG, JPG, JPEG, GIF, WEBP), and a crabcrop URL ends with the source path, so whether a response gets reprocessed would depend on the source filename. The standard boilerplate even unsets the `Vary` header on responses Image Optimizer processed, erasing the record crabcrop's negotiation depends on.

- On the crabcrop service: **do not enable Image Optimizer.** There is nothing to configure around it; off is the whole setting.
- If another site on the account needs it, give that site its own service. The switch is service-wide, so there is no per-hostname opt-out to hide behind.

A response Image Optimizer touched carries a `fastly-io-info` header. Its absence on this hostname is step 2 of the checklist.

## 3. Format negotiation: Vary works here, normalize only if fragmentation bites

Without an explicit `filters:format(...)`, crabcrop negotiates the output format from the request's `Accept` header (AVIF, then JPEG XL, then WebP, by q-value) and can vary dimensions by client hints, all under one URL, and it says so in a `Vary` response header. Fastly supports `Vary` per spec: the named request headers join the cache key for the object, so each distinct `Accept` value gets its own variant under the same URL and every visitor is served a variant their own request negotiated. Three ways to handle it, in order:

1. **Pin the format in the URL** (`filters:format(avif)`). Each format becomes a different signed URL and its own cache object. No edge features involved. This is the recommended default behind any CDN, not just Fastly.
2. **Leave `Vary` in charge.** Correct for every visitor with zero configuration, which is not true of every CDN in this series. The cost is fragmentation: real clients send dozens of `Accept` spellings and orderings that mean the same three formats, plus distinct `DPR` and client-hint values, and each combination is another object filling and another fill from the origin.
3. **Normalize `Accept` with the `crabcrop-normalize-accept` snippet** in `fastly-snippets.vcl` (type recv). It rewrites `Accept` to the single token the origin's own negotiation would pick, following the server's exact rules (q-values, refusal at q=0, an unparseable q counts as 1, the AVIF/JXL/WebP tie order, wildcards are not an offer, a format named twice is governed by its first entry). Snippets run before the cache lookup, so the rewritten header is both what `Vary` matching sees and what reaches the origin on a miss. Meta requests pass through untouched: an `Accept` asking for `application/json` is how crabcrop knows to answer with JSON, and rewriting it would break meta. Requests offering no negotiable format also pass through, and the origin serves the source format.

Client-hint variants are not collapsed by any of the three: build width variants as distinct operations (the SDKs' `build_srcset` does exactly this) instead of relying on client hints behind a CDN.

## 4. Expiring URLs: the edge does not read them

`exp:` expiry lives inside the signed path, and Fastly never looks inside the URL. A response cached before expiry keeps being served until its TTL runs out or it is purged: with step 1 in place, that can be a year. What expiry still guarantees is that the origin refuses the URL, so nothing new gets rendered or re-cached after it passes.

If expiry has to bite at the edge, pick one:

- Cap the edge TTL for expiring paths with the `crabcrop-expiry-ttl-cap` snippet in `fastly-snippets.vcl` (type fetch): when the path carries an `exp:` segment and the origin's TTL exceeds the cap, the TTL is clamped. Set the cap no higher than the shortest signed lifetime you mint. The clamp replaces crabcrop's TTL for those paths only.
- Purge the URL when it expires, from the same job that tracks the expiry (step 5; one URL purge from a generous budget).
- Accept it, when expiry is about minting (no new renders) rather than access control. Most signed-URL deployments mean exactly that.

## 5. Purge mapping

Purge both layers, crabcrop first and Fastly second: purge the edge first and a concurrent request re-fills it from crabcrop's still-stale tiers, pinning the old bytes for another year. Fastly calls all of these Instant Purge; they take effect immediately rather than on a propagation delay, and the checklist still verifies rather than assumes. All three go through the API with a token (console and CLI equivalents exist). Send purges through the API, not as raw `PURGE` requests to the hostname: edge URL purges are unauthenticated by default.

| You changed | crabcrop | Fastly |
| --- | --- | --- |
| One rendered URL | `POST /purge/<signature>/<url>`, or `crabcrop purge` | `POST https://api.fastly.com/purge/<full-url>` |
| A preset (a cache group) | `POST /purge-group/<signature>/<name>`, or `crabcrop purge-group` | `POST /service/{service_id}/purge/<name>` (a surrogate key purge, per below) |
| Everything on the service | Per URL or per group, as above | `POST /service/{service_id}/purge_all` |

Why the group row is a key and not a URL pattern: every signed URL starts with a different signature segment, so no URL prefix names "every spelling of this preset", and Fastly has no wildcard URL purge. Surrogate keys can. crabcrop sends its group names in the `Cache-Group` response header (RFC 9875) on grouped responses; the `crabcrop-surrogate-keys` snippet in `fastly-snippets.vcl` (type fetch) copies them into the `Surrogate-Key` header Fastly indexes, unquoting the list into the space-separated tokens Fastly wants. Group names must work as single tokens (no spaces), which preset names are in practice. Two properties line up exactly: surrogate keys are case-sensitive, and so are crabcrop group names, so a key purge names the group byte-for-byte (not true of every CDN's tag system). Fastly strips `Surrogate-Key` before delivery, so visitors never see it; a request carrying `Fastly-Debug: 1` shows the keys on a cached response, which is how step 4 of the checklist proves they landed.

Costs and limits, so the mapping stays honest: URL and surrogate key purges share a budget of an average of 100,000 purges per hour per customer, and each key targeted counts as one purge no matter how many objects carry it. A batch endpoint (`POST /service/{service_id}/purge`) purges up to 256 keys in one request. Purge all is always a hard purge and counts against the global API rate limits instead. URL and key purges default to hard (the object is gone and the next request refills from the origin); adding the `Fastly-Soft-Purge: 1` header makes one a soft purge, marking the object stale so it can still be served while the origin is refreshed or down. Individual keys are capped at 1,024 bytes and the whole `Surrogate-Key` header at 16,384 bytes, with anything past the limit dropped rather than truncated; crabcrop group lists sit far under both.

Two variant subtleties: a URL purge targets the cache key, so it invalidates every `Vary` variant of that URL at once, while a key purge invalidates only the variants carrying the key (variants of one URL normally carry the same groups, so the rows above behave the same in practice). And one membership subtlety, mirrored on both sides: keys only attach to responses cached while the snippet was live, and crabcrop groups accumulate the same way. Right after installing the snippet, purge all once to re-seed the edge cache with keyed copies.

## 6. Shielding (optional)

Shielding designates one Fastly POP as the shield for the host: edge POPs that miss ask the shield instead of the origin, so crabcrop sees one fill per object from the shield POP rather than one per edge POP, on top of its own two tiers. Pick a shield data center close to the origin (Origins, your host, Shielding). Two things to know before enabling it: inbound traffic to the shield is billed as regular traffic, and VCL executes twice, once at the edge POP and once at the shield, so the fetch snippets above apply at both tiers (they are written to be safe under that: the key bridge sets the same value from the same header, and the TTL cap only ever clamps down). Worth it when the origin is far from most viewers or renders are expensive (AI operations); unnecessary when viewers and origin share a region.

## Checklist

Run against your service after setup:

1. Request a signed image URL twice. The second response shows `x-cache: HIT`, and the first carries crabcrop's own `cache-status` header naming the tier that served the origin fill.
2. Request the same URL and confirm there is no `fastly-io-info` response header: Image Optimizer is off for this service. Then request the URL as meta (`Accept: application/json`) twice. It is never a HIT on repeat: `private, no-store` is passed, which also proves no cache setting is forcing a TTL.
3. Purge one URL in both layers, then request it: `x-cache: MISS`, then HIT again on the next request.
4. If you use key purges: request a `preset:` URL with a `Fastly-Debug: 1` request header and confirm the debug response shows a `Surrogate-Key` header carrying the preset name. Purge that key in both layers and request the URL again. A MISS proves the keys landed.
5. If you left formats unpinned, request one URL with a Chrome `Accept` header and then with a Safari one: different bytes (each browser's negotiated format) prove `Vary` is splitting variants. If you installed the normalize snippet, follow with a different spelling of the Chrome `Accept` (same formats, different order or spacing): a HIT with identical bytes proves the normalization is keying correctly.

## Teardown

Clone the active service version, remove what the recipe added (the snippets from `fastly-snippets.vcl`, and shielding if you enabled it in step 6), and activate. Image Optimizer stays off either way: it was never enabled. To remove crabcrop from Fastly entirely, delete the service and point the domain's DNS elsewhere.
