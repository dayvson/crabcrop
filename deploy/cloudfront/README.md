# crabcrop behind CloudFront

A CloudFront distribution in front of a crabcrop origin: the cache policy that respects the origin, the purge mapping (paths for URLs, cache tags for groups), and the one normalization CloudFront cannot express in a policy.

**Status: doc-verified.** Every CloudFront behavior below was checked against AWS's documentation on 2026-10-25, and the function in this directory was tested against the server's negotiation rules. No live distribution was used; the checklist at the end is the live test, run it against yours.

**The shape.** Give crabcrop its own hostname (`images.example.com`) as an alternate domain name on the distribution, with the crabcrop origin as a custom origin over HTTPS, and one behavior for the hostname: viewer protocol policy redirect-to-https, allowed methods GET and HEAD. A behavior of its own keeps every setting below scoped to image traffic.

Why the pairing works at all: crabcrop puts the signature and the operations in the URL path, and CloudFront's cache key starts from the full path, so every signed variant is its own cache entry with no key configuration. crabcrop also sends `Cache-Control: public, max-age=31536000, immutable` on image responses, so an edge that respects origin headers caches for a year and never revalidates, which matches how crabcrop treats its own tiers.

Do not require CloudFront signed URLs or signed cookies on this behavior. Those are a second signature layer checked against CloudFront's own keys; crabcrop's in-path signatures already do the job, and stacking the two means every client integration carries both.

## 1. A cache policy that respects the origin

Create a custom cache policy for the behavior:

- **Minimum TTL: 0**
- **Default TTL: 0**
- **Maximum TTL: 31536000** (one year, matching what crabcrop sends)
- Cache key: no query strings, no cookies, no headers (step 3 adds one header, deliberately)
- Compression: off (the payloads are already-encoded images)

The managed `CachingOptimized` policy looks close and is wrong for this origin in one specific way: its minimum TTL is 1 second, and **a minimum TTL above 0 makes CloudFront ignore `no-store`, `no-cache`, and `private` from the origin**. crabcrop marks its meta responses and its error responses `private, no-store` on purpose; under a nonzero floor those get cached anyway. Minimum TTL 0 is the whole fix: explicit origin headers then govern, the year-long image TTL is honored (capped by the maximum), and a response with no cache headers at all gets the default TTL of 0, which is to say it is not cached.

Pair it with an origin request policy that forwards what crabcrop actually reads and nothing else:

- Headers: `Accept`, `DPR`, `Sec-CH-DPR`, `Sec-CH-Width`, `Sec-CH-Viewport-Width`
- Query strings: none. Cookies: none.

Query strings carry nothing in a crabcrop URL (the whole request is the path), so keeping them out of both the key and the origin request stops a junk parameter from fragmenting the cache or reaching the origin.

## 2. Nothing to turn off

CloudFront has no built-in image transformation for custom origins, so there is no equivalent of the "second optimizer" trap some other CDNs set: nothing recompresses or converts your images behind your back. The corresponding trap here is additive: the AWS samples and blog posts for "dynamic image transformation with CloudFront" wire a Lambda@Edge function in front of S3 or an origin. Do not put one in front of crabcrop. That is the same double-optimizer mistake, installed by hand.

## 3. Format negotiation: normalize it, or split the cache

Without an explicit `filters:format(...)`, crabcrop negotiates the output format from the request's `Accept` header (AVIF, then JPEG XL, then WebP, by q-value) and can vary dimensions by client hints, all under one URL. With no header in the cache key, the first variant cached is the variant every later visitor gets, whatever their browser sent. Three ways to handle it, in order:

1. **Pin the format in the URL** (`filters:format(avif)`). Each format becomes a different signed URL and its own cache entry. No edge features involved. This is the recommended default behind any CDN, not just CloudFront.
2. **The `normalize-accept.js` function in this directory** (a CloudFront Function, attached as viewer request). A cache policy *can* add the raw `Accept` header to the cache key, but raw values fragment terribly: browsers and clients send dozens of spellings and orderings that all mean the same three formats. The function rewrites `Accept` to the single token the origin's own negotiation would pick (`image/avif`, `image/jxl`, or `image/webp`), following the server's exact rules (q-values, refusal at q=0, the AVIF/JXL/WebP tie order, wildcards are not an offer). Viewer request functions run before the cache lookup, so the rewritten header is both what the cache key sees (add `Accept` to the key when you use the function) and what reaches the origin. Meta requests pass through untouched: an `Accept` asking for `application/json` is how crabcrop knows to answer with JSON, and rewriting it would break meta. Requests offering no negotiable format also pass through, and the origin serves the source format. The function is verified against the server's negotiation tests' cases; it costs $0.10 per million invocations.
3. **Raw `Accept` in the cache key**, no function. Correct, and the cache fills with one entry per spelling. Only reasonable on low-traffic distributions.

DPR and client-hint variants are not separated by any of the three: build width variants as distinct operations (the SDKs' `build_srcset` does exactly this) instead of relying on client hints behind a CDN. The origin request policy still forwards the hints, so an origin reached directly behaves the same as one behind the distribution; they just share an edge entry.

## 4. Expiring URLs: the edge does not read them

`exp:` expiry lives inside the signed path, and CloudFront never looks inside the URL. A response cached before expiry keeps being served until its TTL runs out or it is invalidated: with the policy above, that can be a year. What expiry still guarantees is that the origin refuses the URL, so nothing new gets rendered or re-cached after it passes.

If expiry has to bite at the edge, pick one:

- Add a second behavior for the expiring paths with a maximum TTL no longer than the signed lifetime. The cap replaces crabcrop's TTL for those paths only.
- Invalidate the URL when it expires, from the same job that tracks the expiry (step 5; a path invalidation is one path of quota).
- Accept it, when expiry is about minting (no new renders) rather than access control. Most signed-URL deployments mean exactly that.

## 5. Purge mapping

Purge both layers, crabcrop first and CloudFront second: invalidate the edge first and a concurrent request re-fills it from crabcrop's still-stale tiers, pinning the old bytes for another year. CloudFront's word for a purge is an invalidation, submitted with `CreateInvalidation` (console, CLI, or API) as a batch of items:

| You changed | crabcrop | CloudFront |
| --- | --- | --- |
| One rendered URL | `POST /purge/<signature>/<url>`, or `crabcrop purge` | One item: the exact path |
| A preset (a cache group) | `POST /purge-group/<signature>/<name>`, or `crabcrop purge-group` | One item: `#<name>` (a tag invalidation, per below) |
| Everything on the behavior | Per URL or per group, as above | One item: `/*` |

Why the group row is a tag and not a wildcard: a CloudFront path wildcard must be the path's only wildcard and its last character, and every signed URL starts with a different signature segment, so no path pattern can name "every spelling of this preset". CloudFront's cache tags (a 2026 feature) can. Configure the distribution with a `CacheTagConfig` whose header name is `Cache-Group`: crabcrop already sends its group names in that response header (RFC 9875) on grouped responses, CloudFront stores the values as the cached object's tags, and an invalidation item of `#<name>` removes every object carrying that tag, all variants included. Tag values are case-insensitive at CloudFront while crabcrop group names are case-sensitive, so keep group names lowercase to avoid two groups colliding into one tag. Objects carry at most 50 tags; crabcrop responses carry only their cache groups, far under the cap.

Costs and limits, so the mapping stays honest: the first 1,000 invalidation paths per month are free across the whole AWS account, then $0.005 per path, and a wildcard or tag item counts as one path. Up to 3,000 individually named paths and up to 15 wildcard paths can be in progress at once per distribution. Invalidations take effect in under 5 seconds at p95 by AWS's current numbers, with completion reported in under 25 seconds at p95: unlike a TTL expiry, an invalidation here is effectively immediate, but it is still asynchronous, so the checklist verifies rather than assumes.

Two membership subtleties, mirrored on both sides: tags only attach to responses cached while the `CacheTagConfig` was in place, and crabcrop groups accumulate the same way. Right after enabling the tag config, invalidate `/*` once to re-seed the edge cache with tagged copies.

## 6. Origin Shield (optional)

Origin Shield adds one regional caching layer between the edge locations and your origin, so crabcrop sees one fill per object per region instead of one per edge location, on top of its own two tiers. It charges a small per-request fee for requests that reach it. Worth it when the origin is far from most viewers or renders are expensive (AI operations); unnecessary when viewers and origin share a region.

## Checklist

Run against your distribution after setup:

1. Request a signed image URL twice. The second response shows `x-cache: Hit from cloudfront`, and the first carries crabcrop's own `cache-status` header naming the tier that served the origin fill.
2. Request the same URL as meta (`Accept: application/json`). It is never a Hit on repeat: `private, no-store` is respected, which also proves your minimum TTL is 0.
3. Invalidate one URL in both layers, then request it: `x-cache: Miss from cloudfront`, then a Hit again on the next request.
4. If you use tag invalidation: request a `preset:` URL, invalidate `#<name>`, and request it again. A Miss proves the tags landed.
5. If you left formats unpinned with the function attached, request one URL with a Chrome `Accept` header and then with a Safari one. Different bytes (AVIF versus the Safari-negotiated format) under the two spellings that mean the same thing prove the normalization is keying correctly; identical bytes across an AVIF and a JPEG-only `Accept` mean it is not.
