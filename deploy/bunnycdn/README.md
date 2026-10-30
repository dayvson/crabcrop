# crabcrop behind BunnyCDN

A Bunny Pull Zone in front of a crabcrop origin: origin cache headers left in charge, cache groups bridged to Bunny cache tags for group purges, and the two per-zone switches that must stay off.

**Status: doc-verified.** Every Bunny behavior below was checked against Bunny's documentation on 2026-10-25. No live Pull Zone was used; the checklist at the end is the live test, run it against yours. Two points Bunny's docs do not state outright, flagged again where they bite: that `private` and `no-store` responses stay uncached under the respect-origin setting (step 1), and that cache tags written by the middleware bridge are indexed like origin-written tags (step 5). The checklist proves both.

**The shape.** Give crabcrop its own Pull Zone, with `images.example.com` as a hostname on it and the zone's Origin URL pointing at your crabcrop origin over HTTPS. A dedicated zone keeps every setting below scoped to image traffic, and it matters three times over below: Bunny Optimizer and Token Authentication are per-zone switches (step 2 and the warning below), and a full-zone purge (step 5) can only ever mean this hostname's content.

Why the pairing works at all: crabcrop puts the signature and the operations in the URL path, and Bunny's cache key is the request URL with query strings ignored by default, so every signed variant is its own cache object with no key configuration, and a junk query parameter on a shared link cannot mint a second copy. crabcrop also sends `Cache-Control: public, max-age=31536000, immutable` on image responses, so an edge that respects origin headers caches for a year and never revalidates, which matches how crabcrop treats its own tiers.

Do not enable Token Authentication on this Pull Zone (the zone's Security settings). That is a second signature layer checked against a Bunny-side key: classic tokens travel in the query string, which this cache key ignores and crabcrop URLs never carry, and the V2 path form (`bcdn_token=...` as the first path segment) collides with the signed path layout crabcrop verifies byte-for-byte. crabcrop's in-path signatures already do the job; stacking the two means requests fail one check or the other, and every client integration carries both.

## Prerequisites

- A running crabcrop origin, reachable over HTTPS. Every recipe in this family configures a CDN in front of an origin that already works; deploy one from the platform table in the main README first.
- A Bunny account with a Pull Zone to give crabcrop (see The shape above), and your account API key for step 5's purges: URL purges are API-only, while the dashboard covers the full and tag purges.

## 1. Let origin headers govern

In the Pull Zone's **Caching > General** settings:

- **Cache Expiration Time: Respect Origin Cache-Control.** This is the setting the whole recipe hangs on. Picking any fixed duration instead replaces crabcrop's per-response TTLs with one number: the year on images, the shorter lifetime a `maxage:` segment pins for its own URL, all flattened. Respecting the origin leaves each response class with the TTL crabcrop chose for it.
- **Browser Cache Expiration Time: respect the origin** (match the server value; do not pick an Override). This setting rewrites the `Cache-Control` header browsers see. An override here would strip the `immutable` year crabcrop deliberately sends, or stamp a cacheable header onto responses that were never meant for a browser cache.
- **Smart Cache: off.** Smart Cache decides what may be cached by file extension and MIME type instead of by response headers, and it is on by default for zones accelerated by Bunny DNS. crabcrop decides by header, and its URLs end with the source path, so an extension gate gives the wrong answer in both directions: an extensionless source is never cached at all, and negotiated bytes are gated by the source's name rather than the response's type. With it off, the standard expiration logic above decides, which is the origin's headers.
- **Cache Error Responses: leave on** (the default). With it on, Bunny does not cache error responses at all; crabcrop's errors carry `no-store` anyway, so the two agree. Turning it off restores a brief (5 second) protective cache of error responses that this origin does not need.

Query strings need no work, which is not true of every CDN in this series: the key ignores them by default, and crabcrop carries nothing in one, so leave **Ignore Query Strings** on and do not enable the Vary Cache **URL Query String** option (step 3). The signed path is the whole key, exactly as crabcrop intends.

One honesty note, because the contract matters: Bunny's documentation does not say in so many words that a `private` or `no-store` response stays out of the cache under Respect Origin Cache-Control. Everything the setting promises points that way, and crabcrop's meta responses (`private, no-store`) and error responses (`no-store`) depend on it. Checklist step 2 is the proof on your zone; treat a meta HIT as a misconfiguration, not a quirk.

## 2. Keep Bunny Optimizer off

Bunny Optimizer is the second-optimizer trap on this CDN. It is a per-Pull-Zone add-on, and with it on, Bunny processes image responses itself: Smart Image Optimization recompresses per device class, automatic format optimization converts to WebP or AVIF based on Bunny's own read of the browser when enabled in the Optimizer settings, the Dynamic Images API applies its own query-string transformations, and an image nothing else claims is recompressed at the fallback quality of 85. Every one of those fights crabcrop: bytes get a second lossy pass, Bunny's negotiation (no JPEG XL, no q-values, no refusals) replaces crabcrop's, and because the Dynamic Images API keys off query parameters, its variants would need the query string in the cache key, unwinding step 1's clean key.

- On the crabcrop Pull Zone: **do not enable Bunny Optimizer** (the zone's Optimizer settings). There is nothing to configure around it; off is the whole setting.
- If another site on the account needs it, give that site its own Pull Zone. The switch is zone-wide, so there is no per-hostname opt-out to hide behind.

## 3. Format negotiation: capability classes, not Vary

Without an explicit `filters:format(...)`, crabcrop negotiates the output format from the request's `Accept` header (AVIF, then JPEG XL, then WebP, by q-value) and can vary dimensions by client hints, all under one URL, and it says so in a `Vary` response header. Bunny does not document reading the origin's `Vary` header; its cache key is assembled from its own settings, under **Caching > Vary Cache**. Three ways to handle it, in order:

1. **Pin the format in the URL** (`filters:format(avif)`). Each format becomes a different signed URL and its own cache object. No edge features involved. This is the recommended default behind any CDN, not just Bunny.
2. **Vary Cache: enable AVIF support and WebP support.** The key splits on whether the browser supports each format, giving four capability classes under one URL. Know the limits: the toggles only apply to URLs ending in `jpg`, `jpeg`, `webp`, `png`, or `gif`, and a crabcrop URL ends with the source path, so coverage follows the source filename. There is no JPEG XL class, so a JXL-capable client shares an entry with its AVIF/WebP class and gets whatever that class negotiated; if you serve JPEG XL, pin it (option 1) for those clients. And the split reads detected browser support, not the `Accept` header crabcrop negotiates from, so a client that refuses a format at q=0 can still be keyed into that format's class and served the cached variant. Fragmentation is bounded at four classes, which is the option's real advantage.
3. **Vary Cache > Request Headers, on `Accept`.** Each unique raw `Accept` value becomes its own cache entry. Correct for every visitor, and the cache fills with one entry per spelling: real clients send dozens that mean the same three formats. Bunny has no built-in normalizer to collapse them. A middleware script could rewrite `Accept` before the cache lookup (that needs before-cache execution, enabled per zone under General > Origin), the way the CloudFront recipe's function does, but that is code to own for a problem option 1 already solves, so this recipe does not ship one.

Leave the other Vary Cache settings off: each one multiplies the stored versions, and none of them names a dimension crabcrop varies by. The Mobile/Desktop setting is a device class, not a DPR split, so it does not separate client-hint variants either. DPR and client-hint variants are not separated by any of the three options: build width variants as distinct operations (the SDKs' `build_srcset` does exactly this) instead of relying on client hints behind a CDN.

## 4. Expiring URLs: the edge does not read them

`exp:` expiry lives inside the signed path, and Bunny never looks inside the URL. A response cached before expiry keeps being served until its TTL runs out or it is purged: with step 1 in place, that can be a year. What expiry still guarantees is that the origin refuses the URL, so nothing new gets rendered or re-cached after it passes.

If expiry has to bite at the edge, pick one:

- Cap the edge TTL for expiring paths with an Edge Rule: action **Override Cache Time**, condition **Request URL** matching `*/exp:*/*` (any path carrying an `exp:` segment), value no higher than the shortest signed lifetime you mint, in seconds. The override replaces crabcrop's TTL for those paths only. Cache-time overrides are first-match, so if you add more of them later, order this one deliberately.
- Purge the URL when it expires, from the same job that tracks the expiry (step 5; one URL purge from the generous Exact budget).
- Accept it, when expiry is about minting (no new renders) rather than access control. Most signed-URL deployments mean exactly that.

## 5. Purge mapping

Purge both layers, crabcrop first and Bunny second: purge the edge first and a concurrent request re-fills it from crabcrop's still-stale tiers, pinning the old bytes for another year. Bunny's purge API takes your account key in the `AccessKey` header. URL purges are API-only; the dashboard offers the full and tag purges.

| You changed | crabcrop | Bunny |
| --- | --- | --- |
| One rendered URL | `POST /purge/<signature>/<url>`, or `crabcrop purge` | `POST https://api.bunny.net/purge` with the exact URL in the `url` field |
| A preset (a cache group) | `POST /purge-group/<signature>/<name>`, or `crabcrop purge-group` | `POST https://api.bunny.net/pullzone/{id}/purgeCache` with `{"CacheTag": "<name>"}` (per the bridge below) |
| Everything on the Pull Zone | Per URL or per group, as above | `POST https://api.bunny.net/pullzone/{id}/purgeCache` with no `CacheTag` |

Why the group row is a tag and not a URL pattern: every signed URL starts with a different signature segment, so no path prefix names "every spelling of this preset" (Bunny does have prefix purges, and they cannot express this either). Tags can. Bunny indexes the tags in a response's `CDN-Tag` header; crabcrop sends its group names in the `Cache-Group` header (RFC 9875). An Edge Rule cannot bridge the two: rule values are static text or variables drawn from the request, and nothing in the rule system reads one response header into another, which is why Bunny points response transformations at middleware scripts. The bridge is the middleware script in this directory, `cache-tag-middleware.ts`: link it to the Pull Zone as a middleware script (Edge Scripting), and its `onOriginResponse` hook copies the `Cache-Group` values into `CDN-Tag`, unquoting the list into the comma-separated tags Bunny purges by. Bunny documents that hook as modifying the response before it is cached, which is when tags are recorded, and no before-cache execution is needed because the bridge only touches origin responses.

Two things the docs do not state outright, so the checklist has to: that tags written by a script are indexed exactly like origin-written tags, and the separator when one header carries several tags (the script uses the standard comma-separated list form). Checklist step 4 proves both on your zone. Bunny does document wildcard matching in tag purges, so group names built with a shared prefix can be purged as a family in one call.

Costs and limits, so the mapping stays honest: the purge API is rate-limited by a token bucket, per account and per purge type, one token per item. A single-URL purge and a tag purge are the Exact type: a burst of 120 tokens refilling at 5 per second, roughly 300 a minute sustained. Prefix URL purges and full-zone purges are the Prefix type: a burst of 20 refilling at 0.5 per second, roughly 30 a minute. Over the limit, the API answers 429 with retry guidance. Two consequences: a group purge costs one Exact token no matter how many objects carry the tag, so it is the cheap way to clear a preset; and clearing many URLs one by one is the expensive shape on this CDN, the reverse of some others in this series. Send the URL purge exactly as clients request the URL; with query strings ignored (step 1), the endpoint's exact-versus-prefix distinction turns on the path alone.

One membership subtlety, mirrored on both sides: tags only attach to responses cached while the bridge was live, and crabcrop groups accumulate the same way. Right after linking the script, purge the whole zone once to re-seed the edge cache with tagged copies. And if Perma-Cache is on, tag purging is not supported at all; read step 6 before enabling it.

## 6. Origin Shield, and the Perma-Cache trade (optional)

**Origin Shield** is the direct fit: one secondary caching location between all edge PoPs and your origin (Caching > Origin Shield; pick the region closest to the origin). Edge misses consolidate there, so crabcrop sees one fill per object instead of one per PoP, on top of its own two tiers. It is free, and traffic from the shield to the edge nodes is not charged. The cost is one extra hop of latency on misses.

**Perma-Cache** is the other second layer Bunny offers, and it is a different bargain: a geo-replicated Bunny Storage zone behind the edge that keeps a copy of everything effectively permanently, answering edge misses from storage rather than the origin. Its purge contract is weaker in exactly the places this recipe leans on: a single-URL purge deletes the stored copy and then purges the edge, a full-zone purge deletes nothing (it rotates to a fresh storage directory and leaves the old one for you to remove), and tag purging is not supported with Perma-Cache enabled, which removes the group row of step 5's table entirely. The two layers also cannot be combined. For this pairing, take Origin Shield; treat Perma-Cache as a deliberate trade, hit ratio bought with the purge workflow, not a free upgrade.

## Checklist

Run against your Pull Zone after setup:

1. Request a signed image URL twice. The second response shows `CDN-Cache: HIT`, and the first carries crabcrop's own `cache-status` header naming the tier that served the origin fill.
2. Request the same URL as meta (`Accept: application/json`) twice. It is never a HIT on repeat: `private, no-store` is respected, which also proves Respect Origin Cache-Control is really in effect (the promise Bunny's docs leave implicit). Then compare one image response against the same URL fetched directly from the origin: identical bytes and `Content-Type` prove no Optimizer processing sits in front of crabcrop.
3. Purge one URL in both layers, then request it: `CDN-Cache: MISS`, then HIT again on the next request.
4. If you use the tag bridge: request a `preset:` URL, purge by tag (`CacheTag` set to the preset name), and request it again. A MISS proves the middleware's tags landed and are indexed.
5. If you left formats unpinned with the Vary Cache toggles on, request one URL with an AVIF-capable `Accept` header and then with a WebP-only one. Different bytes prove the capability classes are splitting the key; identical bytes mean the toggles are not in effect (check that the source path ends in one of the extensions the toggles cover).

## Teardown

Delete the Pull Zone and crabcrop is off Bunny entirely. To keep the zone but undo the recipe: set the Cache Expiration Time back to a fixed duration of your choice, remove the step 4 Edge Rule, unlink the middleware script, and turn Origin Shield off. Bunny Optimizer and Token Authentication were never enabled, so there is nothing to switch back there.
