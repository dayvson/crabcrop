# Crabcrop for WordPress

A WordPress plugin that serves a site's media through a self-hosted
[crabcrop](../../README.md) instance. It rewrites media URLs as
WordPress produces them (featured images, images in post content, and
srcset) into crabcrop URLs against an instance the operator already
hosts. There is no hosted service and no account.

The original media library is untouched: originals stay where
WordPress put them, every rewrite happens at render time, and
disabling the plugin returns the site to plain URLs. A problem the
plugin can see is safe: an unmapped size, a missing base URL, a
signed mode without a secret, or a URL the builder rejects all
leave the original URL in place. The plugin never contacts the
instance, though, so it cannot see one that is down or
misconfigured (a preset missing on the server, a secret that does
not match): rewritten images then fail to load until the instance
is fixed or the plugin is disabled.

## Install

1. Copy this `wordpress/` directory into `wp-content/plugins/` as
   `wp-content/plugins/crabcrop/` (the plugin file is
   `crabcrop.php`).
2. Activate **Crabcrop** under Plugins.
3. Configure it under **Settings > Crabcrop**, or in code (below).

## Modes

**Presets only (recommended).** WordPress image sizes map to preset
names defined on the server, and the plugin builds `unsafe` preset
URLs, so no secret ever lives in WordPress. Run the instance with:

```text
CRABCROP_ONLY_PRESETS=1
CRABCROP_ALLOW_UNSAFE=1
CRABCROP_PRESETS_FILE=/etc/crabcrop/presets.txt
```

`CRABCROP_ONLY_PRESETS=1` is what makes this safe rather than merely
unsigned: the server rejects free-form operations with `400`, so the
public URLs can only ever name a preset, and there is no way to
invent a transform. The transform policy (crop, format, quality)
lives in the presets file on the server, next to the instance, and
rotating it never touches WordPress.

**Signed.** Sizes map to operation strings, signed with the
instance's `CRABCROP_SECRET` held in the plugin settings. Choose
this when per-size operations are preferable to maintaining
presets. The secret is only ever used server-side, while WordPress
renders a page; it never appears in the URLs or the page output.

## Configuration

Settings resolve in this order, highest precedence first:

1. **Constants** in `wp-config.php`: `CRABCROP_WP_BASE_URL`,
   `CRABCROP_WP_MODE` (`presets` or `signed`), `CRABCROP_WP_SECRET`,
   `CRABCROP_WP_KEY_ID` (signature key ID for rotation windows),
   `CRABCROP_WP_SIZE_MAP` (an array, or the same text format as the
   settings screen), `CRABCROP_WP_DISABLE_SIZES`.
2. **Filters**: `crabcrop_wp_base_url`, `crabcrop_wp_mode`,
   `crabcrop_wp_secret`, `crabcrop_wp_key_id`,
   `crabcrop_wp_size_map`, `crabcrop_wp_disable_sizes`. A final
   `crabcrop_wp_url` filter receives every built URL, with the size
   name and attachment ID, for last-word adjustments.
3. **The settings screen** (Settings > Crabcrop): base URL, mode,
   secret, size mapping, and the sizes toggle. A field set by a
   constant is shown disabled, with the constant named.

Code wins over the admin screen on purpose: a deployment managed in
`wp-config.php` cannot be silently repointed from the dashboard.

## Size mapping

One mapping per line, as `size = target`:

- Presets mode: the target is a preset name.
- Signed mode: the target is an operation string used verbatim, or
  a preset reference.

```text
# presets mode
thumbnail = wp-thumb
medium    = wp-medium
large     = wp-large

# signed mode
thumbnail = crop:150x150:smart
medium    = fit-in:300x300
large     = preset:card
```

Sizes with no mapping keep their original URLs. The bare attachment
URL (`wp_get_attachment_url`) is rewritten only when `full` is
mapped. In srcset, an entry that belongs to a mapped size is
rewritten from the original file; in signed mode, a width no
registered size owns (typically the full width) becomes a
width-only `Wx0` resize, and in presets mode it keeps its original
URL.

An opt-in toggle, **Stop generating intermediate sizes**, makes
WordPress skip its own resized copies at upload time; crabcrop
renders every displayed size on demand instead. Originals are
uploaded and kept either way.

## Worked example

The server runs presets-only mode with this presets file:

```text
# /etc/crabcrop/presets.txt
wp-thumb  = crop:150x150:smart/filters:format(webp)
wp-medium = fit-in:300x300/filters:format(webp)
wp-large  = fit-in:1024x1024/filters:format(webp)
```

The plugin is in presets mode with base URL
`https://images.example.com` and the size mapping above. A featured
image uploaded as
`https://blog.example.com/wp-content/uploads/2026/11/photo.jpg` is
then served in content as:

```text
https://images.example.com/unsafe/preset:wp-medium/https://blog.example.com/wp-content/uploads/2026/11/photo.jpg
```

with the `thumbnail` and `large` srcset entries following the same
shape through their own presets.

## Tests

`php tests/run.php` runs the suite with no WordPress install: the
WordPress functions the plugin calls are stubbed, the configuration
precedence and the rewrite rules are exercised end to end through
the stubs, and the URL builder runs the canonical vectors from
[`spec/test-vectors.json`](../../spec/test-vectors.json), the same
suite every other implementation runs, so the PHP signatures cannot
drift from the server's.
