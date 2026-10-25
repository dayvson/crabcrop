# crabcrop Fastly snippets.
#
# Four regular VCL snippets for a Fastly CDN service in front of a
# crabcrop origin. Create each block below as its own snippet,
# with the name and type noted above it. Regular snippets are
# versioned service configuration; no custom VCL is needed, and
# Fastly's generated VCL (including the cache boilerplate the
# recipe relies on) stays in charge of everything else.
#
# Only the first snippet is required, and only for purge-group
# mapping. The rest are optional and marked as such. See
# README.md for when each one applies.

# ----------------------------------------------------------------
# Snippet: crabcrop-surrogate-keys
# Type: fetch
# Required for purge-group mapping (README step 5).
#
# crabcrop sends its cache groups in the Cache-Group response
# header (RFC 9875): a list of quoted strings, e.g.
#   Cache-Group: "hero", "product-42"
# Fastly purges groups by surrogate key, which wants bare tokens
# separated by spaces. Copy the groups across, unquoted.
declare local var.keys STRING;

if (beresp.http.Cache-Group) {
  set var.keys = beresp.http.Cache-Group;
  set var.keys = regsuball(var.keys, "\"", "");
  set var.keys = regsuball(var.keys, ",", " ");
  set var.keys = regsuball(var.keys, " +", " ");
  set var.keys = regsub(var.keys, "^ ", "");
  set var.keys = regsub(var.keys, " $", "");
  if (var.keys != "") {
    set beresp.http.Surrogate-Key = var.keys;
  }
}

# ----------------------------------------------------------------
# Snippet: crabcrop-normalize-accept
# Type: recv
# Optional: only when formats are not pinned in the URL
# (README step 3).
#
# Rewrite Accept to the single token the origin's own negotiation
# would pick, so the many spellings real clients send collapse
# into one cache variant per format. The rules mirror the
# server's negotiate_format: among image/avif, image/jxl, and
# image/webp the highest q-value wins, ties break AVIF, then
# JPEG XL, then WebP, a format named twice is governed by its
# first entry, a format whose q does not parse counts as q=1,
# q=0 is a refusal, and wildcards are not an offer.
#
# Two kinds of request pass through untouched: meta requests
# (Accept asks for application/json, which is how crabcrop knows
# to answer with JSON) and requests offering no negotiable
# format, where the origin serves the source format.
declare local var.accept STRING;
declare local var.params STRING;
declare local var.qstr STRING;
declare local var.q FLOAT;
declare local var.best STRING;
declare local var.bestq FLOAT;

if (req.http.Accept) {
  set var.accept = ", " + std.tolower(req.http.Accept) + ",";
  set var.best = "";
  set var.bestq = 0.0;
  if (var.accept !~ "application/json") {
    # image/avif
    set var.params = regsub(var.accept, "^.*?,\s*image/avif\s*(;[^,]*)?,.*$", "\1");
    if (var.params != var.accept) {
      set var.q = 1.0;
      if (var.params ~ ";\s*q=") {
        set var.qstr = regsub(var.params, "^.*;\s*q=\s*([0-9.]+).*$", "\1");
        if (var.qstr != var.params) {
          set var.q = std.atof(var.qstr);
        }
      }
      if (var.q > var.bestq) {
        set var.bestq = var.q;
        set var.best = "image/avif";
      }
    }
    # image/jxl
    set var.params = regsub(var.accept, "^.*?,\s*image/jxl\s*(;[^,]*)?,.*$", "\1");
    if (var.params != var.accept) {
      set var.q = 1.0;
      if (var.params ~ ";\s*q=") {
        set var.qstr = regsub(var.params, "^.*;\s*q=\s*([0-9.]+).*$", "\1");
        if (var.qstr != var.params) {
          set var.q = std.atof(var.qstr);
        }
      }
      if (var.q > var.bestq) {
        set var.bestq = var.q;
        set var.best = "image/jxl";
      }
    }
    # image/webp
    set var.params = regsub(var.accept, "^.*?,\s*image/webp\s*(;[^,]*)?,.*$", "\1");
    if (var.params != var.accept) {
      set var.q = 1.0;
      if (var.params ~ ";\s*q=") {
        set var.qstr = regsub(var.params, "^.*;\s*q=\s*([0-9.]+).*$", "\1");
        if (var.qstr != var.params) {
          set var.q = std.atof(var.qstr);
        }
      }
      if (var.q > var.bestq) {
        set var.bestq = var.q;
        set var.best = "image/webp";
      }
    }
    if (var.best != "") {
      set req.http.Accept = var.best;
    }
  }
}

# ----------------------------------------------------------------
# Snippet: crabcrop-strip-query-string
# Type: recv
# Optional hardening (README step 1).
#
# A crabcrop URL carries everything in its path; a query string
# can only fragment the cache (the default cache key includes
# the full URL, query string and all). Drop it before the
# lookup. The origin never reads one.
set req.url = req.url.path;

# ----------------------------------------------------------------
# Snippet: crabcrop-expiry-ttl-cap
# Type: fetch
# Optional: only when signed expiry must bite at the edge
# (README step 4).
#
# Cap the edge TTL of responses to URLs carrying an exp:
# segment, so a cached copy cannot outlive the signed lifetime
# by much. Set the cap no higher than the shortest lifetime you
# mint. The pattern can also match an exp: segment inside a
# source path; the cost of a false hit is a shorter TTL, never
# a wrong object.
if (req.url ~ "/exp:[0-9]" && beresp.ttl > 300s) {
  set beresp.ttl = 300s;
}
