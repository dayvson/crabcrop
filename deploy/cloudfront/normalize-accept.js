// Canonicalize the Accept header so CloudFront's cache key sees one
// of three values instead of every spelling every client sends.
//
// Mirrors crabcrop's own negotiation (negotiate_format on the
// server): among image/avif, image/jxl, and image/webp the highest
// q-value wins, ties break AVIF, then JPEG XL, then WebP, a format
// named twice is governed by its first entry, q=0 is a refusal, and
// wildcards are not an offer.
//
// Attach as a viewer request function on the crabcrop behavior and
// include the Accept header in the cache policy's cache key. The
// function runs before the cache lookup, so the rewritten header is
// both the cache key input and what the origin negotiates from.
//
// Two kinds of request pass through untouched: meta requests
// (Accept asks for application/json, which is how crabcrop knows to
// answer with JSON), and requests offering no negotiable format,
// where the origin serves the source format. Both stay keyed under
// their original header, exactly as the server treats them.
function handler(event) {
  var request = event.request;
  var accept = request.headers.accept;
  if (!accept) {
    return request;
  }
  var value = accept.value.toLowerCase();
  if (value.indexOf('application/json') !== -1) {
    return request;
  }
  var q = { 'image/avif': -1, 'image/jxl': -1, 'image/webp': -1 };
  var parts = value.split(',');
  for (var i = 0; i < parts.length; i++) {
    var seg = parts[i].split(';');
    var media = seg[0].trim();
    if (!(media in q) || q[media] !== -1) {
      continue;
    }
    var qv = 1.0;
    for (var j = 1; j < seg.length; j++) {
      var p = seg[j].trim();
      if (p.indexOf('q=') === 0) {
        qv = parseFloat(p.substring(2));
        if (isNaN(qv)) {
          qv = 0;
        }
      }
    }
    q[media] = qv;
  }
  var order = ['image/avif', 'image/jxl', 'image/webp'];
  var best = null;
  var bestQ = 0;
  for (var k = 0; k < order.length; k++) {
    if (q[order[k]] > bestQ) {
      bestQ = q[order[k]];
      best = order[k];
    }
  }
  if (best) {
    request.headers.accept = { value: best };
  }
  return request;
}
