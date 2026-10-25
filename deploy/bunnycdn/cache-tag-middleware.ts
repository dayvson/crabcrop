// Bridge crabcrop's Cache-Group response header (RFC 9875) to the
// CDN-Tag header Bunny purges by. Link to the crabcrop Pull Zone as
// a middleware script (Edge Scripting). See README.md step 5.
//
// The bridge runs in onOriginResponse, which Bunny documents as
// modifying the response before it is cached, so the tags are set on
// the copy that gets stored and indexed. No before-cache execution
// is needed: the script only touches origin responses. The url in
// servePullZone is only used for local development; in production
// the Pull Zone's own origin is used.
import * as BunnySDK from "@bunny.net/edgescript-sdk@0.13.0";

BunnySDK.net.http
  .servePullZone({ url: "https://origin.example.com" })
  .onOriginResponse((ctx) => {
    const groups = ctx.response.headers.get("Cache-Group");
    if (!groups) {
      return Promise.resolve(ctx.response);
    }
    // Cache-Group is a list of quoted strings; CDN-Tag wants bare
    // tags, comma-separated.
    const tags = groups
      .split(",")
      .map((t) => t.trim().replace(/^"|"$/g, ""))
      .filter(Boolean);
    if (tags.length === 0) {
      return Promise.resolve(ctx.response);
    }
    const tagged = new Response(ctx.response.body, ctx.response);
    tagged.headers.set("CDN-Tag", tags.join(","));
    return Promise.resolve(tagged);
  });
