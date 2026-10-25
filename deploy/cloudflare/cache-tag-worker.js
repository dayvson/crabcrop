// Bridge crabcrop's Cache-Group response header (RFC 9875) to the
// Cache-Tag header Cloudflare purges by. Deploy on the crabcrop
// hostname's route only. See README.md step 5.
export default {
  async fetch(request) {
    const response = await fetch(request);
    const group = response.headers.get("Cache-Group");
    if (!group) {
      return response;
    }
    // Cache-Group is a list of quoted strings; Cache-Tag wants bare
    // tokens, comma-separated.
    const tags = group
      .split(",")
      .map((t) => t.trim().replace(/^"|"$/g, ""))
      .filter(Boolean);
    if (tags.length === 0) {
      return response;
    }
    const tagged = new Response(response.body, response);
    tagged.headers.set("Cache-Tag", tags.join(","));
    return tagged;
  },
};
