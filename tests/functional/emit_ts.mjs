// Parity emitter (repo TypeScript SDK, built dist). Reads cases JSON
// from argv[2], prints {id: output}. Mirrors emit_py.py and parity-rs.
import { readFileSync } from "node:fs";

const dist = new URL("../../sdks/typescript/dist/index.js", import.meta.url);
const {
  buildURL, buildSrcSet, sign, purgeSignature, purgeGroupSignature,
  uploadSignature,
} = await import(dist);

const cases = JSON.parse(readFileSync(process.argv[2], "utf8"));
const out = {};
for (const c of cases) {
  let v;
  try {
    switch (c.kind) {
      case "build_url":
        v = buildURL(c.base, c.secret ?? null, c.operations, c.source, {
          expires: c.expires ?? undefined,
          keyId: c.key_id ?? undefined,
        });
        break;
      case "srcset":
        v = buildSrcSet(c.base, c.secret ?? null, c.source, c.widths, {
          expires: c.expires ?? undefined,
        });
        break;
      case "sign":
        v = sign(c.secret, c.remainder);
        break;
      case "purge_sig":
        v = purgeSignature(c.secret, c.target);
        break;
      case "group_sig":
        v = purgeGroupSignature(c.secret, c.group);
        break;
      case "upload_sig":
        v = uploadSignature(c.secret, c.name, c.expires ?? undefined);
        break;
      default:
        v = `ERROR:unknown kind ${c.kind}`;
    }
  } catch (e) {
    v = `ERROR:${e.constructor.name}:${e.message}`;
  }
  out[c.id] = v;
}
console.log(JSON.stringify(out));
