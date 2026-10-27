#!/usr/bin/env python3
"""Parity emitter (repo Python SDK). Reads cases JSON from argv[1],
prints {id: output}. Mirrors emit_ts.mjs and parity-rs."""
import json, sys, os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "sdks", "python", "src"))
import crabcrop_url as sdk

cases = json.load(open(sys.argv[1]))
out = {}
for c in cases:
    kind = c["kind"]
    secret = c.get("secret")
    try:
        if kind == "build_url":
            v = sdk.build_url(c["base"], secret, c["operations"],
                              c["source"], expires=c.get("expires"),
                              key_id=c.get("key_id"))
        elif kind == "srcset":
            v = sdk.build_srcset(c["base"], secret, c["source"],
                                 c["widths"], expires=c.get("expires"))
        elif kind == "sign":
            v = sdk.sign(secret, c["remainder"])
        elif kind == "seal":
            # Sealed vectors (issue #255): the one emitter
            # output that may legitimately be a skip, when
            # the cryptography extra is not installed. The
            # parity cells treat the sentinel as "compare
            # the other two SDKs", never as a value.
            try:
                v = sdk.seal_with_nonce(secret, c["remainder"],
                                        bytes.fromhex(c["nonce"]),
                                        c.get("key_id"))
            except Exception as e:
                if "cryptography" in str(e):
                    v = "SKIP:no-extra"
                else:
                    raise
        elif kind == "purge_sig":
            v = sdk.purge_signature(secret, c["target"])
        elif kind == "group_sig":
            v = sdk.purge_group_signature(secret, c["group"])
        elif kind == "upload_sig":
            v = sdk.upload_signature(secret, c["name"], c.get("expires"))
        else:
            v = f"ERROR:unknown kind {kind}"
    except Exception as e:
        v = f"ERROR:{type(e).__name__}:{e}"
    out[c["id"]] = v
print(json.dumps(out))
