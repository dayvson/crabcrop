// Wave 4 parity emitter: reads parity cases (JSON) from the file named
// in argv[1], prints {id: output} JSON to stdout. Mirrors emit_py.py and
// emit_ts.mjs exactly; the driver byte-compares the three maps.
use crabcrop_url::*;
use serde_json::{json, Map, Value};

fn s(v: &Value) -> &str {
    v.as_str().unwrap_or("")
}

fn opt_u64(v: &Value) -> Option<u64> {
    if v.is_null() {
        None
    } else {
        v.as_u64()
    }
}

fn main() {
    let path = std::env::args().nth(1).expect("cases path");
    let text = std::fs::read_to_string(path).unwrap();
    let cases: Vec<Value> = serde_json::from_str(&text).unwrap();
    let mut out = Map::new();
    for c in &cases {
        let id = s(&c["id"]).to_string();
        let base = s(&c["base"]);
        let secret: Option<&[u8]> = if c["secret"].is_null() {
            None
        } else {
            Some(s(&c["secret"]).as_bytes())
        };
        let val = match s(&c["kind"]) {
            "build_url" => {
                let exp = opt_u64(&c["expires"]);
                let r = if c["key_id"].is_null() {
                    build_url(base, secret, s(&c["operations"]), s(&c["source"]), exp)
                } else {
                    let sk = secret.map(|b| (b, s(&c["key_id"])));
                    build_url_with_key_id(base, sk, s(&c["operations"]), s(&c["source"]), exp)
                };
                match r {
                    Ok(u) => u,
                    Err(e) => format!("ERROR:{e:?}"),
                }
            }
            "srcset" => {
                let widths: Vec<u32> = c["widths"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|w| w.as_u64().unwrap() as u32)
                    .collect();
                match build_srcset(base, secret, s(&c["source"]), &widths, opt_u64(&c["expires"])) {
                    Ok(u) => u,
                    Err(e) => format!("ERROR:{e:?}"),
                }
            }
            "sign" => sign(secret.unwrap(), s(&c["remainder"])),
            "purge_sig" => purge_signature(secret.unwrap(), s(&c["target"])),
            "group_sig" => purge_group_signature(secret.unwrap(), s(&c["group"])),
            "upload_sig" => match opt_u64(&c["expires"]) {
                Some(e) => upload_signature_with_expiry(secret.unwrap(), s(&c["name"]), e),
                None => upload_signature(secret.unwrap(), s(&c["name"])),
            },
            other => format!("ERROR:unknown kind {other}"),
        };
        out.insert(id, json!(val));
    }
    println!("{}", Value::Object(out));
}
