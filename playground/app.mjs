import { buildPlaygroundUrl, explainUrl, verifyUrlSignature } from "./lib.mjs";

const $ = (id) => document.getElementById(id);
const fields = ["base", "secret", "source", "operations", "preset", "expiry"];

const CHIPS = [
  "300x200",
  "fit-in:800x600",
  "crop:1200x630:smart",
  "pad:800x800:blur",
  "filters:grayscale",
  "filters:format(webp)",
  "filters:quality(80)",
  "filters:blurfaces",
];

for (const chip of CHIPS) {
  const button = document.createElement("button");
  button.textContent = chip;
  button.addEventListener("click", () => {
    const input = $("operations");
    input.value = input.value ? `${input.value}/${chip}` : chip;
    void rebuild();
  });
  $("chips").appendChild(button);
}

let builtUrl = "";

async function rebuild() {
  const preset = $("preset").value.trim();
  // Presets lead, matching the framework loaders' convention.
  const operations = [preset && `preset:${preset}`, $("operations").value.trim()]
    .filter(Boolean)
    .join("/");
  const expirySeconds = $("expiry").value;
  builtUrl = await buildPlaygroundUrl({
    base: $("base").value.trim(),
    secret: $("secret").value,
    operations,
    source: $("source").value.trim(),
    expires: expirySeconds ? Math.floor(Date.now() / 1000) + Number(expirySeconds) : undefined,
  });
  $("built").value = builtUrl;
  $("preview").src = builtUrl;
  void measureSizes(builtUrl);
}

async function fetchSize(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(String(resp.status));
  return (await resp.blob()).size;
}

function formatBytes(n) {
  return n >= 1024 ? `${(n / 1024).toFixed(1)} KiB` : `${n} B`;
}

async function measureSizes(url) {
  const line = $("sizes");
  const source = $("source").value.trim();
  try {
    const [original, processed] = await Promise.all([fetchSize(source), fetchSize(url)]);
    const saved = original > 0 ? Math.round((1 - processed / original) * 100) : 0;
    line.textContent = `Original ${formatBytes(original)} -> processed ${formatBytes(processed)} (${saved}% smaller).`;
  } catch {
    line.textContent =
      "Preview loaded above; byte sizes unavailable because the endpoint or the source did not send CORS headers for the size check.";
  }
}

for (const id of fields) {
  $(id).addEventListener("input", () => void rebuild());
}

$("copy").addEventListener("click", async () => {
  await navigator.clipboard.writeText(builtUrl);
  $("copy").textContent = "Copied";
  setTimeout(() => ($("copy").textContent = "Copy URL"), 1200);
});

function renderExplanation(input) {
  const out = explainUrl(input);
  const box = $("explanation");
  box.replaceChildren();
  const addRow = (kind, raw, detail, isError) => {
    const div = document.createElement("div");
    div.className = `seg${isError ? " error" : ""}`;
    const k = document.createElement("div");
    k.className = "kind";
    k.textContent = kind;
    const c = document.createElement("code");
    c.textContent = raw;
    const d = document.createElement("div");
    d.className = "detail";
    d.textContent = detail ?? "";
    div.append(k, c, d);
    box.appendChild(div);
  };
  if (out.signature) {
    addRow("signature", out.signature.value, out.signature.detail, out.signature.kind === "invalid");
  }
  for (const seg of out.segments) {
    addRow(seg.kind, seg.raw, seg.error ?? seg.detail, Boolean(seg.error));
  }
  if (out.source) {
    addRow("source", out.source, "fetched by the server under its loader bounds and source policy");
  }
  for (const error of out.errors) {
    const p = document.createElement("p");
    p.className = "err";
    p.textContent = error;
    box.appendChild(p);
  }
  return out;
}

async function explainAndVerify() {
  const input = $("explain-input").value.trim();
  if (!input) return;
  renderExplanation(input);
  const line = $("verify-line");
  const secret = $("secret").value;
  if (!secret) {
    line.textContent = "Paste the secret in the builder above to verify the signature locally.";
    return;
  }
  const verdict = await verifyUrlSignature(secret, input);
  if (verdict === null) {
    line.textContent = "This URL is unsigned or malformed; nothing to verify.";
  } else {
    line.innerHTML = verdict
      ? '<span class="ok">Signature verifies</span> against the secret in the builder.'
      : '<span class="err">Signature does NOT verify</span>: wrong secret, or the URL was altered after signing.';
  }
}

$("explain-input").addEventListener("input", () => void explainAndVerify());
$("explain-built").addEventListener("click", () => {
  $("explain-input").value = builtUrl;
  void explainAndVerify();
});

void rebuild();
