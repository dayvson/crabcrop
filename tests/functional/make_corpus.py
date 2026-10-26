#!/usr/bin/env python3
"""One-time freeze of the campaign corpora into the repo corpus.

Canonical copies come from the wave corpora (shared files are byte-identical
across waves; verified by md5 before this script was written). Three files
are re-encoded to keep the checked-in corpus small; the case tables that
assert on them were checked against the new dimensions first.
"""
import hashlib, json, os, shutil, sys
from PIL import Image

CAMP = os.path.expanduser("~/workspace/spikes/feature-campaign")
DEST = os.path.expanduser("~/workspace/crabcrop/tests/functional/corpus")
os.makedirs(DEST, exist_ok=True)

# file -> (wave dir holding the canonical copy, origin note)
COPY = {
    "dog.jpg": ("wave1", "Unsplash photo-1543466835-00a7907e9de1 (via removebg-test corpus)"),
    "car.jpg": ("wave1", "Unsplash photo-1503376780353-7e6692767b70 (via removebg-test corpus)"),
    "cat.jpg": ("wave1", "Unsplash photo-1514888286974-6c03e2ca1dba (via removebg-test corpus)"),
    "logo.png": ("wave1", "Synthesized locally with PIL: transparent PNG logo"),
    "quadrant.jpg": ("wave1", "Synthesized locally with PIL: four-quadrant test pattern"),
    "small.jpg": ("wave1", "Downscaled crop of dog.jpg (200x150)"),
    "panorama.jpg": ("wave1", "Pexels photo 803224"),
    "screenshot.png": ("wave1", "Synthesized locally with PIL: UI screenshot with sharp text"),
    "person-portrait.jpg": ("wave1", "Unsplash photo-1494790108377-be9c29b29330 (via removebg-test corpus)"),
    "person-fullbody.jpg": ("wave1", "Unsplash photo-1509631179647-0177331693ae (via removebg-test corpus)"),
    "product-shoe.jpg": ("wave1", "Unsplash photo-1542291026-7eec264c27ff (via removebg-test corpus)"),
    "night.webp": ("wave1", "Squarespace CDN blog asset"),
    "bordered.jpg": ("wave1", "Synthesized locally with PIL: photo with a drawn border"),
    "exif-orient.jpg": ("wave1", "Synthesized locally with PIL: JPEG with EXIF orientation 6 and Make tag"),
    "icc.jpg": ("wave1", "Synthesized locally with PIL: JPEG carrying an ICC profile"),
    "alpha-base.png": ("wave2", "Synthesized locally with PIL: half-opaque test base"),
    "base-black.png": ("wave2", "Synthesized locally with PIL: solid black"),
    "base-dark.png": ("wave2", "Synthesized locally with PIL: dark gray"),
    "base-gray64.png": ("wave2", "Synthesized locally with PIL: gray 64"),
    "base-mid.png": ("wave2", "Synthesized locally with PIL: mid gray"),
    "base-rgb.png": ("wave2", "Synthesized locally with PIL: RGB quadrant base"),
    "crab-photo.jpg": ("wave2", "crabcrop website asset ai-skate-before.jpg (Snap the crab photo)"),
    "crab-small.png": ("wave2", "Downscale of crab-cutout.png (200x133)"),
    "crab-cutout.png": ("wave2", "Removebg cutout of crab-photo.jpg, kept at native size: overlay cases sample its pixels"),
    "ov-half-green.png": ("wave2", "Synthesized locally with PIL: half green overlay swatch"),
    "ov-marker.png": ("wave2", "Synthesized locally with PIL: marker overlay swatch"),
    "ov-red.png": ("wave2", "Synthesized locally with PIL: red overlay swatch"),
    "dog-200.jpg": ("wave3", "Downscale of dog.jpg (200x150)"),
    "dog-256.jpg": ("wave3", "Downscale of dog.jpg (256x192)"),
    "dog-480.jpg": ("wave3", "Downscale of dog.jpg (480x360)"),
    "logo-256.png": ("wave3", "Downscale of logo.png (256x160)"),
    "group-faces.jpg": ("wave3", "Synthesized locally with PIL: two frontal faces (portrait + mirror) on one canvas"),
    "gradient.png": ("wave3", "Synthesized locally with PIL: smooth color gradient, no subject"),
    "blue.jpg": ("wave4", "Synthesized locally with PIL: blue photo"),
    "tiny.png": ("wave4", "Synthesized locally with PIL: tiny PNG swatch"),
    "probe.svg": ("wave4", "Synthesized locally: SVG carrying a script tag and an onclick handler"),
}

manifest = []
def record(name, origin, note=None):
    p = os.path.join(DEST, name)
    if name.endswith(".svg"):
        e = {"file": name, "origin": origin, "format": "SVG",
             "width": 200, "height": 100,
             "bytes": os.path.getsize(p), "frames": 1,
             "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]}
        if note:
            e["note"] = note
        manifest.append(e)
        print(f"{name}: SVG 200x100 {e['bytes']}B")
        return
    im = Image.open(p)
    frames = getattr(im, "n_frames", 1)
    e = {"file": name, "origin": origin, "format": im.format,
         "width": im.width, "height": im.height,
         "bytes": os.path.getsize(p), "frames": frames,
         "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]}
    if note:
        e["note"] = note
    manifest.append(e)
    print(f"{name}: {im.format} {im.width}x{im.height} frames={frames} {e['bytes']}B")

for name, (w, origin) in COPY.items():
    shutil.copyfile(f"{CAMP}/{w}/sources/{name}", os.path.join(DEST, name))
    record(name, origin)

# --- large.jpg: recompress at native dimensions first ---
src = Image.open(f"{CAMP}/wave1/sources/large.jpg").convert("RGB")
tmp = os.path.join(DEST, "large.jpg")
src.save(tmp, "JPEG", quality=78, optimize=True)
if os.path.getsize(tmp) > 1_300_000:
    src.resize((4000, 2667), Image.LANCZOS).save(tmp, "JPEG", quality=80, optimize=True)
    record("large.jpg", "Unsplash photo-1503376780353-7e6692767b70 at w=5472 (via wave1 corpus)",
           "downscaled to 4000x2667 for the frozen corpus; large-native case asserts the new dimensions")
else:
    record("large.jpg", "Unsplash photo-1503376780353-7e6692767b70 at w=5472 (via wave1 corpus)",
           "recompressed q78 at native dimensions for the frozen corpus")

# --- anim.gif: same frames and dimensions, adaptive palette re-encode ---
g = Image.open(f"{CAMP}/wave1/sources/anim.gif")
frames, durations = [], []
try:
    while True:
        frames.append(g.convert("RGB"))
        durations.append(g.info.get("duration", 100))
        g.seek(g.tell() + 1)
except EOFError:
    pass
pal = frames[0].quantize(colors=256, method=Image.MEDIANCUT)
qframes = [f.quantize(palette=pal, dither=Image.NONE) for f in frames]
qframes[0].save(os.path.join(DEST, "anim.gif"), save_all=True,
                append_images=qframes[1:], duration=durations, loop=0, optimize=True)
record("anim.gif", "Synthesized locally with PIL: 12-frame animation",
       "re-encoded with a shared adaptive palette at native 480x320 for the frozen corpus")

# --- anim.webp: downscaled to 320x213 (cases assert frame count and type only) ---
w = Image.open(f"{CAMP}/wave1/sources/anim.webp")
frames, durations = [], []
try:
    while True:
        frames.append(w.convert("RGB").resize((320, 213), Image.LANCZOS))
        durations.append(w.info.get("duration", 100))
        w.seek(w.tell() + 1)
except EOFError:
    pass
frames[0].save(os.path.join(DEST, "anim.webp"), save_all=True,
               append_images=frames[1:], duration=durations, loop=0,
               quality=70, method=4)
record("anim.webp", "Synthesized locally with PIL: 12-frame animation",
       "downscaled to 320x213 for the frozen corpus; cases assert frame count and content type only")

# --- PDF fixtures (issue #215): hand-written minimal PDFs ---
def _pdf_doc(page_contents):
    """Assemble a deterministic minimal PDF: Letter pages,
    one Helvetica font object, one content stream per page.
    No timestamps or IDs, so the bytes are stable."""
    kids = []
    bodies = []
    n = 4
    for content in page_contents:
        kids.append(f"{n} 0 R")
        bodies.append((n, n + 1, content))
        n += 2
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(page_contents)} >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for page_n, content_n, content in bodies:
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_n} 0 R /Resources << /Font << /F1 3 0 R >> >> >>")
        objects.append(f"<< /Length {len(content)} >>\nstream\n{content}endstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects):
        offsets.append(len(out))
        out += f"{i + 1} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n").encode()
    return bytes(out)


def _band_page(i, total):
    """One page: a hue band unique to the page (the spike
    formula) plus its label, so a render proves which
    page was selected from the band's center pixel."""
    r = ((i * 37) % 100) / 100
    g = ((i * 53) % 100) / 100
    b = ((i * 91) % 100) / 100
    return (f"{r:.2f} {g:.2f} {b:.2f} rg\n72 560 468 60 re\nf\n"
            f"0 0 0 rg\nBT /F1 24 Tf 72 700 Td (Page {i} of {total}) Tj ET\n")


def record_pdf(name, origin, pages, note=None):
    p = os.path.join(DEST, name)
    e = {"file": name, "origin": origin, "format": "PDF",
         "width": 612, "height": 792,
         "bytes": os.path.getsize(p), "frames": pages,
         "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]}
    if note:
        e["note"] = note
    manifest.append(e)
    print(f"{name}: PDF pages={pages} {e['bytes']}B")


pages_data = _pdf_doc([_band_page(i, 5) for i in range(1, 6)])
open(os.path.join(DEST, "pages.pdf"), "wb").write(pages_data)
record_pdf("pages.pdf",
           "Hand-written minimal PDF: 5 Letter pages, per-page hue band and page label",
           5, "width/height are the page size in points; frames is the page count")
open(os.path.join(DEST, "truncated.pdf"), "wb").write(pages_data[:400])
record_pdf("truncated.pdf",
           "The first 400 bytes of pages.pdf: a truncated document no probe can read",
           0)
# locked.pdf is generated once with pypdf (AES-256 user
# password over a 2-page text document) and frozen into
# the corpus; it is recorded when present, never
# regenerated here.
if os.path.exists(os.path.join(DEST, "locked.pdf")):
    record_pdf("locked.pdf",
               "Generated with pypdf: 2 Letter pages, AES-256 user-password encryption",
               2)

manifest.sort(key=lambda e: e["file"])
json.dump(manifest, open(os.path.join(DEST, "..", "corpus.json"), "w"), indent=1)
total = sum(e["bytes"] for e in manifest)
print(f"\n{len(manifest)} files, total {total} bytes ({total/1048576:.2f} MiB)")
