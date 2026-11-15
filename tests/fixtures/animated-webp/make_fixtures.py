#!/usr/bin/env python3
"""Animated WebP fixtures for the frame-identity tests of issue #646.

    python3 tests/fixtures/animated-webp/make_fixtures.py

Writes the `.webp` files next to this script. `src/engine/webp_anim/tests.rs`
decodes every one of them twice, through the image crate and through libwebp
with the engine's own compositing, and requires the same frames.

The `w*` files are assembled by hand from still bitstreams Pillow encodes, so
every ANMF field (offset, size, duration, blend, dispose), the VP8X alpha flag,
the ANIM background colour and loop count can be set freely. The `p*` files
are animations as a mainstream encoder writes them (libwebp's AnimEncoder,
through Pillow). The pixels come from tests/functional/corpus/large.jpg.

The files are checked in: an encoder upgrade changes their bytes, and the
tests only need them to stay what they are. Needs Pillow and numpy.
"""
import io
import os
import struct

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
PHOTO = Image.open(os.path.join(REPO, "tests", "functional", "corpus", "large.jpg")).convert("RGB")


def crop(i, w, h):
    """A w x h piece of the photo, different for each i."""
    x, y = 300 + 137 * i, 500 + 61 * i
    return PHOTO.crop((x, y, x + 6 * w, y + 6 * h)).resize((w, h), Image.LANCZOS)


def with_alpha(img, kind, i=0):
    w, h = img.size
    yy, xx = np.mgrid[0:h, 0:w]
    if kind == "soft":  # every alpha value appears
        a = ((xx * 255) // max(w - 1, 1) + (yy * 31 + i * 17)) % 256
    elif kind == "binary":  # a transparent hole
        a = np.where(((xx - w / 2) ** 2 + (yy - h / 2) ** 2) < (min(w, h) / 3) ** 2, 0, 255)
    elif kind == "edges":  # opaque with a soft border
        d = np.minimum(np.minimum(xx, w - 1 - xx), np.minimum(yy, h - 1 - yy))
        a = np.clip(d * 40, 0, 255)
    else:
        a = np.full((h, w), 255)
    return Image.fromarray(np.dstack([np.asarray(img), a.astype(np.uint8)]), "RGBA")


def riff_chunks(data):
    assert data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    at, found = 12, []
    while at + 8 <= len(data):
        fourcc, size = data[at:at + 4], struct.unpack("<I", data[at + 4:at + 8])[0]
        found.append((fourcc, data[at + 8:at + 8 + size]))
        at += 8 + size + (size & 1)
    return found


def chunk(fourcc, payload):
    return fourcc + struct.pack("<I", len(payload)) + payload + (b"\0" if len(payload) & 1 else b"")


def bitstream(img, hint=True, **kw):
    """The bitstream chunks of img encoded as a still WebP. With hint=False a
    lossless stream's "alpha is used" header bit is cleared, whatever its
    pixels carry."""
    buf = io.BytesIO()
    img.save(buf, "WEBP", **kw)
    out = b""
    for fourcc, payload in riff_chunks(buf.getvalue()):
        if fourcc == b"VP8L" and not hint:
            payload = payload[:4] + bytes([payload[4] & 0xEF]) + payload[5:]
        if fourcc in (b"ALPH", b"VP8 ", b"VP8L"):
            out += chunk(fourcc, payload)
    return out


def u24(v):
    return struct.pack("<I", v)[:3]


def anmf(x, y, img, dur, blend=True, dispose=False, **kw):
    assert x % 2 == 0 and y % 2 == 0
    w, h = img.size
    flags = (0 if blend else 2) | (1 if dispose else 0)
    header = u24(x // 2) + u24(y // 2) + u24(w - 1) + u24(h - 1) + u24(dur) + bytes([flags])
    return chunk(b"ANMF", header + bitstream(img, **kw))


def animation(name, size, frames, bg=(0, 0, 0, 0), loop=0, alpha_flag=True, before=b"", after=b""):
    """bg is (R, G, B, A); the ANIM chunk stores B, G, R, A. `before` goes
    between VP8X and ANIM, `after` behind the last frame."""
    w, h = size
    vp8x = chunk(b"VP8X", bytes([0x02 | (0x10 if alpha_flag else 0), 0, 0, 0]) + u24(w - 1) + u24(h - 1))
    anim = chunk(b"ANIM", bytes([bg[2], bg[1], bg[0], bg[3]]) + struct.pack("<H", loop))
    body = b"WEBP" + vp8x + before + anim + b"".join(frames) + after
    with open(os.path.join(HERE, name), "wb") as out:
        out.write(b"RIFF" + struct.pack("<I", len(body)) + body)


LOSSY = dict(quality=80, method=4)
LOSSLESS = dict(lossless=True, quality=50, method=3)
C = (72, 54)

# Opaque lossy frames that cover the canvas.
animation("w01_lossy_full_blend.webp", C, [anmf(0, 0, crop(i, *C), 70 + i, **LOSSY) for i in range(3)])
# Lossy frames with every alpha value, blended onto part of the canvas.
animation("w02_lossy_alpha_sub_blend.webp", C, [anmf(0, 0, crop(0, *C), 80, **LOSSY)] + [
    anmf(8 + 2 * i, 10, with_alpha(crop(i, 29, 19), "soft", i), 60, **LOSSY) for i in range(1, 4)])
# Opaque sub-rectangles, each disposed to an opaque background colour.
animation("w03_lossy_sub_dispose_bg_red.webp", C, [anmf(0, 0, crop(0, *C), 80, **LOSSY)] + [
    anmf(4 * i, 2 * i, crop(i, 33, 23), 50, dispose=True, **LOSSY) for i in range(1, 5)], bg=(255, 0, 0, 255))
# Lossless frames with alpha, some disposed to a translucent background.
animation("w04_lossless_alpha_sub_dispose_bg_semi.webp", C, [
    anmf(0, 0, with_alpha(crop(0, *C), "edges"), 80, **LOSSLESS)] + [
    anmf(14, 6 + 4 * i, with_alpha(crop(i, 24, 18), "soft", i), 50, dispose=(i % 2 == 0), **LOSSLESS)
    for i in range(1, 5)], bg=(0, 200, 0, 128))
# The blend flag off, for every kind of frame.
animation("w05_noblend.webp", C, [
    anmf(0, 0, with_alpha(crop(0, *C), "binary"), 80, blend=False, **LOSSY),
    anmf(10, 14, with_alpha(crop(1, 30, 22), "soft"), 80, blend=False, **LOSSY),
    anmf(0, 0, with_alpha(crop(2, *C), "soft"), 80, blend=False, **LOSSLESS),
    anmf(36, 24, crop(3, 22, 14), 80, blend=False, **LOSSY),
    anmf(0, 0, crop(4, *C), 80, blend=False, **LOSSY)], bg=(10, 20, 30, 40))
# A canvas without the VP8X alpha flag: frames come back opaque.
animation("w06_no_alpha_flag.webp", C, [
    anmf(8, 4, with_alpha(crop(0, 44, 32), "soft"), 80, **LOSSY),
    anmf(22, 14, with_alpha(crop(1, 36, 30), "binary"), 80, dispose=True, **LOSSLESS),
    anmf(0, 0, with_alpha(crop(2, 32, 26), "edges"), 80, **LOSSY)], bg=(1, 2, 3, 0), alpha_flag=False)
# Lossless frames without alpha.
animation("w07_lossless_rgb.webp", C, [anmf(0, 0, crop(0, *C), 80, **LOSSLESS)] + [
    anmf(18, 18, crop(i, 26, 18), 40, dispose=(i == 2), **LOSSLESS) for i in range(1, 4)], bg=(255, 255, 255, 255))
# Everything in one file, with a loop count.
animation("w08_mixed.webp", C, [
    anmf(0, 0, crop(0, *C), 30, **LOSSY),
    anmf(4, 4, with_alpha(crop(1, 36, 36), "soft"), 40, dispose=True, **LOSSLESS),
    anmf(22, 8, with_alpha(crop(2, 44, 40), "edges"), 50, **LOSSY),
    anmf(0, 0, crop(3, *C), 60, dispose=True, **LOSSY),
    anmf(0, 0, with_alpha(crop(4, *C), "binary"), 70, **LOSSY),
    anmf(0, 0, with_alpha(crop(5, *C), "soft"), 80, dispose=True, **LOSSLESS),
    anmf(0, 0, crop(6, *C), 90, **LOSSY),
    anmf(10, 10, crop(7, 18, 18), 100, **LOSSLESS)], bg=(255, 255, 255, 255), loop=3)
# One-pixel frames on a 3x3 canvas.
animation("w09_tiny.webp", (3, 3), [
    anmf(0, 0, with_alpha(crop(0, 3, 3), "soft"), 10, **LOSSLESS),
    anmf(2, 2, with_alpha(crop(1, 1, 1), "soft"), 10, **LOSSLESS),
    anmf(0, 2, crop(2, 2, 1), 10, dispose=True, **LOSSY),
    anmf(2, 0, with_alpha(crop(3, 1, 3), "soft", 3), 10, **LOSSY)], bg=(9, 8, 7, 200))
# A first frame that does not cover the canvas: the background shows.
animation("w10_first_frame_small_bg.webp", C, [
    anmf(22, 14, with_alpha(crop(0, 28, 22), "soft"), 80, **LOSSY),
    anmf(0, 0, with_alpha(crop(1, 36, 36), "edges"), 80, **LOSSY),
    anmf(36, 18, crop(2, 36, 36), 80, **LOSSY)], bg=(40, 80, 160, 255))
# Full-canvas frames disposed one after another, with and without alpha.
animation("w11_full_dispose_then_frames.webp", C, [
    anmf(0, 0, crop(0, *C), 80, dispose=True, **LOSSY),
    anmf(0, 0, with_alpha(crop(1, *C), "soft"), 80, dispose=True, **LOSSY),
    anmf(0, 0, crop(2, *C), 80, dispose=True, **LOSSY),
    anmf(8, 8, crop(3, 18, 18), 80, dispose=True, **LOSSY),
    anmf(0, 0, with_alpha(crop(4, *C), "binary"), 80, **LOSSLESS)], bg=(200, 100, 50, 255))
# Odd sizes everywhere.
animation("w12_odd_canvas.webp", (73, 55), [anmf(0, 0, crop(0, 73, 55), 80, **LOSSY)] + [
    anmf(36, 28, with_alpha(crop(i, 37, 27), "soft", i), 33, dispose=(i % 3 == 0),
         **(LOSSY if i % 2 else LOSSLESS)) for i in range(1, 7)], bg=(0, 0, 0, 255))
# Lossless frames whose header says "no alpha" while the pixels carry it.
animation("w13_lossless_alpha_without_the_hint.webp", C, [
    anmf(0, 0, with_alpha(crop(0, *C), "soft"), 80, hint=False, **LOSSLESS),
    anmf(12, 8, with_alpha(crop(1, 40, 30), "edges"), 80, hint=False, dispose=True, **LOSSLESS),
    anmf(30, 20, with_alpha(crop(2, 20, 20), "binary"), 80, hint=False, blend=False, **LOSSLESS)],
    bg=(250, 250, 0, 90))
# The ends of the duration and loop-count ranges.
animation("w14_durations_and_loops.webp", (16, 12), [
    anmf(0, 0, crop(0, 16, 12), 0, **LOSSY),
    anmf(0, 0, crop(1, 16, 12), 1, **LOSSY),
    anmf(2, 2, with_alpha(crop(2, 9, 7), "soft"), 0xFFFFFF, **LOSSY),
    anmf(0, 0, crop(3, 16, 12), 65536, **LOSSLESS)], loop=65535)
# Other chunks around the frames: metadata flagged in VP8X or not, an
# unknown chunk ahead of ANIM, and chunks behind the last frame.
animation("w15_chunks_around_the_frames.webp", C, [
    anmf(0, 0, crop(0, *C), 80, **LOSSY),
    anmf(20, 12, with_alpha(crop(1, 30, 30), "soft"), 80, dispose=True, **LOSSY),
    anmf(0, 0, with_alpha(crop(2, *C), "edges"), 80, **LOSSLESS)],
    bg=(12, 34, 56, 255), before=chunk(b"unkn", b"odd"),
    after=chunk(b"EXIF", b"Exif\0\0not a tiff") + chunk(b"zzzz", b"1234"))

# Animations as a mainstream encoder writes them.
SIZE = (96, 64)
pan = [PHOTO.crop((i * 40, i * 25, i * 40 + 1600, i * 25 + 1067)).resize(SIZE, Image.LANCZOS) for i in range(6)]


def save(name, frames, **kw):
    frames[0].save(os.path.join(HERE, name), save_all=True, append_images=frames[1:], **kw)


save("p01_pillow_lossy.webp", pan, duration=70, quality=80, method=4)
save("p02_pillow_lossless.webp", pan[:3], duration=70, lossless=True, method=3)
save("p03_pillow_mixed_minsize.webp", pan, duration=70, quality=75, method=4, allow_mixed=True, minimize_size=True)
sprite = []
for i in range(6):
    f = Image.new("RGBA", (80, 60), (0, 0, 0, 0))
    f.paste(with_alpha(crop(i, 30, 24), "edges"), (4 + 8 * i, 6 + 5 * i))
    sprite.append(f)
save("p04_pillow_sprite_alpha.webp", sprite, duration=50, quality=80, method=4)
save("p05_pillow_sprite_alpha_lossless.webp", sprite, duration=[50, 60, 70, 80, 90, 100], lossless=True, loop=2)

for name in sorted(os.listdir(HERE)):
    if name.endswith(".webp"):
        print(f"{os.path.getsize(os.path.join(HERE, name)):>9}  {name}")
