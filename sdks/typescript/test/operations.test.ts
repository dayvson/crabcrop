import assert from "node:assert/strict";
import { test } from "node:test";
import { buildURL, filter, joinOps, ops } from "../src/index.ts";

// AP-24: the typed builders compile to the grammar's canonical
// strings, byte for byte. Every expected string below is the
// server's canonical spelling of that operation (the same
// spellings the Rust canonicalization corpus pins).

test("typed segments match the canonical spellings", () => {
  const cases: [string, string][] = [
    [ops.size(300, 200), "300x200"],
    [ops.size(300, 0), "300x0"],
    [ops.size("orig", 200), "origx200"],
    [ops.size(-300, 200), "-300x200"],
    [ops.fitIn(800, 600), "fit-in:800x600"],
    [ops.fitIn(800, 600, true), "fit-in:800x600:adaptive"],
    [ops.crop(1200, 630), "crop:1200x630"],
    [ops.crop(1200, 630, "smart"), "crop:1200x630:smart"],
    [ops.crop(100, 100, "face"), "crop:100x100:face"],
    [ops.crop(100, 100, "top-left"), "crop:100x100:top-left"],
    [ops.crop(100, 100, "bottom-right"), "crop:100x100:bottom-right"],
    [ops.cropRect(10, 20, 110, 220), "crop-rect:10x20:110x220"],
    [ops.focal(10, 20, 110, 220), "focal:10x20:110x220"],
    [ops.pad(100, 100), "pad:100x100"],
    [ops.pad(100, 100, "FF8800"), "pad:100x100:ff8800"],
    [ops.pad(100, 100, "blur"), "pad:100x100:blur"],
    [ops.preset("thumb"), "preset:thumb"],
    [ops.preset("thumb-2_large"), "preset:thumb-2_large"],
    [ops.maxage(60), "maxage:60"],
    [ops.maxage(0), "maxage:0"],
    [ops.maxage(31536000), "maxage:31536000"],
    [ops.debugFocal(), "debug:focal"],
  ];
  for (const [got, want] of cases) {
    assert.equal(got, want);
  }
});

test("typed filters match the canonical spellings", () => {
  const cases: [string, string][] = [
    [filter.grayscale(), "grayscale"],
    [filter.invert(), "invert"],
    [filter.noUpscale(), "no_upscale"],
    [filter.equalize(), "equalize"],
    [filter.removeBg(), "removebg"],
    [filter.keepProfile(), "keep_profile"],
    [filter.sepia(), "sepia"],
    [filter.blurFaces(), "blurfaces"],
    [filter.pixelateFaces(), "pixelatefaces"],
    [filter.blur(2.5), "blur(2.5)"],
    [filter.sharpen(0.5), "sharpen(0.5)"],
    [filter.brighten(-20), "brighten(-20)"],
    [filter.contrast(25), "contrast(25)"],
    [filter.contrast(-100), "contrast(-100)"],
    [filter.hueRotate(90), "huerotate(90)"],
    [filter.saturation(2), "saturation(2)"],
    [filter.saturation(0.5), "saturation(0.5)"],
    [filter.rgb(1, -2, 3), "rgb(1,-2,3)"],
    [
      filter.curve("rgb", [
        [0, 0],
        [255, 255],
      ]),
      "curve(rgb,0,0,255,255)",
    ],
    [
      filter.curve("r", [
        [0, 64],
        [255, 255],
      ]),
      "curve(r,0,64,255,255)",
    ],
    [
      filter.curve("b", [
        [32, 10],
        [224, 245],
      ]),
      "curve(b,32,10,224,245)",
    ],
    [
      filter.curve("g", [
        [0, 0],
        [128, 255],
        [255, 255],
      ]),
      "curve(g,0,0,128,255,255,255)",
    ],
    [
      filter.convolution([0, -1, 0, -1, 5, -1, 0, -1, 0]),
      "convolution(0,-1,0,-1,5,-1,0,-1,0,0)",
    ],
    [
      filter.convolution([0, -1, 0, -1, 5, -1, 0, -1, 0], true),
      "convolution(0,-1,0,-1,5,-1,0,-1,0,1)",
    ],
    [
      filter.convolution(
        [0.0625, 0.125, 0.0625, 0.125, 0.25, 0.125, 0.0625, 0.125, 0.0625],
        1,
      ),
      "convolution(0.0625,0.125,0.0625,0.125,0.25,0.125,0.0625,0.125,0.0625,1)",
    ],
    [
      filter.convolution(
        Array.from({ length: 25 }, (_, i) => (i === 12 ? 1 : 0)),
        0,
      ),
      "convolution(0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0)",
    ],
    [filter.noise(0), "noise(0)"],
    [filter.noise(40), "noise(40)"],
    [filter.noise(100), "noise(100)"],
    [filter.rotate(45.5), "rotate(45.5)"],
    [filter.rotate(90), "rotate(90)"],
    [filter.rotate(-30.25), "rotate(-30.25)"],
    [filter.trim(), "trim(10)"],
    [filter.trim(25), "trim(25)"],
    [filter.quality(80), "quality(80)"],
    [filter.quality(1), "quality(1)"],
    [filter.quality("auto"), "quality(auto)"],
    [filter.format("webp"), "format(webp)"],
    [filter.format("mp4"), "format(mp4)"],
    [filter.format("webm"), "format(webm)"],
    [filter.fill("FF8800"), "fill(ff8800)"],
    [filter.facePad(0.25), "facepad(0.25)"],
    [filter.upscale(2), "upscale(2)"],
    [filter.dpr(2), "dpr(2)"],
    [filter.dpr(1.5), "dpr(1.5)"],
    [filter.maxBytes(100000), "max_bytes(100000)"],
    [filter.radius(24), "radius(24,24,24,24)"],
    [filter.radius([24, 24, 0, 0]), "radius(24,24,0,0)"],
    [filter.border(4, "000000"), "border(4,000000)"],
    [filter.mask("circle"), "mask(circle)"],
    [filter.duotone("1a1a40", "ffd9a0"), "duotone(1a1a40,ffd9a0)"],
    [filter.monochrome("ff8000"), "monochrome(ff8000)"],
    [filter.pixelate(8), "pixelate(8)"],
    [filter.flip("h"), "flip(h)"],
    [
      filter.watermark("https://example.com/wm.png", 10, 10, 80),
      "watermark(https%3A%2F%2Fexample.com%2Fwm.png,10,10,80)",
    ],
    [
      filter.overlay("https://example.com/ov.png", 0, 0, 100, "normal"),
      "overlay(https%3A%2F%2Fexample.com%2Fov.png,0,0,100,normal)",
    ],
    [
      filter.overlay("https://example.com/ov.png", 5, 5, 50, "multiply", 50),
      "overlay(https%3A%2F%2Fexample.com%2Fov.png,5,5,50,multiply,50)",
    ],
    [
      filter.text("Hello, World!", 32, "FF8800", "center"),
      "text(Hello%2C%20World%21,32,ff8800ff,center)",
    ],
    [filter.blurClass("person"), "blurclass(person)"],
    // 32 two-byte characters are exactly the server's
    // 64-byte bound: accepted, and percent-encoded per byte.
    [filter.blurClass("é".repeat(32)), `blurclass(${"%C3%A9".repeat(32)})`],
    [filter.pixelateClass("license plate"), "pixelateclass(license%20plate)"],
    [filter.blurRegion(120, 40, 180, 60), "blur_region(120,40,180,60)"],
    [filter.pixelateRegion(120, 40, 180, 60), "pixelate_region(120,40,180,60)"],
  ];
  for (const [got, want] of cases) {
    assert.equal(got, want);
  }
});

test("a typed pipeline is byte-identical to the string pipeline", () => {
  const typed = joinOps(
    ops.size(300, 200),
    ops.filters(
      filter.hueRotate(90),
      filter.sharpen(0.5),
      filter.saturation(2),
      filter.rgb(1, -2, 3),
      filter.equalize(),
    ),
  );
  assert.equal(
    typed,
    "300x200/filters:huerotate(90):sharpen(0.5):saturation(2):rgb(1,-2,3):equalize",
  );
  const source = "https://example.com/photo.jpg";
  assert.equal(
    buildURL("https://img.example.com", "test-secret", typed, source),
    buildURL(
      "https://img.example.com",
      "test-secret",
      "300x200/filters:huerotate(90):sharpen(0.5):saturation(2):rgb(1,-2,3):equalize",
      source,
    ),
  );
});

test("typed builders reject what the server would reject", () => {
  const bad: (() => string)[] = [
    () => filter.quality(0),
    () => filter.quality(101),
    () => filter.brighten(256),
    () => filter.contrast(101),
    () => filter.contrast(-101),
    () => filter.saturation(256),
    () => filter.blur(0),
    () => filter.rgb(0, 0, 256),
    () => filter.curve("x" as never, [[0, 0], [255, 255]]),
    () => filter.curve("rgb", [[0, 0]]),
    () =>
      filter.curve(
        "rgb",
        Array.from({ length: 17 }, (_, i) => [i * 15, i * 15] as [number, number]),
      ),
    () => filter.curve("rgb", [[0, 0], [0, 255]]),
    () => filter.curve("rgb", [[128, 0], [64, 255]]),
    () => filter.curve("rgb", [[0, 0], [256, 255]]),
    () => filter.curve("rgb", [[0, 0], [255, -1]]),
    () => filter.curve("rgb", [[0, 0], [255, 255.5]]),
    () => filter.convolution([0, 0, 0, 0, 1, 0, 0, 0]),
    () => filter.convolution(Array.from({ length: 26 }, () => 0)),
    () => filter.convolution([0, -1, 0, -1, 1025, -1, 0, -1, 0]),
    () => filter.convolution([0, -1, 0, -1, Number.NaN, -1, 0, -1, 0]),
    () => filter.convolution([0, -1, 0, -1, Infinity, -1, 0, -1, 0]),
    () => filter.convolution([0, -1, 0, -1, 5, -1, 0, -1, 0], 2 as never),
    () => filter.noise(101),
    () => filter.noise(-1),
    () => filter.noise(1.5),
    () => filter.rotate(36001),
    () => ops.size(16384, 10),
    () => ops.size(1.5, 10),
    () => ops.crop(0, 100),
    () => ops.crop(100, 100, "middle" as never),
    () => ops.pad(100, 100, "xyz"),
    () => ops.cropRect(10, 10, 10, 20),
    () => ops.focal(10, 10, 10, 20),
    () => ops.focal(-1, 0, 10, 20),
    () => ops.maxage(-1),
    () => ops.maxage(31536001),
    () => ops.maxage(1.5),
    () => filter.text("", 32, "ff8800", "center"),
    () => filter.text("hi", 32, "ff88", "center"),
    () => filter.watermark("", 0, 0, 50),
    () => filter.watermark("https://x/y.png", 0, 0, 101),
    () => filter.blurClass("bad;name"),
    // 40 two-byte characters are 80 UTF-8 bytes: over the
    // server's 64-byte bound even though the code-unit
    // length is 40.
    () => filter.blurClass("é".repeat(40)),
    () => filter.pixelateClass("é".repeat(33)),
    // Preset names outside the server's configurable
    // charset can only ever answer unknown_preset.
    () => ops.preset("thumb!"),
    () => ops.preset(""),
    () => ops.preset("a/b"),
    () => ops.preset("a:b"),
    () => ops.preset("a b"),
    () => filter.maxBytes(0),
    () => filter.radius(-1),
    () => filter.border(0, "000000"),
  ];
  for (const build of bad) {
    assert.throws(build, Error);
  }
});
