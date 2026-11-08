// Issue #482: the grammar's closed sets are declared once, in
// explain.ts, and both the explainer and the typed builders read
// them. The unions callers type against stay in operations.ts,
// so `pretest` type-checks this file (tsconfig.test.json): a
// value added to a set but not its union, or the other way
// round, fails the build here.
import assert from "node:assert/strict";
import { test } from "node:test";
import {
  BLEND_MODES,
  CROP_ALIGNS,
  CURVE_CHANNELS,
  DPR_FACTORS,
  FLIP_AXES,
  IMAGE_FORMATS,
  JPEG_SUBSAMPLINGS,
  MASK_SHAPES,
  QUALITY_FORMATS,
  STUDIO_BACKDROPS,
  STUDIO_MODES,
  STUDIO_SHADOWS,
  TEXT_ANCHORS,
  UPSCALE_FACTORS,
} from "../src/explain.ts";
import { explainUrl, filter, ops } from "../src/index.ts";
import type {
  Anchor,
  BlendMode,
  CropAlign,
  CurveChannel,
  FormatOptions,
  ImageFormat,
  MaskShape,
  PerFormatQualityFormat,
  StudioMode,
  StudioShadow,
} from "../src/operations.ts";

type Same<A, B> =
  (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2 ? true : false;
type Member<L extends readonly unknown[]> = L[number];

const SAME_SETS: [
  Same<CropAlign, Member<typeof CROP_ALIGNS>>,
  Same<Anchor, Member<typeof TEXT_ANCHORS>>,
  Same<ImageFormat, Member<typeof IMAGE_FORMATS>>,
  Same<PerFormatQualityFormat, Member<typeof QUALITY_FORMATS>>,
  Same<NonNullable<FormatOptions["subsampling"]>, Member<typeof JPEG_SUBSAMPLINGS>>,
  Same<BlendMode, Member<typeof BLEND_MODES>>,
  Same<CurveChannel, Member<typeof CURVE_CHANNELS>>,
  Same<StudioMode, Member<typeof STUDIO_MODES>>,
  Same<StudioShadow, Member<typeof STUDIO_SHADOWS>>,
  Same<Parameters<typeof filter.upscale>[0], Member<typeof UPSCALE_FACTORS>>,
  Same<Parameters<typeof filter.dpr>[0], Member<typeof DPR_FACTORS>>,
  Same<MaskShape, Member<typeof MASK_SHAPES>>,
  Same<Parameters<typeof filter.flip>[0], Member<typeof FLIP_AXES>>,
] = [true, true, true, true, true, true, true, true, true, true, true, true, true];

const SRC = "https://example.com/a.jpg";

test("each closed set's union type names exactly its members", () => {
  assert.ok(SAME_SETS.every((same) => same));
});

test("every member of every shared set builds a segment the explainer accepts", () => {
  const segments = [
    ...CROP_ALIGNS.map((align) => ops.crop(100, 100, align)),
    ...[
      ...TEXT_ANCHORS.map((anchor) => filter.text("hi", 12, "ffffff", anchor)),
      ...IMAGE_FORMATS.map((format) => filter.format(format)),
      ...QUALITY_FORMATS.map((format) => filter.qualityPerFormat({ [format]: 70 })),
      ...JPEG_SUBSAMPLINGS.map((subsampling) => filter.format("jpeg", { subsampling })),
      ...BLEND_MODES.map((blend) => filter.overlay("https://example.com/o.png", 0, 0, 50, blend)),
      ...CURVE_CHANNELS.map((channel) => filter.curve(channel, [[0, 0], [255, 255]])),
      ...STUDIO_MODES.map((mode) => filter.studio(mode)),
      ...STUDIO_BACKDROPS.map((bg) => filter.studio("product", { bg })),
      ...STUDIO_SHADOWS.map((shadow) => filter.studio("product", { shadow })),
      ...UPSCALE_FACTORS.map((factor) => filter.upscale(factor)),
      ...DPR_FACTORS.map((factor) => filter.dpr(factor)),
      ...MASK_SHAPES.map((shape) => filter.mask(shape)),
      ...FLIP_AXES.map((axis) => filter.flip(axis)),
    ].map((call) => `filters:${call}`),
  ];
  for (const segment of segments) {
    assert.deepEqual(explainUrl(`/unsafe/${segment}/${SRC}`).errors, [], segment);
  }
});

// The builders spell a refused value's rule from the shared set;
// these are the messages they gave when each kept its own list.
test("a value outside a shared set throws the builder's rule", () => {
  const cases: [() => unknown, string][] = [
    [() => ops.crop(100, 100, "middle" as CropAlign), 'crop align must be one of center, smart, face, top, bottom, left, right, top-left, top-right, bottom-left, bottom-right, got "middle"'],
    [() => filter.qualityPerFormat({ mp4: 70 } as Partial<Record<PerFormatQualityFormat, number>>), 'quality must be one of avif, gif, jpeg, jxl, png, webp, got "mp4"'],
    [() => filter.format("jpeg", { subsampling: 421 as 420 }), "subsampling must be one of 444, 422, 420, got 421"],
    [() => filter.upscale(3 as 2), "upscale factor must be 2 or 4, got 3"],
    [() => filter.dpr(1.25 as 1), "dpr must be 1, 1.5, 2, or 3, got 1.25"],
    [() => filter.mask("square" as MaskShape), 'mask shape must be "circle" or "ellipse", got "square"'],
    [() => filter.flip("x" as "h"), 'flip axis must be "h" or "v", got "x"'],
    [() => filter.studio("product", { bg: "blur" }), 'studio bg must be six hex digits (RRGGBB), got "blur"'],
  ];
  for (const [call, message] of cases) assert.throws(call, { message });
});
