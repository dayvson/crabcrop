//! Text overlays: rendering a label onto an image with a
//! server-configured font. The engine owns no font; the server
//! builds one renderer at boot from `CRABCROP_FONT_PATH` and
//! lends it to the pipeline, the same arrangement as the AI
//! providers.

use ab_glyph::{Font, FontVec, Glyph, PxScale, ScaleFont, point};
use image::DynamicImage;

use crate::url::Anchor;

/// Draws text onto images. Implemented by the font renderer in
/// production and by fakes in tests.
pub trait TextRenderer: Send + Sync + std::fmt::Debug {
    /// Draw `text` onto a copy of `img`. A renderer checks
    /// `deadline`, the request's processing deadline (`None`
    /// for no deadline), between glyphs, and answers
    /// [`DrawError::Timeout`] once it has passed, since the
    /// work grows with the label's length and size (issue
    /// #606).
    fn draw(
        &self,
        img: &DynamicImage,
        text: &str,
        size: u32,
        color: [u8; 4],
        anchor: Anchor,
        deadline: Option<std::time::Instant>,
    ) -> Result<DynamicImage, DrawError>;
}

/// Why a text overlay was not drawn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DrawError {
    /// The processing deadline passed before the label was
    /// drawn (issue #606). Nothing is returned.
    Timeout,
    /// The renderer failed; the message says why.
    Failed(String),
}

/// Whether `deadline` has passed. `None` never does.
fn passed(deadline: Option<std::time::Instant>) -> bool {
    deadline.is_some_and(|deadline| std::time::Instant::now() >= deadline)
}

/// Text rendering with an ab_glyph font (pure-Rust
/// rasterization; glyphs are drawn at the image's current
/// resolution, so text stays sharp wherever the filter sits in
/// the operation order).
pub struct FontRenderer {
    font: FontVec,
}

impl FontRenderer {
    /// Load a font file (TTF/OTF). Every failure is a plain
    /// message: at boot it becomes the startup diagnostic.
    pub fn load(path: &str) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("cannot read the font {path}: {e}"))?;
        Self::from_bytes(bytes)
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, String> {
        let font =
            FontVec::try_from_vec(bytes).map_err(|e| format!("cannot parse the font: {e}"))?;
        Ok(Self { font })
    }
}

impl std::fmt::Debug for FontRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FontRenderer").finish_non_exhaustive()
    }
}

impl TextRenderer for FontRenderer {
    fn draw(
        &self,
        img: &DynamicImage,
        text: &str,
        size: u32,
        color: [u8; 4],
        anchor: Anchor,
        deadline: Option<std::time::Instant>,
    ) -> Result<DynamicImage, DrawError> {
        let scale = PxScale::from(size as f32);
        let font = self.font.as_scaled(scale);
        // Measure and lay out in one pass: glyphs with their
        // pen positions, advances plus pair kerning.
        let mut glyphs: Vec<(Glyph, f32)> = Vec::new();
        let mut pen = 0.0f32;
        let mut prev: Option<ab_glyph::GlyphId> = None;
        for ch in text.chars() {
            if passed(deadline) {
                return Err(DrawError::Timeout);
            }
            let id = font.glyph_id(ch);
            if let Some(p) = prev {
                pen += font.kern(p, id);
            }
            glyphs.push((
                Glyph {
                    id,
                    scale,
                    position: point(pen, 0.0),
                },
                pen,
            ));
            pen += font.h_advance(id);
            prev = Some(id);
        }
        let text_w = pen.ceil() as i32;
        let ascent = font.ascent();
        let text_h = (ascent - font.descent()).ceil() as i32;
        let (hx, vy) = anchor.axes();
        let pad = ((size / 2) as i32).max(4);
        let (iw, ih) = (img.width() as i32, img.height() as i32);
        let x0 = match hx {
            0 => pad,
            1 => (iw - text_w) / 2,
            _ => iw - text_w - pad,
        };
        let y0 = match vy {
            0 => pad,
            1 => (ih - text_h) / 2,
            _ => ih - text_h - pad,
        };
        let baseline = y0 as f32 + ascent;
        let alpha = color[3] as f32 / 255.0;
        let mut out = img.to_rgba8();
        for (glyph, pen_x) in glyphs {
            if passed(deadline) {
                return Err(DrawError::Timeout);
            }
            let mut glyph = glyph;
            glyph.position = point(x0 as f32 + pen_x, baseline);
            let Some(outlined) = font.outline_glyph(glyph) else {
                continue; // e.g. the space glyph has no outline
            };
            let bounds = outlined.px_bounds();
            let (bx, by) = (bounds.min.x.round() as i32, bounds.min.y.round() as i32);
            // A glyph that lands wholly off the canvas would
            // rasterize for nothing: `draw` visits columns bx
            // to bx + width and rows by to by + height, and the
            // closure below drops every one of them. Skipping
            // it changes no pixel (issue #606).
            let (w, h) = (bounds.width() as i64, bounds.height() as i64);
            let (bx64, by64) = (bx as i64, by as i64);
            if bx64 + w <= 0 || by64 + h <= 0 || bx64 >= iw as i64 || by64 >= ih as i64 {
                continue;
            }
            outlined.draw(|x, y, coverage| {
                let (tx, ty) = (bx + x as i32, by + y as i32);
                if tx < 0 || ty < 0 || tx >= iw || ty >= ih {
                    return;
                }
                let a = coverage * alpha;
                if a <= 0.0 {
                    return;
                }
                let px = out.get_pixel_mut(tx as u32, ty as u32);
                for c in 0..3 {
                    px[c] = (px[c] as f32 * (1.0 - a) + color[c] as f32 * a).round() as u8;
                }
                let sa = px[3] as f32 / 255.0;
                px[3] = ((a + sa * (1.0 - a)) * 255.0).round() as u8;
            });
        }
        Ok(DynamicImage::ImageRgba8(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn renderer() -> FontRenderer {
        FontRenderer::from_bytes(include_bytes!("../tests/fixtures/font.ttf").to_vec()).unwrap()
    }

    fn black(w: u32, h: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(image::RgbImage::from_pixel(w, h, image::Rgb([0, 0, 0])))
    }

    fn brightest(img: &DynamicImage, x0: u32, y0: u32, x1: u32, y1: u32) -> u8 {
        let rgba = img.to_rgba8();
        let mut m = 0u8;
        for y in y0..y1 {
            for x in x0..x1 {
                m = m.max(rgba.get_pixel(x, y)[0]);
            }
        }
        m
    }

    #[test]
    fn text_lands_only_in_its_anchor_region() {
        let r = renderer();
        let out = r
            .draw(
                &black(200, 100),
                "HI",
                24,
                [255, 255, 255, 255],
                Anchor::BottomRight,
                None,
            )
            .unwrap();
        // Something was drawn in the bottom-right...
        assert!(brightest(&out, 100, 50, 200, 100) > 200);
        // ...and nothing anywhere else.
        assert_eq!(brightest(&out, 0, 0, 200, 50), 0);
        assert_eq!(brightest(&out, 0, 50, 100, 100), 0);

        let out = r
            .draw(
                &black(200, 100),
                "HI",
                24,
                [255, 255, 255, 255],
                Anchor::TopLeft,
                None,
            )
            .unwrap();
        assert!(brightest(&out, 0, 0, 100, 50) > 200);
        assert_eq!(brightest(&out, 100, 50, 200, 100), 0);
    }

    #[test]
    fn color_alpha_blends_instead_of_replacing() {
        let r = renderer();
        let out = r
            .draw(
                &black(200, 100),
                "H",
                40,
                [255, 255, 255, 128],
                Anchor::TopLeft,
                None,
            )
            .unwrap();
        let m = brightest(&out, 0, 0, 100, 60);
        assert!((90..=170).contains(&m), "brightest {m}");
    }

    #[test]
    fn whitespace_draws_nothing() {
        let r = renderer();
        let img = black(60, 40);
        let out = r
            .draw(&img, " ", 20, [255, 255, 255, 255], Anchor::Center, None)
            .unwrap();
        assert_eq!(brightest(&out, 0, 0, 60, 40), 0);
    }

    #[test]
    fn drawing_stops_at_the_deadline() {
        // Issue #606: the work grows with the label's length
        // and size, so the renderer checks the deadline
        // between glyphs. A deadline already passed draws
        // nothing; one that passes while a long label is laid
        // out or drawn stops it there.
        let r = renderer();
        let white = [255, 255, 255, 255];
        let past = std::time::Instant::now();
        assert_eq!(
            r.draw(&black(64, 64), "HI", 24, white, Anchor::TopLeft, Some(past)),
            Err(DrawError::Timeout)
        );
        let label = "W".repeat(200_000);
        let started = std::time::Instant::now();
        let deadline = started + std::time::Duration::from_millis(50);
        let drawn = r.draw(
            &black(64, 64),
            &label,
            1000,
            white,
            Anchor::TopLeft,
            Some(deadline),
        );
        assert_eq!(drawn, Err(DrawError::Timeout));
        let late = started.elapsed();
        assert!(late < std::time::Duration::from_secs(2), "{late:?}");
        // A deadline still ahead draws as no deadline does.
        let ahead = std::time::Instant::now() + std::time::Duration::from_secs(600);
        assert_eq!(
            r.draw(
                &black(200, 100),
                "HI",
                24,
                white,
                Anchor::Center,
                Some(ahead)
            )
            .unwrap()
            .to_rgba8()
            .into_raw(),
            r.draw(&black(200, 100), "HI", 24, white, Anchor::Center, None)
                .unwrap()
                .to_rgba8()
                .into_raw()
        );
    }

    /// The draw as it was before glyphs off the canvas were
    /// skipped (issue #606): every glyph rasterizes, and the
    /// closure drops the pixels that miss the canvas.
    fn draw_every_glyph(
        r: &FontRenderer,
        img: &DynamicImage,
        text: &str,
        size: u32,
        color: [u8; 4],
        anchor: Anchor,
    ) -> DynamicImage {
        let scale = PxScale::from(size as f32);
        let font = r.font.as_scaled(scale);
        let mut glyphs: Vec<(Glyph, f32)> = Vec::new();
        let mut pen = 0.0f32;
        let mut prev: Option<ab_glyph::GlyphId> = None;
        for ch in text.chars() {
            let id = font.glyph_id(ch);
            if let Some(p) = prev {
                pen += font.kern(p, id);
            }
            glyphs.push((
                Glyph {
                    id,
                    scale,
                    position: point(pen, 0.0),
                },
                pen,
            ));
            pen += font.h_advance(id);
            prev = Some(id);
        }
        let text_w = pen.ceil() as i32;
        let ascent = font.ascent();
        let text_h = (ascent - font.descent()).ceil() as i32;
        let (hx, vy) = anchor.axes();
        let pad = ((size / 2) as i32).max(4);
        let (iw, ih) = (img.width() as i32, img.height() as i32);
        let x0 = match hx {
            0 => pad,
            1 => (iw - text_w) / 2,
            _ => iw - text_w - pad,
        };
        let y0 = match vy {
            0 => pad,
            1 => (ih - text_h) / 2,
            _ => ih - text_h - pad,
        };
        let baseline = y0 as f32 + ascent;
        let alpha = color[3] as f32 / 255.0;
        let mut out = img.to_rgba8();
        for (glyph, pen_x) in glyphs {
            let mut glyph = glyph;
            glyph.position = point(x0 as f32 + pen_x, baseline);
            let Some(outlined) = font.outline_glyph(glyph) else {
                continue;
            };
            let bounds = outlined.px_bounds();
            let (bx, by) = (bounds.min.x.round() as i32, bounds.min.y.round() as i32);
            outlined.draw(|x, y, coverage| {
                let (tx, ty) = (bx + x as i32, by + y as i32);
                if tx < 0 || ty < 0 || tx >= iw || ty >= ih {
                    return;
                }
                let a = coverage * alpha;
                if a <= 0.0 {
                    return;
                }
                let px = out.get_pixel_mut(tx as u32, ty as u32);
                for c in 0..3 {
                    px[c] = (px[c] as f32 * (1.0 - a) + color[c] as f32 * a).round() as u8;
                }
                let sa = px[3] as f32 / 255.0;
                px[3] = ((a + sa * (1.0 - a)) * 255.0).round() as u8;
            });
        }
        DynamicImage::ImageRgba8(out)
    }

    #[test]
    fn labels_partly_off_the_canvas_draw_the_same_pixels() {
        // Skipping the glyphs that miss the canvas changes no
        // pixel: labels cut by every edge, at every anchor,
        // on canvases narrower and shorter than the label and
        // on a 1x1 canvas, match the draw that rasterized
        // every glyph, byte for byte.
        let r = renderer();
        let gray = |w: u32, h: u32| {
            DynamicImage::ImageRgba8(image::RgbaImage::from_fn(w, h, |x, y| {
                image::Rgba([(x * 7 % 256) as u8, (y * 5 % 256) as u8, 90, 200])
            }))
        };
        let anchors = [
            Anchor::TopLeft,
            Anchor::TopCenter,
            Anchor::TopRight,
            Anchor::CenterLeft,
            Anchor::Center,
            Anchor::CenterRight,
            Anchor::BottomLeft,
            Anchor::BottomCenter,
            Anchor::BottomRight,
        ];
        let labels = ["Sale ends soon", "Hello, world ✓ é", &"Wi".repeat(40)];
        let mut cut = 0;
        for (w, h) in [(200, 100), (37, 300), (1, 1), (600, 40)] {
            let img = gray(w, h);
            for size in [9, 40, 160] {
                for label in labels {
                    for anchor in anchors {
                        for color in [[255, 255, 255, 255], [255, 40, 0, 128]] {
                            let before = draw_every_glyph(&r, &img, label, size, color, anchor);
                            let after = r.draw(&img, label, size, color, anchor, None).unwrap();
                            assert_eq!(
                                after.to_rgba8().into_raw(),
                                before.to_rgba8().into_raw(),
                                "{label:?} at {size} on {w}x{h}, {anchor:?}"
                            );
                            cut += usize::from(before.to_rgba8() != img.to_rgba8());
                        }
                    }
                }
            }
        }
        // The cases draw something, not only blank canvases.
        assert!(cut > 300, "{cut}");
    }
}
