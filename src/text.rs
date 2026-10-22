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
    fn draw(
        &self,
        img: &DynamicImage,
        text: &str,
        size: u32,
        color: [u8; 4],
        anchor: Anchor,
    ) -> Result<DynamicImage, String>;
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
    ) -> Result<DynamicImage, String> {
        let scale = PxScale::from(size as f32);
        let font = self.font.as_scaled(scale);
        // Measure and lay out in one pass: glyphs with their
        // pen positions, advances plus pair kerning.
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
                continue; // e.g. the space glyph has no outline
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
            .draw(&img, " ", 20, [255, 255, 255, 255], Anchor::Center)
            .unwrap();
        assert_eq!(brightest(&out, 0, 0, 60, 40), 0);
    }
}
