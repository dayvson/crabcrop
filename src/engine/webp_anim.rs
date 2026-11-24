//! Animated WebP frames: libwebp decodes each frame's bitstream
//! and this module composites them (issue #646).
//!
//! The image crate's animated WebP decoder (image-webp) spends
//! most of a decode in its own VP8 decoder, which the still path
//! left for libwebp long ago (see `decode_webp_direct`: about
//! 3.5x faster, the same pixels). Here the same swap is made for
//! the frames of an animation, one frame at a time: each ANMF
//! chunk's bitstream is handed to libwebp as a still image,
//! through the `webp` crate, and the canvas rules are a port of
//! image-webp 0.2.4's (`extended::composite_frame` and
//! `alpha_blending`), so every frame comes out byte for byte
//! what the image crate's frame iterator hands back. On a
//! 12-frame 800x533 animation the decode takes 46 ms where the
//! iterator takes 98 ms.
//!
//! "Byte for byte" includes three things image-webp does that
//! libwebp's own animation decoder does not, kept on purpose so
//! that no output changes:
//!
//! - a frame with alpha is blended through the blend formula for
//!   every pixel, opaque ones too, and the formula takes one off
//!   each non-zero colour channel of an opaque pixel;
//! - the canvas starts as the ANIM background colour, and a
//!   disposed frame's rectangle is filled with it;
//! - when the frame after a disposed one has no alpha, that fill
//!   walks the canvas three bytes at a time (see [`clear`]).
//!
//! Whether to render as libwebp does instead is a rendering
//! decision with a cache epoch of its own (issue #339).
//!
//! Anything this module does not recognize ends the walk with
//! [`Stop::Unsupported`], and the caller runs the image crate's
//! decoder from the start, so an unusual or broken file gets the
//! frames or the error it always got. That covers the container:
//! every chunk name, size and field around the frames'
//! compressed data, and a frame libwebp refuses. Damage inside
//! the compressed data is a narrower promise, because two
//! decoders read it: where libwebp decodes a damaged frame that
//! image-webp would have refused, or reads other pixels out of
//! the damage, the answer is libwebp's, as it is for a still
//! WebP in the same state.
//!
//! The limits are the caller's and unchanged: the caller charges
//! the canvas and the frame count from the headers before any
//! frame decodes (`super::check_webp_frame_budget`, the WebP
//! counterpart of `check_gif_frame_budget`), the image crate's
//! decoder opens the file, with the decode limits, and
//! `collect_frames` charges the frame cap, the pixel and
//! allocation budgets and the deadline on the frames this walk
//! yields exactly as on the iterator's. The walk itself holds
//! one canvas, refused past the decode allocation budget, and
//! one decoded frame no larger than it: a stream that codes
//! another size than its frame is refused on its header, before
//! libwebp decodes any of it. While libwebp decodes a lossless
//! frame it also keeps a buffer of that frame's size of its own
//! beside the frame it hands back. The decoded frame is freed
//! once it is on the canvas, before the canvas is copied out
//! for the frame, and the last frame takes the canvas itself,
//! so the walk holds at most two canvases more than the
//! animation has frames, as the charge allows for.

/// Why [`Animation::next_frame`] produced no frame.
pub(super) enum Stop {
    /// Every frame has been produced.
    Done,
    /// The file is not one this walk handles, or libwebp refused
    /// a frame: the image crate's decoder takes the file from the
    /// start.
    Unsupported,
}

/// One ANMF chunk's header and the bitstream chunks it carries.
struct FrameRef<'a> {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    duration: u32,
    blend: bool,
    dispose: bool,
    bitstream: Bitstream<'a>,
}

enum Bitstream<'a> {
    /// A `VP8 ` chunk's payload: lossy, no alpha.
    Lossy(&'a [u8]),
    /// A `VP8L` chunk's payload.
    Lossless(&'a [u8]),
    /// An `ALPH` chunk's payload and the `VP8 ` payload after it.
    LossyAlpha(&'a [u8], &'a [u8]),
}

/// A rectangle on the canvas: x, y, width, height.
type Rect = (u32, u32, u32, u32);

/// The walk over an animated WebP's frames, holding the canvas.
pub(super) struct Animation<'a> {
    bytes: &'a [u8],
    width: u32,
    height: u32,
    /// The VP8X alpha flag: without it every frame is handed
    /// back opaque, as the image crate does.
    has_alpha: bool,
    background: Option<[u8; 4]>,
    frames: u32,
    produced: u32,
    /// Where the next ANMF chunk's header starts.
    next: usize,
    canvas: Option<Vec<u8>>,
    dispose_next: bool,
    previous: Rect,
}

fn u24(bytes: &[u8], at: usize) -> Option<u32> {
    let b = bytes.get(at..at.checked_add(3)?)?;
    Some(u32::from(b[0]) | u32::from(b[1]) << 8 | u32::from(b[2]) << 16)
}

fn u32le(bytes: &[u8], at: usize) -> Option<u32> {
    let b = bytes.get(at..at.checked_add(4)?)?;
    Some(u32::from_le_bytes(b.try_into().ok()?))
}

/// A chunk header at `at`: its fourcc, payload size, and the size
/// rounded up to the even byte the next chunk starts on.
fn chunk_header(bytes: &[u8], at: usize) -> Option<([u8; 4], usize, usize)> {
    let fourcc: [u8; 4] = bytes.get(at..at.checked_add(4)?)?.try_into().ok()?;
    let size = usize::try_from(u32le(bytes, at.checked_add(4)?)?).ok()?;
    Some((fourcc, size, size.checked_add(size & 1)?))
}

/// An animated WebP's canvas size, alpha flag, ANMF frame count,
/// and the offset of its first ANMF chunk, read the way
/// image-webp reads them (`read_data`). `None` when the file is
/// not a plain animated one. The engine's budget charge
/// ([`super::check_webp_frame_budget`]) and [`Animation::open`]
/// read the canvas and the frame count through here, so the
/// pixels charged before a decode are the pixels the decoder
/// allocates.
///
/// image-webp reads the ten bytes of the VP8X chunk's fields
/// whatever the chunk's size field states, and finds the next
/// chunk by that size, so this reads them the same way; a chunk
/// that states fewer than ten bytes is marked (`short_vp8x`),
/// and the walk leaves such a file to the image crate.
pub(super) struct Header {
    pub(super) width: u32,
    pub(super) height: u32,
    has_alpha: bool,
    pub(super) frames: u32,
    first_anmf: Option<usize>,
    short_vp8x: bool,
}

pub(super) fn header(bytes: &[u8]) -> Option<Header> {
    if bytes.get(..4)? != b"RIFF" || bytes.get(8..12)? != b"WEBP" {
        return None;
    }
    let riff_size = usize::try_from(u32le(bytes, 4)?).ok()?;
    let (first, size, rounded) = chunk_header(bytes, 12)?;
    if &first != b"VP8X" {
        return None;
    }
    let flags = *bytes.get(20)?;
    let has_alpha = flags & 0b0001_0000 != 0;
    if flags & 0b0000_0010 == 0 {
        return None;
    }
    let width = u24(bytes, 24)? + 1;
    let height = u24(bytes, 27)? + 1;
    // The ANMF chunks, counted the way image-webp counts them:
    // every chunk after VP8X, up to the RIFF size past it or the
    // end of the bytes.
    let mut at = 20usize.checked_add(rounded)?;
    let end = at.checked_add(riff_size.saturating_sub(12))?;
    let mut first_anmf = None;
    let mut frames = 0u32;
    while at < end {
        let Some((fourcc, _, rounded)) = chunk_header(bytes, at) else {
            break;
        };
        if &fourcc == b"ANMF" {
            first_anmf.get_or_insert(at);
            frames = frames.checked_add(1)?;
        }
        at = at.checked_add(8)?.checked_add(rounded)?;
    }
    Some(Header {
        width,
        height,
        has_alpha,
        frames,
        first_anmf,
        short_vp8x: size < 10,
    })
}

impl<'a> Animation<'a> {
    /// Start the walk over a file the image crate's decoder has
    /// already opened, so the container has passed its checks,
    /// which this repeats none of. `background` is the colour
    /// set on that decoder. The canvas size, the alpha flag, the
    /// first frame and the frame count are read by [`header`],
    /// the way image-webp reads them. `None` when the file is not
    /// a plain animated one, which a VP8X chunk that states fewer
    /// than its ten bytes is not.
    pub(super) fn open(bytes: &'a [u8], background: Option<[u8; 4]>) -> Option<Self> {
        let Header {
            width,
            height,
            has_alpha,
            frames,
            first_anmf,
            short_vp8x,
        } = header(bytes)?;
        if short_vp8x {
            return None;
        }
        // The canvas must fit the allocation budget the frame
        // collector charges each frame to. The engine charges
        // this from the header before any frame decodes
        // ([`super::check_webp_frame_budget`]), so an over-budget
        // canvas is refused there and never reaches the walk;
        // this guard keeps the walk from allocating one should it
        // ever be opened directly.
        let canvas_bytes = u64::from(width) * u64::from(height) * 4;
        if canvas_bytes > super::MAX_DECODE_ALLOC {
            return None;
        }
        Some(Self {
            bytes,
            width,
            height,
            has_alpha,
            background,
            frames,
            produced: 0,
            next: first_anmf?,
            canvas: None,
            dispose_next: true,
            previous: (0, 0, 0, 0),
        })
    }

    /// Read the ANMF chunk at `self.next`, with the checks
    /// image-webp's `read_frame` makes before it decodes, and
    /// where the chunk after it starts. `None` for anything that
    /// decoder refuses, and for a chunk that does not lie whole
    /// inside the file.
    fn frame_ref(&self) -> Option<(FrameRef<'a>, usize)> {
        let bytes = self.bytes;
        let (fourcc, anmf_size, _) = chunk_header(bytes, self.next)?;
        if &fourcc != b"ANMF" || anmf_size < 32 {
            return None;
        }
        let body = self.next.checked_add(8)?;
        let anmf = bytes.get(body..body.checked_add(anmf_size)?)?;
        let x = u24(anmf, 0)? * 2;
        let y = u24(anmf, 3)? * 2;
        let width = u24(anmf, 6)? + 1;
        let height = u24(anmf, 9)? + 1;
        if width > 16384 || height > 16384 {
            return None;
        }
        if x + width > self.width || y + height > self.height {
            return None;
        }
        let duration = u24(anmf, 12)?;
        let info = *anmf.get(15)?;
        let (fourcc, size, rounded) = chunk_header(anmf, 16)?;
        if rounded.checked_add(24)? > anmf_size {
            return None;
        }
        let payload = anmf.get(24..24 + size)?;
        let bitstream = match &fourcc {
            b"VP8 " => Bitstream::Lossy(payload),
            b"VP8L" => Bitstream::Lossless(payload),
            b"ALPH" => {
                if rounded.checked_add(32)? > anmf_size {
                    return None;
                }
                let at = 24 + rounded;
                let (fourcc, lossy_size, _) = chunk_header(anmf, at)?;
                if &fourcc != b"VP8 " || size.checked_add(lossy_size)?.checked_add(32)? > anmf_size
                {
                    return None;
                }
                let lossy = anmf.get(at + 8..(at + 8).checked_add(lossy_size)?)?;
                Bitstream::LossyAlpha(payload, lossy)
            }
            _ => return None,
        };
        let frame = FrameRef {
            x,
            y,
            width,
            height,
            duration,
            blend: info & 0b10 == 0,
            dispose: info & 0b01 != 0,
            bitstream,
        };
        // image-webp steps by the unrounded size.
        Some((frame, body + anmf_size))
    }

    /// Decode and composite the next frame, and hand back the
    /// canvas as the image crate's frame iterator does.
    pub(super) fn next_frame(&mut self) -> Result<image::Frame, Stop> {
        if self.produced == self.frames {
            return Err(Stop::Done);
        }
        let (frame, after) = self.frame_ref().ok_or(Stop::Unsupported)?;
        let (pixels, frame_has_alpha) = decode_bitstream(&frame).ok_or(Stop::Unsupported)?;
        let clear_color = if self.dispose_next {
            self.background
        } else {
            None
        };
        let (width, height) = (self.width, self.height);
        let background = self.background;
        let canvas = self.canvas.get_or_insert_with(|| {
            let mut canvas = vec![0u8; width as usize * height as usize * 4];
            if let Some(color) = background {
                for pixel in canvas.as_chunks_mut::<4>().0 {
                    *pixel = color;
                }
            }
            canvas
        });
        composite(
            canvas,
            (width, height),
            clear_color,
            &pixels,
            &frame,
            frame_has_alpha,
            self.previous,
        );
        // The frame is on the canvas: its decoded pixels go before
        // the canvas is copied out.
        drop(pixels);
        self.previous = (frame.x, frame.y, frame.width, frame.height);
        self.dispose_next = frame.dispose;
        self.next = after;
        self.produced += 1;
        // Nothing composites onto the canvas after the last frame,
        // so that frame takes the canvas itself.
        let mut out = if self.produced == self.frames {
            std::mem::take(canvas)
        } else {
            canvas.clone()
        };
        if !self.has_alpha {
            // The image crate reads such a file as RGB and widens
            // it: the canvas's own alpha never shows.
            for pixel in out.as_chunks_mut::<4>().0 {
                pixel[3] = 255;
            }
        }
        let buffer = image::RgbaImage::from_raw(width, height, out).ok_or(Stop::Unsupported)?;
        Ok(image::Frame::from_parts(
            buffer,
            0,
            0,
            image::Delay::from_numer_denom_ms(frame.duration, 1),
        ))
    }
}

fn push_chunk(out: &mut Vec<u8>, fourcc: &[u8; 4], payload: &[u8]) {
    out.extend_from_slice(fourcc);
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    if payload.len() & 1 == 1 {
        out.push(0);
    }
}

/// Whether a `VP8 ` payload's frame header sets the colour space
/// bit: the first bit of the first partition, which follows the
/// ten bytes of the frame tag, the start code and the size.
/// image-webp refuses such a frame and libwebp ignores the bit,
/// so the walk leaves it to image-webp.
fn other_color_space(vp8: &[u8]) -> bool {
    vp8.get(10).is_some_and(|first| first & 0x80 != 0)
}

/// A still WebP file around `chunks`, for the `webp` crate.
fn still_file(chunks: &[(&[u8; 4], &[u8])]) -> Option<Vec<u8>> {
    let mut file = b"RIFF\0\0\0\0WEBP".to_vec();
    for (fourcc, payload) in chunks {
        push_chunk(&mut file, fourcc, payload);
    }
    let riff_size = u32::try_from(file.len() - 8).ok()?;
    file[4..8].copy_from_slice(&riff_size.to_le_bytes());
    Some(file)
}

/// Where the stream starts in [`lossless_still_file`]: after
/// the RIFF header and the `VP8L` chunk header.
const LOSSLESS_STREAM_AT: usize = 12 + 8;

/// A still WebP file that ends with the lossless stream `vp8l`:
/// its `VP8L` chunk is the last thing in the file, with no
/// padding byte after a stream of odd length (see
/// [`still_of`]). libwebp's still decoder takes such a file;
/// only a demuxer asks for the padding.
fn lossless_still_file(vp8l: &[u8]) -> Option<Vec<u8>> {
    let mut file = Vec::with_capacity(LOSSLESS_STREAM_AT + vp8l.len());
    file.extend_from_slice(b"RIFF\0\0\0\0WEBPVP8L");
    file.extend_from_slice(&u32::try_from(vp8l.len()).ok()?.to_le_bytes());
    file.extend_from_slice(vp8l);
    let riff_size = u32::try_from(file.len() - 8).ok()?;
    file[4..8].copy_from_slice(&riff_size.to_le_bytes());
    Some(file)
}

/// The still WebP file that carries one frame's bitstream to
/// libwebp, and whether the frame's pixels carry alpha, in
/// image-webp's terms: RGB for a lossy frame, RGBA for a
/// lossless one or a lossy one with an ALPH chunk. `None` for a
/// frame image-webp refuses on its header, and for a stream
/// that codes another size than the ANMF header states.
///
/// libwebp must not be given more of a stream than image-webp
/// reads, or it could decode a frame cut short that image-webp
/// refuses. Its still decoder takes a bitstream from the chunk
/// header to the end of the bytes it is handed, the padding
/// byte after a chunk of odd size included. For a lossy frame
/// that byte is harmless: image-webp itself lets each
/// partition run one zero byte past its end. A lossless stream
/// gets no such byte from image-webp, so its file ends with
/// the stream's last byte.
///
/// Nor may libwebp decode a stream before its size is known to
/// be the frame's: a lossless stream of a few bytes can code a
/// 16384x16384 image, which libwebp would allocate and decode
/// whole. So the size is read here, from the stream's header,
/// as image-webp reads a lossless stream's before it decodes.
fn still_of(frame: &FrameRef<'_>) -> Option<(Vec<u8>, bool)> {
    let (file, has_alpha) = match frame.bitstream {
        Bitstream::Lossy(vp8) => {
            if other_color_space(vp8) {
                return None;
            }
            (still_file(&[(b"VP8 ", vp8)])?, false)
        }
        Bitstream::Lossless(vp8l) => {
            // image-webp reads a lossless frame's alpha whatever
            // the header's alpha hint says. The `webp` crate asks
            // libwebp for RGBA only when the hint is set, so the
            // copy handed to it has the hint set (bit 28 of the
            // 32 bits after the signature byte). The hint is not
            // part of the coded pixels.
            let mut file = lossless_still_file(vp8l)?;
            *file.get_mut(LOSSLESS_STREAM_AT + 4)? |= 0x10;
            (file, true)
        }
        Bitstream::LossyAlpha(alph, vp8) => {
            if other_color_space(vp8) {
                return None;
            }
            let mut header = [0u8; 10];
            header[0] = 0x10;
            header[4..7].copy_from_slice(&(frame.width - 1).to_le_bytes()[..3]);
            header[7..10].copy_from_slice(&(frame.height - 1).to_le_bytes()[..3]);
            let chunks: [(&[u8; 4], &[u8]); 3] =
                [(b"VP8X", &header), (b"ALPH", alph), (b"VP8 ", vp8)];
            (still_file(&chunks)?, true)
        }
    };
    let stated = webp::BitstreamFeatures::new(&file)?;
    if stated.width() != frame.width
        || stated.height() != frame.height
        || stated.has_alpha() != has_alpha
    {
        return None;
    }
    Some((file, has_alpha))
}

/// Decode one frame's bitstream through libwebp. Returns the
/// pixels and whether they carry alpha (see [`still_of`]).
/// `None` when the frame is not one to hand to libwebp, or
/// libwebp refuses it.
fn decode_bitstream(frame: &FrameRef<'_>) -> Option<(webp::WebPImage, bool)> {
    let (file, has_alpha) = still_of(frame)?;
    let decoded = webp::Decoder::new(&file).decode()?;
    Some((decoded, has_alpha))
}

/// `(1 << 24) / a` for the blended alphas 1 to 255.
const fn scale_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut a = 1;
    while a < 256 {
        table[a] = (1u32 << 24) / a as u32;
        a += 1;
    }
    table
}
const SCALE: [u32; 256] = scale_table();

/// image-webp's `div_by_255`: a division by 255 rounded to the
/// nearest.
const fn div_by_255(v: u32) -> u32 {
    (((v + 0x80) >> 8) + v + 0x80) >> 8
}

/// image-webp's `do_alpha_blending`: `src` over `dst`, neither
/// premultiplied. The two ends of the alpha range are written out
/// (they are what almost every pixel takes): a transparent source
/// leaves the canvas, and an opaque one gives each channel
/// `(c * 255 * 65793) >> 24`, which is `c - 1` for every `c`
/// above zero.
#[inline(always)]
fn blend(src: [u8; 4], dst: &mut [u8; 4]) {
    match src[3] {
        0 => {}
        255 => {
            *dst = [
                src[0].saturating_sub(1),
                src[1].saturating_sub(1),
                src[2].saturating_sub(1),
                255,
            ];
        }
        src_a => {
            let src_a = u32::from(src_a);
            let dst_factor_a = div_by_255(u32::from(dst[3]) * (255 - src_a));
            let blend_a = src_a + dst_factor_a;
            let scale = SCALE[blend_a as usize];
            let channel = |s: u8, d: u8| {
                let unscaled = u32::from(s) * src_a + u32::from(d) * (dst_factor_a & 0xff);
                (unscaled.wrapping_mul(scale) >> 24) as u8
            };
            *dst = [
                channel(src[0], dst[0]),
                channel(src[1], dst[1]),
                channel(src[2], dst[2]),
                blend_a as u8,
            ];
        }
    }
}

/// image-webp's fill of the previous frame's rectangle with the
/// background colour, before a frame that follows a disposed one.
/// Two of its four cases walk the RGBA canvas as if it were RGB,
/// and are kept as they are.
fn clear(
    canvas: &mut [u8],
    canvas_width: usize,
    color: [u8; 4],
    full_size: bool,
    frame_has_alpha: bool,
    (px, py, pw, ph): Rect,
) {
    match (full_size, frame_has_alpha) {
        (true, true) => {
            for pixel in canvas.as_chunks_mut::<4>().0 {
                *pixel = color;
            }
        }
        (true, false) => {
            for pixel in canvas.as_chunks_mut::<3>().0 {
                pixel.copy_from_slice(&color[..3]);
            }
        }
        (false, true) => {
            for y in 0..ph as usize {
                let start = (px as usize + (y + py as usize) * canvas_width) * 4;
                for pixel in canvas[start..start + pw as usize * 4]
                    .as_chunks_mut::<4>()
                    .0
                {
                    *pixel = color;
                }
            }
        }
        (false, false) => {
            for y in 0..ph as usize {
                let start = (px as usize + (y + py as usize) * canvas_width) * 3;
                for pixel in canvas[start..start + pw as usize * 3]
                    .as_chunks_mut::<3>()
                    .0
                {
                    pixel.copy_from_slice(&color[..3]);
                }
            }
        }
    }
}

/// image-webp's `composite_frame`: put a decoded frame on the
/// canvas.
fn composite(
    canvas: &mut [u8],
    (canvas_width, canvas_height): (u32, u32),
    clear_color: Option<[u8; 4]>,
    frame: &[u8],
    at: &FrameRef<'_>,
    frame_has_alpha: bool,
    previous: Rect,
) {
    let full_size =
        at.x == 0 && at.y == 0 && at.width == canvas_width && at.height == canvas_height;
    if full_size && !at.blend {
        if frame_has_alpha {
            canvas.copy_from_slice(frame);
        } else {
            for (input, output) in frame
                .as_chunks::<3>()
                .0
                .iter()
                .zip(canvas.as_chunks_mut::<4>().0)
            {
                *output = [input[0], input[1], input[2], 255];
            }
        }
        return;
    }
    let cw = canvas_width as usize;
    if let Some(color) = clear_color {
        clear(canvas, cw, color, full_size, frame_has_alpha, previous);
    }
    let width = at.width.min(canvas_width.saturating_sub(at.x)) as usize;
    let height = at.height.min(canvas_height.saturating_sub(at.y)) as usize;
    let fw = at.width as usize;
    for y in 0..height {
        let start = (at.x as usize + (y + at.y as usize) * cw) * 4;
        let output = canvas[start..start + width * 4].as_chunks_mut::<4>().0;
        if frame_has_alpha {
            let input = frame[y * fw * 4..][..width * 4].as_chunks::<4>().0;
            if at.blend {
                for (input, output) in input.iter().zip(output) {
                    blend(*input, output);
                }
            } else {
                output.copy_from_slice(input);
            }
        } else {
            let input = frame[y * fw * 3..][..width * 3].as_chunks::<3>().0;
            for (input, output) in input.iter().zip(output) {
                *output = [input[0], input[1], input[2], 255];
            }
        }
    }
}

#[cfg(test)]
mod tests;
