//! A JPEG source decoded at reduced scale (issue #653).
//!
//! A JPEG holds its picture as 8x8 blocks of frequency
//! coefficients, and a decoder can invert each block at 4x4,
//! 2x2 or 1x1 instead: the picture at a half, a quarter or an
//! eighth of its size, without ever making the full-size
//! pixels. zune-jpeg, which decodes every JPEG here, has no
//! such mode. libjpeg-turbo, which already writes the JPEG
//! output (issue #668), has one, with vector code for it. So
//! where the first operation of a request is a plain resize
//! that shrinks a JPEG far enough, the source is decoded
//! through libjpeg-turbo at reduced scale and the resize
//! finishes from the reduced frame.
//!
//! The pixels of such a render differ from the ones a
//! full-size decode and the same resize give, so the rule is
//! narrow ([`plan`]): the frame must still be at least one and
//! a half times the resize's target on both axes. A real
//! resample always follows the scaled decode, and a scaled
//! decode never lands on the target itself, which is where the
//! two renders sit furthest apart. Every other request decodes
//! as it did.
//!
//! The reduced frame stands for the image the header declares
//! ([`Reduced`]). That image's size is the source size the
//! response reports, the `no_upscale` cap, and what the resize
//! resolves its target against, so every size a request works
//! out is the one it worked out before. And the resize reads
//! the part of the frame that image fills, to the fraction of
//! a pixel: a frame is rounded up to whole pixels, so its last
//! column and row can stand for less than a full pixel of
//! image, and a resize of the whole frame would stretch the
//! picture and shift it by up to half a pixel.
//!
//! The limits are the full image's too. A file whose declared
//! size is past the decode limits is not eligible, and the
//! full-size path refuses it as it always did, before anything
//! is decoded. A scaled decode holds a frame of a quarter of
//! the pixels or fewer where the full one holds them all, and
//! for a progressive file the same coefficients zune-jpeg
//! holds, which both paths charge with the full frame, so a
//! progressive file past the budget is refused at every size;
//! its scans are capped where zune-jpeg caps them
//! ([`MAX_SCANS`]).
//!
//! Two decoders reading one file is the hazard, and the answer
//! to it is the same everywhere: where libjpeg-turbo and
//! zune-jpeg might not read a file alike, the scaled decode
//! stands down and the full-size path answers, with the pixels
//! or the error it always gave. That covers a header the two
//! read differently, an entropy coding or sample depth
//! zune-jpeg does not read, any file libjpeg-turbo refuses,
//! and any file it warns about (a truncated scan, stray bytes,
//! a restart marker out of turn): the library returns an image
//! with a warning, and here a warning is a refusal. libjpeg's
//! errors come back as values through the `turbojpeg` crate;
//! the library does not print or exit.
//!
//! What no header shows is damage inside the coded data. The
//! one place a differential run found the two decoders
//! parting there is the first scan of a progressive file:
//! zune-jpeg refuses a file whose first scan its reader is
//! done with before it comes to the marker that ends the
//! scan, where libjpeg-turbo reads the scan without a
//! warning, so such a file rendered at the sizes that took
//! this path and failed at full size. A progressive file is
//! therefore taken only where zune-jpeg's reader comes to
//! that marker (issue #713, [`first_scan`]): the check
//! follows the reader through the first scan, a table lookup
//! a block, and leaves whatever it does not follow to the
//! full-size path, whose answer, the picture or the error, is
//! then the answer at every size. A damaged file both
//! decoders read still shows its damage differently on the
//! two paths, as every scaled render's pixels differ a little
//! from the full-size render's.

use image::DynamicImage;

use super::{
    EngineError, ImageSize, MAX_DECODE_ALLOC, Operation, ProcessContext, SizeCap,
    check_decoded_dimensions, exif_orientation, has_zero_side, progressive_coefficient_bytes,
    resize_target, zune_jpeg_options,
};

mod first_scan;

/// The most scans a progressive file may carry. A scan can be
/// a few bytes that make the decoder walk every block of the
/// image, so a small file with thousands of them is hours of
/// work. This is zune-jpeg's own cap (its default, which every
/// JPEG decode here runs under): a file past it is refused by
/// both decoders, so the full-size path gives it the refusal
/// it always got.
const MAX_SCANS: u32 = 100;

/// The scales tried, as denominators, from the smallest frame
/// to the largest. libjpeg-turbo scales by any eighth; these
/// three are the ones it has vector code for, and the others
/// measured no faster than the next of these.
const DENOMINATORS: [u32; 3] = [8, 4, 2];

/// A scaled decode a request is eligible for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ScaledDecode {
    /// The frame is the image at one over this: 2, 4 or 8.
    pub(super) denominator: u32,
    /// The size the file's header declares, as stored.
    pub(super) stored: (u32, u32),
    /// The file's EXIF orientation, which says which stored
    /// axis a target's width falls on. The decode stage
    /// applies it to the frame afterwards.
    pub(super) orientation: Option<u32>,
}

impl ScaledDecode {
    /// The size of the frame the decode makes.
    pub(super) fn frame(&self) -> (u32, u32) {
        (
            scaled_side(self.stored.0, self.denominator),
            scaled_side(self.stored.1, self.denominator),
        )
    }

    /// What the frame stands for once the decode stage has
    /// turned it upright.
    ///
    /// As stored, the image fills the frame from its top left
    /// corner: `stored / denominator` pixels across and down,
    /// which is short of the frame's last column or row
    /// wherever the size is not a multiple of the
    /// denominator. Each EXIF orientation carries that corner
    /// with it.
    pub(super) fn reduced(&self) -> Reduced {
        let scale = f64::from(self.denominator);
        let (w, h) = (
            f64::from(self.stored.0) / scale,
            f64::from(self.stored.1) / scale,
        );
        let (fw, fh) = self.frame();
        let (fw, fh) = (f64::from(fw), f64::from(fh));
        let window = match self.orientation {
            // Mirrored, turned half round, flipped.
            Some(2) => (fw - w, 0.0, w, h),
            Some(3) => (fw - w, fh - h, w, h),
            Some(4) => (0.0, fh - h, w, h),
            // The quarter turns, with and without a mirror:
            // the stored rows become the columns.
            Some(5) => (0.0, 0.0, h, w),
            Some(6) => (fh - h, 0.0, h, w),
            Some(7) => (fh - h, fw - w, h, w),
            Some(8) => (0.0, fw - w, h, w),
            _ => (0.0, 0.0, w, h),
        };
        Reduced {
            full: turn(self.stored, self.orientation),
            window,
        }
    }
}

/// What a reduced frame stands for, upright: the image the
/// file's header declares.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Reduced {
    /// The size of that image: the source's size wherever one
    /// is reported or resolved against.
    pub(super) full: (u32, u32),
    /// The part of the frame the image fills, as (left, top,
    /// width, height) in frame pixels. The rest, less than a
    /// pixel along one or two edges, is the padding that
    /// filled out the file's blocks.
    pub(super) window: (f64, f64, f64, f64),
}

/// A `side` pixel side at one over `denominator`. libjpeg
/// rounds up: a block cut short by the image's edge still
/// leaves a pixel.
fn scaled_side(side: u32, denominator: u32) -> u32 {
    side.div_ceil(denominator)
}

/// Whether a `frame` pixel side is at least one and a half
/// times a `target` pixel side, in whole numbers.
fn leaves_a_resample(frame: u32, target: u32) -> bool {
    u64::from(frame) * 2 >= u64::from(target) * 3
}

/// A size carried between the stored axes and the upright
/// ones, either way: EXIF orientations 5 to 8 turn the image
/// a quarter, so its upright width is its stored height.
fn turn((width, height): (u32, u32), orientation: Option<u32>) -> (u32, u32) {
    if matches!(orientation, Some(5..=8)) {
        (height, width)
    } else {
        (width, height)
    }
}

/// The scaled decode `ops` make `bytes` eligible for, or
/// `None` for the full-size decode every other request gets.
///
/// Eligible is: the source is a YCbCr JPEG ([`eligible_size`]),
/// the first operation is a plain resize, and one of a half, a
/// quarter and an eighth leaves a frame at least one and a half
/// times the resize's target on both axes. The smallest such
/// frame is taken.
///
/// The target is the one the resize itself will resolve, with
/// the same function ([`resize_target`]): against the declared
/// size turned upright by the file's EXIF orientation, a pixel
/// side scaled by `dpr`, and held to the declared size under
/// `no_upscale`. For orientations 5 to 8 the stored frame lies
/// on its side, so the target's width is held against the
/// frame's height. The plan carries the orientation it read,
/// which the decode stage then applies to the frame.
///
/// A request that selects a video frame or a PDF page decodes
/// the PNG its tool wrote, never a JPEG; it is turned away by
/// name all the same. A box with a zero pixel side is refused
/// by the resize, so nothing is gained by decoding for it.
pub(super) fn plan(
    bytes: &[u8],
    ops: &[Operation],
    directives: &crate::url::RequestDirectives,
    dpr: f32,
) -> Option<ScaledDecode> {
    #[cfg(test)]
    if FULL_SIZE_ONLY.get() {
        return None;
    }
    let Some(Operation::Size(size)) = ops.first() else {
        return None;
    };
    if directives.frame.is_some() || directives.page.is_some() || has_zero_side(size) {
        return None;
    }
    // A first look, which allocates nothing, turns most
    // requests down (see [`Glance`]); the two decoders'
    // reading of the header and the orientation as the decode
    // stage reads it then decide.
    let glance = glance(bytes)?;
    denominator_for(
        glance.size,
        glance.orientation,
        size,
        dpr,
        directives.no_upscale,
    )?;
    let stored = eligible_size(bytes, &glance)?;
    let orientation = exif_orientation(bytes);
    let denominator = denominator_for(stored, orientation, size, dpr, directives.no_upscale)?;
    Some(ScaledDecode {
        denominator,
        stored,
        orientation,
    })
}

/// The scale a resize to `size` leaves room for on a `stored`
/// image with this EXIF orientation: the rule of [`plan`],
/// without the file.
fn denominator_for(
    stored: (u32, u32),
    orientation: Option<u32>,
    size: &ImageSize,
    dpr: f32,
    no_upscale: bool,
) -> Option<u32> {
    let (width, height) = turn(stored, orientation);
    let cap = no_upscale.then_some(SizeCap::Axes(width, height));
    let target = resize_target(size, dpr, (width, height), cap);
    // Back onto the stored axes, which the decoder scales.
    let (across, down) = turn(target, orientation);
    DENOMINATORS.into_iter().find(|&denominator| {
        leaves_a_resample(scaled_side(stored.0, denominator), across)
            && leaves_a_resample(scaled_side(stored.1, denominator), down)
    })
}

/// What the scaled decode reads of a file before either
/// decoder does: the size its frame header declares, whether
/// it is progressive, whether its first scan names Huffman
/// tables the file defines ahead of it, and its EXIF
/// orientation, from a walk over the marker segments up to
/// the first scan ([`glance`]).
///
/// The walk allocates nothing, and that is its purpose. Most
/// requests whose first operation is a resize do not shrink
/// the source far enough, and the walk turns them down before
/// either decoder reads the header. Such a request then runs
/// the full-size path with exactly the allocations it always
/// made: a header read by either decoder allocates, and in a
/// loop of renders of a phone photo at 800 wide that alone
/// changed which large buffers the allocator handed back to
/// the system between renders, and doubled the page faults.
/// The walk decides nothing on its own: a file or a request
/// it lets through is judged by the decoders' own reading
/// ([`eligible_size`]), so where it reads a file differently
/// the request decodes at full size, as before.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Glance {
    /// The frame header's width and height, as stored.
    size: (u32, u32),
    progressive: bool,
    /// Whether the first scan holds all three components and
    /// codes them with tables the file defines ahead of it.
    /// libjpeg fills in the standard Huffman tables for a
    /// sequential file that leaves them out, as the frames of
    /// motion JPEG do; zune-jpeg does that only for a file
    /// that says it is such a frame, and refuses any other. A
    /// file only one of the two reads must not render at one
    /// size and fail at another, so a sequential file is
    /// taken only when its first scan names tables that are
    /// there; a first scan of fewer components has more scans
    /// behind it, which are not read here. A progressive file
    /// needs no such care: libjpeg fills nothing in for it,
    /// and refuses a scan that names a missing table.
    names_its_tables: bool,
    /// The orientation tag of the first EXIF segment.
    orientation: Option<u32>,
}

/// The [`Glance`] of `bytes`, or `None` for a file that is not
/// an 8-bit, three-component, Huffman-coded JPEG frame with a
/// size on both axes (baseline, extended or progressive), and
/// for anything in the header the walk does not expect.
fn glance(bytes: &[u8]) -> Option<Glance> {
    if bytes.get(..2)? != [0xFF, 0xD8] {
        return None;
    }
    // The tables defined so far: DC and AC, four slots each.
    let mut defined = [[false; 4]; 2];
    let mut frame = None;
    let mut orientation = None;
    let mut exif_seen = false;
    // Past the start-of-image marker.
    let mut at = 2;
    loop {
        // A marker: 0xFF, any fill bytes of the same, its code,
        // then a segment that opens with its own length.
        if *bytes.get(at)? != 0xFF {
            return None;
        }
        while *bytes.get(at + 1)? == 0xFF {
            at += 1;
        }
        let code = bytes[at + 1];
        let length = usize::from(u16::from_be_bytes([
            *bytes.get(at + 2)?,
            *bytes.get(at + 3)?,
        ]));
        let segment = bytes.get(at + 4..(at + 2).checked_add(length)?)?;
        match code {
            // A frame header the decode takes: 8-bit samples,
            // three components, a size on both axes.
            0xC0..=0xC2 => {
                let &[8, h0, h1, w0, w1, 3, ..] = segment else {
                    return None;
                };
                let width = u32::from(u16::from_be_bytes([w0, w1]));
                let height = u32::from(u16::from_be_bytes([h0, h1]));
                if width == 0 || height == 0 || frame.is_some() {
                    return None;
                }
                frame = Some(((width, height), code == 0xC2));
            }
            // Every other frame header: lossless, hierarchical
            // or arithmetic-coded.
            0xC3 | 0xC5..=0xC7 | 0xC9..=0xCB | 0xCD..=0xCF => return None,
            // Huffman tables: a class and slot, sixteen code
            // counts, and as many symbols as they add up to.
            0xC4 => {
                let mut rest = segment;
                while let Some((&id, table)) = rest.split_first() {
                    let class = usize::from(id >> 4);
                    let slot = usize::from(id & 15);
                    let counts = table.get(..16)?;
                    let symbols: usize = counts.iter().map(|n| usize::from(*n)).sum();
                    *defined.get_mut(class)?.get_mut(slot)? = true;
                    rest = table.get(16 + symbols..)?;
                }
            }
            // The first EXIF segment.
            0xE1 if !exif_seen => {
                if let Some(tiff) = segment.strip_prefix(b"Exif\0\0") {
                    exif_seen = true;
                    orientation = tiff_orientation(tiff);
                }
            }
            // The first scan: its components, each with the DC
            // and AC slots it is coded with.
            0xDA => {
                let (size, progressive) = frame?;
                let (&components, selectors) = segment.split_first()?;
                let selectors = selectors.get(..2 * usize::from(components))?;
                let all_there = selectors.as_chunks::<2>().0.iter().all(|[_, tables]| {
                    let (dc, ac) = (usize::from(tables >> 4), usize::from(tables & 15));
                    dc < 4 && ac < 4 && defined[0][dc] && defined[1][ac]
                });
                return Some(Glance {
                    size,
                    progressive,
                    names_its_tables: components == 3 && all_there,
                    orientation,
                });
            }
            // Markers that stand alone have no place ahead of
            // a scan.
            0x00 | 0x01 | 0xD0..=0xD9 => return None,
            _ => {}
        }
        at += 2 + length;
    }
}

/// The orientation tag (0x0112, one SHORT) in the first image
/// file directory of an EXIF segment's TIFF data, in either
/// byte order.
fn tiff_orientation(tiff: &[u8]) -> Option<u32> {
    let big = match tiff.get(..4)? {
        b"MM\0*" => true,
        b"II*\0" => false,
        _ => return None,
    };
    let bytes_at = |at: usize, n: usize| tiff.get(at..at.checked_add(n)?);
    let u16_at = |at: usize| {
        let b: [u8; 2] = bytes_at(at, 2)?.try_into().ok()?;
        Some(if big {
            u16::from_be_bytes(b)
        } else {
            u16::from_le_bytes(b)
        })
    };
    let u32_at = |at: usize| {
        let b: [u8; 4] = bytes_at(at, 4)?.try_into().ok()?;
        Some(if big {
            u32::from_be_bytes(b)
        } else {
            u32::from_le_bytes(b)
        })
    };
    let directory = usize::try_from(u32_at(4)?).ok()?;
    (0..usize::from(u16_at(directory)?)).find_map(|entry| {
        let at = directory.checked_add(2 + 12 * entry)?;
        if u16_at(at)? == 0x0112 && u16_at(at + 2)? == 3 {
            u16_at(at + 8).map(u32::from)
        } else {
            None
        }
    })
}

/// The declared size of a JPEG the scaled decode takes, or
/// `None` for a file it leaves to the full-size path.
///
/// Both decoders read the header, and both must read the same
/// thing as each other and as the first look (`glance`): an
/// 8-bit, Huffman-coded, lossy YCbCr frame of one size.
/// zune-jpeg's reading comes first, since it is the decoder of
/// every other render of the file: a grayscale, CMYK, YCCK or
/// RGB-coded file, a sample depth or a coding it does not
/// read, and a header it refuses all stop here. libjpeg-turbo
/// then has to agree. The two tell a file's colour space from
/// its markers by rules of their own, and a file they would
/// colour differently must not change colour with the size it
/// is asked for. Nor may a file render at one size that fails
/// at another: see [`Glance::names_its_tables`] for the one
/// such header.
///
/// The declared size is held to the decode limits here, ahead
/// of any decode, and so is a progressive file's frame with the
/// coefficients both decoders keep beside it
/// ([`progressive_coefficient_bytes`]): a file past them is not
/// eligible, and the full-size path refuses it by the same
/// checks before it decodes anything.
///
/// Last, a progressive file's first scan must be one
/// zune-jpeg reads to the marker that ends it
/// ([`first_scan::reaches_its_marker`]), or zune-jpeg refuses
/// the file at every size. That reading walks the scan's
/// data, so it comes after every check of the header.
fn eligible_size(bytes: &[u8], glance: &Glance) -> Option<(u32, u32)> {
    use zune_core::colorspace::ColorSpace;
    #[cfg(test)]
    HEADER_READS.set(HEADER_READS.get() + 1);
    // The header only, through a reader of its own: the
    // full-size decode's reader is left to that decode.
    let mut zune = zune_jpeg::JpegDecoder::new_with_options(
        zune_core::bytestream::ZCursor::new(bytes),
        zune_jpeg_options(),
    );
    zune.decode_headers().ok()?;
    if zune.input_colorspace()? != ColorSpace::YCbCr {
        return None;
    }
    let (width, height) = zune.dimensions()?;
    let (width, height) = (u32::try_from(width).ok()?, u32::try_from(height).ok()?);
    if (width, height) != glance.size
        || check_decoded_dimensions(width.into(), height.into()).is_err()
        || (u64::from(width) * u64::from(height) * 4)
            .saturating_add(progressive_coefficient_bytes(bytes, &zune))
            > MAX_DECODE_ALLOC
    {
        return None;
    }
    let header = turbojpeg::read_header(bytes).ok()?;
    let agrees = header.width == width as usize
        && header.height == height as usize
        && header.colorspace == turbojpeg::Colorspace::YCbCr
        && !header.is_lossless
        && !header.is_arithmetic
        && header.is_progressive == glance.progressive
        && (glance.progressive || glance.names_its_tables)
        && (!glance.progressive || first_scan::reaches_its_marker(bytes));
    agrees.then_some((width, height))
}

/// The decode stage's way in: the reduced frame of `bytes`
/// and the plan it was decoded by, where [`plan`] makes the
/// request eligible and libjpeg-turbo decodes the file
/// without a warning, or `None` for the full-size decode
/// every other request gets. A refused scaled decode took
/// time, so the full-size decode after it may start only
/// before the deadline: past it, this is the deadline's
/// error.
pub(super) fn decode_planned(
    bytes: &[u8],
    ops: &[Operation],
    directives: &crate::url::RequestDirectives,
    dpr: f32,
    ctx: &ProcessContext<'_>,
) -> Result<Option<(DynamicImage, ScaledDecode)>, EngineError> {
    let Some(scaled) = plan(bytes, ops, directives, dpr) else {
        return Ok(None);
    };
    match decode(bytes, &scaled) {
        Some(frame) => Ok(Some((frame, scaled))),
        None => {
            ctx.check_deadline()?;
            Ok(None)
        }
    }
}

/// Decode `bytes` as `scaled` plans: an RGB8 frame of
/// [`ScaledDecode::frame`]'s size, in the stored orientation.
/// `None` when libjpeg-turbo refuses the file or warns about
/// it; the caller then decodes at full size, which gives the
/// request the pixels or the error it always had.
///
/// The decoder is libjpeg-turbo as the `turbojpeg` crate sets
/// it up: the accurate integer inverse DCT at each block size,
/// and smooth chroma upsampling. Both are integer arithmetic,
/// and its vector code and its plain C code compute the same
/// values, so a frame is the same bytes on every host.
fn decode(bytes: &[u8], scaled: &ScaledDecode) -> Option<DynamicImage> {
    let (width, height) = scaled.frame();
    let mut decoder = turbojpeg::Decompressor::new().ok()?;
    decoder.set_scan_limit(MAX_SCANS).ok()?;
    decoder
        .set_scaling_factor(turbojpeg::ScalingFactor::new(
            1,
            scaled.denominator as usize,
        ))
        .ok()?;
    let row = width as usize * 3;
    let mut pixels = vec![0u8; row * height as usize];
    let frame = turbojpeg::Image {
        pixels: pixels.as_mut_slice(),
        width: width as usize,
        pitch: row,
        height: height as usize,
        format: turbojpeg::PixelFormat::RGB,
    };
    decoder.decompress(bytes, frame).ok()?;
    image::RgbImage::from_raw(width, height, pixels).map(DynamicImage::ImageRgb8)
}

/// [`decode`] at one over `denominator` (2, 4 or 8) for any
/// file the scaled decode takes, whatever a request would ask
/// of it: the fuzz harness's way in (see
/// [`super::decode_jpeg_scaled`]).
pub(super) fn decode_at(bytes: &[u8], denominator: u32) -> Option<DynamicImage> {
    if !DENOMINATORS.contains(&denominator) {
        return None;
    }
    let glance = glance(bytes)?;
    let scaled = ScaledDecode {
        denominator,
        stored: eligible_size(bytes, &glance)?,
        orientation: None,
    };
    decode(bytes, &scaled)
}

#[cfg(test)]
thread_local! {
    /// Turns [`plan`] off on this thread, so a test can render
    /// a request the way every request rendered before the
    /// scaled decode, and hold the scaled render against it.
    static FULL_SIZE_ONLY: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// How many times [`eligible_size`] has read a header on
    /// this thread, so a test can tell the first look turned
    /// a request down before either decoder read the file.
    static HEADER_READS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
mod tests;
