//! PDF page rendering (issue #215): a `page:`
//! segment turns a PDF the server fetched into an
//! image source by rendering one page of it.
//!
//! Rendering a PDF is a job for a real PDF engine, and
//! PDF parsers are a large attack surface, so rendering
//! follows the pattern frame extraction set in
//! [`crate::frames`]: an operator-configured external
//! program runs as a bounded subprocess, never a shell,
//! with the source bytes on stdin, an empty environment,
//! its own process group (killed whole on timeout), a
//! wall-clock limit, and a cap on what it may write back.
//! A parser crash or a hostile document's memory lands
//! in the subprocess, never in the server's address
//! space. The tool receives only bytes over stdin: no
//! path, no URL, nothing a document's embedded
//! references or actions could resolve against.
//!
//! Poppler splits facts from rendering, so the contract
//! configures two programs where frames configures one:
//!
//! - Probe: the `pdfinfo` sibling of the configured
//!   renderer, in the same directory. Run as
//!   `-f <n> -l <n> -` over the source on stdin, it
//!   prints the document's page count, its encryption
//!   state, and the selected page's size in points: the
//!   size of its crop box (clipped to its media box).
//!   Facts the design needs before rendering are probe
//!   facts, so the probe runs first, every time.
//! - Render: the configured program (Poppler's
//!   `pdftocairo` is the reference), run as
//!   `-png -singlefile -cropbox -f <n> -l <n> -r <dpi> - -`:
//!   the PDF on stdin, one PNG on stdout, at a fixed
//!   scale of 2.0 (144 DPI) clamped so neither raster
//!   axis exceeds the 16383 px output limit, and the
//!   raster area stays within 64 Mpx, at the probed page
//!   size. `-cropbox` makes the renderer draw the crop
//!   box, the box the probe measured, so the clamp is
//!   computed from the box that is drawn. Without it the
//!   renderer draws the media box, which a page may
//!   declare far larger than its crop box.
//!
//! The raster clamp bounds the surface the renderer draws,
//! not the images a page draws on it: Poppler decodes an
//! embedded JPEG 2000 image at its full declared size, four
//! bytes a sample, before scaling it onto the page. So a
//! document that stores a JPEG 2000 image past the decode
//! budget is refused before any tool runs (see
//! `oversized_jpx`), and on Linux both tools run under a
//! 2 GiB address-space cap (`RENDERER_MAX_ADDRESS_SPACE`),
//! sized to the largest page within the raster budget with
//! an image within the decode budget on it, which bounds
//! every other decode a document can ask for. An image
//! whose decode passes the cap fails in the renderer, and
//! the renderer leaves it out of the page, as it leaves out
//! any image it cannot decode.
//!
//! The rendered PNG re-enters the image pipeline as
//! the source, so every downstream budget (decode
//! dimensions and allocation, output pixels) applies to
//! it unchanged.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::toolrun::{ToolCommand, ToolError, ToolInput, ToolRun, run_tool, tool_cpu_limit};

/// The most pages a document may have before the
/// pipeline refuses it (issue #215). The cap bounds
/// page-tree parser exposure and keeps the feature
/// aimed at documents; it sits far above real books and
/// reports, and one page renders per request either way,
/// so the count multiplies no work.
pub const MAX_PDF_PAGES: u32 = 1000;

/// Cap on the rendered page's PNG bytes. Tied to the
/// engine's decode allocation budget (256 MiB), the
/// frames rationale verbatim: a page whose PNG passes
/// this can never survive the decode budgets, so
/// anything larger is a broken or hostile tool, not a
/// page.
pub const MAX_PAGE_PNG_BYTES: usize = 256 * 1024 * 1024;

/// The fixed render scale: pages render at twice their
/// nominal size (144 DPI), the frames precedent of
/// extracting at a defined resolution and letting the
/// pipeline resize. A defined 2x gives preview and
/// unfurl crops real headroom without a URL-visible
/// knob.
pub const RENDER_SCALE: f64 = 2.0;

/// The size a rendered raster axis may not exceed at the
/// probed page size, in pixels: one under the 16383 px
/// output limit, itself one under the decode chain's 16384
/// (issue #528), applied before the tool runs so a poster
/// page renders at a clamped scale instead of asking for a
/// raster the pipeline would refuse. A page rendered at the
/// decode cap that no step resized would end past the output
/// limit, which the engine refuses.
///
/// The pixel of margin is for the probe's rounding: it
/// prints the page size to six significant digits (an
/// 8191.5004 pt side prints as 8191.5), and the renderer
/// rounds the raster up, so a scale that put the printed
/// size at exactly 16383 px could render the true size at
/// 16384. Six digits are off by at most a part in 200,000,
/// under a tenth of a pixel at this size, which the margin
/// absorbs.
const MAX_RASTER_AXIS: f64 = (crate::url::MAX_DIMENSION - 1) as f64;

/// The total raster area a render may not exceed, in
/// pixels: 64 Mpx, the raster the 256 MB decode budget
/// holds at 4 bytes per pixel. The per-axis clamp alone
/// allows 16383 squared (about 268 Mpx, over 1 GB of
/// renderer memory), which a tiny hostile MediaBox
/// reaches trivially (issue #262).
const MAX_RASTER_PIXELS: f64 = 64_000_000.0;

/// The lowest resolution the renderer is asked for, in
/// dots per inch. The clamped scale's DPI is floored to a
/// whole number, and this is the smallest one above zero,
/// so a page that does not fit [`MAX_RASTER_AXIS`] and
/// [`MAX_RASTER_PIXELS`] at this resolution cannot be
/// clamped into them: it is refused before the renderer
/// runs (see [`fits_at_min_dpi`]).
const MIN_RENDER_DPI: f64 = 1.0;

/// How much of a probe report the parser keeps. The
/// report is a few hundred bytes; this is generous
/// headroom for chatty builds.
const PROBE_REPORT_CAP: usize = 1024 * 1024;

/// The PDF tools' address-space cap on Linux, 2 GiB. The
/// heaviest renders within the budgets measured about 1.1
/// GiB with Poppler 24.02: a page holding an 8192x8192
/// JPEG 2000 image (the decode budget, which Poppler holds
/// at four bytes a sample while it draws it) at the 64
/// megapixel raster, and a 64 megapixel page drawing a
/// 12000x12000 JPEG. The cap leaves room over those for a
/// transparency group or two at the full raster, and turns
/// a decode past it into a failure in the tool within about
/// a second.
const RENDERER_MAX_ADDRESS_SPACE: u64 = 2 << 30;

/// The two markers that open every JPEG 2000 codestream:
/// SOC (start of codestream) and SIZ (image and tile size),
/// which the format requires to follow it at once.
const JPX_SOC_SIZ: [u8; 4] = [0xFF, 0x4F, 0xFF, 0x51];

/// Whether bytes can be a PDF at all: the format's
/// header, `%PDF-`, appears within the first 1024 bytes
/// (the format's own tolerance for prefixed junk). The
/// engine sniffs with this before spawning anything:
/// bytes that fail it never reach the tool.
pub fn sniff_pdf(bytes: &[u8]) -> bool {
    let head = &bytes[..bytes.len().min(1024)];
    head.windows(5).any(|w| w == b"%PDF-")
}

/// What a probe learned about a PDF source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PdfProbe {
    /// The document's total page count.
    pub pages: u32,
    /// The document carries encryption. A document the
    /// probe could read opened with the empty password,
    /// so this is a fact for the record, not a refusal:
    /// only a document that actually needs a password
    /// fails, and it fails at the probe.
    pub encrypted: bool,
    /// The selected page's width in points: its crop
    /// box, the area the renderer draws.
    pub page_width: f64,
    /// The selected page's height in points: its crop
    /// box, the area the renderer draws.
    pub page_height: f64,
}

/// One rendered page and the facts meta reports.
#[derive(Debug)]
pub struct RenderedPage {
    /// The page as PNG bytes, ready to decode as the
    /// source image.
    pub png: Vec<u8>,
    /// The document's total page count, for meta.
    pub pages: u32,
    /// The selected page (1-based, after the grammar's
    /// canonicalization), for meta.
    pub page: u32,
    /// The page's width in points, for meta.
    pub page_width: f64,
    /// The page's height in points, for meta.
    pub page_height: f64,
}

/// Why page rendering failed. The class decides the
/// response: a source that is not a usable PDF fails
/// the way bad sources already fail, a tool that cannot
/// do its job is a server-side failure, and a selection
/// the source cannot answer is the request's problem.
#[derive(Debug)]
pub enum PdfError {
    /// The probe could not read the source as a PDF
    /// (not a PDF, a corrupt document), or the document
    /// is past a source budget (the page count cap).
    NotPdf(String),
    /// The selection cannot be answered by this source:
    /// a page past the end, or a document that needs a
    /// password to open.
    Unsupported(String),
    /// The tool itself failed: it could not be spawned,
    /// timed out, exited nonzero, produced no page, or
    /// produced output past the cap or not a PNG.
    Failed(String),
}

impl std::fmt::Display for PdfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PdfError::NotPdf(why) => write!(f, "{why}"),
            PdfError::Unsupported(why) => write!(f, "{why}"),
            PdfError::Failed(why) => write!(f, "{why}"),
        }
    }
}

impl std::error::Error for PdfError {}

/// The stable message for a document that needs a
/// password to open. One class, one text, from either
/// tool: the probe refuses it, and a renderer that
/// refuses it anyway lands in the same class.
const PASSWORD_MESSAGE: &str =
    "the document is password-protected and cannot be opened without its password";

/// A configured PDF renderer: an external render program
/// plus its `pdfinfo` probe sibling, and the bounds every
/// run obeys. Cheap to share behind an `Arc`; each render
/// spawns fresh processes.
#[derive(Debug)]
pub struct PdfRenderer {
    program: PathBuf,
    probe_program: PathBuf,
    timeout: Duration,
}

impl PdfRenderer {
    /// Validate the program paths at boot: the renderer
    /// must exist and be executable, and so must the
    /// `pdfinfo` sibling in its directory, because a
    /// renderer without its probe is a misconfiguration
    /// that should fail here, not on the first `page:`
    /// request.
    pub fn new(program: PathBuf, timeout: Duration) -> Result<Self, PdfError> {
        validate_program(&program, "PDF renderer")?;
        let probe_program = probe_sibling(&program);
        validate_program(&probe_program, "PDF probe (pdfinfo)").map_err(|e| match e {
            PdfError::Failed(why) => PdfError::Failed(format!(
                "{why}; the probe is the pdfinfo program next to the renderer {}",
                program.display()
            )),
            other => other,
        })?;
        Ok(Self {
            program,
            probe_program,
            timeout,
        })
    }

    /// Probe the source for the selected page: page
    /// count, encryption state, and the page's size. A
    /// document the probe cannot read is `NotPdf`, except
    /// one that needs a password, which is its own
    /// `Unsupported` class.
    /// The single-probe entry: the render path probes
    /// through `probe_until` (the shared deadline); this
    /// spelling stays as the entry the test suite
    /// exercises.
    #[allow(dead_code)]
    pub fn probe(&self, pdf: &[u8], page: u32) -> Result<PdfProbe, PdfError> {
        self.probe_until(pdf, page, Instant::now() + self.timeout)
    }

    /// The probe under a caller-supplied deadline, so a
    /// stacked sequence (probe, a confirming second probe,
    /// the render) shares one budget instead of a fresh
    /// timeout per invocation (issue #362).
    fn probe_until(&self, pdf: &[u8], page: u32, deadline: Instant) -> Result<PdfProbe, PdfError> {
        let run = self.run_probe(pdf, Some(page), deadline)?;
        if run.status.success() {
            let report = String::from_utf8_lossy(&run.stdout);
            return parse_probe(&report, page).ok_or_else(not_pdf);
        }
        if run.stderr.contains("Incorrect password") {
            return Err(PdfError::Unsupported(PASSWORD_MESSAGE.to_string()));
        }
        if run.stderr.contains("Wrong page range") {
            // The probe refuses a range past the end of
            // the document (exit 99), so an out-of-range
            // selection never produces a report. Confirm
            // the document with a whole-document probe and
            // answer with the range refusal and its real
            // count; a document the second probe cannot
            // read either is simply not usable.
            let whole = self.run_probe(pdf, None, deadline)?;
            if whole.status.success() {
                let report = String::from_utf8_lossy(&whole.stdout);
                if let Some(probe) = parse_probe(&report, page) {
                    if page > probe.pages {
                        return Err(PdfError::Unsupported(format!(
                            "page {page} is past the end of the document, which has {} {}",
                            probe.pages,
                            if probe.pages == 1 { "page" } else { "pages" }
                        )));
                    }
                    return Ok(probe);
                }
            }
            if whole.stderr.contains("Incorrect password") {
                return Err(PdfError::Unsupported(PASSWORD_MESSAGE.to_string()));
            }
        }
        Err(not_pdf())
    }

    /// One probe run: ranged to the selected page (the
    /// form that reports that page's own size), or over
    /// the whole document when `page` is `None`.
    fn run_probe(
        &self,
        pdf: &[u8],
        page: Option<u32>,
        deadline: Instant,
    ) -> Result<ToolRun, PdfError> {
        match page {
            Some(page) => {
                let page_arg = page.to_string();
                self.run(
                    &self.probe_program,
                    "probe",
                    &["-f", &page_arg, "-l", &page_arg, "-"],
                    pdf,
                    PROBE_REPORT_CAP,
                    deadline,
                )
            }
            None => self.run(
                &self.probe_program,
                "probe",
                &["-"],
                pdf,
                PROBE_REPORT_CAP,
                deadline,
            ),
        }
    }

    /// Render the selected page as PNG bytes. A document
    /// that stores a JPEG 2000 image past the decode budget
    /// is refused before any tool runs. Then the probe: the
    /// page count cap and the page range are enforced
    /// against the probe before the renderer runs, and the
    /// render scale is computed from the probed page size,
    /// the crop box the renderer then draws.
    pub fn render_page(&self, pdf: &[u8], page: u32) -> Result<RenderedPage, PdfError> {
        if let Some(image) = oversized_jpx(pdf) {
            return Err(PdfError::NotPdf(format!(
                "the document embeds a {}x{} JPEG 2000 image with {} {}, past the {} byte \
                 decode allocation budget",
                image.width,
                image.height,
                image.components,
                if image.components == 1 {
                    "component"
                } else {
                    "components"
                },
                crate::engine::MAX_DECODE_ALLOC
            )));
        }
        // One deadline for the probe and the render
        // together (issue #362).
        let deadline = Instant::now() + self.timeout;
        let probe = self.probe_until(pdf, page, deadline)?;
        if probe.pages > MAX_PDF_PAGES {
            return Err(PdfError::NotPdf(format!(
                "the document has {} pages, over the {MAX_PDF_PAGES} page cap",
                probe.pages
            )));
        }
        if page > probe.pages {
            return Err(PdfError::Unsupported(format!(
                "page {page} is past the end of the document, which has {} {}",
                probe.pages,
                if probe.pages == 1 { "page" } else { "pages" }
            )));
        }
        if !fits_at_min_dpi(probe.page_width, probe.page_height) {
            return Err(PdfError::NotPdf(format!(
                "page {page} is {} x {} pts, too large to render within the raster limits ({MAX_RASTER_AXIS} px per axis, {} megapixels) at {MIN_RENDER_DPI} dpi, the lowest resolution the renderer is asked for",
                probe.page_width,
                probe.page_height,
                MAX_RASTER_PIXELS / 1_000_000.0
            )));
        }
        let scale = render_scale(probe.page_width, probe.page_height);
        // Floor the DPI so the clamped scale can never
        // round a raster axis past the axis cap; at the
        // fixed scale this is exactly 144. The page fits the
        // limits at the minimum DPI (checked above), so
        // raising a smaller DPI to the minimum keeps the
        // raster within them.
        let dpi = (scale * 72.0).floor().max(MIN_RENDER_DPI);
        let dpi_arg = format!("{dpi}");
        let page_arg = page.to_string();
        // `-cropbox`: draw the crop box, the size the probe
        // reported and the scale was clamped for, not the
        // media box, which can be larger.
        let run = self.run(
            &self.program,
            "renderer",
            &[
                "-png",
                "-singlefile",
                "-cropbox",
                "-f",
                &page_arg,
                "-l",
                &page_arg,
                "-r",
                &dpi_arg,
                "-",
                "-",
            ],
            pdf,
            MAX_PAGE_PNG_BYTES,
            deadline,
        )?;
        if !run.status.success() {
            if run.stderr.contains("Incorrect password") {
                return Err(PdfError::Unsupported(PASSWORD_MESSAGE.to_string()));
            }
            if run.status.code() == Some(99) {
                return Err(PdfError::Unsupported(format!(
                    "page {page} is past the end of the document, which has {} {}",
                    probe.pages,
                    if probe.pages == 1 { "page" } else { "pages" }
                )));
            }
            return Err(PdfError::Failed(format!(
                "PDF renderer exited with {}: {}",
                run.status,
                run.stderr_excerpt()
            )));
        }
        if run.stdout.is_empty() {
            return Err(PdfError::Failed(
                "the PDF renderer produced no output".to_string(),
            ));
        }
        if !run.stdout.starts_with(b"\x89PNG\r\n\x1a\n") {
            return Err(PdfError::Failed(
                "the PDF renderer's output is not a PNG".to_string(),
            ));
        }
        Ok(RenderedPage {
            png: run.stdout,
            pages: probe.pages,
            page,
            page_width: probe.page_width,
            page_height: probe.page_height,
        })
    }

    /// Run one of the tools once through the shared runner
    /// ([`crate::toolrun::run_tool`]): the source bytes on
    /// stdin, stdout capped at `max_stdout`, and stderr kept
    /// up to the report cap.
    fn run(
        &self,
        program: &Path,
        name: &'static str,
        args: &[&str],
        stdin_bytes: &[u8],
        max_stdout: usize,
        deadline: Instant,
    ) -> Result<ToolRun, PdfError> {
        run_tool(ToolCommand {
            program,
            args,
            input: ToolInput::Bytes(stdin_bytes),
            max_stdout,
            max_stderr: PROBE_REPORT_CAP,
            deadline,
            cpu_limit: tool_cpu_limit(self.timeout),
            max_address_space: RENDERER_MAX_ADDRESS_SPACE,
        })
        .map_err(|e| {
            PdfError::Failed(match e {
                ToolError::Spawn(e) => {
                    format!("could not start the PDF {name} {}: {e}", program.display())
                }
                ToolError::OutputCap(cap) => {
                    format!("the PDF {name}'s output passed the {cap} byte cap")
                }
                ToolError::Read(e) => format!("reading the PDF {name}'s output failed: {e}"),
                ToolError::TimedOut => {
                    format!("PDF {name} timed out after {}s", self.timeout.as_secs())
                }
                ToolError::Wait(e) => format!("waiting on the PDF {name} failed: {e}"),
            })
        })
    }
}

/// The probe program for a renderer path: the `pdfinfo`
/// sibling in the same directory, keeping the renderer's
/// extension if it carries one (a `.exe` renderer pairs
/// with a `.exe` probe).
fn probe_sibling(renderer: &Path) -> PathBuf {
    let mut name = std::ffi::OsString::from("pdfinfo");
    if let Some(ext) = renderer.extension() {
        name.push(".");
        name.push(ext);
    }
    renderer.with_file_name(name)
}

/// Validate one tool path the way the video tools' are
/// validated: it must exist and be executable, so a
/// misconfiguration fails at boot, not on the first
/// `page:` request.
fn validate_program(path: &Path, what: &str) -> Result<(), PdfError> {
    if !path.is_file() {
        return Err(PdfError::Failed(format!(
            "{what} {} does not exist or is not a file",
            path.display()
        )));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(path)
            .map(|m| m.permissions().mode())
            .unwrap_or(0);
        if mode & 0o111 == 0 {
            return Err(PdfError::Failed(format!(
                "{what} {} is not executable",
                path.display()
            )));
        }
    }
    Ok(())
}

/// The render scale for a page of the given size in
/// points: the fixed [`RENDER_SCALE`], clamped down so
/// neither raster axis exceeds [`MAX_RASTER_AXIS`] and the
/// raster area stays within [`MAX_RASTER_PIXELS`].
pub fn render_scale(page_width: f64, page_height: f64) -> f64 {
    let longest = page_width.max(page_height);
    if longest <= 0.0 {
        return RENDER_SCALE;
    }
    let mut scale = RENDER_SCALE.min(MAX_RASTER_AXIS / longest);
    let area = page_width * page_height;
    if area > 0.0 {
        scale = scale.min((MAX_RASTER_PIXELS / area).sqrt());
    }
    scale
}

/// Whether a page of the given size in points renders
/// within [`MAX_RASTER_AXIS`] and [`MAX_RASTER_PIXELS`] at
/// [`MIN_RENDER_DPI`]. [`render_scale`] can clamp any
/// smaller page into the limits, but the DPI it becomes
/// does not go below the minimum, so a page past it (over
/// 16382 inches on a side, or 64 million square inches)
/// would render past the limits at the minimum DPI. A
/// size that is not a finite number never fits.
fn fits_at_min_dpi(page_width: f64, page_height: f64) -> bool {
    let per_point = MIN_RENDER_DPI / 72.0;
    let (width, height) = (page_width * per_point, page_height * per_point);
    width <= MAX_RASTER_AXIS && height <= MAX_RASTER_AXIS && width * height <= MAX_RASTER_PIXELS
}

/// Parse the probe facts out of a `pdfinfo` report: the
/// `Pages:` and `Encrypted:` lines and the selected
/// page's size (printed as `Page size:` for a whole-
/// document report and `Page   <n> size:` when the probe
/// names a page range, the form this probe runs).
/// Returns `None` when any fact is missing or malformed:
/// a report the parser cannot read means the document
/// is not one the pipeline can use.
pub fn parse_probe(report: &str, page: u32) -> Option<PdfProbe> {
    // pdfinfo prints the document metadata (title, author,
    // and the rest) ahead of the document facts, and
    // metadata is document-controlled text: a value
    // containing a newline prints as extra report lines,
    // including lines shaped like the structured fields.
    // A forged small "Page size:" in a title once steered
    // the raster clamp (issue #363). The real fields print
    // in a fixed order after every metadata field, with
    // Pages: first among them, so only the lines from the
    // last Pages: line onward are read as the structured
    // section; anything earlier is metadata.
    let lines: Vec<&str> = report.lines().collect();
    let start = lines.iter().rposition(|l| l.starts_with("Pages:"))?;
    let mut pages = None;
    let mut encrypted = None;
    let mut size = None;
    for line in &lines[start..] {
        if let Some(rest) = line.strip_prefix("Pages:") {
            pages = rest.trim().parse::<u32>().ok();
        } else if let Some(rest) = line.strip_prefix("Encrypted:") {
            encrypted = Some(rest.trim_start().starts_with("yes"));
        } else if size.is_none() {
            size = parse_page_size(line, page);
        }
    }
    let (page_width, page_height) = size?;
    Some(PdfProbe {
        pages: pages?,
        encrypted: encrypted?,
        page_width,
        page_height,
    })
}

/// Parse one report line as a page size, in either of
/// the probe's spellings:
/// `Page size:       612 x 792 pts (letter)` or
/// `Page   50 size:  612 x 792 pts (letter)`.
fn parse_page_size(line: &str, page: u32) -> Option<(f64, f64)> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    let dims: &[&str] = match tokens.as_slice() {
        ["Page", "size:", rest @ ..] => rest,
        ["Page", n, "size:", rest @ ..] if n.parse::<u32>().ok() == Some(page) => rest,
        _ => return None,
    };
    let [w, "x", h, ..] = dims else {
        return None;
    };
    let width: f64 = w.parse().ok()?;
    let height: f64 = h.parse().ok()?;
    // A size must be a positive, finite number. The probe
    // prints a box with an overflowing coordinate as `inf`
    // or `-nan`, which `f64` parses, and a NaN side passes
    // a plain `<= 0.0` test and skips the raster clamp.
    let usable = |side: f64| side.is_finite() && side > 0.0;
    if !usable(width) || !usable(height) {
        return None;
    }
    Some((width, height))
}

/// A JPEG 2000 image's size, as its codestream declares it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct JpxSize {
    /// The image's width in pixels (its reference grid
    /// less the image offset).
    width: u64,
    /// The image's height in pixels.
    height: u64,
    /// How many components each pixel has.
    components: u16,
    /// The samples the decoded image holds, summed over its
    /// components, each at its own subsampling.
    samples: u64,
}

impl JpxSize {
    /// Whether the decoded image passes the engine's decode
    /// allocation budget, counted the way the engine counts
    /// its own decodes, four bytes a pixel, or one byte a
    /// sample where an image has more than four components.
    fn past_the_budget(&self) -> bool {
        let pixels = self.width.saturating_mul(self.height);
        pixels.saturating_mul(4).max(self.samples) > crate::engine::MAX_DECODE_ALLOC
    }
}

/// The first JPEG 2000 image the document stores as is
/// whose decoded size passes the decode budget (see
/// [`JpxSize`]), or `None`. A `JPXDecode` stream holds the
/// codestream itself, so its SOC and SIZ markers sit in the
/// document's bytes, and the SIZ segment declares the size
/// the renderer's decoder allocates for. The scan reads
/// every SOC and SIZ pair whose segment is well formed
/// (its length matches its component count, and the image
/// is not empty); the odds that compressed bytes elsewhere
/// form one are negligible. A codestream behind a further
/// filter or encryption is out of its sight, and the
/// renderer's address-space cap bounds that one instead.
fn oversized_jpx(pdf: &[u8]) -> Option<JpxSize> {
    let mut at = 0;
    while let Some(found) = pdf[at..].iter().position(|&b| b == JPX_SOC_SIZ[0]) {
        let start = at + found;
        if pdf[start..].starts_with(&JPX_SOC_SIZ)
            && let Some(size) = parse_jpx_siz(&pdf[start + JPX_SOC_SIZ.len()..])
            && size.past_the_budget()
        {
            return Some(size);
        }
        at = start + 1;
    }
    None
}

/// Parse a JPEG 2000 SIZ marker segment, from its length
/// field on: `Lsiz Rsiz Xsiz Ysiz XOsiz YOsiz XTsiz YTsiz
/// XTOsiz YTOsiz Csiz`, then `Ssiz XRsiz YRsiz` per
/// component, big-endian. `None` for a segment that is cut
/// short or not well formed.
fn parse_jpx_siz(segment: &[u8]) -> Option<JpxSize> {
    let u16_at = |at: usize| -> Option<u16> {
        Some(u16::from_be_bytes(
            segment.get(at..at + 2)?.try_into().ok()?,
        ))
    };
    let u32_at = |at: usize| -> Option<u64> {
        Some(u64::from(u32::from_be_bytes(
            segment.get(at..at + 4)?.try_into().ok()?,
        )))
    };
    let length = usize::from(u16_at(0)?);
    let (x, y) = (u32_at(4)?, u32_at(8)?);
    let (x0, y0) = (u32_at(12)?, u32_at(16)?);
    let components = u16_at(36)?;
    let count = usize::from(components);
    if count == 0 || count > 16384 || length != 38 + 3 * count || x <= x0 || y <= y0 {
        return None;
    }
    let table = segment.get(38..38 + 3 * count)?;
    let mut samples: u64 = 0;
    for component in 0..count {
        let dx = u64::from(table[3 * component + 1]);
        let dy = u64::from(table[3 * component + 2]);
        if dx == 0 || dy == 0 {
            return None;
        }
        let width = x.div_ceil(dx) - x0.div_ceil(dx);
        let height = y.div_ceil(dy) - y0.div_ceil(dy);
        samples = samples.saturating_add(width.saturating_mul(height));
    }
    Some(JpxSize {
        width: x - x0,
        height: y - y0,
        components,
        samples,
    })
}

/// The not-a-PDF refusal: bytes the probe cannot read
/// as a document at all.
fn not_pdf() -> PdfError {
    PdfError::NotPdf(
        "the source is not a PDF the probe can read (not a PDF, or a corrupt document)".to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A whole-document probe report, recorded from the
    /// reference probe on a real two-page document.
    const PLAIN_REPORT: &str = "\
Title:           untitled
Pages:           2
Encrypted:       no
Page size:       612 x 792 pts (letter)
Page rot:        0
PDF version:     1.3
";

    /// The per-page report form the probe runs produce:
    /// the size line names the probed page.
    const PAGE_REPORT: &str = "\
Pages:           100
Encrypted:       no
Page   50 size:  612 x 792 pts (letter)
Page   50 rot:   0
";

    /// A permissions-only encrypted document the probe
    /// opened with the empty password: the report is
    /// complete and the encryption a fact, not a refusal.
    const PERMISSIONS_REPORT: &str = "\
Pages:           2
Encrypted:       yes (print:yes copy:yes change:yes addNotes:yes algorithm:AES-256)
Page size:       595.28 x 841.89 pts (A4)
";

    #[test]
    fn probe_parses_the_recorded_reports() {
        let probe = parse_probe(PLAIN_REPORT, 1).expect("a plain report parses");
        assert_eq!(probe.pages, 2);
        assert!(!probe.encrypted);
        assert_eq!((probe.page_width, probe.page_height), (612.0, 792.0));

        let probe = parse_probe(PAGE_REPORT, 50).expect("a per-page report parses");
        assert_eq!(probe.pages, 100);
        assert_eq!((probe.page_width, probe.page_height), (612.0, 792.0));

        let probe = parse_probe(PERMISSIONS_REPORT, 1).expect("an encrypted report parses");
        assert_eq!(probe.pages, 2);
        assert!(probe.encrypted);
        assert_eq!((probe.page_width, probe.page_height), (595.28, 841.89));
    }

    #[test]
    fn probe_ignores_metadata_lines_shaped_like_facts() {
        // Issue #363: a title containing a newline prints
        // as extra report lines ahead of the structured
        // section. The forged small page size (and the
        // forged page count) must not steer the parse: the
        // facts come from the structured section only.
        let report = "\
Title:           Forged
Page size:       10 x 10 pts
Pages:           999
Producer:        someone
Pages:           2
Encrypted:       no
Page size:       612 x 792 pts (letter)
";
        let probe = parse_probe(report, 1).expect("the real facts parse");
        assert_eq!(probe.pages, 2);
        assert_eq!((probe.page_width, probe.page_height), (612.0, 792.0));
    }

    #[test]
    fn probe_parses_a_rotated_page_size() {
        let report = "\
Pages:           1
Encrypted:       no
Page size:       792 x 612 pts (letter)
";
        let probe = parse_probe(report, 1).unwrap();
        assert_eq!((probe.page_width, probe.page_height), (792.0, 612.0));
    }

    #[test]
    fn probe_refuses_incomplete_or_foreign_reports() {
        // No page count.
        assert!(parse_probe("Encrypted: no\nPage size: 612 x 792 pts\n", 1).is_none());
        // No page size.
        assert!(parse_probe("Pages: 2\nEncrypted: no\n", 1).is_none());
        // A per-page size for a different page than probed.
        assert!(parse_probe(PAGE_REPORT, 51).is_none());
        // Junk and empty reports.
        assert!(parse_probe("Syntax Error: Couldn't find trailer dictionary\n", 1).is_none());
        assert!(parse_probe("", 1).is_none());
        // A degenerate size is not a size.
        assert!(parse_probe("Pages: 1\nEncrypted: no\nPage size: 0 x 792 pts\n", 1).is_none());
        // Nor is one that is not a finite number: the probe
        // prints a box with an overflowing coordinate this
        // way, and f64 parses every one of these spellings.
        for size in [
            "-nan x -nan",
            "nan x 792",
            "612 x inf",
            "inf x inf",
            "-inf x 792",
        ] {
            let report = format!("Pages: 1\nEncrypted: no\nPage    1 size:  {size} pts\n");
            assert!(parse_probe(&report, 1).is_none(), "{size}");
        }
    }

    #[test]
    fn scale_is_two_until_the_raster_cap_bites() {
        assert_eq!(render_scale(612.0, 792.0), 2.0);
        assert_eq!(render_scale(595.28, 841.89), 2.0);
        // A 200-inch poster page (14400 pt per axis): the
        // area budget bites before the axis cap (issue
        // #262), clamping the raster to 64 Mpx.
        let scale = render_scale(14_400.0, 14_400.0);
        let expected = (64_000_000.0f64 / (14_400.0 * 14_400.0)).sqrt();
        assert!((scale - expected).abs() < 1e-9, "{scale}");
        let axis = 14_400.0 * scale;
        assert!(axis * axis <= 64_000_001.0, "{axis}");
        // A wide-but-short page still clamps by axis, to
        // the output limit rather than the decode cap
        // (issue #528), less the probe's rounding margin: an
        // 8192 pt page at the fixed scale would render 16384
        // px wide.
        let scale = render_scale(20_000.0, 100.0);
        assert!((scale - 16_382.0 / 20_000.0).abs() < 1e-9, "{scale}");
        let scale = render_scale(8_192.0, 100.0);
        assert!((scale - 16_382.0 / 8_192.0).abs() < 1e-9, "{scale}");
        assert!((8_192.0 * (scale * 72.0).floor() / 72.0) <= 16_383.0);
        // The hostile MediaBox from the finding: a
        // 100000 pt square page renders at 8000 px per
        // axis (64 Mpx), not 16383 (268 Mpx).
        let scale = render_scale(100_000.0, 100_000.0);
        assert!((100_000.0 * scale - 8_000.0).abs() < 1e-6, "{scale}");
        // A degenerate size keeps the fixed scale; the
        // renderer run is where that document fails.
        assert_eq!(render_scale(0.0, 0.0), 2.0);
    }

    #[test]
    fn pages_past_the_raster_limits_at_the_minimum_dpi_do_not_fit() {
        // Ordinary and poster pages fit.
        assert!(fits_at_min_dpi(612.0, 792.0));
        assert!(fits_at_min_dpi(14_400.0, 14_400.0));
        // The axis limit at 1 dpi: 16382 px is 1179504 pt.
        assert!(fits_at_min_dpi(1_179_504.0, 10.0));
        assert!(fits_at_min_dpi(10.0, 1_179_504.0));
        assert!(!fits_at_min_dpi(1_179_600.0, 10.0));
        assert!(!fits_at_min_dpi(10.0, 1_179_600.0));
        // The area limit at 1 dpi: 64 Mpx is 576000 pt
        // square.
        assert!(fits_at_min_dpi(570_000.0, 570_000.0));
        assert!(!fits_at_min_dpi(580_000.0, 580_000.0));
        assert!(!fits_at_min_dpi(2_000_000.0, 2_000_000.0));
        // A size that is not a finite number never fits.
        assert!(!fits_at_min_dpi(f64::NAN, 100.0));
        assert!(!fits_at_min_dpi(100.0, f64::NAN));
        assert!(!fits_at_min_dpi(f64::INFINITY, 100.0));
        // Every page that fits renders within both limits at
        // the DPI the clamp gives it.
        for (w, h) in [
            (1_179_504.0, 10.0),
            (570_000.0, 570_000.0),
            (300_000.0, 1_000.0),
            (14_400.0, 14_400.0),
        ] {
            let dpi = (render_scale(w, h) * 72.0).floor().max(MIN_RENDER_DPI);
            let (rw, rh) = (raster_side(w, dpi), raster_side(h, dpi));
            assert!(
                rw <= MAX_RASTER_AXIS && rh <= MAX_RASTER_AXIS,
                "{w}x{h} at {dpi}"
            );
            assert!(
                rw * rh <= MAX_RASTER_PIXELS + rw + rh + 1.0,
                "{w}x{h} at {dpi}"
            );
        }
    }

    /// The raster side the reference renderer draws for a
    /// page side of `points` at `dpi`: the scaled size,
    /// rounded up.
    fn raster_side(points: f64, dpi: f64) -> f64 {
        (points * dpi / 72.0).ceil()
    }

    #[test]
    fn a_page_the_probe_rounds_down_stays_within_the_output_limit() {
        // The probe prints a page side to six significant
        // digits, and the renderer rounds the raster up
        // (issue #528). An 8191.5004 pt side printed as
        // 8191.5, which at the old 16383 px cap kept the
        // scale at 2: the page rendered 16384 px wide and the
        // output check refused it. Every printed size within
        // a pixel or two of a DPI step, short side or long,
        // renders within the limit at the largest true size
        // that prints that way.
        let max = f64::from(crate::url::MAX_DIMENSION);
        let dpi_at = |w: f64, h: f64| (render_scale(w, h) * 72.0).floor().max(1.0);
        assert_eq!(dpi_at(8_191.5, 100.0), 143.0);
        assert_eq!(raster_side(8_191.500_4, 143.0), 16_270.0);
        assert_eq!(raster_side(8_191.500_4, 144.0), 16_384.0);
        // The sides that render at exactly the limit at each
        // DPI the clamp can pick, and two points either way.
        for step in 72..=144u32 {
            let edge = max * 72.0 / f64::from(step);
            for tenths in -20..=20 {
                let printed = sig6(edge + f64::from(tenths) / 10.0);
                let truth = printed + half_unit_of_six_digits(printed);
                for (w, h, tw, th) in [
                    (printed, 100.0, truth, 100.0),
                    (100.0, printed, 100.0, truth),
                ] {
                    let dpi = dpi_at(w, h);
                    assert!(
                        raster_side(tw, dpi) <= max && raster_side(th, dpi) <= max,
                        "{w}x{h} pt printed, {tw}x{th} pt true, at {dpi} dpi"
                    );
                }
            }
        }
    }

    /// `value` rounded to six significant digits, as the
    /// probe's `%g` prints it.
    fn sig6(value: f64) -> f64 {
        format!("{value:.5e}").parse().unwrap()
    }

    /// Half a unit in the sixth significant digit of
    /// `value`: how far the true size can sit from the
    /// printed one.
    fn half_unit_of_six_digits(value: f64) -> f64 {
        0.5 * 10f64.powi(value.abs().log10().floor() as i32 - 5)
    }

    #[test]
    fn sniff_finds_the_header_within_its_window() {
        assert!(sniff_pdf(b"%PDF-1.4\nrest"));
        // Prefixed junk inside the window is the format's
        // own tolerance.
        let mut prefixed = vec![b'x'; 100];
        prefixed.extend_from_slice(b"%PDF-1.7");
        assert!(sniff_pdf(&prefixed));
        // A PNG is not a PDF, and a header past the
        // window does not count.
        assert!(!sniff_pdf(b"\x89PNG\r\n\x1a\n"));
        let mut late = vec![b'x'; 2048];
        late.extend_from_slice(b"%PDF-1.7");
        assert!(!sniff_pdf(&late));
        assert!(!sniff_pdf(b""));
    }

    #[test]
    fn probe_sibling_sits_next_to_the_renderer() {
        assert_eq!(
            probe_sibling(Path::new("/usr/bin/pdftocairo")),
            PathBuf::from("/usr/bin/pdfinfo")
        );
        assert_eq!(
            probe_sibling(Path::new("tools/pdftocairo.exe")),
            PathBuf::from("tools/pdfinfo.exe")
        );
        assert_eq!(
            probe_sibling(Path::new("pdftocairo")),
            PathBuf::from("pdfinfo")
        );
    }

    fn temp_dir(name: &str) -> crate::test_support::ScratchDir {
        crate::test_support::scratch_dir(&format!("pdf-test-{name}"))
    }

    fn write_tool(dir: &Path, name: &str, script: &str) -> PathBuf {
        let path = dir.join(name);
        crate::test_support::write_script(&path, script);
        path
    }

    /// A fake tool pair in one directory: the probe
    /// prints a recorded-shape report, the renderer
    /// prints a PNG. Returns the directory and the
    /// renderer path to build the extractor with.
    fn fake_pair(
        name: &str,
        info_script: &str,
        render_script: &str,
    ) -> (crate::test_support::ScratchDir, PathBuf) {
        let dir = temp_dir(name);
        write_tool(&dir, "pdfinfo", info_script);
        let renderer = write_tool(&dir, "pdftocairo", render_script);
        (dir, renderer)
    }

    const INFO_OK: &str = "#!/bin/sh\n\
cat > /dev/null\n\
printf '%s\\n' 'Pages:           3' 'Encrypted:       no' 'Page size:       612 x 792 pts (letter)'\n\
exit 0\n";

    const RENDER_OK: &str = "#!/bin/sh\n\
cat > /dev/null\n\
printf '\\211PNG\\015\\012\\032\\012fakepng'\n\
exit 0\n";

    #[test]
    fn renderer_validates_both_programs_at_construction() {
        // A missing renderer.
        let err = PdfRenderer::new(
            PathBuf::from("/nonexistent/no-such-renderer"),
            Duration::from_secs(5),
        )
        .unwrap_err();
        assert!(matches!(err, PdfError::Failed(_)));
        assert!(err.to_string().contains("does not exist"), "{err}");

        // A renderer whose probe sibling is missing.
        let dir = temp_dir("no-probe");
        let renderer = write_tool(&dir, "pdftocairo", "#!/bin/sh\nexit 0\n");
        let err = PdfRenderer::new(renderer, Duration::from_secs(5)).unwrap_err();
        assert!(matches!(err, PdfError::Failed(_)), "{err}");
        assert!(err.to_string().contains("pdfinfo"), "{err}");

        // A non-executable renderer.
        let dir = temp_dir("no-exec");
        let path = dir.join("pdftocairo");
        std::fs::write(&path, "#!/bin/sh\nexit 0\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        }
        let err = PdfRenderer::new(path, Duration::from_secs(5)).unwrap_err();
        assert!(err.to_string().contains("not executable"), "{err}");
    }

    #[test]
    fn render_runs_probe_then_renderer() {
        let (_tools, program) = fake_pair("ok", INFO_OK, RENDER_OK);
        let renderer = PdfRenderer::new(program.clone(), Duration::from_secs(10)).unwrap();
        assert_eq!(
            renderer.probe_program,
            program.parent().unwrap().join("pdfinfo")
        );
        let rendered = renderer.render_page(b"%PDF- fake", 2).unwrap();
        assert!(rendered.png.starts_with(b"\x89PNG\r\n\x1a\n"));
        assert_eq!(rendered.pages, 3);
        assert_eq!(rendered.page, 2);
        assert_eq!((rendered.page_width, rendered.page_height), (612.0, 792.0));
    }

    #[test]
    fn render_asks_the_renderer_for_the_crop_box() {
        // The probe reports the crop box as the page size
        // and the scale is clamped for that size, so the
        // renderer is asked to draw the crop box too: its
        // default, the media box, can be larger. The fake
        // renderer records its argv, one argument per line.
        let dir = temp_dir("argv");
        let argv = dir.join("argv");
        write_tool(&dir, "pdfinfo", INFO_OK);
        let render = format!(
            "#!/bin/sh\ncat > /dev/null\nprintf '%s\\n' \"$@\" > '{}'\nprintf '\\211PNG\\015\\012\\032\\012fakepng'\nexit 0\n",
            argv.display()
        );
        let program = write_tool(&dir, "pdftocairo", &render);
        let renderer = PdfRenderer::new(program, Duration::from_secs(10)).unwrap();
        renderer.render_page(b"%PDF- fake", 2).unwrap();
        let recorded = std::fs::read_to_string(&argv).unwrap();
        assert_eq!(
            recorded.lines().collect::<Vec<_>>(),
            [
                "-png",
                "-singlefile",
                "-cropbox",
                "-f",
                "2",
                "-l",
                "2",
                "-r",
                "144",
                "-",
                "-"
            ]
        );
    }

    #[test]
    fn render_refuses_a_password_document_from_the_probe() {
        let (_tools, program) = fake_pair(
            "password",
            "#!/bin/sh\ncat > /dev/null\nprintf 'Command Line Error: Incorrect password\\n' >&2\nexit 1\n",
            RENDER_OK,
        );
        let renderer = PdfRenderer::new(program, Duration::from_secs(10)).unwrap();
        let err = renderer.render_page(b"%PDF- fake", 1).unwrap_err();
        assert!(matches!(err, PdfError::Unsupported(_)), "{err}");
        assert!(err.to_string().contains("password-protected"), "{err}");
    }

    #[test]
    fn render_maps_an_unreadable_probe_to_not_pdf() {
        let (_tools, program) = fake_pair(
            "junk",
            "#!/bin/sh\ncat > /dev/null\nprintf 'Syntax Error: Couldn't find trailer dictionary\\n' >&2\nexit 1\n",
            RENDER_OK,
        );
        let renderer = PdfRenderer::new(program, Duration::from_secs(10)).unwrap();
        let err = renderer.render_page(b"junk", 1).unwrap_err();
        assert!(matches!(err, PdfError::NotPdf(_)), "{err}");
    }

    #[test]
    fn render_refuses_a_page_too_large_for_the_minimum_dpi_before_rendering() {
        // A page past the raster limits even at 1 dpi, the
        // lowest resolution the renderer is asked for, is
        // refused from the probe's size: the renderer never
        // runs (its marker file stays absent).
        for size in [
            "2e+06 x 2e+06",
            "1.5e+06 x 10",
            "10 x 1.5e+06",
            "580000 x 580000",
        ] {
            let dir = temp_dir("too-large");
            let marker = dir.join("rendered");
            let info = format!(
                "#!/bin/sh\ncat > /dev/null\nprintf '%s\\n' 'Pages:           1' 'Encrypted:       no' 'Page    1 size:  {size} pts'\nexit 0\n"
            );
            let render = format!(
                "#!/bin/sh\ncat > /dev/null\ntouch \"{}\"\nprintf '\\211PNG\\015\\012\\032\\012fakepng'\nexit 0\n",
                marker.display()
            );
            write_tool(&dir, "pdfinfo", &info);
            let program = write_tool(&dir, "pdftocairo", &render);
            let renderer = PdfRenderer::new(program, Duration::from_secs(10)).unwrap();
            let err = renderer.render_page(b"%PDF- fake", 1).unwrap_err();
            assert!(matches!(err, PdfError::NotPdf(_)), "{size}: {err}");
            assert!(err.to_string().contains("too large"), "{size}: {err}");
            assert!(!marker.exists(), "{size}: the renderer must not run");
        }
    }

    #[test]
    fn render_refuses_out_of_range_and_over_cap_before_rendering() {
        // The renderer records its invocation in a marker
        // file; a pre-check refusal must never reach it.
        let dir = temp_dir("precheck");
        let marker = dir.join("rendered");
        write_tool(&dir, "pdfinfo", INFO_OK);
        let render = format!(
            "#!/bin/sh\ncat > /dev/null\ntouch '{}'\nprintf '\\211PNG\\015\\012\\032\\012fakepng'\nexit 0\n",
            marker.display()
        );
        let program = write_tool(&dir, "pdftocairo", &render);
        let renderer = PdfRenderer::new(program, Duration::from_secs(10)).unwrap();
        let err = renderer.render_page(b"%PDF- fake", 4).unwrap_err();
        assert!(matches!(err, PdfError::Unsupported(_)), "{err}");
        assert!(err.to_string().contains("page 4"), "{err}");
        assert!(err.to_string().contains("3 pages"), "{err}");
        assert!(!marker.exists(), "the renderer must not run");

        // Over the page cap: the probe answers, the
        // document is refused as a bad source.
        let (_tools, program) = fake_pair(
            "overcap",
            "#!/bin/sh\ncat > /dev/null\nprintf '%s\\n' 'Pages:           5000' 'Encrypted:       no' 'Page size:       612 x 792 pts'\nexit 0\n",
            RENDER_OK,
        );
        let renderer = PdfRenderer::new(program, Duration::from_secs(10)).unwrap();
        let err = renderer.render_page(b"%PDF- fake", 1).unwrap_err();
        assert!(matches!(err, PdfError::NotPdf(_)), "{err}");
        assert!(err.to_string().contains("5000"), "{err}");
        assert!(err.to_string().contains("1000"), "{err}");
    }

    #[test]
    fn probe_range_refusal_falls_back_to_the_whole_document() {
        // The reference probe refuses a range past the
        // end of the document (exit 99, "Wrong page
        // range"), so the probe confirms the document
        // with an unranged run and answers with the
        // range refusal and its real count; the renderer
        // never runs (its marker file stays absent).
        let dir = temp_dir("range-fallback");
        let marker = dir.join("rendered");
        let info = "#!/bin/sh\ncat > /dev/null\nif [ \"$1\" = \"-f\" ] && [ \"$2\" -gt 3 ]; then\n  printf '%s\\n' 'Command Line Error: Wrong page range given: the first page (9) can not be after the last page (3).' >&2\n  exit 99\nfi\nprintf '%s\\n' 'Pages:           3' 'Encrypted:       no' 'Page size:       612 x 792 pts (letter)'\nexit 0\n".to_string();
        let render = format!(
            "#!/bin/sh\ncat > /dev/null\ntouch \"{}\"\nprintf '\\211PNG\\r\\n\\032\\n'\nexit 0\n",
            marker.display()
        );
        let (_tools, program) = fake_pair("range-fallback", &info, &render);
        let renderer = PdfRenderer::new(program, Duration::from_secs(10)).unwrap();
        let err = renderer.render_page(b"%PDF- fake", 9).unwrap_err();
        assert!(matches!(err, PdfError::Unsupported(_)), "{err}");
        assert!(err.to_string().contains("page 9"), "{err}");
        assert!(err.to_string().contains("3 pages"), "{err}");
        assert!(!marker.exists(), "the renderer must not run");
        // An in-range page still probes and renders.
        let rendered = renderer.render_page(b"%PDF- fake", 2).unwrap();
        assert_eq!(rendered.pages, 3);
        assert!(marker.exists());
    }

    #[test]
    fn render_maps_renderer_failure_classes() {
        // Exit 99 is the reference renderer's own
        // out-of-range answer: the same class as the
        // pre-check, should the two ever disagree.
        let (_tools, program) = fake_pair(
            "exit99",
            "#!/bin/sh\ncat > /dev/null\nprintf '%s\\n' 'Pages:           100' 'Encrypted:       no' 'Page size:       612 x 792 pts'\nexit 0\n",
            "#!/bin/sh\ncat > /dev/null\nprintf 'Wrong page range given\\n' >&2\nexit 99\n",
        );
        let renderer = PdfRenderer::new(program, Duration::from_secs(10)).unwrap();
        let err = renderer.render_page(b"%PDF- fake", 50).unwrap_err();
        assert!(matches!(err, PdfError::Unsupported(_)), "{err}");

        // A password refusal at render time lands in the
        // password class too.
        let (_tools, program) = fake_pair(
            "render-password",
            INFO_OK,
            "#!/bin/sh\ncat > /dev/null\nprintf 'Command Line Error: Incorrect password\\n' >&2\nexit 1\n",
        );
        let renderer = PdfRenderer::new(program, Duration::from_secs(10)).unwrap();
        let err = renderer.render_page(b"%PDF- fake", 1).unwrap_err();
        assert!(matches!(err, PdfError::Unsupported(_)), "{err}");
        assert!(err.to_string().contains("password-protected"), "{err}");

        // Any other nonzero exit is a tool failure, with
        // the stderr excerpt in the cause.
        let (_tools, program) = fake_pair(
            "failing",
            INFO_OK,
            "#!/bin/sh\ncat > /dev/null\necho 'boom on stderr' >&2\nexit 3\n",
        );
        let renderer = PdfRenderer::new(program, Duration::from_secs(10)).unwrap();
        let err = renderer.render_page(b"%PDF- fake", 1).unwrap_err();
        assert!(matches!(err, PdfError::Failed(_)), "{err}");
        assert!(err.to_string().contains("boom on stderr"), "{err}");

        // A clean run with no output, and one whose
        // output is not a PNG, are tool failures.
        let (_tools, program) = fake_pair("empty", INFO_OK, "#!/bin/sh\ncat > /dev/null\nexit 0\n");
        let renderer = PdfRenderer::new(program, Duration::from_secs(10)).unwrap();
        let err = renderer.render_page(b"%PDF- fake", 1).unwrap_err();
        assert!(matches!(err, PdfError::Failed(_)), "{err}");
        assert!(err.to_string().contains("no output"), "{err}");

        let (_tools, program) = fake_pair(
            "notpng",
            INFO_OK,
            "#!/bin/sh\ncat > /dev/null\nprintf 'definitely not a png'\nexit 0\n",
        );
        let renderer = PdfRenderer::new(program, Duration::from_secs(10)).unwrap();
        let err = renderer.render_page(b"%PDF- fake", 1).unwrap_err();
        assert!(matches!(err, PdfError::Failed(_)), "{err}");
        assert!(err.to_string().contains("not a PNG"), "{err}");
    }

    #[test]
    fn refusals_keep_their_class_when_stderr_arrives_late() {
        // Issue #566: these classes come from the tools'
        // stderr, which the run gave up on 200 ms after the
        // tool exited. Held back past that, stderr still
        // arrives, and no refusal reads as `NotPdf`.
        let _hold = crate::toolrun::hold_readers(Duration::ZERO, Duration::from_millis(500));
        let password = "#!/bin/sh\ncat > /dev/null\nprintf 'Command Line Error: Incorrect password\\n' >&2\nexit 1\n";

        // A password refusal from the probe.
        let (_tools, program) = fake_pair("late-password", password, RENDER_OK);
        let renderer = PdfRenderer::new(program, Duration::from_secs(10)).unwrap();
        let err = renderer.render_page(b"%PDF- fake", 1).unwrap_err();
        assert!(matches!(err, PdfError::Unsupported(_)), "{err}");
        assert!(err.to_string().contains("password-protected"), "{err}");

        // The probe's page-range refusal, which the
        // whole-document probe confirms.
        let info = "#!/bin/sh\ncat > /dev/null\nif [ \"$1\" = \"-f\" ]; then\n  printf '%s\\n' 'Command Line Error: Wrong page range given: the first page (9) can not be after the last page (3).' >&2\n  exit 99\nfi\nprintf '%s\\n' 'Pages:           3' 'Encrypted:       no' 'Page size:       612 x 792 pts (letter)'\nexit 0\n";
        let (_tools, program) = fake_pair("late-range", info, RENDER_OK);
        let renderer = PdfRenderer::new(program, Duration::from_secs(10)).unwrap();
        let err = renderer.render_page(b"%PDF- fake", 9).unwrap_err();
        assert!(matches!(err, PdfError::Unsupported(_)), "{err}");
        assert!(err.to_string().contains("page 9"), "{err}");
        assert!(err.to_string().contains("3 pages"), "{err}");

        // A password refusal from the renderer.
        let (_tools, program) = fake_pair("late-render-password", INFO_OK, password);
        let renderer = PdfRenderer::new(program, Duration::from_secs(10)).unwrap();
        let err = renderer.render_page(b"%PDF- fake", 1).unwrap_err();
        assert!(matches!(err, PdfError::Unsupported(_)), "{err}");
        assert!(err.to_string().contains("password-protected"), "{err}");

        // Any other renderer failure quotes the end of
        // stderr.
        let (_tools, program) = fake_pair(
            "late-failing",
            INFO_OK,
            "#!/bin/sh\ncat > /dev/null\necho 'boom on stderr' >&2\nexit 3\n",
        );
        let renderer = PdfRenderer::new(program, Duration::from_secs(10)).unwrap();
        let err = renderer.render_page(b"%PDF- fake", 1).unwrap_err();
        assert!(matches!(err, PdfError::Failed(_)), "{err}");
        assert!(err.to_string().contains("boom on stderr"), "{err}");
    }

    #[test]
    fn run_caps_tool_output() {
        let (_tools, program) = fake_pair(
            "chatty",
            INFO_OK,
            "#!/bin/sh\ncat > /dev/null\ndd if=/dev/zero bs=1024 count=4 2>/dev/null\nexit 0\n",
        );
        let renderer = PdfRenderer::new(program.clone(), Duration::from_secs(10)).unwrap();
        let err = renderer
            .run(
                &program,
                "renderer",
                &[],
                b"input",
                1024,
                Instant::now() + Duration::from_secs(10),
            )
            .unwrap_err();
        assert!(matches!(err, PdfError::Failed(_)), "{err}");
        assert!(err.to_string().contains("byte cap"), "{err}");
    }

    #[test]
    fn render_times_out_on_a_hung_renderer() {
        let (_tools, program) = fake_pair("sleeper", INFO_OK, "#!/bin/sh\nsleep 30\n");
        let renderer = PdfRenderer::new(program, Duration::from_secs(1)).unwrap();
        let started = Instant::now();
        let err = renderer.render_page(b"%PDF- fake", 1).unwrap_err();
        assert!(matches!(err, PdfError::Failed(_)), "{err}");
        assert!(err.to_string().contains("timed out"), "{err}");
        assert!(started.elapsed() < Duration::from_secs(10));
    }

    /// A minimal two-page PDF written by hand: page 1 a
    /// full-page red fill, page 2 blue, Letter pages with
    /// uncompressed content streams. Deterministic bytes,
    /// so the real-tool test needs no generator.
    fn two_page_pdf() -> Vec<u8> {
        let content = |color: &str| format!("{color} rg\n0 0 612 792 re\nf\n");
        pdf_document(&[
            ("/MediaBox [0 0 612 792]", content("1 0 0")),
            ("/MediaBox [0 0 612 792]", content("0 0 1")),
        ])
    }

    /// A PDF written by hand, one page per entry: the
    /// page dictionary's box (and rotation) entries, and
    /// its uncompressed content stream.
    fn pdf_document(pages: &[(&str, String)]) -> Vec<u8> {
        let kids: Vec<String> = (0..pages.len())
            .map(|i| format!("{} 0 R", 3 + 2 * i))
            .collect();
        let mut objects: Vec<String> = vec![
            "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
            format!(
                "<< /Type /Pages /Kids [{}] /Count {} >>",
                kids.join(" "),
                pages.len()
            ),
        ];
        for (i, (boxes, content)) in pages.iter().enumerate() {
            objects.push(format!(
                "<< /Type /Page /Parent 2 0 R {boxes} /Contents {} 0 R /Resources << >> >>",
                4 + 2 * i
            ));
            objects.push(format!(
                "<< /Length {} >>\nstream\n{content}endstream",
                content.len()
            ));
        }
        let mut out = Vec::new();
        out.extend_from_slice(b"%PDF-1.4\n");
        let mut offsets = Vec::new();
        for (i, obj) in objects.iter().enumerate() {
            offsets.push(out.len());
            out.extend_from_slice(format!("{} 0 obj\n{}\nendobj\n", i + 1, obj).as_bytes());
        }
        let xref = out.len();
        out.extend_from_slice(format!("xref\n0 {}\n", objects.len() + 1).as_bytes());
        out.extend_from_slice(b"0000000000 65535 f \n");
        for off in offsets {
            out.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
        }
        out.extend_from_slice(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
                objects.len() + 1
            )
            .as_bytes(),
        );
        out
    }

    /// A PNG's width and height from its header, without
    /// decoding the raster.
    fn png_size(png: &[u8]) -> (u32, u32) {
        assert!(png.starts_with(b"\x89PNG\r\n\x1a\n") && png.len() >= 24);
        let be = |at: usize| u32::from_be_bytes(png[at..at + 4].try_into().unwrap());
        (be(16), be(20))
    }

    /// The reference tools' directory, resolved from
    /// PATH (the renderer validates real file paths, it
    /// does not search).
    fn poppler_renderer() -> Option<PathBuf> {
        let path = std::env::var_os("PATH")?;
        std::env::split_paths(&path)
            .map(|dir| dir.join("pdftocairo"))
            .find(|candidate| candidate.is_file())
    }

    /// End to end against the real Poppler when it is on
    /// PATH. A machine without it skips this test and the
    /// other `real_tools_` tests. rust.yml, which runs the
    /// unit tests, does not install Poppler, so CI skips
    /// them unless the runner image ships it. functional.yml
    /// installs it for the HTTP suite, which runs no unit
    /// tests.
    /// The fixture's pages are solid colors, so the
    /// selected page is provable from its pixels.
    #[test]
    fn real_tools_render_the_selected_pages() {
        let Some(program) = poppler_renderer() else {
            return;
        };
        let renderer = PdfRenderer::new(program, Duration::from_secs(30)).unwrap();
        let bytes = two_page_pdf();

        let probe = renderer.probe(&bytes, 1).unwrap();
        assert_eq!(probe.pages, 2);
        assert!(!probe.encrypted);
        assert_eq!((probe.page_width, probe.page_height), (612.0, 792.0));

        let center = |page: &RenderedPage| {
            let img = image::load_from_memory(&page.png).unwrap().to_rgb8();
            assert_eq!((img.width(), img.height()), (1224, 1584));
            let p = img.get_pixel(612, 792);
            (p[0], p[1], p[2])
        };
        let (r, g, b) = center(&renderer.render_page(&bytes, 1).unwrap());
        assert!(
            r > 200 && g < 60 && b < 60,
            "page 1 is red: {:?}",
            (r, g, b)
        );
        let (r, g, b) = center(&renderer.render_page(&bytes, 2).unwrap());
        assert!(
            b > 200 && r < 60 && g < 60,
            "page 2 is blue: {:?}",
            (r, g, b)
        );

        // Page 3 of 2 is the documented refusal, not the
        // last page served under another name.
        let err = renderer.render_page(&bytes, 3).unwrap_err();
        assert!(matches!(err, PdfError::Unsupported(_)), "{err}");
        assert!(err.to_string().contains("page 3"), "{err}");

        // Junk bytes are not a PDF, from the real probe.
        let err = renderer.render_page(b"not a pdf at all", 1).unwrap_err();
        assert!(matches!(err, PdfError::NotPdf(_)), "{err}");
    }

    /// End to end against the real Poppler: a page whose
    /// media box is ten times its crop box on each side.
    /// The probe reports the crop box, the clamp is sized
    /// from it, and the raster is that same box. Drawn
    /// whole at the scale the crop box allows, the 4100 pt
    /// media box would be 8200 px square, 67.24 Mpx, past
    /// the 64 Mpx area clamp.
    #[test]
    fn real_tools_draw_the_crop_box_the_clamp_is_sized_from() {
        let Some(program) = poppler_renderer() else {
            return;
        };
        let renderer = PdfRenderer::new(program, Duration::from_secs(60)).unwrap();
        // The media box is blue and the crop box, at the
        // origin, red: a raster with any blue in it drew
        // past the crop box.
        let bytes = pdf_document(&[(
            "/MediaBox [0 0 4100 4100] /CropBox [0 0 410 410]",
            "0 0 1 rg\n0 0 4100 4100 re\nf\n1 0 0 rg\n0 0 410 410 re\nf\n".to_string(),
        )]);
        let probe = renderer.probe(&bytes, 1).unwrap();
        assert_eq!((probe.page_width, probe.page_height), (410.0, 410.0));
        assert_eq!(render_scale(410.0, 410.0), RENDER_SCALE);

        let page = renderer.render_page(&bytes, 1).unwrap();
        let (width, height) = png_size(&page.png);
        assert_eq!(
            (width, height),
            (820, 820),
            "the raster is the crop box at the clamped scale"
        );
        assert!(f64::from(width) * f64::from(height) <= MAX_RASTER_PIXELS);
        assert_eq!((page.page_width, page.page_height), (410.0, 410.0));
        let img = image::load_from_memory(&page.png).unwrap().to_rgb8();
        for (x, y) in [(0, 0), (819, 0), (0, 819), (819, 819), (410, 410)] {
            let p = img.get_pixel(x, y);
            assert!(
                p[0] > 200 && p[1] < 60 && p[2] < 60,
                "({x}, {y}) is inside the crop box: {p:?}"
            );
        }
    }

    /// End to end against the real Poppler: a page whose
    /// crop box is its media box but too long to fit 16382
    /// px even at 1 dpi is refused from the probe's size,
    /// before the renderer draws it past the axis limit.
    #[test]
    fn real_tools_refuse_a_page_too_large_for_the_minimum_dpi() {
        let Some(program) = poppler_renderer() else {
            return;
        };
        let renderer = PdfRenderer::new(program, Duration::from_secs(30)).unwrap();
        let bytes = pdf_document(&[(
            "/MediaBox [0 0 1500000 10]",
            "0 0 1 rg\n0 0 1500000 10 re\nf\n".to_string(),
        )]);
        let probe = renderer.probe(&bytes, 1).unwrap();
        assert_eq!((probe.page_width, probe.page_height), (1_500_000.0, 10.0));
        let err = renderer.render_page(&bytes, 1).unwrap_err();
        assert!(matches!(err, PdfError::NotPdf(_)), "{err}");
        assert!(err.to_string().contains("too large"), "{err}");
    }

    /// End to end against the real Poppler: a page whose
    /// crop box is its media box, stated or by default, and
    /// a rotated one, render the same bytes as the renderer
    /// run without `-cropbox`, the invocation before the
    /// renderer was asked for the crop box.
    #[test]
    fn real_tools_render_pages_with_matching_boxes_as_before() {
        let Some(program) = poppler_renderer() else {
            return;
        };
        let renderer = PdfRenderer::new(program.clone(), Duration::from_secs(30)).unwrap();
        let fill = |color: &str, w: u32, h: u32| format!("{color} rg\n0 0 {w} {h} re\nf\n");
        let stated = pdf_document(&[
            (
                "/MediaBox [0 0 612 792] /CropBox [0 0 612 792]",
                fill("0 1 0", 300, 400),
            ),
            (
                "/MediaBox [0 0 595 842] /Rotate 90",
                fill("1 0 1", 200, 600),
            ),
        ]);
        for bytes in [two_page_pdf(), stated] {
            for page in 1..=2u32 {
                let rendered = renderer.render_page(&bytes, page).unwrap();
                let page_arg = page.to_string();
                let before = renderer
                    .run(
                        &program,
                        "renderer",
                        &[
                            "-png",
                            "-singlefile",
                            "-f",
                            &page_arg,
                            "-l",
                            &page_arg,
                            "-r",
                            "144",
                            "-",
                            "-",
                        ],
                        &bytes,
                        MAX_PAGE_PNG_BYTES,
                        Instant::now() + Duration::from_secs(30),
                    )
                    .unwrap();
                assert!(before.status.success(), "{}", before.stderr_excerpt());
                assert!(
                    rendered.png == before.stdout,
                    "page {page} renders the bytes it rendered before"
                );
            }
        }
    }

    /// A JPEG 2000 SOC and SIZ pair declaring an image of
    /// `x` by `y` on the reference grid, from `(x0, y0)`,
    /// with one `(XRsiz, YRsiz)` subsampling per component.
    fn jpx_siz(x: u32, y: u32, x0: u32, y0: u32, components: &[(u8, u8)]) -> Vec<u8> {
        let mut out = JPX_SOC_SIZ.to_vec();
        let count = u16::try_from(components.len()).unwrap();
        out.extend_from_slice(&(38 + 3 * count).to_be_bytes());
        out.extend_from_slice(&0u16.to_be_bytes());
        for value in [x, y, x0, y0, x, y, 0, 0] {
            out.extend_from_slice(&value.to_be_bytes());
        }
        out.extend_from_slice(&count.to_be_bytes());
        for &(dx, dy) in components {
            out.extend_from_slice(&[7, dx, dy]);
        }
        out
    }

    const RGB: &[(u8, u8)] = &[(1, 1), (1, 1), (1, 1)];

    #[test]
    fn the_scan_reads_each_jpeg_2000_image_size_against_the_decode_budget() {
        // The budget is the engine's: 256 MiB at four bytes a
        // pixel (8192x8192 fits exactly), or one byte a
        // sample past four components.
        let at = |bytes: Vec<u8>| oversized_jpx(&bytes);
        assert_eq!(at(jpx_siz(8192, 8192, 0, 0, RGB)), None);
        let over = at(jpx_siz(16000, 16000, 0, 0, RGB)).expect("16000x16000 is over");
        assert_eq!(
            (over.width, over.height, over.components),
            (16000, 16000, 3)
        );
        assert_eq!(over.samples, 3 * 16000 * 16000);
        assert!(at(jpx_siz(8193, 8192, 0, 0, RGB)).is_some());
        // The image is the grid less its offset.
        assert_eq!(at(jpx_siz(16000, 8192, 8000, 0, RGB)), None);
        assert!(at(jpx_siz(16000, 8192, 7000, 0, RGB)).is_some());
        // Past four components, the samples count: sixteen
        // at 4096x4096 fit, seventeen do not.
        assert_eq!(at(jpx_siz(4096, 4096, 0, 0, &[(1, 1); 16])), None);
        assert!(at(jpx_siz(4096, 4096, 0, 0, &[(1, 1); 17])).is_some());
        // A subsampled component holds fewer samples, but the
        // pixels alone can pass the budget.
        let half = at(jpx_siz(9000, 9000, 0, 0, &[(1, 1), (2, 2), (2, 2)])).unwrap();
        assert_eq!(half.samples, 81_000_000 + 2 * 4500 * 4500);
        // The image may sit anywhere in the document, after
        // stray marker bytes, and a later one counts too.
        let mut document = b"%PDF-1.7 \xff\xff\x4f\xff junk".to_vec();
        document.extend_from_slice(&jpx_siz(64, 64, 0, 0, RGB));
        document.extend_from_slice(b" more ");
        document.extend_from_slice(&jpx_siz(20000, 20000, 0, 0, &[(1, 1)]));
        assert_eq!(oversized_jpx(&document).map(|s| s.width), Some(20000));
    }

    #[test]
    fn the_scan_ignores_marker_bytes_that_are_not_a_size_segment() {
        let big = jpx_siz(16000, 16000, 0, 0, RGB);
        // The length must match the component count.
        let mut wrong_length = big.clone();
        wrong_length[5] += 1;
        assert_eq!(oversized_jpx(&wrong_length), None);
        // No components, or a zero subsampling.
        assert_eq!(oversized_jpx(&jpx_siz(16000, 16000, 0, 0, &[])), None);
        assert_eq!(
            oversized_jpx(&jpx_siz(16000, 16000, 0, 0, &[(1, 1), (0, 1)])),
            None
        );
        // An empty image (no extent past its offset).
        assert_eq!(oversized_jpx(&jpx_siz(16000, 16000, 16000, 0, RGB)), None);
        // Cut short anywhere.
        for end in 0..big.len() {
            assert_eq!(oversized_jpx(&big[..end]), None, "cut at {end}");
        }
        assert!(oversized_jpx(&big).is_some());
    }

    /// The JPEG 2000 file (a JP2 box sequence around its
    /// codestream) FFmpeg's OpenJPEG encoder writes for one
    /// 16000x16000 gray RGB frame.
    const JPX_16000: &str = "0000000c6a5020200d0a870a00000014667479706a703220000000006a703220\
0000002d6a703268000000166968647200003e8000003e800003070700000000000f636f6c720100000000\
0010000000a76a703263ff4fff51002f000000003e8000003e80000000000000000000003e8000003e8000\
000000000000000003070101070101070101ff52000c00000001000504040001ff5c001340404848504848\
50484850484850484850ff640025000143726561746564206279204f70656e4a5045472076657273696f6e\
20322e352e30ff90000a0000000000200001ff93808080808080808080808080808080808080ffd9";

    fn jpx_16000() -> Vec<u8> {
        (0..JPX_16000.len())
            .step_by(2)
            .map(|at| u8::from_str_radix(&JPX_16000[at..at + 2], 16).unwrap())
            .collect()
    }

    /// A one-page 200x200 pt document that draws `image`, a
    /// 16000x16000 RGB image XObject through `filter`.
    fn image_pdf(filter: &str, image: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"%PDF-1.5\n");
        let content = b"q 200 0 0 200 0 0 cm /Im0 Do Q";
        let mut objects: Vec<Vec<u8>> = vec![
            b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] \
              /Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>"
                .to_vec(),
        ];
        let mut stream = format!("<< /Length {} >>\nstream\n", content.len()).into_bytes();
        stream.extend_from_slice(content);
        stream.extend_from_slice(b"\nendstream");
        objects.push(stream);
        let mut xobject = format!(
            "<< /Type /XObject /Subtype /Image /Width 16000 /Height 16000 \
             /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter {filter} /Length {} >>\nstream\n",
            image.len()
        )
        .into_bytes();
        xobject.extend_from_slice(image);
        xobject.extend_from_slice(b"\nendstream");
        objects.push(xobject);
        let mut offsets = Vec::new();
        for (i, object) in objects.iter().enumerate() {
            offsets.push(out.len());
            out.extend_from_slice(format!("{} 0 obj\n", i + 1).as_bytes());
            out.extend_from_slice(object);
            out.extend_from_slice(b"\nendobj\n");
        }
        let xref = out.len();
        out.extend_from_slice(format!("xref\n0 {}\n", objects.len() + 1).as_bytes());
        out.extend_from_slice(b"0000000000 65535 f \n");
        for off in offsets {
            out.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
        }
        out.extend_from_slice(
            format!(
                "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
                objects.len() + 1
            )
            .as_bytes(),
        );
        out
    }

    #[test]
    fn a_document_with_a_jpeg_2000_image_past_the_decode_budget_is_refused_before_any_tool_runs() {
        // Both fake tools mark that they ran; neither may.
        let dir = temp_dir("jpx-budget");
        let marker = dir.join("ran");
        let mark = |body: &str| {
            format!(
                "#!/bin/sh\ncat > /dev/null\ntouch \"{}\"\n{body}",
                marker.display()
            )
        };
        write_tool(
            &dir,
            "pdfinfo",
            &mark(
                "printf '%s\\n' 'Pages:           1' 'Encrypted:       no' 'Page    1 size:  200 x 200 pts'\nexit 0\n",
            ),
        );
        let program = write_tool(
            &dir,
            "pdftocairo",
            &mark("printf '\\211PNG\\015\\012\\032\\012fakepng'\nexit 0\n"),
        );
        let renderer = PdfRenderer::new(program, Duration::from_secs(10)).unwrap();
        let err = renderer
            .render_page(&image_pdf("/JPXDecode", &jpx_16000()), 1)
            .unwrap_err();
        assert!(matches!(err, PdfError::NotPdf(_)), "{err}");
        assert!(
            err.to_string()
                .contains("16000x16000 JPEG 2000 image with 3 components"),
            "{err}"
        );
        assert!(!marker.exists(), "no tool may run");
        // A document whose images fit runs as before.
        let small = image_pdf("/JPXDecode", &jpx_siz(640, 480, 0, 0, RGB));
        renderer.render_page(&small, 1).unwrap();
        assert!(marker.exists());
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn the_pdf_tools_run_under_the_renderer_address_space_cap() {
        // Each fake tool records its own address-space cap
        // and OOM score: the renderer's cap, sized to the
        // raster and decode budgets, and the highest score.
        let dir = temp_dir("pdf-limits");
        let record = dir.join("limits.log");
        let report = format!(
            "sleep 0.3\n\
             grep '^Max address space' /proc/$$/limits | awk '{{print $4}}' >> \"{0}\"\n\
             cat /proc/$$/oom_score_adj >> \"{0}\"\n",
            record.display()
        );
        write_tool(
            &dir,
            "pdfinfo",
            &format!(
                "#!/bin/sh\ncat > /dev/null\n{report}printf '%s\\n' 'Pages:           1' 'Encrypted:       no' 'Page    1 size:  200 x 200 pts'\nexit 0\n"
            ),
        );
        let program = write_tool(
            &dir,
            "pdftocairo",
            &format!(
                "#!/bin/sh\ncat > /dev/null\n{report}printf '\\211PNG\\015\\012\\032\\012fakepng'\nexit 0\n"
            ),
        );
        let renderer = PdfRenderer::new(program, Duration::from_secs(10)).unwrap();
        renderer.render_page(b"%PDF- fake", 1).unwrap();
        let recorded = std::fs::read_to_string(&record).unwrap();
        let cap = RENDERER_MAX_ADDRESS_SPACE.to_string();
        assert_eq!(cap, (2u64 << 30).to_string());
        let expected: Vec<&str> = [cap.as_str(), "1000"].repeat(2);
        assert_eq!(
            recorded.lines().collect::<Vec<_>>(),
            expected,
            "probe then renderer: {recorded}"
        );
    }

    #[test]
    fn real_tools_refuse_a_document_with_a_jpeg_2000_image_past_the_decode_budget() {
        // A document whose JPEG 2000 image declares
        // 16000x16000 RGB is refused before any tool runs.
        let Some(program) = poppler_renderer() else {
            return;
        };
        let renderer = PdfRenderer::new(program, Duration::from_secs(60)).unwrap();
        let bytes = image_pdf("/JPXDecode", &jpx_16000());
        assert_eq!(bytes.len(), 923);
        let started = Instant::now();
        let err = renderer.render_page(&bytes, 1).unwrap_err();
        assert!(matches!(err, PdfError::NotPdf(_)), "{err}");
        assert!(
            err.to_string().contains("decode allocation budget"),
            "{err}"
        );
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    /// The same image behind an ASCIIHex filter, where the
    /// scan cannot see its size: the renderer's address-space
    /// cap stops the decode, about a gigabyte in, and the
    /// renderer leaves the image out of the page, which
    /// stays white instead of the image's gray. Ignored by
    /// default for that gigabyte. Under a cap that holds the
    /// whole decode, the page comes back gray.
    #[test]
    #[ignore = "allocates about 1 GiB in the renderer before its cap stops it"]
    fn real_renderer_leaves_out_an_image_whose_decode_passes_its_cap() {
        let Some(program) = poppler_renderer() else {
            return;
        };
        let renderer = PdfRenderer::new(program, Duration::from_secs(60)).unwrap();
        let mut hex: Vec<u8> = jpx_16000()
            .iter()
            .flat_map(|b| format!("{b:02x}").into_bytes())
            .collect();
        hex.push(b'>');
        let bytes = image_pdf("[/ASCIIHexDecode /JPXDecode]", &hex);
        assert_eq!(oversized_jpx(&bytes), None, "the scan cannot see it");
        let page = renderer.render_page(&bytes, 1).unwrap();
        let img = image::load_from_memory(&page.png).unwrap().to_rgb8();
        assert_eq!((img.width(), img.height()), (400, 400));
        let p = img.get_pixel(200, 200);
        assert_eq!((p[0], p[1], p[2]), (255, 255, 255), "the image was drawn");
    }
}
