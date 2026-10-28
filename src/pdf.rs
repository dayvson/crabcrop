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
//!   state, and the selected page's size in points.
//!   Facts the design needs before rendering are probe
//!   facts, so the probe runs first, every time.
//! - Render: the configured program (Poppler's
//!   `pdftocairo` is the reference), run as
//!   `-png -singlefile -f <n> -l <n> -r <dpi> - -`: the
//!   PDF on stdin, one PNG on stdout, at a fixed scale
//!   of 2.0 (144 DPI) clamped so neither raster axis
//!   exceeds the decode cap.
//!
//! The rendered PNG re-enters the image pipeline as
//! the source, so every downstream budget (decode
//! dimensions and allocation, output pixels) applies to
//! it unchanged.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

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

/// The decode cap a rendered raster axis may not
/// exceed, in pixels: the same 16384 the decode chain
/// enforces, applied before the tool runs so a poster
/// page renders at a clamped scale instead of asking
/// for a raster the pipeline would refuse.
const MAX_RASTER_AXIS: f64 = 16_384.0;

/// The total raster area a render may not exceed, in
/// pixels: 64 Mpx, the raster the 256 MB decode budget
/// holds at 4 bytes per pixel. The per-axis clamp alone
/// allows 16384 squared (about 268 Mpx, over 1 GB of
/// renderer memory), which a tiny hostile MediaBox
/// reaches trivially (issue #262).
const MAX_RASTER_PIXELS: f64 = 64_000_000.0;

/// How much of a probe report the parser keeps. The
/// report is a few hundred bytes; this is generous
/// headroom for chatty builds.
const PROBE_REPORT_CAP: usize = 1024 * 1024;

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
    /// The selected page's width in points.
    pub page_width: f64,
    /// The selected page's height in points.
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
    pub fn probe(&self, pdf: &[u8], page: u32) -> Result<PdfProbe, PdfError> {
        let run = self.run_probe(pdf, Some(page))?;
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
            let whole = self.run_probe(pdf, None)?;
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
    fn run_probe(&self, pdf: &[u8], page: Option<u32>) -> Result<ToolRun, PdfError> {
        match page {
            Some(page) => {
                let page_arg = page.to_string();
                self.run(
                    &self.probe_program,
                    "probe",
                    &["-f", &page_arg, "-l", &page_arg, "-"],
                    pdf,
                    PROBE_REPORT_CAP,
                )
            }
            None => self.run(&self.probe_program, "probe", &["-"], pdf, PROBE_REPORT_CAP),
        }
    }

    /// Render the selected page as PNG bytes. Probes
    /// first: the page count cap and the page range are
    /// enforced against the probe before the renderer
    /// runs, and the render scale is computed from the
    /// probed page size.
    pub fn render_page(&self, pdf: &[u8], page: u32) -> Result<RenderedPage, PdfError> {
        let probe = self.probe(pdf, page)?;
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
        let scale = render_scale(probe.page_width, probe.page_height);
        // Floor the DPI so the clamped scale can never
        // round a raster axis past the decode cap; at the
        // fixed scale this is exactly 144.
        let dpi = (scale * 72.0).floor().max(1.0);
        let dpi_arg = format!("{dpi}");
        let page_arg = page.to_string();
        let run = self.run(
            &self.program,
            "renderer",
            &[
                "-png",
                "-singlefile",
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
                stderr_excerpt(&run.stderr)
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

    /// Run one of the tools once under the standing
    /// bounds: the source bytes on stdin, stdout capped
    /// at `max_stdout` (past the cap the process group is
    /// killed and the run fails), stderr kept up to a cap
    /// while the rest is drained so the tool never blocks
    /// on a full pipe, an empty environment, and the
    /// wall-clock timeout. Mirrors the frame extractor's
    /// discipline exactly.
    fn run(
        &self,
        program: &Path,
        name: &'static str,
        args: &[&str],
        stdin_bytes: &[u8],
        max_stdout: usize,
    ) -> Result<ToolRun, PdfError> {
        let mut cmd = Command::new(program);
        cmd.args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env_clear()
            .current_dir(std::env::temp_dir());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }
        let mut child = cmd.spawn().map_err(|e| {
            PdfError::Failed(format!(
                "could not start the PDF {name} {}: {e}",
                program.display()
            ))
        })?;
        let pid = child.id();

        let mut stdin_pipe = child.stdin.take().expect("stdin was piped");
        let input = stdin_bytes.to_vec();
        let writer = std::thread::spawn(move || {
            let _ = stdin_pipe.write_all(&input);
        });

        let mut stdout_pipe = child.stdout.take().expect("stdout was piped");
        let (out_tx, out_rx) = std::sync::mpsc::channel::<Result<Vec<u8>, String>>();
        let stdout_thread = std::thread::spawn(move || {
            let mut buf = Vec::new();
            let mut chunk = [0u8; 64 * 1024];
            let result = loop {
                match stdout_pipe.read(&mut chunk) {
                    Ok(0) => break Ok(buf),
                    Ok(n) => {
                        buf.extend_from_slice(&chunk[..n]);
                        if buf.len() > max_stdout {
                            break Err(format!(
                                "the PDF {name}'s output passed the {max_stdout} byte cap"
                            ));
                        }
                    }
                    Err(e) => break Err(format!("reading the PDF {name}'s output failed: {e}")),
                }
            };
            let _ = out_tx.send(result);
        });

        let mut stderr_pipe = child.stderr.take().expect("stderr was piped");
        let (err_tx, err_rx) = std::sync::mpsc::channel::<String>();
        let stderr_thread = std::thread::spawn(move || {
            let mut kept = Vec::new();
            let mut chunk = [0u8; 16 * 1024];
            loop {
                match stderr_pipe.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => {
                        if kept.len() < PROBE_REPORT_CAP {
                            let take = (PROBE_REPORT_CAP - kept.len()).min(n);
                            kept.extend_from_slice(&chunk[..take]);
                        }
                    }
                }
            }
            let _ = err_tx.send(String::from_utf8_lossy(&kept).into_owned());
        });

        let deadline = Instant::now() + self.timeout;
        let timeout = self.timeout;
        // Wait for the exit status without ever blocking
        // past the deadline: a tool that closed its stdout
        // and then hung still gets killed on time.
        let wait_status =
            |child: &mut std::process::Child| -> Result<std::process::ExitStatus, PdfError> {
                loop {
                    match child.try_wait() {
                        Ok(Some(status)) => return Ok(status),
                        Ok(None) => {
                            if Instant::now() >= deadline {
                                kill_process_group(pid);
                                let _ = child.wait();
                                return Err(PdfError::Failed(format!(
                                    "PDF {name} timed out after {}s",
                                    timeout.as_secs()
                                )));
                            }
                            std::thread::sleep(Duration::from_millis(5));
                        }
                        Err(e) => {
                            return Err(PdfError::Failed(format!(
                                "waiting on the PDF {name} failed: {e}"
                            )));
                        }
                    }
                }
            };
        let (status, stdout) = loop {
            match out_rx.try_recv() {
                Ok(Ok(bytes)) => {
                    let status = wait_status(&mut child)?;
                    break (status, bytes);
                }
                Ok(Err(why)) => {
                    kill_process_group(pid);
                    let _ = child.wait();
                    return Err(PdfError::Failed(why));
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    if Instant::now() >= deadline {
                        kill_process_group(pid);
                        let _ = child.wait();
                        return Err(PdfError::Failed(format!(
                            "PDF {name} timed out after {}s",
                            timeout.as_secs()
                        )));
                    }
                    match child.try_wait() {
                        Ok(Some(status)) => {
                            // The process is gone; its stdout
                            // reader finishes at EOF. Give it
                            // a moment, then take what arrived.
                            match out_rx.recv_timeout(Duration::from_secs(2)) {
                                Ok(Ok(bytes)) => break (status, bytes),
                                Ok(Err(why)) => return Err(PdfError::Failed(why)),
                                Err(_) => break (status, Vec::new()),
                            }
                        }
                        Ok(None) => std::thread::sleep(Duration::from_millis(5)),
                        Err(e) => {
                            return Err(PdfError::Failed(format!(
                                "waiting on the PDF {name} failed: {e}"
                            )));
                        }
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    let status = wait_status(&mut child)?;
                    break (status, Vec::new());
                }
            }
        };
        let _ = writer.join();
        // The reader threads are deliberately never
        // joined (issue #263, the video.rs pattern): a
        // grandchild that inherited the pipes and holds
        // them open would block a join for its whole
        // lifetime, defeating the timeout above. stderr
        // is worth a short grace through its channel, no
        // more; the stdout result already arrived through
        // its own channel.
        let stderr = err_rx
            .recv_timeout(Duration::from_millis(200))
            .unwrap_or_default();
        drop(stdout_thread);
        drop(stderr_thread);
        Ok(ToolRun {
            status,
            stdout,
            stderr,
        })
    }
}

/// One finished tool run.
#[derive(Debug)]
struct ToolRun {
    status: std::process::ExitStatus,
    stdout: Vec<u8>,
    stderr: String,
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
/// neither raster axis exceeds the decode cap and the
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

/// Parse the probe facts out of a `pdfinfo` report: the
/// `Pages:` and `Encrypted:` lines and the selected
/// page's size (printed as `Page size:` for a whole-
/// document report and `Page   <n> size:` when the probe
/// names a page range, the form this probe runs).
/// Returns `None` when any fact is missing or malformed:
/// a report the parser cannot read means the document
/// is not one the pipeline can use.
pub fn parse_probe(report: &str, page: u32) -> Option<PdfProbe> {
    let mut pages = None;
    let mut encrypted = None;
    let mut size = None;
    for line in report.lines() {
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
    if width <= 0.0 || height <= 0.0 {
        return None;
    }
    Some((width, height))
}

/// The not-a-PDF refusal: bytes the probe cannot read
/// as a document at all.
fn not_pdf() -> PdfError {
    PdfError::NotPdf(
        "the source is not a PDF the probe can read (not a PDF, or a corrupt document)".to_string(),
    )
}

/// The tail of a tool's stderr, for error text: capped,
/// the same as the frame extractor's excerpts.
fn stderr_excerpt(stderr: &str) -> String {
    let tail: String = stderr
        .chars()
        .rev()
        .take(500)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    let tail = tail.trim();
    if tail.is_empty() {
        "no error output".to_string()
    } else {
        tail.to_string()
    }
}

/// Kill a runaway tool's whole process group (it may
/// have spawned helpers). Mirrors the frame extractor's
/// discipline exactly. The group kill is the whole
/// mechanism (issue #272): a descendant that detaches
/// (setsid, double-fork) escapes it by design of the
/// tool contract, which forbids daemonizing.
#[cfg(unix)]
fn kill_process_group(pid: u32) {
    if let Ok(pid) = i32::try_from(pid)
        && let Some(pid) = rustix::process::Pid::from_raw(pid)
    {
        let _ = rustix::process::kill_process_group(pid, rustix::process::Signal::KILL);
    }
}

#[cfg(not(unix))]
fn kill_process_group(_pid: u32) {}

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
    }

    #[test]
    fn scale_is_two_until_the_decode_cap_bites() {
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
        // A wide-but-short page still clamps by axis.
        let scale = render_scale(20_000.0, 100.0);
        assert!((scale - 16_384.0 / 20_000.0).abs() < 1e-9, "{scale}");
        // The hostile MediaBox from the finding: a
        // 100000 pt square page renders at 8000 px per
        // axis (64 Mpx), not 16384 (268 Mpx).
        let scale = render_scale(100_000.0, 100_000.0);
        assert!((100_000.0 * scale - 8_000.0).abs() < 1e-6, "{scale}");
        // A degenerate size keeps the fixed scale; the
        // renderer run is where that document fails.
        assert_eq!(render_scale(0.0, 0.0), 2.0);
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

    fn temp_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("crabcrop-pdf-test-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_tool(dir: &Path, name: &str, script: &str) -> PathBuf {
        let path = dir.join(name);
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(script.as_bytes()).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        path
    }

    /// A fake tool pair in one directory: the probe
    /// prints a recorded-shape report, the renderer
    /// prints a PNG. Returns the renderer path to build
    /// the extractor with.
    fn fake_pair(name: &str, info_script: &str, render_script: &str) -> PathBuf {
        let dir = temp_dir(name);
        write_tool(&dir, "pdfinfo", info_script);
        write_tool(&dir, "pdftocairo", render_script)
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
        let program = fake_pair("ok", INFO_OK, RENDER_OK);
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
    fn render_refuses_a_password_document_from_the_probe() {
        let program = fake_pair(
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
        let program = fake_pair(
            "junk",
            "#!/bin/sh\ncat > /dev/null\nprintf 'Syntax Error: Couldn't find trailer dictionary\\n' >&2\nexit 1\n",
            RENDER_OK,
        );
        let renderer = PdfRenderer::new(program, Duration::from_secs(10)).unwrap();
        let err = renderer.render_page(b"junk", 1).unwrap_err();
        assert!(matches!(err, PdfError::NotPdf(_)), "{err}");
    }

    #[test]
    fn render_refuses_out_of_range_and_over_cap_before_rendering() {
        // The renderer records its invocation in a marker
        // file; a pre-check refusal must never reach it.
        let dir = temp_dir("precheck");
        let marker = dir.join("rendered");
        write_tool(&dir, "pdfinfo", INFO_OK);
        let render = format!(
            "#!/bin/sh\ncat > /dev/null\ntouch {}\nprintf '\\211PNG\\015\\012\\032\\012fakepng'\nexit 0\n",
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
        let program = fake_pair(
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
        let program = fake_pair("range-fallback", &info, &render);
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
        let program = fake_pair(
            "exit99",
            "#!/bin/sh\ncat > /dev/null\nprintf '%s\\n' 'Pages:           100' 'Encrypted:       no' 'Page size:       612 x 792 pts'\nexit 0\n",
            "#!/bin/sh\ncat > /dev/null\nprintf 'Wrong page range given\\n' >&2\nexit 99\n",
        );
        let renderer = PdfRenderer::new(program, Duration::from_secs(10)).unwrap();
        let err = renderer.render_page(b"%PDF- fake", 50).unwrap_err();
        assert!(matches!(err, PdfError::Unsupported(_)), "{err}");

        // A password refusal at render time lands in the
        // password class too.
        let program = fake_pair(
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
        let program = fake_pair(
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
        let program = fake_pair("empty", INFO_OK, "#!/bin/sh\ncat > /dev/null\nexit 0\n");
        let renderer = PdfRenderer::new(program, Duration::from_secs(10)).unwrap();
        let err = renderer.render_page(b"%PDF- fake", 1).unwrap_err();
        assert!(matches!(err, PdfError::Failed(_)), "{err}");
        assert!(err.to_string().contains("no output"), "{err}");

        let program = fake_pair(
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
    fn run_caps_tool_output() {
        let program = fake_pair(
            "chatty",
            INFO_OK,
            "#!/bin/sh\ncat > /dev/null\ndd if=/dev/zero bs=1024 count=4 2>/dev/null\nexit 0\n",
        );
        let renderer = PdfRenderer::new(program.clone(), Duration::from_secs(10)).unwrap();
        let err = renderer
            .run(&program, "renderer", &[], b"input", 1024)
            .unwrap_err();
        assert!(matches!(err, PdfError::Failed(_)), "{err}");
        assert!(err.to_string().contains("byte cap"), "{err}");
    }

    #[test]
    fn render_times_out_on_a_hung_renderer() {
        let program = fake_pair("sleeper", INFO_OK, "#!/bin/sh\nsleep 30\n");
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
        let objects: Vec<String> = vec![
            "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
            "<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>".to_string(),
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << >> >>"
                .to_string(),
            format!(
                "<< /Length {} >>\nstream\n{}endstream",
                content("1 0 0").len(),
                content("1 0 0")
            ),
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 6 0 R /Resources << >> >>"
                .to_string(),
            format!(
                "<< /Length {} >>\nstream\n{}endstream",
                content("0 0 1").len(),
                content("0 0 1")
            ),
        ];
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
    /// PATH (CI installs it; a machine without it skips).
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
}
