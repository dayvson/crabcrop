//! Observability: one structured JSON log line per request, and
//! Prometheus metrics behind `/metrics`.
//!
//! The log never contains the full source URL, only the operation
//! summary (the path segments between the signature and the source).
//! `cache_hit` records whether the result cache answered the
//! request, as reported by the handler. The lines reach standard
//! output through a [`RequestLog`] (issue #642). A request whose
//! client goes away before its answer still leaves its line and
//! its metrics update, with the status [`CLIENT_CLOSED`] (issue
//! #707): the server's record of the request files them when it
//! is dropped unfiled.

use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

mod request_log;

#[cfg(test)]
pub(crate) use request_log::fake::FakeStdout;
pub use request_log::{QUEUE_BYTES, QueueState, RequestLog, SHUTDOWN_WAIT, STALLED_AFTER};

/// What [`stderr_line!`](crate::stderr_line) writes with.
/// `eprintln!` panics when a write to standard error fails,
/// so a reader of the pipe that went away (a log collector
/// that exited) ended the boot at its first warning, and a
/// request at its first error line. A line that cannot be
/// written is dropped instead, standard output says so once,
/// if it can, and the server goes on: the line after it is
/// tried again, so the lines come back if standard error
/// does. In the library's tests, `eprintln!` keeps the lines
/// in the test's captured output.
#[doc(hidden)]
pub fn print_stderr_line(line: std::fmt::Arguments<'_>) {
    #[cfg(test)]
    eprintln!("{line}");
    #[cfg(not(test))]
    {
        use std::io::Write as _;
        use std::sync::atomic::{AtomicBool, Ordering};
        static TOLD: AtomicBool = AtomicBool::new(false);
        // The lock on standard error is let go before the note
        // takes the one on standard output. A boot line that
        // fails holds standard output while it says so on
        // standard error, so holding both the other way round
        // could deadlock with it, and the request log's writer
        // holds standard output through a write that a slow
        // reader can stall, which would stall every stderr line
        // behind this one.
        let written = writeln!(std::io::stderr().lock(), "{line}");
        if let Err(e) = written
            && !TOLD.swap(true, Ordering::Relaxed)
        {
            let _ = writeln!(
                std::io::stdout().lock(),
                "crabcrop: cannot write to standard error: {e}; the lines it does not take are dropped"
            );
        }
    }
}

/// The status recorded for an image request whose client went away
/// before it was answered (issue #707): 499, the "client closed
/// request" of nginx and the log tools built around it. No response
/// carries it, since nobody is left to receive one. It is a 4xx, and
/// the metrics count it as one.
pub const CLIENT_CLOSED: u16 = 499;

/// Which stage failed, for the error counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailStage {
    None,
    Load,
    Engine,
}

/// Everything one request leaves behind: a log line and a metrics
/// update. Built up by the handler as the request moves through its
/// stages.
#[derive(Debug, Clone)]
pub struct RequestRecord {
    /// The request's method: `GET`, or `HEAD`, which the image
    /// route answers too (issue #707).
    pub method: &'static str,
    /// Operation summary: path segments between signature and source.
    pub ops: String,
    /// HTTP status of the response actually sent, or
    /// [`CLIENT_CLOSED`] when the client went away first.
    pub status: u16,
    pub verify: Duration,
    pub load: Duration,
    pub process: Duration,
    pub total: Duration,
    /// Whether the loader stage ran (attempted), for its averages.
    pub load_ran: bool,
    /// Whether the engine stage ran (attempted).
    pub process_ran: bool,
    /// Whether the result cache was consulted at all.
    /// Requests rejected before the lookup (bad signature,
    /// parse errors) and meta requests (which bypass the
    /// cache by design) never set it, and the hit/miss
    /// counters only move for requests that did.
    pub cache_looked_up: bool,
    pub cache_hit: bool,
    /// Whether the configured fallback image was served
    /// instead of the source.
    pub fallback: bool,
    pub bytes_in: usize,
    pub bytes_out: usize,
    pub fail: FailStage,
    pub request_id: String,
}

impl Default for RequestRecord {
    fn default() -> Self {
        Self {
            method: "GET",
            ops: String::new(),
            status: 0,
            verify: Duration::ZERO,
            load: Duration::ZERO,
            process: Duration::ZERO,
            total: Duration::ZERO,
            load_ran: false,
            process_ran: false,
            cache_looked_up: false,
            cache_hit: false,
            fallback: false,
            bytes_in: 0,
            bytes_out: 0,
            fail: FailStage::None,
            request_id: String::new(),
        }
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Escape a string for inclusion in a JSON string literal.
fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

impl RequestRecord {
    /// The structured log line for this request, as JSON.
    pub fn to_json(&self) -> String {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        format!(
            "{{\"ts\":{ts},\"method\":\"{}\",\"ops\":\"{}\",\"status\":{},\
             \"verify_ms\":{:.1},\"load_ms\":{:.1},\"process_ms\":{:.1},\"total_ms\":{:.1},\
             \"cache\":\"{}\",\"fallback\":{},\"bytes_in\":{},\"bytes_out\":{},\
             \"request_id\":\"{}\"}}",
            escape_json(self.method),
            escape_json(&self.ops),
            self.status,
            ms(self.verify),
            ms(self.load),
            ms(self.process),
            ms(self.total),
            if self.cache_hit { "hit" } else { "miss" },
            self.fallback,
            self.bytes_in,
            self.bytes_out,
            escape_json(&self.request_id),
        )
    }
}

/// Histogram bucket bounds in milliseconds, plus the implicit +Inf.
const BUCKETS_MS: [f64; 10] = [
    5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0,
];

#[derive(Debug, Default)]
struct Inner {
    requests_2xx: u64,
    requests_3xx: u64,
    requests_4xx: u64,
    requests_5xx: u64,
    requests_other: u64,
    load_errors: u64,
    engine_errors: u64,
    bytes_in: u64,
    bytes_served: u64,
    cache_hits: u64,
    cache_misses: u64,
    load_ms_sum: f64,
    load_count: u64,
    purges: u64,
    uploads: u64,
    process_ms_sum: f64,
    process_count: u64,
    duration_buckets: [u64; BUCKETS_MS.len() + 1],
    duration_sum: f64,
    duration_count: u64,
}

/// Request metrics, shared across handlers.
#[derive(Debug, Default)]
pub struct Metrics {
    inner: Mutex<Inner>,
}

impl Metrics {
    /// Fold one finished request into the counters.
    pub fn record(&self, r: &RequestRecord) {
        let mut m = self.inner.lock().unwrap();
        match r.status {
            200..=299 => m.requests_2xx += 1,
            // A revalidation's `304` (issue #707).
            300..=399 => m.requests_3xx += 1,
            // `CLIENT_CLOSED` among them.
            400..=499 => m.requests_4xx += 1,
            500..=599 => m.requests_5xx += 1,
            _ => m.requests_other += 1,
        }
        match r.fail {
            FailStage::Load => m.load_errors += 1,
            FailStage::Engine => m.engine_errors += 1,
            FailStage::None => {}
        }
        m.bytes_in += r.bytes_in as u64;
        m.bytes_served += r.bytes_out as u64;
        if r.cache_looked_up {
            if r.cache_hit {
                m.cache_hits += 1;
            } else {
                m.cache_misses += 1;
            }
        }
        if r.load_ran {
            m.load_ms_sum += ms(r.load);
            m.load_count += 1;
        }
        if r.process_ran {
            m.process_ms_sum += ms(r.process);
            m.process_count += 1;
        }
        let total = ms(r.total);
        for (i, bound) in BUCKETS_MS.iter().enumerate() {
            if total <= *bound {
                m.duration_buckets[i] += 1;
            }
        }
        m.duration_buckets[BUCKETS_MS.len()] += 1;
        m.duration_sum += total;
        m.duration_count += 1;
    }

    /// Count one purge request, accepted or rejected (audit
    /// PL-8: the destructive endpoint must leave a trace).
    pub fn record_purge(&self) {
        self.inner.lock().unwrap().purges += 1;
    }

    /// Count one upload request, accepted or rejected
    /// (issue #168: the write endpoint leaves the same
    /// audit trace purges do).
    pub fn record_upload(&self) {
        self.inner.lock().unwrap().uploads += 1;
    }

    /// Render the counters in Prometheus text format.
    pub fn render(&self) -> String {
        let mut out = String::with_capacity(2048);
        self.render_into(&mut out);
        out
    }

    /// [`Metrics::render`], appended to `out` (issue #643):
    /// `/metrics` writes its whole exposition into one
    /// buffer, and every line here goes straight into it
    /// instead of into a string of its own first.
    pub fn render_into(&self, out: &mut String) {
        use std::fmt::Write as _;
        let m = self.inner.lock().unwrap();
        out.push_str("# HELP crabcrop_requests_total Image requests by status class.\n");
        out.push_str("# TYPE crabcrop_requests_total counter\n");
        // Writing to a `String` cannot fail.
        let _ = writeln!(
            out,
            "crabcrop_requests_total{{status_class=\"2xx\"}} {}",
            m.requests_2xx
        );
        let _ = writeln!(
            out,
            "crabcrop_requests_total{{status_class=\"3xx\"}} {}",
            m.requests_3xx
        );
        let _ = writeln!(
            out,
            "crabcrop_requests_total{{status_class=\"4xx\"}} {}",
            m.requests_4xx
        );
        let _ = writeln!(
            out,
            "crabcrop_requests_total{{status_class=\"5xx\"}} {}",
            m.requests_5xx
        );
        let _ = writeln!(
            out,
            "crabcrop_requests_total{{status_class=\"other\"}} {}",
            m.requests_other
        );
        out.push_str("# HELP crabcrop_request_duration_ms Total request duration.\n");
        out.push_str("# TYPE crabcrop_request_duration_ms histogram\n");
        for (i, bound) in BUCKETS_MS.iter().enumerate() {
            let _ = writeln!(
                out,
                "crabcrop_request_duration_ms_bucket{{le=\"{bound}\"}} {}",
                m.duration_buckets[i]
            );
        }
        let _ = writeln!(
            out,
            "crabcrop_request_duration_ms_bucket{{le=\"+Inf\"}} {}",
            m.duration_buckets[BUCKETS_MS.len()]
        );
        let _ = writeln!(
            out,
            "crabcrop_request_duration_ms_sum {:.1}",
            m.duration_sum
        );
        let _ = writeln!(
            out,
            "crabcrop_request_duration_ms_count {}",
            m.duration_count
        );
        out.push_str("# HELP crabcrop_load_duration_ms Source load duration.\n");
        out.push_str("# TYPE crabcrop_load_duration_ms summary\n");
        let _ = writeln!(out, "crabcrop_load_duration_ms_sum {:.1}", m.load_ms_sum);
        let _ = writeln!(out, "crabcrop_load_duration_ms_count {}", m.load_count);
        out.push_str("# HELP crabcrop_process_duration_ms Engine processing duration.\n");
        out.push_str("# TYPE crabcrop_process_duration_ms summary\n");
        let _ = writeln!(
            out,
            "crabcrop_process_duration_ms_sum {:.1}",
            m.process_ms_sum
        );
        let _ = writeln!(
            out,
            "crabcrop_process_duration_ms_count {}",
            m.process_count
        );
        // These six had no HELP line before issue #707.
        out.push_str(
            "# HELP crabcrop_loader_errors_total Image requests whose source or watermark failed to load.\n",
        );
        out.push_str("# TYPE crabcrop_loader_errors_total counter\n");
        let _ = writeln!(out, "crabcrop_loader_errors_total {}", m.load_errors);
        out.push_str("# HELP crabcrop_engine_errors_total Image requests whose render failed.\n");
        out.push_str("# TYPE crabcrop_engine_errors_total counter\n");
        let _ = writeln!(out, "crabcrop_engine_errors_total {}", m.engine_errors);
        out.push_str("# HELP crabcrop_bytes_received_total Source bytes image requests loaded.\n");
        out.push_str("# TYPE crabcrop_bytes_received_total counter\n");
        let _ = writeln!(out, "crabcrop_bytes_received_total {}", m.bytes_in);
        out.push_str("# HELP crabcrop_bytes_served_total Response body bytes of image requests.\n");
        out.push_str("# TYPE crabcrop_bytes_served_total counter\n");
        let _ = writeln!(out, "crabcrop_bytes_served_total {}", m.bytes_served);
        out.push_str("# HELP crabcrop_cache_hits_total Image requests a cache tier answered.\n");
        out.push_str("# TYPE crabcrop_cache_hits_total counter\n");
        let _ = writeln!(out, "crabcrop_cache_hits_total {}", m.cache_hits);
        out.push_str("# HELP crabcrop_cache_misses_total Image requests no cache tier answered.\n");
        out.push_str("# TYPE crabcrop_cache_misses_total counter\n");
        let _ = writeln!(out, "crabcrop_cache_misses_total {}", m.cache_misses);
        out.push_str("# HELP crabcrop_purges_total Purge requests, accepted or rejected.\n");
        out.push_str("# TYPE crabcrop_purges_total counter\n");
        let _ = writeln!(out, "crabcrop_purges_total {}", m.purges);
        out.push_str("# HELP crabcrop_uploads_total Upload requests, accepted or rejected.\n");
        out.push_str("# TYPE crabcrop_uploads_total counter\n");
        let _ = writeln!(out, "crabcrop_uploads_total {}", m.uploads);
    }
}

/// The series in a Prometheus exposition that lack a `# HELP` or a
/// `# TYPE` line, each named once, in order (issue #707). A
/// histogram's and a summary's `_bucket`, `_sum` and `_count`
/// samples belong to the series their TYPE line names.
#[cfg(test)]
pub(crate) fn undescribed_series(exposition: &str) -> Vec<String> {
    let mut helped = std::collections::HashSet::new();
    let mut typed = std::collections::HashSet::new();
    let mut missing = Vec::new();
    for line in exposition.lines() {
        if let Some(rest) = line.strip_prefix("# HELP ") {
            helped.insert(rest.split(' ').next().unwrap_or("").to_string());
            continue;
        }
        if let Some(rest) = line.strip_prefix("# TYPE ") {
            typed.insert(rest.split(' ').next().unwrap_or("").to_string());
            continue;
        }
        let name = line.split(['{', ' ']).next().unwrap_or("");
        if name.is_empty() {
            continue;
        }
        let series = ["_bucket", "_sum", "_count"]
            .iter()
            .filter_map(|suffix| name.strip_suffix(suffix))
            .find(|base| typed.contains(*base))
            .unwrap_or(name);
        let described = helped.contains(series) && typed.contains(series);
        if !described && !missing.iter().any(|m| m == series) {
            missing.push(series.to_string());
        }
    }
    missing
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(status: u16) -> RequestRecord {
        RequestRecord {
            ops: "300x50/filters:format(jxl)".to_string(),
            status,
            verify: Duration::from_micros(100),
            load: Duration::from_millis(3),
            process: Duration::from_millis(1),
            total: Duration::from_millis(4),
            load_ran: true,
            process_ran: true,
            cache_looked_up: true,
            bytes_in: 1200,
            bytes_out: 300,
            ..Default::default()
        }
    }

    #[test]
    fn log_line_is_json_with_expected_fields() {
        let v: serde_json::Value = serde_json::from_str(&record(200).to_json()).unwrap();
        assert_eq!(v["method"], "GET");
        assert_eq!(v["ops"], "300x50/filters:format(jxl)");
        assert_eq!(v["status"], 200);
        assert_eq!(v["cache"], "miss");
        assert_eq!(v["bytes_in"], 1200);
        assert_eq!(v["bytes_out"], 300);
        assert_eq!(v["load_ms"], 3.0);
        assert_eq!(v["process_ms"], 1.0);
        assert!(v["ts"].as_u64().unwrap() > 0);
    }

    #[test]
    fn log_line_escapes_quotes_in_ops() {
        let mut r = record(400);
        r.ops = "we\"ird\\path".to_string();
        let v: serde_json::Value = serde_json::from_str(&r.to_json()).unwrap();
        assert_eq!(v["ops"], "we\"ird\\path");
    }

    #[test]
    fn log_line_includes_request_id() {
        let mut r = record(200);
        r.request_id = "req-abc123".to_string();
        let v: serde_json::Value = serde_json::from_str(&r.to_json()).unwrap();
        assert_eq!(v["request_id"], "req-abc123");
    }

    #[test]
    fn metrics_count_classes_errors_and_bytes() {
        let metrics = Metrics::default();
        metrics.record(&record(200));
        let mut failed = record(502);
        failed.fail = FailStage::Load;
        metrics.record(&failed);
        metrics.record(&record(403));
        let out = metrics.render();
        assert!(
            out.contains("crabcrop_requests_total{status_class=\"2xx\"} 1"),
            "{out}"
        );
        assert!(
            out.contains("crabcrop_requests_total{status_class=\"4xx\"} 1"),
            "{out}"
        );
        assert!(
            out.contains("crabcrop_requests_total{status_class=\"5xx\"} 1"),
            "{out}"
        );
        assert!(out.contains("crabcrop_loader_errors_total 1"), "{out}");
        assert!(out.contains("crabcrop_engine_errors_total 0"), "{out}");
        assert!(out.contains("crabcrop_bytes_served_total 900"), "{out}");
        assert!(
            out.contains("crabcrop_request_duration_ms_count 3"),
            "{out}"
        );
        assert!(out.contains("crabcrop_cache_misses_total 3"), "{out}");
    }

    #[test]
    fn requests_that_never_reached_the_cache_count_neither_way() {
        let metrics = Metrics::default();
        // An early rejection (bad signature, parse error)
        // and a meta request both return records that never
        // consulted the cache: neither counter may move.
        metrics.record(&RequestRecord {
            status: 403,
            ..Default::default()
        });
        metrics.record(&RequestRecord {
            status: 200,
            ..Default::default()
        });
        let out = metrics.render();
        assert!(out.contains("crabcrop_cache_hits_total 0"), "{out}");
        assert!(out.contains("crabcrop_cache_misses_total 0"), "{out}");
        // A looked-up miss and hit still count.
        let mut miss = record(200);
        miss.cache_hit = false;
        metrics.record(&miss);
        let mut hit = record(200);
        hit.cache_hit = true;
        metrics.record(&hit);
        let out = metrics.render();
        assert!(out.contains("crabcrop_cache_hits_total 1"), "{out}");
        assert!(out.contains("crabcrop_cache_misses_total 1"), "{out}");
    }

    /// [`Metrics::render`] as it was written before issue
    /// #643: a string formatted for each line, then pushed.
    fn render_by_formatting(m: &Inner) -> String {
        let mut out = String::new();
        out.push_str("# HELP crabcrop_requests_total Image requests by status class.\n");
        out.push_str("# TYPE crabcrop_requests_total counter\n");
        out.push_str(&format!(
            "crabcrop_requests_total{{status_class=\"2xx\"}} {}\n",
            m.requests_2xx
        ));
        out.push_str(&format!(
            "crabcrop_requests_total{{status_class=\"3xx\"}} {}\n",
            m.requests_3xx
        ));
        out.push_str(&format!(
            "crabcrop_requests_total{{status_class=\"4xx\"}} {}\n",
            m.requests_4xx
        ));
        out.push_str(&format!(
            "crabcrop_requests_total{{status_class=\"5xx\"}} {}\n",
            m.requests_5xx
        ));
        out.push_str(&format!(
            "crabcrop_requests_total{{status_class=\"other\"}} {}\n",
            m.requests_other
        ));
        out.push_str("# HELP crabcrop_request_duration_ms Total request duration.\n");
        out.push_str("# TYPE crabcrop_request_duration_ms histogram\n");
        for (i, bound) in BUCKETS_MS.iter().enumerate() {
            out.push_str(&format!(
                "crabcrop_request_duration_ms_bucket{{le=\"{bound}\"}} {}\n",
                m.duration_buckets[i]
            ));
        }
        out.push_str(&format!(
            "crabcrop_request_duration_ms_bucket{{le=\"+Inf\"}} {}\n",
            m.duration_buckets[BUCKETS_MS.len()]
        ));
        out.push_str(&format!(
            "crabcrop_request_duration_ms_sum {:.1}\n",
            m.duration_sum
        ));
        out.push_str(&format!(
            "crabcrop_request_duration_ms_count {}\n",
            m.duration_count
        ));
        out.push_str("# HELP crabcrop_load_duration_ms Source load duration.\n");
        out.push_str("# TYPE crabcrop_load_duration_ms summary\n");
        out.push_str(&format!(
            "crabcrop_load_duration_ms_sum {:.1}\n",
            m.load_ms_sum
        ));
        out.push_str(&format!(
            "crabcrop_load_duration_ms_count {}\n",
            m.load_count
        ));
        out.push_str("# HELP crabcrop_process_duration_ms Engine processing duration.\n");
        out.push_str("# TYPE crabcrop_process_duration_ms summary\n");
        out.push_str(&format!(
            "crabcrop_process_duration_ms_sum {:.1}\n",
            m.process_ms_sum
        ));
        out.push_str(&format!(
            "crabcrop_process_duration_ms_count {}\n",
            m.process_count
        ));
        out.push_str(
            "# HELP crabcrop_loader_errors_total Image requests whose source or watermark failed to load.\n",
        );
        out.push_str("# TYPE crabcrop_loader_errors_total counter\n");
        out.push_str(&format!("crabcrop_loader_errors_total {}\n", m.load_errors));
        out.push_str("# HELP crabcrop_engine_errors_total Image requests whose render failed.\n");
        out.push_str("# TYPE crabcrop_engine_errors_total counter\n");
        out.push_str(&format!(
            "crabcrop_engine_errors_total {}\n",
            m.engine_errors
        ));
        out.push_str("# HELP crabcrop_bytes_received_total Source bytes image requests loaded.\n");
        out.push_str("# TYPE crabcrop_bytes_received_total counter\n");
        out.push_str(&format!("crabcrop_bytes_received_total {}\n", m.bytes_in));
        out.push_str("# HELP crabcrop_bytes_served_total Response body bytes of image requests.\n");
        out.push_str("# TYPE crabcrop_bytes_served_total counter\n");
        out.push_str(&format!("crabcrop_bytes_served_total {}\n", m.bytes_served));
        out.push_str("# HELP crabcrop_cache_hits_total Image requests a cache tier answered.\n");
        out.push_str("# TYPE crabcrop_cache_hits_total counter\n");
        out.push_str(&format!("crabcrop_cache_hits_total {}\n", m.cache_hits));
        out.push_str("# HELP crabcrop_cache_misses_total Image requests no cache tier answered.\n");
        out.push_str("# TYPE crabcrop_cache_misses_total counter\n");
        out.push_str(&format!("crabcrop_cache_misses_total {}\n", m.cache_misses));
        out.push_str("# HELP crabcrop_purges_total Purge requests, accepted or rejected.\n");
        out.push_str("# TYPE crabcrop_purges_total counter\n");
        out.push_str(&format!("crabcrop_purges_total {}\n", m.purges));
        out.push_str("# HELP crabcrop_uploads_total Upload requests, accepted or rejected.\n");
        out.push_str("# TYPE crabcrop_uploads_total counter\n");
        out.push_str(&format!("crabcrop_uploads_total {}\n", m.uploads));
        out
    }

    #[test]
    fn the_exposition_is_the_text_it_was_formatted_to() {
        // Issue #643: every line is written into one buffer.
        // The text must be what the per-line strings came
        // to, for counters of every kind: each status class,
        // both failing stages, hits and misses, every
        // histogram bucket, purges and uploads.
        let metrics = Metrics::default();
        let durations_ms = [0, 4, 5, 9, 24, 50, 99, 250, 400, 1000, 2400, 5000, 7000];
        for (n, total) in durations_ms.into_iter().enumerate() {
            let mut r = record([200, 204, 304, 403, 404, 502, 503, 101][n % 8]);
            r.total = Duration::from_millis(total);
            r.load = Duration::from_micros(1500 * n as u64);
            r.process = Duration::from_micros(700 * n as u64);
            r.load_ran = n % 3 != 0;
            r.process_ran = n % 4 != 0;
            r.cache_looked_up = n % 5 != 0;
            r.cache_hit = n % 2 == 0;
            r.fail = [FailStage::None, FailStage::Load, FailStage::Engine][n % 3];
            r.bytes_in = 1000 * n;
            r.bytes_out = 77 * n;
            metrics.record(&r);
        }
        metrics.record_purge();
        metrics.record_purge();
        metrics.record_upload();
        let expected = render_by_formatting(&metrics.inner.lock().unwrap());
        assert_eq!(metrics.render(), expected);
        // 304 is a 3xx since issue #707; 101 alone is other.
        assert!(expected.contains("crabcrop_requests_total{status_class=\"3xx\"} 2\n"));
        assert!(expected.contains("crabcrop_requests_total{status_class=\"other\"} 1\n"));
        assert!(expected.contains("crabcrop_request_duration_ms_bucket{le=\"+Inf\"} 13\n"));
        assert!(expected.contains("crabcrop_request_duration_ms_sum 16241.0\n"));
        assert!(expected.contains("crabcrop_purges_total 2\n"));
        // It is appended to what the buffer holds.
        let mut out = String::from("# before\n");
        metrics.render_into(&mut out);
        assert_eq!(out, format!("# before\n{expected}"));
        // And a server that has seen nothing renders zeros.
        let idle = Metrics::default();
        assert_eq!(
            idle.render(),
            render_by_formatting(&idle.inner.lock().unwrap())
        );
    }

    #[test]
    fn each_status_counts_in_its_class() {
        // Issue #707: a 304 counted as "other", for want of a
        // 3xx class, and a client that left counts as the
        // 4xx it is recorded as.
        let metrics = Metrics::default();
        for status in [200, 206, 304, 301, 400, CLIENT_CLOSED, 500, 503, 101, 600] {
            metrics.record(&record(status));
        }
        let out = metrics.render();
        for class in ["2xx", "3xx", "4xx", "5xx", "other"] {
            assert!(
                out.contains(&format!(
                    "crabcrop_requests_total{{status_class=\"{class}\"}} 2\n"
                )),
                "{class}: {out}"
            );
        }
        // In the order of the classes.
        let classes: Vec<&str> = out
            .lines()
            .filter_map(|line| line.strip_prefix("crabcrop_requests_total{status_class=\""))
            .map(|rest| &rest[..rest.find('"').unwrap()])
            .collect();
        assert_eq!(classes, ["2xx", "3xx", "4xx", "5xx", "other"]);
    }

    #[test]
    fn the_log_line_names_the_method() {
        // Issue #707: it said GET for a HEAD request too.
        let get: serde_json::Value = serde_json::from_str(&record(200).to_json()).unwrap();
        assert_eq!(get["method"], "GET");
        let head = RequestRecord {
            method: "HEAD",
            ..record(200)
        };
        let head: serde_json::Value = serde_json::from_str(&head.to_json()).unwrap();
        assert_eq!(head["method"], "HEAD");
    }

    #[test]
    fn every_counter_has_help_and_type() {
        // Issue #707: six series had a TYPE line and no HELP.
        let metrics = Metrics::default();
        metrics.record(&record(200));
        let out = metrics.render();
        assert_eq!(undescribed_series(&out), Vec::<String>::new(), "{out}");
        assert!(
            out.contains(
                "# HELP crabcrop_cache_misses_total Image requests no cache tier answered.\n\
                 # TYPE crabcrop_cache_misses_total counter\n"
            ),
            "{out}"
        );
        // The check finds a series with no HELP, and one with
        // neither.
        let bare = "# TYPE a_total counter\na_total 1\nb 2\n";
        assert_eq!(undescribed_series(bare), ["a_total", "b"]);
    }
}
