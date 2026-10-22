//! Observability: one structured JSON log line per request, and
//! Prometheus metrics behind `/metrics`.
//!
//! The log never contains the full source URL, only the operation
//! summary (the path segments between the signature and the source).
//! `cache_hit` records whether the result cache answered the
//! request, as reported by the handler.

use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

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
    /// Operation summary: path segments between signature and source.
    pub ops: String,
    /// HTTP status of the response actually sent.
    pub status: u16,
    pub verify: Duration,
    pub load: Duration,
    pub process: Duration,
    pub total: Duration,
    /// Whether the loader stage ran (attempted), for its averages.
    pub load_ran: bool,
    /// Whether the engine stage ran (attempted).
    pub process_ran: bool,
    pub cache_hit: bool,
    pub bytes_in: usize,
    pub bytes_out: usize,
    pub fail: FailStage,
}

impl Default for RequestRecord {
    fn default() -> Self {
        Self {
            ops: String::new(),
            status: 0,
            verify: Duration::ZERO,
            load: Duration::ZERO,
            process: Duration::ZERO,
            total: Duration::ZERO,
            load_ran: false,
            process_ran: false,
            cache_hit: false,
            bytes_in: 0,
            bytes_out: 0,
            fail: FailStage::None,
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
            "{{\"ts\":{ts},\"method\":\"GET\",\"ops\":\"{}\",\"status\":{},\
             \"verify_ms\":{:.1},\"load_ms\":{:.1},\"process_ms\":{:.1},\"total_ms\":{:.1},\
             \"cache\":\"{}\",\"bytes_in\":{},\"bytes_out\":{}}}",
            escape_json(&self.ops),
            self.status,
            ms(self.verify),
            ms(self.load),
            ms(self.process),
            ms(self.total),
            if self.cache_hit { "hit" } else { "miss" },
            self.bytes_in,
            self.bytes_out,
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
        if r.cache_hit {
            m.cache_hits += 1;
        } else {
            m.cache_misses += 1;
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

    /// Render the counters in Prometheus text format.
    pub fn render(&self) -> String {
        let m = self.inner.lock().unwrap();
        let mut out = String::new();
        out.push_str("# HELP crabcrop_requests_total Image requests by status class.\n");
        out.push_str("# TYPE crabcrop_requests_total counter\n");
        out.push_str(&format!(
            "crabcrop_requests_total{{status_class=\"2xx\"}} {}\n",
            m.requests_2xx
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
        out.push_str("# TYPE crabcrop_loader_errors_total counter\n");
        out.push_str(&format!("crabcrop_loader_errors_total {}\n", m.load_errors));
        out.push_str("# TYPE crabcrop_engine_errors_total counter\n");
        out.push_str(&format!(
            "crabcrop_engine_errors_total {}\n",
            m.engine_errors
        ));
        out.push_str("# TYPE crabcrop_bytes_received_total counter\n");
        out.push_str(&format!("crabcrop_bytes_received_total {}\n", m.bytes_in));
        out.push_str("# TYPE crabcrop_bytes_served_total counter\n");
        out.push_str(&format!("crabcrop_bytes_served_total {}\n", m.bytes_served));
        out.push_str("# TYPE crabcrop_cache_hits_total counter\n");
        out.push_str(&format!("crabcrop_cache_hits_total {}\n", m.cache_hits));
        out.push_str("# TYPE crabcrop_cache_misses_total counter\n");
        out.push_str(&format!("crabcrop_cache_misses_total {}\n", m.cache_misses));
        out
    }
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
}
