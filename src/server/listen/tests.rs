use std::collections::VecDeque;
use std::convert::Infallible;
use std::future::Future;
use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::routing::get;
use bytes::Bytes;
use hyper::Version;
use hyper_util::rt::TokioExecutor;
use hyper_util::server::conn::auto;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Semaphore;
use tokio::time::{Instant, sleep, sleep_until, timeout};
use tokio_util::sync::CancellationToken;

use super::{
    ConnectionLimits, Frames, RequestsInFlight, Running, Sending, Watch, bind_listener,
    serve_connection, serve_connections,
};
use crate::test_support::Rng;

fn ms(n: u64) -> Duration {
    Duration::from_millis(n)
}

/// Serve `router` on a free port with the two timeouts: its
/// address, the stop token, and the serve task.
async fn start(
    router: Router,
    header_timeout: Duration,
    idle_timeout: Duration,
) -> (SocketAddr, CancellationToken, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let addr = listener.local_addr().unwrap();
    let stop = CancellationToken::new();
    let limits = ConnectionLimits {
        header_timeout,
        idle_timeout,
        max_connections: 64,
    };
    let served = tokio::spawn(serve_connections(
        listener,
        router,
        limits,
        stop.clone(),
        RequestsInFlight::default(),
    ));
    (addr, stop, served)
}

/// The size of the `/big` body: three times the 64 KiB
/// an HTTP/2 client lets the server send before its first
/// window update.
const BIG: usize = 3 * 64 * 1024;

fn router() -> Router {
    Router::new()
        .route("/", get(|| async { "ok" }))
        .route(
            "/slow",
            get(|| async {
                sleep(ms(900)).await;
                "slow"
            }),
        )
        .route(
            "/trickle",
            get(|| async {
                // Six chunks, 200 ms apart.
                let chunks = futures_util::stream::unfold(0, |n| async move {
                    if n == 6 {
                        return None;
                    }
                    sleep(ms(200)).await;
                    Some((
                        Ok::<_, Infallible>(Bytes::from(format!("chunk{n};"))),
                        n + 1,
                    ))
                });
                Body::from_stream(chunks)
            }),
        )
        .route(
            "/big",
            // One frame, as the server's own responses are.
            get(|| async { Bytes::from(vec![b'x'; BIG]) }),
        )
        .route(
            "/endless",
            get(|| async {
                let chunk = Ok::<_, Infallible>(Bytes::from(vec![b'x'; 64 * 1024]));
                Body::from_stream(futures_util::stream::repeat(chunk))
            }),
        )
}

/// Read one response with a `Content-Length` body from
/// `stream`, leaving the connection open: its head and
/// body.
async fn read_response(stream: &mut TcpStream) -> (String, String) {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let head_end = loop {
        let n = stream.read(&mut chunk).await.unwrap();
        assert!(n > 0, "closed before a response: {buf:?}");
        buf.extend_from_slice(&chunk[..n]);
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break pos + 4;
        }
    };
    let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
    let length: usize = head
        .lines()
        .find_map(|l| {
            l.to_ascii_lowercase()
                .strip_prefix("content-length:")
                .map(|v| v.trim().parse().unwrap())
        })
        .unwrap_or_else(|| panic!("no content-length: {head}"));
    while buf.len() < head_end + length {
        let n = stream.read(&mut chunk).await.unwrap();
        assert!(n > 0, "closed inside a body");
        buf.extend_from_slice(&chunk[..n]);
    }
    let body = String::from_utf8_lossy(&buf[head_end..]).to_string();
    (head, body)
}

/// Read until the server closes `stream` (a reset counts),
/// for at most `within`: how long that took, or `None`.
async fn closed_within(stream: &mut TcpStream, within: Duration) -> Option<Duration> {
    let began = Instant::now();
    let mut sink = Vec::new();
    match timeout(within, stream.read_to_end(&mut sink)).await {
        Ok(_) => Some(began.elapsed()),
        Err(_) => None,
    }
}

#[tokio::test]
async fn a_handler_that_runs_past_both_timeouts_still_answers() {
    // Neither timeout runs while a handler works: a render
    // longer than the header or the idle timeout is not cut
    // off by them.
    let (addr, _stop, _) = start(router(), ms(300), ms(300)).await;
    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream
        .write_all(b"GET /slow HTTP/1.1\r\nHost: x\r\n\r\n")
        .await
        .unwrap();
    let (head, body) = read_response(&mut stream).await;
    assert!(head.starts_with("HTTP/1.1 200"), "{head}");
    assert_eq!(body, "slow");
}

#[tokio::test]
async fn a_response_body_that_keeps_moving_is_not_cut_off() {
    // The body takes 1.2 s in chunks 200 ms apart; each
    // chunk sent is activity, so a 500 ms idle timeout does
    // not end it.
    let (addr, _stop, _) = start(router(), ms(500), ms(500)).await;
    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream
        .write_all(b"GET /trickle HTTP/1.0\r\nHost: x\r\n\r\n")
        .await
        .unwrap();
    let mut response = Vec::new();
    timeout(Duration::from_secs(10), stream.read_to_end(&mut response))
        .await
        .unwrap()
        .unwrap();
    let response = String::from_utf8_lossy(&response);
    assert!(response.starts_with("HTTP/1.0 200"), "{response}");
    assert!(
        response.ends_with("chunk0;chunk1;chunk2;chunk3;chunk4;chunk5;"),
        "{response}"
    );
}

#[tokio::test]
async fn a_response_the_client_stops_reading_is_closed_after_the_idle_timeout() {
    // A client that requests an endless body and never
    // reads it stalls the server's writes; with no bytes
    // moving, the idle timeout closes the connection.
    let (addr, _stop, _) = start(router(), ms(300), ms(300)).await;
    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream
        .write_all(b"GET /endless HTTP/1.0\r\nHost: x\r\n\r\n")
        .await
        .unwrap();
    sleep(ms(1500)).await;
    // What the socket buffers held arrives, then the end.
    assert!(
        closed_within(&mut stream, Duration::from_secs(10))
            .await
            .is_some(),
        "an endless response the client stopped reading is still open"
    );
}

#[tokio::test]
async fn a_partial_head_after_a_keep_alive_request_gets_the_header_timeout() {
    // Between requests the idle timeout applies; once the
    // next head's first byte arrives, the header timeout
    // does, though the idle deadline is further off.
    let (addr, _stop, _) = start(router(), ms(300), Duration::from_secs(5)).await;
    let mut stream = TcpStream::connect(addr).await.unwrap();
    stream
        .write_all(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
        .await
        .unwrap();
    let (head, _) = read_response(&mut stream).await;
    assert!(head.starts_with("HTTP/1.1 200"), "{head}");
    sleep(ms(600)).await;
    stream.write_all(b"GET / HTTP/1.1\r\n").await.unwrap();
    let closed = closed_within(&mut stream, Duration::from_secs(10)).await;
    let closed = closed.expect("a partial head is still open 10 s on");
    assert!(
        closed < Duration::from_secs(3),
        "a partial head closed after {closed:?}, not at the 300 ms header timeout"
    );
}

const DATA: u8 = 0;
const HEADERS: u8 = 1;
const SETTINGS: u8 = 4;
const PING: u8 = 6;
const WINDOW_UPDATE: u8 = 8;
/// The ACK flag on SETTINGS and PING, END_STREAM on DATA
/// and HEADERS.
const ACK: u8 = 1;
const END_STREAM: u8 = 1;

/// An HTTP/2 frame.
fn frame(kind: u8, flags: u8, stream: u32, payload: &[u8]) -> Vec<u8> {
    let len = u32::try_from(payload.len()).unwrap().to_be_bytes();
    let mut out = vec![len[1], len[2], len[3], kind, flags];
    out.extend_from_slice(&stream.to_be_bytes());
    out.extend_from_slice(payload);
    out
}

/// Open an HTTP/2 connection with prior knowledge, and ask
/// for `GET http://x<path>` on stream 1.
async fn h2_get(addr: SocketAddr, path: &str) -> TcpStream {
    let mut stream = TcpStream::connect(addr).await.unwrap();
    let mut hello = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n".to_vec();
    hello.extend(frame(SETTINGS, 0, 0, &[]));
    // The method and scheme from the static table, the path
    // and authority as literals; end of stream and end of
    // headers.
    let mut block = vec![0x82, 0x86, 0x04, u8::try_from(path.len()).unwrap()];
    block.extend_from_slice(path.as_bytes());
    block.extend_from_slice(&[0x41, 0x01, b'x']);
    hello.extend(frame(HEADERS, 0x4 | END_STREAM, 1, &block));
    stream.write_all(&hello).await.unwrap();
    stream
}

/// Read one HTTP/2 frame, acknowledging the server's
/// settings: its type, flags and payload, or `None` once
/// the server has closed the connection.
async fn read_frame(stream: &mut TcpStream) -> Option<(u8, u8, Vec<u8>)> {
    let mut head = [0u8; 9];
    stream.read_exact(&mut head).await.ok()?;
    let len = u32::from_be_bytes([0, head[0], head[1], head[2]]) as usize;
    let mut payload = vec![0u8; len];
    stream.read_exact(&mut payload).await.ok()?;
    if head[3] == SETTINGS && head[4] & ACK == 0 {
        stream.write_all(&frame(SETTINGS, ACK, 0, &[])).await.ok()?;
    }
    Some((head[3], head[4], payload))
}

/// Read to the end of the response on stream 1.
async fn read_h2_response(stream: &mut TcpStream) {
    loop {
        let (kind, flags, _) = read_frame(stream)
            .await
            .expect("closed before the response ended");
        if (kind == DATA || kind == HEADERS) && flags & END_STREAM != 0 {
            return;
        }
    }
}

/// Send a ping and wait for its answer: whether it came
/// before the server closed the connection.
async fn ping(stream: &mut TcpStream) -> bool {
    if stream
        .write_all(&frame(PING, 0, 0, b"12345678"))
        .await
        .is_err()
    {
        return false;
    }
    loop {
        match read_frame(stream).await {
            None => return false,
            Some((kind, flags, _)) if kind == PING && flags & ACK != 0 => return true,
            Some(_) => {}
        }
    }
}

#[test]
fn only_a_data_frames_payload_counts_as_a_body_written() {
    // A ping's answer, then a DATA frame with a 4-byte
    // body, then a settings acknowledgement.
    let ping_answer = frame(PING, ACK, 0, b"12345678");
    let mut out = ping_answer.clone();
    out.extend(frame(DATA, 0, 1, b"body"));
    out.extend(frame(SETTINGS, ACK, 0, &[]));
    // Written a byte at a time, only the body's bytes count.
    let mut frames = Frames::default();
    let counted: Vec<bool> = out.iter().map(|b| frames.wrote(&[*b])).collect();
    let body = ping_answer.len() + 9..ping_answer.len() + 9 + 4;
    let expected: Vec<bool> = (0..out.len()).map(|i| body.contains(&i)).collect();
    assert_eq!(counted, expected);
    // Written at once, the body counts; a ping's answer
    // alone does not.
    assert!(Frames::default().wrote(&out));
    assert!(!Frames::default().wrote(&ping_answer));
}

#[tokio::test]
async fn an_http2_connection_that_only_pings_is_closed_at_the_idle_timeout() {
    // The server has always answered HTTP/2 with prior
    // knowledge. Between requests an HTTP/2 client sends
    // frames, such as pings, that start no request: the
    // header timeout does not end the connection, and they
    // are not activity, so the idle timeout runs from the
    // end of the last response.
    let (addr, _stop, _) = start(router(), ms(300), ms(1500)).await;
    let mut stream = h2_get(addr, "/").await;
    read_h2_response(&mut stream).await;
    let ended = Instant::now();
    // Past the header timeout, a ping is still answered.
    sleep(ms(700)).await;
    assert!(
        ping(&mut stream).await,
        "an idle HTTP/2 connection closed at the header timeout"
    );
    // Then a ping every 500 ms.
    loop {
        sleep(ms(500)).await;
        if !ping(&mut stream).await {
            break;
        }
        assert!(
            ended.elapsed() < Duration::from_secs(10),
            "an HTTP/2 connection that pings is still open 10 s after its last response, \
             with a 1.5 s idle timeout"
        );
    }
    let closed = ended.elapsed();
    assert!(
        closed >= ms(1200) && closed < Duration::from_secs(5),
        "closed {closed:?} after its last response, with a 1.5 s idle timeout"
    );
}

#[tokio::test]
async fn an_http2_response_the_client_stops_reading_is_closed_though_it_pings() {
    // The client never opens its flow control window past
    // the first 64 KiB of an endless body, but keeps
    // pinging. The answers to its pings are not activity,
    // so the idle timeout closes the connection.
    let (addr, _stop, _) = start(router(), ms(300), ms(1500)).await;
    let mut stream = h2_get(addr, "/endless").await;
    let began = Instant::now();
    while ping(&mut stream).await {
        assert!(
            began.elapsed() < Duration::from_secs(10),
            "a stalled HTTP/2 response whose client pings is still open 10 s on, \
             with a 1.5 s idle timeout"
        );
        sleep(ms(500)).await;
    }
}

#[tokio::test]
async fn an_http2_response_the_client_reads_slowly_is_not_cut_off() {
    // hyper takes the whole `/big` body at once and sends it
    // as the client's flow control window allows. The client
    // opens the window 16 KiB at a time, 400 ms apart, so the
    // body takes over 3 s, twice the 1.5 s idle timeout. Each
    // DATA frame sent is activity, though the handler and
    // its body are done.
    const STEP: u32 = 16 * 1024;
    let (addr, _stop, _) = start(router(), ms(300), ms(1500)).await;
    let mut stream = h2_get(addr, "/big").await;
    let began = Instant::now();
    let mut granted: usize = 65_535;
    let mut received = 0;
    let closed = |received: usize| -> ! {
        panic!(
            "closed {:?} in, after {received} of {BIG} body bytes",
            began.elapsed()
        )
    };
    loop {
        let Some((kind, flags, payload)) = read_frame(&mut stream).await else {
            closed(received)
        };
        if kind != DATA {
            continue;
        }
        received += payload.len();
        if flags & END_STREAM != 0 {
            break;
        }
        if received == granted {
            sleep(ms(400)).await;
            for id in [0, 1] {
                let update = frame(WINDOW_UPDATE, 0, id, &STEP.to_be_bytes());
                if stream.write_all(&update).await.is_err() {
                    closed(received);
                }
            }
            granted += STEP as usize;
        }
    }
    assert_eq!(received, BIG);
    assert!(
        began.elapsed() > Duration::from_secs(3),
        "{:?}",
        began.elapsed()
    );
}

#[tokio::test]
async fn stopping_closes_idle_connections_and_waits_for_requests_in_flight() {
    // Shutdown (issue #463) is unchanged: the listener
    // closes, an idle connection closes at once, a request
    // in flight finishes, and the loop returns after it.
    let (addr, stop, served) =
        start(router(), Duration::from_secs(30), Duration::from_secs(30)).await;
    let mut idle = TcpStream::connect(addr).await.unwrap();
    idle.write_all(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n")
        .await
        .unwrap();
    let (head, _) = read_response(&mut idle).await;
    assert!(head.starts_with("HTTP/1.1 200"), "{head}");
    let mut busy = TcpStream::connect(addr).await.unwrap();
    busy.write_all(b"GET /slow HTTP/1.1\r\nHost: x\r\n\r\n")
        .await
        .unwrap();
    sleep(ms(200)).await;
    stop.cancel();
    assert!(
        closed_within(&mut idle, ms(500)).await.is_some(),
        "an idle connection is still open after the stop"
    );
    let (head, body) = read_response(&mut busy).await;
    assert!(head.starts_with("HTTP/1.1 200"), "{head}");
    assert_eq!(body, "slow");
    timeout(Duration::from_secs(5), served)
        .await
        .expect("the loop returns once the request in flight is done")
        .unwrap();
    assert!(TcpStream::connect(addr).await.is_err());
}

#[test]
fn a_request_counts_in_flight_until_its_body_is_sent_or_it_is_dropped() {
    // Issue #715: a shutdown that stops waiting at its
    // deadline says how many requests it cut off. One count
    // spans every connection of the loop, and a request
    // counts from its head until its response body is sent,
    // or until its handler or its body is dropped unfinished.
    let in_flight = RequestsInFlight::default();
    let limits = ConnectionLimits {
        header_timeout: Duration::from_secs(30),
        idle_timeout: Duration::from_secs(30),
        max_connections: 2,
    };
    let one = Arc::new(Watch::new(limits, in_flight.clone()));
    let two = Arc::new(Watch::new(limits, in_flight.clone()));
    assert_eq!(in_flight.count(), 0);
    let answered = one.request(Version::HTTP_11);
    let dropped = two.request(Version::HTTP_11);
    let multiplexed = two.request(Version::HTTP_2);
    assert_eq!(in_flight.count(), 3);
    // Answered, the request counts while its body is sent.
    let sending = answered.answered();
    assert_eq!(in_flight.count(), 3);
    drop(sending);
    assert_eq!(in_flight.count(), 2);
    // A handler dropped unanswered: its client left.
    drop(dropped);
    assert_eq!(in_flight.count(), 1);
    drop(multiplexed.answered());
    assert_eq!(in_flight.count(), 0);
}

#[cfg(unix)]
#[tokio::test]
async fn an_ipv6_listener_takes_ipv4_too_or_says_the_host_has_no_ipv6() {
    // Issue #715: `CRABCROP_BIND=::` listens with
    // IPV6_V6ONLY off, so it takes IPv4 connections too
    // whatever the host's default. A host without IPv6
    // cannot make the socket, and the error says what to
    // set instead; that is the half this test runs there.
    let listener = match bind_listener("[::]:0".parse().unwrap()).await {
        Ok(listener) => listener,
        Err(e) => {
            assert!(
                std::net::TcpListener::bind("[::]:0").is_err(),
                "the host binds [::], but this bind failed: {e}"
            );
            assert!(
                e.to_string().ends_with(
                    "; this host has no IPv6: set CRABCROP_BIND to an IPv4 address, such as 0.0.0.0"
                ),
                "{e}"
            );
            eprintln!("skipping the dual-stack half: this host has no IPv6");
            return;
        }
    };
    assert!(!rustix::net::sockopt::ipv6_v6only(&listener).unwrap());
    let port = listener.local_addr().unwrap().port();
    let ipv4 = TcpStream::connect(("127.0.0.1", port)).await.unwrap();
    let (_, peer) = listener.accept().await.unwrap();
    assert_eq!(peer.ip().to_canonical(), ipv4.local_addr().unwrap().ip());
    if std::net::TcpListener::bind("[::1]:0").is_ok() {
        let ipv6 = TcpStream::connect(("::1", port)).await.unwrap();
        let (_, peer) = listener.accept().await.unwrap();
        assert_eq!(peer.ip(), ipv6.local_addr().unwrap().ip());
    }
    // An IPv4 address binds as it always has.
    let ipv4_only = bind_listener("127.0.0.1:0".parse().unwrap()).await.unwrap();
    assert!(ipv4_only.local_addr().unwrap().is_ipv4());
}

#[cfg(unix)]
#[test]
fn only_a_host_without_ipv6_is_told_to_bind_ipv4() {
    // Issue #715: the hint is for the one failure that means
    // the host has no IPv6. A socket refused for another
    // reason, such as a full file table, keeps its own
    // message, which a hint about IPv6 would only mislead.
    let errno = |e: rustix::io::Errno| std::io::Error::from_raw_os_error(e.raw_os_error());
    let no_ipv6 = super::with_no_ipv6_hint(errno(rustix::io::Errno::AFNOSUPPORT));
    let plain = errno(rustix::io::Errno::AFNOSUPPORT).to_string();
    assert_eq!(
        no_ipv6.to_string(),
        format!(
            "{plain}; this host has no IPv6: set CRABCROP_BIND to an IPv4 address, such as 0.0.0.0"
        )
    );
    for other in [rustix::io::Errno::MFILE, rustix::io::Errno::NOBUFS] {
        let e = super::with_no_ipv6_hint(errno(other));
        assert_eq!(e.raw_os_error(), Some(other.raw_os_error()), "{e}");
        assert!(!e.to_string().contains("IPv6"), "{e}");
    }
}

/// The whole request head the keep-alive tests send.
const GET_ROOT: &[u8] = b"GET / HTTP/1.1\r\nHost: x\r\n\r\n";

/// Send one request on `stream` and read its answer.
async fn one_request(stream: &mut TcpStream) {
    stream.write_all(GET_ROOT).await.unwrap();
    let (head, body) = read_response(stream).await;
    assert!(head.starts_with("HTTP/1.1 200"), "{head}");
    assert_eq!(body, "ok");
}

#[tokio::test]
async fn an_idle_timeout_shorter_than_the_header_timeout_closes_a_connection_after_its_request() {
    // Issue #639: the watch is no longer woken when a
    // handler ends. With the idle timeout the shorter one,
    // the idle deadline after a request comes before the
    // header deadline of the head the watch was waiting
    // for, and the connection must still close on it. The
    // request is sent once the watch has looked at the
    // connection with nothing read.
    let (addr, _stop, _) = start(router(), Duration::from_secs(4), ms(300)).await;
    let mut stream = TcpStream::connect(addr).await.unwrap();
    sleep(ms(100)).await;
    let sent = Instant::now();
    one_request(&mut stream).await;
    let closed = closed_within(&mut stream, Duration::from_secs(10)).await;
    closed.expect("an idle connection is still open 10 s on");
    let closed = sent.elapsed();
    assert!(
        closed >= ms(300) && closed < Duration::from_secs(3),
        "an idle connection closed {closed:?} after its request, \
         not at the 300 ms idle timeout"
    );
}

#[tokio::test]
async fn a_connection_idles_out_after_a_handler_that_ran_past_the_idle_timeout() {
    // The handler runs for three idle timeouts, and the
    // watch looks at the connection while it does. It finds
    // nothing due and must look again in time for the idle
    // deadline that follows the answer, though the header
    // timeout is far longer.
    let (addr, _stop, _) = start(router(), Duration::from_secs(4), ms(300)).await;
    let mut stream = TcpStream::connect(addr).await.unwrap();
    sleep(ms(100)).await;
    let sent = Instant::now();
    stream
        .write_all(b"GET /slow HTTP/1.1\r\nHost: x\r\n\r\n")
        .await
        .unwrap();
    let (head, body) = read_response(&mut stream).await;
    assert!(head.starts_with("HTTP/1.1 200"), "{head}");
    assert_eq!(body, "slow");
    let closed = closed_within(&mut stream, Duration::from_secs(10)).await;
    closed.expect("an idle connection is still open 10 s on");
    let closed = sent.elapsed();
    assert!(
        closed >= ms(1200) && closed < Duration::from_secs(3),
        "closed {closed:?} after a 900 ms request, not at the 300 ms idle timeout after it"
    );
}

#[tokio::test]
async fn an_idle_timeout_shorter_than_the_header_timeout_does_not_cut_the_first_head_short() {
    // The first request head has the header timeout from
    // the connection's start, however short the idle
    // timeout is: a connection that sends nothing closes at
    // the header timeout, not sooner.
    let (addr, _stop, _) = start(router(), ms(1200), ms(200)).await;
    let began = Instant::now();
    let mut stream = TcpStream::connect(addr).await.unwrap();
    let closed = closed_within(&mut stream, Duration::from_secs(10)).await;
    closed.expect("a silent connection is still open 10 s on");
    let closed = began.elapsed();
    assert!(
        closed >= ms(1200) && closed < Duration::from_secs(4),
        "a silent connection closed after {closed:?}, not at the 1.2 s header timeout"
    );
}

#[tokio::test]
async fn a_partial_head_after_a_long_idle_period_closes_at_the_header_timeout() {
    // The connection has been idle for five header
    // timeouts, so the watch sleeps to an idle deadline far
    // off. The first bytes of a head that then stalls bring
    // the deadline forward to the header timeout from those
    // bytes: not before it, and not at the idle deadline.
    let (addr, _stop, _) = start(router(), ms(300), Duration::from_secs(30)).await;
    let mut stream = TcpStream::connect(addr).await.unwrap();
    one_request(&mut stream).await;
    sleep(ms(1500)).await;
    let sent = Instant::now();
    stream.write_all(b"GET / HTTP/1.1\r\nHo").await.unwrap();
    let closed = closed_within(&mut stream, Duration::from_secs(10)).await;
    closed.expect("a partial head is still open 10 s on");
    let closed = sent.elapsed();
    assert!(
        closed >= ms(300) && closed < Duration::from_secs(3),
        "a partial head closed after {closed:?}, not at the 300 ms header timeout"
    );
}

#[tokio::test]
async fn a_head_sent_a_byte_at_a_time_closes_at_the_header_timeout() {
    // Bytes that keep coming do not extend the header
    // timeout, on a connection's first head or a later one.
    // The head takes 4 s to send at a byte every 150 ms;
    // the server closes 500 ms after its first byte and
    // never answers it.
    let (addr, _stop, _) = start(router(), ms(500), Duration::from_secs(30)).await;
    for keep_alive in [false, true] {
        // The first head's timeout runs from the
        // connection's start, a later one's from its
        // first byte.
        let mut began = Instant::now();
        let mut stream = TcpStream::connect(addr).await.unwrap();
        if keep_alive {
            one_request(&mut stream).await;
            began = Instant::now();
        }
        let mut answer = Vec::new();
        let mut sent = 0;
        for byte in GET_ROOT {
            // A write to a connection the server closed
            // fails, at the latest on the second try.
            if stream.write_all(&[*byte]).await.is_err() {
                break;
            }
            sent += 1;
            if let Ok(Ok(0)) = timeout(ms(150), stream.read_buf(&mut answer)).await {
                break;
            }
        }
        let closed = began.elapsed();
        assert!(answer.is_empty(), "keep-alive {keep_alive}: {answer:?}");
        assert!(
            sent < GET_ROOT.len(),
            "keep-alive {keep_alive}: the whole head was sent, over {closed:?}"
        );
        assert!(
            closed >= ms(500) && closed < ms(3000),
            "keep-alive {keep_alive}: closed after {closed:?} and {sent} bytes, \
             not at the 500 ms header timeout"
        );
    }
}

#[tokio::test]
async fn pipelined_requests_are_answered_and_the_connection_then_idles() {
    // Two requests sent at once are both answered, though
    // the second head starts no read of its own, and the
    // connection then has the idle timeout: it is open past
    // the header timeout and closes at the idle one.
    let (addr, _stop, _) = start(router(), ms(300), ms(1500)).await;
    let mut stream = TcpStream::connect(addr).await.unwrap();
    let sent = Instant::now();
    stream
        .write_all(&[GET_ROOT, GET_ROOT].concat())
        .await
        .unwrap();
    // Both answers are read from what arrives.
    let mut answers = Vec::new();
    while answers.windows(2).filter(|w| w == b"ok").count() < 2 {
        let n = stream.read_buf(&mut answers).await.unwrap();
        assert!(n > 0, "closed before both answers: {answers:?}");
    }
    let answers = String::from_utf8_lossy(&answers);
    assert_eq!(answers.matches("HTTP/1.1 200").count(), 2, "{answers}");
    let closed = closed_within(&mut stream, Duration::from_secs(10)).await;
    closed.expect("an idle connection is still open 10 s on");
    let closed = sent.elapsed();
    assert!(
        closed >= ms(1500) && closed < Duration::from_secs(5),
        "closed {closed:?} after two pipelined requests, with a 1.5 s idle timeout"
    );
}

#[tokio::test]
async fn a_partial_head_behind_a_pipelined_request_has_the_idle_timeout_until_its_next_byte() {
    // A partial head that arrives with the request before
    // it starts no header timeout: its bytes came while that
    // request was the one being read. The idle timeout
    // bounds it, until its next byte arrives with nothing in
    // flight and starts the header timeout.
    let (addr, _stop, _) = start(router(), ms(300), Duration::from_secs(30)).await;
    let mut stream = TcpStream::connect(addr).await.unwrap();
    let mut sent = GET_ROOT.to_vec();
    sent.extend_from_slice(b"GET / HTTP/1.1\r\n");
    stream.write_all(&sent).await.unwrap();
    let (head, _) = read_response(&mut stream).await;
    assert!(head.starts_with("HTTP/1.1 200"), "{head}");
    assert!(
        closed_within(&mut stream, ms(1000)).await.is_none(),
        "a partial head behind a pipelined request closed at the header timeout"
    );
    let sent = Instant::now();
    stream.write_all(b"Host").await.unwrap();
    let closed = closed_within(&mut stream, Duration::from_secs(10)).await;
    closed.expect("a partial head is still open 10 s after its next byte");
    let closed = sent.elapsed();
    assert!(
        closed >= ms(300) && closed < Duration::from_secs(3),
        "closed {closed:?} after the head's next byte, not at the 300 ms header timeout"
    );
}

/// A future that counts how often it is polled.
struct CountPolls<F> {
    inner: Pin<Box<F>>,
    polls: Arc<AtomicUsize>,
}

impl<F: Future> Future for CountPolls<F> {
    type Output = F::Output;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<F::Output> {
        self.polls.fetch_add(1, Ordering::Relaxed);
        self.inner.as_mut().poll(cx)
    }
}

#[tokio::test]
async fn a_keep_alive_request_polls_its_connection_once() {
    // Issue #639: the watch used to wake the connection's
    // task when a request head started and again when its
    // handler ended. That task is the one doing both, and a
    // task that wakes itself is polled again: two polls for
    // every request, the second from the back of the run
    // queue and on whichever worker takes it first. A
    // request that arrives whole now wakes nothing.
    const REQUESTS: usize = 200;
    let listener = TcpListener::bind(("127.0.0.1", 0)).await.unwrap();
    let addr = listener.local_addr().unwrap();
    let mut client = TcpStream::connect(addr).await.unwrap();
    let (stream, _) = listener.accept().await.unwrap();
    let limits = ConnectionLimits {
        header_timeout: Duration::from_secs(30),
        idle_timeout: Duration::from_secs(30),
        max_connections: 1,
    };
    let slot = Arc::new(Semaphore::new(1)).acquire_owned().await.unwrap();
    let polls = Arc::new(AtomicUsize::new(0));
    let served = tokio::spawn(CountPolls {
        inner: Box::pin(serve_connection(
            stream,
            router(),
            Arc::new(auto::Builder::new(TokioExecutor::new())),
            limits,
            CancellationToken::new(),
            slot,
            RequestsInFlight::default(),
        )),
        polls: Arc::clone(&polls),
    });
    for _ in 0..REQUESTS {
        one_request(&mut client).await;
    }
    let polled = polls.load(Ordering::Relaxed);
    assert!(
        polled <= REQUESTS + REQUESTS / 4,
        "{REQUESTS} keep-alive requests polled their connection {polled} times"
    );
    drop(client);
    timeout(Duration::from_secs(5), served)
        .await
        .expect("the connection ends when its client leaves")
        .unwrap();
}

/// One thing a connection does that its watch hears of, in
/// the calls hyper and the socket make for it.
#[derive(Clone, Copy, Debug)]
enum Step {
    /// Bytes arrive and the connection goes on waiting for
    /// more: `read`, then a read that finds nothing.
    Bytes,
    /// Bytes arrive that complete a request head, and its
    /// handler starts in the same poll: `read`, `request`.
    Request,
    /// A head is complete in bytes that arrived earlier (a
    /// pipelined request): `request` alone.
    Buffered,
    /// The oldest running handler answers.
    Answer,
    /// The oldest running handler is dropped unanswered.
    Abandon,
    /// Bytes go out; whether they are a response body's.
    Write(bool),
    /// The oldest response body ends.
    BodyEnd,
    /// A read finds nothing.
    Pending,
}

/// A connection's activity and deadline as they were kept
/// before issue #639, when the watch was woken at the start
/// and the end of every request: the three bounds, stated
/// as code. The connection is to close at the first moment
/// this deadline has passed.
struct Reference {
    header_timeout: Duration,
    idle_timeout: Duration,
    handlers: usize,
    bodies: usize,
    http2: bool,
    head_since: Option<Instant>,
    last_active: Instant,
}

impl Reference {
    fn apply(&mut self, step: Step, version: Version, now: Instant) {
        let read = |r: &mut Reference| {
            if r.http2 {
                return;
            }
            r.last_active = now;
            if r.handlers == 0 && r.bodies == 0 && r.head_since.is_none() {
                r.head_since = Some(now);
            }
        };
        let request = |r: &mut Reference| {
            r.handlers += 1;
            r.head_since = None;
            if version == Version::HTTP_2 {
                r.http2 = true;
            }
        };
        match step {
            Step::Bytes => read(self),
            Step::Request => {
                read(self);
                request(self);
            }
            Step::Buffered => request(self),
            Step::Answer => {
                self.handlers -= 1;
                self.bodies += 1;
                self.last_active = now;
            }
            Step::Abandon => {
                self.handlers -= 1;
                self.last_active = now;
            }
            Step::Write(body) => {
                if body || !self.http2 {
                    self.last_active = now;
                }
            }
            Step::BodyEnd => {
                self.bodies -= 1;
                self.last_active = now;
            }
            Step::Pending => {}
        }
    }

    fn deadline(&self) -> Option<Instant> {
        if self.handlers > 0 {
            None
        } else if let Some(since) = self.head_since {
            Some(since + self.header_timeout)
        } else {
            Some(self.last_active + self.idle_timeout)
        }
    }
}

/// A connection's [`Watch`], driven by the calls a
/// connection makes, with its `expired` running as a task.
struct Driven {
    watch: Arc<Watch>,
    running: VecDeque<Running>,
    sending: VecDeque<Sending>,
    /// When `expired` resolved.
    closed: Arc<Mutex<Option<Instant>>>,
}

impl Driven {
    fn new(header_timeout: Duration, idle_timeout: Duration) -> Self {
        let watch = Arc::new(Watch::new(
            ConnectionLimits {
                header_timeout,
                idle_timeout,
                max_connections: 1,
            },
            RequestsInFlight::default(),
        ));
        let closed = Arc::new(Mutex::new(None));
        let (watching, closing) = (Arc::clone(&watch), Arc::clone(&closed));
        tokio::spawn(async move {
            watching.expired().await;
            *closing.lock().unwrap() = Some(Instant::now());
        });
        Driven {
            watch,
            running: VecDeque::new(),
            sending: VecDeque::new(),
            closed,
        }
    }

    fn apply(&mut self, step: Step, version: Version) {
        match step {
            Step::Bytes => {
                self.watch.read();
                self.watch.read_pending();
            }
            Step::Request => {
                self.watch.read();
                self.running.push_back(self.watch.request(version));
            }
            Step::Buffered => self.running.push_back(self.watch.request(version)),
            Step::Answer => {
                let running = self.running.pop_front().unwrap();
                self.sending.push_back(running.answered());
            }
            Step::Abandon => drop(self.running.pop_front().unwrap()),
            Step::Write(body) => self.watch.wrote(body),
            Step::BodyEnd => drop(self.sending.pop_front().unwrap()),
            Step::Pending => self.watch.read_pending(),
        }
    }

    fn closed(&self) -> Option<Instant> {
        *self.closed.lock().unwrap()
    }
}

/// Run one generated connection against the watch and the
/// reference: at every step, and at the end, the watch has
/// closed the connection exactly when the reference
/// deadline first passed, and not otherwise. Steps fall on
/// even milliseconds and both timeouts are odd, so no
/// deadline falls on a step.
async fn run_script(
    header_ms: u64,
    idle_ms: u64,
    version: Version,
    seed: u64,
) -> Result<usize, String> {
    let (header_timeout, idle_timeout) = (ms(header_ms), ms(idle_ms));
    let began = Instant::now();
    let at = |t: Instant| t.duration_since(began).as_millis();
    let mut rng = Rng::new(seed);
    let mut watched = Driven::new(header_timeout, idle_timeout);
    let mut reference = Reference {
        header_timeout,
        idle_timeout,
        handlers: 0,
        bodies: 0,
        http2: false,
        head_since: Some(began),
        last_active: began,
    };
    // Short waits, and waits around both timeouts.
    let short = header_ms.min(idle_ms);
    let brief = [0, 0, 0, 2, 2, 4, 10];
    let long = [
        short - 1,
        short + 1,
        header_ms - 1,
        header_ms + 1,
        idle_ms - 1,
        idle_ms + 1,
        2 * short,
        header_ms + idle_ms,
    ];
    let steps = 6 + rng.below(40);
    let mut log: Vec<(u128, Step)> = Vec::new();
    let mut taken = 0;
    loop {
        // Once the script is over, what is in flight ends,
        // so the connection comes to have a deadline.
        let step = if taken < steps {
            let mut choices = vec![Step::Bytes, Step::Bytes, Step::Pending, Step::Write(false)];
            let (handlers, bodies) = (reference.handlers, reference.bodies);
            if handlers > 0 {
                choices.extend([Step::Answer, Step::Answer, Step::Answer, Step::Abandon]);
            }
            if bodies > 0 {
                choices.extend([Step::Write(true), Step::Write(true), Step::BodyEnd]);
                choices.push(Step::BodyEnd);
            }
            // HTTP/1 serves one request at a time; HTTP/2
            // starts them side by side.
            let free = handlers == 0 && bodies == 0;
            if free || (version == Version::HTTP_2 && handlers < 3) {
                choices.extend([Step::Request, Step::Request, Step::Request, Step::Buffered]);
            }
            *rng.pick(&choices)
        } else if reference.handlers > 0 {
            Step::Answer
        } else if reference.bodies > 0 {
            Step::BodyEnd
        } else {
            break;
        };
        let wait = *if rng.below(4) == 0 {
            rng.pick(&long)
        } else {
            rng.pick(&brief)
        };
        let now = Instant::now() + ms(wait);
        sleep_until(now).await;
        if Instant::now() != now {
            return Err(format!(
                "the clock stands at {}, not {}",
                at(Instant::now()),
                at(now)
            ));
        }
        let due = reference.deadline().filter(|due| *due < now);
        match (due, watched.closed()) {
            (None, None) => {}
            (Some(due), Some(closed)) if due == closed => return Ok(taken),
            (due, closed) => {
                return Err(format!(
                    "at {} ms: due at {:?} ms, closed at {:?} ms, after {log:?}",
                    at(now),
                    due.map(at),
                    closed.map(at)
                ));
            }
        }
        reference.apply(step, version, now);
        watched.apply(step, version);
        log.push((at(now), step));
        taken += 1;
    }
    let due = reference.deadline().expect("nothing is in flight");
    // Past the deadline, so a watch that sleeps too long
    // shows as late and not as a test that never ends.
    sleep_until(due + ms(2 * (header_ms + idle_ms))).await;
    match watched.closed() {
        Some(closed) if closed == due => Ok(taken),
        closed => Err(format!(
            "due at {} ms, closed at {:?} ms, after {log:?}",
            at(due),
            closed.map(at)
        )),
    }
}

#[tokio::test(start_paused = true)]
async fn the_watch_closes_a_connection_exactly_when_its_deadline_passes() {
    // Issue #639: the watch is no longer woken when a
    // request starts or ends; it sleeps to a time no later
    // than any deadline the connection can come to have.
    // So every order of reads, requests, answers, bodies
    // and writes has to leave it closing the connection at
    // the moment the deadline of issue #607 passes: the
    // header timeout from the connection's start and from
    // the first byte of a later head, the idle timeout
    // otherwise, neither while a handler runs, and on
    // HTTP/2 no read and no write but a body's counting.
    // The clock is tokio's paused one, so the comparison is
    // exact, to the millisecond.
    let limits = [
        (31, 301),
        (301, 31),
        (31, 31),
        (5, 1001),
        (1001, 5),
        (101, 151),
        (151, 101),
    ];
    let mut steps = 0;
    for (header_ms, idle_ms) in limits {
        for version in [Version::HTTP_11, Version::HTTP_2] {
            for seed in 0..300 {
                match run_script(header_ms, idle_ms, version, seed).await {
                    Ok(taken) => steps += taken,
                    Err(e) => panic!(
                        "header timeout {header_ms} ms, idle timeout {idle_ms} ms, \
                         {version:?}, seed {seed}: {e}"
                    ),
                }
            }
        }
    }
    // The scripts are long enough to mean something.
    assert!(steps > 40_000, "{steps} steps");
}

#[tokio::test(start_paused = true)]
async fn the_watch_follows_the_named_orders_of_events() {
    // The orders issue #639 names, each as its own script
    // on the paused clock: (header timeout, idle timeout,
    // the steps with the wait before each, when the
    // connection is to close), in milliseconds.
    use Step::*;
    type Script = (u64, u64, &'static [(u64, Step)], u64);
    let scripts: &[Script] = &[
        // Nothing is ever sent: the header timeout from
        // the connection's start, though the idle one is
        // shorter.
        (301, 31, &[], 301),
        (31, 301, &[], 31),
        // A request, answered: the idle timeout from the
        // end of its body, shorter or longer than the
        // header timeout.
        (
            301,
            31,
            &[(10, Request), (2, Answer), (2, Write(true)), (2, BodyEnd)],
            47,
        ),
        (
            31,
            301,
            &[(10, Request), (2, Answer), (2, Write(true)), (2, BodyEnd)],
            317,
        ),
        // A handler that runs past both timeouts.
        (31, 51, &[(10, Request), (500, Answer), (0, BodyEnd)], 561),
        // A partial head after a long idle period: the
        // header timeout from its first byte.
        (
            31,
            5001,
            &[(10, Request), (0, Answer), (0, BodyEnd), (2000, Bytes)],
            2041,
        ),
        // A head a byte at a time: the header timeout from
        // its first byte, whatever arrives later.
        (
            31,
            5001,
            &[
                (10, Request),
                (0, Answer),
                (0, BodyEnd),
                (100, Bytes),
                (10, Bytes),
                (10, Bytes),
                (10, Bytes),
            ],
            141,
        ),
        // The same for the connection's first head.
        (31, 5001, &[(10, Bytes), (10, Bytes), (10, Bytes)], 31),
        // Pipelined requests: the second head starts no
        // read, and the connection then idles.
        (
            31,
            301,
            &[
                (10, Request),
                (0, Answer),
                (0, BodyEnd),
                (0, Buffered),
                (2, Answer),
                (0, BodyEnd),
            ],
            313,
        ),
        // A partial head behind a pipelined request: the
        // idle timeout, then the header timeout from its
        // next byte.
        (
            31,
            301,
            &[(10, Request), (0, Answer), (0, BodyEnd), (0, Pending)],
            311,
        ),
        (
            31,
            301,
            &[
                (10, Request),
                (0, Answer),
                (0, BodyEnd),
                (0, Pending),
                (100, Bytes),
            ],
            141,
        ),
        // A head that starts while a response is still
        // being sent starts no header timeout either.
        (
            31,
            301,
            &[(10, Request), (0, Answer), (2, Bytes), (2, BodyEnd)],
            315,
        ),
        // A request dropped unanswered: idle from then.
        (301, 31, &[(10, Request), (100, Abandon)], 141),
    ];
    for (n, (header_ms, idle_ms, steps, closes)) in scripts.iter().enumerate() {
        let began = Instant::now();
        let mut watched = Driven::new(ms(*header_ms), ms(*idle_ms));
        for (wait, step) in *steps {
            sleep(ms(*wait)).await;
            assert_eq!(watched.closed(), None, "script {n}: closed before {step:?}");
            watched.apply(*step, Version::HTTP_11);
        }
        sleep_until(began + ms(*closes - 1)).await;
        assert_eq!(watched.closed(), None, "script {n}: closed early");
        sleep(ms(2 * (header_ms + idle_ms))).await;
        assert_eq!(
            watched.closed().map(|at| at.duration_since(began)),
            Some(ms(*closes)),
            "script {n}"
        );
    }
}
