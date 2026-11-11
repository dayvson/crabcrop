use std::convert::Infallible;
use std::net::SocketAddr;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::routing::get;
use bytes::Bytes;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::time::{Instant, sleep, timeout};
use tokio_util::sync::CancellationToken;

use super::{ConnectionLimits, serve_connections};

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
    let served = tokio::spawn(serve_connections(listener, router, limits, stop.clone()));
    (addr, stop, served)
}

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

/// An HTTP/2 frame.
fn frame(kind: u8, flags: u8, stream: u32, payload: &[u8]) -> Vec<u8> {
    let len = u32::try_from(payload.len()).unwrap().to_be_bytes();
    let mut out = vec![len[1], len[2], len[3], kind, flags];
    out.extend_from_slice(&stream.to_be_bytes());
    out.extend_from_slice(payload);
    out
}

/// Read one HTTP/2 frame: its type, flags and payload, or
/// `None` once the server has closed the connection.
async fn read_frame(stream: &mut TcpStream) -> Option<(u8, u8, Vec<u8>)> {
    let mut head = [0u8; 9];
    stream.read_exact(&mut head).await.ok()?;
    let len = u32::from_be_bytes([0, head[0], head[1], head[2]]) as usize;
    let mut payload = vec![0u8; len];
    stream.read_exact(&mut payload).await.ok()?;
    Some((head[3], head[4], payload))
}

#[tokio::test]
async fn an_idle_http2_connection_is_closed_at_the_idle_timeout_not_the_header_timeout() {
    // The server has always answered HTTP/2 with prior
    // knowledge. Between requests an HTTP/2 client sends
    // frames (here a ping) that start no request, so the
    // header timeout does not end the connection; the idle
    // timeout does.
    const SETTINGS: u8 = 4;
    const PING: u8 = 6;
    const ACK: u8 = 1;
    let (addr, _stop, _) = start(router(), ms(300), ms(1500)).await;
    let mut stream = TcpStream::connect(addr).await.unwrap();
    let mut hello = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n".to_vec();
    hello.extend(frame(SETTINGS, 0, 0, &[]));
    // GET http://x/ from the static table, end of stream
    // and end of headers.
    hello.extend(frame(1, 0x5, 1, &[0x82, 0x86, 0x84, 0x41, 0x01, b'x']));
    stream.write_all(&hello).await.unwrap();
    // Read to the end of the response on stream 1.
    loop {
        let (kind, flags, _) = read_frame(&mut stream)
            .await
            .expect("closed before the response ended");
        if kind == SETTINGS && flags & ACK == 0 {
            stream
                .write_all(&frame(SETTINGS, ACK, 0, &[]))
                .await
                .unwrap();
        }
        if (kind == 0 || kind == 1) && flags & 0x1 != 0 {
            break;
        }
    }
    // Past the header timeout, a ping is still answered.
    sleep(ms(700)).await;
    stream
        .write_all(&frame(PING, 0, 0, b"12345678"))
        .await
        .unwrap();
    loop {
        let (kind, flags, _) = read_frame(&mut stream)
            .await
            .expect("an idle HTTP/2 connection closed at the header timeout");
        if kind == PING && flags & ACK != 0 {
            break;
        }
    }
    let pinged = Instant::now();
    while read_frame(&mut stream).await.is_some() {
        assert!(
            pinged.elapsed() < Duration::from_secs(10),
            "an idle HTTP/2 connection is still open 10 s on"
        );
    }
    let closed = pinged.elapsed();
    assert!(
        closed >= ms(1200) && closed < Duration::from_secs(5),
        "closed {closed:?} after its last frame, with a 1.5 s idle timeout"
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
