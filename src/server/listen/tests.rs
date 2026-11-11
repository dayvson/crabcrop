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

use super::{ConnectionLimits, Frames, serve_connections};

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
