//! The accept loop, and the bounds it keeps on each
//! connection (issue #607).
//!
//! `axum::serve` set no timer and no cap. A connection that
//! sent nothing, stopped part way through a request head,
//! or went quiet between requests stayed open for good, and
//! enough of them took the process's whole file table. This
//! loop serves the same router through hyper's HTTP/1 and
//! HTTP/2 connections, as `axum::serve` did, with three
//! bounds:
//!
//! - At most `max_connections` connections are open at
//!   once. At the cap the loop stops accepting, so a new
//!   connection waits in the kernel's listen queue until
//!   an open one closes. With a per-client share set
//!   (`per_client`, off by default), one client may hold at
//!   most that many of them, and a connection past its share
//!   is closed as soon as it is accepted.
//! - The header timeout: a new connection must send its
//!   first request head within it. On HTTP/1 each later
//!   request head must also arrive within it, counted from
//!   its first byte. Bytes that keep coming do not extend
//!   it.
//! - The idle timeout: a connection with no request running
//!   and no bytes moving is closed after it. That covers a
//!   keep-alive connection between requests, and a response
//!   whose client stopped reading it. On HTTP/2 only the
//!   bytes of a response body count as moving: pings and
//!   other frames between requests do not, so the idle
//!   timeout runs from the end of the last response.
//!
//! Neither timer runs while a handler works on a request:
//! the processing deadline and the fetch timeouts bound
//! that. hyper's own header timer is left off. The auto
//! builder's protocol sniff runs before it, so it misses a
//! connection that sends nothing, and it also runs while a
//! keep-alive connection waits for its next request, which
//! would make the idle bound the header bound.

use std::convert::Infallible;
use std::future::Future;
use std::io;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::task::{Context, Poll};
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use bytes::Bytes;
use hyper::body::{Body as HttpBody, Frame, Incoming, SizeHint};
use hyper::{Request, Response, Version};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto;
use hyper_util::service::TowerToHyperService;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tokio_util::task::TaskTracker;

/// The header timeout when `CRABCROP_HEADER_TIMEOUT_SECS`
/// is unset: hyper's own default. A client that is not
/// stalling sends a request head in far less.
pub const DEFAULT_HEADER_TIMEOUT_SECS: u64 = 30;

/// The idle timeout when `CRABCROP_IDLE_TIMEOUT_SECS` is
/// unset, 65 minutes: longer than the common edges keep an
/// idle connection to the origin in their pool by default
/// (AWS load balancers 60 s, Caddy 2 minutes, Google Cloud
/// load balancers 10 minutes, Envoy 1 hour). An edge that
/// reuses a connection the server is closing gets an error,
/// and some answer it with a `502`, so the edge should be
/// the one to close.
pub const DEFAULT_IDLE_TIMEOUT_SECS: u64 = 3900;

/// The bounds [`serve_connections`] keeps on each connection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConnectionLimits {
    /// How long a request head may take: from the
    /// connection's start for the first request, and on
    /// HTTP/1 from the first byte for each later one.
    pub header_timeout: Duration,
    /// How long a connection may sit with no request
    /// running and no bytes moving.
    pub idle_timeout: Duration,
    /// How many connections may be open at once.
    pub max_connections: usize,
    /// How many of them one client may hold, or `None` for
    /// no share: a client is an IPv4 address, or the /64 an
    /// IPv6 address is in (see [`ClientKey`]). Behind a proxy
    /// or a load balancer every connection comes from its
    /// few addresses, so the share is for a server that
    /// clients reach directly.
    pub per_client: Option<usize>,
}

/// Serve `router` on `listener` within `limits` until
/// `stop` is cancelled. Then the listener closes, idle
/// connections close, and this returns once the requests
/// in flight have finished; the caller bounds that wait.
pub async fn serve_connections(
    listener: TcpListener,
    router: Router,
    limits: ConnectionLimits,
    stop: CancellationToken,
) {
    let slots = Arc::new(Semaphore::new(limits.max_connections));
    let clients = limits.per_client.map(|share| Arc::new(Clients::new(share)));
    let mut refusals = Refusals::default();
    let connections = TaskTracker::new();
    let builder = Arc::new(auto::Builder::new(TokioExecutor::new()));
    loop {
        // A slot first: at the cap, nothing is accepted, and
        // new connections wait in the listen queue.
        let slot = tokio::select! {
            biased;
            () = stop.cancelled() => break,
            slot = Arc::clone(&slots).acquire_owned() => slot,
        };
        let Ok(slot) = slot else { break };
        let accepted = tokio::select! {
            biased;
            () = stop.cancelled() => break,
            accepted = listener.accept() => accepted,
        };
        let (stream, peer) = match accepted {
            Ok(accepted) => accepted,
            // The connection was reset before it was
            // accepted: take the next one.
            Err(e) if is_per_connection(&e) => continue,
            // Anything else, such as a full file table,
            // waits a second before the next try, as
            // `axum::serve` did.
            Err(e) => {
                eprintln!("crabcrop: cannot accept a connection: {e}; trying again in 1 s");
                tokio::select! {
                    biased;
                    () = stop.cancelled() => break,
                    () = tokio::time::sleep(Duration::from_secs(1)) => continue,
                }
            }
        };
        // A client at its share: the connection closes here,
        // unanswered, and its slot is free again.
        let client = match &clients {
            Some(clients) => match clients.admit(peer.ip()) {
                Some(client) => Some(client),
                None => {
                    refusals.note(clients.share);
                    continue;
                }
            },
            None => None,
        };
        let served = serve_connection(
            stream,
            router.clone(),
            Arc::clone(&builder),
            limits,
            stop.clone(),
            slot,
        );
        connections.spawn(async move {
            let _client = client;
            served.await;
        });
    }
    drop(listener);
    connections.close();
    connections.wait().await;
}

/// A client, as the per-client share counts it: an IPv4
/// address, or the first 64 bits of an IPv6 one, since a
/// host is usually given a whole /64 and can use any address
/// in it. An IPv4 client that reaches a dual-stack listener
/// as an IPv4-mapped address counts as its IPv4 address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum ClientKey {
    V4(std::net::Ipv4Addr),
    V6(u64),
}

impl ClientKey {
    fn of(ip: std::net::IpAddr) -> Self {
        match ip.to_canonical() {
            std::net::IpAddr::V4(v4) => ClientKey::V4(v4),
            std::net::IpAddr::V6(v6) => ClientKey::V6((u128::from(v6) >> 64) as u64),
        }
    }
}

/// The connections each client holds, while the per-client
/// share is on. A client with none has no entry, so the map
/// holds at most one entry per open connection.
struct Clients {
    share: usize,
    held: Mutex<std::collections::HashMap<ClientKey, usize>>,
}

impl Clients {
    fn new(share: usize) -> Self {
        Clients {
            share,
            held: Mutex::new(std::collections::HashMap::new()),
        }
    }

    fn held(&self) -> MutexGuard<'_, std::collections::HashMap<ClientKey, usize>> {
        self.held.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// A place for one more connection from `ip`, held until
    /// it is dropped, or `None` when the client already holds
    /// its share.
    fn admit(self: &Arc<Self>, ip: std::net::IpAddr) -> Option<ClientConnection> {
        let key = ClientKey::of(ip);
        let mut held = self.held();
        let count = held.entry(key).or_insert(0);
        if *count >= self.share {
            return None;
        }
        *count += 1;
        Some(ClientConnection {
            clients: Arc::clone(self),
            key,
        })
    }
}

/// One connection's place in its client's share.
struct ClientConnection {
    clients: Arc<Clients>,
    key: ClientKey,
}

impl Drop for ClientConnection {
    fn drop(&mut self) {
        let mut held = self.clients.held();
        if let Some(count) = held.get_mut(&self.key) {
            *count -= 1;
            if *count == 0 {
                held.remove(&self.key);
            }
        }
    }
}

/// How often the accept loop says that it closed connections
/// past a client's share: one line on stderr, with a count of
/// the rest, at most this often. The line names no address.
const REFUSAL_LINE_EVERY: Duration = Duration::from_secs(10);

/// The connections closed past a client's share since the
/// last line about them.
#[derive(Default)]
struct Refusals {
    unreported: u64,
    next_line: Option<Instant>,
}

impl Refusals {
    fn note(&mut self, share: usize) {
        self.unreported += 1;
        let now = Instant::now();
        if self.next_line.is_some_and(|at| now < at) {
            return;
        }
        eprintln!(
            "crabcrop: closed {} connection(s) from clients already holding {share} \
             (CRABCROP_MAX_CONNECTIONS_PER_CLIENT)",
            self.unreported
        );
        self.unreported = 0;
        self.next_line = Some(now + REFUSAL_LINE_EVERY);
    }
}

/// Whether an accept error concerns only the connection it
/// was about to accept.
fn is_per_connection(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        io::ErrorKind::ConnectionRefused
            | io::ErrorKind::ConnectionAborted
            | io::ErrorKind::ConnectionReset
    )
}

/// Serve one connection, holding its `slot`, until it ends,
/// a timeout closes it, or `stop` asks it to finish its
/// requests and close.
async fn serve_connection(
    stream: TcpStream,
    router: Router,
    builder: Arc<auto::Builder<TokioExecutor>>,
    limits: ConnectionLimits,
    stop: CancellationToken,
    slot: OwnedSemaphorePermit,
) {
    let _slot = slot;
    let watch = Arc::new(Watch::new(limits));
    let io = TokioIo::new(Watched {
        stream,
        watch: Arc::clone(&watch),
        frames: Frames::default(),
    });
    let service = Tracked {
        router: TowerToHyperService::new(router),
        watch: Arc::clone(&watch),
    };
    let mut conn = std::pin::pin!(builder.serve_connection_with_upgrades(io, service));
    let mut expired = std::pin::pin!(watch.expired());
    let mut stopping = false;
    loop {
        tokio::select! {
            // An error here is the client's (a reset, a
            // malformed request); hyper has answered what it
            // could.
            _ = conn.as_mut() => return,
            // Dropping the connection closes the socket.
            () = expired.as_mut() => return,
            () = stop.cancelled(), if !stopping => {
                stopping = true;
                conn.as_mut().graceful_shutdown();
            }
        }
    }
}

/// One connection's activity, which its timeouts are
/// measured against.
struct Watch {
    header_timeout: Duration,
    idle_timeout: Duration,
    activity: Mutex<Activity>,
    /// Woken when a request head has started and been left
    /// incomplete: the one change that can bring the deadline
    /// ahead of the time [`Watch::expired`] sleeps to.
    changed: Notify,
}

/// The shortest time [`Watch::expired`] sleeps before it
/// looks again while nothing is due: the timer's own
/// resolution, so a timeout of zero cannot make it spin.
const SHORTEST_SLEEP: Duration = Duration::from_millis(1);

struct Activity {
    /// Requests whose handler is running.
    handlers: usize,
    /// Responses whose body is still being sent.
    bodies: usize,
    /// Whether the connection speaks HTTP/2, known from its
    /// first request. An HTTP/2 client sends frames between
    /// requests (pings, settings, window updates), and the
    /// server answers some of them. So on HTTP/2 no read
    /// starts a request head or counts as activity, and of
    /// the writes only a response body's DATA frames count.
    http2: bool,
    /// When the request head being read began: the
    /// connection's start until the first request, then on
    /// HTTP/1 the first byte of each later one.
    head_since: Option<Instant>,
    /// Whether [`Watch::expired`] has yet to be told of the
    /// head that began at `head_since`. It is told once that
    /// head leaves the connection waiting for more bytes
    /// ([`Watch::read_pending`]); a head that arrives whole
    /// starts its handler in the same poll and never needs
    /// its deadline.
    head_unseen: bool,
    /// When a byte last moved either way, or a request or
    /// response last ended.
    last_active: Instant,
}

impl Watch {
    fn new(limits: ConnectionLimits) -> Self {
        let now = Instant::now();
        Watch {
            header_timeout: limits.header_timeout,
            idle_timeout: limits.idle_timeout,
            activity: Mutex::new(Activity {
                handlers: 0,
                bodies: 0,
                http2: false,
                head_since: Some(now),
                head_unseen: false,
                last_active: now,
            }),
            changed: Notify::new(),
        }
    }

    fn activity(&self) -> MutexGuard<'_, Activity> {
        self.activity.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Bytes arrived. On HTTP/1 they are activity, and with
    /// nothing in flight they start the next request head.
    /// On HTTP/2 they are neither.
    ///
    /// A head that starts here wakes nothing (issue #639).
    /// The task that reads these bytes is the one
    /// [`Watch::expired`] runs on, and a task that wakes
    /// itself is run again from the back of the queue, on
    /// whichever worker takes it first: a second poll, a
    /// timer re-armed and a parked worker woken, for every
    /// request.
    fn read(&self) {
        let now = Instant::now();
        let mut activity = self.activity();
        if activity.http2 {
            return;
        }
        activity.last_active = now;
        if activity.handlers == 0 && activity.bodies == 0 && activity.head_since.is_none() {
            activity.head_since = Some(now);
            activity.head_unseen = true;
        }
    }

    /// A read found nothing to take, so the connection now
    /// waits on its client. If a request head began since
    /// [`Watch::expired`] last looked, its header deadline
    /// can be sooner than the time that sleeps to, so it is
    /// woken: once for a head that arrives in pieces.
    fn read_pending(&self) {
        let mut activity = self.activity();
        if activity.head_unseen {
            activity.head_unseen = false;
            drop(activity);
            self.changed.notify_one();
        }
    }

    /// Bytes went out; `body` says whether some of them were
    /// an HTTP/2 DATA frame's payload. On HTTP/2 only those
    /// are activity.
    fn wrote(&self, body: bool) {
        let mut activity = self.activity();
        if body || !activity.http2 {
            activity.last_active = Instant::now();
        }
    }

    /// A request head arrived, and its handler starts.
    fn request(self: &Arc<Self>, version: Version) -> Running {
        let mut activity = self.activity();
        activity.handlers += 1;
        activity.head_since = None;
        activity.head_unseen = false;
        if version == Version::HTTP_2 {
            activity.http2 = true;
        }
        Running {
            watch: Some(Arc::clone(self)),
        }
    }

    /// When the connection is to be closed, as things stand:
    /// none while a handler runs.
    fn deadline(&self, activity: &Activity) -> Option<Instant> {
        if activity.handlers > 0 {
            None
        } else if let Some(since) = activity.head_since {
            Some(since + self.header_timeout)
        } else {
            Some(activity.last_active + self.idle_timeout)
        }
    }

    /// `None` once the connection is past its deadline.
    /// Otherwise the time to look again: never later than
    /// any deadline the connection can come to have before
    /// then, unless [`Watch::read_pending`] wakes `changed`
    /// first. So [`Watch::expired`] need not be told when a
    /// request starts or ends (issue #639).
    ///
    /// - While a handler runs, nothing is due. Once it ends,
    ///   the deadline is the idle timeout from then, or the
    ///   header timeout from a later head's first byte:
    ///   never sooner than the shorter of the two from now.
    /// - While a head is read, its header deadline stands.
    ///   The head may still arrive and be answered at once,
    ///   which leaves the idle timeout from then: never
    ///   sooner than the idle timeout from now.
    /// - Between requests the idle deadline stands, and
    ///   activity only moves it later. A head that starts
    ///   and is answered leads to the cases above, later
    ///   still. One that starts and stalls wakes `changed`.
    fn look_again(&self, now: Instant) -> Option<Instant> {
        let activity = self.activity();
        let Some(due) = self.deadline(&activity) else {
            let step = self.header_timeout.min(self.idle_timeout);
            return Some(now + step.max(SHORTEST_SLEEP));
        };
        if due <= now {
            return None;
        }
        Some(if activity.head_since.is_some() {
            due.min(now + self.idle_timeout.max(SHORTEST_SLEEP))
        } else {
            due
        })
    }

    /// Resolves once the connection is past its deadline.
    async fn expired(&self) {
        while let Some(at) = self.look_again(Instant::now()) {
            tokio::select! {
                () = tokio::time::sleep_until(at) => {}
                () = self.changed.notified() => {}
            }
        }
    }
}

/// A request whose handler is running. It becomes a
/// [`Sending`] when the handler answers; dropped first, the
/// client left and the request is over.
struct Running {
    watch: Option<Arc<Watch>>,
}

impl Running {
    fn answered(mut self) -> Sending {
        let watch = self.watch.take();
        if let Some(watch) = &watch {
            let mut activity = watch.activity();
            activity.handlers -= 1;
            activity.bodies += 1;
            activity.last_active = Instant::now();
        }
        Sending { watch }
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        if let Some(watch) = self.watch.take() {
            let mut activity = watch.activity();
            activity.handlers -= 1;
            activity.last_active = Instant::now();
        }
    }
}

/// A response whose body is being sent.
struct Sending {
    watch: Option<Arc<Watch>>,
}

impl Drop for Sending {
    fn drop(&mut self) {
        if let Some(watch) = self.watch.take() {
            let mut activity = watch.activity();
            activity.bodies -= 1;
            activity.last_active = Instant::now();
        }
    }
}

/// The router as hyper calls it, counting the requests on
/// one connection.
struct Tracked {
    router: TowerToHyperService<Router>,
    watch: Arc<Watch>,
}

type TrackedFuture =
    Pin<Box<dyn Future<Output = Result<Response<TrackedBody>, Infallible>> + Send>>;

impl hyper::service::Service<Request<Incoming>> for Tracked {
    type Response = Response<TrackedBody>;
    type Error = Infallible;
    type Future = TrackedFuture;

    fn call(&self, request: Request<Incoming>) -> Self::Future {
        let running = self.watch.request(request.version());
        let response = self.router.call(request);
        Box::pin(async move {
            let response = response.await?;
            let sending = running.answered();
            Ok(response.map(|body| TrackedBody {
                body,
                _sending: sending,
            }))
        })
    }
}

/// A response body, unchanged, that marks its connection
/// as sending until it is dropped.
struct TrackedBody {
    body: Body,
    _sending: Sending,
}

impl HttpBody for TrackedBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, axum::Error>>> {
        Pin::new(&mut self.get_mut().body).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.body.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.body.size_hint()
    }
}

/// A connection's socket, reporting the bytes that move to
/// its [`Watch`].
struct Watched {
    stream: TcpStream,
    watch: Arc<Watch>,
    /// Where the bytes written so far stand in HTTP/2 frames.
    frames: Frames,
}

/// The length of an HTTP/2 frame header.
const FRAME_HEADER_LEN: usize = 9;

/// The type of an HTTP/2 DATA frame.
const DATA_FRAME: u8 = 0;

/// Follows a connection's output as HTTP/2 frames, to tell
/// the DATA frames of a response body from control frames
/// such as the answer to a ping. hyper hands HTTP/2 a body
/// of one chunk, as the server's responses are, all at
/// once, and HTTP/2 sends it as the client's flow control
/// allows. So the DATA frames written, not the body, show
/// that a response is still moving. It reads every
/// connection's output from the first byte, since whether
/// one speaks HTTP/2 is known only at its first request; on
/// HTTP/1 its answer is not used.
#[derive(Default)]
struct Frames {
    /// The next frame header, as far as it has been written.
    header: [u8; FRAME_HEADER_LEN],
    header_len: usize,
    /// The payload bytes of the current frame still to come.
    payload_left: usize,
    /// Whether the current frame is a DATA frame.
    data: bool,
}

impl Frames {
    /// Step over `bytes`, just written: whether any of them
    /// were a DATA frame's payload.
    fn wrote(&mut self, mut bytes: &[u8]) -> bool {
        let mut data = false;
        while !bytes.is_empty() {
            if self.payload_left > 0 {
                let n = self.payload_left.min(bytes.len());
                data |= self.data;
                self.payload_left -= n;
                bytes = &bytes[n..];
                continue;
            }
            let n = (FRAME_HEADER_LEN - self.header_len).min(bytes.len());
            self.header[self.header_len..self.header_len + n].copy_from_slice(&bytes[..n]);
            self.header_len += n;
            bytes = &bytes[n..];
            if self.header_len == FRAME_HEADER_LEN {
                let [a, b, c, kind, ..] = self.header;
                self.payload_left = u32::from_be_bytes([0, a, b, c]) as usize;
                self.data = kind == DATA_FRAME;
                self.header_len = 0;
            }
        }
        data
    }
}

impl AsyncRead for Watched {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let this = self.get_mut();
        let before = buf.filled().len();
        let polled = Pin::new(&mut this.stream).poll_read(cx, buf);
        match polled {
            Poll::Ready(Ok(())) if buf.filled().len() > before => this.watch.read(),
            Poll::Pending => this.watch.read_pending(),
            _ => {}
        }
        polled
    }
}

impl AsyncWrite for Watched {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        data: &[u8],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let polled = Pin::new(&mut this.stream).poll_write(cx, data);
        if let Poll::Ready(Ok(n)) = polled
            && n > 0
        {
            let body = this.frames.wrote(&data[..n]);
            this.watch.wrote(body);
        }
        polled
    }

    fn poll_write_vectored(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        data: &[io::IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        let this = self.get_mut();
        let polled = Pin::new(&mut this.stream).poll_write_vectored(cx, data);
        if let Poll::Ready(Ok(n)) = polled
            && n > 0
        {
            let mut left = n;
            let mut body = false;
            for slice in data {
                let step = left.min(slice.len());
                body |= this.frames.wrote(&slice[..step]);
                left -= step;
                if left == 0 {
                    break;
                }
            }
            this.watch.wrote(body);
        }
        polled
    }

    fn is_write_vectored(&self) -> bool {
        self.stream.is_write_vectored()
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().stream).poll_flush(cx)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().stream).poll_shutdown(cx)
    }
}

#[cfg(test)]
mod tests;
