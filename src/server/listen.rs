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
//!   an open one closes.
//! - The header timeout: a new connection must send its
//!   first request head within it. On HTTP/1 each later
//!   request head must also arrive within it, counted from
//!   its first byte. Bytes that keep coming do not extend
//!   it.
//! - The idle timeout: a connection with no request running
//!   and no bytes moving is closed after it. That covers a
//!   keep-alive connection between requests, and a response
//!   whose client stopped reading it.
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
/// unset: longer than the common edges keep an idle
/// connection to the origin in their pool (AWS load
/// balancers 60 s, Caddy 2 minutes, Google Cloud load
/// balancers 10 minutes). An edge that reuses a connection
/// the server is closing gets an error, and some answer it
/// with a `502`, so the edge should be the one to close.
pub const DEFAULT_IDLE_TIMEOUT_SECS: u64 = 1200;

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
        let stream = match accepted {
            Ok((stream, _)) => stream,
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
        connections.spawn(serve_connection(
            stream,
            router.clone(),
            Arc::clone(&builder),
            limits,
            stop.clone(),
            slot,
        ));
    }
    drop(listener);
    connections.close();
    connections.wait().await;
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
    /// Woken when the deadline may have moved earlier: a
    /// request head started, or a handler finished.
    changed: Notify,
}

struct Activity {
    /// Requests whose handler is running.
    handlers: usize,
    /// Responses whose body is still being sent.
    bodies: usize,
    /// Whether the connection speaks HTTP/2, known from its
    /// first request. HTTP/2 sends frames between requests
    /// (pings, window updates), so its reads never start a
    /// request head.
    http2: bool,
    /// When the request head being read began: the
    /// connection's start until the first request, then on
    /// HTTP/1 the first byte of each later one.
    head_since: Option<Instant>,
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
                last_active: now,
            }),
            changed: Notify::new(),
        }
    }

    fn activity(&self) -> MutexGuard<'_, Activity> {
        self.activity.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Bytes arrived. On an HTTP/1 connection with nothing
    /// in flight they start the next request head.
    fn read(&self) {
        let now = Instant::now();
        let mut activity = self.activity();
        activity.last_active = now;
        if !activity.http2
            && activity.handlers == 0
            && activity.bodies == 0
            && activity.head_since.is_none()
        {
            activity.head_since = Some(now);
            drop(activity);
            self.changed.notify_one();
        }
    }

    /// Bytes went out.
    fn wrote(&self) {
        self.activity().last_active = Instant::now();
    }

    /// A request head arrived, and its handler starts.
    fn request(self: &Arc<Self>, version: Version) -> Running {
        let mut activity = self.activity();
        activity.handlers += 1;
        activity.head_since = None;
        if version == Version::HTTP_2 {
            activity.http2 = true;
        }
        Running {
            watch: Some(Arc::clone(self)),
        }
    }

    /// When the connection is to be closed, as things stand:
    /// none while a handler runs.
    fn deadline(&self) -> Option<Instant> {
        let activity = self.activity();
        if activity.handlers > 0 {
            None
        } else if let Some(since) = activity.head_since {
            Some(since + self.header_timeout)
        } else {
            Some(activity.last_active + self.idle_timeout)
        }
    }

    /// Resolves once the connection is past its deadline.
    async fn expired(&self) {
        loop {
            match self.deadline() {
                None => self.changed.notified().await,
                Some(at) if at <= Instant::now() => return,
                Some(at) => {
                    tokio::select! {
                        () = tokio::time::sleep_until(at) => {}
                        () = self.changed.notified() => {}
                    }
                }
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
            drop(activity);
            watch.changed.notify_one();
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
            drop(activity);
            watch.changed.notify_one();
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
        if let Poll::Ready(Ok(())) = polled
            && buf.filled().len() > before
        {
            this.watch.read();
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
            this.watch.wrote();
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
            this.watch.wrote();
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
