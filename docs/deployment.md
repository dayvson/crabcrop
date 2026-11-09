# Deployment

Every recipe below runs the same server, configured entirely through the environment (see [Configuration](configuration.md#configuration)). Release images are published to GHCR as `ghcr.io/dayvson/crabcrop`, tagged with the release version; the AI variant is `ghcr.io/dayvson/crabcrop-ai`. Pin the exact image by digest where your platform supports it, as the Helm chart does with `image.digest`. Every recipe needs `CRABCROP_SECRET`, the HMAC secret that signs and verifies every URL, at least 16 bytes: the server refuses to start without it.

The repo ships a multi-stage `Dockerfile` (the runtime image carries the binary, CA certificates, and the dav1d library for AVIF decoding):

```bash
docker build -t crabcrop:latest .
docker run -p 3000:3000 -e CRABCROP_SECRET=your-secret-change-me -e CRABCROP_BLOCK_PRIVATE_SOURCES=1 crabcrop:latest
```

For the AI features, build `Dockerfile.ai` on top of that image; see [AI features](ai-features.md#ai-features).

### Choose a platform

| Platform | Recipe | Artifact you apply | Secret lives in | Shape |
| --- | --- | --- | --- | --- |
| Kubernetes | [Helm chart](../deploy/helm/crabcrop/README.md) | The chart, plus your values | A Kubernetes Secret the chart references | One workload, replicas optional, optional Ingress and HPA |
| Kubernetes | [Kustomize](../deploy/kustomize/README.md) | Base plus components and overlays, with `kubectl apply -k` | A Kubernetes Secret you create before applying | The same workload in kustomize form: base, components for presets, the disk tier, and the AI variant, plus single, multi-replica, and full overlays |
| AWS | [ECS Fargate](../deploy/ecs-fargate/README.md) | `task-definition.json`, plus the service commands in the recipe | Secrets Manager | A Fargate service behind an Application Load Balancer |
| GCP | [Cloud Run](../deploy/cloud-run/README.md) | `service.yaml` | Secret Manager | A managed service; the recipe keeps one instance warm |
| Azure | [Container Apps](../deploy/container-apps/README.md) | `containerapp.yaml` | The app secret store, or a Key Vault reference | A managed app with ingress; the recipe keeps one replica warm |
| Single host | [Docker Compose](../deploy/compose/README.md) | `docker-compose.yml` for evaluation, `docker-compose.prod.yml` for production | An environment variable (evaluation) or an owner-only environment file (production) | One host: an evaluation run, or a production run with Caddy and TLS in front and persistent result storage |
| Single host | [Binary on a VM](../deploy/vm/README.md) | The release binary, `crabcrop.service`, and a proxy config | A root-only environment file | One host, for production: systemd plus a TLS proxy |

### Put a CDN in front

Any platform above can sit behind a CDN. Each recipe in this family configures one CDN in front of an already-running crabcrop origin, and each folder carries the config artifact its steps refer to.

| CDN | Recipe | Config artifact | Group purge bridged to | Built-in optimizer to disable |
| --- | --- | --- | --- | --- |
| Cloudflare | [Cloudflare](../deploy/cloudflare/README.md) | `cache-tag-worker.js` (the tag bridge Worker) | Cache tags (the `Cache-Tag` header) | Polish |
| CloudFront | [CloudFront](../deploy/cloudfront/README.md) | `normalize-accept.js` (a CloudFront Function) | Cache tags (the `Cache-Group` header, via `CacheTagConfig`) | None: CloudFront has no built-in image transformation for custom origins |
| Fastly | [Fastly](../deploy/fastly/README.md) | `fastly-snippets.vcl` | Surrogate keys (the `Surrogate-Key` header) | Image Optimizer |
| BunnyCDN | [BunnyCDN](../deploy/bunnycdn/README.md) | `cache-tag-middleware.ts` (a middleware script) | Cache tags (the `CDN-Tag` header) | Bunny Optimizer |
| Akamai | [Akamai](../deploy/akamai/README.md) | `property-rules.json` (a rule fragment) | Cache tags (the `Edge-Cache-Tag` header) | Image and Video Manager |

### Production notes

#### The cache tiers, and what replicas change

Results are cached in up to three tiers: an in-memory LRU (256 MB by default, `CRABCROP_CACHE_MAX_MB`), a disk tier underneath it (`CRABCROP_RESULT_STORAGE_DIR`, 512 MB by default) whose entries survive restarts, and a shared S3 tier (`CRABCROP_RESULT_BUCKET`) that every replica reads and writes. The first two tiers are local to each process, so running more replicas trades cache hit rate: every replica produces identical bytes for a signed URL, and a miss just costs a re-fetch and re-process. Once you run several replicas, the shared tier is what keeps the hit rate whole; each recipe says how to wire it on its platform. A render is written through to the bucket in the background, after its response, with at most 64 bucket writes in flight per replica. When a slow or stalled bucket holds all 64, the next render is not written through: it stays in that replica's own tiers, the next replica to miss renders it again, and the skip counts in `crabcrop_cache_shared_puts_skipped_total`. A hit that adds a cache group to a stored entry also writes the bucket copy, so that a purge of the group finds it there; that write is never skipped. It waits in the background for one of the 64 to free up, and the response does not wait for it.

One behavior does not cross processes: purges. A purge is guaranteed for the process that receives it and for the configured persistent and shared stores: the entry is removed from that process's memory, its disk tier, and the shared bucket. It does not invalidate the memory of any other replica, so a replica that already holds the entry can keep serving it until that process evicts the entry or restarts. Because the removals reach the receiving process's disk tier and the bucket, a replica that misses in its own memory finds nothing underneath in those tiers and re-renders, and in them a purged entry is never resurrected from a lower tier. On a bucket large enough that the purge's scan finishes in the background, the purge's tombstone covers the gap: the receiving process writes it to the bucket (under `purges/` inside `CRABCROP_RESULT_PREFIX`, authenticated with a key derived from the secret, so a planted one is ignored) before it touches any tier, and every replica reads the tombstones at boot and every 5 seconds after. While a tombstone lives, each replica treats a disk or bucket copy it covers that was written before the purge began as a miss, so a bucket copy the scan has not reached yet is not served, and a restart mid-scan does not undo the purge; copies rendered after the purge began serve normally, and the scan leaves them in place. Each replica runs at most two background scans at a time, and each scan reads 16 entries at a time. The parts of the scans that run inside purge requests share 32 reads at a time between them, so a replica keeps at most 64 purge reads in flight however many purges arrive. A repeated purge of a target whose scan is still running there joins that scan instead of scanning the bucket again in parallel; when the scan ends, one more pass runs for every purge that joined it. A tombstone expires 15 minutes after its scan ends, and the replicas delete expired ones as they refresh. One whose scan never ended (the process stopped first) stays in force until a later purge of the same target finishes its scan. So does one whose scan could not reach every entry because the bucket failed to answer a listing, a read, or a delete; each such failure counts in `crabcrop_cache_tier_errors_total`. Copies are compared by the time their render began, so replica clocks should agree to within 5 seconds, as any NTP-synced fleet does; a copy another replica renders in the first 5 seconds after a purge counts as purged, so it is not cached and the next request for it renders again. In the seconds before a replica has read a new tombstone, it can still read a bucket copy the scan has not reached and keep it in its own memory, like a copy it already held. Its own disk directory is the exception of the same kind as its memory: a purge sent to another replica does not delete its disk entries, so with per-replica disk tiers a disk copy that nobody requests while the tombstone lives can serve again afterwards, until the replica receives the purge itself or evicts the entry. Cache entries record when their render began from this release on (entry format `CRABDISK6`); entries written by earlier releases keep serving and count as older than every purge. An earlier release cannot read the new entries: rolled back, its disk tier deletes them at boot and it re-renders over the bucket's copies; during a mixed-version rollout, send purges to upgraded replicas, because an earlier-release replica writes no tombstone and its scan skips entries in the new format. Cross-replica purge invalidation is deliberately out of scope. If a purge has to take effect everywhere immediately, send it to each replica, or run a single replica.

#### Shutting down

On `SIGTERM` or Ctrl-C the server drains, then exits with status 0:

1. It stops accepting connections, closes idle ones, and lets the requests in flight finish.
2. It waits for the background writes those and earlier requests left: shared-tier write-throughs, and purges whose caller disconnected before the answer.
3. It exits. A background purge scan is not waited for: one cut short leaves its purge's tombstone in force until a later purge of the same target finishes its scan (see [the cache tiers](#the-cache-tiers-and-what-replicas-change)), so the purge still holds. The exception is a purge whose tombstone the bucket refused (counted in `crabcrop_cache_tier_errors_total`): it held in the stopping process only, so the copies its scan had not reached can serve again. Send that purge again.

One deadline bounds the first two steps together: `CRABCROP_SHUTDOWN_GRACE_SECS` (8 seconds by default) from the signal on. At the deadline the process exits with whatever is still running: a request still in flight is cut off, and a bucket write still in flight is dropped; stderr says how many writes were dropped. A dropped render write-through costs one render on the next replica that misses. A dropped group write-back leaves the bucket copy outside a later purge of that group, as a failed bucket write does. A second `SIGTERM` or Ctrl-C during the drain ends it at once, in the same way. The server never runs past the grace, so give the platform a stop timeout longer than it, or the platform kills the process mid-drain:

- Kubernetes: the Helm chart passes `shutdownGraceSeconds` (default 8) to the server and sets the pod's `terminationGracePeriodSeconds` to it plus 5, so set the grace there; the chart refuses `CRABCROP_SHUTDOWN_GRACE_SECS` in `env`. Elsewhere `terminationGracePeriodSeconds` defaults to 30 seconds, which covers the default grace.
- `docker stop` and Compose: 10 seconds by default. A grace of 10 or more needs `docker stop --time` or Compose's `stop_grace_period` raised to match.
- Cloud Run: 10 seconds, fixed. Keep the grace under it.
- ECS on Fargate: the container's `stopTimeout` in the task definition, 30 seconds by default (120 at most).
- Azure Container Apps: `terminationGracePeriodSeconds` under the app's `template`, 30 seconds by default.
- systemd (the VM unit): `TimeoutStopSec=` in the unit, 90 seconds by default.

The listener closes at the signal, so a load balancer that still routes to the instance gets refused connections until it notices. Where the platform removes the instance from the balancer while it signals it, as Kubernetes does, a `preStop` hook that sleeps a few seconds keeps it serving meanwhile; add the sleep to `terminationGracePeriodSeconds`. The Helm chart sets no such hook, and its 5-second margin is not room for one.

#### Lock down the edge

Run crabcrop behind a reverse proxy or CDN in production. The application bounds its own work (the processing, AI, video, and animation gates shed overload with `503`) and the source bytes it holds before that work starts (`CRABCROP_SOURCE_BUFFER_MAX_MB`: each body is charged as it arrives with a read-ahead of at most 1 MiB, an origin that stalls or sends slowly holds only what it sent, and a request that cannot get budget within 5 seconds sheds with the same `503`), but connection-level protections belong at the edge: request and header timeouts, connection limits, rate limiting, and TLS termination. Restrict `/health` and `/metrics` to your internal network at the proxy: `/health` is unauthenticated, and `/metrics` is gated by its bearer token (it answers `403` when no `CRABCROP_METRICS_TOKEN` is configured, and `401` for a missing or wrong bearer when one is). Both are served with `Cache-Control: no-store`, and the metrics expose traffic and error counters.

#### Rotating the signing secret

Rotation never invalidates a URL that is already in the wild: during a rotation window the server verifies the outgoing secret alongside the current one, each named by its key ID. See [Rotating the secret](configuration.md#rotating-the-secret) under Configuration for the three steps, using `CRABCROP_SECRET_KEY_ID`, `CRABCROP_SECRET_PREVIOUS`, and `CRABCROP_SECRET_PREVIOUS_KEY_ID`.

## Building from source

Beyond the Rust toolchain, a source build needs a C compiler (the tree vendors C libraries), nasm on x86_64 (the AVIF encoder's assembly; its build script fails without it), and, only for the vendored AVIF decode route (`--features avif-decode`), meson and ninja, or a system libdav1d 1.3 or newer found through pkg-config instead. The prebuilt routes (the installer, the Docker images, the release binaries) need none of this.

AVIF source decoding is behind the `avif-decode` cargo feature (off by default, on in the Docker image and the release builds; see [Source images](url-grammar.md#source-images)). Building with it needs dav1d 1.3 or newer reachable through pkg-config, or a vendored build:

```bash
SYSTEM_DEPS_DAV1D_BUILD_INTERNAL=always cargo build --features avif-decode
