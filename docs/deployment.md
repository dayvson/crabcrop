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

Results are cached in up to three tiers: an in-memory LRU (256 MB by default, `CRABCROP_CACHE_MAX_MB`), a disk tier underneath it (`CRABCROP_RESULT_STORAGE_DIR`, 512 MB by default) whose entries survive restarts, and a shared S3 tier (`CRABCROP_RESULT_BUCKET`) that every replica reads and writes. The first two tiers are local to each process, so running more replicas trades cache hit rate: every replica produces identical bytes for a signed URL, and a miss just costs a re-fetch and re-process. Once you run several replicas, the shared tier is what keeps the hit rate whole; each recipe says how to wire it on its platform.

One behavior does not cross processes: purges. A purge is guaranteed for the process that receives it and for the configured persistent and shared stores: the entry is removed from that process's memory, its disk tier, and the shared bucket. It does not invalidate the memory of any other replica, so a replica that already holds the entry can keep serving it until that process evicts the entry or restarts. Because the removals do reach the disk tier and the bucket, a replica that misses in its own memory finds nothing underneath and re-renders; a purged entry is never resurrected from a lower tier. Cross-replica purge invalidation is deliberately out of scope. If a purge has to take effect everywhere immediately, send it to each replica, or run a single replica.

#### Lock down the edge

Run crabcrop behind a reverse proxy or CDN in production. The application bounds its own work (the processing, AI, video, and animation gates shed overload with `503`), but connection-level protections belong at the edge: request and header timeouts, connection limits, rate limiting, and TLS termination. Restrict `/health` and `/metrics` to your internal network at the proxy: `/health` is unauthenticated, and `/metrics` is gated by its bearer token (it answers `403` when no `CRABCROP_METRICS_TOKEN` is configured, and `401` for a missing or wrong bearer when one is). Both are served with `Cache-Control: no-store`, and the metrics expose traffic and error counters.

#### Rotating the signing secret

Rotation never invalidates a URL that is already in the wild: during a rotation window the server verifies the outgoing secret alongside the current one, each named by its key ID. See [Rotating the secret](configuration.md#rotating-the-secret) under Configuration for the three steps, using `CRABCROP_SECRET_KEY_ID`, `CRABCROP_SECRET_PREVIOUS`, and `CRABCROP_SECRET_PREVIOUS_KEY_ID`.

## Building from source

Beyond the Rust toolchain, a source build needs a C compiler (the tree vendors C libraries), nasm on x86_64 (the AVIF encoder's assembly; its build script fails without it), and, only for the vendored AVIF decode route (`--features avif-decode`), meson and ninja, or a system libdav1d 1.3 or newer found through pkg-config instead. The prebuilt routes (the installer, the Docker images, the release binaries) need none of this.

AVIF source decoding is behind the `avif-decode` cargo feature (off by default, on in the Docker image and the release builds; see [Source images](url-grammar.md#source-images)). Building with it needs dav1d 1.3 or newer reachable through pkg-config, or a vendored build:

```bash
SYSTEM_DEPS_DAV1D_BUILD_INTERNAL=always cargo build --features avif-decode
