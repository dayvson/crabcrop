# crabcrop Helm chart

Deploys the crabcrop image service. The chart is deliberately thin: the server is configured entirely through the environment, so the chart wires the common settings structurally (secrets, presets, storage, AI models) and passes everything else through a free-form `env` map. Secrets never go through `env`: see [Secrets](#secrets).

Two install shapes ship with the chart:

- **Production**: `values-production.yaml`, the shape to run for real. It is the first recipe below.
- **Evaluation**: the chart defaults, enough to try the service with one command. It is labelled as such where it appears; do not mistake it for the production shape (no resource limits, no disk tier, no metrics token).

## Prerequisites

- A Kubernetes cluster, with kubectl and Helm configured against it.
- An Ingress controller, only if you enable the chart's Ingress (`ingress.enabled`).

## Production install

### 1. The secrets

One Secret you manage holds both credentials, as two keys: the signing secret (16+ bytes; it signs and verifies every URL) and the metrics token (the bearer <redacted> `GET /metrics` requires):

```bash
kubectl create secret generic crabcrop-secret \
  --from-literal=CRABCROP_SECRET="$(openssl rand -hex 32)" \
  --from-literal=CRABCROP_METRICS_TOKEN="$(openssl rand -hex 32)"
```

### 2. Install with the production values

```bash
helm install crabcrop deploy/helm/crabcrop \
  -f deploy/helm/crabcrop/values-production.yaml
```

`values-production.yaml` sets the production shape, and everything in it is a value, so nothing in the chart needs editing:

- The current release image, pinned by tag (set `image.digest` to pin the exact build).
- Resource requests and limits (500m CPU and 512Mi requested; 2 CPU and 2Gi at the limit).
- The disk result tier on its own persistent volume (8GiB budget on a 10Gi claim), so renders survive pod restarts.
- Both credentials wired from the Secret above: the signing secret, and the metrics token through `metricsToken.existingSecret`, so `/metrics` is never answerable without its bearer <redacted>.
- The production environment: a 512 MB memory result tier sized against the memory limit, the slow lane at one permit (long-running renders, such as animated encodes, never starve stills), the 30-second processing limit stated, and private-address sources refused.
- Health probes are part of the chart itself: liveness and readiness both read `/health`, and every install gets them.

Serving traffic takes the usual extra step: enable the Ingress (`ingress.enabled`, with your host and TLS) or front the Service your own way.

### 3. The shared result bucket (when you scale out)

A single replica is complete with the two local tiers. Before you run several replicas, wire the shared tier so they read and write one result store instead of diverging. Put the bucket's AWS keys in a Secret first:

```bash
kubectl create secret generic crabcrop-aws \
  --from-literal=AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  --from-literal=AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY"
helm upgrade crabcrop deploy/helm/crabcrop \
  -f deploy/helm/crabcrop/values-production.yaml \
  --set resultBucket.name=<your-bucket> \
  --set resultBucket.credentialsSecret=crabcrop-aws
```

`resultBucket` also takes a `prefix` (the production values set `results`) and, for an S3-compatible store, `s3Endpoint` and `s3Region`. Requests are signed with environment-provided AWS credentials only, because the server reads `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN` once at startup and does not fetch credentials from a pod IAM role or any other part of the AWS credential chain. `resultBucket.credentialsSecret` names the Secret that holds them, under those key names; add an `AWS_SESSION_TOKEN` key for temporary credentials. The tier is best-effort by design: a bucket error reads as a cache miss, never a failed request.

## Secrets

The server treats six variables as secrets. Read each from a Secret you manage, through the value below. A value in `env` or `ai.env` lands in the Deployment as plain text, and anyone who can read the Deployment or its Pods can read it (the built-in `view` role can, though it cannot read Secrets).

| Variable | Value that reads it from a Secret |
|---|---|
| `CRABCROP_SECRET` | `existingSecret`, key `secretKey` |
| `CRABCROP_SECRET_PREVIOUS` | `previousSecret.existingSecret`, key `previousSecret.secretKey` |
| `CRABCROP_METRICS_TOKEN` | `metricsToken.existingSecret`, key `metricsToken.secretKey` |
| `AWS_SECRET_ACCESS_KEY` and `AWS_SESSION_TOKEN` | `resultBucket.credentialsSecret`, with `AWS_ACCESS_KEY_ID`, each under its own name; the session token is optional |
| `CRABCROP_GCS_SECRET_KEY` | `gcsCredentials.existingSecret`, with `CRABCROP_GCS_ACCESS_KEY`, each under its own name |

The AWS keys also sign `s3://` sources and the upload bucket, so `resultBucket.credentialsSecret` works without a bucket name. Two lists cover the rest:

- `extraEnv` takes env entries as written, after the `env` map, so an entry can carry `valueFrom.secretKeyRef`. Use it when your Secret's key names differ.
- `envFrom` maps whole Secrets or ConfigMaps to variables. A variable set in `env` or `extraEnv` wins over one from `envFrom`.

```yaml
extraEnv:
  - name: AWS_SECRET_ACCESS_KEY
    valueFrom:
      secretKeyRef:
        name: my-aws-keys
        key: secret-access-key
```

This release still renders a secret given in `env`, in `ai.env`, or as an `extraEnv` entry's `value`. It warns in the install notes and in a comment in the rendered Deployment, naming the value to set instead. Set `refusePlainSecrets: true` to fail the render instead. A later release will refuse these by default. To migrate, put each key in a Secret and set the value the warning names. Then remove the key from `env`, `ai.env` or `extraEnv`: an entry left there comes after the Secret's and wins. Move `AWS_ACCESS_KEY_ID` and `CRABCROP_GCS_ACCESS_KEY` with their secret keys the same way.

## Evaluation install

For a trial, the chart defaults plus a secret are enough (the defaults carry the evaluation source posture, private-address sources refused):

```bash
helm install crabcrop deploy/helm/crabcrop \
  --set existingSecret=crabcrop-secret
```

Release images are published to GHCR as `ghcr.io/dayvson/crabcrop`, tagged with the release version; the chart defaults point there. For a quick trial only, `--set secret=<value>` lets the chart create the Secret itself; do not use that route for anything real. To run your own build instead:

```bash
docker build -t your-registry/crabcrop:0.11.0 .
docker push your-registry/crabcrop:0.11.0
helm install crabcrop deploy/helm/crabcrop \
  --set image.repository=your-registry/crabcrop \
  --set image.tag=0.11.0 \
  --set existingSecret=crabcrop-secret
```

## Presets (optional)

```yaml
presets:
  thumb: "300x200"
  gray: "filters:grayscale"
onlyPresets: false
```

The map renders a ConfigMap the chart mounts as the presets file and wires through `CRABCROP_PRESETS_FILE`. Preset names also become cache groups (the `Cache-Group` header), so `crabcrop purge-group thumb` invalidates a preset's whole output set at once.

Treat a preset as a stable template: never edit a definition in place. Roll a change out under a new name and update templates to reference it. Image responses are served `immutable` for a year, so an in-place edit strands the old bytes in every browser and CDN that already holds them, where no purge can reach.

## AI variant

```yaml
ai:
  enabled: true
  image:
    repository: your-registry/crabcrop-ai
  models:
    existingClaim: crabcrop-models
  env:
    CRABCROP_ORT_DYLIB: /models/libonnxruntime.so
    CRABCROP_REMOVEBG_MODEL: /models/removebg.onnx
```

`ai.enabled` switches the image to the Dockerfile.ai variant and mounts the models claim at `/models`. Which features load is decided by which model env vars you set (removebg, face, upscale, detection, CLIP tagging; see the main README's AI section). Models are preloaded and validated at boot, so a bad path fails the pod at startup. Model files are deployment artifacts: the chart never carries them.

## Other settings

Any other `CRABCROP_*` variable goes in the free-form map:

```yaml
env:
  CRABCROP_CACHE_MAX_MB: "512"
  CRABCROP_METADATA: "preserve"
  CRABCROP_ALLOWED_SOURCES: "images.example.com,upload.wikimedia.org"
```

Keep secrets out of this map: see [Secrets](#secrets).

The structured values cover the common wiring beyond the signing secret: `metricsToken` (the `/metrics` bearer <redacted>, from an existing Secret), `resultStorage` (the disk tier and its claim), `resultBucket` (the shared tier), and `shutdownGraceSeconds` (how long a pod drains after `SIGTERM`, passed as `CRABCROP_SHUTDOWN_GRACE_SECS`, 8 by default; the pod's `terminationGracePeriodSeconds` is set to it plus 5, so set the grace here: the chart refuses to render with `CRABCROP_SHUTDOWN_GRACE_SECS` in `env`). See the Configuration table in the main README for the full list. An Ingress and an HPA are available behind `ingress.enabled` and `autoscaling.enabled`.

## Smoke checklist

Run against the production install, with the service reachable at `<host>` (your Ingress host, or `kubectl port-forward svc/crabcrop 3000:80` and then `http://localhost:3000`):

1. Health: `curl https://<host>/health` returns 200 with the body `ok`.
2. Sign one request with the same secret and fetch it: `CRABCROP_SECRET=<secret> crabcrop sign "300x200/https://example.com/photo.jpg"`, then `curl -i "https://<host><signed-path>"` returns 200 and image bytes.
3. Fetch the same signed URL again. The `Cache-Status` response header reports a hit.
4. Purge it: `CRABCROP_BASE=https://<host> CRABCROP_SECRET=<secret> crabcrop purge "300x200/https://example.com/photo.jpg"` answers `purged 1` (or more, counting variants), and the next fetch of the signed URL is a miss again.
5. Metrics auth: `curl https://<host>/metrics` without a token returns 401, and `curl -H "Authorization: Bearer <metrics-token>" https://<host>/metrics` returns 200 with the Prometheus counters. (On an install with no metrics token configured, `/metrics` answers 403 to everyone; that is the evaluation shape, not production.)
6. Update the Secret and restart the deployment (`kubectl rollout restart deployment/crabcrop`) and the pods pick it up on their next start; for zero-downtime rotation use the key ID flow in the main README (`CRABCROP_SECRET_PREVIOUS` plus key IDs). Put the outgoing secret in a Secret and set `previousSecret.existingSecret` to it; the key IDs are not secret, so `CRABCROP_SECRET_KEY_ID` and `CRABCROP_SECRET_PREVIOUS_KEY_ID` go in `env`.

## Scaling and cache notes

```bash
helm install crabcrop deploy/helm/crabcrop \
  -f deploy/helm/crabcrop/values-production.yaml \
  -f deploy/helm/crabcrop/values-multireplica.yaml \
  --set resultBucket.name=<your-bucket> \
  --set resultBucket.credentialsSecret=crabcrop-aws
```

**Read this before scaling out.** crabcrop caches processed results in three tiers, and the first two are local:

- The in-memory tier is **per replica**. A request served by replica B does not see what replica A already processed.
- The disk tier (`resultStorage`) is **per pod**, on that pod's volume. It is not shared unless you point every pod at a genuinely shared (ReadWriteMany) filesystem. The chart's own claim is ReadWriteOnce, so the multi-replica overlay disables the disk tier outright: several replicas cannot mount one single-writer volume, and the bucket tier below is the shared one.
- The bucket tier (`resultBucket`) is **shared**: every replica reads and writes it, and a shared hit promotes into the local tiers.

Replicas therefore trade cache efficiency: every replica produces identical bytes for a signed URL, so a cache miss just costs a re-fetch and re-process. One behavior does not cross replicas: a purge removes an entry from the receiving replica and the shared stores, not from other replicas' memory, so a purged entry can still be served by a replica holding it in memory until it evicts or restarts. Expect the aggregate hit rate of N replicas to be lower than one replica with the same total memory on the local tiers alone; the shared bucket is what keeps it whole. The multi-replica overlay lowers the per-replica memory tier to 256 MB for exactly this reason: size each replica's `CRABCROP_CACHE_MAX_MB` for its own share of the working set, and let the bucket hold the full set once.

## Teardown

```bash
helm uninstall crabcrop
kubectl delete secret crabcrop-secret
```

The second command is only for a secret you created yourself, as above: a chart-created Secret (the `--set secret=` trial route) is part of the release, and uninstalling removes it along with the workload.
