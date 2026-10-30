# crabcrop Helm chart

Deploys the crabcrop image service. The chart is deliberately thin: the server is configured entirely through the environment, so the chart wires the common settings structurally (secret, presets, storage, AI models) and passes everything else through a free-form `env` map.

## Prerequisites

- A Kubernetes cluster, with kubectl and Helm configured against it.
- An Ingress controller, only if you enable the chart's Ingress (`ingress.enabled`).

## The signing secret

Create the signing secret (16+ bytes; it signs and verifies every URL):

```bash
kubectl create secret generic crabcrop-secret \
  --from-literal=CRABCROP_SECRET="$(openssl rand -hex 32)"
```

For a quick trial only, `--set secret=<value>` lets the chart create the Secret itself.

## Install

Release images are published to GHCR as `ghcr.io/dayvson/crabcrop`, tagged with the release version; the chart defaults point there. To pin the exact image, set `image.digest` (and `ai.image.digest` for the AI variant) to a `sha256:...` digest. To run your own build instead:

```bash
docker build -t your-registry/crabcrop:0.9.5 .
docker push your-registry/crabcrop:0.9.5
```

Install:

```bash
helm install crabcrop deploy/helm/crabcrop \
  --set image.repository=your-registry/crabcrop \
  --set image.tag=0.9.5 \
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

See the Configuration table in the main README for the full list. An Ingress and an HPA are available behind `ingress.enabled` and `autoscaling.enabled`.

## Smoke checklist

With the service reachable at `<host>` (your Ingress host, or `kubectl port-forward svc/crabcrop 3000:80` and then `http://localhost:3000`):

1. Health: `curl https://<host>/health` returns 200 with the body `ok`.
2. Sign one request with the same secret and fetch it: `CRABCROP_SECRET=<secret> crabcrop sign "300x200/https://example.com/photo.jpg"`, then `curl -i "https://<host><signed-path>"` returns 200 and image bytes.
3. Fetch the same signed URL again. The `Cache-Status` response header reports a hit.
4. Update the Secret and restart the deployment (`kubectl rollout restart deployment/crabcrop`) and the pods pick it up on their next start; for zero-downtime rotation use the key ID flow in the main README (`CRABCROP_SECRET_PREVIOUS` plus key IDs).

## Scaling and cache notes

```bash
helm install crabcrop deploy/helm/crabcrop \
  -f deploy/helm/crabcrop/values-multireplica.yaml \
  --set existingSecret=crabcrop-secret
```

**Read this before scaling out.** crabcrop caches processed results in two tiers, and both are local:

- The in-memory tier is **per replica**. A request served by replica B does not see what replica A already processed.
- The disk tier (`resultStorage`) is **per pod**, on that pod's volume. It is not shared unless you point every pod at a genuinely shared (ReadWriteMany) filesystem.

Replicas therefore trade cache efficiency, never correctness: every replica produces identical bytes for a signed URL, so a cache miss just costs a re-fetch and re-process. Expect the aggregate hit rate of N replicas to be lower than one replica with the same total memory, and size each replica's `CRABCROP_CACHE_MAX_MB` (default 256) for its own share of the working set.

## Teardown

```bash
helm uninstall crabcrop
kubectl delete secret crabcrop-secret
```

The second command is only for a secret you created yourself, as above: a chart-created Secret (the `--set secret=` trial route) is part of the release, and uninstalling removes it along with the workload.
