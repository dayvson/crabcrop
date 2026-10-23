# crabcrop Helm chart

Deploys the crabcrop image service. The chart is deliberately thin: the server is configured entirely through the environment, so the chart wires the common settings structurally (secret, presets, storage, AI models) and passes everything else through a free-form `env` map.

## Install

Nothing is published to a public registry yet. Build the image from the repo and push it somewhere your cluster can pull from:

```bash
docker build -t your-registry/crabcrop:0.1.0 .
docker push your-registry/crabcrop:0.1.0
```

Create the signing secret (16+ bytes; it signs and verifies every URL):

```bash
kubectl create secret generic crabcrop-secret \
  --from-literal=CRABCROP_SECRET="$(openssl rand -hex 32)"
```

Install:

```bash
helm install crabcrop deploy/helm/crabcrop \
  --set image.repository=your-registry/crabcrop \
  --set image.tag=0.1.0 \
  --set existingSecret=crabcrop-secret
```

For a quick trial only, `--set secret=<value>` lets the chart create the Secret itself.

## Multiple replicas, and the cache caveat

```bash
helm install crabcrop deploy/helm/crabcrop \
  -f deploy/helm/crabcrop/values-multireplica.yaml \
  --set existingSecret=crabcrop-secret
```

**Read this before scaling out.** crabcrop caches processed results in two tiers, and both are local:

- The in-memory tier is **per replica**. A request served by replica B does not see what replica A already processed.
- The disk tier (`resultStorage`) is **per pod**, on that pod's volume. It is not shared unless you point every pod at a genuinely shared (ReadWriteMany) filesystem.

Replicas therefore trade cache efficiency, never correctness: every replica produces identical bytes for a signed URL, so a cache miss just costs a re-fetch and re-process. Expect the aggregate hit rate of N replicas to be lower than one replica with the same total memory, and size each replica's `CRABCROP_CACHE_MAX_MB` (default 256) for its own share of the working set.

## Presets

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
