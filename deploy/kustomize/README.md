# crabcrop on Kubernetes with kustomize

The kustomize spelling of the same deployment the Helm
chart under `deploy/helm` renders. One workload, two
front-ends: pick per your platform standard, and keep the
two in step when the workload changes (the chart's values
are the reference for what each piece means).

Release images are published to GHCR as
`ghcr.io/dayvson/crabcrop` and `ghcr.io/dayvson/crabcrop-ai`,
tagged with the release version: point `image:` (base
`deployment.yaml`, or an overlay `images:` transformer) at
one of those, or at your own build of the repo Dockerfile.

## Prerequisites

- A Kubernetes cluster, with kubectl configured against
  it. kubectl carries kustomize (`apply -k`); no separate
  install is needed.

## The signing secret

The base references a Secret named `crabcrop` with the
key `CRABCROP_SECRET` (16+ bytes; it signs and verifies
every URL), and ships none on purpose. Create it with
kubectl or your own secret tooling before applying:

```sh
kubectl create secret generic crabcrop \
  --from-literal=CRABCROP_SECRET="$(openssl rand -hex 32)"
```

## Layout

- `base/`: the workload as the chart's default values
  render it. Deployment (one replica, `/health` probes,
  `PORT`, the secret reference) and Service. The base
  ships no Secret on purpose (audit R2-40): create the
  one from The signing secret above before applying;
  until it exists, the pod cannot start. The Helm
  chart's equivalent is its `existingSecret` value.
- `components/`: the chart's toggles, as kustomize
  components an overlay composes in:
  - `presets`: the presets ConfigMap plus its mount and
    `CRABCROP_PRESETS_FILE`. Edit `presets.txt` in place, or
    replace the ConfigMap with a `configMapGenerator` in
    your overlay. For presets-only mode, add
    `CRABCROP_ONLY_PRESETS=1` to the Deployment env from
    your overlay.
  - `result-storage`: the disk result tier (PVC plus
    `CRABCROP_RESULT_STORAGE_DIR` and its budget). One pod's
    disk: do not share the claim across replicas unless it
    is a genuinely shared (RWX) filesystem.
  - `ai`: the AI variant image (`Dockerfile.ai`) and a
    models volume referencing a claim named
    `crabcrop-models` that you provide, holding the ONNX
    models and runtime library. Point the model env vars
    (`CRABCROP_ORT_DYLIB`, `CRABCROP_REMOVEBG_MODEL`, and
    friends) at the mount from your overlay.
- `overlays/single`: the base as it stands.
- `overlays/multi-replica`: three replicas, resource
  budgets, and pod anti-affinity, mirroring the chart's
  `values-multireplica.yaml`. Its env patch shows the
  shared result tier: for several replicas, prefer
  `CRABCROP_RESULT_BUCKET` (one S3 bucket every replica
  reads and writes) over any shared disk.
- `overlays/full`: every component composed, as a build
  check and a starting point to trim.

## Deploy

```sh
kubectl apply -k deploy/kustomize/overlays/single
```

Free-form environment (the chart's `env` map) is an overlay
patch against the Deployment's container env, as
`overlays/multi-replica/env-patch.yaml` shows. Ingress and
autoscaling are yours to add the same way; the chart ships
them as values because Helm needs them parameterized,
while a kustomize overlay is already your own file.

## Smoke checklist

With the service reachable at `<host>` (an Ingress you
added, or `kubectl port-forward svc/crabcrop 3000:80`
and then `http://localhost:3000`):

1. Health: `curl https://<host>/health` returns 200 with
   the body `ok`.
2. Sign one request with the same secret and fetch it:
   `CRABCROP_SECRET=<secret> crabcrop sign "300x200/https://example.com/photo.jpg"`,
   then `curl -i "https://<host><signed-path>"` returns
   200 and image bytes.
3. Fetch the same signed URL again. The `Cache-Status`
   response header reports a hit.
4. Recreate the Secret with a new value and restart the
   deployment (`kubectl rollout restart deployment/crabcrop`)
   and the pods pick it up on their next start; for
   zero-downtime rotation use the key ID flow in the main
   README (`CRABCROP_SECRET_PREVIOUS` plus key IDs).

## Scaling and cache notes

The workload's caches are local to each pod: the
in-memory tier is per replica, and the result-storage
component's disk tier is one pod's volume. Replicas trade
cache hit rate, never correctness. The multi-replica
overlay composes three replicas with resource budgets
and pod anti-affinity, and its env patch shows the shared
result tier: for several replicas, prefer
`CRABCROP_RESULT_BUCKET` (one S3 bucket every replica
reads and writes) over any shared disk.

## Teardown

```sh
kubectl delete -k deploy/kustomize/overlays/single
kubectl delete secret crabcrop
```

Delete with the same overlay you applied. The Secret is
deleted separately because kustomize never owned it: you
created it, in The signing secret above. Claims created
by the result-storage component go with the overlay; a
shared claim you provided stays yours.
