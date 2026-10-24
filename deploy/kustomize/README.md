# crabcrop on Kubernetes with kustomize

The kustomize spelling of the same deployment the Helm
chart under `deploy/helm` renders. One workload, two
front-ends: pick per your platform standard, and keep the
two in step when the workload changes (the chart's values
are the reference for what each piece means).

Nothing is published to a public registry yet: build the
image from the repo Dockerfile, push it somewhere your
cluster can pull from, and point `image:` (base
`deployment.yaml`, or an overlay `images:` transformer) at
it.

## Layout

- `base/`: the workload as the chart's default values
  render it. Deployment (one replica, `/health` probes,
  `PORT`, the secret reference), Service, and a Secret with
  a **placeholder** value for local evaluation. For anything
  real, manage the Secret yourself (a Secret named
  `crabcrop` with the key `CRABCROP_SECRET`, 16+ bytes) and
  drop `secret.yaml` from the base's resources.
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

## Use

```sh
kubectl apply -k deploy/kustomize/overlays/single
```

Free-form environment (the chart's `env` map) is an overlay
patch against the Deployment's container env, as
`overlays/multi-replica/env-patch.yaml` shows. Ingress and
autoscaling are yours to add the same way; the chart ships
them as values because Helm needs them parameterized,
while a kustomize overlay is already your own file.
