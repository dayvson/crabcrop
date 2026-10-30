# crabcrop with Docker Compose

A one-command evaluation run. It builds the image from the repo Dockerfile, so no registry is involved.

## Prerequisites

Docker with the Compose plugin, and this repo checked out. That is all: no cluster, and no registry to push to or pull from.

## The signing secret

The secret has no default: compose refuses to start when `CRABCROP_SECRET` is unset or empty, rather than booting with a value anyone can read in this repo (audit R2-40). Anyone who knows the secret can sign URLs, so generate a fresh one per deployment. The run command below generates one inline.

## Deploy

```bash
cd deploy/compose
CRABCROP_SECRET=$(openssl rand -hex 32) docker compose up --build
```

## Smoke checklist

1. Health: `curl http://localhost:3000/health` returns 200 with the body `ok`.
2. Sign one request with the secret the run started with, using the CLI built into the image, then fetch it:

   ```bash
   docker compose exec crabcrop \
     env CRABCROP_BASE=http://localhost:3000 crabcrop sign \
     "300x200/https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Fronalpstock_big.jpg/640px-Fronalpstock_big.jpg"
   curl -o out.jpg "<the signed URL it printed>"
   ```

   The fetch returns 200 and image bytes.
3. Fetch the same signed URL again and look at the headers: `curl -i "<the signed URL it printed>"`. The `Cache-Status` header reports a hit.

The image carries its own `HEALTHCHECK` (the `/health` endpoint), so `docker compose ps` reports health once the server is up.

## Scaling and cache notes

One container on one host: the in-memory cache lives in that container (the compose file sets `CRABCROP_CACHE_MAX_MB` to 256), and there is no second replica to share it with. That is the right shape for an evaluation, and the limit of it.

For presets, an AI variant, or anything beyond a trial, use the Helm chart in `deploy/helm/crabcrop`, the single-host recipe in `deploy/vm`, or run the binary directly with the environment from the main README's Configuration table.

## Teardown

```bash
cd deploy/compose
docker compose down
```

Nothing persists: the compose file mounts no volumes. The built image (`crabcrop:latest`) stays in the local image store; remove it with `docker image rm crabcrop:latest` if you want it gone.
