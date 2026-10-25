# crabcrop with Docker Compose

A one-command evaluation run. It builds the image from the repo Dockerfile, so no registry is involved.

```bash
cd deploy/compose
CRABCROP_SECRET=$(openssl rand -hex 32) docker compose up --build
```

The secret has no default: compose refuses to start when `CRABCROP_SECRET` is unset or empty, rather than booting with a value anyone can read in this repo (audit R2-40). Anyone who knows the secret can sign URLs, so generate a fresh one per deployment.

Check it answers:

```bash
# Health
curl http://localhost:3000/health

# Sign a URL with the CLI built into the image, then fetch it
docker compose exec crabcrop \
  env CRABCROP_BASE=http://localhost:3000 crabcrop sign \
  "300x200/https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Fronalpstock_big.jpg/640px-Fronalpstock_big.jpg"
curl -o out.jpg "<the signed URL it printed>"
```

The image carries its own `HEALTHCHECK` (the `/health` endpoint), so `docker compose ps` reports health once the server is up.

For presets, an AI variant, or anything beyond a trial, use the Helm chart in `deploy/helm/crabcrop` or run the binary directly with the environment from the main README's Configuration table.
