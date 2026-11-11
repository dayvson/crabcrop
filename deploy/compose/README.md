# crabcrop with Docker Compose

Two runs ship in this directory:

- **Production** (`docker-compose.prod.yml`): one host as a real answer. Caddy terminates TLS in front, secrets live in an environment file, the disk result tier persists on a named volume, and the app port is never published.
- **Evaluation** (`docker-compose.yml`): a one-command trial. It builds the image from the repo Dockerfile, so no registry is involved, publishes the app port directly, and persists nothing. It is labelled as such where it appears.

Both build the same image from the repo Dockerfile.

## Prerequisites

Docker with the Compose plugin, and this repo checked out. That is all: no cluster, and no registry to push to or pull from. The production run also needs a DNS name pointing at the host, with ports 80 and 443 open, so Caddy can obtain its TLS certificate for the name.

## Production

### 1. The environment file

All of the app's configuration, secrets included, comes from one environment file. Copy the example, set the signing secret and the metrics token (the bearer <redacted> `/metrics` requires), and make the file owner-only:

```bash
cd deploy/compose
cp crabcrop.env.example crabcrop.env
# edit crabcrop.env: replace both CHANGE-ME values, each with
# its own `openssl rand -hex 32` output
chmod 600 crabcrop.env
```

The server refuses to start without `CRABCROP_SECRET`, so a file with the placeholders still in it fails fast at boot rather than running with a guessable secret. `crabcrop.env` is git-ignored; only the example belongs in the repo.

The public host name is not a secret, so it is not in that file: it goes in the compose environment instead (step 2), and never reaches the app or shares a file with the credentials.

### 2. Deploy

```bash
CRABCROP_HOST=images.example.com \
  docker compose -f docker-compose.prod.yml up -d --build
```

What the production file sets, against the evaluation run:

- **Caddy in front** (`caddy:2`, this directory's `Caddyfile`): the only published ports are 80 and 443. Caddy obtains and renews the certificate for `CRABCROP_HOST` automatically once the name resolves to this host, and proxies to the app over the compose network. `/health` and `/metrics` are not served on the public face at all (check them from inside, as the smoke checklist does): `/health` is unauthenticated, and `/metrics` exposes traffic and error counters.
- **The app unpublished**: no host port maps to it. Its resource shape matches the Helm production values (2 CPUs and 2 GB at the limit, the 512 MB memory result tier set in `crabcrop.env` inside that), and Caddy waits for the app's own healthcheck before it starts proxying.
- **Persistent results**: the disk result tier lives on the named volume `results`, so renders survive rebuilds and restarts. Its budget is `CRABCROP_RESULT_STORAGE_MAX_MB` in `crabcrop.env` (8 GiB as shipped). The server runs as `nobody` (uid 65534) and exits at boot when that directory is not writable; the image creates `/var/cache/crabcrop` owned by `nobody`, and Docker copies that ownership into the volume while it is empty. For a `results` volume created by an older image, see [Upgrading an existing volume](#upgrading-an-existing-volume).
- **Restart policy**: both services run `unless-stopped`.

The production environment also states the slow lane at one permit (long-running renders, such as animated encodes, never starve stills) and the 30-second processing limit, and refuses sources that resolve to non-public addresses. Raise the slow lane in `crabcrop.env` on hosts with cores to spare for animation or video traffic.

### Upgrading an existing volume

Images up to 0.9.10 did not create `/var/cache/crabcrop`, so a `results` volume first created by one of them is owned by root, and the server, which runs as `nobody`, could not write it: those servers ran with a dead disk tier. Such a volume is normally still empty, so it takes the new image's ownership when `up --build` recreates the container, and nothing else is needed. If it holds files, Docker leaves its ownership alone, and the server exits at boot with `the result storage directory … is not writable: Permission denied`. Hand the volume to `nobody` once, then start the stack as usual:

```bash
docker compose -f docker-compose.prod.yml run --rm --user root \
  --entrypoint chown crabcrop -R 65534:65534 /var/cache/crabcrop
CRABCROP_HOST=images.example.com \
  docker compose -f docker-compose.prod.yml up -d --build
```

A fresh or empty volume needs none of this.

### Smoke checklist (production)

1. Health, from inside (the public face does not serve it): `docker compose -f docker-compose.prod.yml exec crabcrop crabcrop health` prints `ok` and exits 0.
2. Sign one request and fetch it through the public name, using the CLI built into the image:

   ```bash
   docker compose -f docker-compose.prod.yml exec crabcrop \
     env CRABCROP_BASE=http://localhost:3000 crabcrop sign \
     "300x200/https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Fronalpstock_big.jpg/640px-Fronalpstock_big.jpg"
   ```

   The CLI prints a localhost URL; take its path and fetch it through Caddy: `curl -o out.jpg "https://images.example.com<the signed path>"`. The fetch returns 200 and image bytes.
3. Fetch the same signed URL again and look at the headers: `curl -i "https://images.example.com<the signed path>"`. The `Cache-Status` header reports a hit.
4. Purge it, through the CLI:

   ```bash
   docker compose -f docker-compose.prod.yml exec crabcrop \
     env CRABCROP_BASE=http://localhost:3000 crabcrop purge \
     "300x200/https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Fronalpstock_big.jpg/640px-Fronalpstock_big.jpg"
   ```

   It answers `purged 1` (or more, counting variants), and the next fetch of the signed URL is a miss again.
5. Metrics auth, from inside. The image carries no curl, so run it from a throwaway container on the app's network:

   ```bash
   docker run --rm --network "container:$(docker compose -f docker-compose.prod.yml ps -q crabcrop)" \
     curlimages/curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/metrics
   ```

   It returns 401 without a token. With the token from `crabcrop.env` (`-H "Authorization: Bearer <metrics-token>"` in place of `-o /dev/null -w "%{http_code}"`) it returns 200 and the Prometheus counters.

## Evaluation

### The signing secret

The secret has no default: compose refuses to start when `CRABCROP_SECRET` is unset or empty, rather than booting with a value anyone can read in this repo (audit R2-40). Anyone who knows the secret can sign URLs, so generate a fresh one per deployment. The run command below generates one inline.

### Deploy

```bash
cd deploy/compose
CRABCROP_SECRET=$(openssl rand -hex 32) docker compose up --build
```

### Smoke checklist

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

The image carries its own `HEALTHCHECK` (`crabcrop health`, which asks the server for `/health`), so `docker compose ps` reports health once the server is up.

## Scaling and cache notes

One container on one host: the in-memory tier lives in that container, and there is no second replica to share it with. The difference between the two runs is the second tier: the evaluation run mounts no volumes, so its results die with the container; the production run's disk tier persists, so only the first request for a render pays for it, across restarts as well.

One host is also the limit of this recipe. When it stops being enough, nothing here is throwaway: the same environment drives the Helm chart in `deploy/helm/crabcrop` (whose production values are the same shape, plus a shared bucket tier for several replicas) and the single-host binary recipe in `deploy/vm`.

## Teardown

Evaluation:

```bash
cd deploy/compose
docker compose down
```

Nothing persists: the evaluation file mounts no volumes. The built image (`crabcrop:latest`) stays in the local image store; remove it with `docker image rm crabcrop:latest` if you want it gone.

Production:

```bash
cd deploy/compose
docker compose -f docker-compose.prod.yml down      # keep the volumes
docker compose -f docker-compose.prod.yml down -v   # also delete them
```

The volumes hold the result cache (`results`) and Caddy's certificates and state (`caddy-data`, `caddy-config`); `-v` removes all three, plus `crabcrop.env` stays on disk until you remove it yourself.
