# crabcrop on a single host (VM or bare metal)

The release binary under systemd on one Linux host, with a
reverse proxy (Caddy or nginx) terminating TLS in front of
it. This is the production form of the single-host setup:
for a quick trial, the Compose recipe in `deploy/compose`
is faster to start; this one is the one to keep.

No container and no registry are involved. The caches live
on this host: the in-memory tier in the process, and the
disk tier in a directory the unit owns (step 3).

## Prerequisites

- A Linux host with systemd, and sudo on it.
- A reverse proxy on the host: Caddy or nginx. Step 4
  covers both.
- A DNS name for the service (this recipe uses
  `images.example.com`) pointing at the host, with ports
  80 and 443 open: the proxy gets its TLS certificate for
  that name. Port 3000 stays closed to the public
  internet; only the proxy talks to it. The Production
  notes in the main README's Deployment section explain
  why `/health` and `/metrics` need that protection.

## 1. Install the binary

Download the release tarball for your platform (macOS and
Linux, x86_64 and arm64) from the releases page, unpack
it, and put the binary on the system path:

```bash
tar -xzf crabcrop-<platform>.tar.gz
sudo install -m 0755 crabcrop /usr/local/bin/crabcrop
```

The shell installer from the main README's Install
section works too; note where it puts the binary and use
that path as the unit's `ExecStart` below.

## 2. The signing secret

Create the service user, the config directory, and the
environment file the unit reads. The file is root-only:
it holds the signing secret.

```bash
sudo useradd --system --no-create-home --shell /usr/sbin/nologin crabcrop
sudo install -d -m 0750 /etc/crabcrop
sudo cp crabcrop.env.example /etc/crabcrop/crabcrop.env
sudo chmod 0600 /etc/crabcrop/crabcrop.env
sudoedit /etc/crabcrop/crabcrop.env
```

Set `CRABCROP_SECRET` (16+ bytes; it signs and verifies
every URL) in the editor. `crabcrop.env.example` in this
directory is the starting point: the secret, the port,
and the disk tier directory. Any variable from the main
README's Configuration table can go in the same file.

## 3. The systemd unit

`crabcrop.service` in this directory runs the binary as
the `crabcrop` user, reads the environment file from
step 2, restarts it on failure, and creates the disk
tier directory (`/var/cache/crabcrop`) through
`CacheDirectory`:

```bash
sudo cp crabcrop.service /etc/systemd/system/crabcrop.service
sudo systemctl daemon-reload
sudo systemctl enable --now crabcrop
systemctl status crabcrop
```

## 4. The reverse proxy (TLS)

Pick one. Both examples in this directory proxy
`images.example.com` to `127.0.0.1:3000` and serve
`/health` and `/metrics` to localhost only: `/health` is
unauthenticated, and `/metrics` exposes traffic and
error counters.

**Caddy** (`Caddyfile`): Caddy obtains and renews the
certificate automatically once the DNS name points at
the host:

```bash
sudo cp Caddyfile /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

**nginx** (`nginx.conf`): a server block for the
hostname. Point its `ssl_certificate` lines at your
certificates (Certbot's `--nginx` plugin issues and
installs them), then:

```bash
sudo cp nginx.conf /etc/nginx/sites-available/crabcrop
sudo ln -s /etc/nginx/sites-available/crabcrop /etc/nginx/sites-enabled/crabcrop
sudo nginx -t && sudo systemctl reload nginx
```

## Smoke checklist

1. Health: `curl http://localhost:3000/health` on the
   host returns 200 with the body `ok`. (Through the
   proxy, `/health` answers only from localhost, by
   design.)
2. Sign one request with the same secret and fetch it
   through the proxy:
   `CRABCROP_SECRET=<secret> crabcrop sign "300x200/https://example.com/photo.jpg"`,
   then `curl -i "https://images.example.com<signed-path>"`
   returns 200 and image bytes.
3. Fetch the same signed URL again. The `Cache-Status`
   response header reports a hit.
4. Rotate the secret by editing the environment file and
   restarting (`sudo systemctl restart crabcrop`); for
   zero-downtime rotation use the key ID flow in the main
   README (`CRABCROP_SECRET_PREVIOUS` plus key IDs).

## Scaling and cache notes

One process on one host: the in-memory tier is this
process's (256 MB by default, `CRABCROP_CACHE_MAX_MB`),
and the disk tier is this host's directory, whose
entries survive restarts. That is the whole scaling
story, and for a single host it is enough: results are
content-addressed, so a tier that misses just costs a
re-fetch and re-process.

When one host stops being enough, nothing here is
throwaway: the same environment file drives the
Kubernetes and cloud recipes, and several hosts behind a
load balancer can share the third tier by setting
`CRABCROP_RESULT_BUCKET` (see the main README's
Configuration table).

## Teardown

```bash
sudo systemctl disable --now crabcrop
sudo rm /etc/systemd/system/crabcrop.service
sudo systemctl daemon-reload
sudo rm -rf /etc/crabcrop /var/cache/crabcrop
sudo rm /usr/local/bin/crabcrop
sudo userdel crabcrop
```

Remove the proxy site too: for Caddy, take the site out
of `/etc/caddy/Caddyfile` and reload; for nginx, remove
the `sites-enabled/crabcrop` link and the file in
`sites-available`, then reload.
