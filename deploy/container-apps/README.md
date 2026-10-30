# crabcrop on Azure Container Apps

A Container Apps app from `containerapp.yaml` in this
directory, ingress in front, the signing secret in the app's
secret store.

**Cache caveat, before anything else.** The in-memory result
cache lives inside one replica, and the disk tier is
per-replica (Container Apps volumes are not a shared cache).
Replicas trade cache hit rate, never correctness.
`containerapp.yaml` keeps `minReplicas: 1` so one replica
(and its cache) stays warm. Azure has no S3-compatible store
of its own; for the shared result tier, point
`CRABCROP_S3_ENDPOINT` at an S3-compatible gateway if you
run one, and otherwise plan on per-replica caches.

Release images are published to GHCR as
`ghcr.io/dayvson/crabcrop`, tagged with the release
version. Replace the `REPLACE_ME` values in
`containerapp.yaml` (location, environment resource ID,
image, and the placeholder secret value), pointing the
image at `ghcr.io/dayvson/crabcrop:0.9.5` or at your own
build of the repo Dockerfile pushed to your Azure
Container Registry.

## Prerequisites

- An Azure subscription and a resource group, with the
  Azure CLI configured.
- For the Key Vault route in step 1: a vault, and a
  user-assigned managed identity for the app.

## 1. The signing secret

The shipped YAML carries the secret as a plain value marked
`REPLACE_ME_signing_secret`, which is for a first boot only.
For anything real, use a Key Vault reference. Give the app
a user-assigned managed identity with `Key Vault Secrets
User` on the vault, then in `containerapp.yaml`:

- add the identity at the resource level:

```yaml
identity:
  type: UserAssigned
  userAssignedIdentities:
    "<identity-resource-id>": {}
```

- and replace the secret entry:

```yaml
    secrets:
      - name: crabcrop-secret
        keyVaultUrl: https://<vault>.vault.azure.net/secrets/crabcrop-secret
        identity: <identity-resource-id>
```

The container already reads it as
`CRABCROP_SECRET` through `secretRef`, which does not
change.

## 2. Deploy

```sh
az containerapp env create \
  --name <environment> --resource-group <group> --location <location>

az containerapp create \
  --name crabcrop --resource-group <group> \
  --environment <environment> \
  --yaml containerapp.yaml
```

The YAML sets `PORT=3000` explicitly, matches it with the
ingress `targetPort`, and declares liveness and readiness
probes on `/health`. Expose the app through your edge (Front
Door or your CDN) as you would for any origin; the main
README's Deployment section explains why `/health` and
`/metrics` should stay internal.

## 3. Presets (optional)

Container Apps can mount secrets as files. Store the
presets text as a second secret (`crabcrop-presets`) in the
app's secret store, then add to `containerapp.yaml`: under
`template`, a secret-backed volume and a mount, and the
environment variable beside the others:

```yaml
        env:
          - name: CRABCROP_PRESETS_FILE
            value: /etc/crabcrop/presets.txt
        volumeMounts:
          - volumeName: presets
            mountPath: /etc/crabcrop
    volumes:
      - name: presets
        storageType: Secret
        secrets:
          - secretRef: crabcrop-presets
            path: presets.txt
```

## 4. Disk tier (optional)

A mounted Azure Files volume at `/var/cache/crabcrop` plus
`CRABCROP_RESULT_STORAGE_DIR=/var/cache/crabcrop` and
`CRABCROP_RESULT_STORAGE_MAX_MB` gives each replica a
persistent disk tier. Treat it as per-replica storage:
replicas do not coordinate on a shared volume, which is
safe (entries are content-addressed) but no better than
several private disks.

## Smoke checklist

With the app's FQDN `<host>`:

1. Health: `curl https://<host>/health` returns 200 with
   the body `ok`.
2. Sign one request with the same secret and fetch it:
   `CRABCROP_SECRET=<secret> crabcrop sign "300x200/https://example.com/photo.jpg"`,
   then `curl -i "https://<host><signed-path>"` returns 200
   and image bytes.
3. Fetch the same signed URL again. The `Cache-Status`
   response header reports a hit.
4. Update the secret in Key Vault and start a new revision
   to pick it up; for zero-downtime rotation use the key ID
   flow in the main README (`CRABCROP_SECRET_PREVIOUS` plus
   key IDs).

## Teardown

```sh
az containerapp delete --name crabcrop --resource-group <group> --yes
az containerapp env delete --name <environment> --resource-group <group> --yes
```

Delete the Key Vault secret and the managed identity from
step 1 as well, if the app was their only consumer. The
disk tier's Azure Files volume is yours to keep or delete;
crabcrop does not delete stored results.
