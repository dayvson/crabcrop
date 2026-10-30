# crabcrop on Google Cloud Run

A Cloud Run service from `service.yaml` in this directory,
with the signing secret in Secret Manager.

**Cache caveat, before anything else.** The in-memory result
cache lives inside one instance, and instance disk is
ephemeral, so the disk tier buys little here. Instances
trade cache hit rate, never correctness. `service.yaml`
keeps `minScale: 1` so one instance (and its cache) stays
warm; for a cache every instance shares, use the shared
result tier in step 4.

**The PORT exception.** Cloud Run owns the `PORT`
environment variable, injects it (8080 by default), and the
server reads it, so this recipe does **not** set `PORT`.
The declared `containerPort: 8080` in `service.yaml`
matches the injection. Everywhere else in these recipes
`PORT` is set explicitly; here the platform setting it is
the wiring.

Release images are published to GHCR as
`ghcr.io/dayvson/crabcrop`, tagged with the release
version. Replace the `REPLACE_ME` image in `service.yaml`
with `ghcr.io/dayvson/crabcrop:0.9.5`, or with your own
build of the repo Dockerfile pushed to Artifact Registry.

## Prerequisites

- A GCP project with Cloud Run and Secret Manager
  available, and the gcloud CLI configured for it.
- A runtime service account for the service. Step 1 gives
  it access to the signing secret, and step 4 uses it
  again for the shared result tier.

## 1. The signing secret

```sh
printf 'use-a-long-random-value-here' | \
  gcloud secrets create crabcrop-secret --data-file=-

gcloud secrets add-iam-policy-binding crabcrop-secret \
  --member='serviceAccount:<runtime-service-account>' \
  --role='roles/secretmanager.secretAccessor'
```

`service.yaml` reads it as the `CRABCROP_SECRET`
environment variable through `secretKeyRef` with
`key: latest`.

## 2. Deploy

```sh
gcloud run services replace service.yaml
gcloud run services set-iam-policy crabcrop policy.yaml  # or add --allow-unauthenticated behind your edge
```

The service declares a startup probe on `/health` with
room for the AI variant's model preload (30 tries at 2
seconds) and a liveness probe on the same route. Put the
service behind your load balancer or CDN as you would for
any origin; the main README's Deployment section explains
why `/health` and `/metrics` should stay internal.

## 3. Presets (optional)

Secret Manager can also deliver the presets file. Create a
second secret holding the presets text
(`gcloud secrets create crabcrop-presets`), then add to
`service.yaml`: a volume from that secret, a mount at
`/etc/crabcrop`, and the environment variable. Add under
the container:

```yaml
          env:
            - name: CRABCROP_PRESETS_FILE
              value: /etc/crabcrop/presets.txt
          volumeMounts:
            - name: presets
              mountPath: /etc/crabcrop
              readOnly: true
```

and under the revision spec, beside `containers`:

```yaml
      volumes:
        - name: presets
          secret:
            secretName: crabcrop-presets
            items:
              - key: latest
                path: presets.txt
```

## 4. Shared result tier (optional)

The shared tier speaks S3, and Cloud Storage's XML API is
S3-compatible. Add plain environment entries (the HMAC keys
come from Secret Manager like the signing secret):

```yaml
            - name: CRABCROP_RESULT_BUCKET
              value: your-results-bucket
            - name: CRABCROP_S3_ENDPOINT
              value: https://storage.googleapis.com
            - name: AWS_ACCESS_KEY_ID
              valueFrom:
                secretKeyRef: { name: crabcrop-gcs-hmac-id, key: latest }
            - name: AWS_SECRET_ACCESS_KEY
              valueFrom:
                secretKeyRef: { name: crabcrop-gcs-hmac-key, key: latest }
```

Create the HMAC keys for the runtime service account in
the Cloud Storage settings, and give that account object
admin on the one bucket. (`CRABCROP_S3_ENDPOINT` and
`CRABCROP_S3_REGION` are the same settings the S3 source
loader uses; the main README's configuration table lists
them.)

## Smoke checklist

With the service URL `<host>`:

1. Health: `curl https://<host>/health` returns 200 with
   the body `ok`.
2. Sign one request with the same secret and fetch it:
   `CRABCROP_SECRET=<secret> crabcrop sign "300x200/https://example.com/photo.jpg"`,
   then `curl -i "https://<host><signed-path>"` returns 200
   and image bytes.
3. Fetch the same signed URL again. The `Cache-Status`
   response header reports a hit. (From a cold scale-to-zero
   event the first fetch repopulates instead; that is the
   per-instance cache behaving as documented.)
4. Update the secret's value in Secret Manager and new
   revisions pick it up; for zero-downtime rotation use the
   key ID flow in the main README (`CRABCROP_SECRET_PREVIOUS`
   plus key IDs).

## Teardown

```sh
gcloud run services delete crabcrop
gcloud secrets delete crabcrop-secret
```

If you created the presets secret or the HMAC secrets from
steps 3 and 4, delete those too. The shared result tier's
bucket is yours to keep or empty; crabcrop does not delete
stored results.
