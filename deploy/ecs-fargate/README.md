# crabcrop on AWS ECS Fargate

A Fargate service behind an Application Load Balancer, with
the signing secret in Secrets Manager. `task-definition.json`
in this directory is the whole container wiring; the service
and the load balancer are created with the commands below.

**Cache caveat, before anything else.** The in-memory result
cache lives inside one task, and the disk tier is per-task
unless you attach shared storage. Running more tasks trades
cache hit rate, never correctness. On AWS the shared result
tier is native: set `CRABCROP_RESULT_BUCKET` (see step 5) and
every task reads and writes one S3 bucket.

Nothing is published to a public registry yet. Build the
image from the repo Dockerfile and push it to your ECR
repository, then replace the `REPLACE_ME` values in
`task-definition.json`:

```sh
docker build -t crabcrop:latest .
docker tag crabcrop:latest <account>.dkr.ecr.<region>.amazonaws.com/crabcrop:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/crabcrop:latest
```

## 1. The signing secret

```sh
aws secretsmanager create-secret \
  --name crabcrop/secret \
  --secret-string 'use-a-long-random-value-here'
```

The task's **execution role** needs
`secretsmanager:GetSecretValue` on that secret, or the task
will not start. Put the secret's ARN in the task definition
where it says `REPLACE_ME_secrets_manager_arn_for_crabcrop_secret`.

## 2. Register the task definition

```sh
aws ecs register-task-definition \
  --cli-input-json file://task-definition.json
```

The definition sets `PORT=3000` explicitly and takes
`CRABCROP_SECRET` from Secrets Manager through the
container's `secrets` block. There is deliberately no
container `healthCheck`: the target group health check in
the next step is what gates traffic, and it checks the same
`/health` route.

## 3. Load balancer and service

Create an ALB with a target group whose health check is
`GET /health` on port 3000 (healthy threshold 2), then the
service:

```sh
aws ecs create-service \
  --cluster <cluster> \
  --service-name crabcrop \
  --task-definition crabcrop \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration 'awsvpcConfiguration={subnets=[<subnet>],securityGroups=[<sg>],assignPublicIp=DISABLED}' \
  --load-balancers 'targetGroupArn=<tg-arn>,containerName=crabcrop,containerPort=3000'
```

Restrict the ALB to your edge (CDN or WAF) as you would for
any origin; the main README's Deployment section explains
why `/health` and `/metrics` should stay internal.

## 4. Presets (optional)

Presets are a file (`CRABCROP_PRESETS_FILE`). On Fargate the
two workable routes:

- Bake `presets.txt` into a derived image
  (`FROM your-crabcrop` plus one `COPY`) and set
  `CRABCROP_PRESETS_FILE` to its path as a plain environment
  entry in the task definition.
- Mount an EFS file system holding `presets.txt` (a volume
  plus a mount point in the task definition) with several
  tasks sharing it.

## 5. Shared result tier and disk tier (optional)

Shared tier, recommended once `desired-count` is above 1.
Add plain environment entries:

```json
{"name": "CRABCROP_RESULT_BUCKET", "value": "your-results-bucket"},
{"name": "CRABCROP_RESULT_PREFIX", "value": "prod/"}
```

The **task role** (not the execution role) then needs
`s3:GetObject`, `s3:PutObject`, `s3:DeleteObject`, and
`s3:ListBucket` on that bucket. Credentials come from the
task role through the ambient AWS chain; nothing else to
wire.

Disk tier per task: mount EFS at `/var/cache/crabcrop` and
add `CRABCROP_RESULT_STORAGE_DIR=/var/cache/crabcrop` plus
`CRABCROP_RESULT_STORAGE_MAX_MB`. Without EFS the Fargate
ephemeral disk works too, but it disappears with the task.

## Smoke checklist

With the service behind its ALB DNS name `<host>`:

1. Health: `curl https://<host>/health` returns 200 with
   the body `ok`.
2. Sign one request with the same secret and fetch it:
   `CRABCROP_SECRET=<secret> crabcrop sign "300x200/https://example.com/photo.jpg"`,
   then `curl -i "https://<host><signed-path>"` returns 200
   and image bytes.
3. Fetch the same signed URL again. The `Cache-Status`
   response header reports a hit.
4. Rotate the secret in Secrets Manager later and the tasks
   pick it up on their next start; for zero-downtime
   rotation use the key ID flow in the main README
   (`CRABCROP_SECRET_PREVIOUS` plus key IDs).
