# eased — GPU Worker Orchestrator

Agentic orchestrator that rents cheap GPU servers on [vast.ai](https://vast.ai), bakes
model weights into a Docker image, and proxies OpenAI-compatible vLLM requests through
a single stable endpoint. Production deployment runs on **ECS Fargate Spot** (no GPU
required). Worker nodes run on vast.ai interruptible instances, with automatic on-demand
fallback.

---

## Architecture

```
Client
  │  HTTPS
  ▼
API Gateway  ──►  Lambda Authorizer (Bearer token)
  │
  ▼
ECS Fargate Spot  ──  Orchestrator (FastAPI)
  │  manages + proxies
  ├──► vast.ai interruptible GPU worker  ◄── vLLM + Nemotron 12B
  └──► vast.ai on-demand GPU worker (fallback)
       (spun up only when interruptible is reclaimed and pool is empty)

State:  AWS DynamoDB  (worker records survive orchestrator restarts)
Alerts: Discord webhook
```

### Orchestrator responsibilities

| Concern | Behaviour |
|---|---|
| **Bidding** | Starts at 50 % of median market price; retries every 5 min at +5 % until filled or cap (110 %) is hit |
| **On-demand fallback** | Cold-starts one on-demand worker only when no interruptible worker exists |
| **Proxy** | Forwards all `/v1/*` calls to the active worker; injects per-worker Bearer token |
| **Health checking** | Pings `/health` every 30 s; marks worker UNHEALTHY on first failure, TERMINATED after 3 consecutive failures |
| **Vast.ai sync** | Every 60 s cross-checks DynamoDB workers against live vast.ai instances; detects reclaimed/orphaned instances |
| **Startup discovery** | On start, scans vast.ai for owned instances not in DynamoDB and registers them (prevents redundant bids after a DB wipe) |
| **Orphan cleanup** | Destroys instances with our label or image that have no DB record |
| **Recovery** | Destroys the failed instance, starts a new bid campaign automatically |
| **State persistence** | DynamoDB — survives Fargate task restarts |
| **Notifications** | Discord webhook for every meaningful lifecycle event + periodic fleet status reports (every 30 min) |
| **Cost tracking** | Tracks per-instance running time and cost; reported in periodic Discord summaries and `/admin/health` |
| **Template management** | Creates/reuses a reusable vast.ai instance template on startup; avoids re-specifying all instance fields on each bid |
| **SSH log access** | Injects an Ed25519 public key into every new instance so it can SSH in to tail `/tmp/vllm.log` during startup |

### Worker image

`ghcr.io/easedai/nemotron:latest` — built and published by the
[`build-worker.yml`](.github/workflows/build-worker.yml) workflow in this repo.
Bakes the full Nemotron Nano 12B BF16 weights into the vLLM base image.

- **Size**: ~40 GB (vLLM base ~15 GB + model ~24 GB)
- **Cold start**: ~5–10 min (image pull on a fresh vast.ai host) + model load
- **Warm start** (same host, layer-cached): model load only (~5 min)
- **Security**: `VLLM_API_KEY` is generated per-instance by the orchestrator and
  injected via vast.ai environment variables — never hardcoded
- **vLLM patch**: A startup patch is applied to every new instance via `EXTRA_COMMANDS`
  to fix a vLLM 0.19.0 crash (`encoder_budget.py`) with the NanoNemotronVLProcessor

---

## Local development

### Prerequisites

- Docker + Docker Compose v2
- A [vast.ai](https://vast.ai) account and API key
- An AWS account with DynamoDB access (via SSO profile)
- A Discord webhook URL

### Quick start

```bash
cd eased
cp .env.example .env
# Edit .env — fill in VASTAI_API_KEY, DISCORD_WEBHOOK_URL, ADMIN_TOKEN, and AWS_PROFILE at minimum
# Generate ADMIN_TOKEN with: openssl rand -hex 32

# Authenticate your AWS SSO session (DynamoDB runs in real AWS, not locally)
aws sso login --profile <your-profile>

docker compose up --build
```

The orchestrator starts at **http://localhost:8000** with hot-reload enabled.
AWS credentials are mounted from `~/.aws` — the container uses your host SSO session.

On startup the orchestrator will scan vast.ai for any existing instances it owns,
register them in DynamoDB, and then only bid for a new worker if none are found.

### API endpoints

| Endpoint | Auth | Description |
|---|---|---|
| `GET  /health` | None | Basic liveness probe — returns status + uptime |
| `GET  /admin/health` | Bearer | Detailed fleet health: worker counts, per-instance cost, spend rate |
| `GET  /admin/workers` | Bearer | List all workers and their full state |
| `POST /admin/workers/refresh` | Bearer | Re-sync DynamoDB with live vast.ai state |
| `POST /admin/workers/{id}/terminate` | Bearer | Destroy a specific worker |
| `POST /admin/bid` | Bearer | Manually trigger a new bid campaign |
| `GET  /admin/template` | Bearer | Show the current vast.ai template ID cached by the orchestrator |
| `POST /admin/template/refresh` | Bearer | Re-run template creation (use after manually deleting the template in vast.ai) |
| `GET  /admin/events/worker/{worker_id}` | Bearer | Event history for a worker |
| `GET  /admin/events/instance/{instance_id}` | Bearer | Event history for a vast.ai instance |
| `GET  /admin/events/label/{label}` | Bearer | Event history by instance label (e.g. `eased-abc123`) |
| `GET  /v1/models` | — | Proxied to the active vLLM worker |
| `POST /v1/chat/completions` | — | Proxied to the active vLLM worker |

All `/admin/*` endpoints require `Authorization: Bearer <ADMIN_TOKEN>`.

### Bruno collection

A [Bruno](https://www.usebruno.com/) API collection is included in `bruno/`.

```bash
# Open in Bruno desktop app
# Select the "Local" environment (pre-configured with base_url + admin_token)
```

---

## Worker states

```
BIDDING → PENDING → STARTING → RUNNING
                                  │
                         (health check fail × 1)
                                  │
                              UNHEALTHY
                                  │
                         (health check fail × 3 total)
                                  │
                             TERMINATED → (new bid campaign)
```

| State | Description |
|---|---|
| `BIDDING` | Searching for a GPU offer and placing bids |
| `PENDING` | vast.ai instance created; waiting for container to start |
| `STARTING` | Container running; waiting for vLLM `/health` to return 200 |
| `RUNNING` | vLLM is serving traffic |
| `UNHEALTHY` | Health check failing; traffic paused; watching for recovery |
| `DRAINING` | Worker is being gracefully wound down (not yet terminated) |
| `TERMINATED` | Worker is dead; triggers a new bid campaign if pool is empty |

---

## Building the worker image

### Locally

```bash
cd eased/worker

# HF_TOKEN must be exported or set inline
DOCKER_BUILDKIT=1 docker build \
  --secret id=HF_TOKEN,env=HF_TOKEN \
  -t ghcr.io/easedai/nemotron:latest \
  .
```

The HuggingFace token is passed as a BuildKit secret — it is **never** embedded in
any image layer and will not appear in `docker history`.

### Via GitHub Actions

The `build-worker.yml` workflow runs on standard `ubuntu-latest` runners.
Registry-based layer caching (`type=registry`) is used so the ~24 GB model layer
survives between runs without hitting the 10 GB GitHub Actions cache limit.

**Required GitHub secrets:**

| Secret | Value |
|---|---|
| `HF_TOKEN` | HuggingFace token with access to the gated Nemotron model |
| `GITHUB_TOKEN` | Auto-provisioned — no action needed |

### Orchestrator CI/CD (`build-orchestrator.yml`)

Runs on `ubuntu-latest`. On every push to `main` that touches `orchestrator/` or
`terraform/`, it:

1. Builds and pushes `ghcr.io/easedai/eased-orchestrator:sha-<sha>` to GHCR
2. Runs `terraform apply` via the pinned SHA tag — ensuring the ECS service always
   runs exactly the just-built image

**Required GitHub secrets for orchestrator CI:**

| Secret | Value |
|---|---|
| `AWS_ACCESS_KEY_ID` | AWS credentials for Terraform |
| `AWS_SECRET_ACCESS_KEY` | AWS credentials for Terraform |
| `TF_VAR_VASTAI_API_KEY` | vast.ai API key |
| `TF_VAR_DISCORD_WEBHOOK_URL` | Discord webhook URL |
| `TF_VAR_HF_TOKEN` | HuggingFace token (stored in Secrets Manager by Terraform) |
| `TF_VAR_AUTHORIZER_TOKEN` | API Gateway Lambda authorizer token |
| `TF_VAR_GHCR_PAT` | GitHub PAT (read:packages) for ECS to pull the private orchestrator image |

---

## GPU provider options

The orchestrator currently targets [vast.ai](https://vast.ai). The `source_type` field on each LB worker record is designed to support multiple providers in the future.

### Marketplace / spot (similar model to vast.ai)

| Provider | Notes |
|---|---|
| [RunPod](https://runpod.io) | Most similar to vast.ai — spot pods, REST API, active supply. Easiest next integration. |
| [TensorDock](https://tensordock.com) | GPU marketplace, similar bidding model, often cheaper |
| [Salad.com](https://salad.com) | Distributed consumer GPUs, very cheap, less reliable uptime |
| [FluidStack](https://fluidstack.io) | GPU marketplace with preemptible instances |

### Dedicated cloud (more stable, higher cost)

| Provider | Notes |
|---|---|
| [Lambda Labs](https://lambdalabs.com) | H100 / A100 clusters, REST API, on-demand only |
| [CoreWeave](https://coreweave.com) | Kubernetes-native, best H100 availability, enterprise focus |
| [Hyperstack](https://hyperstack.cloud) | NVIDIA cloud, competitive H100 pricing |
| [DataCrunch](https://datacrunch.io) | European, A100 / H100, spot + on-demand |

### Aggregators (single API across multiple providers)

| Provider | Notes |
|---|---|
| [Shadeform](https://shadeform.ai) | One API across CoreWeave, Lambda, RunPod, and others |
| [Brev.dev](https://brev.dev) | Similar aggregator model |

---

## Bidding strategy

The orchestrator queries the vast.ai marketplace for all interruptible GPU offers with:
- ≥ 40 GB VRAM (A100 40 GB is the practical minimum for useful context lengths)
- ≥ 100 GB disk
- ≥ 300 Mbps download
- ≥ 90 % reliability rating
- 1 GPU
- **North America only** (US and Canada datacenters)

The **market price** is defined as the median `dph_base` across all matching offers.

| Attempt | Bid | Behaviour |
|---|---|---|
| 1 | 50 % of market | Wait 5 min |
| 2 | 55 % of market | Wait 5 min |
| 3 | 60 % of market | Wait 5 min |
| … | … | … |
| Cap | > 110 % of market | Fall back to on-demand |

The winning multiplier from the previous campaign is remembered — the next campaign
starts one step below it to probe whether a cheaper bid will land.

All thresholds are configurable via environment variables (see below).

---

## Production deployment (ECS Fargate Spot)

The orchestrator is stateless (state lives in DynamoDB) and runs without a GPU,
making it a perfect fit for Fargate Spot.

All AWS infrastructure is managed by Terraform in `terraform/`. The `build-orchestrator.yml`
workflow runs `terraform apply` automatically on each deploy. Resources provisioned:

- API Gateway HTTP API + Lambda authorizer (Bearer token)
- ECS Fargate Spot cluster + service (0.5 vCPU / 1 GB)
- DynamoDB tables: `eased-workers` and `eased-instance-events` (7-day TTL)
- VPC, subnets, security groups, IAM roles
- AWS Secrets Manager secrets: vast.ai API key, admin token, Discord webhook, SSH private key, GHCR PAT
- CloudWatch log group (30-day retention)
- SSH key pair (Ed25519) in `ssh_keys.tf` — stable across task restarts

The orchestrator's SSH private key (`ORCHESTRATOR_SSH_PRIVATE_KEY`) is injected at
container start from Secrets Manager. Using a stable key avoids re-uploading keys to
vast.ai on every Fargate task restart.

---

## Environment variables reference

### Required

| Variable | Description |
|---|---|
| `VASTAI_API_KEY` | vast.ai API key |
| `DISCORD_WEBHOOK_URL` | Discord webhook for alerts |
| `ADMIN_TOKEN` | Bearer token for `/admin/*` endpoints. Generate with `openssl rand -hex 32` |

### Worker image

| Variable | Default | Description |
|---|---|---|
| `WORKER_IMAGE` | `easedai/nemotron-vastai:latest` | Docker Hub image for the worker |
| `WORKER_DISK_GB` | `100.0` | Disk space (GB) to request for each worker instance |
| `GHCR_USERNAME` | — | GitHub username/org for private GHCR image pull |
| `GHCR_PAT` | — | GitHub PAT (read:packages) for private GHCR image pull |

### Model / vLLM

| Variable | Default | Description |
|---|---|---|
| `MODEL_ID` | `nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16` | HuggingFace model ID passed to vLLM |
| `HF_HOME` | `/hf` | HuggingFace cache dir — must match the path baked into the worker image |
| `HF_HUB_ENABLE_HF_TRANSFER` | `1` | Enable fast HF transfers (hf_transfer) |
| `VLLM_CACHE_ROOT` | `/vllm-cache` | vLLM compiled Triton kernel / torch.compile cache |
| `VLLM_PORT` | `8080` | Container port vLLM listens on |
| `VLLM_MAX_MODEL_LEN` | `32768` | Context length. Use `131072` only with ≥ 80 GB VRAM |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.95` | Fraction of GPU VRAM reserved for vLLM |
| `VLLM_VIDEO_LOADER_BACKEND` | `opencv` | Required for `--video-pruning-rate` (do not change) |

### vast.ai template

| Variable | Default | Description |
|---|---|---|
| `VASTAI_TEMPLATE_ID` | — | Pre-existing template hash_id to use (skips template creation) |
| `VASTAI_TEMPLATE_NAME` | `eased-nemotron` | Name of the template the orchestrator creates/looks up |

### AWS / DynamoDB

| Variable | Default | Description |
|---|---|---|
| `DYNAMODB_TABLE` | `eased-workers` | DynamoDB worker state table |
| `EVENTS_TABLE` | `eased-instance-events` | Instance event log table (7-day TTL) |
| `DYNAMODB_ENDPOINT_URL` | AWS | Override for local dev (set automatically by docker-compose) |
| `AWS_REGION` | `us-east-1` | AWS region |

### Bidding

| Variable | Default | Description |
|---|---|---|
| `BID_START_PCT` | `0.50` | Starting bid as fraction of market |
| `BID_STEP_PCT` | `0.05` | Increment per retry |
| `BID_RETRY_INTERVAL_SEC` | `300` | Seconds between retries |
| `BID_MAX_MULTIPLIER` | `1.10` | Fallback threshold (> market × this triggers on-demand) |

### GPU requirements

| Variable | Default | Description |
|---|---|---|
| `MIN_GPU_RAM_GB` | `40` | Minimum GPU VRAM (GB) |
| `MIN_DISK_GB` | `100` | Minimum instance disk (GB) |
| `MIN_INET_DOWN_MBPS` | `300` | Minimum download speed (Mbps) |
| `MIN_RELIABILITY` | `0.90` | Minimum host reliability score |

### Health checking

| Variable | Default | Description |
|---|---|---|
| `HEALTH_CHECK_INTERVAL_SEC` | `30` | Worker HTTP ping interval |
| `HEALTH_CHECK_TIMEOUT_SEC` | `10` | Per-request timeout for health pings |
| `HEALTH_CHECK_FAIL_THRESHOLD` | `3` | Consecutive failures before termination |
| `WORKER_STARTUP_TIMEOUT_SEC` | `900` | Max wait for vLLM to become healthy (15 min) |
| `VAST_CHECK_INTERVAL_SEC` | `60` | How often to cross-check against vast.ai instance list |
| `STATUS_REPORT_INTERVAL_SEC` | `1800` | How often to post fleet status to Discord (30 min) |

### Instance limits

| Variable | Default | Description |
|---|---|---|
| `MAX_INSTANCES` | `1` | Maximum number of operational worker instances |
| `KEEP_DEBUG_INSTANCE` | `false` | When `true`, failed instances are kept alive for SSH inspection instead of being destroyed (at most 1 debug slot; set `false` in production) |

### SSH

| Variable | Default | Description |
|---|---|---|
| `ORCHESTRATOR_SSH_PRIVATE_KEY` | — | OpenSSH Ed25519 private key injected by ECS from Secrets Manager. When set, the orchestrator uses this stable key instead of generating an ephemeral one at startup. Leave unset for local dev. |

### Logging

| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING`. `DEBUG` uses human-readable console output; all other levels emit JSON. |
