# qwen3-vl — Vast.ai vLLM worker

Serves **Qwen3-VL-8B-Instruct** (base) + a fine-tuned **Unsloth/PEFT LoRA adapter**
via an OpenAI-compatible API (port 8001). Designed to integrate with the
[eased](..) orchestrator on Vast.ai.

```
Client → POST /v1/chat/completions  (model: "caption")
           ↓
    vLLM (Qwen3-VL-8B-Instruct + LoRA)
           ↓
    /adapters/checkpoint   ← your fine-tuned adapter
```

---

## Files

| File | Purpose |
|---|---|
| `Dockerfile` | Generic image (local dev / any GPU cloud) |
| `Dockerfile.vastai` | Vast.ai image (vastai/vllm base, supervisor) |
| `entrypoint.sh` | Startup script for the generic image |
| `vllm.sh` | Supervisor wrapper for the Vast.ai image |
| `vllm.conf` | Supervisor program definition |
| `onstart.sh` | Vast.ai on-start hook |
| `docker-compose.yml` | Local dev compose |
| `.env.example` | Environment variable template |

---

## Adapter layout

The container expects your LoRA checkpoint mounted at `/adapters/checkpoint`:

```
/adapters/checkpoint/
├── adapter_config.json
└── adapter_model.safetensors   # (or adapter_model.bin)
```

---

## Local dev (docker-compose)

```bash
cd qwen3-vl
cp .env.example .env
# Fill in API_KEY, HF_TOKEN, ADAPTER_PATH

docker compose up --build
```

The base model downloads from HuggingFace on first start (≈ 16 GB). Subsequent
starts reuse the `hf-cache` volume.

Test the API:
```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "caption",
    "messages": [
      {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
        {"type": "text", "text": "Describe this image."}
      ]}
    ]
  }'
```

---

## Build & push to GHCR

```bash
# Authenticate
echo $GITHUB_PAT | docker login ghcr.io -u <github-username> --password-stdin

# Generic image (for local / RunPod / any OCI runtime)
docker build \
  -f Dockerfile \
  -t ghcr.io/<org>/qwen3-vl:latest \
  .

# Vast.ai image (includes supervisor integration)
docker build \
  -f Dockerfile.vastai \
  -t ghcr.io/<org>/qwen3-vl:vastai-latest \
  .

docker push ghcr.io/<org>/qwen3-vl:latest
docker push ghcr.io/<org>/qwen3-vl:vastai-latest
```

> The base model is **not** baked into the image (it changes less often than the adapter
> and is ≈ 16 GB). It downloads from HuggingFace on first container boot.
> Use `HF_HOME` on a persistent Vast.ai volume to avoid re-downloading across restarts.

---

## Deploy on Vast.ai manually (without eased)

1. **Upload your adapter** to a Vast.ai storage volume or an S3/HF repo.

2. **Rent an instance** — A100 40 GB minimum; filter for CUDA ≥ 12.1.

3. **Configure the instance** in the Vast.ai UI:
   - Docker image: `ghcr.io/<org>/qwen3-vl:latest`
   - Environment variables:
     ```
     API_KEY=<secret>
     HF_TOKEN=<hf-token>
     BASE_MODEL=unsloth/Qwen3-VL-8B-Instruct
     LORA_NAME=caption
     ```
   - Port mapping: `8001 → 8001`
   - If using a Vast.ai storage volume, mount it so the adapter is available at
     `/adapters/checkpoint` inside the container.

4. **Without a volume mount** — ssh into the instance and copy the adapter:
   ```bash
   ssh -p <port> root@<host>
   mkdir -p /adapters/checkpoint
   # scp or rclone your adapter_config.json + adapter_model.safetensors here
   ```
   Then restart the container or re-run `entrypoint.sh`.

5. The vLLM OpenAI-compatible API will be available at:
   ```
   http://<vast-host>:<mapped-port>/v1/chat/completions
   ```

---

## Deploy via eased orchestrator

Point eased at the Vast.ai image and inject the required env vars.

In `eased/.env`:
```bash
WORKER_IMAGE=ghcr.io/<org>/qwen3-vl:vastai-latest
GHCR_USERNAME=<github-org>
GHCR_PAT=<github-pat-with-read:packages>

# Injected into every Vast.ai instance via vast.ai env:
VLLM_API_KEY=<per-instance-secret>    # eased generates this automatically
HF_TOKEN=<hf-token>
BASE_MODEL=unsloth/Qwen3-VL-8B-Instruct
LORA_NAME=caption
```

> **Adapter on Vast.ai:** The adapter must exist at `/adapters/checkpoint` when vLLM
> starts. The recommended approach is to:
> 1. Pre-populate a Vast.ai storage volume with the adapter files, or
> 2. Use `EXTRA_COMMANDS` in the eased orchestrator to `rsync`/`aws s3 sync` the
>    adapter into the container before `vllm.sh` runs.

### eased env variable mapping

| eased variable | maps to container variable |
|---|---|
| `VLLM_API_KEY` | `API_KEY` |
| `MODEL_ID` | `BASE_MODEL` |
| `VLLM_PORT` | `PORT` |
| `VLLM_MAX_MODEL_LEN` | `MAX_MODEL_LEN` |
| `VLLM_GPU_MEMORY_UTILIZATION` | `GPU_MEMORY_UTILIZATION` |
| `TENSOR_PARALLEL_SIZE` | `TENSOR_PARALLEL_SIZE` |

---

## Environment variables reference

| Variable | Default | Description |
|---|---|---|
| `API_KEY` | — | Bearer token for the vLLM API. **Required.** |
| `HF_TOKEN` | — | HuggingFace token (needed for `unsloth/` gated models) |
| `BASE_MODEL` | `unsloth/Qwen3-VL-8B-Instruct` | HuggingFace model ID |
| `LORA_NAME` | `caption` | Adapter name served in the API (`model` field in requests) |
| `LORA_PATH` | `/adapters/checkpoint` | Path to the adapter directory inside the container |
| `MAX_LORA_RANK` | `64` | Max LoRA rank (must be ≥ the adapter's rank) |
| `PORT` | `8001` | vLLM listen port |
| `MAX_MODEL_LEN` | `8192` | Max context length in tokens |
| `GPU_MEMORY_UTILIZATION` | `0.90` | Fraction of VRAM reserved for vLLM |
| `TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs for tensor parallelism |
| `LIMIT_MM_PER_PROMPT` | `image=5,video=2` | Per-prompt multimodal input limits |
| `HF_HOME` | `/models/hf` | HuggingFace weight cache directory |
| `VLLM_CACHE_ROOT` | `/models/vllm-cache` | Triton kernel / torch.compile cache |

---

## LiteLLM integration

```python
import litellm

response = litellm.completion(
    model="openai/caption",
    api_base="http://<host>:8001/v1",
    api_key="<API_KEY>",
    messages=[{"role": "user", "content": "Describe this image."}],
)
```
