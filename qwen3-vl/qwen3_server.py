#!/usr/bin/env python3
"""
Qwen3-VL FastAPI inference server — OpenAI-compatible Chat Completions API.

Endpoints:
  POST /v1/chat/completions   (streaming + non-streaming)
  GET  /v1/models
  GET  /health
  GET  /ping

Auth (mirrors vLLM):
  Set API_KEY env var or --api-key flag.  Clients must send:
    Authorization: Bearer <key>
  or:
    x-api-key: <key>
  If API_KEY is unset / empty, auth is disabled (dev mode, logs a warning).

LoRA adapter:
  --adapter /path/to/lora       Load a LoRA adapter at startup.
  --adapter-name <id>           Model ID to expose for the adapter
                                (default: base-model name).
  --no-merge                    Keep the adapter as a PeftModel instead of
                                merging weights into the base.  Required when
                                using --load-in-4bit or --load-in-8bit.

Usage:
  python qwen3_server.py \\
      --base-model Qwen/Qwen3-VL-8B-Instruct \\
      --adapter /adapters/checkpoint \\
      --adapter-name my-ocr-model \\
      --api-key sk-secret \\
      --port 8001

Docker env vars (translated to flags by entrypoint.sh):
  API_KEY, BASE_MODEL, ADAPTER_PATH, ADAPTER_NAME,
  LOAD_IN_4BIT, LOAD_IN_8BIT, NO_MERGE, PORT, HOST
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from threading import Thread
from typing import AsyncIterator

import glob
import tarfile
import tempfile
import zipfile

import httpx
import torch
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from PIL import Image
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Global model state ─────────────────────────────────────────────────────

_model = None
_processor = None
_model_id: str = "local-caption"        # advertised in /v1/models
_base_model_id: str = ""
_adapter_path: str | None = None
_load_start: float = 0.0
_load_end: float = 0.0

_API_KEY: str | None = None             # set in main()


def _resolve_adapter(adapter: str) -> str:
    """Return a local path for the adapter, downloading it first if needed.

    Accepted forms:
      /local/path                   local directory (pass-through)
      username/repo-name            HuggingFace Hub ID (peft handles natively)
      https://…/adapter.zip         HTTP URL to a .zip archive
      https://…/adapter.tar.gz      HTTP URL to a .tar.gz archive
    """
    if not adapter.startswith(("http://", "https://")):
        return adapter  # local path or HF Hub ID — peft handles both

    log.info("Downloading adapter from %s …", adapter)
    resp = httpx.get(adapter, follow_redirects=True, timeout=300)
    resp.raise_for_status()

    tmp_dir = tempfile.mkdtemp(prefix="adapter_dl_")
    content_type = resp.headers.get("content-type", "")

    if adapter.endswith(".zip") or "zip" in content_type:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            zf.extractall(tmp_dir)
    elif any(adapter.endswith(s) for s in (".tar.gz", ".tgz", ".tar")) or "tar" in content_type:
        with tarfile.open(fileobj=io.BytesIO(resp.content)) as tf:
            tf.extractall(tmp_dir)
    else:
        raise ValueError(
            f"Cannot determine archive format for URL: {adapter}\n"
            "Expected a .zip or .tar.gz URL, or a HuggingFace Hub model ID."
        )

    # The archive may wrap everything in a single subdirectory.
    configs = glob.glob(os.path.join(tmp_dir, "**/adapter_config.json"), recursive=True)
    if not configs:
        raise FileNotFoundError(f"No adapter_config.json found after extracting {adapter}")
    resolved = os.path.dirname(configs[0])
    log.info("Adapter extracted to %s", resolved)
    return resolved


def _load(
    base_model: str,
    adapter_path: str | None,
    adapter_name: str | None,
    device: str,
    offload_dir: str,
    load_in_4bit: bool,
    load_in_8bit: bool,
    no_merge: bool,
) -> None:
    global _model, _processor, _model_id, _base_model_id, _adapter_path
    global _load_start, _load_end

    _load_start = time.time()
    _base_model_id = base_model

    from unsloth import FastVisionModel

    resolved = _resolve_adapter(adapter_path) if adapter_path else None
    _adapter_path = resolved

    # When an adapter is provided, Unsloth reads adapter_config.json to locate
    # the base model automatically; otherwise load the base model directly.
    model_name = resolved or base_model
    # Unsloth supports 4-bit; map 8-bit requests to 4-bit (closest equivalent).
    quantize_4bit = load_in_4bit or load_in_8bit
    log.info("Loading model from %s (4bit=%s) …", model_name, quantize_4bit)

    _model, _processor = FastVisionModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=quantize_4bit,
    )
    FastVisionModel.for_inference(_model)

    _model.eval()
    _load_end = time.time()

    _model_id = adapter_name or base_model
    log.info("Model ready as '%s' (%.1fs load time)", _model_id, _load_end - _load_start)


# ── Auth ───────────────────────────────────────────────────────────────────

_bearer = HTTPBearer(auto_error=False)


def _verify_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    if not _API_KEY:
        return  # auth disabled
    token = None
    if credentials and credentials.scheme.lower() == "bearer":
        token = credentials.credentials
    if token is None:
        token = request.headers.get("x-api-key")
    if token != _API_KEY:
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "Invalid API key", "type": "authentication_error", "code": "invalid_api_key"}},
        )


# ── FastAPI app ────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    a = app.state.args
    _load(
        a.base_model, a.adapter, a.adapter_name, a.device,
        a.offload_dir, a.load_in_4bit, a.load_in_8bit, a.no_merge,
    )
    yield


app = FastAPI(title="Qwen3-VL Inference Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── OpenAI request / response schema ──────────────────────────────────────

class ImageUrl(BaseModel):
    url: str
    detail: str = "auto"


class ContentPart(BaseModel):
    type: str
    text: str | None = None
    image_url: ImageUrl | None = None


class Message(BaseModel):
    role: str
    content: str | list[ContentPart]


class ChatRequest(BaseModel):
    model: str = "local-caption"
    messages: list[Message]
    max_tokens: int = Field(default=512)
    max_completion_tokens: int | None = None
    temperature: float = 0.1
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: str | list[str] | None = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    logprobs: bool | None = None
    user: str | None = None

    @property
    def effective_max_tokens(self) -> int:
        return self.max_completion_tokens or self.max_tokens


def _openai_error(message: str, etype: str = "invalid_request_error", status: int = 400, code: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": etype, "param": None, "code": code}},
    )


def _completion_chunk(cid: str, created: int, model: str, delta: dict, finish_reason: str | None) -> str:
    return f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish_reason}]})}\n\n"


# ── Helpers ────────────────────────────────────────────────────────────────

def _fetch_image(url: str) -> Image.Image:
    if url.startswith("data:"):
        _, encoded = url.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    resp = httpx.get(url, follow_redirects=True, timeout=15)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def _to_qwen_messages(messages: list[Message]) -> tuple[list[dict], list[Image.Image]]:
    qwen_msgs: list[dict] = []
    images: list[Image.Image] = []
    for msg in messages:
        if isinstance(msg.content, str):
            qwen_msgs.append({"role": msg.role, "content": msg.content})
            continue
        parts: list[dict] = []
        for part in msg.content:
            if part.type == "text" and part.text:
                parts.append({"type": "text", "text": part.text})
            elif part.type == "image_url" and part.image_url:
                img = _fetch_image(part.image_url.url)
                images.append(img)
                parts.append({"type": "image", "image": img})
        qwen_msgs.append({"role": msg.role, "content": parts})
    return qwen_msgs, images


def _prepare_inputs(req: ChatRequest) -> tuple[dict, int]:
    qwen_msgs, images = _to_qwen_messages(req.messages)
    text = _processor.apply_chat_template(qwen_msgs, tokenize=False, add_generation_prompt=True)
    inputs = _processor(
        text=[text],
        images=images if images else None,
        return_tensors="pt",
    ).to(_model.device)
    return inputs, inputs["input_ids"].shape[1]


def _generate_kwargs(req: ChatRequest, inputs: dict) -> dict:
    return {
        **inputs,
        "max_new_tokens": req.effective_max_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "do_sample": req.temperature > 0,
    }


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/ping")
def ping():
    """Lightweight liveness probe — always returns 200 immediately."""
    return {"status": "pong"}


@app.get("/health")
def health():
    """Readiness probe — mirrors vLLM: 200 when model is loaded, 503 while loading."""
    ready = _model is not None
    body = {
        "status": "ok" if ready else "loading",
        "model_loaded": ready,
        "model_id": _model_id,
        "base_model": _base_model_id,
        "adapter": _adapter_path,
        "load_time_s": round(_load_end - _load_start, 2) if ready else None,
        "auth_enabled": bool(_API_KEY),
    }
    return JSONResponse(content=body, status_code=200 if ready else 503)


@app.get("/v1/models", dependencies=[Depends(_verify_key)])
def list_models():
    created = int(_load_end) if _load_end else 0
    return {
        "object": "list",
        "data": [{
            "id": _model_id,
            "object": "model",
            "created": created,
            "owned_by": "local",
            "permission": [],
            "root": _base_model_id,
            "parent": None,
        }],
    }


@app.post("/v1/chat/completions", dependencies=[Depends(_verify_key)])
async def chat_completions(req: ChatRequest, request: Request):
    if _model is None:
        return _openai_error("Model not loaded yet", etype="server_error", status=503)

    try:
        inputs, prompt_tokens = _prepare_inputs(req)
    except Exception as exc:
        return _openai_error(f"Failed to process messages: {exc}", status=422)

    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if req.stream:
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            _processor.tokenizer if hasattr(_processor, "tokenizer") else _processor,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        thread = Thread(target=_model.generate, kwargs={**_generate_kwargs(req, inputs), "streamer": streamer}, daemon=True)
        thread.start()

        async def sse() -> AsyncIterator[str]:
            yield _completion_chunk(cid, created, _model_id, {"role": "assistant", "content": ""}, None)
            for token in streamer:
                if await request.is_disconnected():
                    break
                yield _completion_chunk(cid, created, _model_id, {"content": token}, None)
            yield _completion_chunk(cid, created, _model_id, {}, "stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    with torch.no_grad():
        output_ids = _model.generate(**_generate_kwargs(req, inputs))

    completion_tokens = output_ids.shape[1] - prompt_tokens
    generated = _processor.decode(output_ids[0][prompt_tokens:], skip_special_tokens=True)

    return JSONResponse({
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": _model_id,
        "system_fingerprint": None,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": generated},
            "logprobs": None,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    })


# ── Entrypoint ─────────────────────────────────────────────────────────────

def main() -> None:
    global _API_KEY

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", default=os.getenv("BASE_MODEL", "Qwen/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--adapter", default=os.getenv("ADAPTER_PATH"),
                        help="Path to a LoRA adapter directory (adapter_config.json + weights)")
    parser.add_argument("--adapter-name", default=os.getenv("ADAPTER_NAME"),
                        help="Model ID to advertise for the adapter (default: base model name)")
    parser.add_argument("--no-merge", action="store_true", default=os.getenv("NO_MERGE", "").lower() in ("1", "true"),
                        help="Keep adapter as PeftModel instead of merging (required for quantized models)")
    parser.add_argument("--api-key", default=os.getenv("API_KEY"),
                        help="Bearer token for auth.  Unset = auth disabled.")
    parser.add_argument("--device", default=os.getenv("DEVICE", "auto"))
    parser.add_argument("--offload-dir", default="/tmp/model_offload")
    quant = parser.add_mutually_exclusive_group()
    quant.add_argument("--load-in-4bit", action="store_true", default=os.getenv("LOAD_IN_4BIT", "").lower() in ("1", "true"))
    quant.add_argument("--load-in-8bit", action="store_true", default=os.getenv("LOAD_IN_8BIT", "").lower() in ("1", "true"))
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8001")))
    args = parser.parse_args()

    _API_KEY = args.api_key or None
    if not _API_KEY:
        log.warning("API_KEY is not set — authentication is DISABLED")

    app.state.args = args
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
