#!/usr/bin/env python3
"""Real-model smoke test: load every loadable local model, generate once
(twice for reasoning-capable models: with and without reasoning), and probe
image/PDF attachment support where the model's capability matrix allows it.

Talks to an already-running `local-ai-doctor serve` instance over HTTP so it
exercises the real request pipeline, not internal APIs. Run one instance
pinned to CPU and one pinned to CUDA and point two copies of this script at
them to cover both devices; the app's single-worker admission means a single
server instance can only run one inference at a time regardless of device.

Usage:
    python scripts/smoke_all_models.py --base-url http://127.0.0.1:18001 --device cpu --label cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

GENERATION_TASKS = {"text_generation", "encoder_decoder_generation"}
EMBEDDING_TASKS = {"embedding", "multimodal_embedding"}
POLL_INTERVAL_S = 1.0
POLL_TIMEOUT_S = 600.0


def _poll_run(client: httpx.Client, run_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        response = client.get(f"/runs/{run_id}")
        response.raise_for_status()
        run = response.json()
        if run.get("status") in {"complete", "cancelled", "failed", "disconnected"}:
            return run
        time.sleep(POLL_INTERVAL_S)
    raise TimeoutError(f"run {run_id} did not finish within {POLL_TIMEOUT_S}s")


def _generate(
    client: httpx.Client,
    *,
    model_id: str,
    chat_id: str,
    prompt: str,
    reasoning: bool | None,
    attachment_ids: list[str],
) -> dict[str, Any]:
    started = time.monotonic()
    body: dict[str, Any] = {
        "chat_id": chat_id,
        "model_id": model_id,
        "prompt": prompt,
        "sampling": {"max_output_tokens": 24, "temperature": 0.0, "top_k": 1, "alternatives": 0},
        "instrumentation": "basic",
        "attachment_ids": attachment_ids,
    }
    if reasoning is not None:
        body["reasoning"] = reasoning
    response = client.post("/runs", json=body)
    if response.status_code >= 400:
        return {"ok": False, "stage": "submit", "status": response.status_code, "body": response.text[:500]}
    run_id = response.json()["runId"]
    try:
        run = _poll_run(client, run_id)
    except TimeoutError as exc:
        return {"ok": False, "stage": "poll", "error": str(exc)}
    elapsed = time.monotonic() - started
    ok = run.get("status") == "complete"
    return {
        "ok": ok,
        "status": run.get("status"),
        "elapsed_s": round(elapsed, 1),
        "output_tokens": len(run.get("tokens") or []),
        "warnings": run.get("warnings", []),
        "error_code": run.get("error_code"),
        "error_message": run.get("error_message"),
    }


def _upload(client: httpx.Client, model_id: str, path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        response = client.post(
            "/attachments",
            data={"model_id": model_id},
            files={"file": (path.name, handle)},
        )
    if response.status_code >= 400:
        return {"ok": False, "status": response.status_code, "body": response.text[:500]}
    return {"ok": True, "attachment": response.json()}


def _embed(client: httpx.Client, model_id: str, inputs: list[dict[str, Any]]) -> dict[str, Any]:
    response = client.post("/embeddings", json={"model_id": model_id, "inputs": inputs})
    if response.status_code >= 400:
        return {"ok": False, "status": response.status_code, "body": response.text[:500]}
    payload = response.json()
    vectors = payload.get("vectors") or []
    return {
        "ok": True,
        "count": len(vectors),
        "dimensions": vectors[0]["dimensions"] if vectors else 0,
    }


def run_model(
    client: httpx.Client,
    model: dict[str, Any],
    device: str,
    image_path: Path | None,
    pdf_path: Path | None,
    probe_pdf: bool,
) -> dict[str, Any]:
    model_id = model["id"]
    name = model["displayName"] if "displayName" in model else model.get("display_name", model_id)
    caps = model["capabilities"]["entries"]
    result: dict[str, Any] = {"model": name, "device": device, "task": model["task"]}

    load_started = time.monotonic()
    load_response = client.post(f"/models/{model_id}/load", json={"device": device}, timeout=600)
    if load_response.status_code >= 400:
        result["load"] = {"ok": False, "status": load_response.status_code, "body": load_response.text[:500]}
        return result
    result["load"] = {"ok": True, "elapsed_s": round(time.monotonic() - load_started, 1)}

    try:
        if model["task"] in GENERATION_TASKS:
            chat = client.post("/chats", json={"title": f"smoke-{name}-{device}"}).json()
            chat_id = chat["id"]
            reasoning_supported = caps.get("reasoning_channel", {}).get("state") == "full"
            plans: list[bool | None] = [False, True] if reasoning_supported else [None]
            result["generation"] = {}
            for reasoning in plans:
                key = "reasoning_on" if reasoning else "reasoning_off" if reasoning is False else "default"
                result["generation"][key] = _generate(
                    client,
                    model_id=model_id,
                    chat_id=chat_id,
                    prompt="Name one moon of Jupiter.",
                    reasoning=reasoning,
                    attachment_ids=[],
                )
            if caps.get("vision", {}).get("state") == "full" and image_path is not None:
                upload = _upload(client, model_id, image_path)
                result["image_upload"] = upload
                if upload["ok"]:
                    result["image_generation"] = _generate(
                        client,
                        model_id=model_id,
                        chat_id=chat_id,
                        prompt="Describe this image in one short sentence.",
                        reasoning=False if reasoning_supported else None,
                        attachment_ids=[upload["attachment"]["id"]],
                    )
        elif model["task"] in EMBEDDING_TASKS:
            result["embedding_text"] = _embed(
                client, model_id, [{"modality": "text", "text": "The quick brown fox jumps."}]
            )
            if caps.get("vision", {}).get("state") == "full" and image_path is not None:
                upload = _upload(client, model_id, image_path)
                result["image_upload"] = upload
                if upload["ok"]:
                    result["embedding_image"] = _embed(
                        client,
                        model_id,
                        [{"modality": "image", "attachment_id": upload["attachment"]["id"]}],
                    )

        if probe_pdf and pdf_path is not None:
            result["pdf_upload"] = _upload(client, model_id, pdf_path)
    finally:
        client.post(f"/models/{model_id}/unload")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--device", required=True, choices=["cpu", "cuda"])
    parser.add_argument("--label", required=True)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--pdf", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url, timeout=60) as client:
        health = client.get("/health")
        health.raise_for_status()
        models_report = client.get("/models").json()
        models = models_report.get("models", [])

        results: list[dict[str, Any]] = []
        pdf_probed = False
        for model in models:
            if not model.get("loadable"):
                results.append(
                    {
                        "model": model.get("displayName") or model.get("display_name"),
                        "device": args.device,
                        "skipped": True,
                        "reason": "not loadable",
                    }
                )
                continue
            probe_pdf = not pdf_probed
            pdf_probed = True
            print(f"[{args.label}] running {model.get('displayName') or model.get('display_name')} ...", flush=True)
            try:
                result = run_model(client, model, args.device, args.image, args.pdf, probe_pdf)
            except Exception as exc:  # noqa: BLE001 - smoke test must keep going
                result = {
                    "model": model.get("displayName") or model.get("display_name"),
                    "device": args.device,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                client.post(f"/models/{model['id']}/unload")
            results.append(result)
            args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"[{args.label}] done {result.get('model')}", flush=True)

    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[{args.label}] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
