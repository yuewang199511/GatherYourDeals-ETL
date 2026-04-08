"""
GatherYourDeals ETL Service
============================
HTTP wrapper around the ETL pipeline.  Implements the contract defined in
openapi.yaml:

    POST /etl   { "source": "<image URL, local path, or Google Drive folder URL>" }
                → { "success": true/false, "message": "..." }
                  or, for Drive folder sources:
                → { "success": true/false, "message": "N/M succeeded",
                    "results": [{"file": "...", "success": ..., "message": "..."}] }

Single image: the full pipeline (ADI OCR → LLM structuring → geocode → GYD upload)
runs synchronously and blocks until complete.

Google Drive folder: all image files directly inside the folder are processed in
sequence.  The folder must be shared as "Anyone with the link can view".
Requires GOOGLE_API_KEY in .env with the Drive API enabled.

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000

Environment (.env):
    Same variables as etl.py — AZURE_DI_*, OPENROUTER_API_KEY / CLOD_API_KEY,
    GYD_SERVER_URL, GYD_ACCESS_TOKEN, etc.
    Additional:
        ETL_DEFAULT_USER=lkim       # username written into receipt JSON metadata
        GOOGLE_API_KEY=<key>        # required for Drive folder ingestion
"""

import asyncio
import json
import math
import os
import random
import tempfile
import time
import re
import urllib.parse
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import etl as _etl
from etl_logger import log_pipeline

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ETL Service API",
    description=(
        "Internal ETL service that accepts a remote address and processes it "
        "synchronously. No authentication required."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DEFAULT_USER = os.getenv("ETL_DEFAULT_USER", "unknown")
_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

_ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".tiff", ".tif", ".bmp"}

_DRIVE_IMAGE_MIME_TYPES = {
    "image/jpeg", "image/png", "image/webp",
    "image/heic", "image/tiff", "image/bmp",
}

# Matches: https://drive.google.com/drive/folders/<id>[?...]
_GDRIVE_FOLDER_RE = re.compile(r"drive\.google\.com/drive/folders/([^/?#]+)")

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class EtlRequest(BaseModel):
    source: str
    refresh_token: str | None = None


class EtlResponse(BaseModel):
    success: bool
    message: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_source(source: str, run_id: str) -> tuple[Path, str, bool]:
    """Download a URL to a named temp file, or validate a local path.

    Returns:
        (image_path, display_name, is_temp)
        image_path:   Path object pointing to the image on disk.
        display_name: Original filename for use in logs and output/  directory.
        is_temp:      True if image_path is a temp file the caller must delete.

    Raises ValueError / FileNotFoundError on bad input.
    """
    if source.startswith(("http://", "https://")):
        # Convert Google Drive viewer/sharing URLs to direct download URLs.
        # Handles: https://drive.google.com/file/d/<id>/view[?...]
        #          https://drive.google.com/file/d/<id>
        _gdrive_match = re.search(r"drive\.google\.com/file/d/([^/?#]+)", source)
        if _gdrive_match:
            source = f"https://drive.google.com/uc?export=download&id={_gdrive_match.group(1)}"

        parsed = urllib.parse.urlparse(source)
        url_filename = Path(parsed.path).name or "receipt.jpg"
        ext = Path(url_filename).suffix.lower()
        if ext not in _ALLOWED_EXTS:
            ext = ".jpg"
            url_filename = Path(url_filename).stem + ext

        display_name = url_filename
        tmp_path = Path(tempfile.gettempdir()) / f"{Path(url_filename).stem}_{run_id[:8]}{ext}"

        try:
            import httpx
            with httpx.Client(follow_redirects=True, timeout=60) as client:
                resp = client.get(source)
                resp.raise_for_status()
                tmp_path.write_bytes(resp.content)
            print(f"  [download] {display_name} — {len(resp.content):,} bytes, content-type: {resp.headers.get('content-type', 'unknown')}")
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            raise ValueError(f"Failed to download source: {exc}") from exc

        return tmp_path, display_name, True

    # Local path
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")
    if p.suffix.lower() not in _ALLOWED_EXTS:
        raise ValueError(
            f"Unsupported file type '{p.suffix}'.  "
            f"Accepted: {', '.join(sorted(_ALLOWED_EXTS))}"
        )
    return p, p.name, False


# ---------------------------------------------------------------------------
# Google Drive folder helpers
# ---------------------------------------------------------------------------


def _list_drive_images(folder_id: str) -> list[dict]:
    """Return all image files directly inside a public Google Drive folder.

    Raises RuntimeError if google-api-python-client is not installed or
    GOOGLE_API_KEY is not set.
    """
    try:
        from googleapiclient.discovery import build
    except ImportError:
        raise RuntimeError(
            "google-api-python-client is not installed. "
            "Run: pip install google-api-python-client"
        )
    if not _GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set — required to list Google Drive folders. "
            "Add it to your .env file."
        )

    service = build("drive", "v3", developerKey=_GOOGLE_API_KEY)
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id, name, mimeType)",
        pageSize=1000,
    ).execute()

    files = results.get("files", [])
    images = [f for f in files if f.get("mimeType") in _DRIVE_IMAGE_MIME_TYPES]
    print(f"[drive] folder={folder_id}: {len(images)} image(s) found "
          f"({len(files)} total files)")
    return images


# ---------------------------------------------------------------------------
# Single-image pipeline (shared by single and batch paths)
# ---------------------------------------------------------------------------


async def _process_one(
    source: str,
    display_name: str,
    jwt_token: str | None,
    refresh_token: str | None = None,
) -> dict:
    """Run the full ETL pipeline for one image URL or path.

    Returns a dict with keys: success (bool), message (str).
    Never raises — all exceptions are caught and returned as failure dicts.
    """
    run_id = str(uuid.uuid4())
    provider = _etl.LLM_PROVIDER
    model = _etl.CLOD_DEFAULT_MODEL if provider == "clod" else _etl.OR_DEFAULT_MODEL

    try:
        image_path, resolved_name, is_temp = _resolve_source(source, run_id)
        # Prefer the caller-supplied display_name (e.g. Drive filename) over
        # the name derived from the URL, which may be a generic token like "uc".
        if display_name:
            resolved_name = display_name
    except (ValueError, FileNotFoundError) as exc:
        return {"success": False, "message": str(exc)}

    pipeline_start = time.monotonic()
    try:
        try:
            data = await asyncio.to_thread(
                _etl.extract, image_path, _DEFAULT_USER, model, run_id, provider
            )
        except Exception as exc:
            total_ms = (time.monotonic() - pipeline_start) * 1000
            log_pipeline(run_id, resolved_name, _DEFAULT_USER, provider, model,
                         total_ms, False, str(exc))
            return {"success": False, "message": f"Failed to parse data from source: {exc}"}

        rows = _etl.flatten_receipt(data)
        model_slug = model.split("/")[-1].lower()
        out_dir = _etl.OUTPUT_DIR / f"{provider}-{model_slug}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / (Path(resolved_name).stem + ".json")).write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        data["imageName"] = resolved_name
        try:
            created = _etl.upload(data, run_id, token=jwt_token, refresh_token=refresh_token)
        except Exception as exc:
            total_ms = (time.monotonic() - pipeline_start) * 1000
            log_pipeline(run_id, resolved_name, _DEFAULT_USER, provider, model,
                         total_ms, False, f"upload: {exc}")
            return {"success": False, "message": f"Unexpected error during ETL process: {exc}"}

        image_stem = Path(resolved_name).stem or resolved_name
        registry = {image_stem: [r.id for r in created]}
        _etl._registry_save(image_stem, [r.id for r in created])

        total_ms = (time.monotonic() - pipeline_start) * 1000
        log_pipeline(run_id, resolved_name, _DEFAULT_USER, provider, model, total_ms, True)
        return {"success": True, "message": "ETL completed successfully", "registry": registry}

    finally:
        if is_temp:
            image_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Mock pipeline helper (Experiment 1 Phase B — zero API cost load test)
# ---------------------------------------------------------------------------

# Lognormal params calibrated to 2026-03-31 baseline (OCR P50=5000ms P95=11000ms,
# LLM P50=3718ms P95=10080ms, geocode P50=500ms P95=1200ms)
def _ln_params(p50_ms: float, p95_ms: float) -> tuple[float, float]:
    mu = math.log(p50_ms)
    sigma = (math.log(p95_ms) - mu) / 1.645
    return mu, sigma


_MOCK_OCR_MU,  _MOCK_OCR_SIG  = _ln_params(5_000,  11_000)
_MOCK_LLM_MU,  _MOCK_LLM_SIG  = _ln_params(3_718,  10_080)
_MOCK_GEO_MU,  _MOCK_GEO_SIG  = _ln_params(500,     1_200)


async def _run_mock_pipeline() -> None:
    """Sleep through a simulated OCR → LLM → geocode pipeline. No API calls."""
    ocr_ms  = random.lognormvariate(_MOCK_OCR_MU,  _MOCK_OCR_SIG)
    llm_ms  = random.lognormvariate(_MOCK_LLM_MU,  _MOCK_LLM_SIG)
    geo_ms  = random.lognormvariate(_MOCK_GEO_MU,  _MOCK_GEO_SIG)
    await asyncio.sleep((ocr_ms + llm_ms + geo_ms) / 1_000)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post(
    "/etl",
    response_model=EtlResponse,
    responses={
        400: {"model": EtlResponse, "description": "Bad request — missing or invalid source"},
        422: {"model": EtlResponse, "description": "ETL failed — source reachable but processing failed"},
        500: {"model": EtlResponse, "description": "Internal server error"},
    },
)
async def run_etl(
    body: EtlRequest,
    request: Request,
    mock: bool = Query(False, description="If true, skip real pipeline and sleep through mock latencies (Experiment 1 Phase B)"),
):
    """
    Run the full ETL pipeline for a receipt image or a Google Drive folder.

    - Single image: pass any image URL, Google Drive file URL, or local path.
    - Batch: pass a Google Drive folder URL (shared as "Anyone with the link can view").
      All image files directly inside the folder are processed in sequence.
      Requires GOOGLE_API_KEY in .env.

    Set `mock=true` to skip real API calls and simulate pipeline latency instead
    (zero cost — for Experiment 1 Phase B load testing).
    """
    # Extract JWT from Authorization header — passed by Yimeng's frontend on
    # behalf of the logged-in user.  Falls back to GYD_ACCESS_TOKEN env var
    # (used for CLI runs and local testing).
    auth_header = request.headers.get("Authorization", "")
    jwt_token = auth_header.removeprefix("Bearer ").strip() or None
    refresh_token = body.refresh_token or None

    source = (body.source or "").strip()
    if not source:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "source address must not be empty"},
        )

    # --- Detect Google Drive folder ----------------------------------------
    folder_match = _GDRIVE_FOLDER_RE.search(source)
    if folder_match:
        folder_id = folder_match.group(1)

        # Mock batch: simulate one pipeline per image slot (use 4 as stand-in)
        if mock:
            for _ in range(4):
                await _run_mock_pipeline()
            return JSONResponse(
                status_code=200,
                content={"success": True, "message": "mock batch ETL completed (4 images simulated)", "results": []},
            )

        try:
            images = await asyncio.to_thread(_list_drive_images, folder_id)
        except RuntimeError as exc:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": str(exc)},
            )

        if not images:
            return JSONResponse(
                status_code=200,
                content={"success": True, "message": "No image files found in folder", "results": []},
            )

        print(f"\n[etl] batch — {len(images)} image(s) from folder {folder_id}\n")
        results = []
        for file in images:
            download_url = f"https://drive.google.com/uc?export=download&id={file['id']}"
            result = await _process_one(download_url, file["name"], jwt_token, refresh_token)
            results.append({"file": file["name"], **result})
            status = "✓" if result["success"] else "✗"
            print(f"  {status}  {file['name']} — {result['message']}")

        succeeded = sum(1 for r in results if r["success"])
        total = len(results)
        return JSONResponse(
            status_code=200,
            content={
                "success": succeeded == total,
                "message": f"{succeeded}/{total} succeeded",
                "results": results,
            },
        )

    # --- Single image -------------------------------------------------------

    # --- Mock mode: no real API calls, just sleep through simulated latency --
    if mock:
        await _run_mock_pipeline()
        return JSONResponse(
            status_code=200,
            content={"success": True, "message": "mock ETL completed"},
        )

    result = await _process_one(source, "", jwt_token, refresh_token)
    status_code = 200 if result["success"] else 422
    return JSONResponse(status_code=status_code, content=result)


@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok"}
