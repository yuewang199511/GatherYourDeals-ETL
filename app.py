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
sequence.  Requires GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, and
GOOGLE_REFRESH_TOKEN in .env (OAuth 2.0 — run scripts/google_oauth_setup.py
once to obtain the refresh token).

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000

Environment (.env):
    Same variables as etl.py — AZURE_DI_*, OPENROUTER_API_KEY / CLOD_API_KEY,
    GYD_SERVER_URL, GYD_ACCESS_TOKEN, etc.
    Additional:
        ETL_DEFAULT_USER=lkim           # username written into receipt JSON metadata
        GOOGLE_CLIENT_ID=<id>           # OAuth 2.0 client ID  (Drive folder ingestion)
        GOOGLE_CLIENT_SECRET=<secret>   # OAuth 2.0 client secret
        GOOGLE_REFRESH_TOKEN=<token>    # long-lived refresh token
"""

import asyncio
import json
import math
import os
import random
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

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleAuthRequest
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

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
_GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
_GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
_GOOGLE_REFRESH_TOKEN = os.getenv("GOOGLE_REFRESH_TOKEN", "")

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


def _resolve_source(source: str) -> tuple[bytes, str]:
    """Download a URL or read a local path into memory.

    Returns:
        (image_bytes, display_name)
        image_bytes:  Raw image bytes.
        display_name: Original filename for use in logs and output/ directory.

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
        try:
            import httpx
            with httpx.Client(follow_redirects=True, timeout=60) as client:
                resp = client.get(source)
                resp.raise_for_status()
            print(f"  [download] {display_name} — {len(resp.content):,} bytes, content-type: {resp.headers.get('content-type', 'unknown')}")
            return resp.content, display_name
        except Exception as exc:
            raise ValueError(f"Failed to download source: {exc}") from exc

    # Local path — read into memory
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"Source path does not exist: {source}")
    if p.suffix.lower() not in _ALLOWED_EXTS:
        raise ValueError(
            f"Unsupported file type '{p.suffix}'.  "
            f"Accepted: {', '.join(sorted(_ALLOWED_EXTS))}"
        )
    return p.read_bytes(), p.name


# ---------------------------------------------------------------------------
# Google Drive folder helpers
# ---------------------------------------------------------------------------


def _build_drive_service():
    """Build an authenticated Drive v3 service from stored OAuth credentials."""
    if not (_GOOGLE_CLIENT_ID and _GOOGLE_CLIENT_SECRET and _GOOGLE_REFRESH_TOKEN):
        raise RuntimeError(
            "Google Drive OAuth not configured. "
            "Set GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, and GOOGLE_REFRESH_TOKEN in .env "
            "(run scripts/google_oauth_setup.py once to obtain the refresh token)."
        )
    creds = Credentials(
        token=None,
        refresh_token=_GOOGLE_REFRESH_TOKEN,
        client_id=_GOOGLE_CLIENT_ID,
        client_secret=_GOOGLE_CLIENT_SECRET,
        token_uri="https://oauth2.googleapis.com/token",
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    creds.refresh(GoogleAuthRequest())
    return build("drive", "v3", credentials=creds)


def _list_drive_images(service, folder_id: str) -> list[dict]:
    """Return all image files directly inside a Google Drive folder."""
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id, name, mimeType)",
        pageSize=1000,
    ).execute()

    files = results.get("files", [])
    images = [f for f in files if f.get("mimeType") in _DRIVE_IMAGE_MIME_TYPES]

    print(
        f"[drive/oauth] folder={folder_id}: {len(images)} image(s) found "
        f"({len(files)} total files)"
    )
    return images


def _download_folder_gdown(folder_url: str) -> list[tuple[bytes, str]]:
    """Download all image files from a public Drive folder using gdown.

    No API key or OAuth required — folder must be shared as
    'Anyone with the link can view'.

    Returns a list of (image_bytes, filename) tuples for each image found.
    """
    import tempfile
    import shutil

    try:
        import gdown
    except ImportError:
        raise RuntimeError(
            "gdown not installed. Run: pip install gdown"
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="gyd_drive_"))
    try:
        paths = gdown.download_folder(
            url=folder_url,
            output=str(tmp_dir),
            quiet=False,
            use_cookies=False,
        )
        if not paths:
            return []

        results = []
        for p in sorted(paths):
            p = Path(p)
            if p.suffix.lower() in _ALLOWED_EXTS:
                results.append((p.read_bytes(), p.name))
                print(f"  [gdown] {p.name} — {p.stat().st_size:,} bytes")

        return results
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _download_drive_file(service, file_id: str, file_name: str) -> bytes:
    """Download a Drive file via the SDK directly into memory."""
    import io

    request = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    data = buf.getvalue()
    print(f"  [drive/sdk] {file_name} — {len(data):,} bytes")
    return data


# ---------------------------------------------------------------------------
# Single-image pipeline (shared by single and batch paths)
# ---------------------------------------------------------------------------


async def _process_one(
    image_bytes: bytes,
    display_name: str,
    jwt_token: str | None,
    refresh_token: str | None = None,
) -> dict:
    """Run the full ETL pipeline for one image already loaded into memory.

    Returns a dict with keys: success (bool), message (str).
    Never raises — all exceptions are caught and returned as failure dicts.
    """
    run_id = str(uuid.uuid4())
    provider = _etl.LLM_PROVIDER
    model = _etl.CLOD_DEFAULT_MODEL if provider == "clod" else _etl.OR_DEFAULT_MODEL

    pipeline_start = time.monotonic()
    try:
        try:
            data = await asyncio.to_thread(
                _etl.extract, image_bytes, display_name, _DEFAULT_USER, model, run_id, provider
            )
        except Exception as exc:
            total_ms = (time.monotonic() - pipeline_start) * 1000
            log_pipeline(run_id, display_name, _DEFAULT_USER, provider, model,
                         total_ms, False, str(exc))
            return {"success": False, "message": f"Failed to parse data from source: {exc}"}

        rows = _etl.flatten_receipt(data)
        model_slug = model.split("/")[-1].lower()
        out_dir = _etl.OUTPUT_DIR / f"{provider}-{model_slug}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / (Path(display_name).stem + ".json")).write_text(
            json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        data["imageName"] = display_name
        try:
            created = _etl.upload(data, run_id, token=jwt_token, refresh_token=refresh_token)
        except Exception as exc:
            total_ms = (time.monotonic() - pipeline_start) * 1000
            log_pipeline(run_id, display_name, _DEFAULT_USER, provider, model,
                         total_ms, False, f"upload: {exc}")
            return {"success": False, "message": f"Unexpected error during ETL process: {exc}"}

        image_stem = Path(display_name).stem or display_name
        registry = {image_stem: [r.id for r in created]}
        _etl._registry_save(image_stem, [r.id for r in created])

        total_ms = (time.monotonic() - pipeline_start) * 1000
        log_pipeline(run_id, display_name, _DEFAULT_USER, provider, model, total_ms, True)
        return {"success": True, "message": "ETL completed successfully", "registry": registry}

    except Exception as exc:
        total_ms = (time.monotonic() - pipeline_start) * 1000
        log_pipeline(run_id, display_name, _DEFAULT_USER, provider, model, total_ms, False, str(exc))
        return {"success": False, "message": f"Unexpected error during ETL process: {exc}"}


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
    - Batch: pass a Google Drive folder URL. Files are downloaded via gdown first
      (public folders, no auth required). If gdown retrieves no files, the service
      falls back to the OAuth SDK (requires GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET,
      and GOOGLE_REFRESH_TOKEN in .env — works for private folders).

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

        # Try gdown first (public folders, no auth required).
        # Fall back to OAuth SDK if gdown finds no files (e.g. private folder)
        # and credentials are configured.
        oauth_configured = bool(_GOOGLE_CLIENT_ID and _GOOGLE_CLIENT_SECRET and _GOOGLE_REFRESH_TOKEN)

        file_pairs: list[tuple] = []
        gdown_error: str | None = None

        print(f"\n[etl] batch/gdown — attempting public download for folder {folder_id}\n")
        try:
            file_pairs = await asyncio.to_thread(_download_folder_gdown, source)
        except Exception as exc:
            gdown_error = str(exc)
            print(f"  [gdown] failed: {gdown_error}")

        if not file_pairs:
            if oauth_configured:
                print(f"  [gdown] no files retrieved — falling back to OAuth SDK\n")
                try:
                    service = await asyncio.to_thread(_build_drive_service)
                    images = await asyncio.to_thread(_list_drive_images, service, folder_id)
                except RuntimeError as exc:
                    return JSONResponse(
                        status_code=400,
                        content={"success": False, "message": str(exc)},
                    )

                print(f"  [oauth] {len(images)} image(s) found\n")
                for file in images:
                    try:
                        image_bytes = await asyncio.to_thread(
                            _download_drive_file, service, file["id"], file["name"]
                        )
                        file_pairs.append((image_bytes, file["name"]))
                    except Exception as exc:
                        file_pairs.append((None, file["name"], str(exc)))
            else:
                msg = (
                    f"gdown could not retrieve files ({gdown_error}). "
                    "If the folder is private, set GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, "
                    "and GOOGLE_REFRESH_TOKEN in .env to enable OAuth fallback."
                ) if gdown_error else "No image files found in folder."
                return JSONResponse(
                    status_code=400 if gdown_error else 200,
                    content={"success": not gdown_error, "message": msg, "results": []},
                )

        results = []
        for entry in file_pairs:
            if len(entry) == 3:  # download error from OAuth path
                _, file_name, err_msg = entry
                results.append({"file": file_name, "success": False, "message": f"Download failed: {err_msg}"})
                print(f"  ✗  {file_name} — download failed: {err_msg}")
                continue

            image_bytes, file_name = entry
            result = await _process_one(image_bytes, file_name, jwt_token, refresh_token)
            results.append({"file": file_name, **result})
            status = "✓" if result["success"] else "✗"
            print(f"  {status}  {file_name} — {result['message']}")

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

    try:
        image_bytes, display_name = await asyncio.to_thread(_resolve_source, source)
    except (ValueError, FileNotFoundError) as exc:
        return JSONResponse(status_code=400, content={"success": False, "message": str(exc)})

    result = await _process_one(image_bytes, display_name, jwt_token, refresh_token)
    status_code = 200 if result["success"] else 422
    return JSONResponse(status_code=status_code, content=result)


@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok"}
