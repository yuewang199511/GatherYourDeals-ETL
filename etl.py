#!/usr/bin/env python3
"""
GatherYourDeals ETL
====================
Three-step pipeline per receipt, orchestrated as a Railtracks Flow:
  Step 1 — Azure Document Intelligence (prebuilt-read)   [ocr_node]
            Sends the receipt image to ADI, returns raw OCR text.
  Step 2 — LLM structuring  (OpenRouter  OR  CLOD)       [structure_node]
            Structures OCR text into the GYD JSON format.
  Step 3 — Azure Maps Geocoding                          [geocode_node]
            Resolves store address → latitude / longitude.

Railtracks broadcasts granular events at each step so you can inspect
runs in the local visualizer: run `railtracks viz` after processing.

Usage
-----
  # Single receipt — OpenRouter (default)
  python etl.py Receipts/2025-10-01Vons.jpg --user lkim016 --no-upload

  # Single receipt — CLOD
  python etl.py Receipts/2025-10-01Vons.jpg --user lkim016 --provider clod --no-upload

  # Whole directory
  python etl.py Receipts/ --user lkim016 --no-upload

  # With SDK upload
  python etl.py Receipts/2025-10-01Vons.jpg --user lkim016

  # Generate usage report from JSONL logs
  python etl.py --report

  # Generate per-model comparison table (Test 2)
  python etl.py --compare

  # Eval output/ against ground_truth/
  python etl.py --eval

  # View Railtracks run visualizer
  railtracks viz

Requirements
------------
  pip install openai python-dotenv azure-ai-documentintelligence
  pip install "railtracks[cli]"
  pip install git+https://github.com/yuewang199511/GatherYourDeals-SDK.git
  pip install matplotlib              # optional — charts in --report
  pip install pillow pillow-heif      # optional — only for HEIC (iPhone) photos

Environment (.env)
------------------
  # Step 1 — Azure Document Intelligence
  AZURE_DI_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/
  AZURE_DI_KEY=<your-key>

  # Step 2 — LLM provider: "openrouter" (default) or "clod"
  OPENROUTER_API_KEY=sk-or-v1-...
  LLM_PROVIDER=openrouter

  # Step 3 — Azure Maps geocoding (optional)
  AZURE_MAPS_KEY=<your-key>

  # GYD data server (leave blank to run extract-only)
  GYD_SERVER_URL=http://localhost:8080/api/v1
  GYD_USERNAME=
  GYD_PASSWORD=
"""

import argparse
import asyncio
import json
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

from reporting import report, compare, eval_receipts, baseline_report  # noqa: E402 (after load_dotenv)

# ---------------------------------------------------------------------------
# Railtracks — flow orchestration + observability
# ---------------------------------------------------------------------------
try:
    import railtracks as rt
    _RT_AVAILABLE = True
except ImportError:
    _RT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
AZURE_DI_ENDPOINT   = os.getenv("AZURE_DI_ENDPOINT", "")
AZURE_DI_KEY        = os.getenv("AZURE_DI_KEY", "")
AZURE_MAPS_KEY      = os.getenv("AZURE_MAPS_KEY", "")

OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
CLOD_API_KEY        = os.getenv("CLOD_API_KEY", "")

# Which LLM backend to use: "openrouter" (default) or "clod"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").lower()

_DEFAULT_MODELS = {
    "openrouter": "anthropic/claude-haiku-4.5",
    "clod":       "Qwen/Qwen2.5-7B-Instruct-Turbo",
}
DEFAULT_MODEL = os.getenv("OR_DEFAULT_MODEL", _DEFAULT_MODELS.get(LLM_PROVIDER, "google/gemini-2.0-flash-exp:free"))

GYD_SERVER_URL = os.getenv("GYD_SERVER_URL", "http://localhost:8080/api/v1")
GYD_USERNAME   = os.getenv("GYD_USERNAME", "")
GYD_PASSWORD   = os.getenv("GYD_PASSWORD", "")

OUTPUT_DIR       = Path("output")
LOGS_DIR         = Path("logs")
REPORTS_DIR      = Path("reports")
GROUND_TRUTH_DIR = Path("ground_truth")


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".tiff", ".tif", ".bmp"}


# ADI cost per page — always log at S0 rate ($0.0015/page) for accurate production cost tracking.
_ADI_COST_PER_PAGE = 0.0015

# Enable Railtracks logging (writes to logs/rt.log + stdout at INFO level)
if _RT_AVAILABLE:
    LOGS_DIR.mkdir(exist_ok=True)
    rt.enable_logging(level="INFO", log_file=str(LOGS_DIR / "rt.log"))

# ---------------------------------------------------------------------------
# LLM prompt — receives OCR text, returns structured JSON
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are a receipt data structuring assistant.
You are given raw OCR text that was extracted from a receipt image by \
Azure Document Intelligence. Your job is to parse it and return ONLY \
valid JSON — no markdown fences, no explanation.

Required top-level fields (use null when not visible in the OCR text):
  imageName     string   (caller injects — output null)
  userName      string   (caller injects — output null)
  storeName     string
  storeAddress  string
  latitude      null     (injected by caller — always output null)
  longitude     null     (injected by caller — always output null)
  purchaseDate  string   format YYYY.MM.DD
  purchaseTime  string   format HH:MM  (24-hour)
  currency      string   "USD", "CAD", etc.
  items         array    (see below)
  totalItems    integer
  subtotal      string   "X.XXUSD" — match currency of receipt
  tax           string   same format, or null
  total         string   same format, or null
  paymentMethod string   or null

Each element of items must have:
  productName   string
  itemCode      string or null   — UPC / item code printed next to the product
  price         string   "X.XXUSD" — the FINAL charged price (after discounts), always include currency code
  amount        string   — quantity or weight as printed on the receipt (e.g. "1", "2", "0.5 lb")
                           DO NOT put a price value here — amount is a count or weight, not a dollar amount
  category      string or null   — department label printed on the receipt (e.g. "Grocery", "Produce")

IMPORTANT rules:
- Include only grocery/food product line items. Skip deposits, recycling fees, donations, bag fees, and miscellaneous non-product charges.
- storeName must be the store's brand/chain name as printed on the receipt header (e.g. "Costco Wholesale", "Your Independent Grocer", "T&T Supermarket"). Do not append address, city, or branch details.
- purchaseDate must be the date the transaction occurred, formatted YYYY.MM.DD. If the year looks implausible (before 2020), you are likely reading a receipt number, time, or barcode — re-examine the receipt for the correct date.
- price is the total amount charged for that line item as shown in the right-hand price column of the receipt. Do not use per-unit rates, per-oz prices, or any divided amount. Always include the currency code: "4.79USD" not "4.79".
- amount is the number of units or weight, never a price.

Also include every other field readable from the OCR text as extra
top-level fields — cashier, member/rewards number, savings, transaction ID,
store phone, operator number, etc. Preserve all receipt detail."""

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _log(entry: dict):
    LOGS_DIR.mkdir(exist_ok=True)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with open(LOGS_DIR / f"etl_{date}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def log_adi(trace_id, image_name, user_id, image_size_bytes,
            pages, cost_usd, latency_ms, success, chars_extracted=None, error=None):
    """Log one Azure Document Intelligence OCR call."""
    _log({
        "time":              datetime.now(timezone.utc).isoformat(),
        "level":             "INFO" if success else "ERROR",
        "service":           "etl",
        "event":             "adi_ocr",
        "trace_id":          trace_id,
        "user_id":           user_id,
        "image_name":        image_name,
        "image_size_bytes":  image_size_bytes,
        "ocr_provider":      "azure-document-intelligence",
        "ocr_latency_ms":    round(latency_ms, 1),
        "ocr_success":       success,
        "pages":             pages,
        "chars_extracted":   chars_extracted,  # OCR equivalent of token count
        "cost_usd":          round(cost_usd, 6),
        "error":             error,
    })

def log_llm(trace_id, image_name, user_id, provider, model,
            input_tokens, output_tokens, cost_usd,
            latency_ms, items_extracted, success, error=None, cost_source="unknown"):
    """Log one LLM structuring call (OpenRouter or Claude)."""
    _log({
        "time":              datetime.now(timezone.utc).isoformat(),
        "level":             "INFO" if success else "ERROR",
        "service":           "etl",
        "event":             "llm_extraction",
        "trace_id":          trace_id,
        "user_id":           user_id,
        "image_name":        image_name,
        "llm_provider":      provider,
        "llm_model":         model,
        "llm_latency_ms":    round(latency_ms, 1),
        "llm_input_tokens":  input_tokens,
        "llm_output_tokens": output_tokens,
        "llm_cost_usd":      round(cost_usd, 8),
        "llm_cost_source":   cost_source,
        "llm_success":       success,
        "items_extracted":   items_extracted,
        "error":             error,
    })

def log_pipeline(trace_id, image_name, user_id, provider, model,
                 total_latency_ms, success, error=None):
    """Log end-to-end pipeline latency for one receipt (OCR + LLM + geocode)."""
    _log({
        "time":              datetime.now(timezone.utc).isoformat(),
        "level":             "INFO" if success else "ERROR",
        "service":           "etl",
        "event":             "pipeline_complete",
        "trace_id":          trace_id,
        "user_id":           user_id,
        "image_name":        image_name,
        "llm_provider":      provider,
        "llm_model":         model,
        "total_latency_ms":  round(total_latency_ms, 1),
        "success":           success,
        "error":             error,
    })

def log_upload(trace_id, image_name, user_id,
               attempted, uploaded, failed, latency_ms, success, error=None):
    """Log one GYD SDK upload batch."""
    _log({
        "time":            datetime.now(timezone.utc).isoformat(),
        "level":           "INFO" if success else "WARN",
        "service":         "etl",
        "event":           "mcp_upload",
        "trace_id":        trace_id,
        "user_id":         user_id,
        "image_name":      image_name,
        "endpoint":        "POST /api/v1/receipts",
        "status":          201 if success else None,
        "latency_ms":      round(latency_ms, 1),
        "items_attempted": attempted,
        "items_uploaded":  uploaded,
        "items_failed":    failed,
        "error":           error,
    })

# ---------------------------------------------------------------------------
# Step 1 — Azure Document Intelligence OCR
# ---------------------------------------------------------------------------
_ADI_MAX_BYTES = 4 * 1024 * 1024  # Azure DI hard limit: 4 MB


def _to_jpeg_bytes(image_path: Path) -> tuple[bytes, str]:
    """Return (image_bytes, content_type). Converts HEIC → JPEG and downscales images >4 MB."""
    import io
    ext = image_path.suffix.lower()

    if ext == ".heic":
        try:
            import pillow_heif
            from PIL import Image
            pillow_heif.register_heif_opener()
            img = Image.open(image_path).convert("RGB")
        except ImportError:
            raise ImportError("HEIC support requires: pip install pillow pillow-heif")
    else:
        raw = image_path.read_bytes()
        if len(raw) <= _ADI_MAX_BYTES:
            content_type = {
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png",  ".webp": "image/webp",
                ".tiff": "image/tiff", ".tif": "image/tiff",
                ".bmp": "image/bmp",
            }.get(ext, "image/jpeg")
            return raw, content_type
        # Image is over 4 MB — need to downscale
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                f"{image_path.name} is {len(raw) // 1024 // 1024} MB (Azure DI limit: 4 MB). "
                "Install Pillow to auto-resize: pip install pillow"
            )
        img = Image.open(io.BytesIO(raw)).convert("RGB")

    # Downscale until the JPEG fits under 4 MB (quality 85 first, then halve dimensions)
    quality = 85
    scale = 1.0
    while True:
        w, h = int(img.width * scale), int(img.height * scale)
        resized = img.resize((w, h), Image.LANCZOS) if scale < 1.0 else img
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=quality)
        data = buf.getvalue()
        if len(data) <= _ADI_MAX_BYTES:
            return data, "image/jpeg"
        if quality > 60:
            quality -= 10
        else:
            scale *= 0.75


def ocr(image_path: Path, run_id: str, user_id: str = "") -> str:
    """
    Send image to Azure Document Intelligence (prebuilt-read).
    Returns the full OCR text extracted from the receipt.
    """
    if not AZURE_DI_ENDPOINT or not AZURE_DI_KEY:
        raise EnvironmentError(
            "AZURE_DI_ENDPOINT and AZURE_DI_KEY must be set in .env\n"
            "Create a Document Intelligence resource in the Azure portal."
        )
    try:
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential
    except ImportError:
        raise ImportError(
            "Azure SDK not installed.\n"
            "pip install azure-ai-documentintelligence"
        )

    client = DocumentIntelligenceClient(
        endpoint=AZURE_DI_ENDPOINT,
        credential=AzureKeyCredential(AZURE_DI_KEY),
    )

    image_bytes, content_type = _to_jpeg_bytes(image_path)
    image_size_bytes = len(image_bytes)
    start = time.monotonic()
    try:
        poller = client.begin_analyze_document(
            "prebuilt-read",
            body=image_bytes,
            content_type=content_type,
        )
        result = poller.result(timeout=120)
        latency_ms = (time.monotonic() - start) * 1000

        pages = len(result.pages) if result.pages else 1
        ocr_text = result.content or ""

        log_adi(run_id, image_path.name, user_id, image_size_bytes,
                pages, pages * _ADI_COST_PER_PAGE, latency_ms, True,
                chars_extracted=len(ocr_text))
        return ocr_text

    except Exception as e:
        latency_ms = (time.monotonic() - start) * 1000
        log_adi(run_id, image_path.name, user_id, image_size_bytes,
                0, 0.0, latency_ms, False, error=str(e))
        raise


# ---------------------------------------------------------------------------
# Step 2 — LLM structuring
# ---------------------------------------------------------------------------

def _parse_llm_json(raw: str) -> dict:
    """Strip markdown fences if present and parse JSON."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
    raw = re.sub(r"\n?```\s*$",          "", raw)
    return json.loads(raw)


def _structure_openrouter(ocr_text: str, model: str) -> tuple[dict, int, int, str]:
    """Call OpenRouter and return (parsed_dict, prompt_tokens, completion_tokens, generation_id)."""
    if not OPENROUTER_API_KEY:
        raise EnvironmentError("OPENROUTER_API_KEY not set — add it to .env")

    from openai import OpenAI
    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": f"Receipt OCR text:\n\n{ocr_text}"},
        ],
        temperature=0,
        max_tokens=4096,
    )
    usage = resp.usage
    pt = usage.prompt_tokens     if usage else 0
    ct = usage.completion_tokens if usage else 0
    return _parse_llm_json(resp.choices[0].message.content), pt, ct, resp.id or ""


def _fetch_openrouter_cost(generation_id: str) -> float | None:
    """
    Fetch actual billed cost from OpenRouter's generation endpoint.
    Returns total_cost in USD, or None if unavailable.
    """
    if not generation_id or not OPENROUTER_API_KEY:
        return None
    import urllib.request
    url = f"https://openrouter.ai/api/v1/generation?id={generation_id}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"})
    # Small delay — OpenRouter may lag a few hundred ms before cost data is ready
    time.sleep(0.5)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read())
            cost = data.get("data", {}).get("total_cost")
            if cost is not None:
                return float(cost)
        except Exception:
            pass
        if attempt < 2:
            time.sleep(1)
    return None


# Token-based cost estimates ($/M tokens) for known OpenRouter models.
# Used as fallback when the generation API doesn't return a cost.
# https://openrouter.ai/anthropic/claude-haiku-4.5
_OR_PRICING: dict[str, tuple[float, float]] = {
    "anthropic/claude-haiku-4.5": (1.00, 5.00),  # $/M input, $/M output
}

def _estimate_openrouter_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost from token counts using known OpenRouter pricing."""
    price_in, price_out = _OR_PRICING.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


# Token-based cost estimates ($/M tokens) for known CLOD models.
# Used as fallback when the CLOD API response does not include a cost field.
# Source: clod.io pricing page (2026-03-29)
# Output price is dynamically priced based on real-time energy costs;
# $0.12/M is the observed running rate, $0.318/M is the price cap.
_CLOD_PRICING: dict[str, tuple[float, float]] = {
    "Qwen2.5-7B-Instruct-Turbo":      (0.30, 0.12),  # $/M input, $/M output (60% discount applied)
    "Qwen/Qwen2.5-7B-Instruct-Turbo": (0.30, 0.12),  # alias with namespace prefix
}

def _estimate_clod_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost from token counts using known CLOD pricing."""
    price_in, price_out = _CLOD_PRICING.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def _structure_clod(ocr_text: str, model: str) -> tuple[dict, int, int, float | None]:
    """Call the CLOD API via httpx and return (parsed_dict, input_tokens, output_tokens, cost_usd_or_None)."""
    if not CLOD_API_KEY:
        raise EnvironmentError("CLOD_API_KEY not set — add it to .env")
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx not installed.\npip install httpx")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": f"Receipt OCR text:\n\n{ocr_text}"},
        ],
    }
    resp = httpx.post(
        "https://api.clod.io/v1/chat/completions",
        headers={"Authorization": f"Bearer {CLOD_API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices")
    if not choices:
        raise ValueError(f"CLOD returned no choices (model={model}): {data.get('error') or data}")
    content = choices[0]["message"]["content"]
    if not content:
        raise ValueError(f"CLOD returned empty content (model={model}).")
    usage = data.get("usage", {})
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    # CLOD may include actual cost in the usage object; fall back to None if absent
    api_cost = usage.get("total_cost") or usage.get("cost") or data.get("total_cost")
    return _parse_llm_json(content), pt, ct, float(api_cost) if api_cost is not None else None


def _load_gt_store_names() -> list[str]:
    """Load canonical store names from ground_truth/ JSON files."""
    names = []
    if not GROUND_TRUTH_DIR.exists():
        return names
    for f in sorted(GROUND_TRUTH_DIR.glob("*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            entry = d[0] if isinstance(d, list) else d
            name = entry.get("storeName")
            if name and name not in names:
                names.append(name)
        except Exception:
            pass
    return names


def normalize_store_name(raw: str, model: str, provider: str) -> str:
    """
    Second LLM call: match raw extracted store name to the closest canonical
    entry in the ground truth store list. Returns the matched name, or the
    original if the list is empty or the call fails.
    """
    known = _load_gt_store_names()
    if not known:
        return raw
    prompt = (
        f'Given this extracted store name: "{raw}"\n'
        f"Match it to the closest entry in this list: {known}\n"
        f"Return only the matched store name, nothing else."
    )
    try:
        if provider == "clod":
            if not CLOD_API_KEY:
                return raw
            import httpx
            resp = httpx.post(
                "https://api.clod.io/v1/chat/completions",
                headers={"Authorization": f"Bearer {CLOD_API_KEY}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}]},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        else:
            from openai import OpenAI
            client = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=64,
            )
            return resp.choices[0].message.content.strip()
    except Exception:
        return raw


def structure(ocr_text: str, image_path: Path, user_name: str,
              model: str, run_id: str, provider: str | None = None) -> dict:
    """
    Send OCR text to the configured LLM provider and return the structured receipt dict.

    :param provider: ``"openrouter"`` or ``"clod"``.  Defaults to the
        ``LLM_PROVIDER`` environment variable (fallback: ``"openrouter"``).
    """
    resolved_provider = (provider or LLM_PROVIDER).lower()

    start = time.monotonic()
    try:
        if resolved_provider == "clod":
            result, pt, ct, api_cost = _structure_clod(ocr_text, model)
            latency_ms = (time.monotonic() - start) * 1000
            if api_cost is not None:
                cost, cost_source = api_cost, "api"
            else:
                cost, cost_source = _estimate_clod_cost(model, pt, ct), "estimate"
        else:
            result, pt, ct, gen_id = _structure_openrouter(ocr_text, model)
            latency_ms = (time.monotonic() - start) * 1000
            api_cost = _fetch_openrouter_cost(gen_id)
            if api_cost is not None:
                cost, cost_source = api_cost, "api"
            else:
                cost, cost_source = _estimate_openrouter_cost(model, pt, ct), "estimate"

        # Inject caller-controlled fields
        result["imageName"] = image_path.name
        result["userName"]  = user_name

        # Geocode store address → lat/lon
        address = result.get("storeAddress") or ""
        lat, lon = geocode(address)
        result["latitude"]  = lat
        result["longitude"] = lon

        log_llm(run_id, image_path.name, user_name, resolved_provider, model,
                pt, ct, cost, latency_ms, len(result.get("items", [])), True,
                cost_source=cost_source)
        return result

    except Exception as e:
        latency_ms = (time.monotonic() - start) * 1000
        log_llm(run_id, image_path.name, user_name, resolved_provider, model,
                0, 0, 0.0, latency_ms, 0, False, str(e))
        raise


# ---------------------------------------------------------------------------
# Geocoding — Azure Maps (optional, skipped if AZURE_MAPS_KEY is unset)
# ---------------------------------------------------------------------------
def geocode(address: str) -> tuple[float | None, float | None]:
    """
    Look up lat/lon for a store address using Azure Maps Search API.
    Returns (latitude, longitude) or (None, None) if unavailable.
    Requires AZURE_MAPS_KEY in .env.
    """
    if not AZURE_MAPS_KEY or not address:
        return None, None
    try:
        import urllib.request
        import urllib.parse
        url = (
            "https://atlas.microsoft.com/search/address/json"
            f"?api-version=1.0&subscription-key={AZURE_MAPS_KEY}"
            f"&query={urllib.parse.quote(address)}&limit=1"
        )
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        results = data.get("results", [])
        if results:
            pos = results[0]["position"]
            return pos["lat"], pos["lon"]
    except Exception as e:
        print(f"  [GEO]  geocode failed for '{address}': {e}")
    return None, None


# ---------------------------------------------------------------------------
# Railtracks — Pydantic context models + function nodes
# ---------------------------------------------------------------------------
if _RT_AVAILABLE:
    from pydantic import BaseModel

    class OcrInput(BaseModel):
        image_path: str   # serialised as string; converted back to Path in node
        run_id:     str
        user_name:  str
        model:      str
        provider:   str

    class OcrOutput(OcrInput):
        ocr_text: str

    class StructureOutput(OcrOutput):
        store_name:    str
        items_count:   int
        receipt_json:  str   # json.dumps of the receipt dict

    @rt.function_node
    async def ocr_node(ctx: OcrInput) -> OcrOutput:
        """Step 1: Azure Document Intelligence OCR."""
        image_path = Path(ctx.image_path)
        await rt.broadcast(f"[OCR] Starting — {image_path.name} ({image_path.stat().st_size:,} bytes)")
        start = time.monotonic()
        # Run blocking ADI call in a thread so it doesn't block the async event loop
        ocr_text = await asyncio.to_thread(ocr, image_path, ctx.run_id, ctx.user_name)
        latency_ms = (time.monotonic() - start) * 1000
        await rt.broadcast(f"[OCR] Done — {len(ocr_text)} chars extracted in {latency_ms:.0f}ms")
        return OcrOutput(**ctx.model_dump(), ocr_text=ocr_text)

    @rt.function_node
    async def structure_node(ctx: OcrOutput) -> StructureOutput:
        """Step 2: LLM structures OCR text into JSON + Azure Maps geocoding."""
        await rt.broadcast(
            f"[LLM] Starting — provider={ctx.provider}  model={ctx.model}  "
            f"input={len(ctx.ocr_text)} chars"
        )
        start = time.monotonic()
        # Run blocking LLM + geocode call in a thread
        result = await asyncio.to_thread(
            structure, ctx.ocr_text, Path(ctx.image_path), ctx.user_name,
            ctx.model, ctx.run_id, ctx.provider
        )
        latency_ms = (time.monotonic() - start) * 1000
        items      = len(result.get("items", []))
        store      = result.get("storeName", "?")
        lat        = result.get("latitude")
        lon        = result.get("longitude")
        await rt.broadcast(
            f"[LLM] Done — {items} items  store={store}  "
            f"lat={lat}  lon={lon}  latency={latency_ms:.0f}ms"
        )
        return StructureOutput(
            **ctx.model_dump(),
            store_name=store,
            items_count=items,
            receipt_json=json.dumps(result),
        )

    @rt.function_node
    async def receipt_pipeline(ctx: OcrInput) -> StructureOutput:
        """Top-level node: chains OCR → LLM/geocode → returns StructureOutput."""
        ocr_out = await ocr_node(ctx)
        return await structure_node(ocr_out)


# ---------------------------------------------------------------------------
# Extract — runs the pipeline (via Railtracks Flow if available)
# ---------------------------------------------------------------------------
def extract(image_path: Path, user_name: str, model: str, run_id: str,
            provider: str | None = None) -> dict:
    resolved_provider = (provider or LLM_PROVIDER).lower()

    if _RT_AVAILABLE:
        flow = rt.Flow(name="receipt_etl", entry_point=receipt_pipeline)
        result: StructureOutput = flow.invoke(OcrInput(
            image_path=str(image_path),
            run_id=run_id,
            user_name=user_name,
            model=model,
            provider=resolved_provider,
        ))
        return json.loads(result.receipt_json)
    else:
        # Fallback if railtracks is not installed
        print(f"  [ADI]  OCR …")
        ocr_text = ocr(image_path, run_id, user_id=user_name)
        print(f"  [LLM]  Structuring via {resolved_provider} ({len(ocr_text)} chars) …")
        return structure(ocr_text, image_path, user_name, model, run_id, provider=resolved_provider)


# ---------------------------------------------------------------------------
# Upload via GYD SDK
# ---------------------------------------------------------------------------
def upload(receipt: dict, run_id: str):
    try:
        from gather_your_deals import GYDClient
    except ImportError:
        raise ImportError(
            "GYD SDK not installed.\n"
            "pip install git+https://github.com/yuewang199511/GatherYourDeals-SDK.git"
        )

    client = GYDClient(GYD_SERVER_URL)
    client.login(GYD_USERNAME, GYD_PASSWORD)

    items   = receipt.get("items", [])
    created, failed = [], 0
    start   = time.monotonic()

    for item in items:
        try:
            r = client.receipts.create(
                product_name=item.get("productName", ""),
                purchase_date=receipt.get("purchaseDate", ""),
                price=item.get("price", "0.00USD"),
                amount=str(item.get("amount", "1")),
                store_name=receipt.get("storeName", ""),
            )
            created.append(r)
        except Exception as e:
            print(f"    [WARN] upload failed for '{item.get('productName')}': {e}")
            failed += 1

    latency_ms = (time.monotonic() - start) * 1000
    log_upload(run_id, receipt.get("imageName", ""), receipt.get("userName", ""),
               len(items), len(created), failed, latency_ms,
               failed == 0, f"{failed} items failed" if failed else None)
    return created


# ---------------------------------------------------------------------------
# Output formatter — flatten receipt → list of per-item dicts
# ---------------------------------------------------------------------------
def flatten_receipt(receipt: dict) -> list[dict]:
    """
    Convert a receipt dict into a list of flat per-item dicts with exactly 7 fields.
    """
    rows = []
    for item in receipt.get("items", []):
        rows.append({
            "productName": item.get("productName"),
            "purchaseDate": receipt.get("purchaseDate"),
            "price":        item.get("price"),
            "amount":       item.get("amount"),
            "storeName":    receipt.get("storeName"),
            "latitude":     receipt.get("latitude"),
            "longitude":    receipt.get("longitude"),
        })
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="GatherYourDeals receipt ETL (ADI + LLM)")
    p.add_argument("path",        nargs="?", help="Image file or directory")
    p.add_argument("--user",      default=GYD_USERNAME or "unknown", help="Username for JSON metadata (defaults to GYD_USERNAME env var)")
    p.add_argument("--provider",  default=LLM_PROVIDER,
                   choices=["openrouter", "clod"],
                   help="LLM provider (default: LLM_PROVIDER env var)")
    p.add_argument("--model",     default=None,
                   help="Model ID — defaults to OR_DEFAULT_MODEL env var or provider default")
    p.add_argument("--no-upload",         action="store_true", help="Skip SDK upload")
    p.add_argument("--report",            action="store_true", help="Generate usage report")
    p.add_argument("--compare",           action="store_true",
                   help="Generate per-model comparison table scoped to current Receipts/")
    p.add_argument("--eval",              action="store_true",
                   help="Compare output/ against ground_truth/ and print scores")
    p.add_argument("--baseline-report",   action="store_true",
                   help="Generate structured baseline experiment report")
    args = p.parse_args()

    # Resolve model: CLI > provider default (when --provider explicit) > env > global default
    resolved_model = args.model or _DEFAULT_MODELS.get(args.provider) or os.getenv("OR_DEFAULT_MODEL") or DEFAULT_MODEL

    if args.report:
        report(); return

    if args.compare:
        compare(); return

    if args.eval:
        eval_receipts(); return

    if args.baseline_report:
        baseline_report(); return

    if not args.path:
        p.print_help(); sys.exit(1)

    target = Path(args.path)
    run_id = str(uuid.uuid4())

    images = (sorted(f for f in target.iterdir() if f.suffix.lower() in IMAGE_EXTS)
              if target.is_dir() else [target] if target.is_file() else [])
    if not images:
        print(f"No images found at {target}"); sys.exit(1)

    do_upload = not args.no_upload and bool(GYD_USERNAME and GYD_PASSWORD)
    if not do_upload and not args.no_upload:
        print("[INFO] GYD credentials not set — extract-only mode.")

    errors = 0
    for img in images:
        print(f"\n→ {img.name}")
        _start = time.monotonic()
        try:
            data = extract(img, args.user, resolved_model, run_id, provider=args.provider)
            total_ms = (time.monotonic() - _start) * 1000
            log_pipeline(run_id, img.name, args.user, args.provider, resolved_model, total_ms, True)
            rows = flatten_receipt(data)
            provider_out_dir = OUTPUT_DIR / args.provider
            provider_out_dir.mkdir(parents=True, exist_ok=True)
            out = provider_out_dir / (img.stem + ".json")
            out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  saved  {out}  ({len(rows)} items)")
            if do_upload:
                created = upload(data, run_id)
                print(f"  uploaded {len(created)}/{len(rows)} items")
        except Exception as e:
            total_ms = (time.monotonic() - _start) * 1000
            log_pipeline(run_id, img.name, args.user, args.provider, resolved_model, total_ms, False, str(e))
            print(f"  ERROR: {e}", file=sys.stderr)
            errors += 1

    print(f"\nDone — {len(images)-errors}/{len(images)} succeeded.")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
