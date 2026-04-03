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
  GYD_ACCESS_TOKEN=<jwt-access-token>   # from: gatherYourDeals show-token
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
from etl_logger import (  # noqa: E402
    log_adi, log_llm, log_pipeline, log_upload, ADI_COST_PER_PAGE, LOGS_DIR,
)

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

# Which LLM backend to use: "clod" (default) or "openrouter"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "clod").lower()

# OR_DEFAULT_MODEL   = os.getenv("OR_DEFAULT_MODEL",   "anthropic/claude-haiku-4.5")  # OpenRouter disabled as default
OR_DEFAULT_MODEL   = os.getenv("OR_DEFAULT_MODEL",   "anthropic/claude-haiku-4.5")  # still usable via --provider openrouter
CLOD_DEFAULT_MODEL = os.getenv("CLOD_DEFAULT_MODEL", "google/gemma-3n-E4B-it")

# API endpoints — read from env to allow routing to proxies or alternate regions
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
CLOD_API_URL        = os.getenv("CLOD_API_URL",        "https://api.clod.io/v1/chat/completions")
AZURE_MAPS_URL      = os.getenv("AZURE_MAPS_URL",      "https://atlas.microsoft.com/search/address/json")

# Operational constants
_ADI_TIMEOUT_S  = int(os.getenv("AZURE_DI_TIMEOUT",  "120"))
_CLOD_TIMEOUT_S = int(os.getenv("CLOD_TIMEOUT",       "120"))
_CLOD_RETRIES   = int(os.getenv("CLOD_RETRIES",       "3"))

GYD_SERVER_URL   = os.getenv("GYD_SERVER_URL", "http://localhost:8080/api/v1")
GYD_ACCESS_TOKEN = os.getenv("GYD_ACCESS_TOKEN", "")

OUTPUT_DIR       = Path("output")
LOGS_DIR         = Path("logs")
REPORTS_DIR      = Path("reports")
GROUND_TRUTH_DIR = Path("ground_truth")

# Registry that maps image stem → list of GYD receipt UUIDs created on upload.
# Used by delete_uploaded() to find and remove records from the database.
_UPLOAD_REGISTRY = OUTPUT_DIR / ".upload_registry.json"


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".tiff", ".tif", ".bmp"}

# Enable Railtracks logging (writes to logs/rt.log + stdout at INFO level)
if _RT_AVAILABLE:
    LOGS_DIR.mkdir(exist_ok=True)
    rt.enable_logging(level="INFO", log_file=str(LOGS_DIR / "rt.log"))

# ---------------------------------------------------------------------------
# LLM prompt — receives markdown-formatted OCR text, returns structured JSON
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are a receipt data structuring assistant.
You are given markdown-formatted OCR text extracted from a receipt image by \
Azure Document Intelligence. The markdown may contain tables — use them to \
accurately identify line items, quantities, and prices.

## Extraction process

Work through the receipt in three steps. Show your work in the tagged sections \
below, then output the final JSON. This reduces errors on messy receipts.

<spans>
Quote verbatim from the OCR text:
- HEADER: the store name, address, and date/time block
- ITEMS: every product line item row exactly as printed
- TOTALS: subtotal, tax, total, and payment lines
</spans>

<extract>
Using the SPATIAL LAYOUT section (if present), reconstruct receipt rows first:
- Each row follows: [L] ITEM NAME  [C] QTY (optional)  [R] PRICE
- Prices are always right-aligned ([R] column) — never assign a [L] or [C] value as price
- Quantities / weights are center-aligned ([C]) or follow the item name in [L]
- If no SPATIAL LAYOUT, use the markdown table rows

Then list raw extracted values:
- date: <raw date string>
- time: <raw time string>
- currency: <symbol or code found>
- For each item row: productName | itemCode | raw_price | raw_amount | category
</extract>

<json>
{final normalized JSON conforming to the schema below}
</json>

## Output schema

Required top-level fields (use null when not visible in the OCR text):
  imageName     string   (caller injects — output null)
  userName      string   (caller injects — output null)
  storeName     string
  storeAddress  string   IMPORTANT: copy the full street address from the receipt header
                         (e.g. "123 Main St, Springfield, CA 90210"). Include street,
                         city, state/province, and postal code if visible. Do NOT leave
                         null unless the receipt truly has no address printed on it.
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

Each element of items must have exactly these fields:
  productName   string
  itemCode      string or null   — UPC / item code printed next to the product
  price         string   "X.XXUSD" — the FINAL charged price (after discounts), always include currency code
  amount        string   — quantity or weight. Use ONLY these unit labels (US/Canada standard):
                             Weight:  lb, lbs, oz, kg, g
                             Volume:  fl oz, gal, qt, pt, L, mL
                             Count:   just the bare number, no unit (e.g. "1", "2", "3")
                           Strip any label not in this list — "W", "EA", "PK", "F", or any
                           non-measurement code — and output only the numeric value.
                           NEVER put a dollar amount in this field.
  category      string or null   — department label printed on the receipt (e.g. "Grocery", "Produce")

## Output schema (place inside <json> tags)

{
  "imageName":     null,
  "userName":      null,
  "storeName":     string | null,
  "storeAddress":  string | null,
  "latitude":      null,
  "longitude":     null,
  "purchaseDate":  string | null,
  "purchaseTime":  string | null,
  "currency":      string | null,
  "items": [
    {
      "productName": string | null,
      "itemCode":    string | null,
      "price":       string | null,
      "amount":      string | null,
      "category":    string | null
    }
  ],
  "totalItems":    integer | null,
  "subtotal":      string | null,
  "tax":           string | null,
  "total":         string | null,
  "paymentMethod": string | null
}

## Rules

- Include only grocery/food product line items. Skip deposits, recycling fees, donations, bag fees, and miscellaneous non-product charges.
- storeName must be the store's brand/chain name as printed on the receipt header (e.g. "Costco Wholesale", "Your Independent Grocer", "T&T Supermarket"). Do not append address, city, or branch details.
- purchaseDate must be the transaction date, formatted YYYY.MM.DD. If the year looks implausible (before 2020) you are likely reading a receipt number, time, or barcode — re-examine the receipt for the correct date.
- price is the total amount charged for that line item as shown in the right-hand price column. Do not use per-unit rates, per-oz prices, or any divided amount. Always include the currency code: "4.79USD" not "4.79".
- amount is a count or weight only — never a price. Strip unrecognised unit codes down to the bare number.
- Weight-priced items: some receipts print "1.160 kg @ $1.72/kg  2.00" — set amount to the weight ("1.160kg") and price to the total charged ("2.00CAD"), not the per-kg rate.
- If a markdown table is present, each row is one line item — do not merge or split rows.
- Column anchoring: each receipt row follows [ITEM NAME] [QTY optional] [PRICE]. Prices are always right-aligned — do not assign a left- or center-column value as a price. Do not assign a right-column value as a quantity.

## Costco receipt rules

- Item numbers: Costco prints a numeric item code before or after the product name (e.g. "47825 GREEN GRAPES" or "GREEN GRAPES 47825"). Put that number in `itemCode` and remove it from `productName`.
- Price letter suffixes: Costco appends a single letter to prices to indicate tax category (e.g. "8.99A", "11.99N", "26.09E"). Strip the trailing letter — `price` must be the numeric value plus currency only: "8.99USD".
- CA Redemption Value / REDEMP VA / CRV lines are deposit fees — skip them entirely, do not include as items. Critically: the dollar amount on a CA REDEMP VA line (e.g. "0.60A", "1.75A") belongs to the fee, NOT to the product printed above it. Never assign a CA REDEMP VA price to a product.
- Abbreviated product names: Costco uses abbreviated names on the receipt (e.g. "KSORGWHLEMLK", "ORG FR EGGS", "HONYCRSP"). Expand these to their full human-readable form using context clues: "KS Organic Whole Milk", "Organic Free Range Eggs", "Honeycrisp Apples", etc.
- Each product appears only once on the receipt. If you see the same item at different points in the text (e.g. once with an item code and once without), output it exactly once.

## Handling ambiguity

- If multiple candidate values exist for the same field (e.g. two dates, two totals), choose the most recent or most prominent one.
- If confidence in a field value is low — the text is unclear, partially obscured, or contradictory — return null rather than guessing.
- Do not fabricate or infer values that are not explicitly present in the OCR text. If a field is not visible, output null.

Also include every other field readable from the OCR text as extra top-level fields — \
cashier, member/rewards number, savings, transaction ID, store phone, operator number, etc. \
Preserve all receipt detail."""

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


def _reconstruct_spatial_rows(result) -> str:
    """
    Build a column-labeled spatial layout from ADI bounding-box data.

    Each line from ADI carries a polygon ([x1,y1,...,x4,y4]) that tells us
    exactly where on the page it sits.  We bucket every line into one of three
    logical columns based on its left-edge X position relative to page width:

        [L] < 40%  — item description / product name
        [C] 40-70% — quantity / unit
        [R] > 70%  — price

    Column-aware row building (two-pass):
      Pass 1 — group [L] tokens by Y proximity into item rows (same as before).
      Pass 2 — assign each [R] token to the nearest [L] row whose Y is AT OR
               BELOW the price token's Y (handles receipts like Costco where the
               price bounding-box sits slightly above the item-name bounding-box
               on the page).  Falls back to the nearest row above when nothing
               is found below.

    This is passed to the LLM as a supplementary section alongside the
    markdown text so it can anchor column assignments without guessing.
    """
    if not result.pages:
        return ""

    page        = result.pages[0]
    page_width  = max(getattr(page, "width",  0) or 0, 1.0)
    page_height = max(getattr(page, "height", 0) or 0, 1.0)
    tolerance   = page_height * 0.015   # lines within 1.5% of page height → same row

    line_data: list[tuple[float, float, str]] = []   # (y_center, x_left, text)
    for line in (page.lines or []):
        poly = getattr(line, "polygon", None)
        text = getattr(line, "content", None)
        if not poly or not text:
            continue
        # polygon: flat list [x1,y1,x2,y2,x3,y3,x4,y4] or list of Point objects
        if hasattr(poly[0], "x"):
            xs = [p.x for p in poly]
            ys = [p.y for p in poly]
        else:
            xs = list(poly[0::2])
            ys = list(poly[1::2])
        line_data.append((sum(ys) / len(ys), min(xs), text.strip()))

    line_data.sort(key=lambda t: t[0])

    # --- Pass 1: separate tokens by column, group [L] rows by Y proximity ---
    left_tokens:   list[tuple[float, float, str]] = []
    center_tokens: list[tuple[float, float, str]] = []
    right_tokens:  list[tuple[float, float, str]] = []
    for y, x, text in line_data:
        pct = x / page_width
        if pct < 0.40:
            left_tokens.append((y, x, text))
        elif pct < 0.70:
            center_tokens.append((y, x, text))
        else:
            right_tokens.append((y, x, text))

    # Group [L] tokens into rows by Y proximity
    left_groups: list[list[tuple[float, float, str]]] = []
    for entry in left_tokens:
        if left_groups and abs(entry[0] - left_groups[-1][0][0]) <= tolerance:
            left_groups[-1].append(entry)
        else:
            left_groups.append([entry])

    if not left_groups:
        return ""

    # Representative Y for each [L] group (first token's Y, groups are sorted)
    group_ys = [g[0][0] for g in left_groups]

    # --- Pass 2: assign [R] tokens to [L] groups ---
    # For each price, prefer the nearest [L] group whose Y >= price_Y
    # (the item is at or below the price on the page — handles Costco format).
    # Fall back to the nearest group above when nothing is found below.
    group_rights: list[list[tuple[float, float, str]]] = [[] for _ in left_groups]

    for r_y, r_x, r_text in right_tokens:
        best_idx, best_dist = 0, float("inf")

        # Forward scan: find nearest group at or below the price
        for i, g_y in enumerate(group_ys):
            if g_y >= r_y:           # group is at or below the price
                dist = g_y - r_y
                if dist < best_dist:
                    best_dist, best_idx = dist, i
                break                # groups sorted ascending; first match is nearest

        # If best_dist is still large, also check nearest group above as fallback
        for i, g_y in enumerate(group_ys):
            dist = abs(g_y - r_y)
            if dist < best_dist:
                best_dist, best_idx = dist, i

        group_rights[best_idx].append((r_y, r_x, r_text))

    # Assign [C] tokens to nearest [L] group by Y
    group_centers: list[list[tuple[float, float, str]]] = [[] for _ in left_groups]
    for c_y, c_x, c_text in center_tokens:
        best_idx = min(range(len(group_ys)), key=lambda i: abs(group_ys[i] - c_y))
        group_centers[best_idx].append((c_y, c_x, c_text))

    # --- Render rows ---
    output_lines: list[str] = []
    for i, group in enumerate(left_groups):
        parts: list[str] = []
        # Left + center tokens sorted left-to-right
        lc = [(t, "L") for t in group] + [(t, "C") for t in group_centers[i]]
        lc.sort(key=lambda tc: tc[0][1])
        for t, col in lc:
            parts.append(f"[{col}] {t[2]}")
        # Right tokens sorted left-to-right
        for _, _, text in sorted(group_rights[i], key=lambda t: t[1]):
            parts.append(f"[R] {text}")
        if parts:
            output_lines.append("  |  ".join(parts))

    return "\n".join(output_lines)


def ocr(image_path: Path, run_id: str, user_id: str = "") -> str:
    """
    Send image to Azure Document Intelligence (prebuilt-read).
    Returns markdown OCR text with a spatial layout section appended.

    The spatial section labels each line [L]eft / [C]enter / [R]ight based on
    its bounding-box X position, preserving column structure for the LLM.
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
            output_content_format="markdown",
        )
        result = poller.result(timeout=_ADI_TIMEOUT_S)
        latency_ms = (time.monotonic() - start) * 1000

        pages    = len(result.pages) if result.pages else 1
        markdown = result.content or ""

        spatial  = _reconstruct_spatial_rows(result)
        ocr_text = (
            markdown
            + "\n\n---\n## SPATIAL LAYOUT\n"
            + "Each token labeled [L]=description  [C]=quantity  [R]=price.\n"
            + "Use this section to extract items — preserves column alignment.\n\n"
            + spatial
        ) if spatial else markdown

        log_adi(run_id, image_path.name, user_id, image_size_bytes,
                pages, pages * ADI_COST_PER_PAGE, latency_ms, True,
                chars_extracted=len(markdown))
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
    """
    Extract and parse JSON from LLM output.

    Tries in order:
      1. Content inside <json>...</json> tags (chain-of-thought response)
      2. Markdown fences stripped (```json ... ```)
      3. Raw string as-is
    """
    raw = raw.strip()

    # 1. Chain-of-thought: extract from <json> tags
    m = re.search(r"<json>\s*(.*?)\s*</json>", raw, re.DOTALL)
    if m:
        return json.loads(m.group(1))

    # 2. Markdown fences — search anywhere in the response (model may prefix with analysis)
    m = re.search(r"```(?:json)?\s*\n(.*?)\n```", raw, re.DOTALL)
    if m:
        raw = m.group(1).strip()

    # 3. Strip trailing commas (some models emit JSON5-style output)
    raw = re.sub(r",\s*([}\]])", r"\1", raw)

    return json.loads(raw)


def _structure_openrouter(ocr_text: str, model: str) -> tuple[dict, int, int, str]:
    """Call OpenRouter and return (parsed_dict, prompt_tokens, completion_tokens, generation_id)."""
    if not OPENROUTER_API_KEY:
        raise EnvironmentError("OPENROUTER_API_KEY not set — add it to .env")

    from openai import OpenAI
    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL, timeout=90)

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


def _fetch_openrouter_generation(generation_id: str) -> dict:
    """
    Fetch generation metadata from OpenRouter's generation endpoint.
    Returns dict with 'cost' (USD float or None) and 'latency_ms' (float or None).
    """
    if not generation_id or not OPENROUTER_API_KEY:
        return {"cost": None, "latency_ms": None}
    import urllib.request
    url = f"{OPENROUTER_BASE_URL}/generation?id={generation_id}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"})
    # Small delay — OpenRouter may lag a few hundred ms before cost data is ready
    time.sleep(0.5)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read())
            gen = data.get("data", {})
            cost = gen.get("total_cost")
            latency = gen.get("generation_time")  # ms, server-side generation time
            return {
                "cost":       float(cost)    if cost    is not None else None,
                "latency_ms": float(latency) if latency is not None else None,
            }
        except Exception:
            pass
        if attempt < 2:
            time.sleep(1)
    return {"cost": None, "latency_ms": None}


# Token-based cost estimates ($/M tokens) for known OpenRouter models.
# Used as fallback when the generation API doesn't return a cost.
# https://openrouter.ai/anthropic/claude-haiku-4.5
_OR_PRICING: dict[str, tuple[float, float]] = {
    "anthropic/claude-haiku-4.5":    (1.00, 5.00),  # $/M input, $/M output
    "qwen/qwen-2.5-7b-instruct":     (0.04, 0.10),  # $/M input, $/M output
}

def _estimate_openrouter_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost from token counts using known OpenRouter pricing."""
    price_in, price_out = _OR_PRICING.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


# Token-based cost estimates ($/M tokens) for known CLOD models.
# Used as fallback when the CLOD API response does not include a cost field.
# Source: clod.io pricing page (2026-03-29); Gemma 3n E4B IT updated 2026-04-01 from Together.ai/Vertex AI.
# Output price is dynamically priced based on real-time energy costs;
# $0.12/M is the observed running rate, $0.318/M is the price cap.
_CLOD_PRICING: dict[str, tuple[float, float]] = {
    "Qwen2.5-7B-Instruct-Turbo":      (0.30, 0.12),  # $/M input, $/M output (60% discount applied)
    "Qwen/Qwen2.5-7B-Instruct-Turbo": (0.30, 0.12),  # alias with namespace prefix
    "qwen/qwen-2.5-7b-instruct":      (0.04, 0.10),  # OpenRouter slug for the same model
    "gemma-3n-E4B-it":                (0.02, 0.04),  # $/M input, $/M output — Together.ai rate (Vertex AI: $0.02/$0.03 with 24% discount)
    "google/gemma-3n-E4B-it":         (0.02, 0.04),  # alias with namespace prefix
    "anthropic/claude-haiku-4-5":     (1.00, 5.00),  # confirm CLOD pricing — using OpenRouter rate as placeholder
}

def _estimate_clod_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Estimate cost from token counts using known CLOD pricing.
    Returns None when the model has no rate card entry (cost unknown)."""
    rates = _CLOD_PRICING.get(model)
    if rates is None or rates == (0.0, 0.0):
        return None
    price_in, price_out = rates
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
    import time
    last_exc: Exception | None = None
    for attempt in range(1, _CLOD_RETRIES + 1):
        try:
            resp = httpx.post(
                CLOD_API_URL,
                headers={"Authorization": f"Bearer {CLOD_API_KEY}", "Content-Type": "application/json"},
                json=payload,
                timeout=_CLOD_TIMEOUT_S,
            )
            resp.raise_for_status()
            break
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError) as exc:
            last_exc = exc
            if attempt < _CLOD_RETRIES:
                wait = 2 ** attempt
                print(f"[etl] CLOD attempt {attempt}/{_CLOD_RETRIES} timed out, retrying in {wait}s…", file=__import__('sys').stderr)
                time.sleep(wait)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (429, 502, 503, 504) and attempt < _CLOD_RETRIES:
                last_exc = exc
                wait = 2 ** attempt
                print(f"[etl] CLOD attempt {attempt}/{_CLOD_RETRIES} got {exc.response.status_code}, retrying in {wait}s…", file=__import__('sys').stderr)
                time.sleep(wait)
            else:
                raise
    else:
        raise last_exc
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
    # CLOD may include actual cost in the usage object; fall back to None if absent.
    # Check all known field names across providers and response levels.
    api_cost = (
        usage.get("total_cost")
        or usage.get("cost")
        or usage.get("cost_usd")
        or data.get("total_cost")
        or data.get("cost")
    )
    api_latency = (
        usage.get("total_time")
        or usage.get("generation_time")
        or data.get("generation_time")
    )
    # If cost is still missing, log the raw usage dict so the field name can be identified
    if api_cost is None:
        import sys
        print(f"[etl_logger] CLOD cost not found in response for model={model}. "
              f"usage keys: {list(usage.keys())}", file=sys.stderr)
    return (
        _parse_llm_json(content),
        pt, ct,
        float(api_cost)    if api_cost    is not None else None,
        float(api_latency) if api_latency is not None else None,
    )


# ---------------------------------------------------------------------------
# Long-receipt chunking
# ---------------------------------------------------------------------------
# LLM attention degrades when OCR text is long and noisy: rows bleed into each
# other, items get merged or dropped, and column alignment is lost.  Receipts
# over _CHUNK_THRESHOLD_CHARS are split into overlapping vertical sections
# before being sent to the LLM.  Results are merged after all chunks return.

_CHUNK_THRESHOLD_CHARS = 1000   # ~1191 chars for a 14-item Costco receipt
_CHUNK_HEADER_LINES    = 6      # lines to prepend to every chunk (store/date context)
_CHUNK_MAX_CHARS       = 700    # max body chars per chunk (excluding prepended header)
_CHUNK_OVERLAP_LINES   = 6      # overlap between chunks — 6 lines keeps item+price+CA REDEMP VA together


def _split_ocr_into_chunks(ocr_text: str) -> list[str]:
    """
    Split long OCR text into overlapping chunks for the LLM.

    When a SPATIAL LAYOUT section is present (appended by the ADI step), only
    the spatial section is chunked — the raw OCR body is excluded.  This avoids
    feeding Qwen/Gemma the price-before-name ordering that some receipts produce
    in the raw scan (e.g. Costco prints a price line before the item-code line).
    The raw OCR header (store name / address — first 3 lines) is still prepended
    to every chunk so the LLM has receipt-level context.

    When no spatial section exists the entire raw OCR is chunked as before.

    In both cases the body is split at line boundaries with _CHUNK_OVERLAP_LINES
    of overlap to prevent cutting a multi-line item row.
    """
    _SPATIAL_MARKER = "\n---\n## SPATIAL LAYOUT\n"
    _DATE_MDY   = re.compile(r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b')  # MM/DD/YYYY
    _DATE_SHORT = re.compile(r'\b(\d{2})/(\d{2})/(\d{2})\b')       # YY/MM/DD or DD/MM/YY

    def _garbled(line: str) -> bool:
        """True when a line is unlikely to be useful receipt header text.

        Catches two common noise patterns:
          - Low alphanumeric density (symbols/punctuation dominate)
          - Bleed-through / reversed text: starts lowercase with no digits
            (real store-name/address lines always start with a capital or digit)
          - Lines starting with a non-alphanumeric character and containing no digits
        """
        s = line.strip()
        if not s:
            return True
        has_digit = any(c.isdigit() for c in s)
        alnum_ratio = sum(c.isalnum() or c.isspace() for c in s) / len(s)
        if alnum_ratio < 0.55:
            return True
        if s[0].islower() and not has_digit:
            return True
        if not s[0].isalnum() and not has_digit:
            return True
        return False

    def _extract_date(text: str) -> str | None:
        """Return YYYY.MM.DD from text, supporting MM/DD/YYYY and YY/MM/DD formats."""
        m = _DATE_MDY.search(text)
        if m:
            mon, day, yr = m.group(1), m.group(2), m.group(3)
            return f"{yr}.{mon.zfill(2)}.{day.zfill(2)}"
        m = _DATE_SHORT.search(text)
        if m:
            a, b, c = int(m.group(1)), int(m.group(2)), int(m.group(3))
            # Determine format: the year component must be ≤ 99 and > 12 to be
            # distinguishable.  If first or last field > 12 it's the year (2-digit).
            if a > 12:          # YY/MM/DD
                return f"20{a:02d}.{b:02d}.{c:02d}"
            elif c > 12:        # MM/DD/YY  (e.g. 02/21/26)
                return f"20{c:02d}.{a:02d}.{b:02d}"
            # Ambiguous (all ≤ 12); skip rather than guess
        return None

    if _SPATIAL_MARKER in ocr_text:
        raw_part, spatial_part = ocr_text.split(_SPATIAL_MARKER, 1)
        raw_lines = raw_part.split('\n')
        # Filter garbled lines (reversed/bleed-through text) so the header
        # only contains clean store-name / address lines.
        clean_lines = [l for l in raw_lines[:10] if not _garbled(l)]
        header = clean_lines[:3]
        # Inject purchase date so every chunk has it (prevents hallucination).
        date_str = _extract_date(raw_part)
        if date_str:
            header.append(f"Purchase date: {date_str}")
        body_lines = spatial_part.split('\n')
    else:
        lines  = ocr_text.split('\n')
        header = lines[:_CHUNK_HEADER_LINES]
        body_lines = lines[_CHUNK_HEADER_LINES:]

    chunks: list[str] = []
    current: list[str] = []
    current_chars = 0

    for line in body_lines:
        if current_chars + len(line) > _CHUNK_MAX_CHARS and current:
            chunks.append('\n'.join(header + current))
            current       = current[-_CHUNK_OVERLAP_LINES:]
            current_chars = sum(len(l) for l in current)
        current.append(line)
        current_chars += len(line)

    if current:
        chunks.append('\n'.join(header + current))

    return chunks or [ocr_text]


def _merge_chunk_results(results: list[dict]) -> dict:
    """
    Merge per-chunk extraction dicts into a single receipt dict.

    - Scalar header fields (storeName, purchaseDate, etc.): from the first chunk
    - items array: union of all chunks, deduplicated by (productName, price)
    - Scalar footer fields (subtotal, tax, total, paymentMethod): from the last
      chunk that carries a non-null value
    """
    if len(results) == 1:
        return results[0]

    merged = {k: v for k, v in results[0].items() if k != 'items'}

    seen: set[tuple] = set()
    all_items: list[dict] = []
    for chunk in results:
        for item in chunk.get('items', []):
            # Strip leading item codes (e.g. "47825 GREEN GRAPES" → "GREEN GRAPES")
            # so the same product extracted with/without its code deduplicates correctly.
            name = re.sub(r'^\d+\s+', '', (item.get('productName') or '').strip()).lower()
            # Strip trailing price letter codes (A/N/E) before comparing prices
            price = re.sub(r'[A-Za-z]+$', '', (item.get('price') or '').strip())
            key = (name, price)
            if key not in seen:
                seen.add(key)
                all_items.append(item)

    merged['items']      = all_items
    merged['totalItems'] = len(all_items)

    # purchaseDate may only appear in a later chunk (e.g. when the spatial-only
    # path is used and the date line falls past the first chunk boundary).
    # Scan forward through all chunks to find the first non-null value.
    for field in ('purchaseDate',):
        if not merged.get(field):
            for chunk in results:
                if chunk.get(field):
                    merged[field] = chunk[field]
                    break

    for field in ('subtotal', 'tax', 'total', 'paymentMethod'):
        for chunk in reversed(results):
            if chunk.get(field) is not None:
                merged[field] = chunk[field]
                break

    return merged


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
                CLOD_API_URL,
                headers={"Authorization": f"Bearer {CLOD_API_KEY}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}]},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        else:
            from openai import OpenAI
            client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=64,
            )
            return resp.choices[0].message.content.strip()
    except Exception:
        return raw


def _validate_and_fix_items(items: list[dict], currency: str = "USD") -> list[dict]:
    """
    Deterministic post-processing rules applied after LLM extraction.

    Rules (in order):
      1. Drop items with no productName.
      2. Drop non-product rows (tax lines, savings/discount lines, redemption fees).
      3. Strip leading item/barcode codes from productName (e.g. "4164501 KS SPARKLING").
      4. Strip trailing letter tax codes from price (Costco "11.99A" → "11.99USD").
      5. Normalize price: ensure currency suffix present; drop items with price ≤ 0.
      6. Drop items with implausibly high prices (> $99) — likely a total/subtotal row.
      7. Clear amount field if it contains a price pattern and a valid price already
         exists — amount must be a count/weight, never a dollar value.
      8. Detect column swap: amount looks like a price and price is missing/zero → swap.
      9. Strip unrecognised unit codes from amount.
    """
    _PRICE_RE       = re.compile(r"(\d+\.?\d*)")
    _UNIT_STRIP     = re.compile(r"\b(W|EA|PK|F|PC|CT|BG|LT|BT|CN|OZ|each)\b", re.IGNORECASE)
    _PRICE_FMT      = re.compile(r"^\d+\.\d{2}\s*[A-Za-z]?$")  # "4.79", "4.79S", "4.79 S"
    _ITEM_CODE      = re.compile(r"^\d{4,}\s+")            # leading 4+ digit item/barcode code
    _PRICE_LETTER   = re.compile(r"^(\d+\.?\d*)[A-Za-z]+$")  # "11.99A", "8.99N"
    _NON_PRODUCT    = re.compile(
        r"\b(tax|saving|savings|discount|instant\s+saving|subtotal|total|"
        r"redemp|crv|deposit|donation|bag\s+fee)\b",
        re.IGNORECASE,
    )
    _MAX_ITEM_PRICE = 99.0   # prices above this are almost certainly totals/subtotals
    _MIN_ITEM_PRICE = 0.50   # prices below this are almost certainly CA CRV / deposit fees bleeding in

    fixed: list[dict] = []
    for item in items:
        name = (item.get("productName") or "").strip()
        if not name:
            continue

        # Drop non-product rows (tax, savings, redemption fees, totals)
        if _NON_PRODUCT.search(name):
            continue

        # Strip leading item/barcode codes the LLM left in productName
        clean_name = _ITEM_CODE.sub("", name).strip()
        if clean_name and clean_name != name:
            item["productName"] = clean_name
            name = clean_name

        # Strip trailing letter tax codes from price (Costco "11.99A" → "11.99")
        raw_price = str(item.get("price") or "")
        m_letter = _PRICE_LETTER.match(raw_price.strip())
        if m_letter:
            raw_price = m_letter.group(1)

        # Parse price to float
        m = _PRICE_RE.search(raw_price)
        price_val = float(m.group(1)) if m else None

        # Drop items with non-positive, implausibly high, or CRV-range price
        if price_val is not None and (price_val <= 0 or price_val < _MIN_ITEM_PRICE or price_val > _MAX_ITEM_PRICE):
            continue

        # Normalize price: ensure currency suffix
        if price_val is not None:
            item["price"] = f"{price_val:.2f}{currency}"

        # --- Amount sanity ---
        raw_amount = str(item.get("amount") or "").strip()

        # Strip trailing tax-code letter from amount (e.g. "4.79 S" → "4.79")
        # before any price-pattern checks so the stripped value is used consistently.
        m_amt_letter = re.match(r"^(\d+\.\d{2})\s*[A-Za-z]$", raw_amount)
        if m_amt_letter:
            raw_amount = m_amt_letter.group(1)

        # Clear amount if it contains a price pattern and a valid price already exists
        # (amount must be a count/weight, not a dollar value)
        if _PRICE_FMT.match(raw_amount) and price_val is not None and price_val > 0:
            raw_amount = ""

        # Detect column swap: amount looks like a price and price is missing/zero
        if _PRICE_FMT.match(raw_amount) and (price_val is None or price_val == 0):
            item["price"] = f"{raw_amount}{currency}"
            raw_amount    = ""

        # Strip unrecognised unit codes the LLM may have left
        if raw_amount:
            cleaned_amount = _UNIT_STRIP.sub("", raw_amount).strip()
            raw_amount = cleaned_amount

        # Default amount to "1" if nothing valid remains
        item["amount"] = raw_amount if raw_amount else "1"

        fixed.append(item)

    return fixed


_CURRENCY_MARKERS = [
    (re.compile(r'\bCAD\b|\bCAD\$|C\$|\$CAD', re.IGNORECASE), "CAD"),
    (re.compile(r'\bGBP\b|£',                                  re.IGNORECASE), "GBP"),
    (re.compile(r'\bEUR\b|€',                                  re.IGNORECASE), "EUR"),
]


def _detect_currency_from_ocr(ocr_text: str) -> str | None:
    """
    Scan OCR text for explicit currency markers and return the currency code.
    Returns None when no non-USD marker is found (caller should default to USD).
    """
    for pattern, code in _CURRENCY_MARKERS:
        if pattern.search(ocr_text):
            return code
    return None


# Matches lines that look like a street address: start with a number followed by
# a street name, optionally followed by city/state/ZIP on the same or next line.
_STREET_LINE = re.compile(
    r"^\s*\d+\s+\w[\w\s,\.#-]{5,}(?:st|street|ave|avenue|blvd|boulevard|rd|road|dr|drive|ln|lane|way|pkwy|hwy|cyn|canyon)\b",
    re.IGNORECASE,
)
_CITY_STATE_ZIP = re.compile(r"[A-Z][a-zA-Z\s]+,?\s+[A-Z]{2}\s+\d{5}", re.IGNORECASE)


def _extract_address_from_ocr(ocr_text: str, store_name: str) -> str:
    """
    Fallback: scan the first 30 lines of OCR text for a street address line.
    Returns the best candidate as a single string, or "" if none found.
    """
    lines = ocr_text.splitlines()[:30]
    street_line = ""
    for i, line in enumerate(lines):
        m = _STREET_LINE.search(line)
        if m:
            # Slice from the match start so we don't capture store-name/phone
            # text that precedes the street number on the same OCR line.
            candidate = line[m.start():].strip()
            # If city/state/ZIP already follows on the same line, we're done.
            if _CITY_STATE_ZIP.search(candidate):
                street_line = candidate
            else:
                street_line = candidate
                # Try to append the next line if it looks like city/state/ZIP
                if i + 1 < len(lines):
                    nxt = lines[i + 1].strip()
                    if _CITY_STATE_ZIP.search(nxt):
                        street_line = f"{street_line}, {nxt}"
            break
    # If no street line found, look for a city/state/ZIP anywhere in a line,
    # then search backwards for the nearest street number to anchor the address.
    if not street_line:
        for line in lines:
            czm = _CITY_STATE_ZIP.search(line)
            if czm:
                before = line[:czm.start()]
                # Find the last street-number candidate (3-5 digits + letter) before the city match
                street_nums = list(re.finditer(r'\b\d{3,5}\s+[A-Za-z]', before))
                if street_nums:
                    # Use the last (closest) street number as the start of the address
                    street_line = line[street_nums[-1].start():czm.end()].strip()
                else:
                    # Check for a street number right at the end of 'before'
                    # (happens when the number and street name span the city-match boundary)
                    dangling = re.search(r'\b(\d{3,5})\s*$', before)
                    if dangling:
                        street_line = f"{dangling.group(1)} {line[czm.start():czm.end()].strip()}"
                    else:
                        # No street number found — just use city/state/zip with store name prefix
                        city_part = line[czm.start():czm.end()].strip()
                        street_line = f"{store_name} {city_part}" if store_name else city_part
                break
    return street_line


def structure(ocr_text: str, image_path: Path, user_name: str,
              model: str, run_id: str, provider: str | None = None) -> dict:
    """
    Send OCR text to the configured LLM provider and return the structured receipt dict.

    Long receipts (> _CHUNK_THRESHOLD_CHARS) are split into overlapping vertical
    sections before extraction to prevent LLM attention degradation on noisy OCR.
    Chunk results are merged before geocoding.

    :param provider: ``"openrouter"`` or ``"clod"``.  Defaults to the
        ``LLM_PROVIDER`` environment variable (fallback: ``"openrouter"``).
    """
    resolved_provider = (provider or LLM_PROVIDER).lower()

    chunks = (
        _split_ocr_into_chunks(ocr_text)
        if len(ocr_text) > _CHUNK_THRESHOLD_CHARS
        else [ocr_text]
    )
    is_chunked = len(chunks) > 1

    start = time.monotonic()
    total_pt, total_ct, total_cost = 0, 0, 0.0
    cost_source    = "estimate"
    latency_source = "local"
    chunk_results: list[dict] = []

    try:
        for chunk_text in chunks:
            if resolved_provider == "clod":
                try:
                    c_result, pt, ct, api_cost, api_latency_ms = _structure_clod(chunk_text, model)
                except json.JSONDecodeError:
                    # Chunk returned unparseable content — skip it, other chunks still contribute
                    continue
                if api_cost is not None:
                    total_cost += api_cost
                    cost_source = "api"
                else:
                    total_cost += _estimate_clod_cost(model, pt, ct) or 0.0
            else:
                c_result, pt, ct, gen_id = _structure_openrouter(chunk_text, model)
                gen = _fetch_openrouter_generation(gen_id)
                if gen["cost"] is not None:
                    total_cost += gen["cost"]
                    cost_source = "api"
                else:
                    total_cost += _estimate_openrouter_cost(model, pt, ct)
                if not is_chunked and gen["latency_ms"] is not None:
                    latency_source = "api"

            total_pt += pt
            total_ct += ct
            chunk_results.append(c_result)

        # Merge chunks (no-op when only one chunk)
        result = _merge_chunk_results(chunk_results)

        # Deterministic post-processing: drop bad rows, fix column swaps
        # Override LLM-extracted currency with deterministic OCR scan so
        # small models that default to "USD" are corrected for CA/GB/EU receipts.
        currency = _detect_currency_from_ocr(ocr_text) or result.get("currency") or "USD"
        result["currency"] = currency
        result["items"] = _validate_and_fix_items(result.get("items", []), currency)
        result["totalItems"] = len(result["items"])

        latency_ms = (time.monotonic() - start) * 1000

        # Inject caller-controlled fields
        result["imageName"] = image_path.name
        result["userName"]  = user_name

        # Geocode once using merged store address.
        # Fallback: if LLM returned no storeAddress, extract it from the OCR text
        # by looking for a line that contains digits + street keywords near the top.
        address = result.get("storeAddress") or ""
        if not address:
            address = _extract_address_from_ocr(ocr_text, result.get("storeName") or "")
            if address:
                result["storeAddress"] = address
        lat, lon = geocode(address)
        result["latitude"]  = lat
        result["longitude"] = lon

        log_llm(run_id, image_path.name, user_name, resolved_provider, model,
                total_pt, total_ct, total_cost, latency_ms,
                len(result.get("items", [])), True,
                cost_source=cost_source, latency_source=latency_source)
        return result, total_pt, total_ct, total_cost

    except Exception as e:
        latency_ms = (time.monotonic() - start) * 1000
        log_llm(run_id, image_path.name, user_name, resolved_provider, model,
                0, 0, 0.0, latency_ms, 0, False, str(e), latency_source="local")
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
            AZURE_MAPS_URL
            + f"?api-version=1.0&subscription-key={AZURE_MAPS_KEY}"
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
        usage: dict          # {input_tokens, output_tokens, total_tokens} — read by Railtracks/AgentHub
        cost:  dict          # {total_usd} — read by Railtracks/AgentHub

    # ---------------------------------------------------------------------------
    # Multi-step nodes (commented out — pipeline runs as a single node below)
    # ---------------------------------------------------------------------------
    # @rt.function_node
    # async def ocr_node(ctx: OcrInput) -> OcrOutput:
    #     """Step 1: Azure Document Intelligence OCR."""
    #     image_path = Path(ctx.image_path)
    #     await rt.broadcast(f"[OCR] Starting — {image_path.name} ({image_path.stat().st_size:,} bytes)")
    #     start = time.monotonic()
    #     ocr_text = await asyncio.to_thread(ocr, image_path, ctx.run_id, ctx.user_name)
    #     latency_ms = (time.monotonic() - start) * 1000
    #     await rt.broadcast(f"[OCR] Done — {len(ocr_text)} chars extracted in {latency_ms:.0f}ms")
    #     return OcrOutput(**ctx.model_dump(), ocr_text=ocr_text)
    #
    # @rt.function_node
    # async def structure_node(ctx: OcrOutput) -> StructureOutput:
    #     """Step 2: LLM structures OCR text into JSON + Azure Maps geocoding."""
    #     await rt.broadcast(
    #         f"[LLM] Starting — provider={ctx.provider}  model={ctx.model}  "
    #         f"input={len(ctx.ocr_text)} chars"
    #     )
    #     start = time.monotonic()
    #     result, total_pt, total_ct, total_cost = await asyncio.to_thread(
    #         structure, ctx.ocr_text, Path(ctx.image_path), ctx.user_name,
    #         ctx.model, ctx.run_id, ctx.provider
    #     )
    #     latency_ms = (time.monotonic() - start) * 1000
    #     items = len(result.get("items", []))
    #     store = result.get("storeName", "?")
    #     lat   = result.get("latitude")
    #     lon   = result.get("longitude")
    #     await rt.broadcast(
    #         f"[LLM] Done — {items} items  store={store}  lat={lat}  lon={lon}  latency={latency_ms:.0f}ms"
    #     )
    #     return StructureOutput(
    #         **ctx.model_dump(), store_name=store, items_count=items,
    #         receipt_json=json.dumps(result),
    #         usage={"input_tokens": total_pt, "output_tokens": total_ct, "total_tokens": total_pt + total_ct},
    #         cost={"total_usd": round(total_cost, 8)},
    #     )
    # ---------------------------------------------------------------------------

    @rt.function_node
    async def receipt_pipeline(ctx: OcrInput) -> StructureOutput:
        """Single-node pipeline: OCR → LLM/geocode in one Railtracks step."""
        image_path = Path(ctx.image_path)
        await rt.broadcast(f"[OCR] Starting — {image_path.name} ({image_path.stat().st_size:,} bytes)")
        ocr_text = await asyncio.to_thread(ocr, image_path, ctx.run_id, ctx.user_name)
        await rt.broadcast(
            f"[LLM] Starting — provider={ctx.provider}  model={ctx.model}  "
            f"input={len(ocr_text)} chars"
        )
        result, total_pt, total_ct, total_cost = await asyncio.to_thread(
            structure, ocr_text, image_path, ctx.user_name,
            ctx.model, ctx.run_id, ctx.provider
        )
        items = len(result.get("items", []))
        store = result.get("storeName", "?")
        await rt.broadcast(
            f"[LLM] Done — {items} items  store={store}  latency logged"
        )
        return StructureOutput(
            **ctx.model_dump(),
            ocr_text=ocr_text,
            store_name=store,
            items_count=items,
            receipt_json=json.dumps(result),
            usage={"input_tokens": total_pt, "output_tokens": total_ct, "total_tokens": total_pt + total_ct},
            cost={"total_usd": round(total_cost, 8)},
        )


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
        result, _, _, _ = structure(ocr_text, image_path, user_name, model, run_id, provider=resolved_provider)
        return result


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

    # Initialize per request with access token — no login() call (Yue's new auth flow).
    # Token source priority: GYD_ACCESS_TOKEN env var > auto-loaded from ~/.GYD_SDK/env.yaml
    # (stored there by `gatherYourDeals login` + `gatherYourDeals show-token`).
    client = GYDClient(GYD_SERVER_URL, auto_persist_tokens=False)
    if GYD_ACCESS_TOKEN:
        client._transport.set_tokens(GYD_ACCESS_TOKEN, "")

    items   = receipt.get("items", [])
    created, failed = [], 0
    start   = time.monotonic()

    # Registry key for this image — used to track uploaded IDs for later deletion.
    # The registry is a local replica of what's in Yue's database; writes happen
    # per item so a mid-upload crash doesn't lose IDs that did land in the DB.
    image_name  = receipt.get("imageName", "")
    image_stem  = Path(image_name).stem or image_name

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
            # Write after each successful create — partial crashes still leave
            # a usable registry entry for whatever items did land in the DB.
            _registry_save(image_stem, [x.id for x in created])
        except Exception as e:
            print(f"    [WARN] upload failed for '{item.get('productName')}': {e}")
            failed += 1

    latency_ms = (time.monotonic() - start) * 1000
    log_upload(run_id, image_name, receipt.get("userName", ""),
               len(items), len(created), failed, latency_ms,
               failed == 0, f"{failed} items failed" if failed else None)

    return created


# ---------------------------------------------------------------------------
# Upload registry — maps image stem → list of GYD receipt UUIDs
# ---------------------------------------------------------------------------

def _registry_load() -> dict:
    if _UPLOAD_REGISTRY.exists():
        try:
            return json.loads(_UPLOAD_REGISTRY.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _registry_save(image_stem: str, ids: list[str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = _registry_load()
    registry[image_stem] = ids
    _UPLOAD_REGISTRY.write_text(json.dumps(registry, indent=2, ensure_ascii=False),
                                encoding="utf-8")


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
    p.add_argument("--user",      default="unknown", help="Username for JSON metadata")
    p.add_argument("--provider",  default=LLM_PROVIDER,
                   choices=["openrouter", "clod"],
                   help="LLM provider (default: LLM_PROVIDER env var)")
    p.add_argument("--model",     default=None,
                   help="Model ID — defaults to OR_DEFAULT_MODEL or CLOD_DEFAULT_MODEL env var")
    p.add_argument("--no-upload",         action="store_true", help="Skip SDK upload")
    p.add_argument("--report",            action="store_true", help="Generate usage report")
    p.add_argument("--compare",           action="store_true",
                   help="Generate per-model comparison table scoped to current Receipts/")
    p.add_argument("--eval",              action="store_true",
                   help="Compare output/ against ground_truth/ and print scores")
    p.add_argument("--baseline-report",   action="store_true",
                   help="Generate structured baseline experiment report")
    args = p.parse_args()

    # Resolve model: CLI flag > .env default for provider
    if args.provider == "clod":
        resolved_model = args.model or CLOD_DEFAULT_MODEL
    else:
        resolved_model = args.model or OR_DEFAULT_MODEL

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

    do_upload = not args.no_upload and bool(GYD_SERVER_URL)
    if not do_upload and not args.no_upload:
        print("[INFO] GYD_SERVER_URL not set — extract-only mode.")

    errors = 0
    for img in images:
        print(f"\n→ {img.name}")
        _start = time.monotonic()
        try:
            data = extract(img, args.user, resolved_model, run_id, provider=args.provider)
            total_ms = (time.monotonic() - _start) * 1000
            log_pipeline(run_id, img.name, args.user, args.provider, resolved_model, total_ms, True)
            rows = flatten_receipt(data)
            model_slug = resolved_model.split("/")[-1].lower()
            provider_out_dir = OUTPUT_DIR / f"{args.provider}-{model_slug}"
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
