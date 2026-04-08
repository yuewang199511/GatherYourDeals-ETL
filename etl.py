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
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
import os

load_dotenv()

from reporting import eval_receipts, baseline_report  # noqa: E402 (after load_dotenv)
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
AZURE_MAPS_URL      = os.getenv("AZURE_MAPS_URL",      "https://atlas.microsoft.com/search/fuzzy/json")

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
OCR_CACHE_DIR    = Path("ocr_cache")

# Registry that maps image stem → list of GYD receipt UUIDs created on upload.
# Used by delete_uploaded() to find and remove records from the database.
_UPLOAD_REGISTRY = OUTPUT_DIR / ".upload_registry.json"


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".tiff", ".tif", ".bmp"}

# When set to a Path, debug mode writes intermediate pipeline files there.
# Set via --debug CLI flag; stays None in normal operation.
DEBUG_DIR: Path | None = None

def _dbg(stem: str, stage: str, text: str) -> None:
    """Write one debug file if DEBUG_DIR is set. No-op otherwise."""
    if DEBUG_DIR is None:
        return
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    (DEBUG_DIR / f"{stem}.{stage}").write_text(text, encoding="utf-8")
    print(f"  [DBG]  saved debug/{stem}.{stage}")

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
- [S] marks a store savings/discount row — its [R] or [C] value is a discount applied to the item in the [L] row immediately above it. Subtract it to get the final charged price (e.g. [R] 2.49 then [S][C] 0.50 → final price 1.99). Do NOT create a separate item for [S] rows. [S] rows always have raw_amount = null.
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

{
  "storeName":     string | null,
  "storeAddress":  string | null,
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

- Include only purchasable product line items. Skip ALL of the following:
  - Deposits, recycling fees, donations, bag fees, CA Redemption Value / CRV lines. Note: recycling fees may appear as OCR variants like "RECYCLING FEL" — skip these regardless of OCR quality.
  - Tax, subtotal, total, payment method, change-due, and savings/discount summary lines.
  - Payment terminal / card slip data: card numbers, transaction IDs, reference numbers (Ref. #), auto numbers (Auto #), approval codes ("APPROVED", "DECLINED"), EMV application identifiers (strings starting with A000...), "CUSTOMER COPY", "RETAIN THIS COPY", "DateTime", "Visa Credit", "Debit Card".
  - Barcodes and item codes: numeric-only strings or alphanumeric codes ending in F, H, N, or X that appear on their own line adjacent to a product name (e.g. "007874237159 F", "003700071650H", "068113102405H"). These are inventory/tax codes — never extract them as a productName. If a barcode line has a price in [C], that price belongs to the product name on the adjacent line above or below.
  - Promotional and footer text: survey URLs (www., .com), loyalty point promotions ("EARN X FUEL POINTS", "FUEL POINTS"), thank-you messages, employee/manager contact lines (MGR:), job listings, store website addresses, feedback solicitations.
  - Sale/discount modifier lines: lines like "(SALE)", "MEMBER SAVINGS", "INSTANT SAVINGS", "DIGITAL COUPON", "PRICE REDUCTION" that appear below an item and modify its price — do not extract these as separate items.
  - Department/section header lines: lines that are purely a category label such as "GROCERY", "PRODUCE", "REFRIG/FROZEN", "MISCELLANEOUS", or numeric department codes like "22-DAIRY", "27-PRODUCE", "31-MEATS", "33-BAKERY INSTORE". These divide the receipt into sections — they are not products.
  - OCR noise and unreadable text: garbled strings that are clearly not product names (e.g. "lonipito diiv", "ug lo zyob", "euoju", "Imt 6", "SC 3547 A", "SC 3547"). If a token does not resemble a real product name, skip it.
  - Never create placeholder items with generic names like "Item", "Food", "Product", "Unknown", or numbered variants like "Item 1". If you cannot identify a clear product name from the spatial layout, skip that row entirely.
- Duplicate purchases: if the same product appears on multiple separate line item rows (the customer bought it more than once), extract it once per row — do NOT deduplicate. Each line is a separate purchase transaction. Example: three separate rows for "BONDUELLE BISTRO" → extract three items, not one.
- storeName must be the store's retail chain brand as printed in the receipt header (e.g. "Target", "Costco Wholesale", "Your Independent Grocer", "T&T Supermarket"). Never use a city, neighborhood, district, or branch location name (e.g. "Anaheim Hills") — those appear in the address block, not as the store brand.
- purchaseDate must be the transaction date, formatted YYYY.MM.DD — always dots as separators, never hyphens or slashes (e.g. "2026-03-08" → "2026.03.08"). If the year looks implausible (before 2021) you are likely reading a receipt number, time, or barcode — re-examine the receipt for the correct date. When multiple dates appear, prefer the date immediately paired with a transaction time (HH:MM or HH:MM:SS) — that pairing identifies the actual transaction timestamp. Dates that appear in promotional or contest text (after words like "through", "until", "expires", "ending", "by", "enter by") are promotion deadlines, not the transaction date — skip them. When a receipt shows both a 2-digit year date (e.g. "02/10/20") and an explicit written date with a 4-digit year (e.g. "Feb 10 2026"), always prefer the 4-digit year date.
- price is the total amount charged for that line item as shown in the right-hand price column. Do not use per-unit rates, per-oz prices, or any divided amount. Always include the currency code: "4.79USD" not "4.79". When a receipt prints two price values per line (e.g. a "Price" column and a "You Pay" or "Sale" column), use the amount actually charged — the lower value or the one marked with S, "You Pay", or "Sale". Example: `[R] 4.99  [R] 4.79 S` → use 4.79 (the S-marked charged price). Never assign the receipt subtotal, tax line, balance, or total as the price of any item — those appear after all line items. Exception — [C] column prices: if an item has a value in [C] but no [R] value, and the receipt consistently shows no right-hand price column at all, treat the [C] value as the price.
- amount is a count or weight only — never a price. Preserve recognized unit suffixes (lb, lbs, kg, oz, g); strip only truly unrecognised codes. Examples: "1.160kg", "4lb", "2" are valid amounts. Important edge cases:
  - Tax/category flag codes that appear after prices (e.g. F, N, X, O, T, or a trailing "0") are tax indicators, not quantities — they are not the amount. Set amount=1 when no explicit count or weight appears for that item.
  - Percentage values embedded in product names are product specifications, not amounts (e.g. "HMGZD MILK 3.25%" — the "3.25%" is the fat content, amount=1).
  - Short qualifier abbreviations appended to product names (e.g. WLD, IQF, MRJ, RQ, FV) are product descriptors, not amounts — amount=1.
  - If the product name begins with a weight or count prefix (e.g. "4LB Honeycrisp Apples", "3-Pack Paper Towels"), extract that prefix as the amount (e.g. "4lb", "3").
- Weight-priced items: some receipts print "1.160 kg @ $1.72/kg  2.00" — set amount to the weight ("1.160kg") and price to the total charged ("2.00CAD"), not the per-kg rate. The weight string itself (e.g. "1.160 kg @ $1.72/kg 2.00" or "0.510 kg @ $4.14/kg") is metadata for the item above it — NEVER extract it as a separate productName. It has no productName of its own.
- If a markdown table is present, each row is one line item — do not merge or split rows.
- Column anchoring: each receipt row follows [ITEM NAME] [QTY optional] [PRICE]. Prices are typically right-aligned — do not assign a [L] value as a price, and do not assign a [R] value as a quantity. If no [R] price column exists for any row in the receipt, a [C] value may be the price (see price rule above).

## Handling ambiguity

- If multiple candidate values exist for the same field (e.g. two dates, two totals), choose the most recent or most prominent one.
- If confidence in a field value is low — the text is unclear, partially obscured, or contradictory — return null rather than guessing.
- Do not fabricate or infer values that are not explicitly present in the OCR text. If a field is not visible, output null."""

# Leaner prompt — no chain-of-thought scaffolding.
# Used for simple (single-chunk, spatial-layout) receipts to cut output tokens
# by ~70%.  Identical rules to _SYSTEM_PROMPT; only the <spans>/<extract>/<json>
# thinking scaffold is removed.
_COT_SECTION = (
    "\n## Extraction process\n\n"
    "Work through the receipt in three steps. Show your work in the tagged sections "
    "below, then output the final JSON. This reduces errors on messy receipts.\n\n"
    "<spans>\nQuote verbatim from the OCR text:\n"
    "- HEADER: the store name, address, and date/time block\n"
    "- ITEMS: every product line item row exactly as printed\n"
    "- TOTALS: subtotal, tax, total, and payment lines\n"
    "</spans>\n\n"
    "<extract>\nUsing the SPATIAL LAYOUT section (if present), reconstruct receipt rows first:\n"
    "- Each row follows: [L] ITEM NAME  [C] QTY (optional)  [R] PRICE\n"
    "- Prices are right-aligned ([R] column) in most receipts — do not assign [L] values as price. Exception: if no [R] price column exists at all and an item has only a [C] value, that [C] value is the price.\n"
    "- When a line shows two [R] prices (regular + sale/You Pay), use the charged amount — the one marked S, 'You Pay', or 'Sale', typically the lower value.\n"
    "- Quantities / weights are center-aligned ([C]) or follow the item name in [L]\n"
    "- [S] marks a store savings/discount row — its [R] or [C] value is a discount applied to the item "
    "in the [L] row immediately above it. Subtract it to get the final charged price "
    "(e.g. [R] 2.49 then [S][C] 0.50 → final price 1.99). Do NOT create a separate item for [S] rows. "
    "[S] rows always have raw_amount = null.\n"
    "- If no SPATIAL LAYOUT, use the markdown table rows\n\n"
    "Then list raw extracted values:\n"
    "- date: <raw date string>\n"
    "- time: <raw time string>\n"
    "- currency: <symbol or code found>\n"
    "- For each item row: productName | itemCode | raw_price | raw_amount | category\n"
    "</extract>\n\n"
    "<json>\n{final normalized JSON conforming to the schema below}\n</json>\n\n"
)
_SYSTEM_PROMPT_DIRECT = _SYSTEM_PROMPT.replace(_COT_SECTION, "")

_COSTCO_PROMPT_ADDENDUM = """\

## Costco receipt rules

- Item numbers: Costco prints a numeric item code before or after the product name (e.g. "47825 GREEN GRAPES" or "GREEN GRAPES 47825"). Put that number in `itemCode` and remove it from `productName`.
- Price letter suffixes: Costco appends a single letter to prices to indicate tax category (e.g. "8.99A", "11.99N", "26.09E"). Strip the trailing letter — `price` must be the numeric value plus currency only: "8.99USD".
- CA Redemption Value / REDEMP VA / CRV lines are deposit fees — skip them entirely, do not include as items. Critically: the dollar amount on a CA REDEMP VA line (e.g. "0.60A", "1.75A") belongs to the fee, NOT to the product printed above it. Never assign a CA REDEMP VA price to a product.
- Abbreviated product names: Costco uses abbreviated names on the receipt (e.g. "KSORGWHLEMLK", "ORG FR EGGS", "HONYCRSP"). Expand these to their full human-readable form using context clues: "KS Organic Whole Milk", "Organic Free Range Eggs", "Honeycrisp Apples", etc.
- Each product appears only once on the receipt. If you see the same item at different points in the text (e.g. once with an item code and once without), output it exactly once.\
"""


def _build_system_prompt(ocr_text: str, use_direct: bool = False) -> str:
    """Return the system prompt, appending Costco rules only when the OCR mentions Costco."""
    base = _SYSTEM_PROMPT_DIRECT if use_direct else _SYSTEM_PROMPT
    if "COSTCO" in ocr_text.upper()[:500]:
        return base + _COSTCO_PROMPT_ADDENDUM
    return base


# ---------------------------------------------------------------------------
# Step 1 — Azure Document Intelligence OCR
# ---------------------------------------------------------------------------
_ADI_MAX_BYTES = 4 * 1024 * 1024  # Azure DI hard limit: 4 MB


def _to_jpeg_bytes(image_data: "Path | bytes", display_name: str = "") -> tuple[bytes, str]:
    """Return (image_bytes, content_type). Converts HEIC → JPEG and downscales images >4 MB.

    Accepts either a Path (read from disk) or raw bytes (in-memory).
    display_name is used for extension detection and error messages when bytes are passed.
    """
    import io

    # Normalise to (raw, ext, img_or_none)
    if isinstance(image_data, Path):
        ext = image_data.suffix.lower()
        name_for_error = image_data.name
        if ext == ".heic":
            try:
                import pillow_heif
                from PIL import Image
                pillow_heif.register_heif_opener()
                img = Image.open(image_data).convert("RGB")
            except ImportError:
                raise ImportError("HEIC support requires: pip install pillow pillow-heif")
            raw = None
        else:
            raw = image_data.read_bytes()
            img = None
    else:
        raw = image_data
        ext = Path(display_name).suffix.lower() if display_name else ".jpg"
        name_for_error = display_name or "image"
        if ext == ".heic":
            try:
                import pillow_heif
                from PIL import Image
                pillow_heif.register_heif_opener()
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            except ImportError:
                raise ImportError("HEIC support requires: pip install pillow pillow-heif")
            raw = None
        else:
            img = None

    if img is None:
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
                f"{name_for_error} is {len(raw) // 1024 // 1024} MB (Azure DI limit: 4 MB). "
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

    Hybrid approach: page.lines for row structure, page.words for intra-line
    column assignment.

    ADI's page.lines correctly groups words into visual rows (row structure),
    but treats each line as one token — so "REYNOLDS WRAP FOIL 0.60" becomes
    a single left-column blob even though "0.60" is in the center column.

    page.words gives each word its own X polygon, so we decompose each ADI
    line into its component words, assign each word to L/C/R by its individual
    X position, then concatenate same-column words within the line into one
    token.  Row grouping still uses page.lines Y positions, so adjacent lines
    ("Your cashier was Jamie" / "REYNOLDS WRAP FOIL") stay separate.

        [L] < 40%  — item description / product name
        [C] 40-70% — quantity / unit
        [R] > 70%  — price
    """
    if not result.pages:
        return ""

    page        = result.pages[0]
    page_width  = max(getattr(page, "width",  0) or 0, 1.0)
    page_height = max(getattr(page, "height", 0) or 0, 1.0)
    tolerance   = page_height * 0.015   # lines within 1.5% of page height → same row

    # ── Build word position maps ─────────────────────────────────────────────
    word_xs:    dict[str, float] = {}   # token → x_left  (for L/C/R column assignment)
    word_ys:    dict[str, float] = {}   # token → y_center (last occurrence; for tilt)
    word_count: dict[str, int]   = {}   # token → total occurrences on this page

    for word in (page.words or []):
        wpoly = getattr(word, "polygon", None)
        wtext = getattr(word, "content", None)
        if not wpoly or not wtext or not wtext.strip():
            continue
        if hasattr(wpoly[0], "x"):
            wxs = [p.x for p in wpoly]
            wys = [p.y for p in wpoly]
        else:
            wxs = list(wpoly[0::2])
            wys = list(wpoly[1::2])
        tok = wtext.strip()
        word_xs[tok]    = min(wxs)
        word_ys[tok]    = sum(wys) / len(wys)
        word_count[tok] = word_count.get(tok, 0) + 1

    # ── Tilt detection ──────────────────────────────────────────────────────
    # ADI often returns axis-aligned word bounding boxes even for tilted
    # receipts, so polygon top-edge slopes are unreliable.  Instead we measure
    # tilt from the Y variation of word centers WITHIN multi-word ADI lines.
    #
    # Pitfall: word_ys stores the LAST occurrence of each token; for repeated
    # tokens (e.g. "F" appears on every food-taxable line, "SC" on every Kroger
    # savings row) the stored Y is wrong for earlier occurrences, which
    # corrupts the slope calculation.  To avoid this, we only include slope
    # samples from lines where ALL tokens are unique across the page.
    #
    # For the Kroger test receipt the estimated tilt is ≈ −0.07 to −0.09
    # (right side ~15 px higher per 200 px horizontal span).
    # If the receipt is straight, all slopes ≈ 0 → tilt = 0 → y_corr = y.
    slope_samples: list[float] = []
    for line in (page.lines or []):
        ltext = getattr(line, "content", None)
        lpoly = getattr(line, "polygon", None)
        if not ltext or not lpoly:
            continue
        tokens = ltext.strip().split()
        if len(tokens) < 2:
            continue
        # Skip noise lines (totals, loyalty text, transaction codes) — they
        # often have different perspective distortion than the item section.
        if _SPATIAL_NOISE_LINE.match(ltext.strip()):
            continue
        # Skip lines containing any token that appears more than once on the
        # page — word_ys for such tokens reflects their last position, not this
        # line's position, which would give a wrong slope.
        if any(word_count.get(t, 0) > 1 for t in tokens):
            continue
        # Only sample from the upper 60% of the receipt — the item section.
        # Footer/ad text at the bottom of receipts often has reduced or reversed
        # perspective tilt, which would drag the median toward zero.
        if hasattr(lpoly[0], "x"):
            line_y_center = sum(p.y for p in lpoly) / len(lpoly)
        else:
            line_y_center = sum(lpoly[1::2]) / (len(lpoly) // 2)
        if line_y_center > page_height * 0.60:
            continue
        xy = [(word_xs[t], word_ys[t]) for t in tokens
              if t in word_xs and t in word_ys]
        if len(xy) < 2:
            continue
        xy.sort(key=lambda p: p[0])                     # sort left → right by X
        x_span = xy[-1][0] - xy[0][0]
        if x_span < page_width * 0.05:                  # need meaningful span (≥5%)
            continue
        slope_samples.append((xy[-1][1] - xy[0][1]) / x_span)

    tilt: float = statistics.median(slope_samples) if slope_samples else 0.0

    def _cy(x: float, y: float) -> float:
        """Return shear-corrected Y: removes the linear tilt across the page."""
        return y - tilt * x

    # ── Build line_data ──────────────────────────────────────────────────────
    # Row structure: page.lines — each ADI line is one visual row.
    # Column assignment: page.words X (word_xs).
    # Y stored in line_data: raw line_y for ALL tokens (not corrected).
    #   • Left tokens  → raw line_y preserves ADI's row separation for grouping
    #   • Center/right → raw line_y; tilt correction is applied ONLY during
    #     assignment to groups (see _best_group below), not in line_data itself.
    #     Applying correction in line_data would shrink the gap between adjacent
    #     rows (e.g. REYNOLDS row ↔ SC KROGER SAVINGS row from 16 px to 8 px),
    #     dropping it below the grouping tolerance and merging them.
    #
    # Column thresholds (fraction of page width):
    #   [L] < 52%  — item description / product name.  52% rather than 40% so
    #                 that long product names whose words spill into the 40–52%
    #                 zone stay in the left column.
    #   [C] 52–70% — quantity / unit price / discount amount
    #   [R] > 70%  — right-column total price
    line_data: list[tuple[float, float, str]] = []   # (raw_line_y, x_left, text)

    for line in (page.lines or []):
        poly = getattr(line, "polygon", None)
        text = getattr(line, "content", None)
        if not poly or not text or not text.strip():
            continue
        if _SPATIAL_NOISE_LINE.match(text.strip()):
            continue
        if hasattr(poly[0], "x"):
            ys        = [p.y for p in poly]
            line_min_x = min(p.x for p in poly)
        else:
            ys        = list(poly[1::2])
            line_min_x = min(poly[0::2])
        line_y = sum(ys) / len(ys)

        left_words:   list[tuple[str, float]] = []
        center_words: list[tuple[str, float]] = []
        right_words:  list[tuple[str, float]] = []

        for token in text.strip().split():
            x   = word_xs.get(token, line_min_x)
            pct = x / page_width
            if pct < 0.52:
                left_words.append((token, x))
            elif pct < 0.70:
                center_words.append((token, x))
            else:
                right_words.append((token, x))

        if left_words:
            line_data.append((line_y, left_words[0][1],
                              " ".join(t[0] for t in left_words)))
        for tok, x in center_words:
            line_data.append((line_y, x, tok))
        if right_words:
            line_data.append((line_y, right_words[0][1],
                              " ".join(t[0] for t in right_words)))

    line_data.sort(key=lambda t: t[0])

    # --- Pass 1: separate tokens by column, group [L] rows by Y proximity ---
    left_tokens:   list[tuple[float, float, str]] = []
    center_tokens: list[tuple[float, float, str]] = []
    right_tokens:  list[tuple[float, float, str]] = []
    for y, x, text in line_data:
        pct = x / page_width
        if pct < 0.52:
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

    # ── Tilt-corrected group assignment ─────────────────────────────────────
    # Assign center and right tokens to [L] groups by comparing tilt-corrected
    # Y values (simple nearest).
    #
    # Why tilt correction here and not in line_data?
    #   Receipt tilt shifts the price column up relative to the item-name column
    #   on the same visual row.  Left groups are built from raw line_y (which
    #   preserves ADI's original row separation), but center/right tokens need
    #   their Y adjusted before comparison so that a price at corrected Y=306
    #   lands on the item at corrected Y=309, not on the cashier header at
    #   corrected Y=267.
    #
    # Corrected Y for a token:    _cy(token_x, raw_line_y)
    # Corrected Y for a group:    _cy(avg_x_of_group, raw_y_of_first_token)
    #   where avg_x = average of stored X values across all entries in the group.
    #   This uses the group's leftmost-word X as a proxy for its horizontal
    #   center, correcting the group Y by approximately the same amount as the
    #   same-row price column.

    def _group_cy(group: list) -> float:
        """Tilt-corrected representative Y for a [L] group.

        Uses the LAST element's (x, y) so that multi-item groups (e.g.
        {KRO CREAMER, IMPR MARGRNE}) expose the bottom item's corrected Y.
        This ensures the group's corrected-Y boundary is >= all prices that
        belong to items inside the group.
        """
        last = group[-1]
        return _cy(last[1], last[0])        # _cy(x, raw_y) of the last element

    def _best_group(raw_y: float, x: float) -> int:
        """Return the index of the best [L] group for a center/right token.

        Strategy: "at or below with margin"
        - A group is a *valid candidate* if its corrected Y is not more than
          `margin` pixels *above* the token's corrected Y
          (i.e. _group_cy(group) >= token_cy - margin).
        - Among valid candidates, prefer the one with the smallest corrected Y
          (the item closest to, but at or below, the price).

        Why margin?  Tilt estimation is imperfect (±5–10 px residual error),
        so the "at or below" boundary is fuzzy.  A small margin prevents
        near-correct assignments from flipping to the wrong row.

        Fallback: if no group qualifies (e.g. token above all items), use the
        topmost group (smallest corrected Y).
        """
        corr_y = _cy(x, raw_y)
        margin = tolerance * 0.5            # ~ half an inter-row spacing
        ranked = sorted(range(len(left_groups)),
                        key=lambda i: _group_cy(left_groups[i]))
        # Filter to valid candidates
        valid = [(i, _group_cy(left_groups[i])) for i in ranked
                 if _group_cy(left_groups[i]) >= corr_y - margin]
        if valid:
            # Among valid candidates, pick the one with smallest corrected Y
            # (the item closest to and just at-or-below the price)
            return min(valid, key=lambda ic: ic[1])[0]
        # Fallback — token above all groups; assign to the topmost group
        return ranked[0]

    # --- Pass 2: assign [R] tokens to [L] groups ---
    group_rights: list[list[tuple[float, float, str]]] = [[] for _ in left_groups]
    for r_y, r_x, r_text in right_tokens:
        group_rights[_best_group(r_y, r_x)].append((r_y, r_x, r_text))

    # Preliminary center assignment — needed before the continuation merge so we
    # can detect groups with a center-column price.  On Kroger-style receipts
    # item prices land in [C] (52-70% X) with F/T suffixes, not [R] (>70% X).
    # Without this, the continuation merge sees r_tokens=[] for every item and
    # collapses the entire item section into one cascade row.
    prelim_centers: list[list[tuple[float, float, str]]] = [[] for _ in left_groups]
    for c_y, c_x, c_text in center_tokens:
        prelim_centers[_best_group(c_y, c_x)].append((c_y, c_x, c_text))

    # --- Pass 3: merge continuation lines into their parent row ---
    # When a product name wraps across two OCR lines they land in separate [L]
    # groups (Y gap > tolerance).  If the second group has no price in ANY column
    # (R or C) it is almost certainly a continuation of the line above.
    continuation_threshold = tolerance * 3   # ~4.5% of page height ≈ 2-3 line heights
    merged_groups:  list[list[tuple[float, float, str]]] = []
    merged_rights:  list[list[tuple[float, float, str]]] = []
    merged_prelim:  list[list[tuple[float, float, str]]] = []

    for i, group in enumerate(left_groups):
        r_tokens = group_rights[i]
        c_tokens = prelim_centers[i]
        has_price = bool(r_tokens or c_tokens)           # price in R or C column
        parent_has_price = bool(
            merged_rights and (merged_rights[-1] or merged_prelim[-1])
        )
        if (merged_groups
                and not has_price                               # no price on this line
                and not parent_has_price                        # parent also has no price yet
                and (group[0][0] - merged_groups[-1][0][0])    # Y gap from parent
                    <= continuation_threshold):
            # Continuation line — absorb into the previous group
            merged_groups[-1].extend(group)
        else:
            merged_groups.append(group)
            merged_rights.append(r_tokens)
            merged_prelim.append(c_tokens)

    # Re-derive group_ys from merged groups for [C] assignment below
    group_ys     = [g[0][0] for g in merged_groups]
    left_groups  = merged_groups
    group_rights = merged_rights
    # (merged_prelim is discarded — final center assignment below uses updated group_ys)

    # Assign [C] tokens to [L] groups — tilt-corrected nearest, same as Pass 2.
    group_centers: list[list[tuple[float, float, str]]] = [[] for _ in left_groups]
    for c_y, c_x, c_text in center_tokens:
        group_centers[_best_group(c_y, c_x)].append((c_y, c_x, c_text))

    # --- Render rows ---
    # Each token is labeled with its column (L/C/R/S).
    output_lines: list[str] = []
    for i, group in enumerate(left_groups):
        group_text = " ".join(t[2] for t in group)
        col_label  = "S" if _SAVINGS_LINE.search(group_text) else "L"

        if len(group) > 1 and col_label == "L":
            # Multiple [L] items landed in the same Y-band (adjacent lines printed
            # very close together).  Render each on its own row so the LLM sees
            # them as separate products, not a name + item-code pair.
            # [C] tokens are distributed to whichever [L] is nearest by Y.
            # [R] tokens (price column) go to the last [L] item in the group.
            sorted_left = sorted(group, key=lambda t: t[0])
            c_tokens    = sorted(group_centers[i], key=lambda t: t[0])
            r_tokens    = sorted(group_rights[i],  key=lambda t: t[1])
            for j, (l_y, l_x, l_text) in enumerate(sorted_left):
                parts: list[str] = [f"[L] {l_text}"]
                for c_y, c_x, c_text in c_tokens:
                    # Use tilt-corrected Y for within-group assignment so that
                    # prices that appear above their item in raw Y (due to tilt)
                    # land on the correct item rather than the one above it.
                    c_corr = _cy(c_x, c_y)
                    nearest = min(range(len(sorted_left)),
                                  key=lambda k: abs(_cy(sorted_left[k][1],
                                                        sorted_left[k][0]) - c_corr))
                    if nearest == j:
                        parts.append(f"[C] {c_text}")
                if j == len(sorted_left) - 1:
                    for _, _, r_text in r_tokens:
                        parts.append(f"[R] {r_text}")
                output_lines.append("  |  ".join(parts))
        else:
            parts = []
            lc = [(t, col_label) for t in group] + [(t, "C") for t in group_centers[i]]
            lc.sort(key=lambda tc: tc[0][1])
            for t, col in lc:
                parts.append(f"[{col}] {t[2]}")
            for r_y, _, text in sorted(group_rights[i], key=lambda t: t[1]):
                parts.append(f"[R] {text}")
            if parts:
                output_lines.append("  |  ".join(parts))

    return "\n".join(output_lines)


def ocr(image_data: "Path | bytes", display_name: str, run_id: str, user_id: str = "", use_cache: bool = True) -> str:
    """
    Send image to Azure Document Intelligence (prebuilt-read).
    Returns markdown OCR text with a spatial layout section appended.

    Accepts either a Path (read from disk) or raw bytes (in-memory).
    display_name is used for cache keys, logging, and error messages.

    When use_cache=True (default), the OCR result is written to
    ocr_cache/<stem>.txt after the first successful call and loaded from there
    on all subsequent calls — skipping the ADI network call entirely.  Use
    --no-ocr-cache to force a fresh ADI call (e.g. after changing the spatial
    reconstruction logic).
    """
    cache_stem = Path(display_name).stem if display_name else "unknown"
    if use_cache:
        cache_file = OCR_CACHE_DIR / (cache_stem + ".txt")
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8")

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

    image_bytes, content_type = _to_jpeg_bytes(image_data, display_name)
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
        # Reconstruct text from page.lines instead of result.content.
        # result.content applies ADI's own merge heuristic, which can concatenate
        # two visually separate receipt rows (e.g. "Your cashier was Jamie
        # REYNOLDS WRAP FOIL 0.60") into one line.  page.lines preserves the
        # per-line bounding boxes ADI detected, giving correct line breaks.
        line_texts: list[str] = []
        for _page in (result.pages or []):
            for _line in (_page.lines or []):
                if getattr(_line, "content", None):
                    line_texts.append(_line.content)
        markdown = "\n".join(line_texts) if line_texts else (result.content or "")

        spatial  = _reconstruct_spatial_rows(result)
        ocr_text = (
            markdown
            + "\n\n---\n## SPATIAL LAYOUT\n"
            + "Each token labeled [COL] where COL=L/C/R/S (L=item name, C=center/price, R=right-col price, S=savings/discount row).\n"
            + "Use this section to extract items — preserves column alignment.\n\n"
            + spatial
        ) if spatial else markdown

        log_adi(run_id, display_name, user_id, image_size_bytes,
                pages, pages * ADI_COST_PER_PAGE, latency_ms, True,
                chars_extracted=len(markdown))

        if use_cache:
            OCR_CACHE_DIR.mkdir(exist_ok=True)
            (OCR_CACHE_DIR / (cache_stem + ".txt")).write_text(
                ocr_text, encoding="utf-8"
            )

        return ocr_text

    except Exception as e:
        latency_ms = (time.monotonic() - start) * 1000
        log_adi(run_id, display_name, user_id, image_size_bytes,
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


def _structure_openrouter(ocr_text: str, model: str, system_prompt: str = _SYSTEM_PROMPT) -> tuple[dict, int, int, str]:
    """Call OpenRouter and return (parsed_dict, prompt_tokens, completion_tokens, generation_id)."""
    if not OPENROUTER_API_KEY:
        raise EnvironmentError("OPENROUTER_API_KEY not set — add it to .env")

    from openai import OpenAI
    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL, timeout=90)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Receipt OCR text:\n\n{ocr_text}"},
        ],
        temperature=0,
        max_tokens=4096,
    )
    usage = resp.usage
    pt = usage.prompt_tokens     if usage else 0
    ct = usage.completion_tokens if usage else 0
    raw = resp.choices[0].message.content
    return _parse_llm_json(raw), pt, ct, resp.id or "", raw


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
    "anthropic/claude-haiku-4.5":    (1.00,  5.00),  # $/M input, $/M output
    "qwen/qwen-2.5-7b-instruct":     (0.04,  0.10),
    "google/gemini-flash-1.5":       (0.075, 0.30),  # repair escalation model
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


def _structure_clod(ocr_text: str, model: str, system_prompt: str = _SYSTEM_PROMPT) -> tuple[dict, int, int, float | None]:
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
            {"role": "system", "content": system_prompt},
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
        content,
    )


# ---------------------------------------------------------------------------
# Long-receipt chunking
# ---------------------------------------------------------------------------
# LLM attention degrades when OCR text is long and noisy: rows bleed into each
# other, items get merged or dropped, and column alignment is lost.  Receipts
# over _CHUNK_THRESHOLD_CHARS are split into overlapping vertical sections
# before being sent to the LLM.  Results are merged after all chunks return.

_CHUNK_THRESHOLD_CHARS = 2000   # raised from 1000 — most receipts fit in one chunk
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


# ---------------------------------------------------------------------------
# Tier 1 — OCR noise filter
# ---------------------------------------------------------------------------
# Strips lines that are almost certainly not product items before the text
# reaches the LLM.  This reduces input tokens (improving speed and cost) and
# prevents the model from confusing totals/tax/payment lines with item rows.

_NOISE_LINE = re.compile(
    r"^\s*(?:"
    r"sub\s*total|subtotal|total|net\s*total|grand\s*total|"
    r"hst|gst|pst|qst|vat|tax|surcharge|"
    r"payment|cash|credit|debit|visa|mastercard|interac|amex|"
    r"change\s*due|balance\s*due|amount\s*due|amount\s*tendered|"
    r"savings?|you\s*saved|instant\s*savings?|member\s*savings?|everyday\s*savings?|"
    r"discount|coupon|points?|rewards?|loyalty|"
    r"thank\s*you|please\s*come|visit\s*us|survey|"
    r"receipt\s*#|store\s*#|ref\s*#|trans\s*#|auth\s*#|approval|"
    r"approved|declined|pin\s*verified|customer\s*copy|merchant\s*copy|"
    r"crv|ca\s*redemp|deposit|bottle\s*dep|bag\s*fee|"
    r"cashier|operator|terminal|"
    r"\*{2,}|={3,}|-{3,}|#{3,}"
    r")\b",
    re.IGNORECASE,
)

# Spatial-layout noise filter — same as _NOISE_LINE but intentionally keeps
# savings/discount lines so the LLM can compute the final discounted price.
_SPATIAL_NOISE_LINE = re.compile(
    r"^\s*(?:"
    r"sub\s*total|subtotal|total|net\s*total|grand\s*total|"
    r"hst|gst|pst|qst|vat|tax|surcharge|"
    r"payment|cash|credit|debit|visa|mastercard|interac|amex|"
    r"us\s+debit|us\s+credit|"                          # "US DEBIT Purchase" etc.
    r"change\s*due|change\b|balance\s*due|balance\b|"   # bare CHANGE / BALANCE lines
    r"amount\s*due|amount\s*tendered|"
    r"purchase\s*:|purchase\b|"                         # "PURCHASE: 9.06"
    r"verified|pin\s*verified|"                         # "VERIFIED BY PIN"
    r"aid\s*:|tc\s*:|ref\s*#|trans\s*#|auth\s*#|approval|"  # transaction codes
    r"thank\s*you|please\s*come|visit\s*us|survey|"
    r"tell\s*us|earn\b|fuel\s*point|fuel\b|"            # loyalty program footer
    r"remaining\b.*point|total\b.*point|"               # "Remaining May Fuel Points"
    r"annual\s*card|you\s*saved|with\s*our|"            # savings summary footer
    r"go\s*to\s*www|www\.|feedback|hiring|"             # URLs / HR footer
    r"receipt\s*#|store\s*#|"
    r"approved|declined|customer\s*copy|merchant\s*copy|"
    r"crv|ca\s*redemp|deposit|bottle\s*dep|bag\s*fee|"
    r"your\s+cashier|cashier|operator|terminal|"        # "Your cashier was Jamie" etc.
    r"\*{4,}|={3,}|-{3,}|#{3,}"                        # symbol-only lines (\b removed — \W next to \W has no boundary)
    r")(?:\b|$|\s)",                                    # word boundary OR end OR whitespace
    re.IGNORECASE,
)

# Identifies savings/discount lines in the spatial layout so they can be
# labeled [S] and associated with the item above them.
_SAVINGS_LINE = re.compile(
    r"\b(savings?|you\s*saved|instant\s*savings?|member\s*savings?|"
    r"everyday\s*savings?|digital\s*coupon|coupon\s*savings?|discount)\b",
    re.IGNORECASE,
)


def _filter_noise_lines(ocr_text: str) -> str:
    """
    Remove non-item lines from OCR text before sending to the LLM.

    Keeps the spatial layout section intact (noise filtering only applies to
    the raw OCR body).  Lines matching payment info, totals, tax, loyalty
    points, and decorative separators are dropped.

    This reduces token count on large receipts (e.g. NoFrills 16-item receipt:
    6985 → ~3500 input tokens) and prevents the model from treating total/tax
    lines as product items.
    """
    _SPATIAL_MARKER = "\n---\n## SPATIAL LAYOUT\n"
    if _SPATIAL_MARKER in ocr_text:
        raw_part, spatial_part = ocr_text.split(_SPATIAL_MARKER, 1)
        filtered_raw = "\n".join(
            line for line in raw_part.splitlines()
            if not _NOISE_LINE.match(line)
        )
        return filtered_raw + _SPATIAL_MARKER + spatial_part
    return "\n".join(
        line for line in ocr_text.splitlines()
        if not _NOISE_LINE.match(line)
    )


# ---------------------------------------------------------------------------
# Tier 3b — Targeted repair for items with null/missing price
# ---------------------------------------------------------------------------
# When primary extraction leaves items with null price, re-query the LLM with
# only the 5-line OCR window around that item — much cheaper than reprocessing
# the full receipt (~50 tokens vs 6985).  If the repair still fails, escalate
# to a stronger model via OpenRouter.

_REPAIR_ESCALATION_MODEL = "google/gemini-flash-1.5"   # cheap + strong structured extraction


def _find_ocr_context(product_name: str, ocr_text: str, window: int = 3) -> str:
    """
    Return the OCR lines surrounding the first occurrence of product_name.
    Searches case-insensitively using the first significant word of the name.
    Returns up to (2*window + 1) lines, or the full text if name not found.
    """
    keyword = product_name.split()[0].upper() if product_name else ""
    lines = ocr_text.splitlines()
    for i, line in enumerate(lines):
        if keyword and keyword in line.upper():
            start = max(0, i - window)
            end   = min(len(lines), i + window + 1)
            return "\n".join(lines[start:end])
    return ocr_text[:500]   # fallback: first 500 chars if name not found


def _repair_failed_items(
    items: list[dict],
    ocr_text: str,
    model: str,
    provider: str,
    currency: str,
) -> list[dict]:
    """
    Tier 3b + Tier 4: targeted re-extraction for items with null price.

    For each item where price is None:
      1. Extract a 7-line OCR window around the product name
      2. Ask the primary model (same provider/model) for just the price
      3. If still null, escalate to OpenRouter google/gemini-flash-1.5

    Items that cannot be repaired are dropped (null price = invalid upload).
    Items with a valid price are updated in-place and kept.
    """
    import httpx
    from openai import OpenAI

    repaired: list[dict] = []
    for item in items:
        price = (item.get("price") or "").strip()
        if price and price.lower() != "null":
            repaired.append(item)
            continue

        name    = item.get("productName", "unknown item")
        context = _find_ocr_context(name, ocr_text)
        prompt  = (
            f'From this receipt section, extract the price of "{name}".\n'
            f"Return only the numeric price (e.g. 3.49). No currency symbol, no explanation.\n\n"
            f"Receipt section:\n{context}"
        )

        found_price: str | None = None

        # --- Tier 3b: retry with same model ---
        try:
            if provider == "clod" and CLOD_API_KEY:
                resp = httpx.post(
                    CLOD_API_URL,
                    headers={"Authorization": f"Bearer {CLOD_API_KEY}", "Content-Type": "application/json"},
                    json={"model": model, "messages": [{"role": "user", "content": prompt}]},
                    timeout=30,
                )
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"].strip()
            elif OPENROUTER_API_KEY:
                client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0, max_tokens=16,
                )
                raw = resp.choices[0].message.content.strip()
            else:
                raw = ""

            m = re.search(r"\d+\.?\d*", raw)
            if m:
                found_price = f"{float(m.group()):.2f}{currency}"
        except Exception:
            pass

        # --- Tier 4: escalate to gemini-flash if still null ---
        if not found_price and OPENROUTER_API_KEY:
            try:
                client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)
                resp = client.chat.completions.create(
                    model=_REPAIR_ESCALATION_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0, max_tokens=16,
                )
                raw = resp.choices[0].message.content.strip()
                m = re.search(r"\d+\.?\d*", raw)
                if m:
                    found_price = f"{float(m.group()):.2f}{currency}"
            except Exception:
                pass

        if found_price:
            item["price"] = found_price
            repaired.append(item)
        # else: drop the item — null price cannot be uploaded

    return repaired


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
    _UNIT_STRIP     = re.compile(r"\b(W|EA|PK|F|PC|CT|BG|LT|BT|CN|OZ|MRJ|each)\b", re.IGNORECASE)
    _PRICE_FMT      = re.compile(r"^\d+\.\d{2}\s*[A-Za-z]?$")  # "4.79", "4.79S", "4.79 S"
    _ITEM_CODE      = re.compile(r"^\d{4,}\s+")            # leading 4+ digit item/barcode code
    _PRICE_LETTER   = re.compile(r"^(\d+\.?\d*)[A-Za-z]+$")  # "11.99A", "8.99N"
    _NON_PRODUCT    = re.compile(
        r"\b(tax|saving|savings|discount|instant\s+saving|subtotal|total|"
        r"redemp|crv|deposit|donation|bag\s+fee|"
        # Payment terminal / card slip lines
        r"approved|customer\s+copy|card\s+number|retain\s+this|"
        r"ref\.?\s*#|auto\s*#|visa\s+credit|entry\s+id|datetime|"
        r"transaction\s+id|debit\s+card|credit\s+card|"
        # Promotional / footer text
        r"fuel\s+points|thank\s+you\s+for\s+shopping|earn\s+\d+|"
        r"opportunity\s+awaits|join\s+our\s+team|feedback|"
        r"closing\s+balance|points\s+redeemed|pc\s+optimum|"
        # Sale/discount modifier lines (not standalone products)
        r"member\s+saving|digital\s+coupon|coupon\s+saving|"
        r"instant\s+saving|price\s+reduction)\b",
        re.IGNORECASE,
    )
    # Structural junk: URLs, EMV AIDs (A000...), "Item N" placeholders,
    # approval code lines ("00 APPROVED"), standalone short codes ("SC")
    _JUNK_NAME = re.compile(
        r"(www\.|\.com\b|\.org\b|jobs\.|"          # URLs
        r"^[Aa][0-9a-fA-F]{8,}|"                   # EMV AID e.g. A0000000031010
        r"^\d{2,3}\s+[A-Z]{2,}|"                   # "00 APPROVED", "03 DECLINED"
        r"^item\s+\d+$|"                            # "Item 1", "Item 2"
        r"^\*{2,}|"                                 # "*** CUSTOMER COPY ***"
        r"^mgr:|^date:|^time:|"                     # manager/timestamp footer fields
        r"^\(sale\)\s*$|"                           # bare "(SALE)" line
        r"^\(?\d{3}\)?[\s\-]\d{3}[\s\-]\d{4}|"     # phone numbers (604) 688-0911
        r"^\d{2}/\d{2}/\d{2,4}\s+\d{1,2}:\d{2}|"  # timestamps 02/19/26 7:53:13 PM
        r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})",      # ISO timestamps 2026-02-19 07:53
        re.IGNORECASE,
    )
    # Sale-modifier prefix — remove "(SALE)" prefix from duplicated item names
    _SALE_PREFIX = re.compile(r"^\(sale\)\s+", re.IGNORECASE)

    _MAX_ITEM_PRICE = 99.0   # prices above this are almost certainly totals/subtotals
    _MIN_ITEM_PRICE = 0.50   # prices below this are almost certainly CA CRV / deposit fees bleeding in

    fixed: list[dict] = []
    for item in items:
        name = (item.get("productName") or "").strip()
        if not name:
            continue

        # Drop non-product rows (tax, savings, redemption fees, totals,
        # payment terminal lines, promotional footer text)
        if _NON_PRODUCT.search(name):
            continue

        # Drop structurally-junk names (URLs, EMV AIDs, "Item N", approval codes,
        # phone numbers, timestamps)
        if _JUNK_NAME.search(name):
            continue

        # Drop names that are too short to be a real product (single chars, "SC", "R")
        # or are purely numeric (OCR noise like "0.41" ending up as a product name).
        if len(name) <= 2 or re.fullmatch(r"[\d\s\.\,\-]+", name):
            continue

        # Drop store-metadata lines that models occasionally extract as items:
        # street addresses ("123 Main St"), phone-number prefixes ("(604)"),
        # "Welcome #", "Ref. #" variants, and lines that are mostly digits/punctuation.
        if re.search(r"\b(welcome|davie|street|avenue|blvd|suite|unit|floor)\b", name, re.IGNORECASE):
            continue

        # Drop items whose name is predominantly CJK characters (bilingual receipt
        # duplicates — T&T extracts each item in both English and Chinese).
        cjk_chars = sum(1 for c in name if '\u4e00' <= c <= '\u9fff')
        if cjk_chars > len(name) * 0.4:
            continue

        # Strip "(SALE)" prefix — these are duplicate rows for a discounted item,
        # not separate products. Remove the prefix so the dedup key matches.
        name = _SALE_PREFIX.sub("", name).strip()
        item["productName"] = name

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

    # --- Dedup pass: remove items extracted more than once ---
    # Catches duplicates that slip through chunk-merge dedup:
    #   1. Exact (name, price) duplicates within a single LLM response.
    #   2. Fuzzy name duplicates — same item named slightly differently
    #      (e.g. "CAMPBELLS SOUP" vs "Campbell's Soup") at the same price.
    import difflib

    def _norm_for_dedup(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    deduped: list[dict] = []
    for item in fixed:
        name_raw  = _norm_for_dedup(item.get("productName") or "")
        price_raw = re.sub(r"[A-Za-z]+$", "", (item.get("price") or "").strip())
        is_dup = False
        for kept in deduped:
            kept_name  = _norm_for_dedup(kept.get("productName") or "")
            kept_price = re.sub(r"[A-Za-z]+$", "", (kept.get("price") or "").strip())
            if kept_price != price_raw:
                continue   # different price → definitely different items
            # Same price: check name similarity
            if name_raw == kept_name:
                is_dup = True
                break
            if name_raw and kept_name:
                ratio = difflib.SequenceMatcher(None, name_raw, kept_name).ratio()
                if ratio >= 0.85:   # 85% similarity at same price → duplicate
                    is_dup = True
                    break
        if not is_dup:
            deduped.append(item)

    return deduped


_CURRENCY_MARKERS = [
    (re.compile(r'\bCAD\b|\bCAD\$|C\$|\$CAD', re.IGNORECASE), "CAD"),
    (re.compile(r'\bGBP\b|£',                                  re.IGNORECASE), "GBP"),
    (re.compile(r'\bEUR\b|€',                                  re.IGNORECASE), "EUR"),
]


_US_STORE_OCR_RE = re.compile(
    r'\b(KROGER|INGLES|INGLE\'?S|WALMART|WAL-MART|TARGET|VONS|RALPHS|SAFEWAY'
    r'|ALBERTSONS|PUBLIX|H-?E-?B|WHOLE\s+FOODS|TRADER\s+JOE\'?S'
    r'|FARM\s*&\s*TABLE|CVS|WALGREENS|RITE\s+AID)\b',
    re.IGNORECASE,
)


def _detect_currency_from_ocr(ocr_text: str) -> str | None:
    """
    Scan OCR text for currency signals and return the currency code.

    Non-USD markers (CAD/GBP/EUR) take priority — if found, return immediately.
    Then check for known US store names in the OCR; if found, return 'USD' to
    override an LLM-hallucinated non-USD currency when the model misread the
    store name (e.g. Qwen returning 'Grocery' for Ingles).
    Returns None when no signal is found (caller falls back to store-name
    inference, then the LLM's own currency guess, then 'USD').
    """
    for pattern, code in _CURRENCY_MARKERS:
        if pattern.search(ocr_text):
            return code
    if _US_STORE_OCR_RE.search(ocr_text):
        return "USD"
    return None


# ---------------------------------------------------------------------------
# Weight-priced item recovery
# ---------------------------------------------------------------------------
# Some receipts (e.g. No Frills) print items as:
#   BANANA
#   1.160 kg @ $1.72/kg  2.00        ← weight, per-unit rate, and total on one line
#   CELERY STICKS
#   0.075 kg @ $3.49/kg              ← total on the next line
#   0.26
#
# The LLM often extracts the item name but returns null price/amount because the
# ---------------------------------------------------------------------------
# Store-name aliases: LLM sometimes returns ALL-CAPS brand codes; map them
# back to the canonical "as-written" store name used in the ground truth.
# Keys are upper-cased for case-insensitive lookup.
# ---------------------------------------------------------------------------
_STORE_ALIASES: dict[str, str] = {
    "NOFRILLS":         "No Frills",
    "NO FRILLS":        "No Frills",
    "COSTCO WHOLESALE": "Costco Wholesale",
    "COSTCO":           "Costco Wholesale",
    "VONS":             "Vons",
    "VONS STORE":       "Vons",
}


def _normalize_store_name(raw: str) -> str:
    return _STORE_ALIASES.get((raw or "").strip().upper(), raw)


# Known Canadian stores — when OCR has no explicit CAD marker, infer from store name.
_CANADIAN_STORE_NAMES: frozenset[str] = frozenset({
    "No Frills",
    "Real Canadian Superstore",
    "T&T Supermarket",
    "House of Dosa",
    "House of Dosa- Downtown",
    "Loblaws",
    "Sobeys",
    "Metro",
    "FreshCo",
    "Food Basics",
    "Independent",
})

# Known US stores — used to override an LLM-hallucinated non-USD currency code.
_US_STORE_KEYWORDS: tuple[str, ...] = (
    "KROGER", "INGLES", "WALMART", "WAL-MART", "TARGET", "VONS",
    "RALPHS", "SAFEWAY", "ALBERTSONS", "PUBLIX", "HEB", "WHOLE FOODS",
    "TRADER JOE", "COSTCO",  # Costco has US and CA locations; OCR marker takes precedence
    "FARM & TABLE", "FARM AND TABLE",
    "CVS", "WALGREENS", "RITE AID",
)


def _infer_currency_from_store(store_name: str) -> str | None:
    """Return 'CAD' or 'USD' if store_name is a known chain, else None.

    Canadian chains → CAD.  US chains → USD (overrides LLM hallucinations in
    either direction).  Returns None when the store is unknown so the LLM's
    own currency guess is used as the fallback.
    """
    name = (store_name or "").strip()
    name_upper = name.upper()

    # Canadian check — exact then partial
    if name in _CANADIAN_STORE_NAMES:
        return "CAD"
    canadian_keywords = ("NO FRILLS", "REAL CANADIAN", "T&T SUPERMARKET",
                         "HOUSE OF DOSA", "LOBLAWS", "SOBEYS", "FRESHCO",
                         "FOOD BASICS")
    for kw in canadian_keywords:
        if kw in name_upper:
            return "CAD"

    # US check — keyword partial match
    for kw in _US_STORE_KEYWORDS:
        if kw in name_upper:
            return "USD"

    return None


# OCR number spacing (`1. 160`, `1. 72`) confuses it.  This regex + function
# parses the raw OCR deterministically and injects the correct price/amount.
_WEIGHT_ITEM_RE = re.compile(
    r'([\d]+\.[\d]+)\s*kg\s*@\s*\$\s*([\d]+\.[\d]+)\s*/kg(?:\s+([\d]+\.[\d]+))?',
    re.IGNORECASE,
)
_NORM_SPACED_NUM  = re.compile(r'(\d)\.\s+(\d)')   # "1. 160" → "1.160"
_DANGLING_PRICE   = re.compile(r'^\$?(\d+\.\d{2})\s*$')   # a line that is ONLY a price: "2.00", "$2.00"
_ENDS_WITH_PRICE  = re.compile(r'\$?\d+\.\d{2}\s*$')      # line already ends with a price


def _join_split_price_lines(ocr_text: str) -> str:
    """
    Join dangling price lines back onto the preceding item line.

    Some receipt scanners emit item name and price on separate lines:
        BANANAS
        2.00
    The LLM sees two unrelated lines and often returns price=null.
    This pass collapses them into one:
        BANANAS  2.00

    Rules (conservative to avoid false merges):
      - Current line is ONLY a price value (no other text).
      - Previous non-blank line does NOT already end with a price.
      - Previous line is not a section header / separator.

    Only applied to the raw OCR body — the SPATIAL LAYOUT section
    (built from bounding-box coordinates) is left untouched.
    """
    _SPATIAL_MARKER = "\n---\n## SPATIAL LAYOUT\n"
    if _SPATIAL_MARKER in ocr_text:
        raw_part, spatial_part = ocr_text.split(_SPATIAL_MARKER, 1)
        return _join_split_price_lines(raw_part) + _SPATIAL_MARKER + spatial_part

    lines  = ocr_text.split('\n')
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if (stripped
                and _DANGLING_PRICE.match(stripped)
                and result):
            # Find the last non-blank line in result
            for i in range(len(result) - 1, -1, -1):
                prev = result[i].rstrip()
                if not prev:
                    continue
                # Only merge if prev doesn't already carry a price
                if not _ENDS_WITH_PRICE.search(prev):
                    result[i] = prev + '  ' + stripped
                break
            else:
                result.append(line)
        else:
            result.append(line)
    return '\n'.join(result)


def _extract_weight_items_from_ocr(ocr_text: str) -> dict[str, tuple[str, str]]:
    """
    Scan raw OCR for weight-priced lines and return:
        { UPPER_ITEM_NAME: (weight_str, total_price_str) }
    """
    # Work only on the raw OCR (before the spatial layout section)
    raw = ocr_text.split("\n---\n")[0] if "\n---\n" in ocr_text else ocr_text
    lines = raw.splitlines()

    def _norm(s: str) -> str:
        return _NORM_SPACED_NUM.sub(r'\1.\2', s).strip()

    result: dict[str, tuple[str, str]] = {}
    for i, line in enumerate(lines):
        m = _WEIGHT_ITEM_RE.search(_norm(line))
        if not m:
            continue
        weight_str = m.group(1)
        total_str  = m.group(3)  # may be on the same line

        # Total not on the same line — scan the next few lines for a bare number
        if not total_str:
            for j in range(i + 1, min(i + 4, len(lines))):
                candidate = _norm(lines[j].strip())
                if re.match(r'^\d+\.\d+$', candidate):
                    total_str = candidate
                    break

        if not total_str:
            continue  # couldn't find a total — skip

        # Item name is on a line preceding the weight line
        # Skip blank lines and MRJ-only lines while looking backwards
        for k in range(i - 1, max(i - 5, -1), -1):
            prev = lines[k].strip()
            if not prev or prev.upper() == 'MRJ':
                continue
            # Strip leading barcode number and trailing MRJ tag
            cleaned = re.sub(r'^\d+\s+', '', prev)
            cleaned = re.sub(r'\s+MRJ\s*$', '', cleaned, flags=re.IGNORECASE).strip()
            # Skip section headers like "27-PRODUCE"
            if re.match(r'^\d+-[A-Z]+$', cleaned):
                continue
            if cleaned:
                result[cleaned.upper()] = (weight_str, total_str)  # injector appends " kg"
            break

    return result


def _inject_weight_prices(
    items: list[dict], ocr_text: str, currency: str
) -> list[dict]:
    """
    For any extracted item that has a null price, check whether the raw OCR
    contains a weight-priced line for that item and inject the total price
    and weight amount.
    """
    weight_map = _extract_weight_items_from_ocr(ocr_text)
    if not weight_map:
        return items
    import difflib
    matched_keys: set[str] = set()
    for item in items:
        name = (item.get("productName") or "").upper()
        # Try exact match first, then fuzzy
        candidates = [name] if name in weight_map else \
            difflib.get_close_matches(name, weight_map.keys(), n=1, cutoff=0.6)
        if not candidates:
            continue
        key = candidates[0]
        matched_keys.add(key)
        weight, total = weight_map[key]
        # Always override: OCR-derived weight data is deterministic; LLM may
        # hallucinate the wrong price for weight-priced items.
        try:
            item["price"]  = f"{float(total):.2f}{currency}"
            item["amount"] = f"{weight} kg"
        except ValueError:
            pass

    # Add any weight-priced items the LLM missed entirely.
    # Seed metadata from the first extracted item (same receipt context).
    proto = items[0] if items else {}
    for key, (weight, total) in weight_map.items():
        if key in matched_keys:
            continue
        try:
            new_item: dict = {
                "productName": key.title(),
                "purchaseDate": proto.get("purchaseDate"),
                "price":  f"{float(total):.2f}{currency}",
                "amount": f"{weight} kg",
                "storeName": proto.get("storeName"),
            }
            items.append(new_item)
        except ValueError:
            pass

    return items


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


def structure(ocr_text: str, display_name: str, user_name: str,
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

    # Tier 0 — global OCR normalisation before any other processing.
    # Step 1: fix spaced decimals ("1. 160" → "1.160", "1. 72" → "1.72").
    # Step 2: join dangling price lines onto the preceding item line so the
    #         LLM sees "BANANAS  2.00" rather than two unrelated lines.
    # Applied once here so every downstream tier — noise filter, chunker,
    # weight-price parser, and LLM prompt — all see clean, aligned text.
    ocr_text = _NORM_SPACED_NUM.sub(r'\1.\2', ocr_text)
    ocr_text = _join_split_price_lines(ocr_text)

    # Tier 1 — strip noise lines before the text reaches the LLM.
    # Reduces token count on large receipts and prevents total/tax rows
    # from being misidentified as product items.
    ocr_text = _filter_noise_lines(ocr_text)

    # Always run the chunker so it strips the raw OCR body when a SPATIAL
    # LAYOUT section is present (saves ~40% tokens on short receipts too).
    # For long receipts it also splits into overlapping sections as before.
    _SPATIAL_MARKER = "\n---\n## SPATIAL LAYOUT\n"
    if _SPATIAL_MARKER in ocr_text or len(ocr_text) > _CHUNK_THRESHOLD_CHARS:
        chunks = _split_ocr_into_chunks(ocr_text)
    else:
        chunks = [ocr_text]
    is_chunked = len(chunks) > 1

    # Use the leaner direct-output prompt for simple receipts (no CoT scaffolding).
    # Simple = single chunk with spatial layout (column-aligned, unambiguous).
    # Complex = chunked or no spatial layout → keep full CoT prompt for accuracy.
    _SPATIAL_MARKER = "\n---\n## SPATIAL LAYOUT\n"
    _use_direct = not is_chunked and _SPATIAL_MARKER in ocr_text
    _is_costco  = "COSTCO" in ocr_text.upper()[:500]
    _prompt = _build_system_prompt(ocr_text, _use_direct)

    # Prompt-path label recorded in the log so per-receipt token counts can be
    # interpreted alongside system-prompt size differences.
    _prompt_path = ("direct" if _use_direct else "cot") + ("+costco" if _is_costco else "")

    # Total OCR content chars sent to the LLM (sum across all chunks, system
    # prompt excluded).  This is the metric we track for token reduction work.
    _input_chars = sum(len(c) for c in chunks)

    start = time.monotonic()
    total_pt, total_ct, total_cost = 0, 0, 0.0
    cost_source    = "estimate"
    latency_source = "local"
    chunk_results: list[dict] = []

    try:
        for chunk_text in chunks:
            if resolved_provider == "clod":
                try:
                    c_result, pt, ct, api_cost, api_latency_ms, _ = _structure_clod(chunk_text, model, _prompt)
                except json.JSONDecodeError:
                    # Chunk returned unparseable content — skip it, other chunks still contribute
                    continue
                if api_cost is not None:
                    total_cost += api_cost
                    cost_source = "api"
                else:
                    total_cost += _estimate_clod_cost(model, pt, ct) or 0.0
            else:
                c_result, pt, ct, gen_id = _structure_openrouter(chunk_text, model, _prompt)
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

        # Tier 2c — normalise store name first so Canadian-store CAD inference works.
        if result.get("storeName"):
            result["storeName"] = _normalize_store_name(result["storeName"])

        # Override LLM-extracted currency with deterministic OCR scan so
        # small models that default to "USD" are corrected for CA/GB/EU receipts.
        # Fall back to Canadian-store inference when OCR has no explicit marker.
        ocr_currency = _detect_currency_from_ocr(ocr_text)
        if ocr_currency is None:
            ocr_currency = _infer_currency_from_store(result.get("storeName") or "")
        currency = ocr_currency or result.get("currency") or "USD"
        result["currency"] = currency

        # Tier 2+3 — deterministic post-processing
        result["items"] = _validate_and_fix_items(result.get("items", []), currency)

        # Tier 2b — recover prices for weight-priced items (e.g. "1.160 kg @ $1.72/kg 2.00")
        # Do this before the null-price repair so weight items don't consume repair budget.
        result["items"] = _inject_weight_prices(result["items"], ocr_text, currency)

        # Tier 3b+4 — targeted repair for items with null price, then escalate
        null_price_count = sum(1 for i in result["items"] if not (i.get("price") or "").strip() or (i.get("price") or "").lower() == "null")
        if null_price_count:
            result["items"] = _repair_failed_items(
                result["items"], ocr_text, model, resolved_provider, currency
            )

        result["totalItems"] = len(result["items"])

        latency_ms = (time.monotonic() - start) * 1000

        # Inject caller-controlled fields
        result["imageName"] = display_name
        result["userName"]  = user_name

        # Geocode once using merged store address.
        # Fallback: if LLM returned no storeAddress, extract it from the OCR text
        # by looking for a line that contains digits + street keywords near the top.
        address = result.get("storeAddress") or ""
        if not address:
            address = _extract_address_from_ocr(ocr_text, result.get("storeName") or "")
            if address:
                result["storeAddress"] = address
        lat, lon = geocode(address, store_name=result.get("storeName") or "")
        result["latitude"]  = lat
        result["longitude"] = lon

        log_llm(run_id, display_name, user_name, resolved_provider, model,
                total_pt, total_ct, total_cost, latency_ms,
                len(result.get("items", [])), True,
                cost_source=cost_source, latency_source=latency_source,
                input_chars=_input_chars, prompt_path=_prompt_path)
        return result, total_pt, total_ct, total_cost

    except Exception as e:
        latency_ms = (time.monotonic() - start) * 1000
        log_llm(run_id, display_name, user_name, resolved_provider, model,
                0, 0, 0.0, latency_ms, 0, False, str(e), latency_source="local",
                input_chars=_input_chars, prompt_path=_prompt_path)
        raise


# ---------------------------------------------------------------------------
# Geocoding — Azure Maps (optional, skipped if AZURE_MAPS_KEY is unset)
# ---------------------------------------------------------------------------
def geocode(address: str, store_name: str = "") -> tuple[float | None, float | None]:
    """
    Look up lat/lon for a store address using Azure Maps Fuzzy Search API.
    Prepends store_name to the query so Azure Maps can match the POI directly
    rather than falling back to the nearest street address (which drifts in longitude).
    Adds countrySet=CA,US to avoid cross-country mismatches.
    Returns (latitude, longitude) or (None, None) if unavailable.
    Requires AZURE_MAPS_KEY in .env.
    """
    if not AZURE_MAPS_KEY or not address:
        return None, None
    try:
        import urllib.request
        import urllib.parse
        query = f"{store_name} {address}".strip() if store_name else address
        url = (
            AZURE_MAPS_URL
            + f"?api-version=1.0&subscription-key={AZURE_MAPS_KEY}"
            f"&query={urllib.parse.quote(query)}&limit=1&countrySet=CA,US"
        )
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        results = data.get("results", [])
        if results:
            pos = results[0]["position"]
            return pos["lat"], pos["lon"]
    except Exception as e:
        print(f"  [GEO]  geocode failed for '{query}': {e}")
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
def extract(image_data: "Path | bytes", display_name: str, user_name: str, model: str, run_id: str,
            provider: str | None = None, use_cache: bool = True) -> dict:
    resolved_provider = (provider or LLM_PROVIDER).lower()

    # Railtracks path requires a file on disk — only available when image_data is a Path.
    if _RT_AVAILABLE and isinstance(image_data, Path):
        flow = rt.Flow(name="receipt_etl", entry_point=receipt_pipeline)
        result: StructureOutput = flow.invoke(OcrInput(
            image_path=str(image_data),
            run_id=run_id,
            user_name=user_name,
            model=model,
            provider=resolved_provider,
        ))
        return json.loads(result.receipt_json)
    else:
        cache_stem = Path(display_name).stem if display_name else "unknown"
        _cache_hit = use_cache and (OCR_CACHE_DIR / (cache_stem + ".txt")).exists()
        print(f"  [ADI]  OCR {'(cached)' if _cache_hit else '…'}")
        ocr_text = ocr(image_data, display_name, run_id, user_id=user_name, use_cache=use_cache)
        print(f"  [LLM]  Structuring via {resolved_provider} ({len(ocr_text)} chars) …")
        result, _, _, _ = structure(ocr_text, display_name, user_name, model, run_id, provider=resolved_provider)
        return result


# ---------------------------------------------------------------------------
# Upload via GYD SDK
# ---------------------------------------------------------------------------
def upload(receipt: dict, run_id: str, token: str | None = None, refresh_token: str | None = None):
    try:
        from gather_your_deals import GYDClient
    except ImportError:
        raise ImportError(
            "GYD SDK not installed.\n"
            "pip install git+https://github.com/yuewang199511/GatherYourDeals-SDK.git"
        )

    # Token priority: per-request JWT (forwarded from caller) >
    #                 GYD_ACCESS_TOKEN env var (CLI / local testing fallback)
    resolved_token = token or GYD_ACCESS_TOKEN
    client = GYDClient(GYD_SERVER_URL, auto_persist_tokens=False)
    if resolved_token:
        client._transport.set_tokens(resolved_token, refresh_token or "")

    items   = receipt.get("items", [])
    created, failed = [], 0
    start   = time.monotonic()

    image_name  = receipt.get("imageName", "")

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
            print(f"    [ERROR] upload failed for '{item.get('productName')}': {e}")
            failed += 1

    latency_ms = (time.monotonic() - start) * 1000
    log_upload(run_id, image_name, receipt.get("userName", ""),
               len(items), len(created), failed, latency_ms,
               failed == 0, f"{failed} items failed" if failed else None)

    if failed:
        raise RuntimeError(
            f"Upload incomplete: {failed}/{len(items)} item(s) failed for '{image_name}'"
        )

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
    p.add_argument("--no-ocr-cache",      action="store_true",
                   help="Force fresh ADI call even if ocr_cache/<stem>.txt exists")
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
            data = extract(img, args.user, resolved_model, run_id, provider=args.provider,
                           use_cache=not args.no_ocr_cache)
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
                data["imageName"] = img.name
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
