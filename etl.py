#!/usr/bin/env python3
"""
GatherYourDeals ETL
====================
Two-step pipeline:
  Step 1 — Azure Document Intelligence (prebuilt-read)
            Sends the receipt image to ADI, which returns the raw OCR text.
  Step 2 — LLM structuring  (OpenRouter  OR  Claude / Anthropic)
            Receives the OCR text and structures it into the GYD JSON format.

Usage
-----
  # Single receipt — OpenRouter (default)
  python etl.py ../Receipts/converted/2025-10-01Vons.jpg --user lkim016 --no-upload

  # Single receipt — Claude (native Anthropic API)
  python etl.py ../Receipts/converted/2025-10-01Vons.jpg --user lkim016 --provider claude --no-upload

  # Whole directory
  python etl.py ../Receipts/converted/ --user lkim016 --no-upload

  # With SDK upload
  python etl.py ../Receipts/converted/2025-10-01Vons.jpg --user lkim016

  # Generate usage report from logs
  python etl.py --report

Requirements
------------
  pip install openai python-dotenv azure-ai-documentintelligence
  pip install anthropic                # required for --provider claude
  pip install git+https://github.com/yuewang199511/GatherYourDeals-SDK.git
  pip install matplotlib              # optional — charts in --report
  pip install pillow pillow-heif      # optional — only for HEIC (iPhone) photos

Environment (.env)
------------------
  # Step 1 — Azure Document Intelligence
  # Create a Document Intelligence resource in Azure portal.
  # Free tier (F0): 500 pages/month, no charge.
  AZURE_DI_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/
  AZURE_DI_KEY=<your-key>

  # Step 2a — OpenRouter LLM  (LLM_PROVIDER=openrouter)
  # Sign up at openrouter.ai → Keys → Create key.
  # Default model is free-tier; no credits needed.
  OPENROUTER_API_KEY=sk-or-v1-...
  ETL_MODEL=google/gemini-2.0-flash-exp:free

  # Step 2b — Claude / Anthropic  (LLM_PROVIDER=claude)
  # Sign up at console.anthropic.com → API Keys.
  ANTHROPIC_API_KEY=sk-ant-...
  ETL_MODEL=claude-haiku-4-5-20251001

  # Which LLM provider to use: "openrouter" (default) or "claude"
  LLM_PROVIDER=openrouter

  # GYD data server (leave blank to run extract-only)
  GYD_SERVER_URL=http://localhost:8080/api/v1
  GYD_USERNAME=
  GYD_PASSWORD=
"""

import argparse
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
AZURE_DI_ENDPOINT = os.getenv("AZURE_DI_ENDPOINT", "")
AZURE_DI_KEY      = os.getenv("AZURE_DI_KEY", "")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY",  "")

# Which LLM backend to use: "openrouter" (default) or "claude"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").lower()

_DEFAULT_MODELS = {
    "openrouter": "anthropic/claude-3-haiku",
    "claude":     "claude-haiku-4-5-20251001",
}
DEFAULT_MODEL = os.getenv("ETL_MODEL", _DEFAULT_MODELS.get(LLM_PROVIDER, "google/gemini-2.0-flash-exp:free"))

GYD_SERVER_URL = os.getenv("GYD_SERVER_URL", "http://localhost:8080/api/v1")
GYD_USERNAME   = os.getenv("GYD_USERNAME", "")
GYD_PASSWORD   = os.getenv("GYD_PASSWORD", "")

OUTPUT_DIR  = Path("output")
LOGS_DIR    = Path("logs")
REPORTS_DIR = Path("reports")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".tiff", ".tif", ".bmp"}

# LLM cost per 1M tokens (input, output) — free models are $0
_LLM_PRICING = {
    # OpenRouter models
    "google/gemini-2.0-flash-exp:free": (0.0,   0.0),   # retired
    "anthropic/claude-3-haiku":         (0.0,   0.0),   # free on OpenRouter
    "deepseek/deepseek-chat":           (0.0,   0.0),   # free on OpenRouter
    "google/gemini-2.0-flash":          (0.10,  0.40),
    "openai/gpt-4o-mini":               (0.15,  0.60),
    "openai/gpt-4o":                    (2.50, 10.00),
    "anthropic/claude-3.5-sonnet":      (3.00, 15.00),
    # Native Claude models (Anthropic API)
    "claude-haiku-4-5-20251001":        (0.80,  4.00),
    "claude-sonnet-4-6":                (3.00, 15.00),
    "claude-opus-4-6":                  (15.00, 75.00),
}

# ADI cost per page — F0 free tier is $0; S0 standard is $0.0015/page
# Set AZURE_DI_TIER=S0 in .env if you upgrade beyond the free tier.
_ADI_COST_PER_PAGE = 0.0015 if os.getenv("AZURE_DI_TIER", "F0").upper() == "S0" else 0.0

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
  latitude      null     (cannot be derived from a receipt)
  longitude     null     (cannot be derived from a receipt)
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
- Include EVERY line item — do not skip donations, fees, or miscellaneous charges.
- price must always include the currency code: "4.79USD" not "4.79".
- amount is the number of units or weight, never a price.
- If a product has a member/club discount applied, use the discounted (final) price for price.

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
            pages, cost_usd, latency_ms, success, error=None):
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
        "ocr_input_tokens":  None,   # ADI is billed per page, not tokens
        "ocr_output_tokens": None,
        "ocr_success":       success,
        "pages":             pages,
        "cost_usd":          round(cost_usd, 6),
        "error":             error,
    })

def log_llm(trace_id, image_name, user_id, provider, model,
            input_tokens, output_tokens, cost_usd,
            latency_ms, items_extracted, success, error=None):
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
        "llm_success":       success,
        "items_extracted":   items_extracted,
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
def _to_jpeg_bytes(image_path: Path) -> tuple[bytes, str]:
    """Return (image_bytes, content_type). Converts HEIC → JPEG if needed."""
    ext = image_path.suffix.lower()
    if ext == ".heic":
        try:
            import pillow_heif
            from PIL import Image
            import io
            pillow_heif.register_heif_opener()
            buf = io.BytesIO()
            Image.open(image_path).convert("RGB").save(buf, format="JPEG")
            return buf.getvalue(), "image/jpeg"
        except ImportError:
            raise ImportError("HEIC support requires: pip install pillow pillow-heif")
    content_type = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".webp": "image/webp",
        ".tiff": "image/tiff", ".tif": "image/tiff",
        ".bmp": "image/bmp",
    }.get(ext, "image/jpeg")
    return image_path.read_bytes(), content_type


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
        result = poller.result()
        latency_ms = (time.monotonic() - start) * 1000

        pages = len(result.pages) if result.pages else 1
        ocr_text = result.content or ""

        log_adi(run_id, image_path.name, user_id, image_size_bytes,
                pages, pages * _ADI_COST_PER_PAGE, latency_ms, True)
        return ocr_text

    except Exception as e:
        latency_ms = (time.monotonic() - start) * 1000
        log_adi(run_id, image_path.name, user_id, image_size_bytes,
                0, 0.0, latency_ms, False, str(e))
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


def _structure_openrouter(ocr_text: str, model: str) -> tuple[dict, int, int]:
    """Call OpenRouter and return (parsed_dict, prompt_tokens, completion_tokens)."""
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
    )
    usage = resp.usage
    pt = usage.prompt_tokens     if usage else 0
    ct = usage.completion_tokens if usage else 0
    return _parse_llm_json(resp.choices[0].message.content), pt, ct


def _structure_claude(ocr_text: str, model: str) -> tuple[dict, int, int]:
    """Call the Anthropic Claude API and return (parsed_dict, input_tokens, output_tokens)."""
    if not ANTHROPIC_API_KEY:
        raise EnvironmentError("ANTHROPIC_API_KEY not set — add it to .env")

    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError("Anthropic SDK not installed.\npip install anthropic")

    client = Anthropic(api_key=ANTHROPIC_API_KEY)

    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        temperature=0,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Receipt OCR text:\n\n{ocr_text}"}],
    )
    pt = resp.usage.input_tokens
    ct = resp.usage.output_tokens
    return _parse_llm_json(resp.content[0].text), pt, ct


def structure(ocr_text: str, image_path: Path, user_name: str,
              model: str, run_id: str, provider: str | None = None) -> dict:
    """
    Send OCR text to the configured LLM provider and return the structured receipt dict.

    :param provider: ``"openrouter"`` or ``"claude"``.  Defaults to the
        ``LLM_PROVIDER`` environment variable (fallback: ``"openrouter"``).
    """
    resolved_provider = (provider or LLM_PROVIDER).lower()

    start = time.monotonic()
    try:
        if resolved_provider == "claude":
            result, pt, ct = _structure_claude(ocr_text, model)
        else:
            result, pt, ct = _structure_openrouter(ocr_text, model)

        latency_ms = (time.monotonic() - start) * 1000
        rates = _LLM_PRICING.get(model, (1.0, 3.0))
        cost  = (pt * rates[0] + ct * rates[1]) / 1_000_000

        # Inject caller-controlled fields
        result["imageName"] = image_path.name
        result["userName"]  = user_name
        result.setdefault("latitude",  None)
        result.setdefault("longitude", None)

        log_llm(run_id, image_path.name, user_name, resolved_provider, model,
                pt, ct, cost, latency_ms, len(result.get("items", [])), True)
        return result

    except Exception as e:
        latency_ms = (time.monotonic() - start) * 1000
        log_llm(run_id, image_path.name, user_name, resolved_provider, model,
                0, 0, 0.0, latency_ms, 0, False, str(e))
        raise


# ---------------------------------------------------------------------------
# Extract — runs Step 1 then Step 2
# ---------------------------------------------------------------------------
def extract(image_path: Path, user_name: str, model: str, run_id: str,
            provider: str | None = None) -> dict:
    print(f"  [ADI]  OCR …")
    ocr_text = ocr(image_path, run_id, user_id=user_name)
    resolved_provider = (provider or LLM_PROVIDER).lower()
    print(f"  [LLM]  Structuring via {resolved_provider} ({len(ocr_text)} chars of OCR text) …")
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
# Report
# ---------------------------------------------------------------------------
def report():
    entries = []
    for f in sorted(LOGS_DIR.glob("etl_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    adi_entries = [e for e in entries if e.get("event") == "adi_ocr"]
    llm_entries = [e for e in entries if e.get("event") == "llm_extraction"]
    ups         = [e for e in entries if e.get("event") == "mcp_upload"]

    if not llm_entries and not adi_entries:
        print("No log entries found. Run the pipeline first.")
        return

    # ---- ADI stats ----
    adi_calls   = len(adi_entries)
    adi_cost    = sum(e.get("cost_usd", 0) for e in adi_entries)
    adi_pages   = sum(e.get("pages", 0)    for e in adi_entries)
    adi_avg_lat = (sum(e.get("ocr_latency_ms", 0) for e in adi_entries) / adi_calls
                   if adi_calls else 0)

    # ---- LLM stats ----
    llm_calls    = len(llm_entries)
    llm_ok       = sum(1 for e in llm_entries if e.get("llm_success"))
    total_prompt = sum(e.get("llm_input_tokens",  0) for e in llm_entries)
    total_comp   = sum(e.get("llm_output_tokens", 0) for e in llm_entries)
    total_tokens = total_prompt + total_comp
    llm_cost     = sum(e.get("llm_cost_usd", 0)     for e in llm_entries)
    llm_avg_lat  = (sum(e.get("llm_latency_ms", 0) for e in llm_entries) / llm_calls
                    if llm_calls else 0)
    total_items  = sum(e.get("items_extracted", 0)  for e in llm_entries)
    uploaded     = sum(e.get("items_uploaded",  0)  for e in ups)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# GatherYourDeals ETL Report",
        f"_Generated: {now}_", "",
        "## Cost & Token Summary", "",
        "| Metric | Value |", "|--------|-------|",
        f"| ADI OCR calls | {adi_calls} ({adi_pages} pages) |",
        f"| ADI estimated cost | ${adi_cost:.4f} USD |",
        f"| ADI avg latency | {adi_avg_lat:.0f} ms |",
        f"| LLM calls | {llm_calls} ({llm_ok} success) |",
        f"| LLM total tokens | {total_tokens:,} |",
        f"| LLM input / output | {total_prompt:,} / {total_comp:,} |",
        f"| LLM estimated cost | ${llm_cost:.6f} USD |",
        f"| LLM avg latency | {llm_avg_lat:.0f} ms |",
        f"| **Total estimated cost** | **${adi_cost + llm_cost:.6f} USD** |",
        f"| Items extracted | {total_items} |",
        f"| Items uploaded | {uploaded} |",
        "", "## Per-Image Breakdown", "",
        "| Image | ADI (ms) | ADI cost | LLM provider | LLM model | Input | Output | LLM cost | LLM (ms) | Items | OK |",
        "|-------|--------:|---------:|--------------|-----------|------:|-------:|---------:|---------:|------:|:--:|",
    ]

    # Merge ADI and LLM rows by image_name
    adi_by_name = {e.get("image_name"): e for e in adi_entries}
    for e in llm_entries:
        name = e.get("image_name", "?")
        adi  = adi_by_name.get(name, {})
        ok   = "✓" if e.get("llm_success") else "✗"
        lines.append(
            f"| {name} "
            f"| {adi.get('ocr_latency_ms', 0):.0f} | ${adi.get('cost_usd', 0):.4f} "
            f"| {e.get('llm_provider', '?')} "
            f"| {e.get('llm_model', '?')} "
            f"| {e.get('llm_input_tokens', 0):,} | {e.get('llm_output_tokens', 0):,} "
            f"| ${e.get('llm_cost_usd', 0):.6f} | {e.get('llm_latency_ms', 0):.0f} "
            f"| {e.get('items_extracted', 0)} | {ok} |"
        )

    REPORTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    md_path = REPORTS_DIR / f"report_{ts}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report → {md_path}")
    _chart(adi_entries, llm_entries, ts, md_path)


def _chart(adi_entries, llm_entries, ts, md_path):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        print("matplotlib not installed — skipping chart. pip install matplotlib")
        return

    if not llm_entries:
        return

    images  = [e.get("image_name", "?")          for e in llm_entries]
    prompts = [e.get("llm_input_tokens",  0)     for e in llm_entries]
    comps   = [e.get("llm_output_tokens", 0)     for e in llm_entries]
    llm_lat = [e.get("llm_latency_ms",   0)      for e in llm_entries]
    adi_by  = {e.get("image_name"): e.get("ocr_latency_ms", 0) for e in adi_entries}
    adi_lat = [adi_by.get(n, 0) for n in images]
    colours = ["#2ca02c" if e.get("llm_success") else "#d62728" for e in llm_entries]
    x = list(range(len(images)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("GatherYourDeals ETL — Usage", fontweight="bold")

    # Token usage
    ax = axes[0]
    ax.bar(x, prompts,              label="Prompt",     color="#4C72B0")
    ax.bar(x, comps, bottom=prompts, label="Completion", color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels(images, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Tokens"); ax.set_title("LLM Token Usage"); ax.legend()
    for i, (p, c) in enumerate(zip(prompts, comps)):
        ax.text(i, p + c, f"{p+c:,}", ha="center", va="bottom", fontsize=6)

    # LLM latency
    ax2 = axes[1]
    ax2.bar(x, llm_lat, color=colours)
    ax2.set_xticks(x); ax2.set_xticklabels(images, rotation=30, ha="right", fontsize=7)
    ax2.set_ylabel("ms"); ax2.set_title("LLM Latency per Image")
    ax2.legend(handles=[Patch(color="#2ca02c", label="OK"),
                        Patch(color="#d62728", label="Error")])

    # ADI latency
    ax3 = axes[2]
    ax3.bar(x, adi_lat, color="#9467BD")
    ax3.set_xticks(x); ax3.set_xticklabels(images, rotation=30, ha="right", fontsize=7)
    ax3.set_ylabel("ms"); ax3.set_title("ADI OCR Latency per Image")

    plt.tight_layout()
    chart = REPORTS_DIR / f"usage_chart_{ts}.png"
    plt.savefig(chart, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Chart  → {chart}")
    with open(md_path, "a", encoding="utf-8") as f:
        f.write(f"\n\n![Usage Chart](./{chart.name})\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="GatherYourDeals receipt ETL (ADI + LLM)")
    p.add_argument("path",        nargs="?", help="Image file or directory")
    p.add_argument("--user",      default="unknown",     help="Username for JSON metadata")
    p.add_argument("--provider",  default=LLM_PROVIDER,
                   choices=["openrouter", "claude"],
                   help="LLM provider (default: LLM_PROVIDER env var)")
    p.add_argument("--model",     default=None,
                   help="Model ID — defaults to ETL_MODEL env var or provider default")
    p.add_argument("--no-upload", action="store_true",   help="Skip SDK upload")
    p.add_argument("--report",    action="store_true",   help="Generate usage report")
    args = p.parse_args()

    # Resolve model: CLI > env > provider default
    resolved_model = args.model or os.getenv("ETL_MODEL") or _DEFAULT_MODELS.get(args.provider, DEFAULT_MODEL)

    if args.report:
        report(); return

    if not args.path:
        p.print_help(); sys.exit(1)

    target    = Path(args.path)
    do_upload = not args.no_upload and bool(GYD_USERNAME and GYD_PASSWORD)
    if not do_upload and not args.no_upload:
        print("[INFO] GYD credentials not set — extract-only mode.")
    run_id = str(uuid.uuid4())

    images = (sorted(f for f in target.iterdir() if f.suffix.lower() in IMAGE_EXTS)
              if target.is_dir() else [target] if target.is_file() else [])
    if not images:
        print(f"No images found at {target}"); sys.exit(1)

    errors = 0
    for img in images:
        print(f"\n→ {img.name}")
        try:
            data = extract(img, args.user, resolved_model, run_id, provider=args.provider)
            OUTPUT_DIR.mkdir(exist_ok=True)
            out = OUTPUT_DIR / (img.stem + ".json")
            out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  saved  {out}  ({len(data.get('items', []))} items)")
            if do_upload:
                created = upload(data, run_id)
                print(f"  uploaded {len(created)}/{len(data.get('items', []))} items")
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            errors += 1

    print(f"\nDone — {len(images)-errors}/{len(images)} succeeded.")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
