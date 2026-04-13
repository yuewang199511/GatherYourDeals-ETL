import re
import json
import time
import httpx
from typing import Dict, Any, Optional, List, Tuple, TypedDict
from dataclasses import dataclass
from openai import OpenAI
from src.core import config, prompts


_CHUNK_THRESHOLD_CHARS = 2000   # raised from 1000 — most receipts fit in one chunk

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class LLMResult:
    data: Dict[str, Any]
    input_tokens: int
    output_tokens: int
    cost_usd: Optional[float]
    latency_ms: Optional[float]
    raw: str
    generation_id: Optional[str]

class NormalizedResponse(TypedDict, total=False):
    raw: str
    input_tokens: int
    output_tokens: int
    cost_usd: Optional[float]
    latency_ms: Optional[float]
    generation_id: Optional[str]
    
# ---------------------------------------------------------------------------
# Helper Functions (The "Missing" logic)
# ---------------------------------------------------------------------------

def _call_openrouter(messages: List[Dict[str, str]], model: str) -> Any:
    """Uses the OpenAI client to hit the OpenRouter endpoint."""
    client = OpenAI(
        base_url=config.OPENROUTER_BASE_URL,
        api_key=config.OPENROUTER_API_KEY,
    )
    
    return client.chat.completions.create(
        model=model,
        messages=messages,
        extra_headers={
            "HTTP-Referer": "https://github.com/your-repo/gather-your-deals",
            "X-Title": "GatherYourDeals",
        }
    )

def _call_clod(messages: List[Dict[str, str]], model: str) -> Dict[str, Any]:
    """Uses httpx to hit the Clod API directly (as per config.CLOD_API_URL)."""
    headers = {
        "Authorization": f"Bearer {config.CLOD_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 4096
    }
    
    with httpx.Client(timeout=config._CLOD_TIMEOUT_S) as client:
        response = client.post(config.CLOD_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()

def _normalize_openai(resp: Any) -> NormalizedResponse:
    """Extracts data from a standard OpenAI/OpenRouter response object."""
    return {
        "raw": resp.choices[0].message.content,
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens,
        "generation_id": getattr(resp, 'id', None)
    }

def _normalize_clod(resp: Dict[str, Any]) -> NormalizedResponse:
    """Extracts data from the trinity-mini / clod response structure."""
    try:
        # Based on your debug: resp['choices'][0]['message']['content']
        choice = resp["choices"][0]
        message = choice.get("message", {})
        
        raw_content = message.get("content", "")
        
        # Based on your debug: resp['usage'] keys are 'prompt_tokens' and 'completion_tokens'
        usage = resp.get("usage", {})
        
        return {
            "raw": raw_content,
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0)
        }
    except (KeyError, IndexError, TypeError) as e:
        print(f"  [LLM]  Normalization failed for Clod: {e}")
        return None

def estimate_cost(provider: str, model: str, input_t: int, output_t: int) -> float:
    """Calculates cost based on the LLM_PRICING table in config.py."""
    pricing = config.LLM_PRICING.get(provider, {}).get(model)
    if not pricing:
        return 0.0
    
    in_rate, out_rate = pricing
    return ((input_t / 1_000_000) * in_rate) + ((output_t / 1_000_000) * out_rate)

# ---------------------------------------------------------------------------
# Step 2 — LLM structuring
# ---------------------------------------------------------------------------

def structure_llm(provider: str, ocr_text: str, model: str, system_prompt: str = prompts.SYSTEM_PROMPT) -> LLMResult:
    """Main entry point for ETL structuring."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Receipt OCR text:\n\n{ocr_text}"},
    ]

    if provider == "openrouter":
        resp = _call_openrouter(messages, model)
        norm = _normalize_openai(resp)
    elif provider == "clod":
        resp = _call_clod(messages, model)
        norm = _normalize_clod(resp)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # 1. Protect against a completely failed normalizer
    if norm is None or norm.get("raw") is None:
        print(f"  [LLM]  CRITICAL: {provider} returned no content. Skipping parse.")
        # Create a dummy norm dict to prevent crashes in the LLMResult return
        norm = {"raw": "", "input_tokens": 0, "output_tokens": 0}
        parsed = None
    else:
        # 2. Only call the parser if we actually have a string
        parsed = parse_llm_json(norm["raw"])

    if parsed is None:
        print(f"  [LLM]  WARNING: Failed to parse JSON from {provider}. Returning empty template.")
        parsed = {
            "storeName": None, 
            "items": [], 
            "total": None, 
            "currency": "USD"
        }

    # Attempt to get cost from API, fall back to estimation
    cost = norm.get("cost_usd")
    if cost is None:
        cost = estimate_cost(provider, model, norm["input_tokens"], norm["output_tokens"])

    return LLMResult(
        data=parsed,
        input_tokens=norm.get("input_tokens", 0),
        output_tokens=norm.get("output_tokens", 0),
        cost_usd=cost,
        latency_ms=norm.get("latency_ms"),
        raw=norm["raw"],
        generation_id=norm.get("generation_id"),
    )

def parse_llm_json(raw: str) -> Dict[str, Any]:
    """Extract and parse JSON from noisy LLM output with None safety."""
    if raw is None:
        raise ValueError("Received None instead of string for LLM parsing.")
        
    raw = raw.strip()
    if not raw:
        raise ValueError("LLM output is empty.")

    # 1. Check for <json> tags (common in reasoning/thought models)
    m = re.search(r"<json>\s*(.*?)\s*</json>", raw, re.DOTALL)
    if m:
        return json.loads(m.group(1).strip())

    # 2. Check for markdown code blocks (```json ... ```)
    m = re.search(r"```(?:json)?\s*\n(.*?)\n```", raw, re.DOTALL)
    if m:
        raw = m.group(1).strip()

    # 3. Sometimes models prepend commentary → try to isolate first JSON object
    # This grabs the first {...} or [...]
    m = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
    if m:
        raw = m.group(1).strip()

    # 4. Remove trailing commas (JSON5-style output from models)
    raw = re.sub(r",\s*([}\]])", r"\1", raw)

    # 5. Final parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        # Optional: make debugging easier upstream
        raise ValueError(
            f"Failed to parse LLM JSON. Raw output was:\n\n{raw}"
        ) from e


# ---------------------------------------------------------------------------
# Long-receipt chunking
# ---------------------------------------------------------------------------
# LLM attention degrades when OCR text is long and noisy: rows bleed into each
# other, items get merged or dropped, and column alignment is lost.  Receipts
# over _CHUNK_THRESHOLD_CHARS are split into overlapping vertical sections
# before being sent to the LLM.  Results are merged after all chunks return.

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
            if provider == "clod" and config.CLOD_API_KEY:
                resp = httpx.post(
                    config.CLOD_API_URL,
                    headers={"Authorization": f"Bearer {config.CLOD_API_KEY}", "Content-Type": "application/json"},
                    json={"model": model, "messages": [{"role": "user", "content": prompt}]},
                    timeout=30,
                )
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"].strip()
            elif config.OPENROUTER_API_KEY:
                client = OpenAI(api_key=config.OPENROUTER_API_KEY, base_url=config.CLOD_DEFAULT_MODEL)
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
        if not found_price and config.OPENROUTER_API_KEY:
            try:
                client = OpenAI(api_key=config.OPENROUTER_API_KEY, base_url=config.CLOD_DEFAULT_MODEL)
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
    _UNIT_STRIP     = re.compile(r"\b(W|EA|PK|F|N|X|O|T|PC|CT|BG|LT|BT|CN|OZ|MRJ|RQ|FV|IQF|WLD|each)\b", re.IGNORECASE)
    _PRICE_FMT      = re.compile(r"^\d+\.\d{2}\s*[A-Za-z]?$")  # "4.79", "4.79S", "4.79 S"
    _ITEM_CODE      = re.compile(r"^\d{4,}\s+")            # leading 4+ digit item/barcode code
    _PRICE_LETTER   = re.compile(r"^(\d+\.?\d*)[A-Za-z]+$")  # "11.99A", "8.99N"
    _NON_PRODUCT    = re.compile(
        r"\b(tax|saving|savings|discount|instant\s+saving|subtotal|total|"
        r"redemp|crv|deposit|donation|charity|bag\s+fee|bottle\s+dep|"
        # Payment terminal / card slip lines
        r"approved|customer\s+copy|card\s+number|retain\s+this|"
        r"ref\.?\s*#|auto\s*#|visa\s+credit|entry\s+id|datetime|"
        r"transaction\s+id|debit\s+card|credit\s+card|"
        # Promotional / footer text
        r"fuel\s+points|thank\s+you\s+for\s+shopping|earn\s+\d+|"
        r"opportunity\s+awaits|join\s+our\s+team|now\s+hiring|feedback|"
        r"closing\s+balance|points\s+redeemed|pc\s+optimum|"
        # Sale/discount modifier lines (not standalone products)
        r"member\s+saving|digital\s+coupon|coupon\s+saving|"
        r"instant\s+saving|price\s+reduction|"
        # Payment noise
        r"balance\s+due|change\s+due|cash\s+back|gift\s+card)\b",
        re.IGNORECASE,
    )
    # Structural junk: URLs, EMV AIDs (A000...), "Item N" placeholders,
    # approval code lines ("00 APPROVED"), standalone short codes ("SC")
    _JUNK_NAME = re.compile(
        r"(www\.|\.com\b|\.org\b|jobs\.|"          # URLs
        r"^[Aa][0-9a-fA-F]{8,}|"                   # EMV AID e.g. A0000000031010
        r"^\d{2,3}\s+[A-Z]{2,}|"                   # "00 APPROVED", "03 DECLINED"
        r"^item\s*\d*$|"                            # "Item", "Item 1", "Item 2"
        r"^\*{2,}|"                                 # "*** CUSTOMER COPY ***"
        r"^\*\d+|"                                  # card number fragments: "*8424"
        r"^mgr:|^date:|^time:|"                     # manager/timestamp footer fields
        r"^account\s*:|^card\s+type\s*:|"           # payment terminal fields
        r"^trans(?:action)?\s*,?\s+type\s*:|"       # "Trans, Type: PURCHASE"
        r"^rec#|"                                   # receipt reference numbers: "REC#2-5279"
        r"^\(sale\)\s*$|"                           # bare "(SALE)" line
        r"^\(?\d{3}\)?[\s\-]\d{3}[\s\-]\d{4}|"     # phone numbers (604) 688-0911
        r"^\d{2}/\d{2}/\d{2,4}\s+\d{1,2}:\d{2}|"  # timestamps 02/19/26 7:53:13 PM
        r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}|"       # ISO timestamps 2026-02-19 07:53
        r"^\d+[A-Za-z]{1,2}\s+\d+$|"               # OCR noise: "1mt 4", "3oz 2"
        r"^[A-Za-z]{1,2}mt\s+\d+$|"                # OCR noise: "Imt 6", "imt 4"
        r"^SC\s+\d+|"                               # Ingles store codes: "SC 3547", "SC 3547 A"
        r"^\$[\d,]+$|"                              # dollar-prefixed numbers: "$15,000,000"
        r"@\s*\$?\d+\.?\d*\s*/|"                   # per-unit pricing lines: "NET 1b @ $1.49/1b", "1.160 kg @ $1.72/kg"
        r"^[a-z]{4,8}[0-9]?$)",                    # all-lowercase garbled OCR: "euoju", "emo2"
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

        # Drop names that are implausibly long — legal notices, Prop-65 warnings,
        # multi-sentence footer text extracted as a single item.
        if len(name) > 100:
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
        item_code = (item.get("itemCode") or "").strip()
        is_dup = False
        for kept in deduped:
            kept_name  = _norm_for_dedup(kept.get("productName") or "")
            kept_price = re.sub(r"[A-Za-z]+$", "", (kept.get("price") or "").strip())
            kept_code  = (kept.get("itemCode") or "").strip()
            if kept_price != price_raw:
                continue   # different price → definitely different items
            # Different non-empty itemCodes → distinct purchases, never deduplicate
            if item_code and kept_code and item_code != kept_code:
                continue
            # Same price: check name similarity
            if name_raw == kept_name:
                # Same name + same price but no itemCode to distinguish → only
                # deduplicate if there is already a kept entry with a null/empty
                # itemCode (chunk-merge artifact). If both have codes they were
                # already handled above; if neither has a code we treat it as a
                # duplicate to avoid double-counting LLM hallucinations.
                is_dup = True
                break
            if name_raw and kept_name:
                ratio = difflib.SequenceMatcher(None, name_raw, kept_name).ratio()
                if ratio >= 0.85:   # 85% similarity at same price → duplicate
                    is_dup = True
                    break
                # Token-overlap check: catches abbreviated vs full-name pairs
                # e.g. "ORG FR EGGS" vs "Organic Free Range Eggs" at same price
                a_toks = set(re.findall(r"[a-z0-9]{3,}", name_raw))
                b_toks = set(re.findall(r"[a-z0-9]{3,}", kept_name))
                if a_toks and b_toks:
                    overlap = len(a_toks & b_toks) / min(len(a_toks), len(b_toks))
                    if overlap >= 0.6:  # ≥60% token overlap at same price → duplicate
                        is_dup = True
                        break
        if not is_dup:
            deduped.append(item)

    return deduped


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


# Regex for dates paired with a transaction time — the most reliable way to
# identify the actual purchase timestamp vs promotional/contest dates.
_TX_DATE_TIME_RE = re.compile(
    r'(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\s+\d{1,2}:\d{2}'   # MM/DD/YY HH:MM
    r'|(\d{4}[/\-]\d{2}[/\-]\d{2})\s+\d{2}:\d{2}',           # YYYY-MM-DD HH:MM
)
# Months for written-out dates like "Feb 10 2026"
_MONTH_NAMES = {
    "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
    "jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12,
}
_WRITTEN_DATE_RE = re.compile(
    r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2})\s+(20\d{2})\b',
    re.IGNORECASE,
)


def _extract_transaction_date(ocr_text: str) -> str | None:
    """
    Scan raw OCR for a date that is paired with a transaction time (HH:MM).
    Returns YYYY.MM.DD string, or None if no clear transaction date found.

    Prefers:
      1. A date+time pair where year ≥ 2022 (filters 2-digit year ambiguity).
      2. A written-out date like "Feb 10 2026" (unambiguous 4-digit year).
    Skips dates that appear after promotional keywords (through/until/expires).
    """
    raw = ocr_text.split("\n---\n")[0] if "\n---\n" in ocr_text else ocr_text

    # Mark lines that are promotional context so we can skip their dates.
    promo_re = re.compile(r'\b(through|until|expires?|ending|by|enter\s+by|earn\s+plays)\b', re.IGNORECASE)

    best: str | None = None

    for m in _TX_DATE_TIME_RE.finditer(raw):
        # Check if this date appears after a promo keyword on the same/preceding line
        line_start = raw.rfind('\n', 0, m.start()) + 1
        surrounding = raw[max(0, line_start - 80):m.start()]
        if promo_re.search(surrounding):
            continue

        raw_date = m.group(1) or m.group(2)
        parts = re.split(r'[/\-]', raw_date)
        if len(parts) != 3:
            continue
        try:
            a, b, c = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue

        # Determine YYYY.MM.DD based on length of first part
        if len(parts[0]) == 4:          # YYYY-MM-DD
            yr, mo, dy = a, b, c
        elif c >= 100:                  # MM/DD/YYYY or DD/MM/YYYY — year is last
            yr = c
            # Guess MM/DD if first value ≤ 12, else DD/MM
            if a <= 12 and b <= 31:
                mo, dy = a, b
            else:
                mo, dy = b, a
        else:                           # 2-digit year: MM/DD/YY
            yr = 2000 + c
            if a <= 12 and b <= 31:
                mo, dy = a, b
            else:
                mo, dy = b, a

        if yr < 2022 or mo < 1 or mo > 12 or dy < 1 or dy > 31:
            continue
        best = f"{yr}.{mo:02d}.{dy:02d}"
        break  # first valid date+time pair wins

    # Fallback: look for an explicit written date with 4-digit year (e.g. "Feb 10 2026")
    if not best:
        for m in _WRITTEN_DATE_RE.finditer(raw):
            surrounding = raw[max(0, m.start() - 80):m.start()]
            if promo_re.search(surrounding):
                continue
            mo = _MONTH_NAMES.get(m.group(1)[:3].lower())
            dy = int(m.group(2))
            yr = int(m.group(3))
            if mo and 2022 <= yr <= 2030 and 1 <= dy <= 31:
                best = f"{yr}.{mo:02d}.{dy:02d}"
                break

    return best


# Known location/neighborhood names that LLMs sometimes hallucinate as store names.
# Maps OCR-visible chain keyword → canonical store name.
_LOCATION_OVERRIDES: dict[str, str] = {
    "TARGET":   "Target",
}
_KNOWN_CHAIN_RE = re.compile(
    r'\b(target|kroger|walmart|wal-mart|vons|ralphs|safeway|albertsons|costco'
    r'|no\s+frills|real\s+canadian|t&t\s+supermarket|your\s+independent\s+grocer'
    r'|independent\s+grocer|kin\'?s\s+farm|house\s+of\s+dosa|ingles|farm\s*&\s*table'
    r'|publix|heb|whole\s+foods|trader\s+joe|marquis\s+wine)\b',
    re.IGNORECASE,
)
_CHAIN_NAMES: dict[str, str] = {
    "target": "Target", "kroger": "Kroger", "walmart": "Walmart",
    "vons": "Vons", "ralphs": "Ralphs", "safeway": "Safeway",
    "costco": "Costco Wholesale", "ingles": "Ingles",
}


def _correct_store_name_from_ocr(store_name: str, ocr_text: str) -> str:
    """
    If the LLM returned a location/neighborhood name instead of a chain name,
    scan the first 10 lines of OCR for a known chain keyword and substitute.
    """
    # If the current store name already contains a known chain keyword, keep it.
    if _KNOWN_CHAIN_RE.search(store_name):
        return store_name

    # Otherwise scan OCR header (first 10 lines) for a chain name.
    header = "\n".join(ocr_text.splitlines()[:10])
    m = _KNOWN_CHAIN_RE.search(header)
    if m:
        key = m.group(1).lower().split()[0]  # first significant word
        return _CHAIN_NAMES.get(key, m.group(1).title())

    return store_name


def _is_hallucinated(result: dict, ocr_text: str) -> bool:
    """
    Return True if the extracted result looks like a hallucination — i.e. the
    LLM invented items that do not appear anywhere in the OCR text.

    Strategy: take the first significant word (>3 chars) of each extracted
    product name and check whether it appears in the OCR.  If NONE of the
    first 5 items have any word found in the OCR, the model almost certainly
    fabricated the receipt.
    """
    items = result.get("items") or []
    if not items:
        return False
    ocr_lower = ocr_text.lower()
    checked = 0
    for item in items[:5]:
        name = (item.get("productName") or "").strip().lower()
        words = [w for w in re.split(r"\W+", name) if len(w) > 3]
        if not words:
            continue
        checked += 1
        if any(w in ocr_lower for w in words):
            return False   # at least one item found → not hallucinated
    # If we checked items and found none in the OCR → hallucination
    return checked > 0


