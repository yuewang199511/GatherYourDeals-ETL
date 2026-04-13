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
import sys
import time
import uuid
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from src.reporting import eval_receipts, baseline_report  # noqa: E402 (after load_dotenv)
from src.etl_logger import (  # noqa: E402
    log_adi, log_llm, log_pipeline, log_upload, ADI_COST_PER_PAGE,
)

# OR, if you want to use them directly without the 'prompts.' prefix:
from src.core import config, prompts
from src.services import llm, ocr, geo

# Then reference it like this:
if not config.AZURE_DI_KEY:
    print("Error: Azure Key missing!")


# 1. Define the folder you want to scan (e.g., your Receipts folder)
# You can use the Path from your config if you set one up
folder = Path("Receipts")

image_list = [f for f in folder.glob("*") if f.suffix.lower() in config.IMAGE_EXTS]


# --- Concurrency Control ---
_MAX_CONCURRENT_OCR = 5
# Note: Ensure ocr_semaphore is defined inside your async loop or 
# at the module level if using a global loop.
ocr_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_OCR)

async def throttled_ocr(image_data, display_name, run_id, user_id, use_cache):
    """
    Limits concurrent calls to Azure to avoid 429 (Too Many Requests) 
    and manage system resources.
    """
    async with ocr_semaphore:
        # Define a small wrapper to handle the class lifecycle
        def run_sync():
            # 1. Instantiate the service
            service = ocr.AzureOCRService(
                image_data, 
                display_name, 
                run_id, 
                user_id=user_id, 
                use_cache=use_cache
            )
            # 2. Call the method that actually returns the string/text
            # Replace '.process()' with whatever your main method is called (e.g., .run() or .get_text())
            return service.process() 

        return await asyncio.to_thread(run_sync)

# ---------------------------------------------------------------------------
# Railtracks — flow orchestration + observability
# ---------------------------------------------------------------------------
try:
    import railtracks as rt
    _RT_AVAILABLE = True
except ImportError:
    _RT_AVAILABLE = False

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
    config.LOGS_DIR.mkdir(exist_ok=True)
    rt.enable_logging(level="INFO", log_file=str(config.LOGS_DIR / "rt.log"))


def _build_system_prompt(ocr_text: str, use_direct: bool = False) -> str:
    # Use the imported variables
    base = prompts.SYSTEM_PROMPT_DIRECT if use_direct else prompts.SYSTEM_PROMPT
    
    ocr_upper = ocr_text.upper()[:1000] 
    addenda = []
    
    # Costco Specifics
    if "COSTCO" in ocr_upper:
        addenda.append(prompts.COSTCO_PROMPT_ADDENDUM)
    
    # You can easily add more store rules here later
    # if "WALMART" in ocr_upper:
    #     addenda.append(prompts.WALMART_ADDENDUM)

    return base + "\n".join(addenda)

# ---------------------------------------------------------------------------
# ETL Pipeline Nodes
# ---------------------------------------------------------------------------
async def ocr_node(image_path, run_id, user_id):
    ocr_service = ocr.AzureOCRService()
    
    # This replaces the massive block of code previously in etl.py
    ocr_text = await ocr_service.perform_ocr(
        image_data=image_path,
        display_name=image_path.name,
        run_id=run_id,
        user_id=user_id
    )
    
    return ocr_text


# ---------------------------------------------------------------------------
# Output flattening — denormalize receipt metadata into per-item records
# ---------------------------------------------------------------------------

_VALID_PRICE_RE = re.compile(r"^\d+\.\d{2}[A-Z]{3}$")
_FLAT_NON_PRODUCT = re.compile(
    r"\b(donation|charity|bag\s+fee|bottle\s+dep|deposit|recycling|crv|redemption|"
    r"tax|subtotal|total|savings?|discount|coupon|reward|point|loyalty|"
    r"balance\s+due|change\s+due|cash\s+back|gift\s+card)\b",
    re.IGNORECASE,
)


def flatten_receipt(receipt: dict) -> list[dict]:
    """
    Convert a structured receipt dict into flat per-item records.

    Each record contains only the 7 target output fields:
      productName, purchaseDate, price, amount, storeName, latitude, longitude

    Items are dropped if:
      - productName or price is null/empty
      - price does not match the X.XXCURRENCY format (e.g. "3.69USD")
      - productName matches a non-product keyword pattern
      - productName is entirely lowercase (garbled OCR — real receipt names are CAPS/Title)
      - productName contains the store name (header line leaked in as a product)

    After filtering, substring deduplication removes fragment names where a longer
    name at the same price already exists (e.g. "GRD A LRG" when "GRD A LRG BRWN MRJ"
    is also present at the same price).
    """
    store_name    = receipt.get("storeName") or None
    purchase_date = receipt.get("purchaseDate") or None
    lat           = receipt.get("latitude")
    lon           = receipt.get("longitude")

    # Normalised store name for containment check (e.g. "no frills", "kroger")
    store_lower = (store_name or "").lower()

    flat_items: list[dict] = []
    for item in receipt.get("items", []):
        name  = (item.get("productName") or "").strip()
        price = (item.get("price") or "").strip()

        if not name or not price:
            continue
        if not _VALID_PRICE_RE.match(price):
            continue
        if _FLAT_NON_PRODUCT.search(name):
            continue

        # General fix 1: all-lowercase name → garbled OCR noise.
        # Real receipt product names are always CAPS or Title Case.
        if name == name.lower() and any(c.isalpha() for c in name):
            continue

        # General fix 2: store name contained in product name → header leaked in.
        if store_lower and store_lower in name.lower():
            continue

        flat_items.append({
            "productName":  name,
            "purchaseDate": purchase_date,
            "price":        price,
            "amount":       item.get("amount") or "1",
            "storeName":    store_name,
            "latitude":     lat,
            "longitude":    lon,
        })

    # General fix 3: substring dedup at same price.
    # If name A is a substring of name B and they share a price, drop A (keep the longer one).
    price_groups: dict[str, list[dict]] = {}
    for item in flat_items:
        price_groups.setdefault(item["price"], []).append(item)

    deduped: list[dict] = []
    for items_at_price in price_groups.values():
        names = [i["productName"].lower() for i in items_at_price]
        keep = []
        for i, item in enumerate(items_at_price):
            n = names[i]
            # Drop if any other name at this price fully contains this one (and is longer)
            if any(n != names[j] and n in names[j] for j in range(len(names))):
                continue
            keep.append(item)
        deduped.extend(keep)

    return deduped



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
    resolved_provider = (provider or config.LLM_PROVIDER).lower()

    # Tier 0 — global OCR normalisation before any other processing.
    # Step 1: fix spaced decimals ("1. 160" → "1.160", "1. 72" → "1.72").
    # Step 2: join dangling price lines onto the preceding item line so the
    #         LLM sees "BANANAS  2.00" rather than two unrelated lines.
    # Applied once here so every downstream tier — noise filter, chunker,
    # weight-price parser, and LLM prompt — all see clean, aligned text.
    ocr_text = llm._NORM_SPACED_NUM.sub(r'\1.\2', ocr_text)
    ocr_text = llm._join_split_price_lines(ocr_text)

    # Tier 1 — strip noise lines before the text reaches the LLM.
    # Reduces token count on large receipts and prevents total/tax rows
    # from being misidentified as product items.
    ocr_text = llm._filter_noise_lines(ocr_text)

    # Always run the chunker so it strips the raw OCR body when a SPATIAL
    # LAYOUT section is present (saves ~40% tokens on short receipts too).
    # For long receipts it also splits into overlapping sections as before.
    _SPATIAL_MARKER = "\n---\n## SPATIAL LAYOUT\n"
    if _SPATIAL_MARKER in ocr_text or len(ocr_text) > llm._CHUNK_THRESHOLD_CHARS:
        chunks = llm._split_ocr_into_chunks(ocr_text)
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
            # Use the imported utility! 
            # It handles provider branching, retries, and JSON parsing internally.
            llm_res = llm.structure_llm(
                provider=resolved_provider,
                ocr_text=chunk_text,
                model=model,
                system_prompt=_prompt
            )
            
            # Accumulate metrics from the LLMResult object
            total_pt += llm_res.input_tokens
            total_ct += llm_res.output_tokens
            total_cost += (llm_res.cost_usd or 0.0)
            
            chunk_results.append(llm_res.data)

        # Merge chunks (no-op when only one chunk)
        result = llm._merge_chunk_results(chunk_results)

        # Hallucination guard: if none of the extracted item names appear in the
        # OCR text, the model fabricated the receipt.  Retry once with the same
        # model; if it hallucinates again, fall back to the other CLOD model.
        if llm._is_hallucinated(result, ocr_text):
            _FALLBACK_MODEL = "openai/gpt-4o-mini" # Or your preferred fallback
            retry_model = _FALLBACK_MODEL if model != _FALLBACK_MODEL else model
            retry_chunks: list[dict] = []
            
            for chunk_text in chunks:
                try:
                    # Use the same utility for the retry!
                    retry_res = llm.structure_llm(resolved_provider, chunk_text, retry_model, _prompt)
                    total_pt += retry_res.input_tokens
                    total_ct += retry_res.output_tokens
                    total_cost += (retry_res.cost_usd or 0.0)
                    retry_chunks.append(retry_res.data)
                except Exception:
                    pass
            if retry_chunks:
                result = llm._merge_chunk_results(retry_chunks)

        # Deterministic post-processing: drop bad rows, fix column swaps

        # Tier 2c — normalise store name first so Canadian-store CAD inference works.
        if result.get("storeName"):
            result["storeName"] = llm._normalize_store_name(result["storeName"])
            result["storeName"] = llm._correct_store_name_from_ocr(result["storeName"], ocr_text)

        # Tier 2d — override LLM date with deterministic OCR scan when model picks
        # a promotional date (e.g. contest end date) instead of the transaction date.
        ocr_date = llm._extract_transaction_date(ocr_text)
        if ocr_date:
            result["purchaseDate"] = ocr_date

        # Override LLM-extracted currency with deterministic OCR scan so
        # small models that default to "USD" are corrected for CA/GB/EU receipts.
        # Fall back to Canadian-store inference when OCR has no explicit marker.
        ocr_currency = ocr._detect_currency_from_ocr(ocr_text)
        if ocr_currency is None:
            ocr_currency = llm._infer_currency_from_store(result.get("storeName") or "")
        currency = ocr_currency or result.get("currency") or "USD"
        result["currency"] = currency

        # Tier 2+3 — deterministic post-processing
        result["items"] = llm._validate_and_fix_items(result.get("items", []), currency)

        # Tier 2b — recover prices for weight-priced items (e.g. "1.160 kg @ $1.72/kg 2.00")
        # Do this before the null-price repair so weight items don't consume repair budget.
        result["items"] = llm._inject_weight_prices(result["items"], ocr_text, currency)

        # Tier 3b+4 — targeted repair for items with null price, then escalate
        null_price_count = sum(1 for i in result["items"] if not (i.get("price") or "").strip() or (i.get("price") or "").lower() == "null")
        if null_price_count:
            result["items"] = llm._repair_failed_items(
                result["items"], ocr_text, model, resolved_provider, currency
            )

        latency_ms = (time.monotonic() - start) * 1000

        # Inject caller-controlled fields
        result["totalItems"] = len(result["items"])
        result["imageName"] = display_name
        result["userName"]  = user_name
        # Flatten and then OVERWRITE the items list with the cleaned version
        # This ensures that 'items' only contains real products for the upload.
        cleaned_rows = flatten_receipt(result)
        result["items"] = cleaned_rows

        # Geocode once using merged store address.
        # Fallback: if LLM returned no storeAddress, extract it from the OCR text
        # by looking for a line that contains digits + street keywords near the top.
        # 1. Resolve the address (use fallback if LLM missed it)
        address = result.get("storeAddress", "").strip()
        if not address or len(address) < 12:
            address = llm._extract_address_from_ocr(ocr_text, result.get("storeName") or "")
            result["storeAddress"] = address

        # 2. Get the coordinates from your updated geo.py
        short_name = (result.get("storeName") or "").split(" ")[0]
        lat, lon = geo.geocode(address, store_name=short_name)

        # 3. Update the nested items list with coordinates and filter junk
        if "items" in result:
            valid_items = []
            
            for item in result["items"]:
                name = item.get("productName", "").strip()
                
                # --- NEW: Aggressive Barcode/Junk Filter ---
                # This catches: "003700071650H" (numbers + H) 
                # or anything that is mostly just a long ID number.
                is_barcode = any([
                    name.isdigit() and len(name) > 8,
                    len(name) > 10 and any(char.isdigit() for char in name[:8]) # Starts with numbers
                ])

                if is_barcode:
                    print(f"  [LLM] Skipping barcode/ID found in name: {name}")
                    continue # Drop it from the final list
                
                # --- FILTERING LOGIC ---
                if "%" in name or "TAX" in name.upper():
                    continue
                
                if len(name) <= 1:
                    continue
                
                if any(stop in name.upper() for stop in ["SUBTOTAL", "TOTAL", "CASH", "CHANGE"]):
                    continue
                # -----------------------

                # --- BARCODE WARNING (Inside the loop) ---
                if name.isdigit() and len(name) > 10:
                     print(f"  [LLM]  Warning: Found barcode instead of name: {name}")

                # Attach the data
                item["latitude"] = lat
                item["longitude"] = lon
                item["purchaseDate"] = result.get("purchaseDate")
                item["storeName"] = result.get("storeName")
                
                valid_items.append(item)
            
            # Replace with the cleaned version
            result["items"] = valid_items
                
            # Final Status Print (outside the loop)
            if lat and lon:
                print(f"  [GEO]  Successfully tagged {len(result['items'])} items with coordinates.")
            else:
                print(f"  [GEO]  Warning: Geocoding failed. Coordinates will be null.")

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


    @rt.function_node
    async def receipt_pipeline(ctx: OcrInput) -> StructureOutput:
        """Single-node pipeline: OCR → LLM/geocode in one Railtracks step."""
        image_path = Path(ctx.image_path)        
        
        # Use the throttled OCR function instead of direct to_thread
        ocr_text = await throttled_ocr(image_path, image_path.name, ctx.run_id, ctx.user_name, True)
        # 2. Explicitly check for None or empty string
        if ocr_text is None:
            print(f"❌ ERROR: throttled_ocr returned None for {image_path.name}")
            # Handle the error or return early
            return 
            
        print(f"DEBUG: OCR Result: {len(ocr_text)} chars found.")
        # asyncio.to_thread(AzureOCRService, image_path, image_path.name, ctx.run_id, ctx.user_name)

        # await rt.broadcast(f"[OCR] Starting — {image_path.name} ({image_path.stat().st_size:,} bytes)")
        await rt.broadcast(
            f"[LLM] Starting — provider={ctx.provider}  model={ctx.model}  "
            f"input={len(ocr_text)} chars"
        )
        result, total_pt, total_ct, total_cost = await asyncio.to_thread(
            structure, ocr_text, image_path.name, ctx.user_name,
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
    resolved_provider = (provider or config.LLM_PROVIDER).lower()

    # Railtracks path requires a file on disk — only available when image_data is a Path.
    if isinstance(image_data, bytes):
        # Create a temp file so Railtracks has a physical path to work with
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(image_data)
            tmp_path = tmp.name
        image_to_process = tmp_path
    else:
        image_to_process = str(image_data)

    flow = rt.Flow(name="receipt_etl", entry_point=receipt_pipeline)
    result = flow.invoke(OcrInput(
        image_path=image_to_process,
        run_id=run_id,
        user_name=user_name,
        model=model,
        provider=resolved_provider,
    ))
    return json.loads(result.receipt_json)

    # if _RT_AVAILABLE and isinstance(image_data, Path):
    #     flow = rt.Flow(name="receipt_etl", entry_point=receipt_pipeline)
    #     result: StructureOutput = flow.invoke(OcrInput(
    #         image_path=str(image_data),
    #         run_id=run_id,
    #         user_name=user_name,
    #         model=model,
    #         provider=resolved_provider,
    #     ))
    #     return json.loads(result.receipt_json)
    # else:
    #     cache_stem = Path(display_name).stem if display_name else "unknown"
    #     _cache_hit = use_cache and (config.OCR_CACHE_DIR / (cache_stem + ".txt")).exists()
    #     print(f"  [ADI]  OCR {'(cached)' if _cache_hit else '…'}")
    #     ocr_text = ocr.AzureOCRService(image_data, display_name, run_id, user_id=user_name, use_cache=use_cache)
    #     print(f"  [LLM]  Structuring via {resolved_provider} ({len(ocr_text)} chars) …")
    #     result, _, _, _ = structure(ocr_text, display_name, user_name, model, run_id, provider=resolved_provider)
    #     return result


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
    resolved_token = token or config.GYD_ACCESS_TOKEN
    client = GYDClient(config.GYD_SERVER_URL, auto_persist_tokens=False)
    if resolved_token:
        client._transport.set_tokens(resolved_token, refresh_token or "")

    items   = receipt.get("items", [])
    created, failed = [], 0
    start   = time.monotonic()

    image_name  = receipt.get("imageName", "")

    _UPLOAD_MAX_RETRIES = 3
    _UPLOAD_RETRY_DELAY = 1.0  # seconds between retries

    for item in items:
        product_name = item.get("productName", "")
        # --- DEFINE THESE FIRST ---
        purchase_date = receipt.get("purchaseDate", "0000.00.00")
        price = item.get("price", "0.00USD")
        store = receipt.get("storeName", "Unknown")
        # --------------------------
        last_exc = None
        for attempt in range(1, _UPLOAD_MAX_RETRIES + 1):
            try:
                # Keep product_name as a string unless the SDK specifically asked for a tuple
                r = client.receipts.create(
                    product_name=product_name, 
                    purchase_date=purchase_date,
                    price=price,
                    amount=str(item.get("amount", "1")),
                    store_name=store,
                )
                # Now print after successful creation
                print(f"[{run_id[:8]:<8}] {purchase_date:10}  {product_name:<25}  {price:>10}  @ {store}")
                
                created.append(r)
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                if attempt < _UPLOAD_MAX_RETRIES:
                    print(f"    [WARN] upload attempt {attempt}/{_UPLOAD_MAX_RETRIES} failed for '{product_name}': {e} — retrying in {_UPLOAD_RETRY_DELAY}s")
                    time.sleep(_UPLOAD_RETRY_DELAY)
        if last_exc is not None:
            print(f"    [ERROR] upload failed for '{product_name}' after {_UPLOAD_MAX_RETRIES} attempts: {last_exc}")
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
    if config._UPLOAD_REGISTRY.exists():
        try:
            return json.loads(config._UPLOAD_REGISTRY.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _registry_save(image_stem: str, ids: list[str]) -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = _registry_load()
    registry[image_stem] = ids
    config._UPLOAD_REGISTRY.write_text(json.dumps(registry, indent=2, ensure_ascii=False),
                                encoding="utf-8")




# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="GatherYourDeals receipt ETL (ADI + LLM)")
    p.add_argument("path",        nargs="?", help="Image file or directory")
    p.add_argument("--user",      default="unknown", help="Username for JSON metadata")
    p.add_argument("--provider",  default=config.LLM_PROVIDER,
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
        resolved_model = args.model or config.CLOD_DEFAULT_MODEL
    else:
        resolved_model = args.model or config.OR_DEFAULT_MODEL

    if args.eval:
        eval_receipts(); return

    if args.baseline_report:
        baseline_report(); return

    if not args.path:
        p.print_help(); sys.exit(1)

    target = Path(args.path)
    run_id = str(uuid.uuid4())

    images = (sorted(f for f in target.iterdir() if f.suffix.lower() in config.IMAGE_EXTS)
              if target.is_dir() else [target] if target.is_file() else [])
    if not images:
        print(f"No images found at {target}"); sys.exit(1)

    do_upload = not args.no_upload and bool(config.GYD_SERVER_URL)
    if not do_upload and not args.no_upload:
        print("[INFO] GYD_SERVER_URL not set — extract-only mode.")

    errors = 0
    for img in images:
        print(f"\n→ {img.name}")
        _start = time.monotonic()
        try:
            data = extract(img, img.name, args.user, resolved_model, run_id, provider=args.provider,
                           use_cache=not args.no_ocr_cache)
            total_ms = (time.monotonic() - _start) * 1000
            log_pipeline(run_id, img.name, args.user, args.provider, resolved_model, total_ms, True)
            
            rows = data["items"]

            model_slug = resolved_model.split("/")[-1].lower()
            provider_out_dir = config.OUTPUT_DIR / f"{args.provider}-{model_slug}"
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
