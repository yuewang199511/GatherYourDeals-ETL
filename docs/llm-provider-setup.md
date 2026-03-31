# LLM Provider Setup & Change Log

This document covers:
1. How to configure the LLM provider for Step 2 of the ETL pipeline
2. A record of all prompt and model changes made during development

---

## Providers

The ETL supports two LLM providers for Step 2 (OCR text → structured JSON).
Select one via the `LLM_PROVIDER` environment variable or the `--provider` CLI flag.

| Provider | Env var | Default model | Cost |
|----------|---------|---------------|------|
| OpenRouter | `OPENROUTER_API_KEY` | `anthropic/claude-haiku-4.5` | ~$0.004/receipt |
| CLOD | `CLOD_API_KEY` | `Qwen/Qwen2.5-7B-Instruct-Turbo` | ~$0.0004/receipt (sponsored) |

---

## Configuration (`.env`)

```env
# ── LLM Provider ──────────────────────────────────────────────────
# Which provider to use: "openrouter" (default) or "clod"
LLM_PROVIDER=openrouter

# Step 2a — OpenRouter
# Sign up at openrouter.ai → Keys → Create Key
OPENROUTER_API_KEY=sk-or-v1-...
OR_DEFAULT_MODEL=anthropic/claude-haiku-4.5

# Step 2b — CLOD  (set LLM_PROVIDER=clod to use)
# Sign up at app.clod.io → API Keys → Create Key
# CLOD_API_KEY=<your-key>
# OR_DEFAULT_MODEL=Qwen/Qwen2.5-7B-Instruct-Turbo
```

---

## CLI Usage

```bash
# OpenRouter (default)
python3 etl.py receipt.jpg --user $GYD_USERNAME --no-upload

# OpenRouter with explicit model
python3 etl.py receipt.jpg --user $GYD_USERNAME --provider openrouter --model anthropic/claude-haiku-4.5 --no-upload

# CLOD
python3 etl.py receipt.jpg --user $GYD_USERNAME --provider clod --model Qwen/Qwen2.5-7B-Instruct-Turbo --no-upload

# Whole directory
python3 etl.py Receipts/ --user $GYD_USERNAME --no-upload

# With upload to GYD data service
python3 etl.py Receipts/ --user $GYD_USERNAME
```

Model resolution order: `--model` flag → `OR_DEFAULT_MODEL` env var → provider default.

---

## Available Free Models on OpenRouter

The model `google/gemini-2.0-flash-exp:free` was retired in March 2026.
The following models are confirmed free as of 2026-03-25:

| Model ID | Notes |
|----------|-------|
| `anthropic/claude-haiku-4.5` | **Current default** — best accuracy for receipts |
| `deepseek/deepseek-chat` | Good alternative |
| `amazon/nova-lite-v1` | Fast, reliable |
| `amazon/nova-micro-v1` | Fastest, smallest |

To see all currently free models:

```bash
python3 -c "
import json, urllib.request, os
from dotenv import load_dotenv; load_dotenv()
req = urllib.request.Request('https://openrouter.ai/api/v1/models',
    headers={'Authorization': f'Bearer {os.getenv(\"OPENROUTER_API_KEY\")}'})
data = json.loads(urllib.request.urlopen(req).read())
for m in sorted(data['data'], key=lambda x: x['id']):
    if float(m.get('pricing', {}).get('prompt', 1)) == 0:
        print(m['id'])
"
```

---

## Model Pricing Reference

| Model | Provider | Input (per 1M) | Output (per 1M) | Notes |
|-------|----------|----------------|-----------------|-------|
| `anthropic/claude-haiku-4.5` | OpenRouter | $1.00 | $5.00 | Current default |
| `Qwen/Qwen2.5-7B-Instruct-Turbo` | CLOD | $0.30 | $0.12 | 60% discount applied; output dynamically priced, cap ~$0.318/M |
| `google/gemini-2.0-flash` | OpenRouter | $0.10 | $0.40 | |
| `openai/gpt-4o-mini` | OpenRouter | $0.15 | $0.60 | |

---

## Change Log

### 2026-03-29 — Provider overhaul, cost logging fixes, new metrics

**Changes:**

| Area | Change |
|------|--------|
| Provider | Replaced Claude/Anthropic native provider with CLOD (`Qwen/Qwen2.5-7B-Instruct-Turbo`) as the alternative LLM provider |
| Model slug | `anthropic/claude-haiku-4-5` → `anthropic/claude-haiku-4.5` (correct OpenRouter slug) |
| ADI cost | Always logged at S0 rate ($0.0015/page) for production-accurate cost tracking |
| OpenRouter cost | Added `_fetch_openrouter_cost()` with 0.5s delay + 3 retries; token-based fallback via `_OR_PRICING` when API unavailable |
| CLOD cost | Added `_CLOD_PRICING` token-based fallback ($0.30/$0.12 per M tokens) when API response omits cost |
| Store normalization | Added `normalize_store_name()` — second LLM call matches raw extracted store name to canonical ground truth list |
| New log event | `pipeline_complete` — logs end-to-end wall time per receipt (OCR + LLM + geocode + normalization) |
| Baseline report | Now includes E2E P50/P95, throughput (receipts/min), and failure rate per provider |
| Baseline report | Field-level accuracy (`--eval`) automatically appended to `--baseline-report` output |
| Baseline runs | `run_baseline.sh` updated to 3× runs per provider for statistical significance |

### 2026-03-25 — OpenRouter provider fixes

**Problem:** `google/gemini-2.0-flash-exp:free` returned HTTP 404 (model retired).

**Fix:** Updated default model to `anthropic/claude-3-haiku` in both `.env` and
`etl.py` (`_DEFAULT_MODELS["openrouter"]`).

---

### 2026-03-25 — Added Claude (Anthropic) as a second LLM provider

**Motivation:** Support native Anthropic API as an alternative to OpenRouter,
enabling use of the latest Claude models directly.

**Changes to `etl.py`:**

| Area | Change |
|------|--------|
| Config | Added `ANTHROPIC_API_KEY`, `LLM_PROVIDER` env vars |
| Pricing | Added `claude-haiku-4-5-20251001`, `claude-sonnet-4-6`, `claude-opus-4-6` entries |
| `structure()` | Refactored into `_structure_openrouter()` + `_structure_claude()` backends; `structure()` dispatches based on `provider` arg |
| `extract()` | Passes `provider` through to `structure()` |
| CLI | Added `--provider {openrouter,claude}` flag; `--model` now defaults per-provider |
| Install | Added `anthropic` to required packages |

**New `.env` variables:**
```env
ANTHROPIC_API_KEY=sk-ant-...
LLM_PROVIDER=openrouter          # or "claude"
```

**New CLI flags:**
```bash
--provider {openrouter,claude}   # selects the LLM backend
--model MODEL                    # overrides per-provider default
```

---

### 2026-03-25 — System prompt improvements

**Problem:** Running the Vons receipt (`2025-10-01Vons.jpg`) with `anthropic/claude-3-haiku`
via OpenRouter produced three errors:

| Field | Actual output | Expected |
|-------|---------------|----------|
| `amount` | `"4.99"` (original price) | `"1"` (quantity) |
| `price` | `"4.79"` (no currency) | `"4.79USD"` |
| Item count | 6 items | 7 items (Donation was skipped) |

**Root cause:** The prompt did not explicitly distinguish `amount` (count/weight) from
price, did not require the currency code suffix, and did not require all line items
to be included.

**Fix:** Added the following rules to `_SYSTEM_PROMPT` in `etl.py`:

```
- price must always include the currency code: "4.79USD" not "4.79"
- amount is the number of units or weight, never a price
- Include EVERY line item — do not skip donations, fees, or miscellaneous charges
- If a product has a member/club discount, use the discounted (final) price
```

**Result after fix:** 7/7 items extracted, all prices include currency code,
all amounts are quantities. Output matches ground truth.

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `No endpoints found for google/gemini-2.0-flash-exp:free` | Model retired | Change `OR_DEFAULT_MODEL` to `anthropic/claude-haiku-4.5` |
| `OPENROUTER_API_KEY not set` | Missing `.env` entry | Add key to `.env` |
| `ANTHROPIC_API_KEY not set` | Using `--provider claude` without key | Add `ANTHROPIC_API_KEY` to `.env` |
| `Anthropic SDK not installed` | Missing package | `pip install anthropic` |
| `401 Unauthorized` | Wrong API key | Check key in `.env` matches dashboard |
| `429 Too Many Requests` | Rate limit hit | Wait 1 min or switch to a paid model |
