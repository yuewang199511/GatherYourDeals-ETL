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
| OpenRouter | `OPENROUTER_API_KEY` | `anthropic/claude-3-haiku` | Free |
| Claude (Anthropic) | `ANTHROPIC_API_KEY` | `claude-haiku-4-5-20251001` | Paid |

---

## Configuration (`.env`)

```env
# ── LLM Provider ──────────────────────────────────────────────────
# Which provider to use: "openrouter" (default) or "claude"
LLM_PROVIDER=openrouter

# Step 2a — OpenRouter
# Sign up at openrouter.ai → Keys → Create Key
OPENROUTER_API_KEY=sk-or-v1-...
ETL_MODEL=anthropic/claude-3-haiku

# Step 2b — Claude / Anthropic  (set LLM_PROVIDER=claude to use)
# Sign up at console.anthropic.com → API Keys
ANTHROPIC_API_KEY=sk-ant-...
# ETL_MODEL=claude-haiku-4-5-20251001
```

---

## CLI Usage

```bash
# OpenRouter (default, uses LLM_PROVIDER from .env)
python3 etl.py receipt.jpg --user alice --no-upload

# OpenRouter with explicit provider and model
python3 etl.py receipt.jpg --user alice --provider openrouter --model anthropic/claude-3-haiku --no-upload

# Claude (Anthropic native API)
python3 etl.py receipt.jpg --user alice --provider claude --model claude-haiku-4-5-20251001 --no-upload

# Whole directory
python3 etl.py ./Receipts/converted/ --user alice --no-upload

# With upload to GYD API
python3 etl.py ./Receipts/converted/ --user alice
```

Model resolution order: `--model` flag → `ETL_MODEL` env var → provider default.

---

## Available Free Models on OpenRouter

The model `google/gemini-2.0-flash-exp:free` was retired in March 2026.
The following models are confirmed free as of 2026-03-25:

| Model ID | Notes |
|----------|-------|
| `anthropic/claude-3-haiku` | **Current default** — best accuracy for receipts |
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

| Model | Provider | Input (per 1M) | Output (per 1M) |
|-------|----------|----------------|-----------------|
| `anthropic/claude-3-haiku` | OpenRouter | $0.00 | $0.00 |
| `deepseek/deepseek-chat` | OpenRouter | $0.00 | $0.00 |
| `google/gemini-2.0-flash` | OpenRouter | $0.10 | $0.40 |
| `openai/gpt-4o-mini` | OpenRouter | $0.15 | $0.60 |
| `openai/gpt-4o` | OpenRouter | $2.50 | $10.00 |
| `anthropic/claude-3.5-sonnet` | OpenRouter | $3.00 | $15.00 |
| `claude-haiku-4-5-20251001` | Anthropic | $0.80 | $4.00 |
| `claude-sonnet-4-6` | Anthropic | $3.00 | $15.00 |
| `claude-opus-4-6` | Anthropic | $15.00 | $75.00 |

---

## Change Log

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
| `No endpoints found for google/gemini-2.0-flash-exp:free` | Model retired | Change `ETL_MODEL` to `anthropic/claude-3-haiku` |
| `OPENROUTER_API_KEY not set` | Missing `.env` entry | Add key to `.env` |
| `ANTHROPIC_API_KEY not set` | Using `--provider claude` without key | Add `ANTHROPIC_API_KEY` to `.env` |
| `Anthropic SDK not installed` | Missing package | `pip install anthropic` |
| `401 Unauthorized` | Wrong API key | Check key in `.env` matches dashboard |
| `429 Too Many Requests` | Rate limit hit | Wait 1 min or switch to a paid model |
