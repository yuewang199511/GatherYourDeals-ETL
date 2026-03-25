# GatherYourDeals-ETL

Two-step receipt pipeline that reads a photo of a receipt and produces structured JSON.

```
Receipt photo  (JPG / PNG / WEBP / HEIC / TIFF / BMP)
    │
    ▼  Step 1 — Azure Document Intelligence  (prebuilt-read)
    │           Extracts all text from the image via OCR
    │
    ▼  Step 2 — LLM  (OpenRouter  or  Claude / Anthropic)
    │           Structures the OCR text into the GYD JSON format
    │
    ▼  output/<image-name>.json  +  optional upload to GYD data service
```

**Test run results (7 receipts, 2026-03-25):** 7/7 success · 89 items extracted · $0.00 cost

---

## Setup

### 1. Install dependencies

```bash
pip install openai anthropic python-dotenv azure-ai-documentintelligence
pip install git+https://github.com/yuewang199511/GatherYourDeals-SDK.git
pip install matplotlib              # optional — for --report charts
pip install pillow pillow-heif      # optional — only for HEIC (iPhone) photos
```

### 2. Azure Document Intelligence (Step 1 — OCR)

1. Go to [portal.azure.com](https://portal.azure.com)
2. **Create a resource** → search **Document Intelligence** → Create
3. Choose **Free tier (F0)**: 500 pages/month, no charge
4. After deployment: **Keys and Endpoint** → copy **Endpoint** and **Key 1**

See `docs/setup-azure-di.md` for detailed instructions.

### 3. LLM provider (Step 2 — structuring)

**Option A — OpenRouter** (default, free tier available)

1. Sign up at [openrouter.ai](https://openrouter.ai)
2. **Keys** → Create key
3. Default model is `anthropic/claude-3-haiku` — free, no credits needed

**Option B — Claude / Anthropic** (native API)

1. Sign up at [console.anthropic.com](https://console.anthropic.com)
2. **API Keys** → Create key
3. Set `LLM_PROVIDER=claude` in `.env`

See `docs/llm-provider-setup.md` for model options, pricing, and troubleshooting.

### 4. Configure `.env`

```bash
cp .env.example .env
```

Fill in your credentials:

```env
# Step 1 — Azure Document Intelligence
AZURE_DI_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/
AZURE_DI_KEY=<your-key>

# Step 2 — OpenRouter (default)
OPENROUTER_API_KEY=sk-or-v1-...
ETL_MODEL=anthropic/claude-3-haiku
LLM_PROVIDER=openrouter

# Step 2 — Claude / Anthropic (set LLM_PROVIDER=claude to use instead)
# ANTHROPIC_API_KEY=sk-ant-...
# ETL_MODEL=claude-haiku-4-5-20251001

# GYD data service — leave blank to run in extract-only mode
GYD_SERVER_URL=http://localhost:8080/api/v1
GYD_USERNAME=
GYD_PASSWORD=
```

---

## Usage

```bash
# Single receipt — extract only (no server needed)
python etl.py Receipts/converted/2025-10-01Vons.jpg --user lkim016 --no-upload

# Whole directory
python etl.py Receipts/converted/ --user lkim016 --no-upload

# Use Claude instead of OpenRouter
python etl.py Receipts/converted/ --user lkim016 --provider claude --no-upload

# With upload to GYD data service (set GYD_* in .env first)
python etl.py Receipts/converted/ --user lkim016

# Generate usage report + charts from logs
python etl.py --report
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--user USER` | `unknown` | Username written into the output JSON |
| `--provider {openrouter,claude}` | `LLM_PROVIDER` env var | LLM backend to use |
| `--model MODEL` | `ETL_MODEL` env var | Model ID — overrides provider default |
| `--no-upload` | off | Skip upload to GYD data service |
| `--report` | off | Generate Markdown report + chart from logs |

---

## Output format

Each receipt produces `output/<image-name>.json`.
`imageName` and `userName` are always present; all other fields are extracted from the receipt.

```json
{
  "imageName": "2025-10-01Vons.jpg",
  "userName": "lkim016",
  "storeName": "VONS",
  "storeAddress": "8010 East Santa Ana Cyn. ANAHEIM HILLS CA 92808",
  "purchaseDate": "2025.10.01",
  "purchaseTime": "17:34",
  "currency": "USD",
  "items": [
    {
      "productName": "OPN NAT GRANOLA",
      "itemCode": "7989311116",
      "price": "4.79USD",
      "amount": "1",
      "category": "GROCERY"
    }
  ],
  "totalItems": 7,
  "subtotal": "27.36USD",
  "tax": "0.00USD",
  "total": "27.36USD",
  "paymentMethod": "Visa",
  "cashier": "Jael"
}
```

Ground truth examples for all 7 test receipts are in `ReceiptJson/`.

---

## Logging

Structured JSON logs are written to `logs/etl_YYYY-MM-DD.jsonl` — one line per event.
All entries share these common fields: `time`, `level`, `service`, `event`, `trace_id`, `user_id`, `image_name`.

**ADI OCR event (`adi_ocr`)**
```json
{
  "time": "2026-03-25T12:34:47Z",
  "level": "INFO",
  "service": "etl",
  "event": "adi_ocr",
  "trace_id": "1b9ba35a-...",
  "user_id": "lkim016",
  "image_name": "2025-10-01Vons.jpg",
  "image_size_bytes": 1480423,
  "ocr_provider": "azure-document-intelligence",
  "ocr_latency_ms": 4936.0,
  "ocr_input_tokens": null,
  "ocr_output_tokens": null,
  "ocr_success": true,
  "pages": 1,
  "cost_usd": 0.0,
  "error": null
}
```

**LLM structuring event (`llm_extraction`)**
```json
{
  "time": "2026-03-25T12:34:57Z",
  "level": "INFO",
  "service": "etl",
  "event": "llm_extraction",
  "trace_id": "1b9ba35a-...",
  "user_id": "lkim016",
  "image_name": "2025-10-01Vons.jpg",
  "llm_provider": "openrouter",
  "llm_model": "anthropic/claude-3-haiku",
  "llm_latency_ms": 6985.0,
  "llm_input_tokens": 997,
  "llm_output_tokens": 674,
  "llm_cost_usd": 0.0,
  "llm_success": true,
  "items_extracted": 7,
  "error": null
}
```

**Upload event (`mcp_upload`)**
```json
{
  "time": "2026-03-25T12:35:10Z",
  "level": "INFO",
  "service": "etl",
  "event": "mcp_upload",
  "trace_id": "1b9ba35a-...",
  "user_id": "lkim016",
  "image_name": "2025-10-01Vons.jpg",
  "endpoint": "POST /api/v1/receipts",
  "status": 201,
  "latency_ms": 45.0,
  "items_attempted": 7,
  "items_uploaded": 7,
  "items_failed": 0,
  "error": null
}
```

Generate a report from the logs:

```bash
python etl.py --report
# → reports/report_YYYYMMDD_HHMMSS.md
# → reports/usage_chart_YYYYMMDD_HHMMSS.png
```

---

## Pricing

| Service | Free tier | Paid |
|---------|-----------|------|
| Azure Document Intelligence | F0: 500 pages/month, $0 | S0: $1.50 / 1,000 pages |
| OpenRouter `anthropic/claude-3-haiku` | Free (no credits needed) | — |
| OpenRouter `google/gemini-2.0-flash` | — | $0.10 / $0.40 per 1M tokens |
| Anthropic `claude-haiku-4-5-20251001` | — | $0.80 / $4.00 per 1M tokens |
| Anthropic `claude-sonnet-4-6` | — | $3.00 / $15.00 per 1M tokens |

**7 receipts · 89 items · avg ~1,100 input + ~950 output tokens per receipt → $0.00 on free tier**

---

## Project structure

```
etl.py                  # Main pipeline script
.env.example            # Environment variable template
requirements.txt        # Python dependencies
Receipts/converted/     # Input receipt images (7 test receipts)
ReceiptJson/            # Ground truth JSON for each test receipt
output/                 # Generated output JSON (one file per receipt)
logs/                   # JSONL structured logs (one file per day)
reports/                # Markdown reports + usage charts
docs/
  setup-azure-di.md         # Azure DI setup guide
  setup-openrouter.md       # OpenRouter setup guide
  llm-provider-setup.md     # LLM provider config + change log
```

---

## Remote storage

Receipt images are tracked via Git LFS (configured in `.gitattributes`).
Output JSON and logs are committed as regular files.

```bash
git lfs install        # once per machine
git add Receipts/ output/ logs/
git commit -m "receipts YYYY-MM-DD"
git push
```
