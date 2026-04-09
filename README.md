# GatherYourDeals-ETL

Receipt digitization pipeline exposed as a REST service. Accepts a receipt image (or a Google Drive folder of images) and returns structured line-item JSON via OCR + LLM, orchestrated as a [Railtracks](https://railtracks.org/) Flow.

```
POST /etl  { "source": "<image URL, local path, or Google Drive folder URL>" }
    │
    ▼  Step 1 — Azure Document Intelligence  (prebuilt-read)
    │           Extracts raw text + per-word bounding-box coordinates
    │
    ▼  Step 2 — Spatial Reconstruction
    │           Tokens classified into columns [L] / [C] / [R] by X-position
    │           [L] tokens grouped into rows by Y-proximity (tolerance 12 px)
    │           [R] price tokens assigned to nearest [L] row at-or-below their Y
    │           Garbage header lines filtered; purchase date injected into chunk
    │
    ▼  Step 3 — LLM Structuring  (OpenRouter / CLOD)
    │           Chunked spatial layout → JSON extraction
    │           Fields: productName, price, amount, purchaseDate, storeName
    │           Chunks merged; purchaseDate forward-scanned across all chunks
    │
    ▼  Step 4 — Deterministic Post-Processing
    │           Tier 1  Currency detection — scans OCR for CAD/GBP/EUR markers;
    │                   overrides LLM default (which often assumes USD)
    │           Tier 2  Price/amount validation — strips tax-code letters
    │                   (e.g. "4.79 S" → "4.79"), removes unit codes (EA/MRJ/PK)
    │           Tier 3  Weight-item recovery — regex parses
    │                   "X.XXX kg @ $Y.YY/kg TOTAL" lines from raw OCR;
    │                   injects correct price + weight amount; adds items the
    │                   LLM missed entirely
    │           Tier 4  Store name normalization — maps ALL-CAPS brand codes
    │                   (e.g. NOFRILLS) to canonical form (No Frills)
    │           Tier 5  Null-price repair — targeted re-extraction for any
    │                   items still missing a price after tiers 1–4
    │
    ▼  Step 5 — Azure Maps Geocoding  (optional)
    │           Resolves store address → latitude / longitude
    │           Address extracted from OCR if LLM returns none
    │
    ▼  Step 6 — GYD Upload
                Uploads structured items to the GYD data service via SDK
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install "railtracks[cli]"      # optional — flow observability
pip install pillow pillow-heif     # optional — only for HEIC (iPhone) photos
pip install matplotlib             # optional — for --report charts
```

> Azure Maps geocoding uses Python's built-in `urllib` — no extra package needed.

### 2. Azure Document Intelligence (Step 1 — OCR)

1. Go to [portal.azure.com](https://portal.azure.com)
2. **Create a resource** → search **Document Intelligence** → Create
3. Choose **Free tier (F0)**: 500 pages/month, no charge
4. After deployment: **Keys and Endpoint** → copy **Endpoint** and **Key 1**

See `docs/setup-azure-di.md` for detailed instructions.

### 3. LLM provider (Step 2 — structuring)

**Option A — OpenRouter** (default)

1. Sign up at [openrouter.ai](https://openrouter.ai)
2. **Keys** → Create key
3. Default model: `anthropic/claude-haiku-4.5`

**Option B — CLOD**

1. Sign up at [app.clod.io](https://app.clod.io)
2. **API Keys** → Create key
3. Set `LLM_PROVIDER=clod` in `.env`
4. Default model: `Qwen/Qwen2.5-7B-Instruct-Turbo`

See `docs/llm-provider-setup.md` for model options, pricing, and troubleshooting.

### 4. Google Drive folder ingestion (optional)

`POST /etl` accepts a Google Drive folder URL and processes all images inside
it via **gdown** — no API key or OAuth required.

The folder must be shared as **"Anyone with the link can view"**. No setup
needed — `gdown` is included in `requirements.txt`.

> For private folder support, see `docs/general.md` → *Google Drive Folder Ingestion*.

### 5. Azure Maps Geocoding (Step 3 — optional)

Resolves the store address into `latitude` / `longitude`.
If `AZURE_MAPS_KEY` is not set, lat/lon will be `null` in the output.

1. In the Azure Portal, add **Azure Maps** to your existing resource group
2. Choose **Gen2** pricing tier (free: 5,000 geocode requests/month)
3. After deployment: **Authentication** → copy **Primary Key**

### 6. GYD data service token

The ETL uploads structured receipts to the GYD data service using a JWT access token — no username/password needed.

```bash
# One-time login via the GYD CLI
gatherYourDeals login

# Print your current access token and copy it into .env
gatherYourDeals show-token
```

```env
GYD_SERVER_URL=http://localhost:8080/api/v1
GYD_ACCESS_TOKEN=<paste token here>
```

> The token is initialized **per request** (no shared client). If `GYD_ACCESS_TOKEN` is not set, the SDK falls back to tokens auto-loaded from `~/.GYD_SDK/env.yaml` stored by the CLI login.

### 7. Configure `.env`

```bash
cp .env.example .env
```

```env
# Step 1 — Azure Document Intelligence
AZURE_DI_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/
AZURE_DI_KEY=<your-key>

# Step 2 — LLM provider: openrouter (default) or clod
LLM_PROVIDER=openrouter

# Option A — OpenRouter
OPENROUTER_API_KEY=sk-or-v1-...
OR_DEFAULT_MODEL=anthropic/claude-haiku-4.5

# Option B — CLOD (set LLM_PROVIDER=clod to use)
# CLOD_API_KEY=<your-key>
CLOD_DEFAULT_MODEL=Qwen/Qwen2.5-7B-Instruct-Turbo

# Step 3 — Azure Maps geocoding (optional — leave blank to skip)
AZURE_MAPS_KEY=<your-key>

# GYD data service — leave blank to run in extract-only mode
GYD_SERVER_URL=http://localhost:8080/api/v1
# JWT access token: run `gatherYourDeals login` then `gatherYourDeals show-token`
GYD_ACCESS_TOKEN=

# ETL service username written into receipt JSON metadata
ETL_DEFAULT_USER=lkim

# Google Drive private folder access (optional — public folders use gdown automatically)
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GOOGLE_REFRESH_TOKEN=
```

---

## Running as a Service

Start the ETL service with uvicorn:

```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/etl` | Run the full ETL pipeline — single image or Google Drive folder |
| `GET` | `/health` | Liveness check |

Interactive API docs available at `http://localhost:8000/docs` once running.

### Example requests

**Single image:**
```bash
curl -X POST http://localhost:8000/etl \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-jwt>" \
  -d '{"source": "https://example.com/receipts/2026-01-03Costco.jpg"}'
```

**Google Drive folder (batch):**
```bash
curl -X POST http://localhost:8000/etl \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-jwt>" \
  -d '{"source": "https://drive.google.com/drive/folders/<folder-id>"}'
```

The `source` field accepts: HTTP/HTTPS image URLs, Google Drive file/folder URLs, or local file paths.
Google Drive viewer URLs (`/file/d/<id>/view`) are automatically converted to direct download URLs.

### Response

**Single image:**
```json
{ "success": true, "message": "ETL completed successfully" }
```

**Google Drive folder (batch):**
```json
{
  "success": true,
  "message": "4/4 succeeded",
  "results": [
    {"file": "receipt1.jpg", "success": true,  "message": "ETL completed successfully"},
    {"file": "receipt2.jpg", "success": false, "message": "Failed to parse data from source: ..."}
  ]
}
```

| Status | Meaning |
|--------|---------|
| `200` | Pipeline + upload completed successfully (single or batch) |
| `400` | Empty source, unreachable URL, unsupported file type, or missing Google OAuth credentials |
| `422` | Source reachable but OCR / LLM processing failed |
| `500` | Upload to GYD service failed |

---

---

## Usage (CLI)

```bash
# Single receipt
python etl.py Receipts/2026-01-03Costco.jpg --user $GYD_USERNAME --no-upload

# Whole directory
python etl.py Receipts/ --user $GYD_USERNAME --no-upload

# Use CLOD instead of OpenRouter
python etl.py Receipts/ --user $GYD_USERNAME --provider clod --no-upload

# With upload to GYD data service (set GYD_* in .env first)
python etl.py Receipts/ --user $GYD_USERNAME

# Full baseline experiment (both providers, 3× each)
bash scripts/run_baseline.sh

# Generate baseline experiment report (last run only)
python etl.py --baseline-report

# Generate cumulative usage report + charts from logs
python etl.py --report

# Generate per-model comparison table
python etl.py --compare

# Evaluate output/ against ground_truth/
python etl.py --eval

# View Railtracks run visualizer
railtracks viz
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--user USER` | `unknown` | Username written into the output JSON |
| `--provider {openrouter,clod}` | `LLM_PROVIDER` env var | LLM backend to use |
| `--model MODEL` | `OR_DEFAULT_MODEL` env var | Model ID — overrides provider default |
| `--no-upload` | off | Skip upload to GYD data service |
| `--baseline-report` | off | Structured provider egress report scoped to last run |
| `--report` | off | Cumulative usage report + chart from all logs |
| `--compare` | off | Per-model comparison table from all logs |
| `--eval` | off | Score output/ against ground_truth/ |

---

## Output format

Each receipt produces `output/<provider>/<image-name>.json` as a **list of flat per-item dicts**.
Each element has exactly 7 fields:

```json
[
  {
    "productName": "OPN NAT GRANOLA",
    "purchaseDate": "2026.01.03",
    "price": "4.79USD",
    "amount": "1",
    "storeName": "COSTCO",
    "latitude": 33.8673096,
    "longitude": -117.7515869
  }
]
```

Output is separated by provider (`output/openrouter/`, `output/clod/`) so each can be evaluated independently against ground truth.

---

## `ground_truth/`

Contains reference JSON files — one per receipt — used to evaluate pipeline accuracy. Files are named after the source image (e.g. `2026-01-03Costco.json`) and follow the full receipt schema.

Ground truth files in this repo were **written manually** — each receipt image was read by hand and transcribed into JSON, keeping the reference fully independent from the pipeline.

```bash
python etl.py --eval
```

Scores every file in `output/<provider>/` against `ground_truth/` field by field:

- **Store name, date, lat/lon** — receipt-level scalar fields
- **Item count** — correct number of line items?
- **Item name / price / amount** — per-item matching

Overall 0–100 score (50% scalar fields, 50% item accuracy) saved to `reports/`.

---

## LLM Extraction

### Chunking strategy

Long OCR outputs are split into overlapping chunks (~2,500 chars each, 10-line overlap) so the LLM context window is never exceeded. Each chunk is prefixed with a 3-line header (store name, address, purchase date) so every chunk has enough context to populate all fields.

When Azure Document Intelligence returns a spatial layout (bounding-box data), the pipeline switches to a **spatial-only path**: the raw OCR is discarded in favor of a column-tagged layout built from the word coordinates. This eliminates the raw OCR line-ordering confusion that causes price misalignment on receipts like Costco where prices are printed before item names.

### Spatial reconstruction

Tokens are classified into three columns by their X-centre relative to the page width:

| Column | Tag | Content |
|--------|-----|---------|
| Left (< 35%) | `[L]` | Product name |
| Centre (35–65%) | `[C]` | Quantity / unit code |
| Right (> 65%) | `[R]` | Price |

`[L]` tokens are grouped into rows by Y-proximity (12 px tolerance). `[R]` price tokens are then assigned to the nearest `[L]` row **at or below** the price's Y position — a two-pass algorithm that handles the common Costco pattern where the price bounding box falls between two item rows.

### Post-processing pipeline

After the LLM returns JSON, several deterministic passes correct common model errors:

| Tier | Function | What it fixes |
|------|----------|---------------|
| 1 | `_detect_currency_from_ocr` | Models default to USD; scans raw OCR for `CAD`, `C$`, `£`, `€` and overrides |
| 2 | `_validate_and_fix_items` | Strips tax-code letters from amounts (`4.79 S` → `4.79`); removes unit codes (EA, MRJ, PK); rejects non-product lines (subtotal, tax, change) |
| 3 | `_inject_weight_prices` | Parses `X.XXX kg @ $Y.YY/kg TOTAL` patterns from raw OCR; injects correct price and weight-amount for all matching items; adds any weight-priced items the LLM missed entirely |
| 4 | `_normalize_store_name` | Maps ALL-CAPS brand codes (e.g. `NOFRILLS`) to canonical form (`No Frills`) via a static alias table |
| 5 | `_repair_failed_items` | Re-extracts any item still carrying a null price after tiers 1–4 |

### Date handling

Receipts use several date formats. The pipeline detects and normalises all of them to `YYYY.MM.DD`:

| Format | Example | Detection |
|--------|---------|-----------|
| `MM/DD/YYYY` | `01/03/2026` | `_DATE_MDY` regex |
| `MM/DD/YY` | `01/03/26` | `_DATE_SHORT` (C < 12 → year) |
| `YY/MM/DD` | `26/02/21` | `_DATE_SHORT` (A > 12 → year) |

The extracted date is injected into every chunk header so that even when the date line falls in a later chunk, all chunks can populate `purchaseDate`. After merging, `purchaseDate` is forward-scanned across all chunks to find the first non-null value.

### Multi-currency support

| Currency | OCR markers detected |
|----------|---------------------|
| USD (default) | fallback when no other marker found |
| CAD | `CAD`, `CAD$`, `C$`, `$CAD` |
| GBP | `GBP`, `£` |
| EUR | `EUR`, `€` |

---

## Observability — Railtracks

The pipeline runs as a Railtracks Flow (`receipt_etl`). Railtracks provides a local browser UI to inspect per-receipt run timelines and pass/fail history.

### Setup

```bash
# Inside a virtual environment (recommended)
pip install "railtracks[cli]"

# System Python
pip install "railtracks[cli]" --break-system-packages
```

> If Railtracks is not installed the pipeline falls back to a plain sequential run automatically.

### Usage

```bash
# 1. Run the ETL as normal
python etl.py Receipts/ --user $GYD_USERNAME --no-upload

# 2. Open the viz dashboard
railtracks viz
# → http://localhost:3030
```

### What Railtracks shows vs JSONL logs

| Metric | Railtracks | JSONL logs |
|---|---|---|
| E2E duration per receipt | ✓ | ✓ (`total_latency_ms`) |
| Pass / fail per run | ✓ | ✓ (`llm_success`) |
| Cost per receipt | ✓ (`cost.total_usd`) | ✓ (`llm_cost_usd`) |
| Token counts | ✓ (`usage.*_tokens`) | ✓ (`llm_input_tokens`, `llm_output_tokens`) |
| OCR vs LLM latency split | — | ✓ (`ocr_latency_ms`, `llm_latency_ms`) |

---

## Pricing

| Service | Rate | Notes |
|---------|------|-------|
| Azure Document Intelligence | $0.0015/page (S0) | F0: 500 pages/month free |
| Azure Maps Geocoding | ~$4.50/1,000 req | Gen2: 5,000 req/month free |
| OpenRouter `anthropic/claude-haiku-4.5` | $1.00/$5.00 per M tokens | ~$0.0038/receipt |
| CLOD `Qwen/Qwen2.5-7B-Instruct-Turbo` | $0.30/$0.12 per M tokens | ~$0.0003/receipt (sponsored) |

> ADI cost is always logged at the S0 rate for accurate production cost tracking. Free tier quotas are not subtracted.

---

## Remote storage

Receipt images and large output files are stored on SharePoint:

[Google Drive - Receipts](https://drive.google.com/drive/folders/1_IiL3p5N3djcDsc1ITYYniyDSqamB_fi)
