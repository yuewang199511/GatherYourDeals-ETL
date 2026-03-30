# GatherYourDeals-ETL

Two-step receipt pipeline that reads a photo of a receipt and produces structured JSON, orchestrated as a [Railtracks](https://railtracks.ai) Flow.

```
Receipt photo  (JPG / PNG / WEBP / HEIC / TIFF / BMP)
    │
    ▼  Step 1 — Azure Document Intelligence  (prebuilt-read)
    │           Extracts all text from the image via OCR
    │
    ▼  Step 2 — LLM  (OpenRouter  /  CLOD)
    │           Structures the OCR text into the GYD JSON format
    │
    ▼  Step 2b — Store Name Normalization  (second LLM call)
    │           Matches raw store name to canonical entry in ground_truth/ corpus
    │
    ▼  Step 3 — Azure Maps Geocoding  (optional)
    │           Resolves store address → latitude / longitude
    │
    ▼  output/<provider>/<image-name>.json  +  optional upload to GYD data service
```

**Baseline (9 receipts, 2026-03-30):** 9/9 OpenRouter · 9/9 CLOD · $0.1921 total (last run) · OpenRouter $0.0038/receipt · CLOD $0.0003/receipt

---

## Setup

### 1. Install dependencies

```bash
pip install openai python-dotenv azure-ai-documentintelligence
pip install "railtracks[cli]"
pip install git+https://github.com/yuewang199511/GatherYourDeals-SDK.git
pip install matplotlib              # optional — for --report charts
pip install pillow pillow-heif      # optional — only for HEIC (iPhone) photos
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

### 4. Azure Maps Geocoding (Step 3 — optional)

Resolves the store address into `latitude` / `longitude`.
If `AZURE_MAPS_KEY` is not set, lat/lon will be `null` in the output.

1. In the Azure Portal, add **Azure Maps** to your existing resource group
2. Choose **Gen2** pricing tier (free: 5,000 geocode requests/month)
3. After deployment: **Authentication** → copy **Primary Key**

### 5. Configure `.env`

```bash
cp .env.example .env
```

```env
# Step 1 — Azure Document Intelligence
AZURE_DI_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/
AZURE_DI_KEY=<your-key>

# Step 2 — LLM provider: openrouter (default) or clod
LLM_PROVIDER=openrouter
ETL_MODEL=anthropic/claude-haiku-4.5

# OpenRouter
OPENROUTER_API_KEY=sk-or-v1-...

# CLOD (set LLM_PROVIDER=clod to use)
# CLOD_API_KEY=<your-key>

# Step 3 — Azure Maps geocoding (optional — leave blank to skip)
AZURE_MAPS_KEY=<your-key>

# GYD data service — leave blank to run in extract-only mode
GYD_SERVER_URL=http://localhost:8080/api/v1
GYD_USERNAME=
GYD_PASSWORD=
```

---

## Usage

```bash
# Single receipt
python etl.py Receipts/2026-01-03Costco.jpg --user xxx --no-upload

# Whole directory
python etl.py Receipts/ --user xxx --no-upload

# Use CLOD instead of OpenRouter
python etl.py Receipts/ --user xxx --provider clod --no-upload

# With upload to GYD data service (set GYD_* in .env first)
python etl.py Receipts/ --user xxx

# Full baseline experiment (both providers, 3× each)
bash run_baseline.sh

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
| `--model MODEL` | `ETL_MODEL` env var | Model ID — overrides provider default |
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

## Logging

Structured JSON logs are written to `logs/etl_YYYY-MM-DD.jsonl` — one line per event.
All entries share: `time`, `level`, `service`, `event`, `trace_id`, `user_id`, `image_name`.

**`adi_ocr`**
```json
{
  "event": "adi_ocr",
  "ocr_latency_ms": 4936.0,
  "ocr_success": true,
  "pages": 1,
  "chars_extracted": 1842,
  "cost_usd": 0.0015
}
```

> `cost_usd` is always logged at the S0 rate ($0.0015/page) for accurate production cost tracking, regardless of free tier usage.

**`llm_extraction`**
```json
{
  "event": "llm_extraction",
  "llm_provider": "openrouter",
  "llm_model": "anthropic/claude-haiku-4.5",
  "llm_latency_ms": 6985.0,
  "llm_input_tokens": 997,
  "llm_output_tokens": 674,
  "llm_cost_usd": 0.003276,
  "llm_cost_source": "estimate",
  "llm_success": true,
  "items_extracted": 7
}
```

> `llm_cost_source` is `"api"` when the provider returns a billed cost, `"estimate"` when falling back to token-based pricing.

**`pipeline_complete`**
```json
{
  "event": "pipeline_complete",
  "llm_provider": "openrouter",
  "llm_model": "anthropic/claude-haiku-4.5",
  "total_latency_ms": 12847.0,
  "success": true
}
```

> End-to-end wall time per receipt — covers OCR + LLM + geocoding. Used to compute E2E P50/P95 in `--baseline-report`.

**`mcp_upload`**
```json
{
  "event": "mcp_upload",
  "endpoint": "POST /api/v1/receipts",
  "status": 201,
  "latency_ms": 45.0,
  "items_attempted": 7,
  "items_uploaded": 7,
  "items_failed": 0
}
```

`reporting.py` can also be run directly:

```bash
python reporting.py --baseline-report
python reporting.py --report
python reporting.py --compare
python reporting.py --eval
```

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
python etl.py Receipts/ --user xxx --no-upload

# 2. Open the viz dashboard
railtracks viz
# → http://localhost:3030
```

### What Railtracks shows vs JSONL logs

| Metric | Railtracks | JSONL logs |
|---|---|---|
| E2E duration per receipt | ✓ | ✓ (`total_latency_ms`) |
| Pass / fail per run | ✓ | ✓ (`llm_success`) |
| Cost per receipt | — | ✓ (`llm_cost_usd`) |
| Token counts | — | ✓ (`llm_input_tokens`, `llm_output_tokens`) |
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

## Project structure

```
etl.py                  # Main pipeline script (ADI OCR → LLM → geocode → upload)
reporting.py            # Reporting module (--baseline-report, --report, --compare, --eval)
run_baseline.sh         # Baseline experiment script (3× per provider)
.env.example            # Environment variable template
requirements.txt        # Python dependencies
Receipts/               # Input receipt images (9 receipts)
ground_truth/           # Manually-verified reference JSON for each receipt
output/
  openrouter/           # Output JSON from OpenRouter runs
  clod/                 # Output JSON from CLOD runs
logs/                   # JSONL structured logs (etl_YYYY-MM-DD.jsonl) + Railtracks rt.log
reports/                # Markdown reports + usage charts
docs/
  HW9Report.md              # CS6650 HW9 report — problem, methodology, results
  setup-azure-di.md         # Azure DI setup guide
  llm-provider-setup.md     # LLM provider config + model change log
```

---

## Remote storage

Receipt images are tracked via Git LFS (configured in `.gitattributes`).

```bash
git lfs install        # once per machine
git add Receipts/ output/ logs/
git commit -m "receipts YYYY-MM-DD"
git push
```
