# Azure Document Intelligence Setup Guide (Free F0 Tier)

Azure Document Intelligence (ADI) handles Step 1 of the ETL pipeline — reading
all text off the receipt image (OCR). The free F0 tier gives you 500 pages/month
at no cost. 7 receipts = 7 pages.

---

## Step 1 — Create an Azure account

1. Go to **https://portal.azure.com**
2. Click **Create account** and sign up with a Microsoft account, GitHub, or email
3. A credit card is required for identity verification — you will **not** be charged
   for F0 tier usage

---

## Step 2 — Create a Document Intelligence resource

1. Sign in to the Azure portal
2. Click **Create a resource** on the home page
3. In the search bar type **Document Intelligence** and select it
4. Click **Create**
5. Fill in the form:

   | Field | What to enter |
   |-------|--------------|
   | Subscription | Your current subscription |
   | Resource group | Create new → give it a name e.g. `gatheryourdeals-rg` |
   | Region | Pick the one closest to you |
   | Name | Something descriptive e.g. `gatheryourdeals-di` |
   | **Pricing tier** | **Free F0** ← important |

6. Click **Review + create**
7. Wait for **Validation Passed** (a few seconds), then click **Create**
8. Wait for **Your deployment is complete**

---

## Step 3 — Get your endpoint and key

1. Click **Go to resource**
2. In the left sidebar click **Keys and Endpoint**
3. Copy:
   - **Endpoint** — looks like `https://gatheryourdeals-di.cognitiveservices.azure.com/`
   - **Key 1** — a long hex string

---

## Step 4 — Add to your `.env`

Open `.env` in the project root and fill in:

```
AZURE_DI_ENDPOINT=https://gatheryourdeals-di.cognitiveservices.azure.com/
AZURE_DI_KEY=your-key-1-here
```

---

## Step 5 — Verify ADI works

Test OCR on the first receipt without running the full pipeline:

```bash
python3 - <<'EOF'
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import sys
sys.path.insert(0, ".")
from etl import ocr

text = ocr(Path("../Receipts/converted/2025-10-01Vons.jpg"), run_id="test")
print(text)
EOF
```

**Expected output:** raw receipt text extracted by ADI, e.g.:

```
Vons
8010 East Santa Ana Cyn
Anaheim Hills CA 92808
...
OPN Nat Granola        4.79
Granola Blubry Flx     4.79
...
```

If you see receipt text, ADI is working correctly.

---

## Free tier limits

| | |
|---|---|
| Pages per month | 500 |
| Cost | $0 |
| Overage behaviour | Stops (returns error) — does **not** auto-charge |
| Credit card charge | None for F0 usage |

For 7 receipts you could run the pipeline **71 times** before approaching the monthly limit.

---

## Monitoring usage in the Azure portal

After running the pipeline, you can see ADI metrics at:

**Azure portal → your Document Intelligence resource → Metrics**

Useful metrics to add:
- **Processed Pages** — how many pages have been sent
- **Successful Calls** — request success rate
- **Latency** — response time per request

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `AZURE_DI_ENDPOINT not set` | Check `.env` has the endpoint with no trailing spaces |
| `401 Unauthorized` | Key is wrong — re-copy Key 1 from the portal |
| `ResourceNotFound` | Endpoint URL is wrong — must end with `.cognitiveservices.azure.com/` |
| `QuotaExceeded` | Hit 500 pages for the month — wait for reset or upgrade to S0 |
| `ImportError: azure-ai-documentintelligence` | Run `pip install azure-ai-documentintelligence` |
