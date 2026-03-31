# OpenRouter Setup Guide

OpenRouter provides access to many LLMs through a single API. This project uses it for
Step 2 of the ETL pipeline — structuring raw OCR text into JSON.
The default model is on the **free tier**: no credit card required.

---

## Step 1 — Create an account

1. Go to **https://openrouter.ai**
2. Click **Sign Up**
3. Register with email, GitHub, or Google

No credit card is required for the free tier.

---

## Step 2 — Generate an API key

1. Log in to your OpenRouter dashboard
2. Click **Keys** in the top navigation
3. Click **Create Key**
4. Copy the key immediately — it is only shown once

Your key will look like: `sk-or-v1-abc123...`

---

## Step 3 — Add the key to your `.env`

From the project root:

```bash
cp .env.example .env
```

Open `.env` and fill in:

```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
OR_DEFAULT_MODEL=google/gemini-2.0-flash-exp:free
```

---

## Step 4 — Verify the key works

Run this before the full pipeline to confirm authentication is working:

```bash
python3 -c "
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    api_key=os.getenv('OPENROUTER_API_KEY'),
    base_url='https://openrouter.ai/api/v1'
)
r = client.chat.completions.create(
    model='google/gemini-2.0-flash-exp:free',
    messages=[{'role': 'user', 'content': 'Reply with the word OK only.'}]
)
print(r.choices[0].message.content)
"
```

**Expected output:** `OK`

If you see an authentication error, check that the key in `.env` matches what was shown in the dashboard.

---

## Free tier limits

| | |
|---|---|
| Credit card required | No |
| Cost | $0 |
| Rate limit | ~20 requests/min, ~200 requests/day |
| Eligible models | Any model with a `:free` suffix |

For 7 receipts in the incubation stage this is well within limits.

---

## Available free models

| Model | Notes |
|-------|-------|
| `google/gemini-2.0-flash-exp:free` | **Default** — best accuracy for receipt structuring |
| `meta-llama/llama-3.2-11b-vision-instruct:free` | Backup option |

To see all currently available free models, go to **openrouter.ai/models** and filter by **Free**.

---

## Upgrading beyond the free tier

If you hit rate limits when scaling to more receipts, upgrade to the paid model:

In `.env`, change:
```
OR_DEFAULT_MODEL=google/gemini-2.0-flash
```

This costs approximately **$0.10 per 1M input tokens / $0.40 per 1M output tokens**.
At roughly 1,000 tokens per receipt, processing 1,000 receipts costs under $0.02.
Add credits at **openrouter.ai/credits**.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `401 Unauthorized` | Key is wrong or not set in `.env` |
| `429 Too Many Requests` | Hit rate limit — wait 1 minute or upgrade model |
| `model not found` | Free model may be temporarily unavailable — try the backup |
| `OPENROUTER_API_KEY not set` | Run `cp .env.example .env` and fill in the key |
