"""
Structured JSONL logger for the GatherYourDeals ETL pipeline.

Writes one JSON object per line to logs/etl_YYYY-MM-DD.jsonl.
All events share a common envelope: time, level, service, event,
trace_id, user_id, image_name.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from src.core import config

# ADI cost per page — always log at S0 rate ($0.0015/page) for accurate
# production cost tracking, regardless of free tier usage.
ADI_COST_PER_PAGE = 0.0015


def _log(entry: dict):
    config.LOGS_DIR.mkdir(exist_ok=True)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    with open(config.LOGS_DIR / f"etl_{date}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def log_adi(trace_id, image_name, user_id, image_size_bytes,
            pages, cost_usd, latency_ms, success, chars_extracted=None, error=None):
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
        "ocr_success":       success,
        "pages":             pages,
        "chars_extracted":   chars_extracted,
        "cost_usd":          round(cost_usd, 6),
        "error":             error,
    })


def log_llm(trace_id, image_name, user_id, provider, model,
            input_tokens, output_tokens, cost_usd,
            latency_ms, items_extracted, success, error=None,
            cost_source="unknown", latency_source="local",
            input_chars=None, prompt_path=None):
    """Log one LLM structuring call (OpenRouter or CLOD).

    input_chars  — chars of OCR content sent to the LLM (after all filtering /
                   spatial stripping), excluding the system prompt.  Lets you
                   track the impact of token reduction changes independently of
                   the API-reported prompt_tokens (which include system prompt).
    prompt_path  — which system prompt variant fired:
                   "direct", "cot", "direct+costco", "cot+costco"
    """
    _log({
        "time":                datetime.now(timezone.utc).isoformat(),
        "level":               "INFO" if success else "ERROR",
        "service":             "etl",
        "event":               "llm_extraction",
        "trace_id":            trace_id,
        "user_id":             user_id,
        "image_name":          image_name,
        "llm_provider":        provider,
        "llm_model":           model,
        "llm_latency_ms":      round(latency_ms, 1),
        "llm_latency_source":  latency_source,
        "llm_input_tokens":    input_tokens,
        "llm_output_tokens":   output_tokens,
        "llm_input_chars":     input_chars,
        "llm_prompt_path":     prompt_path,
        "llm_cost_usd":        round(cost_usd, 8),
        "llm_cost_source":     cost_source,
        "llm_success":         success,
        "items_extracted":     items_extracted,
        "error":               error,
        # Standardized usage block read by Railtracks / AgentHub observability.
        "usage": {
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "total_tokens":  (input_tokens or 0) + (output_tokens or 0),
        },
        "cost": {
            "total_usd": round(cost_usd, 8),
        },
    })


def log_pipeline(trace_id, image_name, user_id, provider, model,
                 total_latency_ms, success, error=None):
    """Log end-to-end pipeline latency for one receipt (OCR + LLM + geocode)."""
    _log({
        "time":              datetime.now(timezone.utc).isoformat(),
        "level":             "INFO" if success else "ERROR",
        "service":           "etl",
        "event":             "pipeline_complete",
        "trace_id":          trace_id,
        "user_id":           user_id,
        "image_name":        image_name,
        "llm_provider":      provider,
        "llm_model":         model,
        "total_latency_ms":  round(total_latency_ms, 1),
        "success":           success,
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


def log_delete(trace_id, image_name, user_id,
               attempted, deleted, failed, latency_ms, success, error=None):
    """Log one GYD SDK delete batch."""
    _log({
        "time":            datetime.now(timezone.utc).isoformat(),
        "level":           "INFO" if success else "WARN",
        "service":         "etl",
        "event":           "mcp_delete",
        "trace_id":        trace_id,
        "user_id":         user_id,
        "image_name":      image_name,
        "endpoint":        "DELETE /api/v1/receipts/{id}",
        "status":          200 if success else None,
        "latency_ms":      round(latency_ms, 1),
        "items_attempted": attempted,
        "items_deleted":   deleted,
        "items_failed":    failed,
        "error":           error,
    })
