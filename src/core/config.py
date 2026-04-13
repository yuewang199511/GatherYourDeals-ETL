# core/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# This ensures .env is loaded as soon as config is imported
load_dotenv()

# Define the project root relative to this file (up one level from 'core/')
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# --- API Keys ---
AZURE_DI_ENDPOINT   = os.getenv("AZURE_DI_ENDPOINT", "")
AZURE_DI_KEY        = os.getenv("AZURE_DI_KEY", "")
AZURE_MAPS_KEY      = os.getenv("AZURE_MAPS_KEY", "")

OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
CLOD_API_KEY        = os.getenv("CLOD_API_KEY", "")

# --- LLM Defaults ---
# Which LLM backend to use: "clod" (default) or "openrouter"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "clod").lower()

# OR_DEFAULT_MODEL   = os.getenv("OR_DEFAULT_MODEL",   "anthropic/claude-haiku-4.5")  # OpenRouter disabled as default
OR_DEFAULT_MODEL   = os.getenv("OR_DEFAULT_MODEL",   "anthropic/claude-haiku-4.5")  # still usable via --provider openrouter
CLOD_DEFAULT_MODEL = os.getenv("CLOD_DEFAULT_MODEL", "google/gemma-3n-E4B-it")

# API endpoints — read from env to allow routing to proxies or alternate regions
# --- URLs ---
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
CLOD_API_URL        = os.getenv("CLOD_API_URL",        "https://api.clod.io/v1/chat/completions")
AZURE_MAPS_URL      = os.getenv("AZURE_MAPS_URL",      "https://atlas.microsoft.com/search/fuzzy/json")

# Operational constants
_ADI_TIMEOUT_S  = int(os.getenv("AZURE_DI_TIMEOUT",  "120"))
_CLOD_TIMEOUT_S = int(os.getenv("CLOD_TIMEOUT",       "120"))
_CLOD_RETRIES   = int(os.getenv("CLOD_RETRIES",       "3"))

# --- Server / Upload ---
GYD_SERVER_URL   = os.getenv("GYD_SERVER_URL", "http://localhost:8080/api/v1")
GYD_ACCESS_TOKEN = os.getenv("GYD_ACCESS_TOKEN", "")

# --- Paths (Using ROOT_DIR ensures these work anywhere) ---
# --- Paths (Anchored to ROOT_DIR ensures these work anywhere) ---
# ROOT_DIR is already Path(__file__).resolve().parent.parent (which is GatherYourDeals-ETL/)
OUTPUT_DIR       = ROOT_DIR / "output"
LOGS_DIR         = ROOT_DIR / "logs"
REPORTS_DIR      = ROOT_DIR / "reports"
GROUND_TRUTH_DIR = ROOT_DIR / "ground_truth"
OCR_CACHE_DIR    = ROOT_DIR / "ocr_cache"

# 3. Add this loop to actually CREATE the folders so the script doesn't crash
for _path in [OUTPUT_DIR, LOGS_DIR, REPORTS_DIR, OCR_CACHE_DIR]:
    _path.mkdir(parents=True, exist_ok=True)

# Registry that maps image stem → list of GYD receipt UUIDs created on upload.
# Used by delete_uploaded() to find and remove records from the database.
_UPLOAD_REGISTRY = OUTPUT_DIR / ".upload_registry.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".tiff", ".tif", ".bmp"}

# --- Model Pricing ---
# ---------------------------------------------------------------------------
# LLM PRICING (Rates per 1,000,000 tokens)
# Format: "model_name": (input_cost_usd, output_cost_usd)
# ---------------------------------------------------------------------------

LLM_PRICING = {
    "openrouter": {
        "anthropic/claude-haiku-4.5":    (1.00,  5.00),
        "qwen/qwen-2.5-7b-instruct":     (0.04,  0.10),
        "google/gemini-flash-1.5":       (0.075, 0.30),
    },
    "clod": {
        # Qwen variants (60% discount applied to output where applicable)
        "Qwen2.5-7B-Instruct-Turbo":     (0.30,  0.12),
        "Qwen/Qwen2.5-7B-Instruct-Turbo": (0.30,  0.12),
        "qwen/qwen-2.5-7b-instruct":      (0.04,  0.10),
        
        # Gemma 3n (Rates updated 2026-04-01 from Together.ai)
        "gemma-3n-E4B-it":               (0.02,  0.04),
        "google/gemma-3n-E4B-it":        (0.02,  0.04),
        
        # Anthropic (Placeholders confirmed 2026-04-13)
        "anthropic/claude-haiku-4-5":    (1.00,  5.00),
    }
}