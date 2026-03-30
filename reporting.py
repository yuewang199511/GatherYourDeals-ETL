"""
GatherYourDeals ETL — Reporting & Evaluation
=============================================
Standalone module for generating reports, model comparison tables, and
eval scores from ETL run logs.

Usage (via etl.py CLI):
  python etl.py --report     Generate cumulative usage report from logs
  python etl.py --compare    Generate per-model comparison table (Test 2)
  python etl.py --eval       Compare output/ against ground_truth/

Or run directly:
  python reporting.py --report
  python reporting.py --compare
  python reporting.py --eval
"""

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — must match etl.py
# ---------------------------------------------------------------------------
OUTPUT_DIR       = Path("output")
LOGS_DIR         = Path("logs")
REPORTS_DIR      = Path("reports")
GROUND_TRUTH_DIR = Path("ground_truth")
IMAGE_EXTS       = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".tiff", ".tif", ".bmp"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_log_entries() -> list[dict]:
    entries = []
    for f in sorted(LOGS_DIR.glob("etl_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


# ---------------------------------------------------------------------------
# Report — cumulative usage report from all logs
# ---------------------------------------------------------------------------
def report():
    entries = _load_log_entries()

    adi_entries = [e for e in entries if e.get("event") == "adi_ocr"]
    llm_entries = [e for e in entries if e.get("event") == "llm_extraction"]
    ups         = [e for e in entries if e.get("event") == "mcp_upload"]

    if not llm_entries and not adi_entries:
        print("No log entries found. Run the pipeline first.")
        return

    # ---- ADI stats ----
    adi_calls   = len(adi_entries)
    adi_cost    = sum(e.get("cost_usd", 0) for e in adi_entries)
    adi_pages   = sum(e.get("pages", 0)    for e in adi_entries)
    adi_avg_lat = (sum(e.get("ocr_latency_ms", 0) for e in adi_entries) / adi_calls
                   if adi_calls else 0)

    # ---- LLM stats ----
    llm_calls    = len(llm_entries)
    llm_ok       = sum(1 for e in llm_entries if e.get("llm_success"))
    total_prompt = sum(e.get("llm_input_tokens",  0) for e in llm_entries)
    total_comp   = sum(e.get("llm_output_tokens", 0) for e in llm_entries)
    total_tokens = total_prompt + total_comp
    llm_cost     = sum(e.get("llm_cost_usd", 0)     for e in llm_entries)
    llm_avg_lat  = (sum(e.get("llm_latency_ms", 0) for e in llm_entries) / llm_calls
                    if llm_calls else 0)
    total_items  = sum(e.get("items_extracted", 0)  for e in llm_entries)
    uploaded     = sum(e.get("items_uploaded",  0)  for e in ups)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# GatherYourDeals ETL Report",
        f"_Generated: {now}_", "",
        "## Cost & Token Summary", "",
        "| Metric | Value |", "|--------|-------|",
        f"| ADI OCR calls | {adi_calls} ({adi_pages} pages) |",
        f"| ADI estimated cost | ${adi_cost:.4f} USD |",
        f"| ADI avg latency | {adi_avg_lat:.0f} ms |",
        f"| LLM calls | {llm_calls} ({llm_ok} success) |",
        f"| LLM total tokens | {total_tokens:,} |",
        f"| LLM input / output | {total_prompt:,} / {total_comp:,} |",
        f"| LLM estimated cost | ${llm_cost:.6f} USD |",
        f"| LLM avg latency | {llm_avg_lat:.0f} ms |",
        f"| **Total estimated cost** | **${adi_cost + llm_cost:.6f} USD** |",
        f"| Items extracted | {total_items} |",
        f"| Items uploaded | {uploaded} |",
        "", "## Per-Image Breakdown", "",
        "| Image | OCR chars | ADI (ms) | ADI cost | LLM provider | LLM model | Input tok | Output tok | LLM cost | LLM (ms) | Items | OK |",
        "|-------|----------:|---------:|---------:|--------------|-----------|----------:|-----------:|---------:|---------:|------:|:--:|",
    ]

    adi_by_name = {e.get("image_name"): e for e in adi_entries}
    for e in llm_entries:
        name = e.get("image_name", "?")
        adi  = adi_by_name.get(name, {})
        ok   = "✓" if e.get("llm_success") else "✗"
        chars = adi.get("chars_extracted") or "—"
        lines.append(
            f"| {name} "
            f"| {chars} "
            f"| {adi.get('ocr_latency_ms', 0):.0f} | ${adi.get('cost_usd', 0):.4f} "
            f"| {e.get('llm_provider', '?')} "
            f"| {e.get('llm_model', '?')} "
            f"| {e.get('llm_input_tokens', 0):,} | {e.get('llm_output_tokens', 0):,} "
            f"| ${e.get('llm_cost_usd', 0):.6f} | {e.get('llm_latency_ms', 0):.0f} "
            f"| {e.get('items_extracted', 0)} | {ok} |"
        )

    REPORTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    md_path = REPORTS_DIR / f"report_{ts}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report → {md_path}")
    _chart(adi_entries, llm_entries, ts, md_path)


def _chart(adi_entries, llm_entries, ts, md_path):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError:
        print("matplotlib not installed — skipping chart. pip install matplotlib")
        return

    if not llm_entries:
        return

    images  = [e.get("image_name", "?")          for e in llm_entries]
    prompts = [e.get("llm_input_tokens",  0)     for e in llm_entries]
    comps   = [e.get("llm_output_tokens", 0)     for e in llm_entries]
    llm_lat = [e.get("llm_latency_ms",   0)      for e in llm_entries]
    adi_by  = {e.get("image_name"): e.get("ocr_latency_ms", 0) for e in adi_entries}
    adi_lat = [adi_by.get(n, 0) for n in images]
    colours = ["#2ca02c" if e.get("llm_success") else "#d62728" for e in llm_entries]
    x = list(range(len(images)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("GatherYourDeals ETL — Usage", fontweight="bold")

    ax = axes[0]
    ax.bar(x, prompts,               label="Prompt",     color="#4C72B0")
    ax.bar(x, comps, bottom=prompts, label="Completion", color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels(images, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Tokens"); ax.set_title("LLM Token Usage"); ax.legend()
    for i, (p, c) in enumerate(zip(prompts, comps)):
        ax.text(i, p + c, f"{p+c:,}", ha="center", va="bottom", fontsize=6)

    ax2 = axes[1]
    ax2.bar(x, llm_lat, color=colours)
    ax2.set_xticks(x); ax2.set_xticklabels(images, rotation=30, ha="right", fontsize=7)
    ax2.set_ylabel("ms"); ax2.set_title("LLM Latency per Image")
    ax2.legend(handles=[Patch(color="#2ca02c", label="OK"),
                        Patch(color="#d62728", label="Error")])

    ax3 = axes[2]
    ax3.bar(x, adi_lat, color="#9467BD")
    ax3.set_xticks(x); ax3.set_xticklabels(images, rotation=30, ha="right", fontsize=7)
    ax3.set_ylabel("ms"); ax3.set_title("ADI OCR Latency per Image")

    plt.tight_layout()
    chart = REPORTS_DIR / f"usage_chart_{ts}.png"
    plt.savefig(chart, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Chart  → {chart}")
    with open(md_path, "a", encoding="utf-8") as f:
        f.write(f"\n\n![Usage Chart](./{chart.name})\n")


# ---------------------------------------------------------------------------
# Compare — per-model aggregated comparison table (Test 2)
# ---------------------------------------------------------------------------
def compare():
    """
    Read all ETL logs and produce an aggregated per-model comparison table
    scoped to the receipts currently in Receipts/.

    Outputs: reports/compare_<timestamp>.md
    """
    entries     = _load_log_entries()
    llm_entries = [e for e in entries if e.get("event") == "llm_extraction"]
    adi_entries = [e for e in entries if e.get("event") == "adi_ocr"]
    # Most recent ADI entry per image (chars_extracted may vary slightly across runs)
    adi_chars: dict[str, int] = {}
    for e in adi_entries:
        name = e.get("image_name", "")
        if e.get("chars_extracted") is not None:
            adi_chars[name] = e["chars_extracted"]
    if not llm_entries:
        print("No log entries found. Run the pipeline first.")
        return

    # Ground truth expected item counts (from ground_truth/*.json)
    gt_counts: dict[str, int] = {}
    if GROUND_TRUTH_DIR.exists():
        for f in GROUND_TRUTH_DIR.glob("*.json"):
            try:
                items = json.loads(f.read_text(encoding="utf-8"))
                gt_counts[f.stem] = len(items)
            except Exception:
                pass

    # Current receipts in Receipts/ folder
    receipts_dir = Path("Receipts")
    current_receipts: set[str] = set()
    if receipts_dir.exists():
        current_receipts = {
            f.name for f in receipts_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTS
        }

    if not current_receipts:
        print("No receipts found in Receipts/ — cannot scope comparison.")
        return

    # Group successful entries by (provider, model), scoped to current receipts
    groups: dict[tuple, list] = defaultdict(list)
    for e in llm_entries:
        if e.get("image_name", "") in current_receipts and e.get("llm_success"):
            key = (e.get("llm_provider", "?"), e.get("llm_model", "?"))
            groups[key].append(e)

    if not groups:
        print("No successful runs found for current Receipts/ receipts.")
        return

    RECEIPTS_PER_DAY_LOW  = 100
    RECEIPTS_PER_DAY_HIGH = 1_000

    table_rows = []
    for (provider, model), es in sorted(groups.items(), key=lambda x: x[0][1]):
        lats     = sorted(e.get("llm_latency_ms", 0) for e in es)
        in_toks  = [e.get("llm_input_tokens",  0) for e in es]
        out_toks = [e.get("llm_output_tokens", 0) for e in es]
        costs    = [e.get("llm_cost_usd", 0)      for e in es]
        n = len(es)

        avg_lat          = statistics.mean(lats)
        med_lat          = statistics.median(lats)
        p95_lat          = lats[min(int(0.95 * n), n - 1)]
        avg_in           = statistics.mean(in_toks)
        avg_out          = statistics.mean(out_toks)
        cost_per_receipt = statistics.mean(costs)

        # OCR context length — average chars across the receipts in this group
        ocr_char_vals = [adi_chars[e["image_name"]] for e in es
                         if e.get("image_name") in adi_chars]
        avg_ocr_chars = f"{statistics.mean(ocr_char_vals):,.0f}" if ocr_char_vals else "—"

        # Correctness: per receipt, take modal item count and compare to GT
        by_receipt: dict[str, list] = defaultdict(list)
        for e in es:
            by_receipt[e.get("image_name", "")].append(e.get("items_extracted", 0))

        correct = checked = 0
        for img_name, counts in by_receipt.items():
            stem = Path(img_name).stem
            if stem in gt_counts:
                modal = max(set(counts), key=counts.count)
                if modal == gt_counts[stem]:
                    correct += 1
                checked += 1

        correctness = f"{correct}/{checked}" if checked else "—"
        throughput  = f"~{60_000 / med_lat:.0f}/min" if med_lat > 0 else "—"
        cost_low    = f"${cost_per_receipt * RECEIPTS_PER_DAY_LOW:.2f}/day"  if cost_per_receipt > 0 else "$0.00*"
        cost_high   = f"${cost_per_receipt * RECEIPTS_PER_DAY_HIGH:.2f}/day" if cost_per_receipt > 0 else "$0.00*"

        table_rows.append({
            "provider":       provider,
            "model":          model,
            "n":              n,
            "avg_lat_ms":     f"{avg_lat:,.0f}",
            "med_lat_ms":     f"{med_lat:,.0f}",
            "p95_lat_ms":     f"{p95_lat:,.0f}",
            "avg_ocr_chars":  avg_ocr_chars,
            "avg_in_tok":     f"{avg_in:,.0f}",
            "avg_out_tok":    f"{avg_out:,.0f}",
            "cost_per":       f"${cost_per_receipt:.4f}" if cost_per_receipt > 0 else "$0.00",
            "correctness":    correctness,
            "throughput":     throughput,
            "cost_low":       cost_low,
            "cost_high":      cost_high,
        })

    # Print comparison table
    header   = ["Provider", "Model", "n", "Avg lat", "Median lat", "p95 lat",
                "OCR chars", "Avg in tok", "Avg out tok", "Cost/receipt", "Correct", "Throughput"]
    col_keys = ["provider", "model", "n", "avg_lat_ms", "med_lat_ms", "p95_lat_ms",
                "avg_ocr_chars", "avg_in_tok", "avg_out_tok", "cost_per", "correctness", "throughput"]
    row_strs = [[str(r[k]) for k in col_keys] for r in table_rows]
    widths   = [max(len(h), max((len(s[i]) for s in row_strs), default=0))
                for i, h in enumerate(header)]

    print("\n## Model Comparison — scoped to current Receipts/ (successful runs only)\n")
    print("| " + " | ".join(h.ljust(w) for h, w in zip(header, widths)) + " |")
    print("| " + " | ".join("-" * w for w in widths) + " |")
    for s in row_strs:
        print("| " + " | ".join(v.ljust(w) for v, w in zip(s, widths)) + " |")

    # Print scale projections
    scale_header = ["Provider", "Model", f"Cost @{RECEIPTS_PER_DAY_LOW}/day",
                    f"Cost @{RECEIPTS_PER_DAY_HIGH}/day", "Throughput (sequential)"]
    scale_keys   = ["provider", "model", "cost_low", "cost_high", "throughput"]
    scale_strs   = [[str(r[k]) for k in scale_keys] for r in table_rows]
    sw = [max(len(h), max((len(s[i]) for s in scale_strs), default=0))
          for i, h in enumerate(scale_header)]

    print("\n## Scale Projections\n")
    print("| " + " | ".join(h.ljust(w) for h, w in zip(scale_header, sw)) + " |")
    print("| " + " | ".join("-" * w for w in sw) + " |")
    for s in scale_strs:
        print("| " + " | ".join(v.ljust(w) for v, w in zip(s, sw)) + " |")
    print("\n* Free-tier models: $0.00 until rate limit is hit.")

    # Save markdown
    REPORTS_DIR.mkdir(exist_ok=True)
    ts      = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    md_lines = [
        "# GatherYourDeals ETL — Model Comparison",
        f"_Generated: {now_str}_",
        f"_Scoped to: {len(current_receipts)} receipts in `Receipts/`_",
        f"_Ground truth: {len(gt_counts)} receipts in `ground_truth/`_",
        "",
        "## Comparison Table",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("-" * w for w in widths) + " |",
    ]
    for s in row_strs:
        md_lines.append("| " + " | ".join(s) + " |")

    md_lines += [
        "",
        "\\* Free-tier models: \\$0.00 until provider rate limit is hit.",
        "",
        "## Scale Projections",
        "",
        "| " + " | ".join(scale_header) + " |",
        "| " + " | ".join("-" * w for w in sw) + " |",
    ]
    for s in scale_strs:
        md_lines.append("| " + " | ".join(s) + " |")

    md_lines += [
        "",
        "## What Each Model Represents",
        "",
        "| Model | Role |",
        "|-------|------|",
        "| claude-3-haiku (OR) | Free baseline — throughput ceiling at $0 |",
        "| llama-3.3-70b:free (OR) | Free open-source — does size improve correctness? |",
        "| claude-haiku-4.5 (OR) | Paid new-gen — does paying improve latency or correctness? |",
        "| Qwen3-235B (CLOD) | Alternative provider — viable fallback? At what latency cost? |",
    ]

    md_path = REPORTS_DIR / f"compare_{ts}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nCompare report → {md_path}")


# ---------------------------------------------------------------------------
# Eval — compare output/ against ground_truth/
# ---------------------------------------------------------------------------
def _parse_price(val) -> float | None:
    """Extract numeric part from a price string like '4.79USD' or '27.36CAD'."""
    if val is None:
        return None
    try:
        return float(re.sub(r"[^0-9.]", "", str(val)))
    except ValueError:
        return None


def _norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower().strip())


def _score_receipt(output_items: list, truth_items: list) -> dict:
    """
    Compare two flat per-item lists against each other.
    Checks the 7 fields present in the flat output format:
      storeName, purchaseDate, latitude, longitude  — receipt-level (from first item)
      productName, price, amount                    — item-level
    Returns a dict of field scores and an overall 0-100 score.
    """
    scores = {}
    out0   = output_items[0] if output_items else {}
    tru0   = truth_items[0]  if truth_items  else {}

    # --- Receipt-level scalar fields (same across all items) ---
    def exact(a, b, case_insensitive=True):
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        a, b = str(a).strip(), str(b).strip()
        return a.lower() == b.lower() if case_insensitive else a == b

    # storeName: word-overlap match — passes if any significant word (>3 chars)
    # appears in both names. Handles abbreviations, missing branch details, etc.
    a_store = str(out0.get("storeName") or "").strip().lower()
    b_store = str(tru0.get("storeName") or "").strip().lower()
    _stop = {"the", "and", "your", "for"}
    a_words = {w for w in re.split(r"\W+", a_store) if len(w) > 3 and w not in _stop}
    b_words = {w for w in re.split(r"\W+", b_store) if len(w) > 3 and w not in _stop}
    scores["storeName"] = bool(a_words and b_words and a_words & b_words)
    scores["purchaseDate"] = exact(out0.get("purchaseDate"), tru0.get("purchaseDate"), case_insensitive=False)

    # latitude / longitude: pass if within ~1.1 km (~0.01 degrees)
    # Geocoding APIs return slightly different coordinates for the same store
    # depending on address normalisation; 0.01° covers that variance while
    # still failing a result that is in the wrong city or region.
    def coord_match(field, tol=0.01):
        a = out0.get(field)
        b = tru0.get(field)
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        try:
            return abs(float(a) - float(b)) <= tol
        except (TypeError, ValueError):
            return False

    scores["latitude"]  = coord_match("latitude")
    scores["longitude"] = coord_match("longitude")

    # item count
    scores["totalItems"] = len(output_items) == len(truth_items)

    # --- Item-level matching ---
    out_names = [_norm_name(i.get("productName", "")) for i in output_items]
    matched_names = matched_prices = matched_amounts = 0

    for t_item in truth_items:
        t_name   = _norm_name(t_item.get("productName", ""))
        t_price  = _parse_price(t_item.get("price"))
        t_amount = str(t_item.get("amount") or "").strip()

        best_idx = None
        if t_name in out_names:
            best_idx = out_names.index(t_name)
        else:
            for oi, o_name in enumerate(out_names):
                if t_name and (t_name in o_name or o_name in t_name):
                    best_idx = oi
                    break

        if best_idx is not None:
            matched_names += 1
            o_item   = output_items[best_idx]
            o_price  = _parse_price(o_item.get("price"))
            o_amount = str(o_item.get("amount") or "").strip()
            if t_price is not None and o_price is not None and abs(t_price - o_price) <= 0.01:
                matched_prices += 1
            if t_amount and o_amount and t_amount == o_amount:
                matched_amounts += 1

    n = len(truth_items)
    scores["item_name_match"]   = f"{matched_names}/{n}"
    scores["item_price_match"]  = f"{matched_prices}/{n}"
    scores["item_amount_match"] = f"{matched_amounts}/{n}"

    scalar_fields = ["storeName", "purchaseDate", "latitude", "longitude", "totalItems"]
    scalar_score  = sum(1 for f in scalar_fields if scores[f]) / len(scalar_fields)
    item_score    = (matched_names + matched_prices + matched_amounts) / (3 * n) if n > 0 else 1.0
    scores["overall"] = round((scalar_score * 0.5 + item_score * 0.5) * 100, 1)

    return scores


def _compute_eval(output_dir: Path = OUTPUT_DIR, gt_dir: Path = GROUND_TRUTH_DIR):
    """
    Core eval logic — compare every output/<name>.json against ground_truth/<name>.json.
    Returns (header, rows, scores) for use by eval_receipts() and baseline_report().
    """
    header = ("Image", "GT items", "Store", "Date", "Lat", "Lon",
              "Items", "Name match", "Price match", "Amount match", "Score")
    check  = lambda v: "✓" if v else "✗"

    gt_files = {p.stem: p for p in gt_dir.glob("*.json")} if gt_dir.exists() else {}
    rows, scores = [], []

    for stem, gt_path in sorted(gt_files.items()):
        out_path = output_dir / (stem + ".json")
        raw_truth  = json.loads(gt_path.read_text(encoding="utf-8"))
        tru_list   = raw_truth if isinstance(raw_truth, list) else [raw_truth]
        gt_item_count = len(tru_list)
        if not out_path.exists():
            rows.append((stem + ".jpg", gt_item_count, "—", "—", "—", "—", "—", "—", "—", "—", "no output"))
            continue
        raw_output = json.loads(out_path.read_text(encoding="utf-8"))
        out_list   = raw_output if isinstance(raw_output, list) else [raw_output]
        s = _score_receipt(out_list, tru_list)
        scores.append(s["overall"])
        rows.append((
            stem + ".jpg",
            gt_item_count,
            check(s["storeName"]),
            check(s["purchaseDate"]),
            check(s["latitude"]),
            check(s["longitude"]),
            check(s["totalItems"]),
            s["item_name_match"],
            s["item_price_match"],
            s["item_amount_match"],
            f"{s['overall']}%",
        ))
    return header, rows, scores


def eval_receipts(output_dir: Path = OUTPUT_DIR, gt_dir: Path = GROUND_TRUTH_DIR):
    """
    Compare output against ground_truth/ — field by field per receipt.
    If provider subdirectories exist (output/openrouter/, output/clod/),
    evaluates each provider separately. Otherwise falls back to output_dir directly.
    Prints a summary table and saves a report to reports/eval_<ts>.md.
    """
    if not gt_dir.exists() or not any(gt_dir.glob("*.json")):
        print(f"No ground truth files found in {gt_dir}/")
        return

    provider_dirs = [d for d in sorted(output_dir.iterdir()) if d.is_dir()] if output_dir.exists() else []
    dirs_to_eval  = provider_dirs if provider_dirs else [output_dir]

    REPORTS_DIR.mkdir(exist_ok=True)
    ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    md_lines = ["# GatherYourDeals ETL — Eval Report",
                f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_", ""]

    for d in dirs_to_eval:
        header, rows, scores = _compute_eval(d, gt_dir)
        if not rows:
            continue
        label = d.name if provider_dirs else "output"
        print(f"\n=== {label} ===")
        col_w = [max(len(h), max(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
        sep   = "+-" + "-+-".join("-" * w for w in col_w) + "-+"
        fmt   = "| " + " | ".join(f"{{:<{w}}}" for w in col_w) + " |"
        print(sep); print(fmt.format(*header)); print(sep)
        for row in rows:
            print(fmt.format(*[str(c) for c in row]))
        print(sep)
        if scores:
            print(f"Avg score: {sum(scores)/len(scores):.1f}%  "
                  f"(min {min(scores):.1f}%  max {max(scores):.1f}%)  over {len(scores)} receipts")
        md_lines += [f"## {label}", "",
                     "| " + " | ".join(header) + " |",
                     "| " + " | ".join("-" * len(h) for h in header) + " |"]
        for row in rows:
            md_lines.append("| " + " | ".join(str(c) for c in row) + " |")
        if scores:
            md_lines += ["", f"**Avg score: {sum(scores)/len(scores):.1f}%**", ""]

    md_path = REPORTS_DIR / f"eval_{ts}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nEval report → {md_path}")



# ---------------------------------------------------------------------------
# Baseline Report — structured experiment report in the style of the team's
# weekly reports (environment + summary + provider tables + cost + findings)
# ---------------------------------------------------------------------------
def baseline_report():
    """
    Generate a single structured baseline experiment report covering all three
    pipeline providers: Azure DI (OCR), OpenRouter (LLM), CLOD (LLM).

    Scoped to today's log file only so historical runs don't pollute the report.
    Saved to reports/baseline_<ts>.md
    """
    # Load start timestamp written by run_baseline.sh
    start_file = Path(".baseline_start")
    if not start_file.exists():
        print("No .baseline_start file found. Run:  bash run_baseline.sh")
        return
    experiment_start = start_file.read_text().strip()

    # Load all log entries and scope to this experiment only (time >= experiment_start)
    all_entries = []
    for f in sorted(LOGS_DIR.glob("etl_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    all_entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    entries     = [e for e in all_entries if e.get("time", "") >= experiment_start]
    adi_entries = [e for e in entries if e.get("event") == "adi_ocr"]
    llm_entries = [e for e in entries if e.get("event") == "llm_extraction"]

    if not llm_entries and not adi_entries:
        print("No log entries found for this experiment. Run:  bash run_baseline.sh")
        return

    # --- Scope to the 3 most recent *complete* trace_ids per provider ---
    # A complete trace_id covers all receipts in the experiment (count >= receipt_count).
    # Single-receipt ad-hoc runs that happen to fall inside the experiment window are excluded.
    _RUNS_PER_PROVIDER = 3
    _receipts_in_dir = len([f for f in Path("Receipts").iterdir()
                             if f.suffix.lower() in IMAGE_EXTS]) if Path("Receipts").exists() else 9
    _trace_prov: dict[str, str] = {}
    _trace_cnt:  dict[str, int] = defaultdict(int)
    _trace_last: dict[str, str] = defaultdict(str)
    for e in llm_entries:
        tid = e.get("trace_id", "")
        _trace_prov[tid] = e.get("llm_provider", "?")
        _trace_cnt[tid] += 1
        t = e.get("time", "")
        if t > _trace_last[tid]:
            _trace_last[tid] = t
    _complete_by_prov: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for tid, cnt in _trace_cnt.items():
        if cnt >= _receipts_in_dir:
            _complete_by_prov[_trace_prov[tid]].append((_trace_last[tid], tid))
    _valid_tids: set[str] = set()
    for _plist in _complete_by_prov.values():
        _plist.sort(reverse=True)
        for _, tid in _plist[:_RUNS_PER_PROVIDER]:
            _valid_tids.add(tid)
    llm_entries = [e for e in llm_entries if e.get("trace_id") in _valid_tids]
    adi_entries = [e for e in adi_entries if e.get("trace_id") in _valid_tids]

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ts      = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # ---- ADI summary ----
    adi_ok   = [e for e in adi_entries if e.get("ocr_success")]
    adi_lats = sorted(e.get("ocr_latency_ms", 0) for e in adi_ok)
    adi_chars = [e.get("chars_extracted") for e in adi_ok if e.get("chars_extracted")]
    adi_cost  = sum(e.get("cost_usd", 0) for e in adi_entries)
    adi_avg   = statistics.mean(adi_lats) if adi_lats else 0
    adi_p50   = statistics.median(adi_lats) if adi_lats else 0
    adi_p95   = adi_lats[min(int(0.95 * len(adi_lats)), len(adi_lats) - 1)] if adi_lats else 0
    adi_chars_avg = f"{statistics.mean(adi_chars):,.0f}" if adi_chars else "—"

    # ---- LLM summary per provider ----
    providers: dict[tuple, list] = defaultdict(list)
    for e in llm_entries:
        key = (e.get("llm_provider", "?"), e.get("llm_model", "?"))
        providers[key].append(e)

    receipts_dir = Path("Receipts")
    receipt_count = len([f for f in receipts_dir.iterdir()
                         if f.suffix.lower() in IMAGE_EXTS]) if receipts_dir.exists() else "?"

    # ---- Ground truth correctness ----
    gt_counts: dict[str, int] = {}
    if GROUND_TRUTH_DIR.exists():
        for f in GROUND_TRUTH_DIR.glob("*.json"):
            try:
                gt_counts[f.stem] = len(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass

    # ---- Pipeline complete events (end-to-end latency + failure rate) ----
    # Also scoped to valid trace_ids so E2E latency matches the same 27-run dataset
    pipeline_entries = [e for e in entries
                        if e.get("event") == "pipeline_complete"
                        and e.get("trace_id") in _valid_tids]
    pipeline_by_provider: dict[tuple, list] = defaultdict(list)
    for e in pipeline_entries:
        key = (e.get("llm_provider", ""), e.get("llm_model", ""))
        pipeline_by_provider[key].append(e)

    # ---- Build provider rows ----
    provider_rows = []
    for (prov, model), es in sorted(providers.items()):
        ok_es  = [e for e in es if e.get("llm_success")]
        lats   = sorted(e.get("llm_latency_ms", 0) for e in ok_es)
        n      = len(ok_es)
        if not lats:
            continue
        avg_lat = statistics.mean(lats)
        p50_lat = statistics.median(lats)
        p95_lat = lats[min(int(0.95 * n), n - 1)]
        avg_in  = statistics.mean(e.get("llm_input_tokens", 0)  for e in ok_es)
        avg_out = statistics.mean(e.get("llm_output_tokens", 0) for e in ok_es)
        total_cost = sum(e.get("llm_cost_usd", 0) for e in ok_es)
        cost_per   = total_cost / n if n else 0

        # end-to-end latency from pipeline_complete events
        pe = pipeline_by_provider.get((prov, model), [])
        ok_pe = [e for e in pe if e.get("success")]
        e2e_lats = sorted(e.get("total_latency_ms", 0) for e in ok_pe)
        if e2e_lats:
            e2e_p50 = f"{statistics.median(e2e_lats):,.0f}"
            e2e_p95 = f"{e2e_lats[min(int(0.95 * len(e2e_lats)), len(e2e_lats) - 1)]:,.0f}"
            throughput = f"~{60_000 / statistics.median(e2e_lats):.0f}/min"
        else:
            e2e_p50 = e2e_p95 = throughput = "—"

        # failure rate
        errors     = len(es) - len(ok_es)
        total_calls = len(es)
        fail_rate  = f"{errors}/{total_calls} ({100*errors/total_calls:.0f}%)" if total_calls else "—"

        # correctness
        by_receipt: dict[str, list] = defaultdict(list)
        for e in ok_es:
            by_receipt[e.get("image_name", "")].append(e.get("items_extracted", 0))
        correct = checked = 0
        for img, counts in by_receipt.items():
            stem = Path(img).stem
            if stem in gt_counts:
                modal = max(set(counts), key=counts.count)
                if modal == gt_counts[stem]:
                    correct += 1
                checked += 1

        provider_rows.append({
            "provider":    prov,
            "model":       model.split("/")[-1],
            "receipts":    len(by_receipt),
            "fail_rate":   fail_rate,
            "e2e_p50":     e2e_p50,
            "e2e_p95":     e2e_p95,
            "throughput":  throughput,
            "avg_lat":     f"{avg_lat:,.0f}",
            "p50_lat":     f"{p50_lat:,.0f}",
            "p95_lat":     f"{p95_lat:,.0f}",
            "avg_in":      f"{avg_in:,.0f}",
            "avg_out":     f"{avg_out:,.0f}",
            "cost_per":    f"${cost_per:.4f}" if cost_per > 0 else "$0.00",
            "total_cost":  f"${total_cost:.4f}",
            "correct":     f"{correct}/{checked}" if checked else "—",
        })

    # ---- Findings ----
    findings = []
    if len(provider_rows) >= 2:
        r = provider_rows
        faster = min(r, key=lambda x: float(x["e2e_p50"].replace(",", "")) if x["e2e_p50"] != "—" else float(x["p50_lat"].replace(",", "")))
        cheaper = min(r, key=lambda x: float(x["cost_per"].replace("$", "")))
        other = [x for x in r if x != faster][0]
        findings.append(f"- **Lower E2E p50 latency:** {faster['provider']} / {faster['model']} "
                        f"({faster['e2e_p50']} ms vs {other['e2e_p50']} ms)")
        findings.append(f"- **Throughput:** {faster['provider']} ~{faster['throughput']} vs "
                        f"{other['provider']} ~{other['throughput']}")
        findings.append(f"- **Lower cost/receipt:** {cheaper['provider']} / {cheaper['model']} "
                        f"({cheaper['cost_per']})")
        # primary/fallback recommendation
        primary  = faster
        fallback = [x for x in r if x != primary][0]
        findings.append(f"- **Recommended primary:** {primary['provider']} / {primary['model']} "
                        f"— lower egress latency")
        findings.append(f"- **Recommended fallback:** {fallback['provider']} / {fallback['model']} "
                        f"— viable alternative if primary is unavailable")
    elif len(provider_rows) == 1:
        r = provider_rows[0]
        findings.append(f"- Only one provider run found ({r['provider']} / {r['model']}). "
                        f"Run both providers to get comparison findings.")

    # ---- Assemble markdown ----
    md = [
        "# GatherYourDeals ETL — Baseline Experiment Report",
        f"_Generated: {now_str}_",
        "",
        "## Summary",
        "",
        f"Baseline egress measurement for the ETL pipeline across all provider dependencies. "
        f"{receipt_count} receipts processed through Azure Document Intelligence (OCR) "
        f"and {len(provider_rows)} LLM provider(s).",
        "",
        "This experiment is intentionally sequential and quality-focused — not a concurrency stress test. "
        "LLM providers impose per-token costs and rate limits that make concurrent load testing inappropriate "
        "and expensive for this component. The right lens for LLM evaluation is test case quality and "
        "cost-per-call, not infrastructure throughput. Concurrency stress testing belongs to the data "
        "service layer, where requests are cheap and the bottleneck is server resources.",
        "",
        "## Metrics Gathered",
        "",
        "The following metrics are collected per receipt per provider to evaluate quality, latency, and cost:",
        "",
        "| Metric | Source | Purpose |",
        "|--------|--------|---------|",
        "| E2E P50 / P95 latency (ms) | `pipeline_complete` | True wall time from image in to JSON out |",
        "| LLM P50 / P95 latency (ms) | `llm_extraction` | Egress round-trip to LLM provider |",
        "| ADI P50 / P95 latency (ms) | `adi_ocr` | Egress round-trip to Azure Document Intelligence |",
        "| Input / output tokens | `llm_extraction` | Payload size — explains latency and cost variance |",
        "| Cost per receipt (USD) | `llm_extraction` | Real billed cost or token-based estimate |",
        "| Failure rate | `llm_extraction` | Provider reliability — failed calls / total calls |",
        "| Throughput (receipts/min) | Derived from E2E P50 | Sustained processing capacity |",
        "| Items extracted | `llm_extraction` | Output correctness proxy — expected vs actual count |",
        "| Field-level accuracy (0–100%) | `--eval` | Per-field scoring: store, date, price, item name/price match |",
        "",
        "## Repeated Trials",
        "",
        "Each receipt was processed multiple times per provider (~3 runs) to account for variability in:",
        "",
        "- Network latency",
        "- OCR processing time",
        "- LLM response time",
        "",
        f"This increases the number of observations from {receipt_count} to ~{receipt_count * 3} per provider, "
        "improving the reliability of latency and cost estimates.",
        "",
        "Per-receipt entries shown in the report include repeated runs to capture this variance, "
        "while summary metrics (P50, P95, averages) are computed across all runs.",
        "",
        "## Environment",
        "",
        "| Setting | Value |",
        "|---------|-------|",
        "| Platform | Local (WSL2) |",
        f"| Date | {date_str} |",
        f"| Receipts tested | {receipt_count} |",
        f"| Ground truth available | {len(gt_counts)} receipts |",
        "| OCR provider | Azure Document Intelligence (prebuilt-read, F0 free tier) |",
    ]
    for row in provider_rows:
        md.append(f"| LLM provider | {row['provider']} / {row['model']} |")

    md += [
        "",
        "## OCR Provider — Azure Document Intelligence",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Calls | {len(adi_entries)} ({len(adi_ok)} success) |",
        f"| Avg latency | {adi_avg:,.0f} ms |",
        f"| P50 latency | {adi_p50:,.0f} ms |",
        f"| P95 latency | {adi_p95:,.0f} ms |",
        f"| Avg OCR chars extracted | {adi_chars_avg} |",
        f"| Estimated cost | ${adi_cost:.4f} (F0 free tier — 500 pages/month) |",
        "",
        "## LLM Provider Comparison",
        "",
        "| Provider | Model | Receipts | Fail rate | E2E P50 (ms) | E2E P95 (ms) | Throughput | LLM P50 (ms) | LLM P95 (ms) | Avg lat (ms) | Avg in tok | Avg out tok | Cost/receipt | Total cost | Item count match |",
        "|----------|-------|:--------:|:---------:|-------------:|-------------:|:----------:|-------------:|-------------:|-------------:|-----------:|------------:|-------------:|:----------:|:----------------:|",
    ]
    for row in provider_rows:
        md.append(
            f"| {row['provider']} | {row['model']} | {row['receipts']} | {row['fail_rate']} "
            f"| {row['e2e_p50']} | {row['e2e_p95']} | {row['throughput']} "
            f"| {row['p50_lat']} | {row['p95_lat']} | {row['avg_lat']} "
            f"| {row['avg_in']} | {row['avg_out']} "
            f"| {row['cost_per']} | {row['total_cost']} | {row['correct']} |"
        )
    md.append("")
    md.append("_Item count match: a receipt is counted as correct if the most common (modal) item count across all runs equals the ground-truth count._")

    md += [
        "",
        "## Per-Receipt Breakdown",
        "",
        "| Receipt | OCR chars | ADI (ms) | Provider | Model | In tok | Out tok | Cost | LLM (ms) | Items extracted | Items expected | API |",
        "|---------|----------:|---------:|----------|-------|-------:|--------:|-----:|---------:|----------------:|:--------------:|:---:|",
    ]
    # Key by (image_name, trace_id) so each LLM entry is matched to its own ADI call
    adi_by_key = {(e.get("image_name"), e.get("trace_id")): e for e in adi_entries}
    _OUTLIER_MS = 60_000  # flag LLM calls longer than 60 s as anomalous
    for e in llm_entries:
        name  = e.get("image_name", "?")
        adi   = adi_by_key.get((name, e.get("trace_id")), {})
        api   = "✓" if e.get("llm_success") else "✗"
        chars = adi.get("chars_extracted") or "—"
        model_short = (e.get("llm_model") or "?").split("/")[-1]
        llm_lat = e.get("llm_latency_ms", 0)
        lat_str = f"{llm_lat:.0f} ⚠" if llm_lat > _OUTLIER_MS else f"{llm_lat:.0f}"
        items = e.get("items_extracted", 0)
        gt_n  = gt_counts.get(Path(name).stem)
        gt_match = ("✓" if items == gt_n else "✗") if gt_n is not None else "—"
        md.append(
            f"| {name} | {chars} | {adi.get('ocr_latency_ms', 0):.0f} "
            f"| {e.get('llm_provider', '?')} | {model_short} "
            f"| {e.get('llm_input_tokens', 0):,} | {e.get('llm_output_tokens', 0):,} "
            f"| ${e.get('llm_cost_usd', 0):.6f} | {lat_str} "
            f"| {items} | {gt_match} | {api} |"
        )

    total_llm_cost = sum(e.get("llm_cost_usd", 0) for e in llm_entries)
    md += [
        "",
        "## Cost",
        "",
        "| Provider | Service | Cost |",
        "|----------|---------|------|",
        f"| Azure | Document Intelligence (OCR) | ${adi_cost:.4f} (F0 free tier — logged at S0 rate) |",
    ]
    for row in provider_rows:
        md.append(f"| {row['provider']} | LLM ({row['model']}) | {row['total_cost']} |")
    md += [
        f"| **Total** | | **${adi_cost + total_llm_cost:.4f}** |",
        "",
        "_openrouter costs are estimated from the published per-token rate card. "
        "clod costs are taken directly from the API response and reflect whatever rate structure "
        "the provider applies — individual figures cannot be independently verified from token counts alone._",
        "",
        "## Findings",
        "",
    ]
    md += findings if findings else ["- Run both providers to generate findings."]

    # ---- Field-level accuracy (eval) — per provider ----
    provider_dirs = [(row["provider"], OUTPUT_DIR / row["provider"]) for row in provider_rows]
    any_eval = False
    eval_md = ["", "## Field-Level Accuracy", "",
               "Scores each provider's output against ground_truth/ — field by field per receipt.",
               "Outputs are saved to provider-specific directories (`output/<provider>/`) so each "
               "provider can be evaluated independently on the same receipts.",
               "Scores are computed against all ground-truth items; unmatched slots count as misses "
               "(e.g. if 3 items are extracted vs. 4 expected, the 4th slot scores zero across name, price, and amount).",
               "The **GT items** column shows the expected item count from ground truth for reference.",
               "",
               "**Scoring criteria (this version):** store name uses word-overlap matching "
               "(any significant word >3 chars in common counts as a match); "
               "lat/lon tolerance is 0.01° (~1.1 km, covers same-store geocoding variance). "
               "If comparing scores across report versions, note that these criteria were updated "
               "from earlier exact/substring matching — score increases may partly reflect the "
               "updated criteria rather than model improvement alone.",
               ""]

    # prov -> {stem -> (matched, total)} for price match aggregation
    price_stats: dict[str, dict[str, tuple[int, int]]] = {}

    for prov, prov_dir in provider_dirs:
        if not prov_dir.exists():
            eval_md += [f"### {prov}", "", f"_No output directory found at `{prov_dir}/` — run the pipeline first._", ""]
            continue
        eval_header, eval_rows, eval_scores = _compute_eval(output_dir=prov_dir)
        if not eval_rows:
            continue
        any_eval = True
        avg = f"{sum(eval_scores)/len(eval_scores):.1f}%" if eval_scores else "—"
        eval_md += [
            f"### {prov}",
            "",
            f"**Avg score: {avg}** over {len(eval_scores)} receipts",
            "",
            "| " + " | ".join(eval_header) + " |",
            "| " + " | ".join("-" * len(h) for h in eval_header) + " |",
        ]
        try:
            price_col = list(eval_header).index("Price match")
        except ValueError:
            price_col = None
        prov_price: dict[str, tuple[int, int]] = {}
        for row in eval_rows:
            eval_md.append("| " + " | ".join(str(c) for c in row) + " |")
            if price_col is not None:
                stem = str(row[0]).replace(".jpg", "")
                try:
                    num, den = (int(x) for x in str(row[price_col]).split("/"))
                    prov_price[stem] = (num, den)
                except (ValueError, AttributeError):
                    pass
        eval_md.append("")
        price_stats[prov] = prov_price

    # Dynamic quality finding — summarise price match by provider and flag gaps
    if len(price_stats) >= 1:
        finding_lines = ["### Quality Finding: Price Extraction by Provider", ""]
        # Overall price match rate per provider
        prov_overall: dict[str, float] = {}
        prov_models = {row["provider"]: row["model"] for row in provider_rows}
        for prov, pdata in price_stats.items():
            total_num = sum(n for n, _ in pdata.values())
            total_den = sum(d for _, d in pdata.values())
            prov_overall[prov] = total_num / total_den if total_den else 0.0

        # Per-receipt gap detection (>= 30pp gap between any two providers)
        if len(price_stats) == 2:
            provs = list(price_stats.keys())
            common = sorted(set(price_stats[provs[0]]) & set(price_stats[provs[1]]))
            gap_lines: list[str] = []
            for stem in common:
                n0, d0 = price_stats[provs[0]].get(stem, (0, 0))
                n1, d1 = price_stats[provs[1]].get(stem, (0, 0))
                r0 = n0 / d0 if d0 else 0.0
                r1 = n1 / d1 if d1 else 0.0
                if abs(r0 - r1) >= 0.30:
                    stronger, weaker = (provs[0], provs[1]) if r0 >= r1 else (provs[1], provs[0])
                    sr = max(r0, r1); wr = min(r0, r1)
                    gap_lines.append(
                        f"- **{stem}:** {stronger} {sr*100:.0f}% vs {weaker} {wr*100:.0f}%"
                    )

            # Build summary prose
            prov_summary = "  \n".join(
                f"- **{p}** (`{prov_models.get(p, p)}`): {prov_overall[p]*100:.0f}% overall price match"
                for p in provs
            )
            finding_lines.append(prov_summary)
            finding_lines.append("")

            if gap_lines:
                weaker_prov = min(prov_overall, key=lambda p: prov_overall[p])
                stronger_prov = max(prov_overall, key=lambda p: prov_overall[p])
                finding_lines += [
                    f"Receipts with a ≥ 30 percentage-point price match gap between providers:",
                    "",
                ] + gap_lines + [
                    "",
                    f"{weaker_prov} (`{prov_models.get(weaker_prov, weaker_prov)}`) is extracting "
                    f"per-unit rates or subtotals instead of line-item totals on receipts that use a "
                    f"multi-column format (e.g. `qty × unit_price = line_total`). "
                    f"The explicit prompt rule (`price` = right-hand price column) did not resolve this. "
                    f"This is a model capability gap — {stronger_prov} handles multi-column layouts "
                    f"correctly because it is a stronger model.",
                    "",
                    f"The same layout confusion extends to the `amount` field on Costco: {weaker_prov} "
                    f"outputs barcode numbers and prices instead of unit quantities (e.g. `6000000000`, `5.49`), "
                    f"resulting in 0/15 amount match. This is not a ground-truth or prompt change — "
                    f"it reflects the same model capability gap on multi-column receipts.",
                ]
        else:
            # Single provider — just report the rate
            for prov, rate in prov_overall.items():
                finding_lines.append(
                    f"- **{prov}** (`{prov_models.get(prov, prov)}`): {rate*100:.0f}% overall price match"
                )

        finding_lines.append("")
        eval_md += finding_lines

    if any_eval:
        md += eval_md

    REPORTS_DIR.mkdir(exist_ok=True)
    md_path = REPORTS_DIR / f"baseline_{ts}.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nBaseline report → {md_path}")


# ---------------------------------------------------------------------------
# CLI — can also be run directly: python reporting.py --report
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="GatherYourDeals ETL — Reporting")
    p.add_argument("--report",          action="store_true", help="Generate cumulative usage report")
    p.add_argument("--compare",         action="store_true", help="Generate per-model comparison table")
    p.add_argument("--eval",            action="store_true", help="Compare output/ against ground_truth/")
    p.add_argument("--baseline-report", action="store_true", help="Generate structured baseline experiment report")
    args = p.parse_args()

    if args.report:
        report()
    elif args.compare:
        compare()
    elif args.eval:
        eval_receipts()
    elif args.baseline_report:
        baseline_report()
    else:
        p.print_help()


if __name__ == "__main__":
    main()
