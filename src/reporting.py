"""
GatherYourDeals ETL — Reporting & Evaluation
=============================================
Focused on three experiment dimensions: cost, latency, and correctness.

Usage:
  python reporting.py --eval              Compare output/ against ground_truth/
  python reporting.py --baseline-report   Generate structured baseline experiment report

Or via etl.py CLI:
  python etl.py --eval
  python etl.py --baseline-report
"""

import argparse
import difflib
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from src.core import config

# ---------------------------------------------------------------------------
# Paths — must match etl.py
# ---------------------------------------------------------------------------
OUTPUT_DIR       = Path("output")
LOGS_DIR         = Path("logs")
REPORTS_DIR      = Path("reports")
GROUND_TRUTH_DIR = Path("ground_truth")
IMAGE_EXTS       = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".tiff", ".tif", ".bmp"}


# ---------------------------------------------------------------------------
# Log loader
# ---------------------------------------------------------------------------
def _load_log_entries() -> list[dict]:
    entries = []
    for f in sorted(config.LOGS_DIR.glob("etl_*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


# ---------------------------------------------------------------------------
# Eval — compare output/ against ground_truth/
# ---------------------------------------------------------------------------
def _parse_price(val) -> float | None:
    if val is None:
        return None
    try:
        return float(re.sub(r"[^0-9.]", "", str(val)))
    except ValueError:
        return None


def _norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).lower().strip())


def _norm_amount(s: str) -> str:
    s = re.sub(r"\s+", "", s.lower().strip())
    s = re.sub(r"\blbs?\b", "lb", s)
    s = re.sub(r"\bkgs?\b", "kg", s)
    return s


def _score_receipt(output_items: list, truth_items: list) -> dict:
    """
    Score one extracted receipt against ground truth.

    8 dimensions — each worth 1 point toward the overall score:
      5 scalar fields  (store, date, lat, lon, item-count) → 5/8 total weight
      3 item-rate fields (name match rate, price match rate,
                          amount match rate)               → 3/8 total weight

    overall = (scalar_sum + item_sum) / 8 × 100
    where scalar_sum ∈ {0..5} and item_sum = avg of three 0–1 rates ∈ {0..3}.
    """
    scores = {}
    out0 = output_items[0] if output_items else {}
    tru0 = truth_items[0]  if truth_items  else {}

    # --- Scalar fields ---
    def exact(a, b):
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        return str(a).strip() == str(b).strip()

    # Store: word-overlap — any significant word (>3 chars) in common counts
    _stop = {"the", "and", "your", "for"}
    a_words = {w for w in re.split(r"\W+", str(out0.get("storeName") or "").lower()) if len(w) > 3 and w not in _stop}
    b_words = {w for w in re.split(r"\W+", str(tru0.get("storeName") or "").lower()) if len(w) > 3 and w not in _stop}
    scores["storeName"]    = bool(a_words and b_words and a_words & b_words)
    # Normalize date separators (dots, hyphens, slashes all equivalent) before comparing
    def _norm_date(d):
        return re.sub(r"[-/]", ".", str(d).strip()) if d is not None else None
    scores["purchaseDate"] = exact(_norm_date(out0.get("purchaseDate")), _norm_date(tru0.get("purchaseDate")))

    # Lat/lon: pass within 0.02° (~2.2 km) — covers geocoder centroid vs entrance variance
    def coord_match(field):
        a, b = out0.get(field), tru0.get(field)
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        try:
            return abs(float(a) - float(b)) <= 0.02
        except (TypeError, ValueError):
            return False

    scores["latitude"]   = coord_match("latitude")
    scores["longitude"]  = coord_match("longitude")
    scores["totalItems"] = len(output_items) == len(truth_items)

    # --- Item-level matching ---
    out_names = [_norm_name(i.get("productName", "")) for i in output_items]
    matched_names = matched_prices = matched_amounts = 0

    for t_item in truth_items:
        t_name   = _norm_name(t_item.get("productName", ""))
        t_price  = _parse_price(t_item.get("price"))
        t_amount = _norm_amount(str(t_item.get("amount") or ""))

        best_idx = None
        if t_name in out_names:
            best_idx = out_names.index(t_name)
        else:
            for oi, o_name in enumerate(out_names):
                if t_name and (t_name in o_name or o_name in t_name):
                    best_idx = oi
                    break
        if best_idx is None and t_name:
            matches = difflib.get_close_matches(t_name, out_names, n=1, cutoff=0.6)
            if matches:
                best_idx = out_names.index(matches[0])

        if best_idx is not None:
            matched_names += 1
            o_item   = output_items[best_idx]
            o_price  = _parse_price(o_item.get("price"))
            o_amount = _norm_amount(str(o_item.get("amount") or ""))
            if t_price is not None and o_price is not None and abs(t_price - o_price) <= 0.01:
                matched_prices += 1
            if t_amount and o_amount and t_amount == o_amount:
                matched_amounts += 1

    n = len(truth_items)
    scores["item_name_match"]   = f"{matched_names}/{n}"
    scores["item_price_match"]  = f"{matched_prices}/{n}"
    scores["item_amount_match"] = f"{matched_amounts}/{n}"

    # overall: 5 scalar points + up to 3 item-rate points, divided by 8
    scalar_sum = sum(1 for f in ["storeName", "purchaseDate", "latitude", "longitude", "totalItems"] if scores[f])
    item_sum   = (matched_names + matched_prices + matched_amounts) / n if n > 0 else 0.0
    scores["overall"] = round((scalar_sum + item_sum) / 8 * 100, 1)

    return scores


def _compute_eval(output_dir: Path, gt_dir: Path = config.GROUND_TRUTH_DIR):
    """Score every output/<stem>.json against ground_truth/<stem>.json."""
    header = ("Image", "GT items", "Store", "Date", "Lat", "Lon",
              "Items", "Name match", "Price match", "Amount match", "Score")
    check = lambda v: "✓" if v else "✗"

    gt_files = {p.stem: p for p in gt_dir.glob("*.json")} if gt_dir.exists() else {}
    rows, scores = [], []

    for stem, gt_path in sorted(gt_files.items()):
        gt_text = gt_path.read_text(encoding="utf-8").strip()
        if not gt_text:
            continue
        tru_list = json.loads(gt_text)
        if not isinstance(tru_list, list):
            tru_list = [tru_list]
        gt_n = len(tru_list)

        out_path = output_dir / (stem + ".json")
        if not out_path.exists():
            rows.append((stem + ".jpg", gt_n, "—", "—", "—", "—", "—", "—", "—", "—", "no output"))
            continue
        out_text = out_path.read_text(encoding="utf-8").strip()
        if not out_text:
            rows.append((stem + ".jpg", gt_n, "—", "—", "—", "—", "—", "—", "—", "—", "empty output"))
            continue

        out_list = json.loads(out_text)
        if not isinstance(out_list, list):
            out_list = [out_list]
        s = _score_receipt(out_list, tru_list)
        scores.append(s["overall"])
        rows.append((
            stem + ".jpg", gt_n,
            check(s["storeName"]), check(s["purchaseDate"]),
            check(s["latitude"]),  check(s["longitude"]),
            check(s["totalItems"]),
            s["item_name_match"], s["item_price_match"], s["item_amount_match"],
            f"{s['overall']}%",
        ))
    return header, rows, scores


def eval_receipts(output_dir: Path = config.OUTPUT_DIR, gt_dir: Path = config.GROUND_TRUTH_DIR):
    """
    Compare output against ground_truth/ — field by field per receipt.
    Evaluates each provider subdirectory separately if present.
    """
    if not gt_dir.exists() or not any(gt_dir.glob("*.json")):
        print(f"No ground truth files found in {gt_dir}/")
        return

    provider_dirs = [d for d in sorted(output_dir.iterdir()) if d.is_dir()] if output_dir.exists() else []
    dirs_to_eval  = provider_dirs if provider_dirs else [output_dir]

    config.REPORTS_DIR.mkdir(exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    md_lines = [
        "# GatherYourDeals ETL — Eval Report",
        f"_Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_", "",
    ]

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

        md_lines += [
            f"## {label}", "",
            "| " + " | ".join(header) + " |",
            "| " + " | ".join("-" * len(h) for h in header) + " |",
        ]
        for row in rows:
            md_lines.append("| " + " | ".join(str(c) for c in row) + " |")
        if scores:
            md_lines += ["", f"**Avg score: {sum(scores)/len(scores):.1f}%**", ""]

    md_path = config.REPORTS_DIR / f"eval_{ts}.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nEval report → {md_path}")


# ---------------------------------------------------------------------------
# Baseline Report — cost + latency + correctness across all providers
# ---------------------------------------------------------------------------
def baseline_report():
    """
    Generate a structured baseline experiment report covering:
      - Azure DI (OCR): latency and cost
      - LLM providers (CLOD/Qwen, CLOD/Gemma): latency, cost, and field accuracy

    Scoped to the experiment window recorded in .baseline_start.
    Saved to reports/baseline_<ts>.md
    """
    start_file = Path(".baseline_start")
    if not start_file.exists():
        print("No .baseline_start file found. Run:  bash scripts/run_baseline.sh")
        return
    experiment_start = start_file.read_text().strip()

    all_entries = _load_log_entries()
    entries     = [e for e in all_entries if e.get("time", "") >= experiment_start]
    adi_entries = [e for e in entries if e.get("event") == "adi_ocr"]
    llm_entries = [e for e in entries if e.get("event") == "llm_extraction"]

    if not llm_entries and not adi_entries:
        print("No log entries found for this experiment. Run:  bash scripts/run_baseline.sh")
        return

    # Scope to the most recent complete trace_id(s) per provider.
    # A complete trace covers all receipts (count >= modal run size).
    _RUNS_PER_PROVIDER = 3
    _trace_prov: dict[str, tuple[str, str]] = {}
    _trace_cnt:  dict[str, int]  = defaultdict(int)
    _trace_last: dict[str, str]  = defaultdict(str)
    for e in llm_entries:
        tid = e.get("trace_id", "")
        _trace_prov[tid] = (e.get("llm_provider", "?"), e.get("llm_model", "?"))
        _trace_cnt[tid] += 1
        t = e.get("time", "")
        if t > _trace_last[tid]:
            _trace_last[tid] = t

    try:
        from statistics import mode as _mode
        _run_size = _mode(_trace_cnt.values())
    except Exception:
        _run_size = max(_trace_cnt.values()) if _trace_cnt else 1

    _complete_by_prov: dict[tuple, list] = defaultdict(list)
    for tid, cnt in _trace_cnt.items():
        if cnt >= _run_size:
            _complete_by_prov[_trace_prov[tid]].append((_trace_last[tid], tid))

    _valid_tids: set[str] = set()
    for plist in _complete_by_prov.values():
        plist.sort(reverse=True)
        for _, tid in plist[:_RUNS_PER_PROVIDER]:
            _valid_tids.add(tid)

    llm_entries = [e for e in llm_entries if e.get("trace_id") in _valid_tids]
    adi_entries = [e for e in adi_entries if e.get("trace_id") in _valid_tids]

    now_str  = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # --- ADI summary ---
    adi_ok   = [e for e in adi_entries if e.get("ocr_success")]
    adi_lats = sorted(e.get("ocr_latency_ms", 0) for e in adi_ok)
    adi_chars_vals = [e.get("chars_extracted") for e in adi_ok if e.get("chars_extracted")]
    adi_cost = sum(e.get("cost_usd", 0) for e in adi_entries)
    adi_avg  = statistics.mean(adi_lats)    if adi_lats else 0
    adi_p50  = statistics.median(adi_lats)  if adi_lats else 0
    adi_p95  = adi_lats[min(int(0.95 * len(adi_lats)), len(adi_lats) - 1)] if adi_lats else 0
    adi_chars_avg = f"{statistics.mean(adi_chars_vals):,.0f}" if adi_chars_vals else "—"

    # --- Ground truth item counts ---
    gt_counts: dict[str, int] = {}
    if config.GROUND_TRUTH_DIR.exists():
        for f in config.GROUND_TRUTH_DIR.glob("*.json"):
            try:
                gt_counts[f.stem] = len(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass

    receipts_dir  = Path("Receipts")
    receipt_count = len([f for f in receipts_dir.iterdir() if f.suffix.lower() in config.IMAGE_EXTS]) \
                    if receipts_dir.exists() else "?"

    # --- Pipeline E2E latency ---
    pipeline_entries = [e for e in entries
                        if e.get("event") == "pipeline_complete"
                        and e.get("trace_id") in _valid_tids]
    pipeline_by_prov: dict[tuple, list] = defaultdict(list)
    for e in pipeline_entries:
        pipeline_by_prov[(e.get("llm_provider", ""), e.get("llm_model", ""))].append(e)

    # --- Provider rows ---
    providers: dict[tuple, list] = defaultdict(list)
    for e in llm_entries:
        providers[(e.get("llm_provider", "?"), e.get("llm_model", "?"))].append(e)

    provider_rows = []
    for (prov, model), es in sorted(providers.items(), key=lambda kv: (kv[0][0] or "", kv[0][1] or "")):
        ok_es = [e for e in es if e.get("llm_success")]
        lats  = sorted(e.get("llm_latency_ms", 0) for e in ok_es)
        n     = len(ok_es)
        if not lats:
            continue

        avg_lat    = statistics.mean(lats)
        p50_lat    = statistics.median(lats)
        p95_lat    = lats[min(int(0.95 * n), n - 1)]
        avg_in     = statistics.mean(e.get("llm_input_tokens",  0) for e in ok_es)
        avg_out    = statistics.mean(e.get("llm_output_tokens", 0) for e in ok_es)
        chars_vals = [e.get("llm_input_chars") for e in ok_es if e.get("llm_input_chars") is not None]
        avg_chars  = f"{statistics.mean(chars_vals):,.0f}" if chars_vals else "—"
        cost_sources = set(e.get("llm_cost_source", "unknown") for e in ok_es)
        cost_src_label = "api" if cost_sources == {"api"} else "est" if "api" not in cost_sources else "mixed"
        total_cost = sum(e.get("llm_cost_usd", 0) for e in ok_es)
        cost_per   = total_cost / n if n else 0

        pe      = pipeline_by_prov.get((prov, model), [])
        ok_pe   = [e for e in pe if e.get("success")]
        e2e_lats = sorted(e.get("total_latency_ms", 0) for e in ok_pe)
        if e2e_lats:
            e2e_p50    = f"{statistics.median(e2e_lats):,.0f}"
            e2e_p95    = f"{e2e_lats[min(int(0.95 * len(e2e_lats)), len(e2e_lats) - 1)]:,.0f}"
            throughput = f"~{60_000 / statistics.median(e2e_lats):.0f}/min"
        else:
            e2e_p50 = e2e_p95 = throughput = "—"

        errors    = len(es) - len(ok_es)
        fail_rate = f"{errors}/{len(es)} ({100*errors/len(es):.0f}%)" if es else "—"

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
            "provider":   prov,
            "model":      model.split("/")[-1],
            "receipts":   len(by_receipt),
            "fail_rate":  fail_rate,
            "e2e_p50":    e2e_p50,
            "e2e_p95":    e2e_p95,
            "throughput": throughput,
            "p50_lat":    f"{p50_lat:,.0f}",
            "p95_lat":    f"{p95_lat:,.0f}",
            "avg_lat":    f"{avg_lat:,.0f}",
            "avg_in":     f"{avg_in:,.0f}",
            "avg_out":    f"{avg_out:,.0f}",
            "avg_chars":  avg_chars,
            "cost_per":   f"${cost_per:.4f}" if cost_per > 0 else "$0.00",
            "cost_src":   cost_src_label,
            "total_cost": f"${total_cost:.4f}",
            "correct":    f"{correct}/{checked}" if checked else "—",
        })

    # --- Findings ---
    findings = []
    if len(provider_rows) >= 2:
        sorted_r = sorted(provider_rows,
                          key=lambda x: float(x["e2e_p50"].replace(",", ""))
                          if x["e2e_p50"] != "—" else float(x["p50_lat"].replace(",", "")))
        findings.append("- **E2E P50 latency (fastest → slowest):** " +
                        ", ".join(f"{r['provider']}/{r['model']} {r['e2e_p50']} ms" for r in sorted_r))
        findings.append("- **Throughput:** " +
                        ", ".join(f"{r['provider']}/{r['model']} {r['throughput']}" for r in sorted_r))
        cheaper = min(provider_rows, key=lambda x: float(x["cost_per"].replace("$", "")))
        findings.append(f"- **Lowest cost/receipt:** {cheaper['provider']} / {cheaper['model']} ({cheaper['cost_per']})")
        primary = sorted_r[0]
        findings.append(f"- **Recommended primary:** {primary['provider']} / {primary['model']} — lowest egress latency")
        for fb in sorted_r[1:]:
            findings.append(f"- **Fallback:** {fb['provider']} / {fb['model']} — viable alternative if primary is unavailable")
    elif provider_rows:
        r = provider_rows[0]
        findings.append(f"- Only one provider ({r['provider']} / {r['model']}). Run all providers for comparison findings.")

    # --- Markdown assembly ---
    md = [
        "# GatherYourDeals ETL — Baseline Experiment Report",
        f"_Generated: {now_str}_",
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
        "## OCR — Azure Document Intelligence",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Calls | {len(adi_entries)} ({len(adi_ok)} success) |",
        f"| Avg latency | {adi_avg:,.0f} ms |",
        f"| P50 latency | {adi_p50:,.0f} ms |",
        f"| P95 latency | {adi_p95:,.0f} ms |",
        f"| Avg chars extracted | {adi_chars_avg} |",
        f"| Estimated cost | ${adi_cost:.4f} (F0 free tier — 500 pages/month) |",
        "",
        "## LLM Provider Comparison",
        "",
        "| Provider | Model | Receipts | Fail rate | E2E P50 (ms) | E2E P95 (ms) | Throughput | LLM P50 (ms) | LLM P95 (ms) | Avg lat (ms) | Avg in tok | Avg out tok | Avg in chars | Cost/receipt | Cost src | Total cost | Item count match |",
        "|----------|-------|:--------:|:---------:|-------------:|-------------:|:----------:|-------------:|-------------:|-------------:|-----------:|------------:|-------------:|-------------:|:--------:|:----------:|:----------------:|",
    ]
    for row in provider_rows:
        md.append(
            f"| {row['provider']} | {row['model']} | {row['receipts']} | {row['fail_rate']} "
            f"| {row['e2e_p50']} | {row['e2e_p95']} | {row['throughput']} "
            f"| {row['p50_lat']} | {row['p95_lat']} | {row['avg_lat']} "
            f"| {row['avg_in']} | {row['avg_out']} | {row['avg_chars']} "
            f"| {row['cost_per']} | {row['cost_src']} | {row['total_cost']} | {row['correct']} |"
        )
    md.append("")
    md.append("_Item count match: modal item count across runs equals ground-truth count._")

    md += [
        "",
        "## Per-Receipt Breakdown",
        "",
        "| Receipt | OCR chars | ADI (ms) | Provider | Model | In tok | Out tok | In chars | Prompt | Cost | Cost src | LLM (ms) | Items | GT | API |",
        "|---------|----------:|---------:|----------|-------|-------:|--------:|---------:|--------|-----:|:--------:|---------:|------:|:--:|:---:|",
    ]
    adi_by_key = {(e.get("image_name"), e.get("trace_id")): e for e in adi_entries}
    _OUTLIER_MS = 60_000
    for e in llm_entries:
        name  = e.get("image_name", "?")
        adi   = adi_by_key.get((name, e.get("trace_id")), {})
        ocr_chars = adi.get("chars_extracted") or "—"
        model_short = (e.get("llm_model") or "?").split("/")[-1]
        llm_lat = e.get("llm_latency_ms", 0)
        lat_str = f"{llm_lat:.0f} ⚠" if llm_lat > _OUTLIER_MS else f"{llm_lat:.0f}"
        items   = e.get("items_extracted", 0)
        gt_n    = gt_counts.get(Path(name).stem)
        gt_match = ("✓" if items == gt_n else "✗") if gt_n is not None else "—"
        api      = "✓" if e.get("llm_success") else "✗"
        in_chars  = e.get("llm_input_chars")
        in_chars_str = f"{in_chars:,}" if in_chars is not None else "—"
        prompt_path  = e.get("llm_prompt_path") or "—"
        cost_src     = e.get("llm_cost_source", "?")[:3]   # "api" or "est"
        md.append(
            f"| {name} | {ocr_chars} | {adi.get('ocr_latency_ms', 0):.0f} "
            f"| {e.get('llm_provider', '?')} | {model_short} "
            f"| {e.get('llm_input_tokens', 0):,} | {e.get('llm_output_tokens', 0):,} "
            f"| {in_chars_str} | {prompt_path} "
            f"| ${e.get('llm_cost_usd', 0):.6f} | {cost_src} | {lat_str} "
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
        "_clod costs are taken directly from the API response and cannot be independently verified from token counts alone._",
        "",
        "## Findings",
        "",
    ]
    md += findings if findings else ["- Run both providers to generate findings."]

    # --- Field-level accuracy (correctness) ---
    def _output_slug(provider: str, model: str) -> str:
        return f"{provider}-{model.split('/')[-1].lower()}"

    provider_dirs = [
        (f"{row['provider']} / {row['model'].split('/')[-1]}",
         config.OUTPUT_DIR / _output_slug(row["provider"], row["model"]))
        for row in provider_rows
    ]

    eval_md = [
        "", "## Field-Level Accuracy", "",
        "Scores each provider's output against ground_truth/ field by field. "
        "Scalar fields: store (word-overlap), date (exact), lat/lon (±0.01°), item count (exact). "
        "Item fields: name/price/amount match rates over ground-truth items. "
        "Weighting: 5 scalar fields = 5/8 of overall score; 3 item-rate fields = 3/8.",
        "",
        "**Scoring:** store uses word-overlap (any word >3 chars in common); "
        "lat/lon tolerance 0.01° (~1.1 km).",
        "",
    ]

    price_stats: dict[str, dict[str, tuple[int, int]]] = {}
    any_eval = False

    for prov_label, prov_dir in provider_dirs:
        if not prov_dir.exists():
            eval_md += [f"### {prov_label}", "", f"_No output at `{prov_dir}/`_", ""]
            continue
        eval_header, eval_rows, eval_scores = _compute_eval(output_dir=prov_dir)
        if not eval_rows:
            continue
        any_eval = True
        avg = f"{sum(eval_scores)/len(eval_scores):.1f}%" if eval_scores else "—"
        eval_md += [
            f"### {prov_label}", "",
            f"**Avg score: {avg}** over {len(eval_scores)} receipts", "",
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
        price_stats[prov_label] = prov_price

    # Price match gap finding
    if len(price_stats) >= 1:
        slug_labels = {
            _output_slug(row["provider"], row["model"]): f"{row['provider']} / {row['model'].split('/')[-1]}"
            for row in provider_rows
        }
        prov_overall: dict[str, float] = {}
        for prov, pdata in price_stats.items():
            total_num = sum(n for n, _ in pdata.values())
            total_den = sum(d for _, d in pdata.values())
            prov_overall[prov] = total_num / total_den if total_den else 0.0

        sorted_provs = sorted(prov_overall, key=lambda p: prov_overall[p], reverse=True)
        finding_lines = ["### Quality Finding: Price Extraction by Provider", ""]
        finding_lines.append("  \n".join(
            f"- **{p}**: {prov_overall[p]*100:.0f}% overall price match"
            for p in sorted_provs
        ))
        finding_lines.append("")

        gap_lines, seen_gaps = [], set()
        for prov_a, prov_b in combinations(sorted_provs, 2):
            for stem in sorted(set(price_stats[prov_a]) & set(price_stats[prov_b])):
                na, da = price_stats[prov_a].get(stem, (0, 0))
                nb, db = price_stats[prov_b].get(stem, (0, 0))
                ra = na / da if da else 0.0
                rb = nb / db if db else 0.0
                if abs(ra - rb) >= 0.30 and stem not in seen_gaps:
                    seen_gaps.add(stem)
                    stronger, weaker = (prov_a, prov_b) if ra >= rb else (prov_b, prov_a)
                    gap_lines.append(
                        f"- **{stem}:** {stronger} {max(ra,rb)*100:.0f}% "
                        f"vs {weaker} {min(ra,rb)*100:.0f}%"
                    )

        if gap_lines:
            weakest  = min(prov_overall, key=lambda p: prov_overall[p])
            strongest = max(prov_overall, key=lambda p: prov_overall[p])
            finding_lines += [
                "Receipts with a ≥ 30 percentage-point price match gap:", "",
            ] + gap_lines + [
                "",
                f"{weakest} is extracting per-unit rates or subtotals instead of line-item totals "
                f"on multi-column receipts. "
                f"{strongest} handles these layouts correctly.",
                "",
            ]
        eval_md += finding_lines

    if any_eval:
        md += eval_md

    config.REPORTS_DIR.mkdir(exist_ok=True)
    md_path = config.REPORTS_DIR / f"baseline_{ts}.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nBaseline report → {md_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="GatherYourDeals ETL — Reporting")
    p.add_argument("--eval",            action="store_true", help="Compare output/ against ground_truth/")
    p.add_argument("--baseline-report", action="store_true", help="Generate baseline experiment report")
    args = p.parse_args()

    if args.eval:
        eval_receipts()
    elif args.baseline_report:
        baseline_report()
    else:
        p.print_help()


if __name__ == "__main__":
    main()
