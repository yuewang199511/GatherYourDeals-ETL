# src/core/ocr.py
import time
import re
import statistics
from pathlib import Path

from src.core import config
from src.utils import image_proc
# Also import these from etl_logger for the log_adi call
from src.etl_logger import log_adi, ADI_COST_PER_PAGE


# Identifies savings/discount lines in the spatial layout so they can be
# labeled [S] and associated with the item above them.
_SAVINGS_LINE = re.compile(
    r"\b(savings?|you\s*saved|instant\s*savings?|member\s*savings?|"
    r"everyday\s*savings?|digital\s*coupon|coupon\s*savings?|discount)\b",
    re.IGNORECASE,
)

# Spatial-layout noise filter — same as _NOISE_LINE but intentionally keeps
# savings/discount lines so the LLM can compute the final discounted price.
_SPATIAL_NOISE_LINE = re.compile(
    r"^\s*(?:"
    r"sub\s*total|subtotal|total|net\s*total|grand\s*total|"
    r"hst|gst|pst|qst|vat|tax|surcharge|"
    r"payment|cash|credit|debit|visa|mastercard|interac|amex|"
    r"us\s+debit|us\s+credit|"                          # "US DEBIT Purchase" etc.
    r"change\s*due|change\b|balance\s*due|balance\b|"   # bare CHANGE / BALANCE lines
    r"amount\s*due|amount\s*tendered|"
    r"purchase\s*:|purchase\b|"                         # "PURCHASE: 9.06"
    r"verified|pin\s*verified|"                         # "VERIFIED BY PIN"
    r"aid\s*:|tc\s*:|ref\s*#|trans\s*#|auth\s*#|approval|"  # transaction codes
    r"thank\s*you|please\s*come|visit\s*us|survey|"
    r"tell\s*us|earn\b|fuel\s*point|fuel\b|"            # loyalty program footer
    r"remaining\b.*point|total\b.*point|"               # "Remaining May Fuel Points"
    r"annual\s*card|you\s*saved|with\s*our|"            # savings summary footer
    r"go\s*to\s*www|www\.|feedback|hiring|"             # URLs / HR footer
    r"receipt\s*#|store\s*#|"
    r"approved|declined|customer\s*copy|merchant\s*copy|"
    r"crv|ca\s*redemp|deposit|bottle\s*dep|bag\s*fee|"
    r"your\s+cashier|cashier|operator|terminal|"        # "Your cashier was Jamie" etc.
    r"\*{4,}|={3,}|-{3,}|#{3,}"                        # symbol-only lines (\b removed — \W next to \W has no boundary)
    r")(?:\b|$|\s)",                                    # word boundary OR end OR whitespace
    re.IGNORECASE,
)


def _reconstruct_spatial_rows(result) -> str:
    """
    Build a column-labeled spatial layout from ADI bounding-box data.

    Hybrid approach: page.lines for row structure, page.words for intra-line
    column assignment.

    ADI's page.lines correctly groups words into visual rows (row structure),
    but treats each line as one token — so "REYNOLDS WRAP FOIL 0.60" becomes
    a single left-column blob even though "0.60" is in the center column.

    page.words gives each word its own X polygon, so we decompose each ADI
    line into its component words, assign each word to L/C/R by its individual
    X position, then concatenate same-column words within the line into one
    token.  Row grouping still uses page.lines Y positions, so adjacent lines
    ("Your cashier was Jamie" / "REYNOLDS WRAP FOIL") stay separate.

        [L] < 40%  — item description / product name
        [C] 40-70% — quantity / unit
        [R] > 70%  — price
    """
    if not result.pages:
        return ""

    page        = result.pages[0]
    page_width  = max(getattr(page, "width",  0) or 0, 1.0)
    page_height = max(getattr(page, "height", 0) or 0, 1.0)
    tolerance   = page_height * 0.015   # lines within 1.5% of page height → same row

    # ── Build word position maps ─────────────────────────────────────────────
    word_xs:    dict[str, float] = {}   # token → x_left  (for L/C/R column assignment)
    word_ys:    dict[str, float] = {}   # token → y_center (last occurrence; for tilt)
    word_count: dict[str, int]   = {}   # token → total occurrences on this page

    for word in (page.words or []):
        wpoly = getattr(word, "polygon", None)
        wtext = getattr(word, "content", None)
        if not wpoly or not wtext or not wtext.strip():
            continue
        if hasattr(wpoly[0], "x"):
            wxs = [p.x for p in wpoly]
            wys = [p.y for p in wpoly]
        else:
            wxs = list(wpoly[0::2])
            wys = list(wpoly[1::2])
        tok = wtext.strip()
        word_xs[tok]    = min(wxs)
        word_ys[tok]    = sum(wys) / len(wys)
        word_count[tok] = word_count.get(tok, 0) + 1

    # ── Tilt detection ──────────────────────────────────────────────────────
    # ADI often returns axis-aligned word bounding boxes even for tilted
    # receipts, so polygon top-edge slopes are unreliable.  Instead we measure
    # tilt from the Y variation of word centers WITHIN multi-word ADI lines.
    #
    # Pitfall: word_ys stores the LAST occurrence of each token; for repeated
    # tokens (e.g. "F" appears on every food-taxable line, "SC" on every Kroger
    # savings row) the stored Y is wrong for earlier occurrences, which
    # corrupts the slope calculation.  To avoid this, we only include slope
    # samples from lines where ALL tokens are unique across the page.
    #
    # For the Kroger test receipt the estimated tilt is ≈ −0.07 to −0.09
    # (right side ~15 px higher per 200 px horizontal span).
    # If the receipt is straight, all slopes ≈ 0 → tilt = 0 → y_corr = y.
    slope_samples: list[float] = []
    for line in (page.lines or []):
        ltext = getattr(line, "content", None)
        lpoly = getattr(line, "polygon", None)
        if not ltext or not lpoly:
            continue
        tokens = ltext.strip().split()
        if len(tokens) < 2:
            continue
        # Skip noise lines (totals, loyalty text, transaction codes) — they
        # often have different perspective distortion than the item section.
        if _SPATIAL_NOISE_LINE.match(ltext.strip()):
            continue
        # Skip lines containing any token that appears more than once on the
        # page — word_ys for such tokens reflects their last position, not this
        # line's position, which would give a wrong slope.
        if any(word_count.get(t, 0) > 1 for t in tokens):
            continue
        # Only sample from the upper 60% of the receipt — the item section.
        # Footer/ad text at the bottom of receipts often has reduced or reversed
        # perspective tilt, which would drag the median toward zero.
        if hasattr(lpoly[0], "x"):
            line_y_center = sum(p.y for p in lpoly) / len(lpoly)
        else:
            line_y_center = sum(lpoly[1::2]) / (len(lpoly) // 2)
        if line_y_center > page_height * 0.60:
            continue
        xy = [(word_xs[t], word_ys[t]) for t in tokens
              if t in word_xs and t in word_ys]
        if len(xy) < 2:
            continue
        xy.sort(key=lambda p: p[0])                     # sort left → right by X
        x_span = xy[-1][0] - xy[0][0]
        if x_span < page_width * 0.05:                  # need meaningful span (≥5%)
            continue
        slope_samples.append((xy[-1][1] - xy[0][1]) / x_span)

    tilt: float = statistics.median(slope_samples) if slope_samples else 0.0

    def _cy(x: float, y: float) -> float:
        """Return shear-corrected Y: removes the linear tilt across the page."""
        return y - tilt * x

    # ── Calculate Effective Width ───────────────────────────────────────────
    # Find the rightmost point where text actually exists.
    # This prevents narrow receipts from being 'squished' in the calculation.
    all_xs = word_xs.values()
    effective_width = max(all_xs) if all_xs else page_width

    # ── Build line_data ──────────────────────────────────────────────────────
    # Row structure: page.lines — each ADI line is one visual row.
    # Column assignment: page.words X (word_xs).
    # Y stored in line_data: raw line_y for ALL tokens (not corrected).
    #   • Left tokens  → raw line_y preserves ADI's row separation for grouping
    #   • Center/right → raw line_y; tilt correction is applied ONLY during
    #     assignment to groups (see _best_group below), not in line_data itself.
    #     Applying correction in line_data would shrink the gap between adjacent
    #     rows (e.g. REYNOLDS row ↔ SC KROGER SAVINGS row from 16 px to 8 px),
    #     dropping it below the grouping tolerance and merging them.
    #
    # Column thresholds (fraction of page width):
    #   [L] < 52%  — item description / product name.  52% rather than 40% so
    #                 that long product names whose words spill into the 40–52%
    #                 zone stay in the left column.
    #   [C] 52–70% — quantity / unit price / discount amount
    #   [R] > 70%  — right-column total price
    line_data: list[tuple[float, float, str]] = []   # (raw_line_y, x_left, text)

    for line in (page.lines or []):
        poly = getattr(line, "polygon", None)
        text = getattr(line, "content", None)
        if not poly or not text or not text.strip():
            continue
        if _SPATIAL_NOISE_LINE.match(text.strip()):
            continue
        if hasattr(poly[0], "x"):
            ys        = [p.y for p in poly]
            line_min_x = min(p.x for p in poly)
        else:
            ys        = list(poly[1::2])
            line_min_x = min(poly[0::2])
        line_y = sum(ys) / len(ys)

        left_words:   list[tuple[str, float]] = []
        center_words: list[tuple[str, float]] = []
        right_words:  list[tuple[str, float]] = []

        for token in text.strip().split():
            x   = word_xs.get(token, line_min_x)
            pct = x / effective_width
            if pct < 0.48:
                left_words.append((token, x))
            elif pct < 0.75:
                center_words.append((token, x))
            else:
                right_words.append((token, x))

        if left_words:
            line_data.append((line_y, left_words[0][1],
                              " ".join(t[0] for t in left_words)))
        for tok, x in center_words:
            line_data.append((line_y, x, tok))
        if right_words:
            line_data.append((line_y, right_words[0][1],
                              " ".join(t[0] for t in right_words)))

    line_data.sort(key=lambda t: t[0])

    # --- Pass 1: separate tokens by column, group [L] rows by Y proximity ---
    left_tokens:   list[tuple[float, float, str]] = []
    center_tokens: list[tuple[float, float, str]] = []
    right_tokens:  list[tuple[float, float, str]] = []
    for y, x, text in line_data:
        pct = x / effective_width
        if pct < 0.48:
            left_tokens.append((y, x, text))
        elif pct < 0.75:
            center_tokens.append((y, x, text))
        else:
            right_tokens.append((y, x, text))

    # Group [L] tokens into rows by Y proximity
    left_groups: list[list[tuple[float, float, str]]] = []
    for entry in left_tokens:
        if left_groups and abs(entry[0] - left_groups[-1][0][0]) <= tolerance:
            left_groups[-1].append(entry)
        else:
            left_groups.append([entry])

    if not left_groups:
        return ""

    # Representative Y for each [L] group (first token's Y, groups are sorted)
    group_ys = [g[0][0] for g in left_groups]

    # ── Tilt-corrected group assignment ─────────────────────────────────────
    # Assign center and right tokens to [L] groups by comparing tilt-corrected
    # Y values (simple nearest).
    #
    # Why tilt correction here and not in line_data?
    #   Receipt tilt shifts the price column up relative to the item-name column
    #   on the same visual row.  Left groups are built from raw line_y (which
    #   preserves ADI's original row separation), but center/right tokens need
    #   their Y adjusted before comparison so that a price at corrected Y=306
    #   lands on the item at corrected Y=309, not on the cashier header at
    #   corrected Y=267.
    #
    # Corrected Y for a token:    _cy(token_x, raw_line_y)
    # Corrected Y for a group:    _cy(avg_x_of_group, raw_y_of_first_token)
    #   where avg_x = average of stored X values across all entries in the group.
    #   This uses the group's leftmost-word X as a proxy for its horizontal
    #   center, correcting the group Y by approximately the same amount as the
    #   same-row price column.

    def _group_cy(group: list) -> float:
        """Tilt-corrected representative Y for a [L] group.

        Uses the LAST element's (x, y) so that multi-item groups (e.g.
        {KRO CREAMER, IMPR MARGRNE}) expose the bottom item's corrected Y.
        This ensures the group's corrected-Y boundary is >= all prices that
        belong to items inside the group.
        """
        last = group[-1]
        return _cy(last[1], last[0])        # _cy(x, raw_y) of the last element

    def _best_group(raw_y: float, x: float) -> int:
        """Return the index of the best [L] group for a center/right token.

        Strategy: "at or below with margin"
        - A group is a *valid candidate* if its corrected Y is not more than
          `margin` pixels *above* the token's corrected Y
          (i.e. _group_cy(group) >= token_cy - margin).
        - Among valid candidates, prefer the one with the smallest corrected Y
          (the item closest to, but at or below, the price).

        Why margin?  Tilt estimation is imperfect (±5–10 px residual error),
        so the "at or below" boundary is fuzzy.  A small margin prevents
        near-correct assignments from flipping to the wrong row.

        Fallback: if no group qualifies (e.g. token above all items), use the
        topmost group (smallest corrected Y).
        """
        corr_y = _cy(x, raw_y)
        margin = tolerance * 0.5            # ~ half an inter-row spacing
        ranked = sorted(range(len(left_groups)),
                        key=lambda i: _group_cy(left_groups[i]))
        # Filter to valid candidates
        valid = [(i, _group_cy(left_groups[i])) for i in ranked
                 if _group_cy(left_groups[i]) >= corr_y - margin]
        if valid:
            # Among valid candidates, pick the one with smallest corrected Y
            # (the item closest to and just at-or-below the price)
            return min(valid, key=lambda ic: ic[1])[0]
        # Fallback — token above all groups; assign to the topmost group
        return ranked[0]

    # --- Pass 2: assign [R] tokens to [L] groups ---
    group_rights: list[list[tuple[float, float, str]]] = [[] for _ in left_groups]
    for r_y, r_x, r_text in right_tokens:
        group_rights[_best_group(r_y, r_x)].append((r_y, r_x, r_text))

    # Preliminary center assignment — needed before the continuation merge so we
    # can detect groups with a center-column price.  On Kroger-style receipts
    # item prices land in [C] (52-70% X) with F/T suffixes, not [R] (>70% X).
    # Without this, the continuation merge sees r_tokens=[] for every item and
    # collapses the entire item section into one cascade row.
    prelim_centers: list[list[tuple[float, float, str]]] = [[] for _ in left_groups]
    for c_y, c_x, c_text in center_tokens:
        prelim_centers[_best_group(c_y, c_x)].append((c_y, c_x, c_text))

    # --- Pass 3: merge continuation lines into their parent row ---
    # When a product name wraps across two OCR lines they land in separate [L]
    # groups (Y gap > tolerance).  If the second group has no price in ANY column
    # (R or C) it is almost certainly a continuation of the line above.
    continuation_threshold = tolerance * 3   # ~4.5% of page height ≈ 2-3 line heights
    merged_groups:  list[list[tuple[float, float, str]]] = []
    merged_rights:  list[list[tuple[float, float, str]]] = []
    merged_prelim:  list[list[tuple[float, float, str]]] = []

    for i, group in enumerate(left_groups):
        r_tokens = group_rights[i]
        c_tokens = prelim_centers[i]
        has_price = bool(r_tokens or c_tokens)           # price in R or C column
        parent_has_price = bool(
            merged_rights and (merged_rights[-1] or merged_prelim[-1])
        )
        if (merged_groups
                and not has_price                               # no price on this line
                and not parent_has_price                        # parent also has no price yet
                and (group[0][0] - merged_groups[-1][0][0])    # Y gap from parent
                    <= continuation_threshold):
            # Continuation line — absorb into the previous group
            merged_groups[-1].extend(group)
        else:
            merged_groups.append(group)
            merged_rights.append(r_tokens)
            merged_prelim.append(c_tokens)

    # Re-derive group_ys from merged groups for [C] assignment below
    group_ys     = [g[0][0] for g in merged_groups]
    left_groups  = merged_groups
    group_rights = merged_rights
    # (merged_prelim is discarded — final center assignment below uses updated group_ys)

    # Assign [C] tokens to [L] groups — tilt-corrected nearest, same as Pass 2.
    group_centers: list[list[tuple[float, float, str]]] = [[] for _ in left_groups]
    for c_y, c_x, c_text in center_tokens:
        group_centers[_best_group(c_y, c_x)].append((c_y, c_x, c_text))

    # --- Render rows ---
    # Each token is labeled with its column (L/C/R/S).
    output_lines: list[str] = []
    for i, group in enumerate(left_groups):
        group_text = " ".join(t[2] for t in group)
        col_label  = "S" if _SAVINGS_LINE.search(group_text) else "L"

        if len(group) > 1 and col_label == "L":
            # Multiple [L] items landed in the same Y-band (adjacent lines printed
            # very close together).  Render each on its own row so the LLM sees
            # them as separate products, not a name + item-code pair.
            # [C] tokens are distributed to whichever [L] is nearest by Y.
            # [R] tokens (price column) go to the last [L] item in the group.
            sorted_left = sorted(group, key=lambda t: t[0])
            c_tokens    = sorted(group_centers[i], key=lambda t: t[0])
            r_tokens    = sorted(group_rights[i],  key=lambda t: t[1])
            for j, (l_y, l_x, l_text) in enumerate(sorted_left):
                parts: list[str] = [f"[L] {l_text}"]
                for c_y, c_x, c_text in c_tokens:
                    # Use tilt-corrected Y for within-group assignment so that
                    # prices that appear above their item in raw Y (due to tilt)
                    # land on the correct item rather than the one above it.
                    c_corr = _cy(c_x, c_y)
                    nearest = min(range(len(sorted_left)),
                                  key=lambda k: abs(_cy(sorted_left[k][1],
                                                        sorted_left[k][0]) - c_corr))
                    if nearest == j:
                        parts.append(f"[C] {c_text}")
                if j == len(sorted_left) - 1:
                    for _, _, r_text in r_tokens:
                        parts.append(f"[R] {r_text}")
                output_lines.append("  |  ".join(parts))
        else:
            parts = []
            lc = [(t, col_label) for t in group] + [(t, "C") for t in group_centers[i]]
            lc.sort(key=lambda tc: tc[0][1])
            for t, col in lc:
                parts.append(f"[{col}] {t[2]}")
            for r_y, _, text in sorted(group_rights[i], key=lambda t: t[1]):
                parts.append(f"[R] {text}")
            if parts:
                output_lines.append("  |  ".join(parts))

    return "\n".join(output_lines)

def AzureOCRService(image_data: "Path | bytes", display_name: str, run_id: str, user_id: str = "", use_cache: bool = True) -> str:
    """
    Send image to Azure Document Intelligence (prebuilt-read).
    Returns markdown OCR text with a spatial layout section appended.

    Accepts either a Path (read from disk) or raw bytes (in-memory).
    display_name is used for cache keys, logging, and error messages.

    When use_cache=True (default), the OCR result is written to
    ocr_cache/<stem>.txt after the first successful call and loaded from there
    on all subsequent calls — skipping the ADI network call entirely.  Use
    --no-ocr-cache to force a fresh ADI call (e.g. after changing the spatial
    reconstruction logic).
    """
    cache_stem = Path(display_name).stem if display_name else "unknown"
    if use_cache:
        cache_file = config.OCR_CACHE_DIR / (cache_stem + ".txt")
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8")

    if not config.AZURE_DI_ENDPOINT or not config.AZURE_DI_KEY:
        raise EnvironmentError(
            "AZURE_DI_ENDPOINT and AZURE_DI_KEY must be set in .env\n"
            "Create a Document Intelligence resource in the Azure portal."
        )
    try:
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential
    except ImportError:
        raise ImportError(
            "Azure SDK not installed.\n"
            "pip install azure-ai-documentintelligence"
        )

    client = DocumentIntelligenceClient(
        endpoint=config.AZURE_DI_ENDPOINT,
        credential=AzureKeyCredential(config.AZURE_DI_KEY),
    )

    image_bytes, content_type = image_proc._to_jpeg_bytes(image_data, display_name)
    image_size_bytes = len(image_bytes)
    start = time.monotonic()
    try:
        poller = client.begin_analyze_document(
            "prebuilt-read",
            body=image_bytes,
            content_type=content_type,
            output_content_format="markdown",
        )
        result = poller.result(timeout=config._ADI_TIMEOUT_S)
        latency_ms = (time.monotonic() - start) * 1000

        pages    = len(result.pages) if result.pages else 1
        # Reconstruct text from page.lines instead of result.content.
        # result.content applies ADI's own merge heuristic, which can concatenate
        # two visually separate receipt rows (e.g. "Your cashier was Jamie
        # REYNOLDS WRAP FOIL 0.60") into one line.  page.lines preserves the
        # per-line bounding boxes ADI detected, giving correct line breaks.
        line_texts: list[str] = []
        for _page in (result.pages or []):
            for _line in (_page.lines or []):
                if getattr(_line, "content", None):
                    line_texts.append(_line.content)
        markdown = "\n".join(line_texts) if line_texts else (result.content or "")

        spatial  = _reconstruct_spatial_rows(result)
        ocr_text = (
            markdown
            + "\n\n---\n## SPATIAL LAYOUT\n"
            + "Each token labeled [COL] where COL=L/C/R/S (L=item name, C=center/price, R=right-col price, S=savings/discount row).\n"
            + "Use this section to extract items — preserves column alignment.\n\n"
            + spatial
        ) if spatial else markdown

        log_adi(run_id, display_name, user_id, image_size_bytes,
                pages, pages * ADI_COST_PER_PAGE, latency_ms, True,
                chars_extracted=len(markdown))

        if use_cache:
            config.OCR_CACHE_DIR.mkdir(exist_ok=True)
            (config.OCR_CACHE_DIR / (cache_stem + ".txt")).write_text(
                ocr_text, encoding="utf-8"
            )

        return ocr_text

    except Exception as e:
        latency_ms = (time.monotonic() - start) * 1000
        log_adi(run_id, display_name, user_id, image_size_bytes,
                0, 0.0, latency_ms, False, error=str(e))
        raise



_CURRENCY_MARKERS = [
    (re.compile(r'\bCAD\b|\bCAD\$|C\$|\$CAD', re.IGNORECASE), "CAD"),
    (re.compile(r'\bGBP\b|£',                                  re.IGNORECASE), "GBP"),
    (re.compile(r'\bEUR\b|€',                                  re.IGNORECASE), "EUR"),
]


_US_STORE_OCR_RE = re.compile(
    r'\b(KROGER|INGLES|INGLE\'?S|WALMART|WAL-MART|TARGET|VONS|RALPHS|SAFEWAY'
    r'|ALBERTSONS|PUBLIX|H-?E-?B|WHOLE\s+FOODS|TRADER\s+JOE\'?S'
    r'|FARM\s*&\s*TABLE|CVS|WALGREENS|RITE\s+AID)\b',
    re.IGNORECASE,
)

def _detect_currency_from_ocr(ocr_text: str) -> str | None:
    """
    Scan OCR text for currency signals and return the currency code.

    Non-USD markers (CAD/GBP/EUR) take priority — if found, return immediately.
    Then check for known US store names in the OCR; if found, return 'USD' to
    override an LLM-hallucinated non-USD currency when the model misread the
    store name (e.g. Qwen returning 'Grocery' for Ingles).
    Returns None when no signal is found (caller falls back to store-name
    inference, then the LLM's own currency guess, then 'USD').
    """
    for pattern, code in _CURRENCY_MARKERS:
        if pattern.search(ocr_text):
            return code
    if _US_STORE_OCR_RE.search(ocr_text):
        return "USD"
    return None
