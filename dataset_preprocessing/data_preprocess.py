#!/usr/bin/env python3
"""Engineer a training dataset from FoodData Central exports.

Inputs:
  - JSON exported by fetch_foods_100.py (list of food dicts)
  - or the CSV exported by fetch_foods_100.py

Outputs a numeric-feature CSV suitable for training.

Defaults follow the spec in the assignment prompt:
  - drops identifiers/text fields
  - keeps core nutrients
  - adds engineered ratios/densities
  - adds one creative feature: processed_level (0/1/2)
  - adds required label: health_label (Healthy/Moderate/Unhealthy)

Labeling:
  - Default is quantile-based (tertiles) using a composite health_score.
    This avoids hard-coding nutrition thresholds and ensures all 3 classes appear.
  - You can switch to threshold mode and provide thresholds if desired.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


CORE_FEATURES: Tuple[str, ...] = (
    "protein_g",
    "fat_g",
    "carbs_g",
    "energy_kcal",
    "sugars_g",
    "fiber_g",
    "sodium_mg",
    "potassium_mg",
    "cholesterol_mg",
)


NUTRIENT_NUMBER_TO_COL: Dict[str, str] = {
    "203": "protein_g",
    "204": "fat_g",
    "205": "carbs_g",
    "208": "energy_kcal",
    "269": "sugars_g",
    "291": "fiber_g",
    "306": "potassium_mg",
    "307": "sodium_mg",
    "601": "cholesterol_mg",
}


def _safe_float(val: object) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if isinstance(val, bool):
            return None
        out = float(val)
        if math.isnan(out):
            return None
        return out
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        try:
            out = float(s)
            if math.isnan(out):
                return None
            return out
        except ValueError:
            return None
    return None


def _div(numer: Optional[float], denom: Optional[float]) -> Optional[float]:
    if numer is None or denom is None:
        return None
    if denom == 0:
        return None
    return numer / denom


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _extract_nutrients_from_json_food(food: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Extract nutrient amounts from a detailed FDC food JSON object."""
    out: Dict[str, Optional[float]] = {col: None for col in NUTRIENT_NUMBER_TO_COL.values()}

    nutrients = food.get("foodNutrients")
    if not isinstance(nutrients, list):
        return out

    for item in nutrients:
        if not isinstance(item, dict):
            continue

        # Common detailed JSON format:
        #   {"nutrient": {"number": "203"}, "amount": 12.3, ...}
        nutrient = item.get("nutrient")
        amount = item.get("amount")
        if isinstance(nutrient, dict):
            num = nutrient.get("number")
            if num is not None:
                col = NUTRIENT_NUMBER_TO_COL.get(str(num))
                if col:
                    out[col] = _safe_float(amount)
                continue

        # Some foods (like FNDDS) include:
        #   {"number": "203", "amount": 12.3, ...}
        num2 = item.get("number")
        if num2 is not None:
            col2 = NUTRIENT_NUMBER_TO_COL.get(str(num2))
            if col2:
                out[col2] = _safe_float(item.get("amount"))

    return out


_ULTRA_PROCESSED_HINTS = re.compile(
    r"\b(artificial|flavor|colour|color|preservative|emulsifier|stabilizer|stabiliser|"
    r"maltodextrin|dextrose|high\s*fructose|corn\s*syrup|hydrogenated|modified\s*starch|"
    r"monosodium\s*glutamate|msg|sucralose|acesulfame|aspartame|sodium\s*benzoate|"
    r"potassium\s*sorbate|calcium\s*propionate)\b",
    flags=re.IGNORECASE,
)


def processed_level(description: Optional[str], ingredients: Optional[str], data_type: Optional[str]) -> int:
    """Heuristic creative feature: 0 whole, 1 processed, 2 ultra-processed."""
    desc = (description or "").strip()
    ingr = (ingredients or "").strip()
    dtype = (data_type or "").strip()

    text = " ".join([desc, ingr, dtype])

    # If ingredients list exists, it's likely at least processed.
    if ingr:
        if _ULTRA_PROCESSED_HINTS.search(text):
            return 2
        # Long ingredient lists are often more processed.
        comma_count = ingr.count(",")
        if comma_count >= 8:
            return 2
        return 1

    # Branded foods with no ingredients captured: still usually processed.
    if dtype.lower() == "branded":
        return 1

    return 0


def _compute_engineered(row: Dict[str, Any]) -> None:
    protein = _safe_float(row.get("protein_g"))
    fat = _safe_float(row.get("fat_g"))
    carbs = _safe_float(row.get("carbs_g"))
    energy = _safe_float(row.get("energy_kcal"))
    sugars = _safe_float(row.get("sugars_g"))
    fiber = _safe_float(row.get("fiber_g"))
    sodium = _safe_float(row.get("sodium_mg"))

    row["protein_ratio"] = _div(protein, energy)
    row["fat_ratio"] = _div(fat, energy)
    row["carb_ratio"] = _div(carbs, energy)
    row["sugar_ratio"] = _div(sugars, carbs)
    row["fiber_density"] = _div(fiber, carbs)
    row["sodium_density"] = _div(sodium, energy)


def _fill_missing_numeric_with_zero(row: Dict[str, Any], cols: Iterable[str]) -> None:
    for c in cols:
        v = row.get(c)
        if v is None:
            row[c] = 0.0
            continue
        if isinstance(v, float) and math.isnan(v):
            row[c] = 0.0


def _health_score(row: Dict[str, Any]) -> Optional[float]:
    """Composite score used for quantile labeling.

    Higher is healthier (more protein/fiber density, less sugar/sodium/fat per kcal).
    """

    pr = _safe_float(row.get("protein_ratio"))
    fr = _safe_float(row.get("fat_ratio"))
    sr = _safe_float(row.get("sugar_ratio"))
    fd = _safe_float(row.get("fiber_density"))
    sd = _safe_float(row.get("sodium_density"))
    chol = _safe_float(row.get("cholesterol_mg"))

    if pr is None or fr is None or sr is None or fd is None or sd is None:
        return None

    chol_penalty = 0.0
    if chol is not None:
        # scale: 0..300mg -> 0..0.3
        chol_penalty = _clamp(chol / 1000.0, 0.0, 1.0)

    # Clamp ratios to avoid extreme outliers dominating.
    pr_c = _clamp(pr, 0.0, 1.0)
    fr_c = _clamp(fr, 0.0, 1.0)
    sr_c = _clamp(sr, 0.0, 5.0)
    fd_c = _clamp(fd, 0.0, 2.0)
    sd_c = _clamp(sd, 0.0, 50.0)

    return (2.0 * pr_c + 1.5 * fd_c) - (1.2 * fr_c + 1.3 * sr_c + 0.3 * sd_c + 0.5 * chol_penalty)


def _quantile_thresholds(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("No values for quantile")
    s = sorted(values)
    if q <= 0:
        return s[0]
    if q >= 1:
        return s[-1]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def label_quantiles(rows: List[Dict[str, Any]]) -> None:
    scores: List[float] = []
    for r in rows:
        sc = _health_score(r)
        if sc is None:
            continue
        scores.append(sc)

    if len(scores) < 3:
        # Fallback: mark all as Moderate.
        for r in rows:
            r["health_label"] = "Moderate"
        return

    t1 = _quantile_thresholds(scores, 1 / 3)
    t2 = _quantile_thresholds(scores, 2 / 3)

    for r in rows:
        sc = _health_score(r)
        if sc is None:
            r["health_label"] = "Moderate"
            continue
        if sc <= t1:
            r["health_label"] = "Unhealthy"
        elif sc <= t2:
            r["health_label"] = "Moderate"
        else:
            r["health_label"] = "Healthy"


def _normalize_per_100g(row: Dict[str, Any], nutrient_cols: Iterable[str]) -> None:
    serving_size = _safe_float(row.get("servingSize"))
    unit = (row.get("servingSizeUnit") or "").strip().lower()
    if serving_size is None or serving_size <= 0:
        return
    if unit != "g":
        return

    factor = 100.0 / serving_size
    for col in nutrient_cols:
        v = _safe_float(row.get(col))
        if v is None:
            continue
        row[col] = v * factor


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _read_json_foods(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("JSON input must be a list")
    return [d for d in data if isinstance(d, dict)]


def _from_json_to_rows(foods: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for food in foods:
        row: Dict[str, Any] = {
            "fdcId": food.get("fdcId"),
            "description": food.get("description"),
            "dataType": food.get("dataType"),
            "publicationDate": food.get("publicationDate"),
            "brandOwner": food.get("brandOwner"),
            "brandName": food.get("brandName"),
            "ingredients": food.get("ingredients"),
            "servingSize": food.get("servingSize"),
            "servingSizeUnit": food.get("servingSizeUnit"),
            "householdServingFullText": food.get("householdServingFullText"),
        }
        row.update(_extract_nutrients_from_json_food(food))
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description="Engineer an ML-ready dataset from FDC foods JSON/CSV")
    p.add_argument("--in", dest="in_path", required=True, help="Input path (.json or .csv)")
    p.add_argument("--out", dest="out_path", required=True, help="Output CSV path")
    p.add_argument(
        "--include-description",
        dest="include_description",
        action="store_true",
        default=True,
        help="Include the food description column in output (default: on; exclude from training features)",
    )
    p.add_argument(
        "--no-include-description",
        dest="include_description",
        action="store_false",
        help="Do not include description column",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If provided, output at most N rows (after dropping missing core)",
    )
    p.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle rows before applying --limit",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed used when --shuffle is set (default: 0)",
    )
    p.add_argument(
        "--normalize-per-100g",
        action="store_true",
        help="If servingSizeUnit is 'g', normalize nutrients to per-100g using servingSize",
    )
    p.add_argument(
        "--include-datatype",
        action="store_true",
        help="Keep dataType as a feature (otherwise dropped)",
    )
    p.add_argument(
        "--keep-debug-columns",
        action="store_true",
        help="Keep fdcId/description/ingredients columns (for debugging only; not recommended for training)",
    )
    p.add_argument(
        "--drop-missing-core",
        action="store_true",
        default=True,
        help="Drop rows missing any core nutrient features (default: on)",
    )
    p.add_argument(
        "--no-drop-missing-core",
        dest="drop_missing_core",
        action="store_false",
        help="Keep rows even if core nutrients are missing",
    )

    args = p.parse_args(argv)

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    if in_path.suffix.lower() == ".csv":
        rows = _read_csv_rows(in_path)
    elif in_path.suffix.lower() == ".json":
        foods = _read_json_foods(in_path)
        rows = _from_json_to_rows(foods)
    else:
        raise SystemExit("--in must be .json or .csv")

    # Normalize if requested.
    if args.normalize_per_100g:
        for r in rows:
            _normalize_per_100g(r, CORE_FEATURES)

    # Add engineered features + creative feature.
    out_rows: List[Dict[str, Any]] = []
    numeric_feature_cols = [
        *CORE_FEATURES,
        "protein_ratio",
        "fat_ratio",
        "carb_ratio",
        "sugar_ratio",
        "fiber_density",
        "sodium_density",
    ]
    for r in rows:
        # Ensure numeric parsing for core features
        for c in CORE_FEATURES:
            r[c] = _safe_float(r.get(c))

        if args.drop_missing_core and any(r.get(c) is None for c in CORE_FEATURES):
            continue

        _compute_engineered(r)
        _fill_missing_numeric_with_zero(r, numeric_feature_cols)
        r["processed_level"] = processed_level(
            description=r.get("description") if isinstance(r.get("description"), str) else None,
            ingredients=r.get("ingredients") if isinstance(r.get("ingredients"), str) else None,
            data_type=r.get("dataType") if isinstance(r.get("dataType"), str) else None,
        )
        out_rows.append(r)

    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be > 0")
        if args.shuffle:
            import random

            rng = random.Random(args.seed)
            rng.shuffle(out_rows)
        if len(out_rows) > args.limit:
            out_rows = out_rows[: args.limit]

    # Required label.
    label_quantiles(out_rows)

    # Build output schema.
    engineered_cols = [
        *CORE_FEATURES,
        "protein_ratio",
        "fat_ratio",
        "carb_ratio",
        "sugar_ratio",
        "fiber_density",
        "sodium_density",
        "processed_level",
        "health_label",
    ]

    if args.include_datatype:
        engineered_cols.insert(0, "dataType")

    if args.include_description:
        engineered_cols.insert(0, "description")

    if args.keep_debug_columns:
        engineered_cols = [
            "fdcId",
            "description",
            "ingredients",
            *engineered_cols,
        ]

    _write_csv(out_path, out_rows, engineered_cols)
    print(f"Wrote {len(out_rows)} engineered rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))
