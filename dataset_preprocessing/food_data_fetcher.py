#!/usr/bin/env python3
"""Fetch foods from USDA FoodData Central and save as JSON.

Uses the /foods/list endpoint (abridged format).
API key is read from, in order of precedence:
  1) --api-key CLI arg
  2) Environment variables: API_KEY, FDC_API_KEY
  3) A .env file (searched automatically; see --env-file)

Example:
    python fetch_foods_100.py --page-size 5 --out foods_5.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


FDC_BASE_URL = "https://api.nal.usda.gov/fdc/v1"


# Nutrient numbers used by FoodData Central (legacy nutrient numbers).
# These are stable identifiers and are better suited than matching on names.
COMMON_NUTRIENT_NUMBERS: Tuple[str, ...] = (
    "203",  # Protein
    "204",  # Total lipid (fat)
    "205",  # Carbohydrate, by difference
    "208",  # Energy (kcal)
    "269",  # Sugars, total including NLEA
    "291",  # Fiber, total dietary
    "306",  # Potassium, K
    "307",  # Sodium, Na
    "601",  # Cholesterol
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


def _parse_dotenv_line(line: str) -> Optional[tuple[str, str]]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if "=" not in line:
        return None
    key, value = line.split("=", 1)
    key = key.strip()
    value = value.strip().strip('"').strip("'")
    if not key:
        return None
    return key, value


def load_dotenv_file(env_path: Path) -> Dict[str, str]:
    """Parse a .env file and return key/value pairs (does not overwrite os.environ)."""
    values: Dict[str, str] = {}
    try:
        text = env_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return values

    for raw_line in text.splitlines():
        parsed = _parse_dotenv_line(raw_line)
        if not parsed:
            continue
        key, value = parsed
        values[key] = value
    return values


def find_default_env_files(script_dir: Path, cwd: Path) -> Iterable[Path]:
    # Common locations in this repo
    candidates = [
        cwd / ".env",
        cwd / "Warren_datasets" / ".env",
        script_dir / ".env",
        script_dir / "Warren_datasets" / ".env",
    ]
    seen = set()
    for p in candidates:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        yield p


def get_api_key(cli_key: Optional[str], env_file: Optional[Path]) -> str:
    if cli_key:
        return cli_key.strip()

    for env_var in ("API_KEY", "FDC_API_KEY"):
        val = os.getenv(env_var)
        if val and val.strip():
            return val.strip()

    if env_file:
        values = load_dotenv_file(env_file)
        for env_var in ("API_KEY", "FDC_API_KEY"):
            val = values.get(env_var)
            if val and val.strip():
                return val.strip()

    raise RuntimeError(
        "Missing API key. Provide --api-key, or set API_KEY/FDC_API_KEY, "
        "or point --env-file to a .env containing API_KEY."
    )


def _request_json(url: str, retries: int, backoff_seconds: float) -> object:
    req = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "Healthy_Foods_Classifier/1.0 (fetch_foods_100.py)",
        },
        method="GET",
    )

    transient_codes = {429, 500, 502, 503, 504}
    attempt = 0
    while True:
        try:
            with urlopen(req, timeout=30) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"
                body = resp.read().decode(charset, errors="replace")
            return json.loads(body)
        except HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass

            if e.code in transient_codes and attempt < retries:
                sleep_for = backoff_seconds * (2**attempt)
                attempt += 1
                print(
                    f"Transient HTTP {e.code}; retrying in {sleep_for:.1f}s ({attempt}/{retries})",
                    file=sys.stderr,
                )
                time.sleep(sleep_for)
                continue

            raise RuntimeError(f"HTTP {e.code} from FoodData Central. {detail}".strip()) from e
        except URLError as e:
            # URLError can be transient too; retry similarly.
            if attempt < retries:
                sleep_for = backoff_seconds * (2**attempt)
                attempt += 1
                print(
                    f"Network error; retrying in {sleep_for:.1f}s ({attempt}/{retries}): {e}",
                    file=sys.stderr,
                )
                time.sleep(sleep_for)
                continue
            raise RuntimeError(f"Network error calling FoodData Central: {e}") from e
        except json.JSONDecodeError as e:
            raise RuntimeError("Response was not valid JSON.") from e


def fetch_foods_list(
    api_key: str,
    page_size: int,
    page_number: int,
    retries: int,
    backoff_seconds: float,
) -> object:
    params = {
        "api_key": api_key,
        "pageSize": page_size,
        "pageNumber": page_number,
    }

    url = f"{FDC_BASE_URL}/foods/list?{urlencode(params)}"
    return _request_json(url, retries=retries, backoff_seconds=backoff_seconds)


def fetch_food_detail(
    api_key: str,
    fdc_id: int,
    retries: int,
    backoff_seconds: float,
) -> Dict[str, Any]:
    params = {"api_key": api_key}
    url = f"{FDC_BASE_URL}/food/{fdc_id}?{urlencode(params)}"
    try:
        parsed = _request_json(url, retries=retries, backoff_seconds=backoff_seconds)
    except RuntimeError as e:
        # Random ID probing will naturally hit many 404s; treat as "not found".
        msg = str(e)
        if "HTTP 400" in msg or "HTTP 404" in msg:
            raise KeyError(f"Food not found: {fdc_id}") from e
        raise

    if not isinstance(parsed, dict):
        raise RuntimeError("Unexpected food detail response type.")
    return parsed


def _nutrients_to_number_amount(food: Dict[str, Any]) -> Dict[str, float]:
    """Extract {nutrientNumber: amount} from either abridged list items or detailed food."""
    out: Dict[str, float] = {}
    nutrients = food.get("foodNutrients")
    if not isinstance(nutrients, list):
        return out

    for item in nutrients:
        if not isinstance(item, dict):
            continue

        # Abridged format (foods/list) typically uses keys like nutrientNumber/value.
        num = item.get("nutrientNumber")
        val = item.get("value")
        if num is not None and val is not None:
            try:
                out[str(num)] = float(val)
                continue
            except (TypeError, ValueError):
                pass

        # Detailed format (food/{fdcId}) commonly uses nutrient.number and amount.
        nutrient = item.get("nutrient")
        amount = item.get("amount")
        if isinstance(nutrient, dict) and amount is not None:
            nnum = nutrient.get("number")
            if nnum is None:
                continue
            try:
                out[str(nnum)] = float(amount)
            except (TypeError, ValueError):
                continue

    return out


def food_to_row(food: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a food object into a CSV-friendly dict."""
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

    nutrients = _nutrients_to_number_amount(food)
    for num, col in NUTRIENT_NUMBER_TO_COL.items():
        row[col] = nutrients.get(num)
    return row


def write_csv(foods: List[Dict[str, Any]], out_path: Path) -> None:
    rows = [food_to_row(f) for f in foods]

    # Stable column ordering for ML pipelines.
    fieldnames = [
        "fdcId",
        "description",
        "dataType",
        "publicationDate",
        "brandOwner",
        "brandName",
        "ingredients",
        "servingSize",
        "servingSizeUnit",
        "householdServingFullText",
        *[NUTRIENT_NUMBER_TO_COL[n] for n in COMMON_NUTRIENT_NUMBERS],
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _write_json_atomic(data: object, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_path.replace(out_path)


def main(argv: list[str]) -> int:
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()

    parser = argparse.ArgumentParser(
        description="Fetch foods from FoodData Central as JSON and optionally export to CSV"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="FoodData Central API key (otherwise uses API_KEY/FDC_API_KEY env vars or .env)",
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Path to a .env file that contains API_KEY (optional; auto-searches common locations)",
    )
    parser.add_argument("--out", default="foods_5.json", help="Output JSON file path")
    parser.add_argument(
        "--csv-out",
        default=None,
        help="Optional CSV output path (flat columns for ML training)",
    )
    parser.add_argument(
        "--print-nutrients",
        dest="print_nutrients",
        action="store_true",
        default=None,
        help="Print each food's full foodNutrients payload to stdout",
    )
    parser.add_argument(
        "--no-print-nutrients",
        dest="print_nutrients",
        action="store_false",
        default=None,
        help="Disable printing foodNutrients to stdout",
    )
    parser.add_argument(
        "--mode",
        choices=("list", "random"),
        default="random",
        help="Fetch mode: 'list' uses /foods/list; 'random' probes random FDC IDs via /food/{fdcId}",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=4,
        help="Number of retries for transient HTTP/network errors",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=1.0,
        help="Initial backoff seconds for retries (exponential)",
    )
    parser.add_argument("--page-size", type=int, default=300, help="Number of foods to fetch")
    parser.add_argument("--page-number", type=int, default=1, help="Page number to fetch (list mode)")
    parser.add_argument(
        "--random-id-min",
        type=int,
        default=1,
        help="Minimum FDC ID to try (random mode)",
    )
    parser.add_argument(
        "--random-id-max",
        type=int,
        default=3000000,
        help="Maximum FDC ID to try (random mode)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Max random ID attempts to collect page-size foods (random mode); default scales with --page-size",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Write --out periodically every N collected foods (random mode)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume collection from an existing --out JSON file (random mode)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible random sampling (random mode)",
    )

    args = parser.parse_args(argv)

    env_file: Optional[Path]
    if args.env_file:
        env_file = Path(args.env_file)
    else:
        env_file = None
        for candidate in find_default_env_files(script_dir, cwd):
            if candidate.exists():
                env_file = candidate
                break

    api_key = get_api_key(args.api_key, env_file)

    # Never print full key; show only last 4.
    masked = f"***{api_key[-4:]}" if len(api_key) >= 4 else "***"
    if env_file:
        print(f"Using API key {masked} (env file: {env_file})")
    else:
        print(f"Using API key {masked} (from CLI/env)")

    if args.page_size <= 0:
        raise SystemExit("--page-size must be >= 1")
    if args.mode == "list" and args.page_size > 200:
        # FDC typically allows <=200 for list endpoints.
        raise SystemExit("--page-size must be between 1 and 200 in list mode")
    if args.retries < 0:
        raise SystemExit("--retries must be >= 0")
    if args.retry_backoff <= 0:
        raise SystemExit("--retry-backoff must be > 0")

    # Default nutrient printing: on for small runs, off for large runs.
    if args.print_nutrients is None:
        args.print_nutrients = args.page_size <= 10

    # Default max-attempts: scale with desired sample size.
    if args.max_attempts is None:
        # Random probing hits many missing IDs; pick a conservative default.
        args.max_attempts = max(2000, args.page_size * 200)

    if args.checkpoint_every <= 0:
        raise SystemExit("--checkpoint-every must be > 0")

    foods: List[Dict[str, Any]]
    if args.mode == "list":
        data = fetch_foods_list(
            api_key=api_key,
            page_size=args.page_size,
            page_number=args.page_number,
            retries=args.retries,
            backoff_seconds=args.retry_backoff,
        )
        if not isinstance(data, list):
            raise RuntimeError("Expected /foods/list response to be a JSON list")
        foods = [d for d in data if isinstance(d, dict)]
    else:
        if args.random_id_min >= args.random_id_max:
            raise SystemExit("--random-id-min must be < --random-id-max")
        if args.max_attempts <= 0:
            raise SystemExit("--max-attempts must be > 0")
        rng = random.Random(args.seed)

        out_path = Path(args.out)
        foods = []
        seen_fdc_ids: set[int] = set()
        attempts = 0

        if args.resume and out_path.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
            except Exception as e:
                raise RuntimeError(f"Failed to read resume file: {out_path}: {e}") from e

            if isinstance(existing, list):
                foods = [d for d in existing if isinstance(d, dict)]
                for f in foods:
                    fid = f.get("fdcId")
                    if isinstance(fid, int):
                        seen_fdc_ids.add(fid)
                if foods:
                    print(f"Resuming from {out_path}: {len(foods)} foods already collected")

        tried_candidates: set[int] = set()
        while len(foods) < args.page_size and attempts < args.max_attempts:
            attempts += 1
            candidate = rng.randint(args.random_id_min, args.random_id_max)
            if candidate in tried_candidates:
                continue
            tried_candidates.add(candidate)

            try:
                food = fetch_food_detail(
                    api_key=api_key,
                    fdc_id=candidate,
                    retries=args.retries,
                    backoff_seconds=args.retry_backoff,
                )
            except KeyError:
                continue

            fdc_id = food.get("fdcId")
            if isinstance(fdc_id, int) and fdc_id in seen_fdc_ids:
                continue
            if isinstance(fdc_id, int):
                seen_fdc_ids.add(fdc_id)
            foods.append(food)

            if len(foods) % 10 == 0:
                print(f"Collected {len(foods)}/{args.page_size} foods (attempts: {attempts})")
            if len(foods) % args.checkpoint_every == 0:
                _write_json_atomic(foods, out_path)
                print(f"Checkpoint: wrote {len(foods)} foods to {out_path}")

        if len(foods) < args.page_size:
            print(
                f"Warning: only collected {len(foods)}/{args.page_size} foods after {attempts} attempts. "
                "Try increasing --max-attempts or adjusting the random ID range.",
                file=sys.stderr,
            )

    if args.print_nutrients:
        for idx, food in enumerate(foods, start=1):
            fdc_id = food.get("fdcId")
            desc = food.get("description")
            print("=" * 80)
            print(f"Food {idx}/{len(foods)}: fdcId={fdc_id} | {desc}")

            detail: Dict[str, Any]
            if isinstance(fdc_id, int):
                try:
                    detail = fetch_food_detail(
                        api_key=api_key,
                        fdc_id=fdc_id,
                        retries=args.retries,
                        backoff_seconds=args.retry_backoff,
                    )
                except KeyError:
                    detail = food
            else:
                detail = food

            nutrients = detail.get("foodNutrients", [])
            print(json.dumps(nutrients, indent=2, ensure_ascii=False))

    out_path = Path(args.out)
    _write_json_atomic(foods, out_path)
    print(f"Wrote {len(foods)} foods to {out_path}")

    if args.csv_out:
        csv_path = Path(args.csv_out)
        write_csv(foods, csv_path)
        print(f"Wrote CSV to {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
