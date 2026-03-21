
# Data Fetching + Preprocessing Pipeline (USDA FoodData Central → ML Dataset)

This document explains, in detail, how the dataset is:

1) **Fetched** from the USDA **FoodData Central (FDC)** API using `food_data_fetcher.py`, and
2) **Preprocessed / engineered** into a training-ready CSV using `data_preprocess.py`.

The overall flow is:

1. Provide an FDC API key (CLI/env/.env)
2. Fetch foods into a JSON list (and optionally a flat CSV)
3. Convert raw foods into an engineered dataset:
	 - parse core nutrients
	 - add engineered ratios/densities
	 - add one creative feature (`processed_level`)
	 - add a label column (`health_label`)

---

## 0) Prerequisites

### API key

FoodData Central requires an API key.

The fetcher reads the key in this order:
1. `--api-key <KEY>` (CLI)
2. Environment variables: `API_KEY` or `FDC_API_KEY`
3. A `.env` file (optional)

Example `.env` (keep it private; do not commit):

```bash
API_KEY=YOUR_FDC_KEY_HERE
```

`.env` auto-search locations (from `food_data_fetcher.py`):
- `./.env`
- `./Warren_datasets/.env`
- `<script_dir>/.env`
- `<script_dir>/Warren_datasets/.env`

---

## 1) Data Fetching (food_data_fetcher.py)

### 1.1 What this script does

`food_data_fetcher.py` downloads foods from FoodData Central and writes:
- a JSON file (always): `--out <path>.json`
- and optionally a CSV file (flat table): `--csv-out <path>.csv`

The JSON output is a **list of food objects** (each item is a dict).

### 1.2 Endpoints used

The script supports two fetch modes:

#### Mode A — `list` (paged list)

- Endpoint: `GET /fdc/v1/foods/list`
- Returns: **abridged** food records (not always full nutrient detail)
- CLI flags:
	- `--mode list`
	- `--page-size` (must be `<= 200` in list mode; enforced)
	- `--page-number`

Example:

```bash
python3 dataset_preprocessing/food_data_fetcher.py \
	--mode list \
	--page-size 200 \
	--page-number 1 \
	--out foods_list_200.json \
	--csv-out foods_list_200.csv \
	--no-print-nutrients
```

#### Mode B — `random` (random ID probing)

- Endpoint: `GET /fdc/v1/food/{fdcId}`
- Returns: **detailed** food records (includes `foodNutrients` in the detailed format)
- CLI flags:
	- `--mode random`
	- `--page-size` (how many valid foods to collect)
	- `--random-id-min`, `--random-id-max` (random ID range)
	- `--max-attempts` (maximum random IDs to try)
	- `--resume` (continue from an existing JSON output)
	- `--checkpoint-every N` (periodically save progress)

Example (robust long run):

```bash
python3 dataset_preprocessing/food_data_fetcher.py \
	--mode random \
	--page-size 800 \
	--out foods_random_800.json \
	--csv-out foods_random_800.csv \
	--resume \
	--checkpoint-every 50 \
	--no-print-nutrients
```

**How random mode works internally**:
- It repeatedly samples a random integer `candidate = randint(random_id_min, random_id_max)`.
- It calls `fetch_food_detail()` on that candidate.
- Many candidates are not real foods and will return HTTP 400/404; those are treated as “not found” and skipped.
- It deduplicates by `fdcId` to avoid collecting the same item twice.
- It prints progress every 10 collected foods.
- It writes checkpoints using an atomic write method (`_write_json_atomic`) every `--checkpoint-every` foods.

### 1.3 Network retries + backoff

All API calls go through `_request_json()` which:
- Uses a 30 second request timeout
- Retries transient failures with exponential backoff
- Treats these HTTP codes as transient: `429, 500, 502, 503, 504`

Backoff schedule (approx): `retry_backoff * 2^attempt`.

### 1.4 Nutrients extracted for the CSV export

The script exports a “flat” CSV using nutrient **numbers** (stable IDs). These are mapped to columns:

| Nutrient number | Meaning | Output column |
|---:|---|---|
| 203 | Protein | `protein_g` |
| 204 | Total lipid (fat) | `fat_g` |
| 205 | Carbohydrate | `carbs_g` |
| 208 | Energy (kcal) | `energy_kcal` |
| 269 | Sugars | `sugars_g` |
| 291 | Fiber | `fiber_g` |
| 306 | Potassium | `potassium_mg` |
| 307 | Sodium | `sodium_mg` |
| 601 | Cholesterol | `cholesterol_mg` |

CSV is produced by:
- `food_to_row()` which extracts:
	- basic metadata fields (`fdcId`, `description`, `dataType`, brand fields, serving size, ingredients, etc.)
	- and the nutrient columns above

**Important detail**: nutrient extraction supports both formats:
- Abridged `/foods/list` items commonly provide `nutrientNumber` + `value`
- Detailed `/food/{fdcId}` items commonly provide `nutrient.number` + `amount`

### 1.5 Optional: printing nutrient payloads

If `--print-nutrients` is enabled (default for small runs), the script prints each food’s `foodNutrients` JSON to stdout.

For each food, it prefers the detailed `GET /food/{fdcId}` record (if `fdcId` is present) to show a complete nutrient payload.

---

## 2) Data Preprocessing / Feature Engineering (data_preprocess.py)

### 2.1 What this script does

`data_preprocess.py` converts raw foods (JSON or CSV) into an **ML-ready engineered dataset**.

Inputs:
- `--in <path>.json` from `food_data_fetcher.py` (list of food dicts), OR
- `--in <path>.csv` from `food_data_fetcher.py` (flat table)

Output:
- `--out <path>.csv`: engineered features + label

Typical run (shuffle + take a fixed-size subset):

```bash
python3 dataset_preprocessing/data_preprocess.py \
	--in foods_random_800.json \
	--out foods_engineered_300.csv \
	--limit 300 \
	--shuffle \
	--seed 42
```

### 2.2 Reading raw input

#### If input is JSON

- `_read_json_foods()` loads the JSON list.
- `_from_json_to_rows()` creates one row per food, copying useful fields (e.g., `description`, `ingredients`, `dataType`, serving sizes).
- Nutrients are extracted from `food['foodNutrients']` using `_extract_nutrients_from_json_food()`.

Supported detailed nutrient item formats:
- `{"nutrient": {"number": "203"}, "amount": 12.3, ...}`
- `{"number": "203", "amount": 12.3, ...}`

#### If input is CSV

- `_read_csv_rows()` reads each row into a dict.
- Nutrient columns arrive as strings; numeric parsing happens later via `_safe_float()`.

### 2.3 Optional normalization per 100g

If `--normalize-per-100g` is enabled:
- If `servingSizeUnit` is `g` and `servingSize > 0`, then each nutrient in `CORE_FEATURES` is multiplied by:

$$\text{factor} = \frac{100}{\text{servingSize}}$$

This attempts to put all foods on a comparable per-100g basis when serving size information is usable.

### 2.4 Core nutrient features kept

The preprocessing script defines `CORE_FEATURES` as:
- `protein_g`
- `fat_g`
- `carbs_g`
- `energy_kcal`
- `sugars_g`
- `fiber_g`
- `sodium_mg`
- `potassium_mg`
- `cholesterol_mg`

These are parsed with `_safe_float()`:
- returns `None` for missing, blank, or NaN
- returns a `float` for valid numeric values

### 2.5 Dropping incomplete rows (default)

By default `--drop-missing-core` is enabled.

That means a row is **discarded** if any core feature is missing:

```python
if drop_missing_core and any(r.get(c) is None for c in CORE_FEATURES):
		continue
```

You can override with `--no-drop-missing-core`.

### 2.6 Engineered features added

After core parsing, `_compute_engineered()` adds these engineered features:

1. `protein_ratio = protein_g / energy_kcal`
2. `fat_ratio = fat_g / energy_kcal`
3. `carb_ratio = carbs_g / energy_kcal`
4. `sugar_ratio = sugars_g / carbs_g`
5. `fiber_density = fiber_g / carbs_g`
6. `sodium_density = sodium_mg / energy_kcal`

Division uses `_div(numer, denom)`:
- If either side is missing (`None`) → result is `None`
- If denominator is zero → result is `None`

### 2.7 Missing numeric values are filled with 0

After engineered features are computed, the script forces numeric completeness using:

```python
_fill_missing_numeric_with_zero(row, numeric_feature_cols)
```

This fills:
- `None` → `0.0`
- `float('nan')` → `0.0`

This is specifically important because engineered ratios can become missing (e.g., `energy_kcal == 0`).

### 2.8 Creative feature: processed_level

`processed_level` is a heuristic integer feature in `{0, 1, 2}`:

- `0` = whole / minimally processed
- `1` = processed
- `2` = ultra-processed

Logic summary (`processed_level(description, ingredients, data_type)`):
- If an ingredient list exists:
	- If it matches “ultra-processed hints” (regex keywords like artificial flavor, preservatives, maltodextrin, sucralose, MSG, etc.) → `2`
	- If ingredient list is long (>= 8 commas) → `2`
	- Otherwise → `1`
- If no ingredients but `dataType == 'branded'` → `1`
- Otherwise → `0`

### 2.9 Label creation: health_label (quantile-based)

The label is computed by `label_quantiles(out_rows)` using a composite `health_score`.

#### Health score

`_health_score(row)` uses the engineered ratios and adds a small cholesterol penalty:

- It clamps ratios to reduce extreme outlier influence.
- It returns `None` if required ratios are missing.

Score formula (as implemented):

```text
score = (2.0*protein_ratio + 1.5*fiber_density)
				- (1.2*fat_ratio + 1.3*sugar_ratio + 0.3*sodium_density + 0.5*chol_penalty)
```

Where:
- `chol_penalty = clamp(cholesterol_mg / 1000, 0, 1)`

#### Quantile thresholds and class assignment

The script computes tertile thresholds over all available scores:
- `t1` = 1/3 quantile
- `t2` = 2/3 quantile

Then:
- `score <= t1` → `health_label = 'Unhealthy'`
- `t1 < score <= t2` → `health_label = 'Moderate'`
- `score > t2` → `health_label = 'Healthy'`

If a row cannot produce a score, it is labeled `Moderate`.

This approach avoids hard-coded nutrition thresholds and (when scores exist) tends to produce a balanced 3-class dataset.

### 2.10 Output schema

The output CSV columns are constructed in this order:

1. Optional `description` (enabled by default)
2. Optional `dataType` (only if `--include-datatype`)
3. Core nutrient features (`CORE_FEATURES`)
4. Engineered features:
	 - `protein_ratio`, `fat_ratio`, `carb_ratio`, `sugar_ratio`, `fiber_density`, `sodium_density`
5. `processed_level`
6. `health_label`

For debugging, `--keep-debug-columns` can also keep identifiers like `fdcId` and `ingredients`.

---

## 3) Practical Validation Checklist

After running preprocessing, it’s good to confirm:

1. **No missing numeric values** in features (especially engineered ratios)
2. **Label distribution** is reasonable (e.g., not all one class)
3. **Feature ranges** look plausible (check outliers)

Example quick check in Python:

```python
import pandas as pd
df = pd.read_csv('foods_engineered_300.csv')
print(df.isna().sum().sort_values(ascending=False).head())
print(df['health_label'].value_counts())
print(df.describe())
```

---

## 4) Summary

- `food_data_fetcher.py` handles data acquisition from FoodData Central, including robust retry/backoff and optional resume/checkpointing in random mode.
- `data_preprocess.py` converts raw foods into a clean ML dataset by parsing core nutrients, engineering ratios, adding a creative processing feature, generating labels via quantiles, and filling any missing numeric engineered values with zeros.

