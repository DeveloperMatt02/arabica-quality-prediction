"""Project-wide constants: paths, random seed, column groups and the excellence rule.

Everything that the notebooks, the scripts and the tests need to agree on lives
here, so that a change (e.g. a different excellence threshold) propagates
everywhere.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "coffee_data.csv"
COORDS_PATH = DATA_DIR / "raw" / "extracted_coords.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42
TEST_SIZE = 0.20
N_FOLDS = 5

# --------------------------------------------------------------------------- #
# Target definition
# --------------------------------------------------------------------------- #
TARGET_SCORE = "Total.Cup.Points"
TARGET = "Excellent"
EXCELLENT_THRESHOLD = 85.0  # a batch is "excellent" when Total.Cup.Points >= 85

# --------------------------------------------------------------------------- #
# Column groups
# --------------------------------------------------------------------------- #
# Sensory scores assigned by the cupping panel. They are the *components* of
# Total.Cup.Points, hence they leak the target and are never used as agronomic
# predictors (see docs/LESSONS_LEARNED.md).
SENSORY_COLS = [
    "Aroma",
    "Flavor",
    "Aftertaste",
    "Acidity",
    "Body",
    "Balance",
    "Uniformity",
    "Clean.Cup",
    "Sweetness",
    "Cupper.Points",
]

# Sensory scores actually modelled in the sensory track (near-constant ones
# such as Uniformity / Clean.Cup / Sweetness are dropped, Cupper.Points is a
# global judgement rather than a sensory dimension).
SENSORY_MODEL_COLS = ["Aroma", "Flavor", "Aftertaste", "Acidity", "Body", "Balance"]

# Pre-harvest / physical attributes available *before* tasting.
NUMERIC_FEATURES = [
    "Moisture",
    "Category.One.Defects",
    "Category.Two.Defects",
    "altitude_mean_meters",
]
# ``Moisture == 0`` is how the CQI export encodes "not measured" (18% of the
# rows, concentrated in a few certifying labs). It is recoded as missing and
# imputed inside the pipeline together with an explicit indicator column.
MOISTURE_MISSING_FLAG = "Moisture_missing"
NUMERIC_MODEL_FEATURES = NUMERIC_FEATURES + [MOISTURE_MISSING_FLAG]
GEO_COLS = ["Country.of.Origin", "Region"]  # consumed by the transformers, never one-hot encoded
CATEGORICAL_FEATURES = ["MacroArea", "Color", "Processing.Method", "Variety"]

# Raw columns handed to the modelling pipelines (before macro-area mapping).
MODEL_INPUT_COLS = NUMERIC_FEATURES + GEO_COLS + ["Variety", "Processing.Method", "Color"]

# Administrative / identifier columns that carry no agronomic information.
ADMIN_COLS = [
    "Unnamed: 0",
    "Species",  # constant: every sample is Arabica
    "Owner",
    "Owner.1",
    "Farm.Name",
    "Lot.Number",
    "Mill",
    "ICO.Number",
    "Company",
    "Producer",
    "In.Country.Partner",
    "Certification.Body",
    "Certification.Address",
    "Certification.Contact",
    "Grading.Date",
    "Expiration",
    "Altitude",  # free-text altitude, superseded by altitude_mean_meters
    "unit_of_measurement",
    "altitude_low_meters",
    "altitude_high_meters",
    "Bag.Weight",
    "Number.of.Bags",
    "Harvest.Year",
]

# Plausible altitude range for Arabica cultivation (metres). Values outside are
# data-entry errors (the raw file contains altitudes up to 190,164 m).
ALTITUDE_MIN, ALTITUDE_MAX = 200.0, 3300.0

PROCESSING_MAP = {
    "Washed / Wet": "Washed",
    "Natural / Dry": "Natural",
    "Semi-washed / Semi-pulped": "Semi-washed",
    "Pulped natural / honey": "Honey",
    "Other": "Other",
}

MACRO_AREA_MAP = {
    # Central / North America & Caribbean
    "Guatemala": "Central/North America",
    "United States": "Central/North America",
    "United States (Puerto Rico)": "Central/North America",
    "Costa Rica": "Central/North America",
    "Mexico": "Central/North America",
    "Honduras": "Central/North America",
    "Panama": "Central/North America",
    "El Salvador": "Central/North America",
    "Nicaragua": "Central/North America",
    "Haiti": "Central/North America",
    # South America
    "Brazil": "South America",
    "Colombia": "South America",
    "Ecuador": "South America",
    "Peru": "South America",
    # Africa
    "Ethiopia": "Africa",
    "Uganda": "Africa",
    "Kenya": "Africa",
    "Tanzania, United Republic Of": "Africa",
    "Burundi": "Africa",
    "Rwanda": "Africa",
    "Malawi": "Africa",
    "Zambia": "Africa",
    "Mauritius": "Africa",
    "Cote d?Ivoire": "Africa",
    # Mainland Asia
    "China": "Mainland Asia",
    "India": "Mainland Asia",
    "Laos": "Mainland Asia",
    "Myanmar": "Mainland Asia",
    "Vietnam": "Mainland Asia",
    "Thailand": "Mainland Asia",
    # Maritime Asia / Pacific
    "Indonesia": "Maritime Asia/Pacific",
    "Japan": "Maritime Asia/Pacific",
    "Taiwan": "Maritime Asia/Pacific",
    "Papua New Guinea": "Maritime Asia/Pacific",
    "Philippines": "Maritime Asia/Pacific",
    "United States (Hawaii)": "Maritime Asia/Pacific",
}

# Maximum number of levels kept per categorical variable before one-hot
# encoding; rarer levels are pooled into an "infrequent" bucket. Only Variety
# (28 levels) is affected.
MAX_CATEGORIES = 10
