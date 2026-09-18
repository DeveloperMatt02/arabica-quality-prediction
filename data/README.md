# Data

## `raw/coffee_data.csv`

1,311 graded Arabica lots exported from the **Coffee Quality Institute (CQI)**
database (Q-grader cupping scores 2010–2018), in the widely used tabular form
compiled by James LeDoux (`jldbc/coffee-quality-database`) and redistributed
on Kaggle. 44 columns: administrative metadata, ten sensory scores, the total
cup score, and physical / agronomic attributes (altitude, moisture, defects,
variety, processing method, bean colour).

The file is versioned here unchanged (650 KB) so that the whole analysis is
reproducible offline. The original data are published by CQI for public use;
this repository adds no claim over them.

## `raw/extracted_coords.csv`

Lookup table *(Country.of.Origin, Region) → (Latitude, Longitude)* obtained
once with `scripts/geocode_regions.py` (OpenStreetMap Nominatim, OpenCage as
fallback). Coordinates are region- or country-level centroids, used only for
the exploratory map and the unsupervised analysis - never as model inputs.

## Column dictionary (columns used downstream)

| Column | Type | Meaning |
|---|---|---|
| `Total.Cup.Points` | float | Q-grader total score (0–100). **Target:** `Excellent = Total.Cup.Points >= 85` |
| `Aroma` … `Cupper.Points` | float | The ten sensory scores that add up to the total - used only in the sensory track |
| `altitude_mean_meters` | float | Mean farm altitude (m). 17% missing; values outside 200–3,300 m are errors |
| `Moisture` | float | Green-bean moisture fraction. `0` means *not measured* (18% of rows) |
| `Category.One.Defects` | int | Primary (disqualifying) defects per 300 g sample |
| `Category.Two.Defects` | int | Secondary defects per 300 g sample |
| `Country.of.Origin`, `Region` | str | Origin; collapsed into five macro-areas for modelling |
| `Variety` | str | Botanical variety (28 levels; the 10 most frequent are kept, the rest pooled) |
| `Processing.Method` | str | Washed / Natural / Semi-washed / Honey / Unknown |
| `Color` | str | Green-bean colour: green / blue-green / Unknown |
