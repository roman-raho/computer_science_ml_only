# Synthetic Data Generator (v1)

## Purpose

To generate a reproducible synthetic dataset covering the ERD entities for Section B v1.  
Outputs: raw entity tables + compressed score tables.

## inputs

- **seed**: integer, fixed in /ml/SEED.txt. Guarantees deterministic output
- **Config**: number of artists, artworks, galleries, auction houses, etc

## Process

1. **Generate raw entities**

   - Artists, artworks, galleries, auction houses, museums, provenance.
   - Use random distributions (uniform, normal, categorical) with realistic ranges.
   - Ensure relations are respected (artworks -> artists, provenance -> artworks, etc.).

2. **Derive score tables**

   - _Artist reputation_: weighted avg of exhibitions + sales. Normalised 0–100.
   - _Gallery score_: exhibitions hosted × reputation of artists.
   - _Auction house score_: assign Tier 1–3 by sales volume/price thresholds.
   - _Museum prestige_: scale prestige_rank to 0–100.
   - _Provenance score_: combine count + age of ownership records, normalised.

3. **Normalisation**

   - Apply min–max scaling for all scores to ensure 0–100 comparability.
   - Store drivers in `drivers_json` for explainability.

4. **Determinism**
   - Re-running with the same seed produces identical outputs.
   - File hashes recorded in `/ml/CHECKSUMS.md`.

## Outputs

- `/ml/data/raw/*.csv` – base entity tables.
- `/ml/data/processed/*.csv` – score tables.
- `/ml/reports/*.md` – “Top 10” galleries/auction houses etc.
- `/ml/CHECKSUMS.md` – file hashes.
