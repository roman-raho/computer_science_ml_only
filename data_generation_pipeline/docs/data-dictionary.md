# Data Dictionary - Score Tables (v1)

Raw entity tables (artists, artworks, galleries, auction houses, museums, provenance)  
Are shown in ER diagram PNG

## Table: artist_reputation

- **artist_id** (UUID, FK → artists).
- **reputation_score** (INT, 0–100)  
  Formula: weighted average of (number of exhibitions, average sale value).
- **drivers_json** (JSON) → { "exhibitions": int, "avg_sale": float }.
- **created_at** (DATETIME).  
  **Purpose:** Quantifies how established an artist is.

## Table: gallery_score

- **gallery_id** (UUID, FK → galleries).
- **score** (INT, 0–100).  
  Formula: function of number of exhibitions × average artist reputation.
- **drivers_json** (JSON).
- **created_at**.  
  **Purpose:** Captures gallery strength based on output + prestige.

## Table: auction_house_score

- **auction_house_id** (UUID).
- **tier** (VARCHAR: "Tier 1", "Tier 2", "Tier 3").  
  Rule: based on sales volume and average hammer price thresholds.
- **drivers_json** (JSON).
- **created_at**.  
  **Purpose:** Categorises auction houses by market influence.

## Table: museum_prestige

- **museum_id** (UUID).
- **prestige_score** (INT, 0–100).  
  Derived directly from a synthetic prestige ranking.
- **drivers_json** (JSON).
- **created_at**.  
  **Purpose:** Encodes cultural validation of artworks.

## Table: provenance_score

- **artwork_id** (UUID, FK → artworks).
- **provenance_score** (INT, 0–100).  
  Formula: combines number of prior owners + age of ownership records.
- **drivers_json** (JSON).
- **created_at**.  
  **Purpose:** Rewards artworks with rich, verifiable ownership history.
