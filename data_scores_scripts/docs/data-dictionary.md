# Score Tables – Data Dictionary (v1)

## artist_reputation

artist_id (UUID, FK -> artists)  
reputation_score
Based on exhibition count and average sale value.  
drivers_json (json)  
created_at (datetime)

## gallery_score

gallery_id (UUID, FK -> galleries)  
score
Based on number of exhibitions and average artist reputation.  
drivers_json (json)  
created_at (datetime)

## auction_house_score

auction_house_id (UUID)  
tier (Tier 1, Tier 2, Tier 3)  
Tier set from sales volume and average hammer price.  
drivers_json (json)  
created_at (datetime)

## museum_prestige

museum_id (UUID)  
prestige_score
From synthetic prestige ranking.  
drivers_json (json)  
created_at (datetime)

## provenance_score

artwork_id (UUID, FK -> artworks)  
provenance_score (int 0–100)  
From number of past owners and age of ownership records.  
drivers_json (json)  
created_at (datetime)
