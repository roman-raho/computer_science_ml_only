This directory contains the data generation + scoring pipeline for Section B, v1 (Data Foundation).

# Version 1

## What is covers

First I need to generate relation aware synthetic datasets for all ERD tables with timestamps

The I need to create score tables for each ERD section

- [x] Artist_reputation - rolling median sale, volume, museum expose -> score 0-100
- [x] Gallery_score - avg outcomes of represented artists, % above estimate, number of artists and years active
- [x] Auction_house_tier - sell through, avg hammer, volatility, global reach -> tier
- [x] Museum_prestige - visitors, age, exhibitions -> score
- [x] Provenance_score - chain length, museum ownership/lons, gaps -> score

What v1 covers: synthetic ERD data, reproducible with a fixed seed, compressed into scores.

Deliverables list: Data dictionary, ERD snapshot, generator script placeholder, CSV/Parquet dumps, checksums, Top-10 reports.
