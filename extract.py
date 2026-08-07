"""Stage 1 -- turn five raw CERT logs into one row per user per day.

Stage 1 of the pipeline.
  reads:  the five raw CSVs (see config.log_path)
  writes: data/daily_counts.parquet -- user, day, 18 counts

Stub. Stage 0 writes no logic -- this file exists so that the imports of the
stage that follows already resolve.
"""

from __future__ import annotations

pass
