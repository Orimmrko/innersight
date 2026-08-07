"""Stage 3 -- replace every raw count with a z-score against that person's own history.

Stage 3 of the pipeline.
  reads:  data/labelled_counts.parquet
  writes: data/dataset.parquet -- 18 deviations per user-day

Stub. Stage 0 writes no logic -- this file exists so that the imports of the
stage that follows already resolve.
"""

from __future__ import annotations

pass
