"""
Ingest real raw station files from ``data/raw/`` and write the canonical
CSV used by the rest of the pipeline.

Usage:
    python scripts/build_canonical.py

The script:
1. Locates one file per station under ``data/raw/<Station>*.xls*``
2. Applies the alias-tolerant loader in :mod:`data_ingestion.loaders`
3. Runs the cleaning pipeline (imputation + outlier flagging)
4. Writes ``data/processed/bekaa_valley_clean.csv``

Drop your Ammik / Doures / Ras_Baalbeck / Tal_Amara files in ``data/raw/``
and re-run.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_ingestion.cleaners import clean_pipeline
from data_ingestion.loaders import build_canonical_dataset


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    log = logging.getLogger("bekaasense")

    repo = Path(__file__).resolve().parent.parent
    raw_dir = repo / "data" / "raw"
    out_path = repo / "data" / "processed" / "bekaa_valley_clean.csv"

    if not any(raw_dir.iterdir()):
        log.error("No files in %s. "
                  "Drop your raw station files there, or run "
                  "`python scripts/generate_synthetic.py` for a demo.", raw_dir)
        sys.exit(1)

    df = build_canonical_dataset(raw_dir, out_path)
    df = clean_pipeline(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("Wrote %d rows to %s", len(df), out_path)


if __name__ == "__main__":
    main()
