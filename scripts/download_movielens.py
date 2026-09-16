"""Download and extract the MovieLens 100k dataset (opt-in; requires network).

MovieLens is a public research dataset from GroupLens (grouplens.org). It is not
redistributed in this repo; this script fetches it on demand. Not run in CI -
the eval uses synthetic data offline.

    python scripts/download_movielens.py --out data/ml-100k
"""

from __future__ import annotations

import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/ml-100k", help="destination directory")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {URL} …")
    with urllib.request.urlopen(URL) as resp:  # noqa: S310 - fixed, trusted URL
        blob = resp.read()

    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        # The archive nests files under "ml-100k/"; flatten the ones we use.
        for member in ("u.data", "u.item", "u.user"):
            data = zf.read(f"ml-100k/{member}")
            (out / member).write_bytes(data)
            print(f"  wrote {out / member} ({len(data):,} bytes)")

    print(f"Done. Train with:  python -m twotower.train --movielens {out}")


if __name__ == "__main__":
    main()
