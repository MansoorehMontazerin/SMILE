"""Download the Feynman benchmark into data/ in the format smile_pipeline.py reads.

    python download_feynman.py                    # all 100 equations
    python download_feynman.py --eqn I.12.1 I.6.2a

The original AI Feynman download page (space.mit.edu/home/tegmark/aifeynman.html) is
offline. This script uses:
  - FeynmanEquations.csv  from the Internet Archive copy of the AI Feynman page;
  - the equation data     from PMLB (github.com/EpistasisLab/pmlb), the copy used by
                          SRBench: 100,000 rows per equation (the original files had
                          1,000,000 rows).

Each PMLB table (tab-separated, header row, target last) is written as
data/Feynman_with_units/<eqn>: space-separated, no header, inputs then target, the same
layout as the original AI Feynman files.
"""
from __future__ import annotations

import argparse
import gzip
import io
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
META_URL = ("https://web.archive.org/web/20250630050842id_/"
            "https://space.mit.edu/home/tegmark/aifeynman/FeynmanEquations.csv")
PMLB_URL = ("https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/"
            "datasets/{name}/{name}.tsv.gz")


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "smile-download/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def pmlb_name(eqn: str) -> str:
    """I.12.1 -> feynman_I_12_1, II.6.15a -> feynman_II_6_15a."""
    return "feynman_" + eqn.replace(".", "_")


def variable_names(row: pd.Series) -> list[str]:
    names = []
    for k in range(1, 11):
        value = row.get(f"v{k}_name")
        if isinstance(value, str) and value.strip():
            names.append(value.strip())
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eqn", nargs="+", help="Equations to download (default: all).")
    parser.add_argument("--out", type=Path, default=ROOT / "data", help="Output folder (default: data/).")
    parser.add_argument("--overwrite", action="store_true", help="Download files that already exist.")
    args = parser.parse_args()

    out_dir = args.out / "Feynman_with_units"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_path = args.out / "FeynmanEquations.csv"
    if args.overwrite or not meta_path.exists():
        print(f"Downloading FeynmanEquations.csv -> {meta_path}")
        meta_path.write_bytes(fetch(META_URL))
    meta = pd.read_csv(meta_path, encoding="utf-8-sig").dropna(subset=["Filename"])
    meta = meta.set_index("Filename")

    equations = args.eqn or list(meta.index)
    unknown = [e for e in equations if e not in meta.index]
    if unknown:
        sys.exit(f"Not in FeynmanEquations.csv: {unknown}")

    failed = []
    for k, eqn in enumerate(equations, 1):
        target = out_dir / eqn
        if target.exists() and not args.overwrite:
            print(f"[{k}/{len(equations)}] {eqn}: exists, skipped")
            continue
        try:
            raw = gzip.decompress(fetch(PMLB_URL.format(name=pmlb_name(eqn))))
        except (urllib.error.URLError, OSError) as error:
            print(f"[{k}/{len(equations)}] {eqn}: download failed ({error})")
            failed.append(eqn)
            continue
        table = pd.read_csv(io.BytesIO(raw), sep="\t", float_precision="round_trip")  # exact values
        names = variable_names(meta.loc[eqn])
        inputs = [c for c in table.columns if c != "target"]
        if sorted(inputs) != sorted(names) or "target" not in table.columns:
            print(f"[{k}/{len(equations)}] {eqn}: columns {list(table.columns)} do not match "
                  f"the metadata variables {names}; skipped")
            failed.append(eqn)
            continue
        table = table[names + ["target"]]  # metadata variable order, target last
        with open(target, "w", newline="\n") as stream:
            for row in table.itertuples(index=False):
                stream.write(" ".join(repr(float(v)) for v in row) + " \n")
        print(f"[{k}/{len(equations)}] {eqn}: {len(table)} rows -> {target}")

    print(f"\nDone: {len(equations) - len(failed)} of {len(equations)} equations in {out_dir}")
    if failed:
        print(f"Failed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
