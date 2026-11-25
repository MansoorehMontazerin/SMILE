"""Download the original AI Feynman benchmark into data/.

    python download_feynman.py                 # download, unpack, remove the archive
    python download_feynman.py --keep-archive  # keep Feynman_with_units.tar.gz as well

The AI Feynman page (space.mit.edu/home/tegmark/aifeynman.html) is no longer online, but
its Internet Archive copy links the data files:
  - Feynman_with_units.tar.gz (4.1 GB; about 8 GB unpacked, 1,000,000 rows per equation)
    from the Dropbox link given on that page;
  - FeynmanEquations.csv (variable names, ranges, formulas) from the archived page.

Result:
  data/Feynman_with_units/<eqn>     one space-separated file per equation
  data/FeynmanEquations.csv
"""
from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_URL = "https://www.dropbox.com/s/7kgfr00qpokgz8w/Feynman_with_units.tar.gz?dl=1"
META_URL = ("https://web.archive.org/web/20250630050842id_/"
            "https://space.mit.edu/home/tegmark/aifeynman/FeynmanEquations.csv")
ARCHIVE_PAGE = "https://web.archive.org/web/20251212164109/https://space.mit.edu/home/tegmark/aifeynman.html"


def download(url: str, target: Path) -> None:
    """Stream url to target (via a .part file), printing progress."""
    part = target.with_name(target.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "smile-download/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, open(part, "wb") as stream:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while chunk := response.read(1 << 20):
            stream.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e9:.2f} / {total / 1e9:.2f} GB ({100 * done / total:.0f}%)",
                      end="", flush=True)
    print()
    part.replace(target)


def extract(archive: Path, out_dir: Path) -> None:
    """Unpack the archive into out_dir, refusing any member that would land outside it."""
    root = out_dir.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            path = (root / member.name).resolve()
            if root not in path.parents and path != root:
                sys.exit(f"Unsafe path in archive: {member.name}")
            if not (member.isfile() or member.isdir()):
                continue
            tar.extract(member, root)
            if member.isfile():
                print(f"\r  extracted {member.name:<40}", end="", flush=True)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "data", help="Output folder (default: data/).")
    parser.add_argument("--keep-archive", action="store_true", help="Keep the downloaded .tar.gz.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    meta = args.out / "FeynmanEquations.csv"
    if meta.exists():
        print(f"{meta} exists, skipped")
    else:
        print("Downloading FeynmanEquations.csv")
        download(META_URL, meta)

    data_dir = args.out / "Feynman_with_units"
    if data_dir.is_dir() and any(data_dir.iterdir()):
        print(f"{data_dir} exists, skipped")
    else:
        archive = args.out / "Feynman_with_units.tar.gz"
        if not archive.exists():
            print(f"Downloading Feynman_with_units.tar.gz (4.1 GB)\n  source: {ARCHIVE_PAGE}")
            download(DATA_URL, archive)
        print(f"Unpacking into {args.out}")
        extract(archive, args.out)
        if not args.keep_archive:
            archive.unlink()

    count = sum(1 for f in data_dir.iterdir() if f.is_file()) if data_dir.is_dir() else 0
    print(f"\nDone: {count} equation files in {data_dir}, metadata in {meta}")


if __name__ == "__main__":
    main()
