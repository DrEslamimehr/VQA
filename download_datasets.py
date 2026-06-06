#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PHYSIONET_2015_ZIP = "https://physionet.org/files/challenge-2015/1.0.0/training.zip"
WESAD_UCI = "https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection"
WESAD_HOME = "https://www.eti.uni-siegen.de/ubicomp/home/datasets/icmi18/?lang=en"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("content-length", 0) or 0)
        done = 0
        with dest.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r{done / total:6.1%}", end="", flush=True)
        if total:
            print()
    print(f"saved {dest}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download or locate datasets used by QA-TM.")
    parser.add_argument("--physionet2015", action="store_true", help="Download PhysioNet/CinC 2015 training.zip")
    parser.add_argument("--wesad", action="store_true", help="Print WESAD official download instructions")
    args = parser.parse_args()

    if args.physionet2015:
        download(PHYSIONET_2015_ZIP, ROOT / "data/raw/physionet_challenge_2015/training.zip")
    if args.wesad or not args.physionet2015:
        print("WESAD has academic/non-commercial terms; download from the official pages:")
        print(f"  UCI:  {WESAD_UCI}")
        print(f"  Home: {WESAD_HOME}")
        print("Then extract so files look like data/raw/WESAD/S2/S2.pkl, etc.")


if __name__ == "__main__":
    main()

