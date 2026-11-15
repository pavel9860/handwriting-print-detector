"""Rebuilds the detection set into a new folder, checks it, packs it and pushes a new Kaggle version
python scripts/forms/release.py SET_DIR PACK_DIR MESSAGE [--workers 8]"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
TOTAL = 3250
KAGGLE_ID = "pavel9860/hwdet-detection-768x1024"


def run(*cmd):
    print("$", *cmd, flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("set_dir")
    ap.add_argument("pack_dir")
    ap.add_argument("message")
    ap.add_argument("--workers", default="8")
    a = ap.parse_args()
    run(sys.executable, HERE / "build_3250.py", a.set_dir, "--workers", a.workers)
    run(sys.executable, HERE / "check_set.py", a.set_dir)
    pages = len(list(Path(a.set_dir).glob("*/images/*.jpg")))
    if pages != TOTAL:
        sys.exit(f"{pages} pages, not {TOTAL}: nothing pushed")
    run(sys.executable, HERE / "pack_ds.py", a.set_dir, a.pack_dir, KAGGLE_ID)
    run("kaggle", "datasets", "version", "-p", a.pack_dir, "-m", a.message, "--delete-old-versions")


if __name__ == "__main__":
    main()
