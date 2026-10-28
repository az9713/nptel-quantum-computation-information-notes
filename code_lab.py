"""Run the exported laboratory companions from any working directory."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List chapter laboratories")
    parser.add_argument("--chapter", type=int, choices=range(1, 21))
    args = parser.parse_args()
    if args.list:
        for number in range(1, 21):
            meta = json.loads((ROOT / f"chapter-{number:02}.meta.json").read_text(encoding="utf-8"))
            print(f"{number:02}: {meta['title']}")
        return 0
    for number in ([args.chapter] if args.chapter else range(1, 21)):
        path = ROOT / f"code/ch{number:02}_lab.py"
        print(f"\nChapter {number:02} laboratory", flush=True)
        result = subprocess.run([sys.executable, str(path)], cwd=path.parent)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
