"""Verify the assembled book, independent source spans, and exact laboratory outputs."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT


class Inspect(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids = []
        self.links = []
        self.codes = []
        self.active = None
        self.exercises = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        for key in ["href", "src"]:
            if attrs.get(key):
                self.links.append(attrs[key])
        if tag == "article" and "exercise" in attrs.get("class", "").split():
            self.exercises += 1
        if tag == "code" and ("data-file" in attrs or
                             "output" in attrs.get("class", "").split()):
            self.active = {"attributes": attrs, "text": ""}

    def handle_data(self, data):
        if self.active is not None:
            self.active["text"] += data

    def handle_endtag(self, tag):
        if tag == "code" and self.active is not None:
            self.codes.append(self.active)
            self.active = None


def norm(text):
    return text.replace("\r\n", "\n").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--skip-execution", action="store_true")
    args = parser.parse_args()
    failures = []
    pages = {}
    for path in OUT.glob("*.html"):
        inspector = Inspect()
        inspector.feed(path.read_text(encoding="utf-8"))
        if len(inspector.ids) != len(set(inspector.ids)):
            failures.append(f"{path.name}: duplicate HTML ID")
        pages[path.name] = inspector
    for name, page in pages.items():
        for link in page.links:
            parts = urlsplit(link)
            if parts.scheme:
                if parts.scheme not in ["https", "http", "mailto", "data"]:
                    failures.append(f"{name}: unsafe scheme {parts.scheme}")
                continue
            if parts.netloc:
                failures.append(f"{name}: protocol-relative URL")
                continue
            target = (OUT / unquote(parts.path or name)).resolve()
            if not target.is_relative_to(OUT):
                failures.append(f"{name}: local link escapes export: {link}")
            elif not target.exists():
                failures.append(f"{name}: missing local target: {link}")
            elif parts.fragment and target.name in pages:
                if unquote(parts.fragment) not in pages[target.name].ids:
                    failures.append(f"{name}: missing fragment: {link}")
    chapters = []
    for number in range(1, 21):
        name = f"chapter-{number:02}.html"
        canonical = OUT / f"chapter-{number:02}.html"
        meta_path = canonical.with_suffix(".meta.json")
        if name not in pages or not meta_path.exists():
            if not args.allow_partial:
                failures.append(f"chapter {number}: incomplete draft")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        page = pages[name]
        if page.exercises != 6:
            failures.append(f"chapter {number}: {page.exercises} exercises")
        source = {v["video_id"]: v for v in meta["recordings"]}
        for section in meta.get("sections", []):
            video_id = section.get("video_id")
            if not video_id:
                continue
            if video_id not in source:
                failures.append(f"chapter {number}: unknown section video")
                continue
            start = section.get("start_seconds", section.get("start"))
            end = section.get("end_seconds", section.get("end"))
            if start is not None and end is not None:
                if not 0 <= start <= end <= source[video_id]["duration_seconds"]:
                    failures.append(f"chapter {number}: span out of duration {section.get('id')}")
        for video_id, seconds in re.findall(r"watch\?v=([A-Za-z0-9_-]+)(?:&amp;|&)t=([\d.]+)s",
                                            canonical.read_text(encoding="utf-8")):
            if video_id not in source or not 0 <= float(seconds) <= source[video_id]["duration_seconds"]:
                failures.append(f"chapter {number}: timestamp outside assigned source")
        code = OUT / f"code/ch{number:02}_lab.py"
        output = code.with_suffix(".out")
        embedded = [c for c in page.codes if c["attributes"].get("data-file") == code.name]
        if len(embedded) != 1:
            failures.append(f"chapter {number}: missing or duplicated full laboratory")
        elif norm(embedded[0]["text"]) != norm(code.read_text(encoding="utf-8")):
            failures.append(f"chapter {number}: embedded source differs")
        actual_outputs = [c["text"] for c in page.codes if "output" in c["attributes"].get("class", "").split()]
        saved = output.read_text(encoding="utf-8") if output.exists() else ""
        if not any(norm(value) == norm(saved) for value in actual_outputs):
            failures.append(f"chapter {number}: saved output absent from page")
        result = {"chapter": number, "exercises": page.exercises,
                  "code_sha256": hashlib.sha256(code.read_bytes()).hexdigest(),
                  "body_sha256": hashlib.sha256(canonical.read_bytes()).hexdigest()}
        if not args.skip_execution:
            execution = subprocess.run([sys.executable, str(code)], cwd=code.parent,
                                       capture_output=True, text=True, timeout=60)
            result["exit_code"] = execution.returncode
            result["stdout_matches"] = norm(execution.stdout) == norm(saved)
            if execution.returncode or not result["stdout_matches"]:
                failures.append(f"chapter {number}: execution/output mismatch: {execution.stderr[:500]}")
        chapters.append(result)
    report = {"chapters_checked": len(chapters),
              "pages_checked": len(pages),
              "laboratory_execution_performed": not args.skip_execution,
              "failures": failures, "chapters": chapters,
              "scope": "Structure, local links, source-duration bounds, exact embedded source/output and finite program execution; not a mathematical proof audit."}
    (OUT / "package-validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"chapters_checked": len(chapters), "pages_checked": len(pages),
                      "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
