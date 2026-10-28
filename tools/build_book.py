"""Refresh the exported reader without private sources or authoring inputs."""
import hashlib
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
esc = html.escape


def shell(title, body, chapters, current=None):
    options = "".join(
        f'<li><a href="chapter-{number:02}.html">'
        f'{number}. {esc(meta["title"])}</a></li>'
        for number, meta, fragment in chapters)
    page_turn = ""
    if current is not None:
        numbers = [number for number, _, _ in chapters]
        previous = max((n for n in numbers if n < current), default=None)
        following = min((n for n in numbers if n > current), default=None)
        links = []
        for value, label in [(previous, "Previous chapter"),
                             (following, "Next chapter")]:
            if value is not None:
                links.append(f'<a href="chapter-{value:02}.html">{label}</a>')
        page_turn = '<nav class="page-turn" aria-label="Chapter navigation">' + " ".join(links) + "</nav>"
    sidebar = (
        '<aside class="toc" id="chapter-toc" aria-label="Contents"></aside>'
        if current else
        '<aside class="toc"><h2>Chapters</h2><ol>' + options + "</ol></aside>")
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'self\'; '
        'script-src \'self\'; style-src \'self\' \'unsafe-inline\'; '
        'img-src \'self\' data:; connect-src \'none\'; object-src \'none\'; '
        'base-uri \'none\'"><title>' + esc(title) + '</title>'
        '<link rel="stylesheet" href="assets/book.css">'
        '<script src="assets/math-config.js"></script>'
        '<script defer src="assets/mathjax-tex-svg.js"></script>'
        '<script defer src="assets/book.js"></script></head><body>'
        '<a class="skip" href="#chapter-content">Skip to content</a>'
        '<header class="site-header"><a class="brand" href="index.html">'
        'Quantum Computation and Quantum Information</a>'
        '<nav class="course-nav" aria-label="Course">'
        '<a href="index.html">Contents</a><a href="source-map.html">Source map</a>'
        '<a href="references.html">Primary references</a>'
        '<a href="notation.html">Notation</a>'
        '<a href="code-lab.html">Code laboratory</a>'
        '<a href="evidence.html">Evidence and limitations</a></nav></header>'
        '<div class="layout">' + sidebar + '<main>'
        '<div class="reading-tools">'
        '<button id="theme-toggle" type="button">Light theme</button>'
        '<button id="text-toggle" type="button">Larger text</button>'
        '<button id="solutions-toggle" type="button" aria-pressed="false">All solutions</button>'
        '<button id="print-page" type="button">Print / PDF</button></div>'
        '<article id="chapter-content">' + body + '</article>' + page_turn +
        '</main></div><footer class="site-footer">'
        'Independent educational rewrite of Prof. Chiranjib Mitra’s NPTEL course, '
        'IISER Kolkata; coordinated by IIT Madras. Recordings and papers are linked, '
        'not redistributed. <a href="evidence.html">Verification scope</a>.'
        '</footer></body></html>')


def body_of(page):
    marker = '<article id="chapter-content">'
    if page.count(marker) != 1:
        raise ValueError('Expected one chapter-content article')
    body, end, _ = page.split(marker, 1)[1].rpartition('</article>')
    if not end:
        raise ValueError('Missing content article end')
    return body


def main():
    chapters = []
    pages = []
    for number in range(1, 21):
        path = ROOT / f'chapter-{number:02}.html'
        meta = json.loads(path.with_suffix('.meta.json').read_text(encoding='utf-8'))
        body = body_of(path.read_text(encoding='utf-8'))
        if hashlib.sha256(body.encode()).hexdigest() != meta['body_sha256']:
            raise ValueError(f'{path.name}: body changed; obtain review and update its sidecar before rebuilding')
        chapters.append((number, meta, body))
    # Prepare every page before writing, so a failed digest changes no files.
    for number, meta, body in chapters:
        pages.append((ROOT / f'chapter-{number:02}.html', shell(meta['title'], body, chapters, number)))
    for name in ['index', 'source-map', 'references', 'notation', 'code-lab', 'evidence']:
        path = ROOT / f'{name}.html'
        page = path.read_text(encoding='utf-8')
        title = html.unescape(re.search(r'<title>(.*?)</title>', page, re.S).group(1))
        pages.append((path, shell(title, body_of(page), chapters)))
    for path, page in pages:
        with path.open('w', encoding='utf-8', newline='') as stream:
            stream.write(page)
    print(json.dumps({'pages_refreshed': len(pages), 'chapter_body_digests_verified': len(chapters), 'private_inputs_required': False}))


if __name__ == '__main__':
    main()
