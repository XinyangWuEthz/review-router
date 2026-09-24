#!/usr/bin/env python
"""Stage the saved experiment pages and check their local links, without rerunning ML.

    python scripts/build_pages.py --output dist/pages

The output directory must not exist. Only the listed public record assets are copied.
"""

from __future__ import annotations

import argparse
import shutil
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SUPPORT_FILES = (
    "configs/jev_questions.json",
    "analysis/jev/protocol.json",
    "analysis/jev/cohort.csv",
    "LICENSE",
)
LANDING = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="0; url=record/index.html">
  <title>review-router | Human-review routing benchmark</title>
  <meta name="description" content="A reproducible benchmark for human-review routing:
  compare admission policies and queue ordering under fixed reviewer capacity.">
  <link rel="canonical" href="https://xinyangwuethz.github.io/review-router/record/index.html">
</head>
<body>
  <main>
    <h1>review-router</h1>
    <p><a href="record/index.html">Read the experiment results and method.</a></p>
  </main>
</body>
</html>
"""


class PageLinks(HTMLParser):
    """Collect local resource references and named fragment targets."""

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if value is None:
                continue
            if name in {"href", "src"}:
                self.links.append(value)
            if name == "id" or (tag == "a" and name == "name"):
                self.ids.add(value)


def validate_links(output: Path) -> int:
    pages: dict[Path, PageLinks] = {}
    for path in output.rglob("*.html"):
        page = PageLinks()
        page.feed(path.read_text(encoding="utf-8"))
        pages[path] = page
    failures = []
    checked = 0
    for path, page in pages.items():
        for link in page.links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            target = (path.parent / unquote(url.path)).resolve() if url.path else path
            if target.is_dir():
                target /= "index.html"
            if not target.is_relative_to(output) or not target.is_file():
                failures.append(f"{path.relative_to(output)}: missing local target {link}")
            elif (
                url.fragment
                and target in pages
                and unquote(url.fragment) not in pages[target].ids
            ):
                failures.append(f"{path.relative_to(output)}: missing fragment {link}")
            checked += 1
    if failures:
        raise ValueError("\n".join(failures))
    return checked


def build(output: Path) -> None:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    paths = [
        *sorted((ROOT / "record").glob("*.html")),
        *sorted((ROOT / "record").glob("*.json")),
        *sorted((ROOT / "record" / "figures").glob("*.png")),
        *(ROOT / name for name in SUPPORT_FILES),
    ]
    for source in paths:
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Expected a regular source file: {source}")
        target = output / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    (output / "index.html").write_text(LANDING, encoding="utf-8")
    (output / ".nojekyll").touch()
    checked = validate_links(output)
    print(f"Staged {len(paths) + 2} files in {output}; checked {checked} local links.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "pages")
    build(parser.parse_args().output)
