"""Save 12 USDA food-safety pages as Markdown in knowledge/.

A normal urllib or requests call is rejected by the USDA site. curl_cffi
impersonates a browser so the download succeeds. Run from the project root:

    uv add curl-cffi beautifulsoup4
    uv run python scripts/fetch_usda.py
"""

import re
import time
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag
from curl_cffi import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "knowledge"
BASE = "https://www.fsis.usda.gov/food-safety/safe-food-handling-and-preparation/food-safety-basics"

PAGES = [
    ("steps-to-keep-food-safe", f"{BASE}/steps-keep-food-safe"),
    ("danger-zone", f"{BASE}/danger-zone-40f-140f"),
    ("how-temperatures-affect-food", f"{BASE}/how-temperatures-affect-food"),
    ("safe-temperature-chart", f"{BASE}/safe-temperature-chart"),
    ("leftovers-and-food-safety", f"{BASE}/leftovers-and-food-safety"),
    ("refrigeration", f"{BASE}/refrigeration"),
    ("freezing-and-food-safety", f"{BASE}/freezing-and-food-safety"),
    ("big-thaw", f"{BASE}/big-thaw-safe-defrosting-methods"),
    ("grilling-and-food-safety", f"{BASE}/grilling-and-food-safety"),
    ("cooking-with-microwave-ovens", f"{BASE}/cooking-microwave-ovens"),
    ("food-thermometers", f"{BASE}/food-thermometers"),
    ("cutting-boards", f"{BASE}/cutting-boards"),
]


def squash(text: str) -> str:
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def inline(node: Tag) -> str:
    parts: list[str] = []
    for child in node.children:
        if isinstance(child, NavigableString):
            parts.append(str(child).replace("\xa0", " "))
        elif isinstance(child, Tag):
            if child.name in {"ul", "ol", "table", "script", "style", "img", "picture", "svg"}:
                continue
            if child.name in {"strong", "b"}:
                parts.append(f"**{inline(child)}**")
            elif child.name in {"em", "i"}:
                parts.append(f"*{inline(child)}*")
            elif child.name == "br":
                parts.append(" ")
            else:
                parts.append(inline(child))
    return squash("".join(parts))


def list_to_markdown(lst: Tag, ordered: bool, indent: int = 0) -> str:
    lines: list[str] = []
    for index, item in enumerate(lst.find_all("li", recursive=False), start=1):
        marker = f"{index}." if ordered else "-"
        lines.append(f"{'  ' * indent}{marker} {inline(item)}")
        for nested in item.find_all(["ul", "ol"], recursive=False):
            lines.append(list_to_markdown(nested, nested.name == "ol", indent + 1))
    return "\n".join(lines)


def table_to_markdown(table: Tag) -> str:
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = [squash(cell.get_text(" ", strip=True)).replace("|", "/") for cell in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]

    def format_row(row: list[str]) -> str:
        return "| " + " | ".join(row) + " |"

    header = format_row(rows[0])
    separator = "| " + " | ".join("---" for _ in rows[0]) + " |"
    body = "\n".join(format_row(row) for row in rows[1:])
    return "\n".join(part for part in (header, separator, body) if part)


def blocks(parent: Tag) -> list[str]:
    chunks: list[str] = []
    for child in parent.children:
        if not isinstance(child, Tag):
            continue
        if child.name in {"h2", "h3", "h4"}:
            chunks.append(f"{'#' * int(child.name[1])} {squash(child.get_text(' ', strip=True))}")
        elif child.name == "p":
            text = inline(child)
            if text:
                chunks.append(text)
        elif child.name == "ul":
            chunks.append(list_to_markdown(child, ordered=False))
        elif child.name == "ol":
            chunks.append(list_to_markdown(child, ordered=True))
        elif child.name == "table":
            rendered = table_to_markdown(child)
            if rendered:
                chunks.append(rendered)
        elif child.name in {"div", "section", "figure", "article"}:
            chunks.extend(blocks(child))
    return [chunk for chunk in chunks if chunk.strip()]


def page_to_markdown(html: str, url: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    main = soup.select_one("div.l-sidebar__main")
    if main is None:
        raise ValueError("could not find the article body")

    title_node = main.select_one("h1.introduction__title")
    title = squash(title_node.get_text(" ", strip=True)) if title_node else url.rsplit("/", 1)[-1]

    body: list[str] = []
    for section in main.find_all("section", recursive=False):
        classes = set(section.get("class") or [])
        if "section-cards" in classes or "modified" in classes:
            continue
        body.extend(blocks(section))

    return f"# {title}\n\nSource: {url}\n\n" + "\n\n".join(body).strip() + "\n"


def fetch(url: str) -> str:
    response = requests.get(url, impersonate="chrome", timeout=40)
    response.raise_for_status()
    return response.text


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in PAGES:
        html = fetch(url)
        markdown = page_to_markdown(html, url)
        path = OUT_DIR / f"{name}.md"
        path.write_text(markdown, encoding="utf-8")
        print(f"wrote {path.name} ({len(markdown):,} characters)")
        time.sleep(0.4)


if __name__ == "__main__":
    main()
