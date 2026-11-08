"""Check every external link in docs/*.md. Run manually (needs network):

    python tools/check_links.py

arXiv links are looked up through the arXiv API, YouTube links through oEmbed, and every other
link must answer HTTP 200 (redirects followed). Exit code 1 if anything fails.
"""
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")


def collect():
    links = {}
    for md in sorted((ROOT / "docs").glob("*.md")):
        for text, url in LINK.findall(md.read_text(encoding="utf-8")):
            links.setdefault(url, md.name)
    return links


def check_arxiv(ids):
    api = "http://export.arxiv.org/api/query?max_results=200&id_list=" + ",".join(sorted(ids))
    root = ET.fromstring(urllib.request.urlopen(api, timeout=60).read())
    ns = {"a": "http://www.w3.org/2005/Atom"}
    found = {e.find("a:id", ns).text.rsplit("/", 1)[1].split("v")[0] for e in root.findall("a:entry", ns)}
    return sorted(set(ids) - found)


def check_youtube(url):
    try:
        api = "https://www.youtube.com/oembed?format=json&url=" + urllib.parse.quote(url, safe="")
        json.load(urllib.request.urlopen(api, timeout=30))
        return True
    except Exception:
        return False


def check_http(url):
    r = subprocess.run(["curl", "-s", "-o", os.devnull, "-L", "-m", "45", "-w", "%{http_code}", "-A", "Mozilla/5.0", url],
                       capture_output=True, text=True)
    return r.stdout.strip() == "200"


def main():
    links = collect()
    print(f"{len(links)} distinct external links")
    failures = []
    arxiv_ids = [u.rsplit("/", 1)[1] for u in links if "arxiv.org/abs/" in u]
    failures += [(links[f"https://arxiv.org/abs/{i}"], f"arXiv:{i} not found") for i in check_arxiv(arxiv_ids)]
    for url, md in links.items():
        if "arxiv.org/abs/" in url:
            continue
        ok = check_youtube(url) if "youtube.com/watch" in url else check_http(url)
        if not ok:
            failures.append((md, url))
    for md, what in failures:
        print(f"FAIL {md}: {what}")
    print("all links OK" if not failures else f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
