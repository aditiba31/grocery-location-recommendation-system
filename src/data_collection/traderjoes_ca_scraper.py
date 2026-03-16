# Trader Joe's California Store Scraper
# Scrapes all CA store locations from locations.traderjoes.com

import csv
import random
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Config ---

BASE_URL   = "https://locations.traderjoes.com"
CA_URL     = f"{BASE_URL}/ca/"
OUTPUT_CSV = Path("data/trader_joes/tj_locations_raw.csv")

MIN_SLEEP = 2.0
MAX_SLEEP = 4.0
TIMEOUT   = 20

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# --- Session ---

def build_session():
    session = requests.Session()
    session.headers.update(HEADERS)
    retry = Retry(
        total=2,
        read=2,
        connect=2,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://",  adapter)
    return session


# --- Helpers ---

def polite_sleep():
    time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))


def fetch_soup(session, url):
    resp = session.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    polite_sleep()
    return BeautifulSoup(resp.text, "html.parser")


def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def is_city_path(path):
    return bool(re.fullmatch(r"/ca/[^/]+/?", path))


def is_store_path(path):
    return bool(re.fullmatch(r"/ca/[^/]+/\d+/?", path))


# --- Scraping ---

def get_city_urls(session):
    soup      = fetch_soup(session, CA_URL)
    city_urls = set()

    for a in soup.find_all("a", href=True):
        parsed = urlparse(urljoin(BASE_URL, a["href"].strip()))
        if parsed.netloc != urlparse(BASE_URL).netloc:
            continue
        if is_city_path(parsed.path) and parsed.path.rstrip("/") != "/ca":
            city_urls.add(
                f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}/"
            )

    return sorted(city_urls)


def get_store_urls(session, city_url):
    soup       = fetch_soup(session, city_url)
    store_urls = set()

    for a in soup.find_all("a", href=True):
        full   = urljoin(BASE_URL, a["href"].strip())
        parsed = urlparse(full)
        if parsed.netloc != urlparse(BASE_URL).netloc:
            continue
        if is_store_path(parsed.path):
            store_urls.add(
                f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}/"
            )

    return sorted(store_urls)


def find_phone(soup):
    tel = soup.find("a", href=re.compile(r"^tel:"))
    if tel:
        phone = clean_text(tel.get_text(" ", strip=True))
        return phone if phone else tel.get("href", "").replace("tel:", "").strip()
    match = re.search(
        r"\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}",
        clean_text(soup.get_text(" ", strip=True))
    )
    return match.group(0) if match else ""


def parse_store(session, store_url):
    soup  = fetch_soup(session, store_url)
    h1    = soup.find("h1")
    name  = clean_text(h1.get_text(" ", strip=True)) if h1 else ""
    lines = [clean_text(t) for t in soup.stripped_strings if clean_text(t)]

    street = city = state = zip_code = ""
    csz    = re.compile(r"^(.*?),\s*([A-Z]{2})\s+(\d{5})(?:\s+US)?$")
    st_pat = re.compile(r"^\d+\s+.+")

    for i, line in enumerate(lines):
        if not street and st_pat.match(line):
            street = line
            if i + 1 < len(lines):
                m = csz.match(lines[i + 1])
                if m:
                    city, state, zip_code = m.group(1), m.group(2), m.group(3)
                    break

    if not city:
        for line in lines:
            m = csz.match(line)
            if m:
                city, state, zip_code = m.group(1), m.group(2), m.group(3)
                break

    return {
        "store_name": name,
        "street":     street,
        "city":       city,
        "state":      state,
        "zip_code":   zip_code,
        "phone":      find_phone(soup),
        "store_url":  store_url,
    }


# --- Save ---

def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["store_name", "street", "city", "state",
                  "zip_code", "phone", "store_url"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# --- Run ---

if __name__ == "__main__":
    session = build_session()

    print("Fetching city pages...")
    city_urls = get_city_urls(session)
    print(f"Found {len(city_urls)} cities.")

    all_store_urls = set()
    for i, city_url in enumerate(city_urls, 1):
        print(f"[{i}/{len(city_urls)}] {city_url}")
        try:
            urls = get_store_urls(session, city_url)
            all_store_urls.update(urls)
        except Exception as e:
            print(f"  Skipped: {e}")

    print(f"\nFound {len(all_store_urls)} store URLs.")

    rows = []
    for i, store_url in enumerate(sorted(all_store_urls), 1):
        print(f"[{i}/{len(all_store_urls)}] {store_url}")
        try:
            rows.append(parse_store(session, store_url))
        except Exception as e:
            print(f"  Skipped: {e}")

    rows.sort(key=lambda r: (r["city"], r["store_name"]))
    write_csv(rows, OUTPUT_CSV)
    print(f"\nSaved {len(rows)} stores to {OUTPUT_CSV}")