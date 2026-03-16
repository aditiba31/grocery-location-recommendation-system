import sys
import time
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[2]))
from config import (
    YELP_API_KEY, YELP_DIR,
    YELP_SEARCH_RADIUS_M, YELP_REQUEST_DELAY,
    YELP_MAX_RESULTS, YELP_MAX_REVIEWS,
    COMPETITOR_CATEGORIES, COMPETITOR_BRANDS,
    CA_SEARCH_GRID
)

BASE_URL = "https://api.yelp.com/v3"
HEADERS  = {
    "Authorization": f"Bearer {YELP_API_KEY}",
    "Accept": "application/json",
}


# --- API Calls ---

def search_businesses(lat, lon, term=None, categories=None):
    params = {
        "latitude":  lat,
        "longitude": lon,
        "radius":    YELP_SEARCH_RADIUS_M,
        "limit":     YELP_MAX_RESULTS,
        "sort_by":   "rating",
    }
    if term:
        params["term"] = term
    if categories:
        params["categories"] = categories

    try:
        resp = requests.get(
            f"{BASE_URL}/businesses/search",
            headers=HEADERS,
            params=params,
            timeout=10
        )
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 429:
            time.sleep(15)
            return None
        return None
    except requests.RequestException:
        return None


def get_reviews(business_id):
    try:
        resp = requests.get(
            f"{BASE_URL}/businesses/{business_id}/reviews",
            headers=HEADERS,
            params={"limit": YELP_MAX_REVIEWS},
            timeout=10
        )
        if resp.status_code == 200:
            return resp.json().get("reviews", [])
        return []
    except requests.RequestException:
        return []


# --- Parsing ---

def parse_business(biz, search_label):
    location   = biz.get("location", {})
    coords     = biz.get("coordinates", {})
    categories = biz.get("categories", [])

    return {
        "business_id":      biz.get("id"),
        "name":             biz.get("name"),
        "rating":           biz.get("rating"),
        "review_count":     biz.get("review_count"),
        "price":            biz.get("price"),
        "is_closed":        biz.get("is_closed"),
        "latitude":         coords.get("latitude"),
        "longitude":        coords.get("longitude"),
        "address":          location.get("address1"),
        "city":             location.get("city"),
        "state":            location.get("state"),
        "zip_code":         location.get("zip_code"),
        "categories":       ", ".join([c["title"] for c in categories]),
        "category_aliases": ", ".join([c["alias"] for c in categories]),
        "phone":            biz.get("phone"),
        "distance_meters":  biz.get("distance"),
        "search_area":      search_label,
    }


def parse_reviews(reviews, business_id):
    return [{
        "business_id":  business_id,
        "review_id":    r.get("id"),
        "rating":       r.get("rating"),
        "text":         r.get("text", "").strip(),
        "time_created": r.get("time_created"),
        "user_name":    r.get("user", {}).get("name"),
    } for r in reviews]


# --- Collection ---

def collect_competitors(fetch_reviews=True):
    all_businesses = {}
    all_reviews    = []

    for lat, lon, label in tqdm(CA_SEARCH_GRID, desc="Locations"):
        for cat in COMPETITOR_CATEGORIES:
            result = search_businesses(lat, lon, categories=cat)
            if result:
                for biz in result.get("businesses", []):
                    if biz.get("location", {}).get("state") != "CA":
                        continue
                    bid = biz.get("id")
                    if bid and bid not in all_businesses:
                        all_businesses[bid] = parse_business(biz, label)
            time.sleep(YELP_REQUEST_DELAY)

        for brand in COMPETITOR_BRANDS:
            result = search_businesses(lat, lon, term=brand)
            if result:
                for biz in result.get("businesses", []):
                    if biz.get("location", {}).get("state") != "CA":
                        continue
                    bid = biz.get("id")
                    if bid and bid not in all_businesses:
                        all_businesses[bid] = parse_business(biz, label)
            time.sleep(YELP_REQUEST_DELAY)

    if fetch_reviews and all_businesses:
        for bid in tqdm(all_businesses, desc="Reviews"):
            reviews = get_reviews(bid)
            if reviews:
                all_reviews.extend(parse_reviews(reviews, bid))
            time.sleep(YELP_REQUEST_DELAY)

    businesses_df = pd.DataFrame(list(all_businesses.values()))
    reviews_df    = pd.DataFrame(all_reviews) if all_reviews else pd.DataFrame()

    return businesses_df, reviews_df


# --- Feature Engineering ---

def compute_zip_features(businesses_df):
    if businesses_df.empty:
        return pd.DataFrame()

    df = businesses_df.copy()
    df = df[df["state"] == "CA"].dropna(subset=["zip_code", "rating"])

    price_map       = {"$": 1, "$$": 2, "$$$": 3, "$$$$": 4}
    df["price_num"] = df["price"].map(price_map)

    agg = df.groupby("zip_code").agg(
        competitor_count         = ("business_id", "count"),
        avg_competitor_rating    = ("rating", "mean"),
        median_competitor_rating = ("rating", "median"),
        avg_review_count         = ("review_count", "mean"),
        total_reviews            = ("review_count", "sum"),
        avg_price_tier           = ("price_num", "mean"),
        open_count               = ("is_closed", lambda x: (~x).sum()),
    ).reset_index()

    agg["market_saturation_score"] = (
        agg["competitor_count"] * agg["avg_competitor_rating"] / 5.0
    ).round(3)

    agg["opportunity_score"] = (
        1 / (agg["market_saturation_score"] + 0.1)
    ).round(3)

    return agg


def compute_sentiment(reviews_df):
    if reviews_df.empty:
        return pd.DataFrame()

    POSITIVE = ["fresh", "organic", "clean", "friendly", "quality",
                "great", "love", "excellent", "healthy", "variety", "affordable"]
    NEGATIVE = ["expensive", "crowded", "poor", "bad", "dirty",
                "slow", "overpriced", "rude", "limited"]

    def score(text):
        if not isinstance(text, str):
            return 0
        t = text.lower()
        return sum(1 for w in POSITIVE if w in t) - sum(1 for w in NEGATIVE if w in t)

    reviews_df = reviews_df.copy()
    reviews_df["sentiment_score"] = reviews_df["text"].apply(score)
    return reviews_df


# --- Save ---

def save_outputs(businesses_df, reviews_df, zip_features_df, reviews_sentiment_df):
    YELP_DIR.mkdir(parents=True, exist_ok=True)

    if not businesses_df.empty:
        businesses_df.to_csv(YELP_DIR / "businesses_raw.csv", index=False)
    if not reviews_df.empty:
        reviews_df.to_csv(YELP_DIR / "reviews_raw.csv", index=False)
    if not zip_features_df.empty:
        zip_features_df.to_csv(YELP_DIR / "zip_features.csv", index=False)
    if not reviews_sentiment_df.empty:
        reviews_sentiment_df.to_csv(YELP_DIR / "reviews_with_sentiment.csv", index=False)


# --- Run ---

if __name__ == "__main__":
    if not YELP_API_KEY:
        print("ERROR: YELP_API_KEY not found. Check your .env file.")
        sys.exit(1)

    businesses_df, reviews_df = collect_competitors(fetch_reviews=True)
    zip_features_df           = compute_zip_features(businesses_df)
    reviews_sentiment_df      = compute_sentiment(reviews_df)

    save_outputs(businesses_df, reviews_df, zip_features_df, reviews_sentiment_df)