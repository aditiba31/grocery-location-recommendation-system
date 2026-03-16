# Trader Joe's Location Processor
# Reads raw scraped CSV, geocodes addresses, splits 80/20 for train/test.
# Run: python src/data_collection/tj_locations.py

import os
import pandas as pd
from pathlib import Path
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# --- Config ---

INPUT_PATH   = Path("data/trader_joes/tj_locations_raw.csv")
OUTPUT_PATH  = Path("data/trader_joes/tj_locations_ca.csv")
TRAIN_PATH   = Path("data/trader_joes/tj_train.csv")
TEST_PATH    = Path("data/trader_joes/tj_test.csv")
TRAIN_RATIO  = 0.80
RANDOM_STATE = 42


# --- Load ---

def load_locations():
    df = pd.read_csv(INPUT_PATH)
    df = df[df["state"] == "CA"].reset_index(drop=True)
    df["zip_code"] = df["zip_code"].astype(str).str.zfill(5)
    return df


# --- Geocode ---

def add_coordinates(df):
    geolocator = Nominatim(user_agent="tj_location_project")
    geocode    = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    latitudes  = []
    longitudes = []

    for i, row in df.iterrows():
        print(f"  Geocoding [{i+1}/{len(df)}] {row['street']}, {row['city']}")
        query    = f"{row['street']}, {row['city']}, {row['state']} {row['zip_code']}"
        location = geocode(query)
        if location:
            latitudes.append(location.latitude)
            longitudes.append(location.longitude)
        else:
            latitudes.append(None)
            longitudes.append(None)

    df["latitude"]  = latitudes
    df["longitude"] = longitudes
    return df


# --- Split ---

def split_locations(df):
    # Deduplicate by zip code first to avoid data leakage
    df_unique  = df.drop_duplicates(subset=["zip_code"]).copy()
    df_unique  = df_unique.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    split_idx  = int(len(df_unique) * TRAIN_RATIO)
    train_zips = set(df_unique.iloc[:split_idx]["zip_code"].tolist())
    test_zips  = set(df_unique.iloc[split_idx:]["zip_code"].tolist())

    train_df = df[df["zip_code"].isin(train_zips)].copy()
    test_df  = df[df["zip_code"].isin(test_zips)].copy()
    train_df["split"] = "train"
    test_df["split"]  = "test"
    return train_df, test_df


# --- Run ---

if __name__ == "__main__":
    os.makedirs("data/trader_joes", exist_ok=True)

    print("Loading locations...")
    df = load_locations()
    print(f"Loaded {len(df)} CA locations.")

    # Only geocode if coordinates are missing
    if "latitude" not in df.columns or df["latitude"].isnull().all():
        print("Geocoding addresses (5-10 minutes)...")
        df = add_coordinates(df)
        failed = df["latitude"].isna().sum()
        if failed:
            print(f"Could not geocode {failed} addresses.")
    else:
        print("Coordinates already exist — skipping geocoding.")

    df.to_csv(OUTPUT_PATH, index=False)

    train_df, test_df = split_locations(df)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH,   index=False)

    print(f"Total locations : {len(df)}")
    print(f"Train           : {len(train_df)}")
    print(f"Test            : {len(test_df)}")