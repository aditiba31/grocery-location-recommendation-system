# Trader Joe's Location Processor
# Reads raw scraped CSV, geocodes addresses, splits 80/20 for train/test.
# Run: python src/data_collection/tj_locations.py

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
    df        = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    split_idx = int(len(df) * TRAIN_RATIO)
    train_df  = df.iloc[:split_idx].copy()
    test_df   = df.iloc[split_idx:].copy()
    train_df["split"] = "train"
    test_df["split"]  = "test"
    return train_df, test_df


# --- Run ---

if __name__ == "__main__":
    print("Loading locations...")
    df = load_locations()
    print(f"Loaded {len(df)} CA locations.")

    print("Geocoding addresses")
    df = add_coordinates(df)

    failed = df["latitude"].isna().sum()
    if failed:
        print(f"Could not geocode {failed} addresses.")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} locations to {OUTPUT_PATH}")

    train_df, test_df = split_locations(df)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH,   index=False)

    print(f"Train : {len(train_df)}")
    print(f"Test  : {len(test_df)}")