import os
import time
import pandas as pd
import numpy as np
from census import Census
from dotenv import load_dotenv

load_dotenv()

CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")
ACS_YEAR       = 2022
OUTPUT_PATH    = "data/census/ca_demographics.csv"

ACS_VARIABLES = {
    "B01003_001E": "total_population",
    "B01002_001E": "median_age",
    "B19013_001E": "median_household_income",
    "B19301_001E": "per_capita_income",
    "B17001_002E": "population_below_poverty",
    "B15003_022E": "bachelors_degree_holders",
    "B15003_023E": "masters_degree_holders",
    "B15003_025E": "doctorate_holders",
    "B15003_001E": "education_population_base",
    "B02001_002E": "white_alone",
    "B02001_003E": "black_alone",
    "B02001_005E": "asian_alone",
    "B03001_003E": "hispanic_or_latino",
    "B25064_001E": "median_gross_rent",
    "B25077_001E": "median_home_value",
    "B25002_002E": "occupied_housing_units",
    "B25002_001E": "total_housing_units",
    "B08301_001E": "total_commuters",
    "B08301_010E": "public_transit_commuters",
    "B11001_001E": "total_households",
    "B23025_002E": "labor_force",
    "B23025_005E": "unemployed",
}


def fetch_census_data():
    c        = Census(CENSUS_API_KEY, year=ACS_YEAR)
    var_list = list(ACS_VARIABLES.keys())
    batches  = [var_list[i:i+45] for i in range(0, len(var_list), 45)]
    all_dfs  = []

    for i, batch in enumerate(batches):
        print(f"  Fetching batch {i+1}/{len(batches)}...")
        try:
            results = c.acs5.get(
                ["NAME"] + batch,
                {"for": "zip code tabulation area:*"}
            )
            all_dfs.append(pd.DataFrame(results))
            time.sleep(0.3)
        except Exception as e:
            print(f"  Batch {i+1} failed: {e}")

    df = all_dfs[0]
    for extra in all_dfs[1:]:
        df = df.merge(
            extra,
            on=["zip code tabulation area"],
            how="outer",
            suffixes=("", "_dup")
        )
        df = df.loc[:, ~df.columns.str.endswith("_dup")]

    return df


def clean_and_engineer(raw_df):
    df = raw_df.copy()

    df = df.rename(columns={"zip code tabulation area": "zip"})
    df = df.rename(columns=ACS_VARIABLES)
    df["zip"] = df["zip"].astype(str).str.zfill(5)
    df = df.drop(columns=["NAME", "NAME_x", "NAME_y"], errors="ignore")

    for col in [c for c in df.columns if c != "zip"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([-666666666, -999999999, -888888888], np.nan)
    df = df[df["zip"].str.startswith("9")].copy()
    df = df[df["total_population"] > 100].copy()

    df["pct_bachelors_plus"] = (
        (df["bachelors_degree_holders"].fillna(0) +
         df["masters_degree_holders"].fillna(0) +
         df["doctorate_holders"].fillna(0)) /
        df["education_population_base"].replace(0, np.nan)
    ).clip(0, 1)

    df["poverty_rate"] = (
        df["population_below_poverty"] /
        df["total_population"].replace(0, np.nan)
    ).clip(0, 1)

    race_cols = ["white_alone", "black_alone", "asian_alone", "hispanic_or_latino"]
    race_sum  = df[race_cols].sum(axis=1).replace(0, np.nan)
    df["diversity_index"] = (
        1 - sum((df[c] / race_sum) ** 2 for c in race_cols)
    ).clip(0, 1)

    df["pct_transit_commuters"] = (
        df["public_transit_commuters"] /
        df["total_commuters"].replace(0, np.nan)
    ).clip(0, 1)

    df["housing_occupancy_rate"] = (
        df["occupied_housing_units"] /
        df["total_housing_units"].replace(0, np.nan)
    ).clip(0, 1)

    df["pct_hispanic"] = (
        df["hispanic_or_latino"] /
        df["total_population"].replace(0, np.nan)
    ).clip(0, 1)

    df["income_rent_ratio"] = (
        df["median_household_income"] /
        (df["median_gross_rent"] * 12).replace(0, np.nan)
    )

    df["unemployment_rate"] = (
        df["unemployed"] /
        df["labor_force"].replace(0, np.nan)
    ).clip(0, 1)

    print(f"{len(df):,} California zip codes, {df.shape[1]} columns")
    return df


if __name__ == "__main__":
    os.makedirs("data/census", exist_ok=True)
    print(f"Fetching ACS {ACS_YEAR} data...")
    raw_df   = fetch_census_data()
    clean_df = clean_and_engineer(raw_df)
    clean_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved → {OUTPUT_PATH}")
    