import pandas as pd

CA_ZIP_PREFIXES = tuple(str(i) for i in range(900, 967))

census = pd.read_csv('data/census/ca_demographics.csv')
train  = pd.read_csv('data/trader_joes/tj_train.csv')

census["zip"]     = census["zip"].astype(str).str.zfill(5)
train["zip_code"] = train["zip_code"].astype(str).str.zfill(5)

census_ca = census[census["zip"].str.startswith(CA_ZIP_PREFIXES)]
train_ca  = train[train["zip_code"].str.startswith(CA_ZIP_PREFIXES)]

census_zips = set(census_ca["zip"].tolist())
train_zips  = set(train_ca["zip_code"].tolist())

missing = train_zips - census_zips
print(f"TJ zips not in Census : {len(missing)}")
print(missing)

# Also check for duplicate zip codes in census
dupes = census_ca[census_ca["zip"].duplicated()]
print(f"\nDuplicate zips in Census : {len(dupes)}")

# Check label counts after merge
merged = census_ca.merge(
    train_ca[["zip_code"]].assign(has_tj=1),
    left_on="zip", right_on="zip_code", how="left"
)
merged["has_tj"] = merged["has_tj"].fillna(0).astype(int)
print(f"\nAfter merge - TJ count : {merged['has_tj'].sum()}")
print(f"Census rows            : {len(census_ca)}")