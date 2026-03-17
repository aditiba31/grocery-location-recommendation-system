import pandas as pd

train_tj = pd.read_csv('data/trader_joes/tj_train.csv')
test_tj  = pd.read_csv('data/trader_joes/tj_test.csv')
train_f  = pd.read_csv('data/processed/train_features.csv')
test_f   = pd.read_csv('data/processed/test_features.csv')

train_tj["zip_code"] = train_tj["zip_code"].astype(str).str.zfill(5)
test_tj["zip_code"]  = test_tj["zip_code"].astype(str).str.zfill(5)

# Check 1: No overlap between train and test TJ locations
overlap = set(train_tj["zip_code"]) & set(test_tj["zip_code"])
print(f"TJ zip overlap between train and test : {len(overlap)}")
print(f"Expected                              : 0")

# Check 2: Label counts are correct
print(f"\nTrain feature matrix - has_tj = 1 : {train_f['has_tj'].sum()}")
print(f"TJ train locations                 : {len(train_tj)}")

print(f"\nTest feature matrix - has_tj = 1  : {test_f['has_tj'].sum()}")
print(f"TJ test locations                  : {len(test_tj)}")

# Check 3: No nulls in feature matrices
print(f"\nTrain feature matrix nulls : {train_f.isnull().sum().sum()}")
print(f"Test feature matrix nulls  : {test_f.isnull().sum().sum()}")

# Check 4: Same zip codes in both feature matrices
train_zips = set(train_f["zip_code"].astype(str).tolist())
test_zips  = set(test_f["zip_code"].astype(str).tolist())
print(f"\nTrain zip codes  : {len(train_zips)}")
print(f"Test zip codes   : {len(test_zips)}")
print(f"Same zip codes   : {train_zips == test_zips}")