import pandas as pd

test  = pd.read_csv('data/trader_joes/tj_test.csv')
train = pd.read_csv('data/trader_joes/tj_train.csv')

test["zip_code"]  = test["zip_code"].astype(str).str.zfill(5)
train["zip_code"] = train["zip_code"].astype(str).str.zfill(5)

top_zips = ["92101", "91505", "95014", "92618", "90066",
            "90025", "92122", "95051", "92126", "94086"]

print("Checking top 10 recommended zips against train and test sets:\n")
print(f"{'Zip':<10} {'In Train?':<12} {'In Test?'}")
print("-" * 35)

for z in top_zips:
    in_train = z in train["zip_code"].values
    in_test  = z in test["zip_code"].values
    print(f"{z:<10} {str(in_train):<12} {str(in_test)}")