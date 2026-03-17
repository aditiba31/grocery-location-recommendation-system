import pandas as pd
import joblib

test     = pd.read_csv('data/processed/test_features.csv')
tj_test  = pd.read_csv('data/trader_joes/tj_test.csv')
model    = joblib.load('reports/models/best_model.pkl')

tj_test["zip_code"] = tj_test["zip_code"].astype(str).str.zfill(5)
test["zip_code"]    = test["zip_code"].astype(str).str.zfill(5)

FEATURE_COLS = [
    "competitor_count", "avg_competitor_rating", "market_saturation_score",
    "opportunity_score", "avg_price_tier", "total_reviews", "total_population",
    "median_age", "median_household_income", "per_capita_income",
    "pct_bachelors_plus", "poverty_rate", "diversity_index", "pct_hispanic",
    "median_gross_rent", "median_home_value", "housing_occupancy_rate",
    "unemployment_rate", "income_rent_ratio", "pct_transit_commuters",
    "total_households",
]

X     = test[FEATURE_COLS].fillna(test[FEATURE_COLS].median())
probs = model.predict_proba(X)[:, 1]

test["tj_probability"] = probs.round(4)

# Check scores for actual TJ test locations
tj_zips    = set(tj_test["zip_code"].tolist())
tj_scores  = test[test["zip_code"].isin(tj_zips)][["zip_code","has_tj","tj_probability"]]
tj_scores  = tj_scores.sort_values("tj_probability", ascending=False)

print(f"TJ test locations and their predicted probabilities:\n")
print(tj_scores.to_string(index=False))
print(f"\nMean probability for real TJ zips : {tj_scores['tj_probability'].mean():.4f}")
print(f"Min probability                   : {tj_scores['tj_probability'].min():.4f}")
print(f"Max probability                   : {tj_scores['tj_probability'].max():.4f}")