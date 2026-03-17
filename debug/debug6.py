import pandas as pd

biz = pd.read_csv('data/yelp/businesses_raw.csv')
print("Biz columns:", biz.columns.tolist())
print("Zip sample:", biz["zip_code"].head(10).tolist())
print("Total businesses:", len(biz))

reviews = pd.read_csv('data/yelp/reviews_raw.csv')
print("\nReview columns:", reviews.columns.tolist())
print("Total reviews:", len(reviews))