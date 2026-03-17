import joblib

model = joblib.load('reports/models/best_model.pkl')
print(f"Best model type : {type(model)}")
print(f"Model steps     : {model.steps}")