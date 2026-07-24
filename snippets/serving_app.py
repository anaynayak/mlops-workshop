from mlops_workshop import registry

# on startup — same alias, still the source of truth
model = registry.load_model("champion")


@app.post("/predict")
def predict(trip):
    return model.predict(features(trip))
