from feast import FeatureStore

store = FeatureStore(repo_path="feature_repo")
training_df = store.get_historical_features(...)   # point-in-time-correct training set
online = store.get_online_features(...)            # latest values for serving
