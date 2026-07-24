import mlflow, mlflow.sklearn

mlflow.set_tracking_uri("sqlite:///mlruns/mlflow.db")
mlflow.set_experiment("nyc-taxi-duration")

with mlflow.start_run(run_name="rf_baseline"):
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)

    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    metrics = evaluate_model(y_test, model.predict(X_test))
    print_metrics(metrics)

    mlflow.log_metric("rmse", metrics["rmse"])
    mlflow.log_metric("r2", metrics["r2"])
    mlflow.sklearn.log_model(model, name="model")  # artifact only — register later
