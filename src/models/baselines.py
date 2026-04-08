from sklearn.neural_network import MLPRegressor

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


def fit_baseline_regressor(model_name: str, x_train, y_train, seed: int = 42, **kwargs):
    model_name = model_name.lower()
    if model_name == "mlp":
        hidden = kwargs.get("hidden_layer_sizes", (256, 128))
        model = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            max_iter=kwargs.get("max_iter", 500),
            random_state=seed,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.1,
        )
        model.fit(x_train, y_train)
        return model

    if model_name == "xgb":
        if not HAS_XGBOOST:
            raise ImportError("xgboost is required for model_name='xgb'")
        model = xgb.XGBRegressor(
            n_estimators=kwargs.get("n_estimators", 800),
            learning_rate=kwargs.get("learning_rate", 0.05),
            max_depth=kwargs.get("max_depth", 6),
            subsample=kwargs.get("subsample", 0.8),
            colsample_bytree=kwargs.get("colsample_bytree", 0.8),
            objective="reg:squarederror",
            random_state=seed,
        )
        model.fit(x_train, y_train)
        return model

    raise ValueError(f"Unsupported baseline model: {model_name}")
