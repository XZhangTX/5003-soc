import copy

import numpy as np
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
        x_val = kwargs.get("x_val")
        y_val = kwargs.get("y_val")
        if x_val is None or y_val is None:
            raise ValueError("MLP baseline requires x_val and y_val for validation-based early stopping")

        model = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            max_iter=1,
            random_state=seed,
            early_stopping=False,
            warm_start=True,
        )

        max_iter = kwargs.get("max_iter", 500)
        patience = kwargs.get("patience", 20)
        best_model = None
        best_rmse = float("inf")
        bad_rounds = 0

        for _ in range(max_iter):
            model.fit(x_train, y_train)
            y_pred = model.predict(x_val)
            rmse = float(np.sqrt(np.mean(np.square(y_pred - y_val))))
            if rmse < best_rmse:
                best_rmse = rmse
                bad_rounds = 0
                best_model = copy.deepcopy(model)
            else:
                bad_rounds += 1
                if bad_rounds >= patience:
                    break

        return best_model if best_model is not None else model

    if model_name == "xgb":
        if not HAS_XGBOOST:
            raise ImportError("xgboost is required for model_name='xgb'")
        x_val = kwargs.get("x_val")
        y_val = kwargs.get("y_val")
        if x_val is None or y_val is None:
            raise ValueError("XGB baseline requires x_val and y_val for validation-based early stopping")

        model = xgb.XGBRegressor(
            n_estimators=kwargs.get("n_estimators", 800),
            learning_rate=kwargs.get("learning_rate", 0.05),
            max_depth=kwargs.get("max_depth", 6),
            subsample=kwargs.get("subsample", 0.8),
            colsample_bytree=kwargs.get("colsample_bytree", 0.8),
            objective="reg:squarederror",
            random_state=seed,
            eval_metric="rmse",
            early_stopping_rounds=kwargs.get("patience", 20),
        )
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            verbose=False,
        )
        return model

    raise ValueError(f"Unsupported baseline model: {model_name}")
