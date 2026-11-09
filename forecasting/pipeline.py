from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import generate_sample_data


@dataclass(frozen=True)
class ForecastArtifacts:
    metrics: dict[str, float]
    forecast: pd.DataFrame
    validation: pd.DataFrame


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy().sort_values(["sku", "date"])
    frame["date"] = pd.to_datetime(frame["date"])
    grouped = frame.groupby("sku", group_keys=False)["demand"]
    frame["lag_1"] = grouped.shift(1)
    frame["lag_4"] = grouped.shift(4)
    frame["rolling_4"] = grouped.transform(lambda s: s.shift(1).rolling(4).mean())
    frame["week_sin"] = np.sin(2 * np.pi * frame["date"].dt.isocalendar().week.astype(int) / 52)
    frame["week_cos"] = np.cos(2 * np.pi * frame["date"].dt.isocalendar().week.astype(int) / 52)
    frame["trend"] = frame.groupby("sku").cumcount()
    return frame.dropna().reset_index(drop=True)


def _model() -> Pipeline:
    numeric = ["price", "promotion", "lag_1", "lag_4", "rolling_4", "week_sin", "week_cos", "trend"]
    prep = ColumnTransformer(
        [("sku", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["sku"]), ("num", "passthrough", numeric)]
    )
    return Pipeline(
        [
            ("features", prep),
            ("model", HistGradientBoostingRegressor(max_iter=180, learning_rate=0.06, max_leaf_nodes=18, l2_regularization=0.2, random_state=42)),
        ]
    )


def train_and_forecast(data: pd.DataFrame, validation_weeks: int = 12) -> ForecastArtifacts:
    frame = build_features(data)
    cutoff = frame["date"].max() - pd.Timedelta(weeks=validation_weeks - 1)
    train, valid = frame[frame["date"] < cutoff], frame[frame["date"] >= cutoff].copy()
    feature_cols = ["sku", "price", "promotion", "lag_1", "lag_4", "rolling_4", "week_sin", "week_cos", "trend"]
    model = _model().fit(train[feature_cols], train["demand"])
    valid["prediction"] = np.maximum(0, model.predict(valid[feature_cols]))
    error = valid["demand"] - valid["prediction"]
    metrics = {
        "mae": round(float(mean_absolute_error(valid["demand"], valid["prediction"])), 3),
        "rmse": round(float(mean_squared_error(valid["demand"], valid["prediction"]) ** 0.5), 3),
        "wmape": round(float(error.abs().sum() / valid["demand"].abs().sum()), 4),
    }

    latest = frame.sort_values("date").groupby("sku").tail(1).copy()
    next_date = frame["date"].max() + pd.Timedelta(weeks=1)
    latest["date"] = next_date
    latest["lag_4"] = frame.groupby("sku")["demand"].nth(-4).to_numpy()
    latest["lag_1"] = frame.groupby("sku")["demand"].last().to_numpy()
    latest["rolling_4"] = frame.groupby("sku")["demand"].tail(4).groupby(frame["sku"]).mean().to_numpy()
    latest["week_sin"] = np.sin(2 * np.pi * int(next_date.isocalendar().week) / 52)
    latest["week_cos"] = np.cos(2 * np.pi * int(next_date.isocalendar().week) / 52)
    latest["trend"] += 1
    latest["forecast_demand"] = np.maximum(0, model.predict(latest[feature_cols])).round(1)
    latest["safety_stock"] = (1.65 * error.std() * np.sqrt(latest["lead_time_weeks"])).round().astype(int)
    latest["reorder_quantity"] = np.maximum(
        0,
        latest["forecast_demand"] * latest["lead_time_weeks"] + latest["safety_stock"] - latest["inventory_on_hand"],
    ).round().astype(int)
    forecast = latest[["date", "sku", "forecast_demand", "inventory_on_hand", "lead_time_weeks", "safety_stock", "reorder_quantity"]]
    return ForecastArtifacts(metrics, forecast.reset_index(drop=True), valid.reset_index(drop=True))


def main() -> None:
    output = Path("outputs")
    output.mkdir(exist_ok=True)
    artifacts = train_and_forecast(generate_sample_data())
    (output / "metrics.json").write_text(json.dumps(artifacts.metrics, indent=2), encoding="utf-8")
    artifacts.forecast.to_csv(output / "forecast.csv", index=False)
    print(json.dumps(artifacts.metrics, indent=2))
    print(artifacts.forecast.to_string(index=False))


if __name__ == "__main__":
    main()

