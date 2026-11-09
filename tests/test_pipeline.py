from forecasting.data import generate_sample_data
from forecasting.pipeline import build_features, train_and_forecast


def test_feature_lags_use_prior_demand():
    raw = generate_sample_data(weeks=30)
    features = build_features(raw)
    sku_a = features[features["sku"] == "SKU-A"].iloc[0]
    source = raw[raw["sku"] == "SKU-A"].reset_index(drop=True)
    assert sku_a["lag_1"] == source.iloc[3]["demand"]
    assert sku_a["lag_4"] == source.iloc[0]["demand"]


def test_pipeline_returns_metrics_and_non_negative_forecasts():
    artifacts = train_and_forecast(generate_sample_data(weeks=70), validation_weeks=8)
    assert set(artifacts.metrics) == {"mae", "rmse", "wmape"}
    assert len(artifacts.forecast) == 3
    assert (artifacts.forecast["forecast_demand"] >= 0).all()
    assert (artifacts.forecast["reorder_quantity"] >= 0).all()

