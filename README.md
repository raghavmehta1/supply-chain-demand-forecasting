# Supply Chain Demand Forecasting

A compact forecasting baseline for weekly SKU demand. The project generates reproducible synthetic supply-chain data, creates calendar and lag features, trains a gradient-boosting model with time-ordered validation, and produces inventory reorder recommendations.

## Features

- Deterministic multi-SKU sample data generation
- Lag, rolling-mean, trend, price, promotion, and calendar features
- Time-aware train/test split with no random leakage
- MAE, RMSE, and WMAPE evaluation
- Next-week forecasts and reorder quantities based on lead time and safety stock
- Automated tests for data shape, leakage controls, and non-negative forecasts

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m forecasting.pipeline
pytest -q
```

The command writes `outputs/metrics.json` and `outputs/forecast.csv`.

## Model

The baseline uses `HistGradientBoostingRegressor`, which captures nonlinear relationships between recent demand, promotions, price, trend, and seasonality. Validation always uses the most recent weeks so future rows never leak into training.

## Limitations

The bundled data is synthetic and the output is a decision-support example, not a production inventory policy. A production system should include service-level calibration, supplier constraints, stockouts/censored demand, backtesting across multiple cutoffs, drift monitoring, and business-approved ordering rules.

