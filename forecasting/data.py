from __future__ import annotations

import numpy as np
import pandas as pd


def generate_sample_data(weeks: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=weeks, freq="W-MON")
    rows = []
    price_offsets = {"SKU-A": 2, "SKU-B": 5, "SKU-C": 1}
    for sku, base, trend, lead_time in [
        ("SKU-A", 110, 0.18, 2),
        ("SKU-B", 75, 0.08, 3),
        ("SKU-C", 42, 0.04, 4),
    ]:
        for i, date in enumerate(dates):
            promotion = int(rng.random() < 0.14)
            price = round(20 + price_offsets[sku] - promotion * 2 + rng.normal(0, 0.4), 2)
            seasonal = 12 * np.sin(2 * np.pi * i / 52)
            demand = max(0, base + trend * i + seasonal + promotion * 24 - 1.8 * (price - 20) + rng.normal(0, 7))
            rows.append(
                {
                    "date": date,
                    "sku": sku,
                    "demand": round(demand, 1),
                    "price": price,
                    "promotion": promotion,
                    "inventory_on_hand": int(max(10, demand * rng.uniform(1.2, 3.2))),
                    "lead_time_weeks": lead_time,
                }
            )
    return pd.DataFrame(rows).sort_values(["sku", "date"]).reset_index(drop=True)

