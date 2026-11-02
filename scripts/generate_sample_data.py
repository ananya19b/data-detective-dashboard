from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "sample_data"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(42)

# Dataset 1: realistic e-commerce orders with cross-column relationships.
n = 1200
dates = pd.date_range("2026-01-01", periods=n, freq="3h")
products = rng.choice(["Notebook", "Headphones", "Monitor", "Keyboard", "Webcam", "Office Chair"], n, p=[0.20, 0.19, 0.14, 0.19, 0.16, 0.12])
base_prices = {"Notebook": 8, "Headphones": 55, "Monitor": 220, "Keyboard": 45, "Webcam": 70, "Office Chair": 180}
channels = rng.choice(["Website", "Mobile App", "Marketplace", "Corporate"], n, p=[0.45, 0.30, 0.15, 0.10])
segments = rng.choice(["New", "Regular", "Loyal", "Enterprise"], n, p=[0.20, 0.45, 0.27, 0.08])
retail = pd.DataFrame(
    {
        "transaction_id": [f"TX-{i:05d}" for i in range(1, n + 1)],
        "date": dates,
        "product": products,
        "region": rng.choice(["North", "South", "East", "West", "Central"], n),
        "sales_channel": channels,
        "customer_segment": segments,
        "payment_method": rng.choice(["Card", "UPI", "Wallet", "Bank Transfer", "COD"], n, p=[0.34, 0.30, 0.12, 0.14, 0.10]),
        "quantity": np.where(segments == "Enterprise", rng.poisson(10, n) + 1, rng.poisson(2, n) + 1),
        "unit_price": np.round([base_prices[p] for p in products] * rng.lognormal(0, 0.10, n), 2),
        "discount_pct": np.round(np.clip(rng.normal(8 + (segments == "Loyal") * 4 + (segments == "Enterprise") * 7, 3, n), 0, 30), 1),
        "customer_tenure_months": rng.integers(1, 85, n),
        "delivery_days": np.maximum(1, np.round(rng.normal(4, 1.3, n))).astype(int),
        "satisfaction_score": np.round(np.clip(rng.normal(4.2, 0.55, n), 1, 5), 1),
        "returned": rng.choice(["No", "Yes"], n, p=[0.91, 0.09]),
    }
)
retail["revenue"] = np.round(retail.quantity * retail.unit_price * (1 - retail.discount_pct / 100), 2)
retail.loc[48, "quantity"] = 250                         # bulk purchase or entry error
retail.loc[119, "unit_price"] = -499.0                  # impossible price
retail.loc[207, "discount_pct"] = 96.0                  # extreme discount
retail.loc[331, "region"] = "Atlantis"                  # invalid category
retail.loc[410:424, "customer_tenure_months"] = np.nan  # missingness cluster
retail.loc[560:572, "satisfaction_score"] = np.nan      # survey outage
retail.loc[688, ["quantity", "discount_pct"]] = [75, 85] # suspicious combination
retail.loc[804, ["delivery_days", "satisfaction_score", "returned"]] = [32, 1.0, "Yes"]
retail.loc[1000:1003, "payment_method"] = "Crypto"
retail.loc[1199] = retail.loc[1198].values
retail["revenue"] = np.round(retail.quantity * retail.unit_price * (1 - retail.discount_pct / 100), 2)
retail.to_csv(OUT / "retail_transactions.csv", index=False)

# Dataset 2: daily operational incidents and time-shaped data.
n = 365
dates = pd.date_range("2025-01-01", periods=n, freq="D")
weekly = 120 * np.sin(2 * np.pi * np.arange(n) / 7)
operations = pd.DataFrame(
    {
        "date": dates,
        "orders": np.round(1000 + weekly + rng.normal(0, 55, n)).astype(int),
        "avg_latency_ms": np.round(rng.normal(210, 18, n), 1),
        "support_tickets": rng.poisson(24, n),
        "refund_rate_pct": np.round(np.clip(rng.normal(2.2, 0.45, n), 0, None), 2),
        "downtime_minutes": np.where(rng.random(n) < 0.94, 0, rng.integers(1, 12, n)),
        "traffic_source": rng.choice(["Organic", "Paid", "Partner"], n, p=[0.58, 0.32, 0.10]),
    }
)
operations.loc[72, "orders"] = 4800
operations.loc[181, "orders"] = 90
operations.loc[245, "avg_latency_ms"] = 1100
operations.loc[246, "support_tickets"] = -8
operations.loc[300, "downtime_minutes"] = 240
operations.loc[330, "traffic_source"] = "UnknownBot"
operations.loc[112:125, "refund_rate_pct"] = np.nan
operations.to_csv(OUT / "daily_operations.csv", index=False)

print(f"Created sample files in {OUT}")
