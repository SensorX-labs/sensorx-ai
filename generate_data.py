"""
Tạo synthetic dataset: sales_profiles.xlsx, leads.xlsx, training_data.xlsx
Output: 100 leads × 5 sales = 500 rows với đầy đủ features + label
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from ml_core.config.weight_config import PRODUCT_CATEGORIES
from ml_core.scoring.score_normalizer import ScoreNormalizer
from ml_core.features.sales_features import SalesFeatureExtractor
from ml_core.features.lead_features import LeadContextExtractor
from ml_core.features.context_features import PairFeatureBuilder

np.random.seed(42)
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

# tạo dữ liệu sales
sales_raw = [
    
    {"sales_id": "S01", "sales_name": "Nguyen Van An",  "years_experience": 8,
     "strong_categories": "Sensor, Gateway",
     "won_leads_last_90d": 18, "total_leads_last_90d": 25,
     "total_revenue_last_90d": 450_000_000, "avg_deal_size": 35_000_000,
     "current_active_leads": 5},

    {"sales_id": "S02", "sales_name": "Tran Thi Binh", "years_experience": 3,
     "strong_categories": "Controller, Software",
     "won_leads_last_90d": 8,  "total_leads_last_90d": 20,
     "total_revenue_last_90d": 160_000_000, "avg_deal_size": 20_000_000,
     "current_active_leads": 2},

    {"sales_id": "S03", "sales_name": "Le Van Cuong",  "years_experience": 12,
     "strong_categories": "Sensor, Controller, Gateway, Software, Service",
     "won_leads_last_90d": 25, "total_leads_last_90d": 30,
     "total_revenue_last_90d": 750_000_000, "avg_deal_size": 55_000_000,
     "current_active_leads": 8},

    {"sales_id": "S04", "sales_name": "Pham Thi Dung", "years_experience": 5,
     "strong_categories": "Software, Service",
     "won_leads_last_90d": 12, "total_leads_last_90d": 22,
     "total_revenue_last_90d": 280_000_000, "avg_deal_size": 42_000_000,
     "current_active_leads": 3},

    {"sales_id": "S05", "sales_name": "Hoang Van Em",  "years_experience": 1,
     "strong_categories": "Controller",
     "won_leads_last_90d": 3,  "total_leads_last_90d": 10,
     "total_revenue_last_90d": 45_000_000,  "avg_deal_size": 15_000_000,
     "current_active_leads": 1},

    {"sales_id": "S06", "sales_name": "Vu Van Phong",  "years_experience": 2,
     "strong_categories": "Sensor",
     "won_leads_last_90d": 5, "total_leads_last_90d": 15,
     "total_revenue_last_90d": 80_000_000, "avg_deal_size": 16_000_000,
     "current_active_leads": 12},

    {"sales_id": "S07", "sales_name": "Ngo Thi Quynh", "years_experience": 6,
     "strong_categories": "Gateway, Software",
     "won_leads_last_90d": 14,  "total_leads_last_90d": 25,
     "total_revenue_last_90d": 300_000_000, "avg_deal_size": 21_000_000,
     "current_active_leads": 6},

    {"sales_id": "S08", "sales_name": "Bui Van Tuan",  "years_experience": 10,
     "strong_categories": "Controller, Service",
     "won_leads_last_90d": 20, "total_leads_last_90d": 28,
     "total_revenue_last_90d": 500_000_000, "avg_deal_size": 25_000_000,
     "current_active_leads": 2},

    {"sales_id": "S09", "sales_name": "Do Thi Uyen", "years_experience": 4,
     "strong_categories": "Sensor, Gateway",
     "won_leads_last_90d": 10, "total_leads_last_90d": 18,
     "total_revenue_last_90d": 220_000_000, "avg_deal_size": 22_000_000,
     "current_active_leads": 7},

    {"sales_id": "S10", "sales_name": "Ly Van Vinh",  "years_experience": 7,
     "strong_categories": "Software, Service",
     "won_leads_last_90d": 16,  "total_leads_last_90d": 24,
     "total_revenue_last_90d": 400_000_000,  "avg_deal_size": 25_000_000,
     "current_active_leads": 3},
]

sales_df = pd.DataFrame(sales_raw)
sales_df.to_excel(OUTPUT_DIR / "sales_profiles.xlsx", index=False)

# tạo dữ liệu yêu cầu báo giá
CUSTOMER_POOL = [f"C{i:04d}" for i in range(1, 501)]
leads_raw = []
for i in range(1, 5001):
    quote_value = int(np.clip(np.random.lognormal(17.5, 0.6), 5_000_000, 300_000_000))
    
    # Sinh 1-3 sản phẩm ngẫu nhiên
    num_cats = np.random.randint(1, 4)
    lead_cats = np.random.choice(PRODUCT_CATEGORIES, size=num_cats, replace=False)
    
    leads_raw.append({
        "lead_id":             f"L{i:04d}",
        "customer_id":         np.random.choice(CUSTOMER_POOL),
        "product_categories":  ", ".join(lead_cats),  # Lưu dạng string để Excel dễ đọc
        "quote_value":         quote_value,
        "lead_source":         np.random.choice(["Inbound", "Outbound", "Referral"]),
    })
leads_df = pd.DataFrame(leads_raw)
leads_df.to_excel(OUTPUT_DIR / "leads.xlsx", index=False)
print(f"✅ leads.xlsx ({len(leads_df)} leads)")

# tạo dữ liệu làm việc của khách hàng - sales
all_sales_ids = sales_df["sales_id"].tolist()
customer_history: dict[str, list[str]] = {
    cust: list(np.random.choice(all_sales_ids, size=np.random.randint(0, 3), replace=False))
    for cust in CUSTOMER_POOL
}
with open(OUTPUT_DIR / "customer_history.json", "w") as f:
    json.dump(customer_history, f, indent=2)

# tạo dữ liệu training
normalizer      = ScoreNormalizer().fit(sales_df)
sales_extractor = SalesFeatureExtractor(sales_df, normalizer)
pair_builder    = PairFeatureBuilder(sales_extractor, customer_history)

rows = []
for _, lead_row in leads_df.iterrows():
    lead = LeadContextExtractor(lead_row.to_dict())
    for sid in sales_extractor.all_sales_ids():
        features = pair_builder.build(lead, sid)
        # Label dựa trên weighted sum của features + noise để tạo dữ liệu realistic
        # Phạt (trừ điểm) nếu sales đang quá tải: -0.1 cho mỗi 5 leads đang xử lý
        workload_penalty = 0.10 * (features["active_leads"] / 5.0)

        win_prob = (
            0.30 * features["conversion_rate"]
            + 0.25 * features["product_match"]
            + 0.10 * features["experience"]
            + 0.15 * features["customer_relation"]
            + 0.10 * features["deal_size_match"]
            + 0.10 * features["performance_score"]
            - workload_penalty
        )
        win_prob = float(np.clip(win_prob + np.random.normal(0, 0.05), 0, 1))
        rows.append({"lead_id": lead.lead_id, "sales_id": sid, **features,
                     "label": 1 if np.random.random() < win_prob else 0})

train_df = pd.DataFrame(rows)
train_df.to_excel(OUTPUT_DIR / "training_data.xlsx", index=False)
print(f"✅ training_data.xlsx ({len(train_df)} rows)")
print(f"\n📊 Label distribution:\n{train_df['label'].value_counts().to_string()}")
print(f"\n📋 Sample:\n{train_df.head(10).to_string(index=False)}")
