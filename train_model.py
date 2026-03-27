import numpy as np
import pandas as pd
from pathlib import Path

from ml_core.config.weight_config import FEATURE_COLS
from ml_core.models.ml_model import LeadRankerModel

DATA_DIR  = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

print("📂 Loading training_data.xlsx...")
df = pd.read_excel(DATA_DIR / "training_data.xlsx")
print(f"   {len(df)} rows | Label dist: {df['label'].value_counts().to_dict()}")

# XGBRanker yêu cầu data sort theo lead_id và groups = số sales trong mỗi lead
df     = df.sort_values("lead_id").reset_index(drop=True)
X      = df[FEATURE_COLS]
y      = df["label"].values
groups = df.groupby("lead_id", sort=False).size().tolist()

print(f"\n🔢 Leads: {len(groups)} | Sales/lead: {groups[0]} | Total: {sum(groups)}")

print("\n🚀 Training XGBRanker...")
model = LeadRankerModel()
model.train(X, y, groups)

print("\n🏆 Feature Importance:")
for feat, score in model.feature_importances.items():
    print(f"   {feat:<22} {'█' * int(score * 40)} {score:.4f}")

model.save(MODEL_DIR / "xgb_ranker.pkl")
print(f"\n✅ Done. Model saved → models/xgb_ranker.pkl")
