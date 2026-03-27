import json
import pandas as pd
from pathlib import Path

from ml_core.scoring.score_normalizer import ScoreNormalizer
from ml_core.features.sales_features import SalesFeatureExtractor
from ml_core.features.lead_features import LeadContextExtractor
from ml_core.features.context_features import PairFeatureBuilder
from ml_core.models.ml_model import LeadRankerModel
from ml_core.router import get_eligible_sales

DATA_DIR  = Path("data")
MODEL_DIR = Path("models")

sales_df     = pd.read_excel(DATA_DIR / "sales_profiles.xlsx")
normalizer   = ScoreNormalizer().fit(sales_df)
sales_ext    = SalesFeatureExtractor(sales_df, normalizer)

# Lịch sử quan hệ khách hàng — load từ file đã generate
with open(DATA_DIR / "customer_history.json") as f:
    CUSTOMER_HISTORY: dict[str, list[str]] = json.load(f)

pair_builder = PairFeatureBuilder(sales_ext, CUSTOMER_HISTORY)
model        = LeadRankerModel.load(MODEL_DIR / "xgb_ranker.pkl")


def rank_lead(lead_dict: dict) -> pd.DataFrame:
    """
    Nhận thông tin lead → trả về DataFrame sales được xếp hạng bởi XGBRanker.
    (ML model đã tự học feature active_leads để cân bằng tải)
    """
    lead           = LeadContextExtractor(lead_dict)
    eligible_ids   = get_eligible_sales(lead, sales_ext, CUSTOMER_HISTORY)
    feature_matrix = pair_builder.build_matrix(lead).loc[eligible_ids]
    result         = model.rank_sales(feature_matrix)

    name_map = sales_df.set_index("sales_id")["sales_name"].to_dict()
    result.insert(2, "sales_name", result["sales_id"].map(name_map))
    return result


def print_ranking(lead_dict: dict, result: pd.DataFrame) -> None:
    lead = LeadContextExtractor(lead_dict)
    print(f"\n{'─'*66}")
    print(f"  Lead      : {lead.lead_id}")
    print(f"  Customer  : {lead.customer_id}")
    print(f"  Products  : {', '.join(lead.product_categories)}")
    print(f"  Quote     : {lead.quote_value:>15,.0f} VND")
    print(f"{'─'*66}")
    cols = ["rank", "sales_name", "score", "product_match", "customer_relation",
            "deal_size_match", "performance_score", "active_leads"]
    print(result[cols].to_string(index=False))


if __name__ == "__main__":
    print("=" * 62)
    print("   LEAD → SALES RANKING SYSTEM  |  Multi-Product Support")
    print("=" * 62)

    # Case 1: Đa sản phẩm, khách quen
    print_ranking(
        lead_1 := {"lead_id": "L_NEW_001", "customer_id": "C001",
                   "product_categories": ["Sensor", "Software"], "quote_value": 80_000_000},
        rank_lead(lead_1),
    )

    # Case 2: Deal nhỏ, khách mới, sản phẩm đơn lẻ
    print_ranking(
        lead_2 := {"lead_id": "L_NEW_002", "customer_id": "C099",
                   "product_categories": ["Software"], "quote_value": 8_000_000},
        rank_lead(lead_2),
    )

    # Case 3: Gateway & Service, deal trung bình
    print_ranking(
        lead_3 := {"lead_id": "L_NEW_003", "customer_id": "C010",
                   "product_categories": ["Gateway", "Service"], "quote_value": 45_000_000},
        rank_lead(lead_3),
    )

    print(f"\n{'='*62}\n  Done.")
