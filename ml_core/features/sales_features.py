import pandas as pd
from ml_core.config.weight_config import MAX_YEARS_EXPERIENCE
from ml_core.utils.math_utils import safe_div, normalize
from ml_core.scoring.score_formula import compute_performance_score
from ml_core.scoring.score_normalizer import ScoreNormalizer


class SalesFeatureExtractor:
    """
    Tính các features độc lập của sales từ profile (không phụ thuộc vào lead).

    Usage:
        normalizer = ScoreNormalizer().fit(sales_df)
        extractor = SalesFeatureExtractor(sales_df, normalizer)
    """

    def __init__(self, sales_df: pd.DataFrame, normalizer: ScoreNormalizer) -> None:
        self._df = sales_df.set_index("sales_id")
        self._normalizer = normalizer
        # Parse strong_categories từ chuỗi CSV sang list
        self._categories: dict[str, list[str]] = {}
        for sid, row in self._df.iterrows():
            raw = row.get("strong_categories", "")
            if isinstance(raw, str):
                self._categories[sid] = [c.strip() for c in raw.split(",") if c.strip()]
            elif isinstance(raw, list):
                self._categories[sid] = raw 
            else:
                self._categories[sid] = []

    def get_conversion_rate(self, sales_id: str) -> float:
        row = self._get(sales_id)
        return round(safe_div(row["won_leads_last_90d"], row["total_leads_last_90d"]), 4)

    def get_experience_norm(self, sales_id: str) -> float:
        row = self._get(sales_id)
        return round(normalize(row["years_experience"], MAX_YEARS_EXPERIENCE), 4)

    def get_performance_score(self, sales_id: str) -> float:
        row = self._get(sales_id)
        norm_rev = self._normalizer.normalize_revenue(row["total_revenue_last_90d"])
        win_rate = safe_div(row["won_leads_last_90d"], row["total_leads_last_90d"])
        return compute_performance_score(win_rate, norm_rev)

    def get_avg_deal_size(self, sales_id: str) -> float:
        return float(self._get(sales_id)["avg_deal_size"])

    def get_total_leads(self, sales_id: str) -> int:
        """Tổng leads đã xử lý trong 90 ngày — dùng để nhận diện sales mới."""
        return int(self._get(sales_id)["total_leads_last_90d"])

    def get_current_workload(self, sales_id: str) -> int:
        """Số lead đang xử lý hiện tại — dùng để cân bằng phân bổ."""
        return int(self._get(sales_id).get("current_active_leads", 0))

    def get_product_categories(self, sales_id: str) -> list[str]:
        return self._categories.get(sales_id, [])

    def all_sales_ids(self) -> list[str]:
        return list(self._df.index)

    def _get(self, sales_id: str) -> pd.Series:
        if sales_id not in self._df.index:
            raise KeyError(f"sales_id '{sales_id}' không tồn tại trong profiles.")
        return self._df.loc[sales_id]
