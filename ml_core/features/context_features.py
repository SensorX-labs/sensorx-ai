import pandas as pd
from ml_core.features.lead_features import LeadContextExtractor
from ml_core.features.sales_features import SalesFeatureExtractor
from ml_core.utils.math_utils import deal_size_match
from ml_core.config.weight_config import FEATURE_COLS


class PairFeatureBuilder:
    """
    Tính feature vector cho từng cặp (lead, sales_id).
    Nhận SalesFeatureExtractor và customer_history để tính các features
    phụ thuộc vào mối quan hệ giữa lead và sales.
    """

    def __init__(
        self,
        sales_extractor: SalesFeatureExtractor,
        customer_history: dict[str, list[str]],
    ) -> None:
        self._sales = sales_extractor
        self._history = customer_history

    def build(self, lead: LeadContextExtractor, sales_id: str) -> dict:
        """Trả về dict chứa đầy đủ 6 features cho 1 cặp (lead, sales)."""
        return {
            "conversion_rate":   self._sales.get_conversion_rate(sales_id),
            "product_match":     self._product_match(lead, sales_id),
            "experience":        self._sales.get_experience_norm(sales_id),
            "customer_relation": self._customer_relation(lead, sales_id),
            "deal_size_match":   self._deal_size_match(lead, sales_id),
            "performance_score": self._sales.get_performance_score(sales_id),
            "active_leads":      self._sales.get_current_workload(sales_id),
        }

    def build_matrix(self, lead: LeadContextExtractor) -> pd.DataFrame:
        """
        Build feature matrix cho 1 lead × toàn bộ sales.
        Trả về DataFrame với index = sales_id, columns = FEATURE_COLS.
        """
        rows = [{"sales_id": sid, **self.build(lead, sid)} for sid in self._sales.all_sales_ids()]
        df = pd.DataFrame(rows).set_index("sales_id")
        return df[FEATURE_COLS]

    def _product_match(self, lead: LeadContextExtractor, sales_id: str) -> float:
        """
        Mức độ khớp sản phẩm:
          Tỷ lệ sản phẩm của lead mà sales có khả năng xử lý.
          score = (số sản phẩm khớp) / (tổng số sản phẩm của lead)
        """
        sales_cats = set(self._sales.get_product_categories(sales_id))
        lead_cats = set(lead.product_categories)

        if not lead_cats:
            return 0.0

        matches = lead_cats.intersection(sales_cats)
        return round(len(matches) / len(lead_cats), 2)

    def _customer_relation(self, lead: LeadContextExtractor, sales_id: str) -> int:
        """1 nếu sales từng làm việc với khách hàng này, 0 nếu chưa."""
        return 1 if sales_id in self._history.get(lead.customer_id, []) else 0

    def _deal_size_match(self, lead: LeadContextExtractor, sales_id: str) -> float:
        """1 - |quote_value - avg_deal_size| / max(quote_value, avg_deal_size)"""
        avg = self._sales.get_avg_deal_size(sales_id)
        return round(deal_size_match(lead.quote_value, avg), 4)
