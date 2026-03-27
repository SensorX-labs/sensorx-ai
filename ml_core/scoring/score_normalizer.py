import pandas as pd


class ScoreNormalizer:
    """MinMax normalize revenue về [0, 1]. Cần fit từ toàn bộ sales pool trước."""

    def __init__(self) -> None:
        self._max_revenue: float = 1.0
        self._fitted: bool = False

    def fit(self, sales_df: pd.DataFrame) -> "ScoreNormalizer":
        """Fit max_revenue từ cột total_revenue_last_90d."""
        self._max_revenue = float(sales_df["total_revenue_last_90d"].max()) or 1.0
        self._fitted = True
        return self

    def normalize_revenue(self, value: float) -> float:
        self._check_fitted()
        return round(min(value / self._max_revenue, 1.0), 4)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("ScoreNormalizer chưa được fit. Gọi .fit(sales_df) trước.")