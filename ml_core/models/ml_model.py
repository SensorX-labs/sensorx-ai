import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRanker
from ml_core.config.weight_config import FEATURE_COLS, RANKER_PARAMS


class LeadRankerModel:
    """
    XGBRanker wrapper cho bài toán Learning-to-Rank.

    Mỗi lead là 1 query, mỗi sales là 1 document. Model tối ưu thứ tự
    xếp hạng trong cùng một group (lead) thay vì phân loại từng row độc lập.

    groups format: [5, 5, 5, ...] — số sales trong mỗi lead query.
    """

    def __init__(self) -> None:
        self._model = XGBRanker(**RANKER_PARAMS)
        self._trained = False

    def train(
        self,
        X: pd.DataFrame,
        y: np.ndarray | list,
        groups: np.ndarray | list,
    ) -> "LeadRankerModel":
        """
        Train XGBRanker.
        X: feature matrix (columns phải khớp FEATURE_COLS)
        y: relevance labels (0 = lost, 1 = won)
        groups: số sales trong mỗi lead, ví dụ [5, 5, 5, ...]
        """
        X = X[FEATURE_COLS]
        self._model.fit(X, y, qid=self._groups_to_qid(groups))
        self._trained = True
        return self

    def rank(self, X: pd.DataFrame) -> np.ndarray:
        """Trả về ranking scores — score cao hơn = phù hợp hơn."""
        self._check_trained()
        return self._model.predict(X[FEATURE_COLS])

    def rank_sales(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Nhận feature_matrix (index = sales_id).
        Trả về DataFrame sort theo score giảm dần, có cột 'rank' và 'score'.
        """
        scores = self.rank(feature_matrix)
        result = feature_matrix.copy()
        result["score"] = scores
        result = result.sort_values("score", ascending=False).reset_index()
        result.insert(0, "rank", range(1, len(result) + 1))
        return result

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"✅ Model saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "LeadRankerModel":
        with open(path, "rb") as f:
            instance = pickle.load(f)
        if not isinstance(instance, cls):
            raise TypeError(f"File không chứa LeadRankerModel: {path}")
        return instance

    @staticmethod
    def _groups_to_qid(groups: list[int]) -> np.ndarray:
        """Chuyển [5, 5, 5] → qid [0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, ...]"""
        qid = []
        for i, g in enumerate(groups):
            qid.extend([i] * g)
        return np.array(qid)

    def _check_trained(self) -> None:
        if not self._trained:
            raise RuntimeError("Model chưa được train. Hãy gọi .train() hoặc .load() trước.")

    @property
    def feature_importances(self) -> pd.Series:
        self._check_trained()
        return pd.Series(
            self._model.feature_importances_,
            index=FEATURE_COLS,
        ).sort_values(ascending=False)
