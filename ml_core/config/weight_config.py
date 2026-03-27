"""
Toàn bộ hằng số cấu hình cho hệ thống ranking.
Thay đổi giá trị ở đây để tune model mà không cần chạm code logic.
"""

# Routing rules
SMALL_DEAL_THRESHOLD: float = 10_000_000    # định nghĩa deal nhỏ < 10 triệu
NEW_SALES_LEAD_THRESHOLD: int = 10          # định nghĩa sales mới nếu đã xử lý < 10 leads

# Thứ tự phải khớp với lúc train
FEATURE_COLS = [
    "conversion_rate",
    "product_match",
    "experience",
    "customer_relation",
    "deal_size_match",
    "performance_score",
    "active_leads",
]

MAX_YEARS_EXPERIENCE: int = 12           # Năm kinh nghiệm tối đa trong hệ thống
DEAL_SIZE_BENCHMARK: float = 50_000_000  # 50 triệu VND — mức deal tham chiếu

PRODUCT_CATEGORIES = ["Sensor", "Controller", "Gateway", "Software", "Service"]

# performance_score = 0.6 * win_rate + 0.4 * normalized_revenue
PERF_WEIGHTS = {
    "win_rate": 0.6,
    "revenue":  0.4,
}

RANKER_PARAMS = {
    "objective":        "rank:ndcg",
    "n_estimators":     300,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "random_state":     42,
    "verbosity":        0,
}