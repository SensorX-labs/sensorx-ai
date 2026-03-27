from ml_core.config.weight_config import PERF_WEIGHTS
from ml_core.utils.math_utils import clip


def compute_performance_score(win_rate: float, normalized_revenue: float) -> float:
    """
    performance_score = 0.6 * win_rate + 0.4 * normalized_revenue

    win_rate:           won_leads / total_leads (90 ngày gần nhất)
    normalized_revenue: total_revenue / max_revenue trong toàn bộ sales pool
    """
    score = PERF_WEIGHTS["win_rate"] * win_rate + PERF_WEIGHTS["revenue"] * normalized_revenue
    return round(clip(score), 4)