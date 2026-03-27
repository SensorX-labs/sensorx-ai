def deal_size_match(quote_value: float, avg_deal_size: float) -> float:
    """1 - |quote - avg| / max(quote, avg)"""
    denominator = max(quote_value, avg_deal_size)
    if denominator == 0:
        return 1.0
    return 1.0 - abs(quote_value - avg_deal_size) / denominator


def normalize(value: float, max_value: float, default: float = 0.0) -> float:
    """Min-max normalize về [0, 1], assumes min = 0."""
    if max_value == 0:
        return default
    return min(value / max_value, 1.0)


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Chia an toàn, trả về default nếu denominator == 0."""
    if denominator == 0:
        return default
    return numerator / denominator


def clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp giá trị vào khoảng [low, high]."""
    return max(low, min(high, value))
