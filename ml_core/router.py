from ml_core.features.lead_features import LeadContextExtractor
from ml_core.features.sales_features import SalesFeatureExtractor
from ml_core.config.weight_config import SMALL_DEAL_THRESHOLD, NEW_SALES_LEAD_THRESHOLD


def get_eligible_sales(
    lead: LeadContextExtractor,
    sales_extractor: SalesFeatureExtractor,
    customer_history: dict[str, list[str]],
) -> list[str]:
    """
    Trả về danh sách sales_id được phép nhận lead này.

    Rule: Nếu khách mới (chưa từng giao dịch) VÀ deal nhỏ (< 20tr)
          → chỉ assign cho sales mới (< 20 leads xử lý)
          → giúp phân bổ deal nhỏ/dễ cho sales chưa có kinh nghiệm

    Mọi trường hợp khác → toàn bộ sales đều eligible.
    """
    all_ids = sales_extractor.all_sales_ids()

    is_new_customer = len(customer_history.get(lead.customer_id, [])) == 0
    is_small_deal   = lead.quote_value < SMALL_DEAL_THRESHOLD

    if is_new_customer and is_small_deal:
        new_sales = [
            sid for sid in all_ids
            if sales_extractor.get_total_leads(sid) < NEW_SALES_LEAD_THRESHOLD
        ]
        return new_sales if new_sales else all_ids

    return all_ids
