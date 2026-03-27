# lấy thông tin của lead chuyển hóa thành dạng class
class LeadContextExtractor:
    def __init__(self, lead_dict: dict) -> None:
        self._data = lead_dict

    @property
    def lead_id(self) -> str:
        return str(self._data["lead_id"])

    @property
    def customer_id(self) -> str:
        return str(self._data["customer_id"])

    @property
    def product_categories(self) -> list[str]:
        raw = self._data.get("product_categories", [])
        if isinstance(raw, str):
            return [cat.strip() for cat in raw.split(",")]
        return list(raw)

    @property
    def quote_value(self) -> float:
        return float(self._data["quote_value"])

    @property
    def lead_source(self) -> str:
        return str(self._data.get("lead_source", "Unknown"))

    def to_dict(self) -> dict:
        return {
            "lead_id":            self.lead_id,
            "customer_id":        self.customer_id,
            "product_categories": self.product_categories,
            "quote_value":        self.quote_value,
            "lead_source":        self.lead_source,
        }
