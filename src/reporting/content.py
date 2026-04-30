import json
from pathlib import Path

class ContentManager:
    """
    Manager class to handle loading and providing textual content for the dashboard.
    This enforces the 'Zero-Hardcode' rule by storing all descriptive text, insights,
    and literature data in `dashboard_content.json`.
    """
    def __init__(self, content_path: str = None):
        if content_path is None:
            # Default to research/experiments/dashboard_content.json
            project_root = Path(__file__).resolve().parent.parent.parent
            self.content_path = project_root / "research" / "experiments" / "dashboard_content.json"
        else:
            self.content_path = Path(content_path)
        
        self.data = self._load_content()

    def _load_content(self) -> dict:
        if not self.content_path.exists():
            return {"versions": {}, "global": {}}
        try:
            with open(self.content_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"versions": {}, "global": {}}

    def get_version_content(self, version: str) -> dict:
        """Get content specific to a snapshot version."""
        return self.data.get("versions", {}).get(version, {})

    def get_global_content(self) -> dict:
        """Get global content that is shared across versions (e.g. literature, general info cards)."""
        return self.data.get("global", {})
    
    # === Helpers for Overview Page ===
    def get_overview_achievements(self, version: str) -> list:
        v_data = self.get_version_content(version)
        return v_data.get("overview", {}).get("achievements", ["Đang cập nhật..."])

    def get_overview_limitations(self, version: str) -> list:
        v_data = self.get_version_content(version)
        return v_data.get("overview", {}).get("limitations", ["Đang cập nhật..."])
    
    def get_overview_experiments(self, version: str) -> list:
        v_data = self.get_version_content(version)
        return v_data.get("overview", {}).get("experiments", [])

    # === Helpers for Multi-Horizon Page ===
    def get_multi_horizon_insight(self) -> dict:
        return self.get_global_content().get("multi_horizon", {}).get("insight_no_single_best", {})
    
    def get_dm_test_data(self) -> list:
        return self.get_global_content().get("multi_horizon", {}).get("dm_test", [])

    def get_literature_intl(self) -> list:
        return self.get_global_content().get("multi_horizon", {}).get("literature_intl", [])

    def get_literature_vn(self) -> list:
        return self.get_global_content().get("multi_horizon", {}).get("literature_vn", [])

    # === Helpers for Info Cards ===
    def get_info_card_text(self, key: str, default: str = "") -> str:
        return self.get_global_content().get("info_cards", {}).get(key, default)
