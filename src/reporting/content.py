import json
from pathlib import Path

class ContentManager:
    """
    Manager class to handle loading and providing textual content for the dashboard.
    This enforces the 'Zero-Hardcode' rule by storing all descriptive text, insights,
    and literature data in `dashboard_content.json`.

    Data access follows a 3-tier fallback pattern:
        1. API (PostgreSQL via FastAPI) — primary source in Docker
        2. JSON export files (db_export/) — fallback for local dev
        3. Default string — last resort
    """
    def __init__(self, content_path: str = None):
        self._project_root = Path(__file__).resolve().parent.parent.parent
        if content_path is None:
            # Default to research/experiments/dashboard_content.json
            self.content_path = self._project_root / "research" / "experiments" / "dashboard_content.json"
        else:
            self.content_path = Path(content_path)
        
        self.data = self._load_content()
        self._info_cards_cache: dict | None = None

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
    def _load_info_cards_json(self) -> dict:
        """Lazy-load info cards from JSON export file (Tier 2 fallback).

        Caches the result so file is only read once per ContentManager instance.
        """
        if self._info_cards_cache is not None:
            return self._info_cards_cache

        json_path = self._project_root / "research" / "experiments" / "db_export" / "info_cards.json"
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    self._info_cards_cache = json.load(f)
                    return self._info_cards_cache
            except Exception:
                pass

        self._info_cards_cache = {}
        return self._info_cards_cache

    def get_info_card_text(self, key: str, default: str = "") -> str:
        """Get info card content with 3-tier fallback.

        Tier 1: API (PostgreSQL via FastAPI) — works in Docker
        Tier 2: JSON export file (db_export/info_cards.json) — works on local dev
        Tier 3: Default string — last resort
        """
        # Tier 1: API (PostgreSQL)
        try:
            from src.frontend.api_client import APIClient
            client = APIClient()
            result = client.get_info_card(key, quiet=True)
            if isinstance(result, dict) and "content" in result:
                return result["content"]
        except Exception:
            pass

        # Tier 2: JSON export file
        cards_json = self._load_info_cards_json()
        if key in cards_json:
            content = cards_json[key].get("content", "")
            if content:
                return content

        # Tier 3: Default string
        return default

