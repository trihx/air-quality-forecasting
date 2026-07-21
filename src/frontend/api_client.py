"""API Client — Thin HTTP client for Streamlit frontend.

Abstracts all API calls so Streamlit pages never import
src.pipelines, src.models, or src.evaluation directly.

Usage:
    from src.frontend.api_client import APIClient
    client = APIClient()
    result = client.predict(horizon=6, model_name="gru")
    experiments = client.get_experiments()
"""

from __future__ import annotations

import os
from typing import Any

import requests
from loguru import logger

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


class APIClient:
    """Thin HTTP client for the PM2.5 Forecasting API."""

    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or API_BASE_URL).rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    # ── Health ──

    def health(self) -> dict:
        """Check API health."""
        return self._get("/health")

    # ── Inference ──

    def predict(self, horizon: int, model_name: str = "gru") -> dict:
        """Run PM2.5 prediction."""
        return self._post("/api/v1/predict", json={
            "horizon": horizon,
            "model_name": model_name,
        })

    # ── Experiments ──

    def get_experiments(self, limit: int = 50) -> list[dict]:
        """List all experiments."""
        return self._get(f"/api/v1/experiments?limit={limit}")

    def get_experiment(self, experiment_id: int) -> dict:
        """Get a single experiment."""
        return self._get(f"/api/v1/experiments/{experiment_id}")

    def create_experiment(self, name: str, **kwargs) -> dict:
        """Create a new experiment."""
        return self._post("/api/v1/experiments", json={"name": name, **kwargs})

    # ── Runs ──

    def get_runs(self, experiment_id: int) -> list[dict]:
        """List runs for an experiment."""
        return self._get(f"/api/v1/experiments/{experiment_id}/runs")

    def create_run(self, experiment_id: int, horizon: int) -> dict:
        """Create a new run."""
        return self._post("/api/v1/runs", json={
            "experiment_id": experiment_id,
            "horizon": horizon,
        })

    # ── Run Models ──

    def create_run_model(self, run_id: int, model_name: str, **kwargs) -> dict:
        """Log a model within a run."""
        return self._post("/api/v1/run-models", json={
            "run_id": run_id,
            "model_name": model_name,
            **kwargs,
        })

    # ── Metrics ──

    def create_metric(self, run_model_id: int, **kwargs) -> dict:
        """Log metrics for a model."""
        return self._post("/api/v1/metrics", json={
            "run_model_id": run_model_id,
            **kwargs,
        })

    def get_metrics(self, run_model_id: int) -> list[dict]:
        """Get metrics for a model."""
        return self._get(f"/api/v1/run-models/{run_model_id}/metrics")

    # ── Audit ──

    def get_data_hashes(self) -> list[dict]:
        """Get data file hashes for audit."""
        return self._get("/api/v1/audit/data-hashes")

    def get_model_weights(self) -> list[dict]:
        """Get model weight hashes for audit."""
        return self._get("/api/v1/audit/model-weights")

    def get_audit_report(self) -> dict:
        """Get full audit report."""
        return self._get("/api/v1/audit/report")

    def verify_integrity(self) -> dict:
        """Verify file integrity against manifest.json expected MD5 hashes."""
        return self._get("/api/v1/audit/verify")

    # ── Content (Info Cards) ──

    def get_info_cards(self, page: str | None = None) -> list[dict]:
        """List all info cards, optionally filtered by page."""
        path = "/api/v1/content/info-cards"
        if page:
            path += f"?page={page}"
        return self._get(path)

    def get_info_card(self, card_key: str, quiet: bool = False) -> dict:
        """Get a single info card by key."""
        return self._get(f"/api/v1/content/info-cards/{card_key}", quiet=quiet)

    def update_info_card(self, card_key: str, title: str | None = None, content: str | None = None) -> dict:
        """Update an info card's title and/or content."""
        payload = {}
        if title is not None:
            payload["title"] = title
        if content is not None:
            payload["content"] = content
        return self._put(f"/api/v1/content/info-cards/{card_key}", json=payload)

    # ── Internal ──

    def _get(self, path: str, quiet: bool = False) -> Any:
        """HTTP GET request."""
        url = f"{self.base_url}{path}"
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.ConnectionError:
            if not quiet:
                logger.error(f"API connection failed: {url}")
            return {"error": "API server not available"}
        except requests.HTTPError as e:
            if not quiet:
                logger.error(f"API error: {e}")
            return {"error": str(e)}

    def _post(self, path: str, json: dict | None = None) -> Any:
        """HTTP POST request."""
        url = f"{self.base_url}{path}"
        try:
            resp = self.session.post(url, json=json, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.ConnectionError:
            logger.error(f"API connection failed: {url}")
            return {"error": "API server not available"}
        except requests.HTTPError as e:
            logger.error(f"API error: {e}")
            return {"error": str(e)}

    def _put(self, path: str, json: dict | None = None) -> Any:
        """HTTP PUT request."""
        url = f"{self.base_url}{path}"
        try:
            resp = self.session.put(url, json=json, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.ConnectionError:
            logger.error(f"API connection failed: {url}")
            return {"error": "API server not available"}
        except requests.HTTPError as e:
            logger.error(f"API error: {e}")
            return {"error": str(e)}
