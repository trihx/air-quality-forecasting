"""API tests — Experiments CRUD, Inference, Audit endpoints.

Uses FastAPI TestClient (no actual server needed).
Uses SQLite in-memory for test isolation.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.api.database import Base, get_db
from src.api.main import app

# ── Test database (in-memory SQLite) ──
TEST_DATABASE_URL = "sqlite:///./tests/api/test_forecasting.db"
test_engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestSession = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    """Override DB dependency for tests."""
    db = TestSession()
    try:
        yield db
    finally:
        db.close()


# Override the DB dependency
app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def setup_db():
    """Create tables before each test, drop after."""
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture()
def client():
    """FastAPI test client."""
    return TestClient(app)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Health Check
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestHealthCheck:
    """Test /health and /api/health endpoints."""

    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "1.0.0"

    def test_api_health_returns_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "1.0.0"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Experiments CRUD
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestExperiments:
    """Test /api/v1/experiments endpoints."""

    def test_create_experiment(self, client):
        resp = client.post("/api/v1/experiments", json={
            "name": "v8_test",
            "pipeline_version": "v8",
            "description": "Test experiment",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "v8_test"
        assert data["pipeline_version"] == "v8"
        assert data["id"] >= 1

    def test_list_experiments(self, client):
        # Create 2 experiments
        client.post("/api/v1/experiments", json={"name": "exp1"})
        client.post("/api/v1/experiments", json={"name": "exp2"})

        resp = client.get("/api/v1/experiments")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_get_experiment_by_id(self, client):
        create_resp = client.post("/api/v1/experiments", json={"name": "find_me"})
        exp_id = create_resp.json()["id"]

        resp = client.get(f"/api/v1/experiments/{exp_id}")
        assert resp.status_code == 200
        assert resp.json()["name"] == "find_me"

    def test_get_experiment_not_found(self, client):
        resp = client.get("/api/v1/experiments/9999")
        assert resp.status_code == 404


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Runs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRuns:
    """Test /api/v1/runs endpoints."""

    def test_create_run(self, client):
        # Create experiment first
        exp = client.post("/api/v1/experiments", json={"name": "run_test"}).json()

        resp = client.post("/api/v1/runs", json={
            "experiment_id": exp["id"],
            "horizon": 6,
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["horizon"] == 6
        assert data["status"] == "pending"

    def test_create_run_invalid_experiment(self, client):
        resp = client.post("/api/v1/runs", json={
            "experiment_id": 9999,
            "horizon": 1,
        })
        assert resp.status_code == 404

    def test_list_runs_for_experiment(self, client):
        exp = client.post("/api/v1/experiments", json={"name": "runs_list"}).json()
        client.post("/api/v1/runs", json={"experiment_id": exp["id"], "horizon": 1})
        client.post("/api/v1/runs", json={"experiment_id": exp["id"], "horizon": 6})
        client.post("/api/v1/runs", json={"experiment_id": exp["id"], "horizon": 24})

        resp = client.get(f"/api/v1/experiments/{exp['id']}/runs")
        assert resp.status_code == 200
        assert len(resp.json()) == 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run Models & Metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestRunModelsAndMetrics:
    """Test model logging and metric recording."""

    def _create_chain(self, client):
        """Helper: create experiment → run → return run_id."""
        exp = client.post("/api/v1/experiments", json={"name": "metric_test"}).json()
        run = client.post("/api/v1/runs", json={
            "experiment_id": exp["id"], "horizon": 6,
        }).json()
        return run["id"]

    def test_create_run_model(self, client):
        run_id = self._create_chain(client)
        resp = client.post("/api/v1/run-models", json={
            "run_id": run_id,
            "model_name": "GRU",
            "training_time_s": 45.2,
            "hyperparameters": {"hidden_dim": 64, "lr": 0.001},
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["model_name"] == "GRU"
        assert data["training_time_s"] == 45.2

    def test_create_metrics(self, client):
        run_id = self._create_chain(client)
        rm = client.post("/api/v1/run-models", json={
            "run_id": run_id,
            "model_name": "LightGBM",
        }).json()

        resp = client.post("/api/v1/metrics", json={
            "run_model_id": rm["id"],
            "mae": 2.145,
            "rmse": 3.567,
            "mase": 0.737,
            "r2": 0.85,
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["mae"] == 2.145
        assert data["mase"] == 0.737

    def test_get_metrics_for_model(self, client):
        run_id = self._create_chain(client)
        rm = client.post("/api/v1/run-models", json={
            "run_id": run_id, "model_name": "GRU",
        }).json()

        client.post("/api/v1/metrics", json={
            "run_model_id": rm["id"], "mae": 1.5, "mase": 0.8,
        })

        resp = client.get(f"/api/v1/run-models/{rm['id']}/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["mae"] == 1.5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Audit
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestAudit:
    """Test /api/v1/audit endpoints."""

    def test_audit_report_structure(self, client):
        resp = client.get("/api/v1/audit/report")
        assert resp.status_code == 200
        data = resp.json()
        assert "data_hashes" in data
        assert "model_weights" in data
        assert "computed_at" in data

    def test_data_hashes_returns_list(self, client):
        resp = client.get("/api/v1/audit/data-hashes")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_model_weights_returns_list(self, client):
        resp = client.get("/api/v1/audit/model-weights")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
