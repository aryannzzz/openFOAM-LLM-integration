"""
Test suite for LLM-OpenFOAM Orchestrator
"""
import pytest
from fastapi.testclient import TestClient
import json

from main import app
from app.models import SimulationRequest, SimulationType
from app.llm_converter import LLMConverter


client = TestClient(app)


class TestHealthCheck:
    """Test health check endpoint"""

    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "docs" in response.json()


class TestConversion:
    """Test LLM conversion functionality"""

    def test_llm_conversion(self):
        """Test LLM conversion endpoint"""
        response = client.post(
            "/api/convert",
            params={"description": "Simulate flow around a cylinder at 10 m/s"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "json_configuration" in data
        assert "confidence_score" in data

    def test_llm_converter_instance(self):
        """Test LLM converter"""
        converter = LLMConverter(provider="openai", model="gpt-4")
        result = converter.convert_to_json("Flow simulation")
        assert result.confidence_score >= 0
        assert result.json_configuration is not None


class TestSimulation:
    """Test simulation endpoints"""

    def test_submit_simulation(self):
        """Test simulation submission"""
        request_data = {
            "description": "Simulate incompressible flow around a cylinder",
            "case_name": "test_cylinder_flow",
            "max_runtime": 600
        }
        response = client.post("/api/simulate", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert data["case_name"] == "test_cylinder_flow"
        assert "case_id" in data

    def test_get_simulation_status(self):
        """Test getting simulation status"""
        # First submit a simulation
        request_data = {
            "description": "Test simulation",
            "case_name": "test_status"
        }
        submit_response = client.post("/api/simulate", json=request_data)
        case_id = submit_response.json()["case_id"]

        # Check status
        response = client.get(f"/api/status/{case_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["case_id"] == case_id

    def test_list_simulations(self):
        """Test listing simulations"""
        response = client.get("/api/simulations")
        assert response.status_code == 200
        assert "total" in response.json()
        assert "simulations" in response.json()


class TestErrorHandling:
    """Test error handling"""

    def test_invalid_case_id(self):
        """Test invalid case ID handling"""
        response = client.get("/api/status/invalid_case_id")
        assert response.status_code == 404

    def test_invalid_request_data(self):
        """Test invalid request data"""
        response = client.post("/api/simulate", json={})
        assert response.status_code == 422  # Validation error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
