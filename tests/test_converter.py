"""
Test LLM converter module
"""
import pytest
from app.llm_converter import LLMConverter
from app.models import SimulationType, MeshType, SolverType


class TestLLMConverter:
    """Test LLM converter"""

    def test_mock_conversion_incompressible(self):
        """Test mock conversion for incompressible flow"""
        converter = LLMConverter(provider="mock")
        result = converter.convert_to_json("Flow around a cylinder")

        assert result.json_configuration.simulation_type == SimulationType.INCOMPRESSIBLE_FLOW
        assert result.confidence_score >= 0

    def test_mock_conversion_heat_transfer(self):
        """Test mock conversion for heat transfer"""
        converter = LLMConverter(provider="mock")
        result = converter.convert_to_json("Heat transfer in a pipe")

        assert result.json_configuration.simulation_type == SimulationType.HEAT_TRANSFER

    def test_mock_conversion_combustion(self):
        """Test mock conversion for combustion"""
        converter = LLMConverter(provider="mock")
        result = converter.convert_to_json("Combustion simulation in a burner")

        assert result.json_configuration.simulation_type == SimulationType.COMBUSTION

    def test_configuration_validity(self):
        """Test that configuration is valid"""
        converter = LLMConverter(provider="mock")
        result = converter.convert_to_json("Test simulation")
        config = result.json_configuration

        # Check required fields exist
        assert config.domain is not None
        assert config.boundary_conditions is not None
        assert config.material_properties is not None
        assert config.numerics is not None
        assert config.simulation_parameters is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
