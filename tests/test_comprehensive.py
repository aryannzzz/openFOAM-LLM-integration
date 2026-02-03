#!/usr/bin/env python3
"""
Comprehensive Test Suite for LLM-Driven OpenFOAM Orchestration System
Tests all components: schemas, validation, security, LLM conversion, case generation, OpenFOAM execution
"""
import os
import sys
import json
import shutil
import subprocess
import tempfile
import pytest
from pathlib import Path
from datetime import datetime

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas import (
    CFDSpecification, Metadata, Geometry, GeometryType, GeometryDimensions, GeometryDomain,
    Flow, FlowRegime, TimeDependence, TurbulenceModel,
    Fluid, Solver, SolverType, SolverAlgorithm, ConvergenceCriteria,
    TimeSettings, Mesh, MeshResolution, BoundaryLayer,
    Boundaries, InletBoundary, OutletBoundary, WallBoundary, SymmetryBoundary,
    InitialConditions, Outputs, Execution
)
from app.validation import PhysicsValidator, get_recommended_solver, get_recommended_turbulence_model
from app.security import SecurityChecker
from app.llm_converter import LLMConverter
from app.errors import ErrorCode, CFDError


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def sample_laminar_spec():
    """Create a sample laminar flow specification"""
    return CFDSpecification(
        metadata=Metadata(name="test_laminar", description="Laminar test case", version="1.0"),
        geometry=Geometry(
            type=GeometryType.CYLINDER_2D,
            dimensions=GeometryDimensions(characteristic_length=0.01),
            domain=GeometryDomain(upstream=10, downstream=30, lateral=10)
        ),
        flow=Flow(
            regime=FlowRegime.LAMINAR,
            reynolds_number=100,
            time_dependence=TimeDependence.STEADY,
            turbulence_model=TurbulenceModel.NONE
        ),
        fluid=Fluid(name="air", density=1.225, kinematic_viscosity=1.5e-5),
        solver=Solver(
            type=SolverType.SIMPLE_FOAM,
            algorithm=SolverAlgorithm.SIMPLE,
            max_iterations=5000,
            convergence_criteria=ConvergenceCriteria(p=1e-5, U=1e-5)
        ),
        mesh=Mesh(resolution=MeshResolution.MEDIUM, boundary_layer=BoundaryLayer(enabled=True, num_layers=10)),
        boundaries=Boundaries(
            inlet=InletBoundary(type="velocity_inlet", velocity=[0.15, 0, 0]),
            outlet=OutletBoundary(type="pressure_outlet", pressure=0),
            walls=WallBoundary(type="no_slip"),
            symmetry=SymmetryBoundary(planes=["front", "back"])
        ),
        initial_conditions=InitialConditions(velocity=[0.15, 0, 0], pressure=0),
        outputs=Outputs(fields=["p", "U"], derived_quantities=["drag_coefficient"]),
        execution=Execution(parallel=False, num_processors=1)
    )


@pytest.fixture
def sample_turbulent_spec():
    """Create a sample turbulent flow specification"""
    return CFDSpecification(
        metadata=Metadata(name="test_turbulent", description="Turbulent test case", version="1.0"),
        geometry=Geometry(
            type=GeometryType.CYLINDER_2D,
            dimensions=GeometryDimensions(characteristic_length=0.01),
            domain=GeometryDomain(upstream=10, downstream=30, lateral=10)
        ),
        flow=Flow(
            regime=FlowRegime.TURBULENT_RANS,
            reynolds_number=10000,
            time_dependence=TimeDependence.STEADY,
            turbulence_model=TurbulenceModel.K_OMEGA_SST
        ),
        fluid=Fluid(name="air", density=1.225, kinematic_viscosity=1.5e-5),
        solver=Solver(
            type=SolverType.SIMPLE_FOAM,
            algorithm=SolverAlgorithm.SIMPLE,
            max_iterations=5000,
            convergence_criteria=ConvergenceCriteria(p=1e-5, U=1e-5, k=1e-5, omega=1e-5)
        ),
        mesh=Mesh(resolution=MeshResolution.FINE, boundary_layer=BoundaryLayer(enabled=True, num_layers=20)),
        boundaries=Boundaries(
            inlet=InletBoundary(type="velocity_inlet", velocity=[15, 0, 0]),
            outlet=OutletBoundary(type="pressure_outlet", pressure=0),
            walls=WallBoundary(type="no_slip")
        ),
        initial_conditions=InitialConditions(velocity=[15, 0, 0], pressure=0),
        outputs=Outputs(fields=["p", "U", "k", "omega"], derived_quantities=["drag_coefficient", "lift_coefficient"]),
        execution=Execution(parallel=True, num_processors=4)
    )


@pytest.fixture
def sample_transient_spec():
    """Create a sample transient flow specification"""
    return CFDSpecification(
        metadata=Metadata(name="test_transient", description="Transient test case", version="1.0"),
        geometry=Geometry(
            type=GeometryType.CYLINDER_2D,
            dimensions=GeometryDimensions(characteristic_length=0.01),
            domain=GeometryDomain(upstream=10, downstream=30, lateral=10)
        ),
        flow=Flow(
            regime=FlowRegime.LAMINAR,
            reynolds_number=200,
            time_dependence=TimeDependence.TRANSIENT,
            turbulence_model=TurbulenceModel.NONE
        ),
        fluid=Fluid(name="air", density=1.225, kinematic_viscosity=1.5e-5),
        solver=Solver(
            type=SolverType.PIMPLE_FOAM,
            algorithm=SolverAlgorithm.PIMPLE,
            max_iterations=5000,
            convergence_criteria=ConvergenceCriteria(p=1e-5, U=1e-5)
        ),
        time=TimeSettings(end_time=2.0, delta_t=0.0001, adjustable_time_step=True, max_courant=1.0, write_interval=0.1),
        mesh=Mesh(resolution=MeshResolution.MEDIUM, boundary_layer=BoundaryLayer(enabled=True, num_layers=10)),
        boundaries=Boundaries(
            inlet=InletBoundary(type="velocity_inlet", velocity=[0.3, 0, 0]),
            outlet=OutletBoundary(type="pressure_outlet", pressure=0),
            walls=WallBoundary(type="no_slip"),
            symmetry=SymmetryBoundary(planes=["front", "back"])
        ),
        initial_conditions=InitialConditions(velocity=[0.3, 0, 0], pressure=0),
        outputs=Outputs(fields=["p", "U"], derived_quantities=["drag_coefficient", "lift_coefficient"]),
        execution=Execution(parallel=False, num_processors=1)
    )


# =============================================================================
# SCHEMA TESTS
# =============================================================================

class TestSchemas:
    """Test CFDSpecification schema creation and validation"""
    
    def test_create_minimal_spec(self):
        """Test creating a minimal valid specification"""
        spec = CFDSpecification(
            metadata=Metadata(name="minimal_test", description="Minimal test", version="1.0"),
            geometry=Geometry(
                type=GeometryType.CYLINDER_2D,
                dimensions=GeometryDimensions(characteristic_length=0.01),
                domain=GeometryDomain(upstream=10, downstream=30, lateral=10)
            ),
            flow=Flow(regime=FlowRegime.LAMINAR, reynolds_number=100, time_dependence=TimeDependence.STEADY, turbulence_model=TurbulenceModel.NONE),
            fluid=Fluid(name="air", density=1.225, kinematic_viscosity=1.5e-5),
            solver=Solver(type=SolverType.SIMPLE_FOAM, algorithm=SolverAlgorithm.SIMPLE, max_iterations=1000, convergence_criteria=ConvergenceCriteria(p=1e-5, U=1e-5)),
            mesh=Mesh(resolution=MeshResolution.COARSE, boundary_layer=BoundaryLayer(enabled=False)),
            boundaries=Boundaries(inlet=InletBoundary(type="velocity_inlet", velocity=[1, 0, 0]), outlet=OutletBoundary(type="pressure_outlet", pressure=0), walls=WallBoundary(type="no_slip")),
            initial_conditions=InitialConditions(velocity=[1, 0, 0], pressure=0),
            outputs=Outputs(fields=["p", "U"]),
            execution=Execution(parallel=False, num_processors=1)
        )
        assert spec.metadata.name == "minimal_test"
        assert spec.flow.reynolds_number == 100
        
    def test_all_geometry_types(self):
        """Test all supported geometry types"""
        geometry_types = [
            GeometryType.CYLINDER_2D, GeometryType.CYLINDER_3D, GeometryType.SPHERE,
            GeometryType.FLAT_PLATE_2D, GeometryType.BACKWARD_FACING_STEP,
            GeometryType.CHANNEL_2D, GeometryType.CHANNEL_3D,
            GeometryType.AIRFOIL_NACA_4DIGIT, GeometryType.BOX_WITH_OBSTACLE, GeometryType.PIPE_3D
        ]
        for gt in geometry_types:
            geom = Geometry(type=gt, dimensions=GeometryDimensions(characteristic_length=0.01), domain=GeometryDomain(upstream=10, downstream=30, lateral=10))
            assert geom.type == gt
    
    def test_all_flow_regimes(self):
        """Test all supported flow regimes"""
        regimes = [FlowRegime.LAMINAR, FlowRegime.TURBULENT_RANS, FlowRegime.TURBULENT_LES]
        for regime in regimes:
            flow = Flow(regime=regime, reynolds_number=100, time_dependence=TimeDependence.STEADY, turbulence_model=TurbulenceModel.NONE)
            assert flow.regime == regime
    
    def test_all_solver_types(self):
        """Test all supported solver types"""
        solver_types = [
            SolverType.SIMPLE_FOAM, SolverType.PIMPLE_FOAM, SolverType.PISO_FOAM,
            SolverType.ICO_FOAM, SolverType.RHO_SIMPLE_FOAM, SolverType.RHO_PIMPLE_FOAM
        ]
        for st in solver_types:
            solver = Solver(type=st, algorithm=SolverAlgorithm.SIMPLE, max_iterations=1000, convergence_criteria=ConvergenceCriteria(p=1e-5, U=1e-5))
            assert solver.type == st
    
    def test_all_turbulence_models(self):
        """Test all supported turbulence models"""
        models = [
            TurbulenceModel.NONE, TurbulenceModel.K_EPSILON, TurbulenceModel.K_OMEGA,
            TurbulenceModel.K_OMEGA_SST, TurbulenceModel.SPALART_ALLMARAS,
            TurbulenceModel.SMAGORINSKY, TurbulenceModel.WALE
        ]
        for model in models:
            flow = Flow(regime=FlowRegime.LAMINAR, reynolds_number=100, time_dependence=TimeDependence.STEADY, turbulence_model=model)
            assert flow.turbulence_model == model
    
    def test_spec_to_json(self, sample_laminar_spec):
        """Test specification serialization to JSON"""
        json_str = sample_laminar_spec.model_dump_json()
        assert json_str is not None
        data = json.loads(json_str)
        assert data["metadata"]["name"] == "test_laminar"
        assert data["flow"]["reynolds_number"] == 100
    
    def test_spec_from_json(self, sample_laminar_spec):
        """Test specification deserialization from JSON"""
        json_str = sample_laminar_spec.model_dump_json()
        restored = CFDSpecification.model_validate_json(json_str)
        assert restored.metadata.name == sample_laminar_spec.metadata.name
        assert restored.flow.reynolds_number == sample_laminar_spec.flow.reynolds_number


# =============================================================================
# PHYSICS VALIDATION TESTS
# =============================================================================

class TestPhysicsValidation:
    """Test physics validation logic"""
    
    def test_valid_laminar_spec(self, sample_laminar_spec):
        """Test validation of a valid laminar specification"""
        validator = PhysicsValidator()
        is_valid, errors = validator.validate(sample_laminar_spec)
        assert is_valid == True
        assert len([e for e in errors if e.severity == "error"]) == 0
    
    def test_valid_turbulent_spec(self, sample_turbulent_spec):
        """Test validation of a valid turbulent specification"""
        validator = PhysicsValidator()
        is_valid, errors = validator.validate(sample_turbulent_spec)
        assert is_valid == True
    
    def test_valid_transient_spec(self, sample_transient_spec):
        """Test validation of a valid transient specification"""
        validator = PhysicsValidator()
        is_valid, errors = validator.validate(sample_transient_spec)
        assert is_valid == True
    
    def test_invalid_solver_regime_combination(self):
        """Test detection of invalid solver-regime combination"""
        spec = CFDSpecification(
            metadata=Metadata(name="invalid", description="Invalid combo", version="1.0"),
            geometry=Geometry(type=GeometryType.CYLINDER_2D, dimensions=GeometryDimensions(characteristic_length=0.01), domain=GeometryDomain(upstream=10, downstream=30, lateral=10)),
            flow=Flow(regime=FlowRegime.TURBULENT_LES, reynolds_number=10000, time_dependence=TimeDependence.STEADY, turbulence_model=TurbulenceModel.SMAGORINSKY),
            fluid=Fluid(name="air", density=1.225, kinematic_viscosity=1.5e-5),
            solver=Solver(type=SolverType.ICO_FOAM, algorithm=SolverAlgorithm.PISO, max_iterations=1000, convergence_criteria=ConvergenceCriteria(p=1e-5, U=1e-5)),
            mesh=Mesh(resolution=MeshResolution.MEDIUM, boundary_layer=BoundaryLayer(enabled=True, num_layers=10)),
            boundaries=Boundaries(inlet=InletBoundary(type="velocity_inlet", velocity=[1, 0, 0]), outlet=OutletBoundary(type="pressure_outlet", pressure=0), walls=WallBoundary(type="no_slip")),
            initial_conditions=InitialConditions(velocity=[1, 0, 0], pressure=0),
            outputs=Outputs(fields=["p", "U"]),
            execution=Execution(parallel=False, num_processors=1)
        )
        validator = PhysicsValidator()
        is_valid, errors = validator.validate(spec)
        # icoFoam is for laminar only, not LES
        assert any("icoFoam" in str(e.message).lower() or "laminar" in str(e.message).lower() for e in errors)
    
    def test_reynolds_number_guidance(self):
        """Test Reynolds number regime guidance"""
        # Laminar steady should suggest simpleFoam
        assert get_recommended_solver(FlowRegime.LAMINAR, TimeDependence.STEADY) == SolverType.SIMPLE_FOAM
        # Turbulent steady should suggest simpleFoam
        assert get_recommended_solver(FlowRegime.TURBULENT_RANS, TimeDependence.STEADY) == SolverType.SIMPLE_FOAM
        # Laminar transient should suggest icoFoam
        assert get_recommended_solver(FlowRegime.LAMINAR, TimeDependence.TRANSIENT) == SolverType.ICO_FOAM
        # Turbulent transient should suggest pimpleFoam
        assert get_recommended_solver(FlowRegime.TURBULENT_RANS, TimeDependence.TRANSIENT) == SolverType.PIMPLE_FOAM
    
    def test_turbulence_model_recommendations(self):
        """Test turbulence model recommendations"""
        assert get_recommended_turbulence_model(FlowRegime.LAMINAR, re=100) == TurbulenceModel.NONE
        assert get_recommended_turbulence_model(FlowRegime.TURBULENT_RANS, re=10000) == TurbulenceModel.K_OMEGA_SST
        assert get_recommended_turbulence_model(FlowRegime.TURBULENT_LES, re=10000) == TurbulenceModel.WALE


# =============================================================================
# SECURITY VALIDATION TESTS
# =============================================================================

class TestSecurityValidation:
    """Test security validation logic"""
    
    def test_safe_specification(self, sample_laminar_spec):
        """Test that a safe specification passes security checks"""
        checker = SecurityChecker()
        is_safe, details = checker.check(sample_laminar_spec)
        assert is_safe == True
        assert len(details.get("security_issues", [])) == 0
    
    def test_resource_limits_normal(self, sample_laminar_spec):
        """Test that normal resources pass validation"""
        checker = SecurityChecker()
        is_safe, details = checker.check(sample_laminar_spec)
        assert is_safe == True
        assert len(details.get("resource_violations", [])) == 0
    
    def test_excessive_processors_warning(self):
        """Test that processor count is validated at schema level"""
        # Pydantic schema enforces max 64 processors
        with pytest.raises(Exception):  # ValidationError
            spec = CFDSpecification(
                metadata=Metadata(name="excessive", description="Too many CPUs", version="1.0"),
                geometry=Geometry(type=GeometryType.CYLINDER_2D, dimensions=GeometryDimensions(characteristic_length=0.01), domain=GeometryDomain(upstream=10, downstream=30, lateral=10)),
                flow=Flow(regime=FlowRegime.LAMINAR, reynolds_number=100, time_dependence=TimeDependence.STEADY, turbulence_model=TurbulenceModel.NONE),
                fluid=Fluid(name="air", density=1.225, kinematic_viscosity=1.5e-5),
                solver=Solver(type=SolverType.SIMPLE_FOAM, algorithm=SolverAlgorithm.SIMPLE, max_iterations=1000, convergence_criteria=ConvergenceCriteria(p=1e-5, U=1e-5)),
                mesh=Mesh(resolution=MeshResolution.MEDIUM, boundary_layer=BoundaryLayer(enabled=True, num_layers=10)),
                boundaries=Boundaries(inlet=InletBoundary(type="velocity_inlet", velocity=[1, 0, 0]), outlet=OutletBoundary(type="pressure_outlet", pressure=0), walls=WallBoundary(type="no_slip")),
                initial_conditions=InitialConditions(velocity=[1, 0, 0], pressure=0),
                outputs=Outputs(fields=["p", "U"]),
                execution=Execution(parallel=True, num_processors=100)  # Should fail validation
            )


# =============================================================================
# LLM CONVERTER TESTS
# =============================================================================

class TestLLMConverter:
    """Test LLM conversion from natural language to CFDSpecification"""
    
    def test_mock_conversion_cylinder(self):
        """Test mock conversion for cylinder flow"""
        converter = LLMConverter(provider="mock")
        response = converter.convert_to_cfd_specification(
            "Simulate laminar flow around a 2D cylinder at Reynolds number 100"
        )
        assert response.specification is not None
        assert response.specification.flow.reynolds_number == 100
        assert response.specification.geometry.type == GeometryType.CYLINDER_2D
        assert response.specification.flow.regime == FlowRegime.LAMINAR
    
    def test_mock_conversion_channel(self):
        """Test mock conversion for channel flow"""
        converter = LLMConverter(provider="mock")
        response = converter.convert_to_cfd_specification(
            "Simulate flow in a 2D channel with Re=500"
        )
        assert response.specification is not None
        assert response.specification.flow.reynolds_number == 500
        assert response.specification.geometry.type == GeometryType.CHANNEL_2D
    
    def test_mock_conversion_sphere(self):
        """Test mock conversion for sphere"""
        converter = LLMConverter(provider="mock")
        response = converter.convert_to_cfd_specification(
            "Model flow around a sphere at Reynolds 200"
        )
        assert response.specification is not None
        assert response.specification.geometry.type == GeometryType.SPHERE
    
    def test_mock_conversion_pipe(self):
        """Test mock conversion for pipe flow"""
        converter = LLMConverter(provider="mock")
        response = converter.convert_to_cfd_specification(
            "Analyze flow in a 3D pipe at Re=2000"
        )
        assert response.specification is not None
        assert response.specification.geometry.type == GeometryType.PIPE_3D
    
    def test_mock_conversion_turbulent_detection(self):
        """Test automatic turbulent regime detection at high Re"""
        converter = LLMConverter(provider="mock")
        response = converter.convert_to_cfd_specification(
            "Simulate flow at Reynolds number 50000"
        )
        assert response.specification is not None
        assert response.specification.flow.reynolds_number == 50000
        assert response.specification.flow.regime == FlowRegime.TURBULENT_RANS
    
    def test_mock_conversion_transient_detection(self):
        """Test transient flow detection"""
        converter = LLMConverter(provider="mock")
        response = converter.convert_to_cfd_specification(
            "Simulate transient vortex shedding behind a cylinder"
        )
        assert response.specification is not None
        assert response.specification.flow.time_dependence == TimeDependence.TRANSIENT
    
    def test_confidence_score(self):
        """Test that confidence scores are reasonable"""
        converter = LLMConverter(provider="mock")
        response = converter.convert_to_cfd_specification(
            "Simulate laminar flow around a cylinder at Re=100"
        )
        assert 0 <= response.confidence_score <= 1.0
    
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not set")
    def test_real_llm_conversion(self):
        """Test real LLM conversion (requires API key)"""
        converter = LLMConverter(provider="openai", model="gpt-4o-mini")
        response = converter.convert_to_cfd_specification(
            "Simulate laminar incompressible flow around a 2D cylinder at Reynolds number 50"
        )
        assert response.specification is not None
        assert response.confidence_score >= 0.5


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the full pipeline"""
    
    def test_full_pipeline_mock(self, sample_laminar_spec):
        """Test full pipeline with mock LLM"""
        # Step 1: LLM Conversion (mock)
        converter = LLMConverter(provider="mock")
        response = converter.convert_to_cfd_specification(
            "Simulate laminar flow around a cylinder at Re=100"
        )
        spec = response.specification
        
        # Step 2: Physics Validation
        validator = PhysicsValidator()
        is_valid, errors = validator.validate(spec)
        assert is_valid == True
        
        # Step 3: Security Validation
        checker = SecurityChecker()
        is_safe, details = checker.check(spec)
        assert is_safe == True
    
    def test_spec_roundtrip(self, sample_laminar_spec):
        """Test specification serialization roundtrip"""
        # Serialize
        json_str = sample_laminar_spec.model_dump_json()
        
        # Deserialize
        restored = CFDSpecification.model_validate_json(json_str)
        
        # Verify
        assert restored.metadata.name == sample_laminar_spec.metadata.name
        assert restored.flow.reynolds_number == sample_laminar_spec.flow.reynolds_number
        assert restored.solver.type == sample_laminar_spec.solver.type
        
        # Validate restored spec
        validator = PhysicsValidator()
        is_valid, errors = validator.validate(restored)
        assert is_valid == True


# =============================================================================
# OPENFOAM EXECUTION TESTS (require OpenFOAM installed)
# =============================================================================

def openfoam_available():
    """Check if OpenFOAM is available"""
    try:
        result = subprocess.run(
            "source /opt/openfoam11/etc/bashrc && blockMesh -help",
            shell=True, executable='/bin/bash',
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except:
        return False


@pytest.mark.skipif(not openfoam_available(), reason="OpenFOAM not installed")
class TestOpenFOAMExecution:
    """Tests requiring OpenFOAM installation"""
    
    def test_blockmesh_execution(self, tmp_path):
        """Test blockMesh execution"""
        case_dir = tmp_path / "test_case"
        case_dir.mkdir()
        (case_dir / "system").mkdir()
        (case_dir / "constant").mkdir()
        (case_dir / "0").mkdir()
        
        # Create minimal blockMeshDict
        block_mesh = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}

scale 0.1;

vertices
(
    (0 0 0) (1 0 0) (1 1 0) (0 1 0)
    (0 0 0.1) (1 0 0.1) (1 1 0.1) (0 1 0.1)
);

blocks ( hex (0 1 2 3 4 5 6 7) (10 10 1) simpleGrading (1 1 1) );

boundary
(
    allBoundary { type patch; faces ( (3 7 6 2) (0 4 7 3) (2 6 5 1) (1 5 4 0) ); }
    frontAndBack { type empty; faces ( (0 3 2 1) (4 5 6 7) ); }
);
"""
        (case_dir / "system" / "blockMeshDict").write_text(block_mesh)
        
        # Create OpenFOAM v11 compatible controlDict
        control_dict = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}
application foamRun;
solver incompressibleFluid;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 1;
deltaT 0.01;
writeControl timeStep;
writeInterval 100;
purgeWrite 0;
writeFormat ascii;
writePrecision 6;
writeCompression off;
timeFormat general;
timePrecision 6;
runTimeModifiable true;
"""
        (case_dir / "system" / "controlDict").write_text(control_dict)
        
        # Run blockMesh
        result = subprocess.run(
            f"source /opt/openfoam11/etc/bashrc && cd {case_dir} && blockMesh",
            shell=True, executable='/bin/bash',
            capture_output=True, text=True, timeout=60
        )
        
        assert result.returncode == 0
        assert (case_dir / "constant" / "polyMesh").exists()


# =============================================================================
# PROMPT VARIATION TESTS
# =============================================================================

class TestPromptVariations:
    """Test various natural language prompt formats"""
    
    @pytest.mark.parametrize("prompt,expected_re", [
        ("Simulate flow at Reynolds 100", 100),
        ("Re=500 cylinder flow", 500),
        ("Reynolds number 1000 simulation", 1000),
        ("Re: 50", 50),
    ])
    def test_reynolds_extraction(self, prompt, expected_re):
        """Test Reynolds number extraction from various formats"""
        converter = LLMConverter(provider="mock")
        response = converter.convert_to_cfd_specification(prompt)
        assert response.specification.flow.reynolds_number == expected_re
    
    @pytest.mark.parametrize("prompt,expected_geom", [
        ("cylinder flow", GeometryType.CYLINDER_2D),
        ("sphere simulation", GeometryType.SPHERE),
        ("channel analysis", GeometryType.CHANNEL_2D),
        ("pipe flow", GeometryType.PIPE_3D),
        ("airfoil NACA", GeometryType.AIRFOIL_NACA_4DIGIT),
        ("backward facing step", GeometryType.BACKWARD_FACING_STEP),
    ])
    def test_geometry_detection(self, prompt, expected_geom):
        """Test geometry type detection from prompts"""
        converter = LLMConverter(provider="mock")
        response = converter.convert_to_cfd_specification(prompt)
        assert response.specification.geometry.type == expected_geom


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
