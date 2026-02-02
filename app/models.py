"""
Data models for API requests and responses
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum


class SimulationType(str, Enum):
    """Supported simulation types"""
    INCOMPRESSIBLE_FLOW = "incompressible_flow"
    COMPRESSIBLE_FLOW = "compressible_flow"
    HEAT_TRANSFER = "heat_transfer"
    COMBUSTION = "combustion"
    MULTIPHASE = "multiphase"


class MeshType(str, Enum):
    """Supported mesh generation methods"""
    SNAPPY_HEX_MESH = "snappyHexMesh"
    BLOCK_MESH = "blockMesh"
    SIMPLE_MESH = "simple"


class SolverType(str, Enum):
    """Supported OpenFOAM solvers"""
    SIMPLE_FOAM = "simpleFoam"
    INTERFOAM = "interFoam"
    RHOSIMPLEFOAM = "rhoSimpleFoam"
    BUOYANTFOAM = "buoyantFoam"
    REACTINGFOAM = "reactingFoam"


class SimulationRequest(BaseModel):
    """Main simulation request from user (natural language)"""
    description: str = Field(..., description="Natural language description of the simulation")
    case_name: str = Field(..., description="Name for the simulation case")
    simulation_type: Optional[SimulationType] = Field(None, description="Override simulation type")
    max_runtime: Optional[int] = Field(3600, description="Maximum simulation runtime in seconds")
    visualization: Optional[bool] = Field(True, description="Generate visualization output")

    class Config:
        examples = [{
            "description": "Simulate flow around a cylinder at 10 m/s with water",
            "case_name": "cylinder_flow_10ms",
            "max_runtime": 1800
        }]


class JSONConfiguration(BaseModel):
    """OpenFOAM configuration in JSON format"""
    simulation_type: SimulationType
    mesh_type: MeshType
    solver: SolverType
    domain: Dict[str, Any]
    boundary_conditions: Dict[str, Any]
    material_properties: Dict[str, Any]
    numerics: Dict[str, Any]
    simulation_parameters: Dict[str, Any]

    class Config:
        json_schema_extra = {
            "example": {
                "simulation_type": "incompressible_flow",
                "mesh_type": "snappyHexMesh",
                "solver": "simpleFoam",
                "domain": {"type": "3D", "length": 1.0},
                "boundary_conditions": {"inlet": "fixedValue", "outlet": "zeroGradient"},
                "material_properties": {"nu": 1e-5},
                "numerics": {"tolerance": 1e-6},
                "simulation_parameters": {"time_steps": 1000}
            }
        }


class SimulationResult(BaseModel):
    """Simulation results"""
    case_id: str
    case_name: str
    status: str
    simulation_type: SimulationType
    configuration: JSONConfiguration
    runtime_seconds: Optional[float] = None
    convergence_status: Optional[str] = None
    residuals: Optional[Dict[str, List[float]]] = None
    output_path: str
    visualization_url: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "case_id": "sim_20240101_001",
                "case_name": "cylinder_flow",
                "status": "completed",
                "simulation_type": "incompressible_flow",
                "runtime_seconds": 1234.5,
                "convergence_status": "converged",
                "output_path": "/tmp/foam_simulations/cylinder_flow"
            }
        }


class LLMConversionResponse(BaseModel):
    """Response from LLM conversion step"""
    original_description: str
    json_configuration: JSONConfiguration
    confidence_score: float = Field(..., ge=0, le=1)
    interpretation_notes: str


class SimulationStatus(BaseModel):
    """Current status of a running simulation"""
    case_id: str
    case_name: str
    status: str  # pending, running, completed, failed
    progress: Optional[float] = Field(None, ge=0, le=100)
    current_step: Optional[str] = None
    error_message: Optional[str] = None
    time_elapsed: Optional[float] = None


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    details: Optional[str] = None
    error_code: int
