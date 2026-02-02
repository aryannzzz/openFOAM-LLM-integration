"""
CFDSpecification JSON Schema - The Intermediate Representation (IR)
This is the core schema that sits between LLM output and OpenFOAM case generation.
Based on the design document specifications.
"""
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Dict, Any, List, Literal
from enum import Enum


# ============================================================================
# ENUMS - Strictly defined allowed values
# ============================================================================

class GeometryType(str, Enum):
    """Predefined geometry types with templates"""
    CYLINDER_2D = "cylinder_2d"
    CYLINDER_3D = "cylinder_3d"
    SPHERE = "sphere"
    FLAT_PLATE_2D = "flat_plate_2d"
    BACKWARD_FACING_STEP = "backward_facing_step"
    CHANNEL_2D = "channel_2d"
    CHANNEL_3D = "channel_3d"
    AIRFOIL_NACA_4DIGIT = "airfoil_naca_4digit"
    BOX_WITH_OBSTACLE = "box_with_obstacle"
    PIPE_3D = "pipe_3d"


class FlowRegime(str, Enum):
    """Flow regime classification"""
    LAMINAR = "laminar"
    TURBULENT_RANS = "turbulent_rans"
    TURBULENT_LES = "turbulent_les"


class TimeDependence(str, Enum):
    """Time dependence of simulation"""
    STEADY = "steady"
    TRANSIENT = "transient"


class TurbulenceModel(str, Enum):
    """Supported turbulence models"""
    NONE = "none"
    K_EPSILON = "kEpsilon"
    K_OMEGA = "kOmega"
    K_OMEGA_SST = "kOmegaSST"
    SPALART_ALLMARAS = "SpalartAllmaras"
    SMAGORINSKY = "Smagorinsky"
    WALE = "WALE"
    DYNAMIC_K_EQN = "dynamicKEqn"


class SolverType(str, Enum):
    """Supported OpenFOAM solvers"""
    ICO_FOAM = "icoFoam"
    SIMPLE_FOAM = "simpleFoam"
    PIMPLE_FOAM = "pimpleFoam"
    PISO_FOAM = "pisoFoam"
    RHO_SIMPLE_FOAM = "rhoSimpleFoam"
    RHO_PIMPLE_FOAM = "rhoPimpleFoam"


class SolverAlgorithm(str, Enum):
    """Pressure-velocity coupling algorithms"""
    SIMPLE = "SIMPLE"
    PISO = "PISO"
    PIMPLE = "PIMPLE"


class BoundaryType(str, Enum):
    """Boundary condition types"""
    VELOCITY_INLET = "velocity_inlet"
    PRESSURE_INLET = "pressure_inlet"
    PRESSURE_OUTLET = "pressure_outlet"
    FREESTREAM = "freestream"
    NO_SLIP = "no_slip"
    SLIP = "slip"
    SYMMETRY = "symmetry"
    PERIODIC = "periodic"


class MeshResolution(str, Enum):
    """Mesh resolution levels"""
    COARSE = "coarse"
    MEDIUM = "medium"
    FINE = "fine"
    VERY_FINE = "very_fine"


class SimulationStatus(str, Enum):
    """Simulation execution status"""
    QUEUED = "queued"
    PARSING = "parsing"
    VALIDATING = "validating"
    GENERATING = "generating"
    MESHING = "meshing"
    RUNNING = "running"
    POSTPROCESSING = "postprocessing"
    COMPLETED = "completed"
    FAILED = "failed"


# ============================================================================
# SCHEMA COMPONENTS
# ============================================================================

class Metadata(BaseModel):
    """Simulation metadata"""
    name: str = Field(..., min_length=1, max_length=100, pattern=r'^[a-zA-Z0-9_-]+$')
    description: str = Field(..., max_length=500)
    version: str = Field(default="1.0")


class GeometryDimensions(BaseModel):
    """Geometry dimensions"""
    characteristic_length: float = Field(..., gt=0, description="Characteristic length in meters")
    # NACA airfoil specific
    naca_code: Optional[str] = Field(None, pattern=r'^\d{4}$')
    chord: Optional[float] = Field(None, gt=0)
    angle_of_attack: Optional[float] = Field(None, ge=-90, le=90)


class GeometryDomain(BaseModel):
    """Computational domain extent (in characteristic lengths)"""
    upstream: float = Field(default=10, gt=0)
    downstream: float = Field(default=30, gt=0)
    lateral: float = Field(default=10, gt=0)
    height: Optional[float] = Field(None, gt=0)  # For 3D


class Geometry(BaseModel):
    """Geometry specification"""
    type: GeometryType
    dimensions: GeometryDimensions
    domain: GeometryDomain = Field(default_factory=GeometryDomain)


class Flow(BaseModel):
    """Flow conditions"""
    regime: FlowRegime
    reynolds_number: float = Field(..., gt=0, le=1e9)
    time_dependence: TimeDependence = Field(default=TimeDependence.STEADY)
    turbulence_model: TurbulenceModel = Field(default=TurbulenceModel.NONE)
    mach_number: Optional[float] = Field(None, ge=0, le=10)


class Fluid(BaseModel):
    """Fluid properties (SI units)"""
    name: str = Field(default="air")
    density: float = Field(default=1.225, gt=0, description="kg/m³")
    kinematic_viscosity: float = Field(default=1.5e-5, gt=0, description="m²/s")
    specific_heat: Optional[float] = Field(None, gt=0, description="J/(kg·K)")
    thermal_conductivity: Optional[float] = Field(None, gt=0, description="W/(m·K)")


class ConvergenceCriteria(BaseModel):
    """Solver convergence criteria"""
    p: float = Field(default=1e-5, gt=0, le=1)
    U: float = Field(default=1e-5, gt=0, le=1)
    k: Optional[float] = Field(None, gt=0, le=1)
    omega: Optional[float] = Field(None, gt=0, le=1)
    epsilon: Optional[float] = Field(None, gt=0, le=1)


class Solver(BaseModel):
    """Solver configuration"""
    type: SolverType
    algorithm: SolverAlgorithm = Field(default=SolverAlgorithm.SIMPLE)
    max_iterations: int = Field(default=5000, gt=0, le=100000)
    convergence_criteria: ConvergenceCriteria = Field(default_factory=ConvergenceCriteria)


class TimeSettings(BaseModel):
    """Time settings for transient simulations"""
    end_time: float = Field(..., gt=0)
    delta_t: float = Field(..., gt=0)
    adjustable_time_step: bool = Field(default=True)
    max_courant: float = Field(default=1.0, gt=0, le=10)
    write_interval: float = Field(..., gt=0)


class BoundaryLayer(BaseModel):
    """Boundary layer mesh settings"""
    enabled: bool = Field(default=True)
    num_layers: int = Field(default=10, ge=1, le=50)
    expansion_ratio: float = Field(default=1.2, ge=1.0, le=2.0)
    target_y_plus: float = Field(default=1, gt=0, le=300)


class Mesh(BaseModel):
    """Mesh configuration"""
    resolution: MeshResolution = Field(default=MeshResolution.MEDIUM)
    boundary_layer: BoundaryLayer = Field(default_factory=BoundaryLayer)
    max_cells: Optional[int] = Field(None, gt=0, le=10_000_000)


class InletBoundary(BaseModel):
    """Inlet boundary condition"""
    type: Literal["velocity_inlet", "pressure_inlet", "freestream"]
    velocity: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    pressure: Optional[float] = None
    
    @field_validator('velocity')
    @classmethod
    def validate_velocity(cls, v):
        if v is not None:
            if len(v) != 3:
                raise ValueError("Velocity must have 3 components [Ux, Uy, Uz]")
        return v


class OutletBoundary(BaseModel):
    """Outlet boundary condition"""
    type: Literal["pressure_outlet", "freestream"]
    pressure: float = Field(default=0)


class WallBoundary(BaseModel):
    """Wall boundary condition"""
    type: Literal["no_slip", "slip"]


class SymmetryBoundary(BaseModel):
    """Symmetry boundary condition"""
    planes: List[str] = Field(default_factory=list)


class Boundaries(BaseModel):
    """All boundary conditions"""
    inlet: InletBoundary
    outlet: OutletBoundary
    walls: WallBoundary
    symmetry: Optional[SymmetryBoundary] = None


class InitialConditions(BaseModel):
    """Initial field values"""
    velocity: List[float] = Field(..., min_length=3, max_length=3)
    pressure: float = Field(default=0)
    k: Optional[float] = None
    omega: Optional[float] = None
    epsilon: Optional[float] = None
    nut: Optional[float] = None


class VisualizationOutputs(BaseModel):
    """Visualization settings"""
    contour_plots: List[str] = Field(default_factory=lambda: ["p", "U_magnitude"])
    streamlines: bool = Field(default=False)
    vector_plots: List[str] = Field(default_factory=list)


class Outputs(BaseModel):
    """Output configuration"""
    fields: List[str] = Field(default_factory=lambda: ["p", "U"])
    derived_quantities: List[str] = Field(default_factory=list)
    visualization: VisualizationOutputs = Field(default_factory=VisualizationOutputs)
    residual_plots: bool = Field(default=True)


class Execution(BaseModel):
    """Execution settings"""
    parallel: bool = Field(default=False)
    num_processors: int = Field(default=1, ge=1, le=64)


# ============================================================================
# MAIN CFD SPECIFICATION SCHEMA
# ============================================================================

class CFDSpecification(BaseModel):
    """
    Complete CFD Specification - The Intermediate Representation (IR)
    
    This is the core schema that ensures:
    - Deterministic execution
    - Schema validation
    - Security boundary (data only, never code)
    - Full reproducibility
    - Testability
    """
    metadata: Metadata
    geometry: Geometry
    flow: Flow
    fluid: Fluid = Field(default_factory=Fluid)
    solver: Solver
    time: Optional[TimeSettings] = None
    mesh: Mesh = Field(default_factory=Mesh)
    boundaries: Boundaries
    initial_conditions: InitialConditions
    outputs: Outputs = Field(default_factory=Outputs)
    execution: Execution = Field(default_factory=Execution)

    @model_validator(mode='after')
    def validate_transient_has_time(self):
        """Ensure transient simulations have time settings"""
        if self.flow.time_dependence == TimeDependence.TRANSIENT:
            if self.time is None:
                raise ValueError("Transient simulation requires 'time' settings")
        return self

    class Config:
        json_schema_extra = {
            "example": {
                "metadata": {
                    "name": "cylinder_re100_laminar",
                    "description": "Laminar flow over 2D cylinder at Re=100",
                    "version": "1.0"
                },
                "geometry": {
                    "type": "cylinder_2d",
                    "dimensions": {"characteristic_length": 0.01},
                    "domain": {"upstream": 10, "downstream": 30, "lateral": 10}
                },
                "flow": {
                    "regime": "laminar",
                    "reynolds_number": 100,
                    "time_dependence": "transient",
                    "turbulence_model": "none"
                },
                "fluid": {
                    "name": "air",
                    "density": 1.225,
                    "kinematic_viscosity": 1.5e-5
                },
                "solver": {
                    "type": "pimpleFoam",
                    "algorithm": "PIMPLE",
                    "max_iterations": 10000,
                    "convergence_criteria": {"p": 1e-5, "U": 1e-5}
                },
                "time": {
                    "end_time": 2.0,
                    "delta_t": 0.0001,
                    "adjustable_time_step": True,
                    "max_courant": 1.0,
                    "write_interval": 0.1
                },
                "mesh": {
                    "resolution": "medium",
                    "boundary_layer": {
                        "enabled": True,
                        "num_layers": 10,
                        "expansion_ratio": 1.2,
                        "target_y_plus": 1
                    }
                },
                "boundaries": {
                    "inlet": {"type": "velocity_inlet", "velocity": [0.15, 0, 0]},
                    "outlet": {"type": "pressure_outlet", "pressure": 0},
                    "walls": {"type": "no_slip"},
                    "symmetry": {"planes": ["front", "back"]}
                },
                "initial_conditions": {
                    "velocity": [0.15, 0, 0],
                    "pressure": 0
                },
                "outputs": {
                    "fields": ["p", "U"],
                    "derived_quantities": ["drag_coefficient", "lift_coefficient", "strouhal_number"],
                    "visualization": {
                        "contour_plots": ["p", "U_magnitude", "vorticity"],
                        "streamlines": True
                    },
                    "residual_plots": True
                },
                "execution": {
                    "parallel": False,
                    "num_processors": 1
                }
            }
        }


# ============================================================================
# API REQUEST/RESPONSE MODELS
# ============================================================================

class SimulationRequest(BaseModel):
    """Request to run a simulation"""
    prompt: Optional[str] = Field(None, description="Natural language description", max_length=2000)
    specification: Optional[CFDSpecification] = Field(None, description="Direct CFDSpecification JSON")

    @model_validator(mode='after')
    def validate_has_input(self):
        """Ensure either prompt or specification is provided"""
        if not self.prompt and not self.specification:
            raise ValueError("Either 'prompt' or 'specification' must be provided")
        return self


class SimulationStatusResponse(BaseModel):
    """Current status of a simulation"""
    job_id: str
    status: SimulationStatus
    progress: float = Field(ge=0, le=1)
    message: Optional[str] = None
    current_step: Optional[str] = None
    time_elapsed_seconds: Optional[float] = None


class FieldSummary(BaseModel):
    """Summary statistics for a field"""
    type: str
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    magnitude: Optional[Dict[str, float]] = None


class DerivedQuantity(BaseModel):
    """Derived quantity result"""
    final: float
    mean: Optional[float] = None
    rms: Optional[float] = None


class SimulationResult(BaseModel):
    """Complete simulation result"""
    job_id: str
    status: SimulationStatus
    specification: CFDSpecification
    results: Dict[str, Any]
    summary: str
    artifacts: List[str] = Field(default_factory=list)
    runtime_seconds: float
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class LLMConversionResponse(BaseModel):
    """Response from LLM conversion step"""
    original_prompt: str
    specification: CFDSpecification
    confidence_score: float = Field(ge=0, le=1)
    interpretation_notes: str
    inferred_parameters: List[str] = Field(default_factory=list)


class CapabilitiesResponse(BaseModel):
    """System capabilities and constraints"""
    geometries: List[str]
    solvers: List[str]
    turbulence_models: List[str]
    flow_regimes: List[str]
    limits: Dict[str, Any]
