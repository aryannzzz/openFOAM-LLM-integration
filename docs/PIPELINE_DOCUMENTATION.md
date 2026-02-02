# LLM-Driven OpenFOAM Orchestration System
## Complete End-to-End Pipeline Documentation

---

## Table of Contents

1. [Introduction & Motivation](#1-introduction--motivation)
2. [System Overview](#2-system-overview)
3. [Pipeline Stages](#3-pipeline-stages)
4. [Detailed Component Breakdown](#4-detailed-component-breakdown)
5. [Data Flow & Transformations](#5-data-flow--transformations)
6. [OpenFOAM Integration](#6-openfoam-integration)
7. [Complete Workflow Example](#7-complete-workflow-example)
8. [Error Handling & Recovery](#8-error-handling--recovery)
9. [Security Considerations](#9-security-considerations)
10. [Deployment Architecture](#10-deployment-architecture)

---

## 1. Introduction & Motivation

### 1.1 The Problem

Computational Fluid Dynamics (CFD) simulations using OpenFOAM require:
- Deep knowledge of fluid mechanics and numerical methods
- Understanding of OpenFOAM's complex directory structure and dictionary syntax
- Manual creation of multiple configuration files (controlDict, fvSchemes, fvSolution, etc.)
- Expertise in mesh generation, boundary conditions, and solver selection
- Time-consuming trial-and-error for parameter tuning

**This creates a significant barrier to entry** for researchers, engineers, and students who need CFD results but lack OpenFOAM expertise.

### 1.2 The Solution

This system provides a **Natural Language Interface** to OpenFOAM:

```
"Simulate turbulent flow around a cylinder at Reynolds number 10000"
                              ↓
                    [Automated Pipeline]
                              ↓
              Complete CFD Results (Cd, Cl, pressure fields, etc.)
```

### 1.3 Key Innovation: The Intermediate Representation (IR)

The core innovation is the **CFDSpecification** - a structured JSON schema that serves as an intermediate representation between natural language and OpenFOAM:

```
Natural Language → [LLM] → CFDSpecification (JSON) → [Generator] → OpenFOAM Case
```

This IR provides:
- **Determinism**: Same JSON always produces same OpenFOAM case
- **Validation**: Physics and security checks before execution
- **Reproducibility**: JSON can be stored, versioned, and shared
- **Security Boundary**: LLM output is constrained to safe parameters

---

## 2. System Overview

### 2.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE LAYER                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │   REST API      │  │   CLI Tool      │  │  Web Interface  │              │
│  │   (FastAPI)     │  │   (Click)       │  │  (Future)       │              │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘              │
└───────────┼────────────────────┼────────────────────┼────────────────────────┘
            │                    │                    │
            └────────────────────┼────────────────────┘
                                 ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│                        LLM CONVERSION LAYER                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     LLM Converter Module                             │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │    │
│  │  │   OpenAI     │  │  Anthropic   │  │    Ollama    │               │    │
│  │  │   GPT-4      │  │   Claude     │  │   (Local)    │               │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │    │
│  │                           ↓                                          │    │
│  │              CFDSpecification (JSON IR)                              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
                                 ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│                        VALIDATION LAYER                                       │
│  ┌─────────────────────────────┐  ┌─────────────────────────────────────┐   │
│  │     Physics Validator       │  │       Security Validator            │   │
│  │  • Solver-regime compat.    │  │  • Injection detection (40+ rules) │   │
│  │  • Reynolds number check    │  │  • Resource limits enforcement      │   │
│  │  • Turbulence model valid.  │  │  • Path traversal blocking          │   │
│  │  • Time dependence check    │  │  • Sandboxing requirements          │   │
│  └─────────────────────────────┘  └─────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
                                 ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│                        CASE GENERATION LAYER                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Case Generator Module                             │    │
│  │                                                                      │    │
│  │  CFDSpecification → OpenFOAM Case Directory Structure                │    │
│  │                                                                      │    │
│  │  Generated Files:                                                    │    │
│  │  • system/controlDict, fvSchemes, fvSolution, blockMeshDict         │    │
│  │  • constant/transportProperties, turbulenceProperties               │    │
│  │  • 0/p, U, k, omega, epsilon, nut (initial/boundary conditions)     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
                                 ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│                        OPENFOAM EXECUTION LAYER                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   OpenFOAM Wrapper Module                            │    │
│  │                                                                      │    │
│  │  Execution Pipeline:                                                 │    │
│  │  1. blockMesh (or snappyHexMesh for complex geometries)             │    │
│  │  2. decomposePar (if parallel)                                       │    │
│  │  3. Solver (icoFoam, simpleFoam, pimpleFoam, etc.)                  │    │
│  │  4. reconstructPar (if parallel)                                     │    │
│  │  5. postProcess -func (force coefficients, etc.)                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
                                 ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│                        POST-PROCESSING LAYER                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   PostProcessor Module                               │    │
│  │                                                                      │    │
│  │  • Field extraction (p, U, turbulence quantities)                   │    │
│  │  • Force coefficient calculation (Cd, Cl)                           │    │
│  │  • Residual convergence analysis                                    │    │
│  │  • Strouhal number computation (via FFT)                            │    │
│  │  • Result aggregation and JSON output                               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
                                 ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│                        RESULTS DELIVERY                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • JSON results file with computed quantities                       │    │
│  │  • VTK files for visualization (ParaView compatible)                │    │
│  │  • Residual plots and convergence data                              │    │
│  │  • Case directory preserved for further analysis                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Pipeline Stages

### Stage 0: User Input
**Input**: Natural language description of desired CFD simulation
**Output**: Text string passed to LLM

```python
# Example inputs:
"Simulate laminar flow around a 2D cylinder at Re=100"
"I need a turbulent simulation of air flow over a backward-facing step at Re=5000"
"Calculate drag on a NACA 0012 airfoil at 5 degrees angle of attack, Re=1e6"
```

### Stage 1: LLM Conversion (NL → JSON)
**Input**: Natural language prompt
**Output**: CFDSpecification JSON

The LLM receives a carefully crafted system prompt that:
1. Defines the exact JSON schema required
2. Lists all valid geometry types, solvers, and turbulence models
3. Provides physics guidance (Re ranges, solver selection)
4. Enforces security constraints (no shell commands, no file paths)

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM SYSTEM PROMPT                            │
├─────────────────────────────────────────────────────────────────┤
│ "You are a CFD specification generator. Your role is to        │
│  translate natural language requests into structured JSON       │
│  specifications for OpenFOAM simulations.                       │
│                                                                 │
│  CRITICAL CONSTRAINTS:                                          │
│  1. Output ONLY valid JSON conforming to CFDSpecification      │
│  2. NEVER include shell commands, file paths, or code          │
│  3. Use ONLY predefined geometry types and solvers             │
│  4. Select appropriate parameters based on physics:            │
│     - Re < 2300 → laminar                                      │
│     - Re > 4000 → turbulent                                    │
│     - Mach < 0.3 → incompressible                              │
│  ..."                                                          │
└─────────────────────────────────────────────────────────────────┘
```

### Stage 2: Schema Validation
**Input**: Raw JSON from LLM
**Output**: Validated CFDSpecification object (or error)

```python
# Pydantic validation ensures:
# - All required fields present
# - Enum values are valid (geometry_type, solver, turbulence_model)
# - Numeric constraints satisfied (Re > 0, viscosity > 0, etc.)
# - Nested objects properly structured

from app.schemas import CFDSpecification
spec = CFDSpecification(**json_from_llm)  # Raises ValidationError if invalid
```

### Stage 3: Physics Validation
**Input**: CFDSpecification object
**Output**: Validation result with warnings/errors

Physics rules enforced:

| Rule | Description | Example |
|------|-------------|---------|
| Solver-Regime Compatibility | Laminar flows can't use simpleFoam | Re=100 + simpleFoam → Error |
| Solver-Time Compatibility | Steady solvers can't do transient | icoFoam + steady → Error |
| Reynolds-Regime Consistency | Re > 4000 should be turbulent | Re=10000 + laminar → Warning |
| Turbulence Model Validity | RANS needs turbulence model | turbulent + none → Error |
| Compressibility Check | High Mach needs compressible solver | Ma=0.8 + simpleFoam → Error |

```python
# Physics validation matrix example
SOLVER_REGIME_MATRIX = {
    SolverType.ICO_FOAM: [FlowRegime.LAMINAR],
    SolverType.SIMPLE_FOAM: [FlowRegime.TURBULENT_RANS],
    SolverType.PIMPLE_FOAM: [FlowRegime.LAMINAR, FlowRegime.TURBULENT_RANS, FlowRegime.TRANSITIONAL],
    # ...
}
```

### Stage 4: Security Validation
**Input**: CFDSpecification object
**Output**: Security check result (pass/fail)

Security checks:
1. **Injection Detection**: Scan all string fields for dangerous patterns
2. **Resource Limits**: Ensure mesh size, CPU, memory within bounds
3. **Path Validation**: Block any path traversal attempts

```python
DANGEROUS_PATTERNS = [
    r'[;&|`$]',           # Shell metacharacters
    r'\.\.',              # Path traversal
    r'(?:rm|mv|cp)\s',    # Destructive commands
    r'(?:curl|wget)\s',   # Network access
    r'__\w+__',           # Python dunder methods
    r'(?:exec|eval)\s*\(', # Code execution
    # ... 40+ patterns total
]
```

### Stage 5: Case Generation
**Input**: Validated CFDSpecification
**Output**: Complete OpenFOAM case directory

The case generator creates the following structure:

```
case_directory/
├── system/
│   ├── controlDict          # Simulation control parameters
│   ├── fvSchemes            # Discretization schemes
│   ├── fvSolution           # Solver settings and tolerances
│   ├── blockMeshDict        # Mesh definition
│   └── decomposeParDict     # Parallel decomposition (if needed)
├── constant/
│   ├── transportProperties  # Fluid properties (nu, rho)
│   ├── turbulenceProperties # Turbulence model selection
│   └── polyMesh/            # (Generated by blockMesh)
└── 0/                       # Initial & boundary conditions
    ├── p                    # Pressure field
    ├── U                    # Velocity field
    ├── k                    # Turbulent kinetic energy (if turbulent)
    ├── omega                # Specific dissipation rate (if k-omega)
    ├── epsilon              # Dissipation rate (if k-epsilon)
    └── nut                  # Turbulent viscosity
```

### Stage 6: Mesh Generation
**Input**: Case directory with blockMeshDict
**Output**: Mesh in constant/polyMesh/

```bash
# For simple geometries (cylinder, channel, etc.)
blockMesh

# For complex geometries (airfoil, car, etc.)
surfaceFeatureExtract
snappyHexMesh -overwrite
```

### Stage 7: Solver Execution
**Input**: Complete case with mesh
**Output**: Time directories with solution fields

```bash
# Solver selection based on CFDSpecification
# Laminar, transient, incompressible:
icoFoam

# Turbulent, steady, incompressible:
simpleFoam

# Turbulent, transient, incompressible:
pimpleFoam

# Compressible flows:
rhoSimpleFoam / rhoPimpleFoam
```

### Stage 8: Post-Processing
**Input**: Completed simulation case
**Output**: Extracted results (JSON + visualization files)

```python
# Extracted quantities:
results = {
    "drag_coefficient": 1.24,
    "lift_coefficient": 0.02,
    "pressure_drop": 45.2,
    "strouhal_number": 0.21,  # Computed via FFT of lift history
    "final_residuals": {
        "p": 1.2e-6,
        "Ux": 3.4e-7,
        "Uy": 2.1e-7
    },
    "convergence": True,
    "iterations": 4523
}
```

---

## 4. Detailed Component Breakdown

### 4.1 CFDSpecification Schema (The IR)

The complete schema with all fields:

```json
{
  "metadata": {
    "name": "string (required) - Case identifier",
    "description": "string (optional) - Human-readable description",
    "version": "string (default: 1.0)",
    "author": "string (optional)",
    "tags": ["array", "of", "strings"]
  },
  
  "geometry": {
    "type": "enum: cylinder_2d | cylinder_3d | sphere | flat_plate_2d | 
             backward_facing_step | channel_2d | channel_3d | 
             airfoil_naca_4digit | box_with_obstacle | pipe_3d",
    "dimensions": {
      "characteristic_length": "float (m) - Reference length for Re calculation",
      "diameter": "float (optional) - For cylinders/spheres",
      "chord": "float (optional) - For airfoils",
      "step_height": "float (optional) - For backward-facing step",
      "span": "float (optional) - For 3D cases"
    },
    "domain": {
      "upstream": "float - Domain lengths in characteristic lengths",
      "downstream": "float",
      "lateral": "float",
      "height": "float (optional, for 3D)"
    },
    "airfoil_params": {
      "naca_code": "string (e.g., '0012')",
      "angle_of_attack": "float (degrees)"
    }
  },
  
  "flow": {
    "regime": "enum: laminar | transitional | turbulent_rans | turbulent_les",
    "reynolds_number": "float (required) - Re = U*L/nu",
    "mach_number": "float (optional, for compressible)",
    "time_dependence": "enum: steady | transient",
    "turbulence_model": "enum: none | kEpsilon | kOmegaSST | 
                         SpalartAllmaras | realizableKE | LES_Smagorinsky"
  },
  
  "fluid": {
    "name": "string (e.g., 'air', 'water')",
    "density": "float (kg/m³)",
    "kinematic_viscosity": "float (m²/s)",
    "dynamic_viscosity": "float (optional, Pa·s)",
    "specific_heat_cp": "float (optional, J/kg·K)",
    "thermal_conductivity": "float (optional, W/m·K)"
  },
  
  "solver": {
    "type": "enum: icoFoam | simpleFoam | pimpleFoam | pisoFoam | 
             rhoSimpleFoam | rhoPimpleFoam",
    "algorithm": "enum: SIMPLE | PISO | PIMPLE",
    "max_iterations": "int (for steady solvers)",
    "convergence_criteria": {
      "p": "float (default: 1e-5)",
      "U": "float (default: 1e-5)",
      "k": "float (optional)",
      "omega": "float (optional)",
      "epsilon": "float (optional)"
    },
    "relaxation_factors": {
      "p": "float (default: 0.3 for SIMPLE)",
      "U": "float (default: 0.7 for SIMPLE)"
    }
  },
  
  "time": {
    "end_time": "float (s) - Total simulation time",
    "delta_t": "float (s) - Time step (for transient)",
    "adjustable_time_step": "bool (default: false)",
    "max_courant": "float (default: 1.0)",
    "write_interval": "float (s) - Output frequency",
    "write_format": "enum: ascii | binary (default: ascii)"
  },
  
  "mesh": {
    "resolution": "enum: coarse | medium | fine | very_fine",
    "base_cell_size": "float (optional, m)",
    "refinement_regions": [
      {
        "type": "box | sphere | cylinder",
        "level": "int (1-5)",
        "bounds": {}
      }
    ],
    "boundary_layer": {
      "enabled": "bool",
      "num_layers": "int (default: 10)",
      "expansion_ratio": "float (default: 1.2)",
      "first_layer_thickness": "float (optional, m)",
      "y_plus_target": "float (optional, for wall functions)"
    }
  },
  
  "boundaries": {
    "inlet": {
      "type": "velocity_inlet | pressure_inlet | mass_flow_inlet",
      "velocity": "[Ux, Uy, Uz] (m/s)",
      "turbulent_intensity": "float (optional, 0-1)",
      "turbulent_length_scale": "float (optional, m)"
    },
    "outlet": {
      "type": "pressure_outlet | outflow | convective",
      "pressure": "float (Pa, gauge)"
    },
    "walls": {
      "type": "no_slip | slip | moving_wall",
      "velocity": "[Ux, Uy, Uz] (optional, for moving walls)"
    },
    "symmetry": {
      "planes": ["front", "back", "top", "bottom"]
    }
  },
  
  "initial_conditions": {
    "velocity": "[Ux, Uy, Uz] (m/s)",
    "pressure": "float (Pa)",
    "turbulent_kinetic_energy": "float (optional, m²/s²)",
    "turbulent_dissipation": "float (optional)"
  },
  
  "outputs": {
    "fields": ["p", "U", "k", "omega", "nut", "vorticity", "Q"],
    "derived_quantities": [
      "drag_coefficient", "lift_coefficient", "pressure_drop",
      "strouhal_number", "mass_flow_rate", "wall_shear_stress"
    ],
    "probes": [
      {"name": "wake_probe", "location": [0.1, 0, 0]}
    ],
    "visualization": {
      "format": "vtk | ensight | raw",
      "fields": ["p", "U"]
    }
  },
  
  "execution": {
    "parallel": "bool",
    "num_processors": "int (1-8)",
    "decomposition_method": "enum: simple | scotch | hierarchical"
  }
}
```

### 4.2 LLM Converter Module

```python
# app/llm_converter.py - Key components

class LLMConverter:
    """Converts natural language to CFDSpecification"""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4"):
        self.provider = provider
        self.model = model
        self.client = self._initialize_client()
    
    def convert_to_cfd_specification(self, prompt: str) -> LLMConversionResponse:
        """
        Main conversion method.
        
        Returns:
            LLMConversionResponse containing:
            - specification: CFDSpecification object
            - confidence_score: 0.0-1.0 indicating LLM confidence
            - interpretation_notes: How the LLM interpreted the request
            - inferred_parameters: List of parameters that were inferred
        """
        if self.client:
            return self._convert_via_llm(prompt)
        else:
            return self._mock_conversion(prompt)  # Rule-based fallback
    
    def _convert_via_llm(self, prompt: str) -> LLMConversionResponse:
        """Call LLM API with system prompt"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Low temperature for consistency
            max_tokens=3000,
            response_format={"type": "json_object"}  # Force JSON output
        )
        return self._parse_response(response)
```

### 4.3 Physics Validator Module

```python
# app/validation.py - Key validation rules

PHYSICS_CONSTRAINTS = {
    "solver_regime_matrix": {
        # Which solvers are valid for which flow regimes
        "icoFoam": ["laminar"],
        "simpleFoam": ["turbulent_rans"],
        "pimpleFoam": ["laminar", "transitional", "turbulent_rans"],
        "pisoFoam": ["laminar", "transitional"],
        "rhoSimpleFoam": ["turbulent_rans"],  # Compressible
        "rhoPimpleFoam": ["laminar", "transitional", "turbulent_rans"],
    },
    
    "solver_time_matrix": {
        # Which solvers support which time dependencies
        "icoFoam": ["transient"],
        "simpleFoam": ["steady"],
        "pimpleFoam": ["transient", "steady"],
        "pisoFoam": ["transient"],
        "rhoSimpleFoam": ["steady"],
        "rhoPimpleFoam": ["transient", "steady"],
    },
    
    "reynolds_regime_guidance": {
        # Recommended regime based on Reynolds number
        (0, 2300): "laminar",
        (2300, 4000): "transitional",
        (4000, float('inf')): "turbulent_rans",
    },
    
    "turbulence_model_regime": {
        # Which turbulence models are valid for which regimes
        "none": ["laminar"],
        "kEpsilon": ["turbulent_rans"],
        "kOmegaSST": ["turbulent_rans", "transitional"],
        "SpalartAllmaras": ["turbulent_rans"],
    }
}

class PhysicsValidator:
    def validate(self, spec: CFDSpecification) -> dict:
        """Run all physics validation checks"""
        errors = []
        warnings = []
        
        # Check solver-regime compatibility
        if spec.flow.regime.value not in \
           PHYSICS_CONSTRAINTS["solver_regime_matrix"][spec.solver.type.value]:
            errors.append(f"Solver {spec.solver.type.value} incompatible with {spec.flow.regime.value}")
        
        # Check Reynolds-regime consistency
        recommended = self._get_recommended_regime(spec.flow.reynolds_number)
        if spec.flow.regime.value != recommended:
            warnings.append(f"Re={spec.flow.reynolds_number} typically uses {recommended}")
        
        # More checks...
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
```

### 4.4 Case Generator Module

```python
# app/case_generator.py - Key generation logic

class CaseGenerator:
    """Generates OpenFOAM case directories from CFDSpecification"""
    
    def generate_case(self, spec: CFDSpecification, output_dir: str):
        """Generate complete OpenFOAM case"""
        
        # Create directory structure
        self._create_directories(output_dir)
        
        # Generate system files
        self._generate_control_dict(spec, output_dir)
        self._generate_fv_schemes(spec, output_dir)
        self._generate_fv_solution(spec, output_dir)
        self._generate_block_mesh_dict(spec, output_dir)
        
        # Generate constant files
        self._generate_transport_properties(spec, output_dir)
        self._generate_turbulence_properties(spec, output_dir)
        
        # Generate initial/boundary condition files
        self._generate_p_field(spec, output_dir)
        self._generate_u_field(spec, output_dir)
        if spec.flow.regime != FlowRegime.LAMINAR:
            self._generate_turbulence_fields(spec, output_dir)
    
    def _generate_control_dict(self, spec: CFDSpecification, output_dir: str):
        """Generate system/controlDict"""
        content = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}

application     {spec.solver.type.value};
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {spec.time.end_time if spec.time else 1000};
deltaT          {spec.time.delta_t if spec.time else 1};
writeControl    {self._get_write_control(spec)};
writeInterval   {spec.time.write_interval if spec.time else 100};
purgeWrite      0;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

functions
{{
    {self._generate_function_objects(spec)}
}}
"""
        self._write_file(f"{output_dir}/system/controlDict", content)
```

### 4.5 Post-Processing Module

```python
# app/postprocessing.py - Result extraction

class PostProcessor:
    """Extract results from completed OpenFOAM simulation"""
    
    def __init__(self, case_dir: str):
        self.case_dir = Path(case_dir)
    
    def extract_results(self, outputs: Outputs) -> dict:
        """Extract all requested outputs"""
        results = {}
        
        # Extract force coefficients
        if "drag_coefficient" in outputs.derived_quantities:
            results["drag_coefficient"] = self._read_force_coefficients("Cd")
        if "lift_coefficient" in outputs.derived_quantities:
            results["lift_coefficient"] = self._read_force_coefficients("Cl")
        
        # Read residuals
        results["residuals"] = self._read_residuals()
        results["convergence"] = self._check_convergence(results["residuals"])
        
        # Compute Strouhal number (for vortex shedding)
        if "strouhal_number" in outputs.derived_quantities:
            results["strouhal_number"] = self._compute_strouhal()
        
        return results
    
    def _read_force_coefficients(self, coefficient: str) -> float:
        """Read force coefficients from postProcessing directory"""
        coeff_file = self.case_dir / "postProcessing/forceCoeffs/0/coefficient.dat"
        # Parse file and extract coefficient
        ...
    
    def _compute_strouhal(self) -> float:
        """Compute Strouhal number via FFT of lift coefficient history"""
        # Read time history of Cl
        cl_history = self._read_force_coefficient_history("Cl")
        
        # Perform FFT
        from numpy.fft import fft, fftfreq
        n = len(cl_history)
        frequencies = fftfreq(n, d=self.delta_t)
        spectrum = np.abs(fft(cl_history))
        
        # Find dominant frequency
        dominant_freq = frequencies[np.argmax(spectrum[1:n//2]) + 1]
        
        # St = f * D / U
        strouhal = dominant_freq * self.characteristic_length / self.velocity
        return strouhal
```

---

## 5. Data Flow & Transformations

### 5.1 Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 0: USER INPUT                                                          │
│ "Simulate turbulent flow around a cylinder at Re=10000"                     │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: LLM CONVERSION                                                      │
│ Input: Natural language string                                               │
│ Process: GPT-4 with CFD-specific system prompt                              │
│ Output: Raw JSON string                                                      │
│                                                                              │
│ {                                                                            │
│   "metadata": {"name": "cylinder_re10000", ...},                            │
│   "geometry": {"type": "cylinder_2d", "dimensions": {...}},                 │
│   "flow": {"regime": "turbulent_rans", "reynolds_number": 10000, ...},      │
│   "solver": {"type": "simpleFoam", ...},                                    │
│   ...                                                                        │
│ }                                                                            │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: SCHEMA VALIDATION                                                   │
│ Input: Raw JSON string                                                       │
│ Process: Pydantic model validation                                          │
│ Output: CFDSpecification Python object (or ValidationError)                 │
│                                                                              │
│ CFDSpecification(                                                           │
│   metadata=Metadata(name="cylinder_re10000", ...),                          │
│   geometry=Geometry(type=GeometryType.CYLINDER_2D, ...),                    │
│   flow=Flow(regime=FlowRegime.TURBULENT_RANS, reynolds_number=10000, ...),  │
│   ...                                                                        │
│ )                                                                            │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: PHYSICS VALIDATION                                                  │
│ Input: CFDSpecification object                                              │
│ Process: Check solver-regime compatibility, Reynolds consistency, etc.      │
│ Output: ValidationResult(valid=True, errors=[], warnings=[...])             │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: SECURITY VALIDATION                                                 │
│ Input: CFDSpecification object                                              │
│ Process: Scan for injection patterns, check resource limits                 │
│ Output: SecurityResult(valid=True, threats_detected=[])                     │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: CASE GENERATION                                                     │
│ Input: Validated CFDSpecification                                           │
│ Process: Template-based OpenFOAM dictionary generation                      │
│ Output: OpenFOAM case directory structure                                   │
│                                                                              │
│ /data/cases/sim_20240131_143022_a1b2c3d4/                                   │
│ ├── system/                                                                 │
│ │   ├── controlDict         ← Solver control, time stepping                │
│ │   ├── fvSchemes           ← Discretization schemes                       │
│ │   ├── fvSolution          ← Linear solver settings                       │
│ │   └── blockMeshDict       ← Mesh definition                              │
│ ├── constant/                                                               │
│ │   ├── transportProperties ← nu = 1.5e-5                                  │
│ │   └── turbulenceProperties ← RAS + kOmegaSST                            │
│ └── 0/                                                                      │
│     ├── p                   ← Pressure BC: inlet=zeroGradient, outlet=0    │
│     ├── U                   ← Velocity BC: inlet=(1.5,0,0), wall=noSlip    │
│     ├── k                   ← Turbulent KE initial + BC                    │
│     ├── omega               ← Specific dissipation initial + BC            │
│     └── nut                 ← Turbulent viscosity (calculated)             │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 6: MESH GENERATION                                                     │
│ Input: blockMeshDict in case directory                                      │
│ Process: Execute blockMesh (or snappyHexMesh)                               │
│ Output: constant/polyMesh/ directory with mesh files                        │
│                                                                              │
│ $ blockMesh                                                                 │
│ --> Creating block edges                                                    │
│ --> Creating block faces                                                    │
│ --> Creating points with Grading                                            │
│ --> Creating topology blocks                                                │
│ --> Creating cell shapes                                                    │
│     Cells: 45000                                                            │
│     Points: 91202                                                           │
│     Faces: 135100                                                           │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 7: SOLVER EXECUTION                                                    │
│ Input: Complete case with mesh                                              │
│ Process: Execute OpenFOAM solver (simpleFoam for this case)                 │
│ Output: Time directories with solution fields                               │
│                                                                              │
│ $ simpleFoam                                                                │
│ Starting time loop                                                          │
│ Time = 1                                                                    │
│ smoothSolver: Solving for Ux, Initial residual = 0.123, Final = 1.2e-6     │
│ smoothSolver: Solving for Uy, Initial residual = 0.089, Final = 8.9e-7     │
│ GAMG: Solving for p, Initial residual = 0.234, Final = 2.3e-7              │
│ ...                                                                         │
│ Time = 5000                                                                 │
│ Converged in 4523 iterations                                                │
│                                                                              │
│ Generated directories: 100/, 200/, 300/, ..., 5000/                         │
│ Each containing: p, U, k, omega, nut, phi                                   │
└─────────────────────────────────┬───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 8: POST-PROCESSING                                                     │
│ Input: Completed simulation case                                            │
│ Process: Extract fields, compute derived quantities                         │
│ Output: JSON results + VTK visualization files                              │
│                                                                              │
│ $ postProcess -func 'forceCoeffs'                                           │
│ Reading forceCoeffs settings from controlDict                               │
│ Writing force coefficients to postProcessing/forceCoeffs/                   │
│                                                                              │
│ Results:                                                                    │
│ {                                                                           │
│   "job_id": "sim_20240131_143022_a1b2c3d4",                                │
│   "status": "completed",                                                    │
│   "results": {                                                              │
│     "drag_coefficient": 1.18,                                               │
│     "lift_coefficient": 0.003,                                              │
│     "final_residuals": {"p": 2.3e-7, "Ux": 1.2e-6, "Uy": 8.9e-7},         │
│     "iterations": 4523,                                                     │
│     "convergence": true                                                     │
│   },                                                                        │
│   "case_path": "/data/cases/sim_20240131_143022_a1b2c3d4",                 │
│   "result_path": "/data/results/sim_20240131_143022_a1b2c3d4"              │
│ }                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Key Transformations

| Stage | Input Format | Output Format | Transformation |
|-------|--------------|---------------|----------------|
| 0→1 | String (NL) | String (JSON) | LLM inference |
| 1→2 | String (JSON) | Python object | Pydantic parsing |
| 2→3 | CFDSpecification | ValidationResult | Rule evaluation |
| 3→4 | CFDSpecification | SecurityResult | Pattern matching |
| 4→5 | CFDSpecification | File system | Template rendering |
| 5→6 | blockMeshDict | polyMesh/ | OpenFOAM meshing |
| 6→7 | Case directory | Time dirs | CFD solving |
| 7→8 | Time dirs | JSON + VTK | Data extraction |

---

## 6. OpenFOAM Integration

### 6.1 OpenFOAM Directory Structure

OpenFOAM requires a specific directory structure for each simulation case:

```
case/
├── system/                    # Simulation control
│   ├── controlDict            # Main control file
│   ├── fvSchemes              # Discretization schemes
│   ├── fvSolution             # Solver/preconditioner settings
│   ├── blockMeshDict          # Mesh definition (for blockMesh)
│   ├── snappyHexMeshDict      # Complex mesh (optional)
│   └── decomposeParDict       # Parallel decomposition (optional)
│
├── constant/                  # Physical properties (time-invariant)
│   ├── transportProperties    # Fluid properties (viscosity, etc.)
│   ├── turbulenceProperties   # Turbulence model selection
│   ├── polyMesh/              # Mesh files (generated)
│   │   ├── points             # Vertex coordinates
│   │   ├── faces              # Face definitions
│   │   ├── owner              # Face-to-cell mapping
│   │   ├── neighbour          # Internal face neighbors
│   │   └── boundary           # Boundary patch definitions
│   └── triSurface/            # STL files (for snappyHexMesh)
│
└── 0/                         # Initial conditions (t=0)
    ├── p                      # Pressure field
    ├── U                      # Velocity field
    ├── k                      # Turbulent kinetic energy (turbulent)
    ├── omega                  # Specific dissipation (k-omega models)
    ├── epsilon                # Dissipation rate (k-epsilon models)
    └── nut                    # Turbulent viscosity
```

### 6.2 Key OpenFOAM Files Generated

#### 6.2.1 controlDict (Simulation Control)

```cpp
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}

application     simpleFoam;      // Solver to use
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         5000;            // Run for 5000 iterations
deltaT          1;               // Pseudo-time step (steady)
writeControl    timeStep;
writeInterval   100;             // Write every 100 iterations
purgeWrite      3;               // Keep last 3 time dirs
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

// Function objects for post-processing
functions
{
    forceCoeffs
    {
        type            forceCoeffs;
        libs            (forces);
        writeControl    timeStep;
        writeInterval   1;
        
        patches         (cylinder);  // Patch to compute forces on
        rho             rhoInf;
        rhoInf          1.225;       // Freestream density
        liftDir         (0 1 0);
        dragDir         (1 0 0);
        CofR            (0 0 0);     // Center of rotation
        pitchAxis       (0 0 1);
        magUInf         1.5;         // Freestream velocity
        lRef            0.01;        // Reference length
        Aref            0.0001;      // Reference area
    }
    
    fieldAverage
    {
        type            fieldAverage;
        libs            (fieldFunctionObjects);
        writeControl    writeTime;
        
        fields
        (
            U { mean on; prime2Mean on; base time; }
            p { mean on; prime2Mean on; base time; }
        );
    }
}
```

#### 6.2.2 fvSchemes (Discretization)

```cpp
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}

ddtSchemes
{
    default         steadyState;    // Steady-state simulation
}

gradSchemes
{
    default         Gauss linear;
    grad(p)         Gauss linear;
    grad(U)         cellLimited Gauss linear 1;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);  // Convection
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}

wallDist
{
    method          meshWave;
}
```

#### 6.2.3 fvSolution (Solver Settings)

```cpp
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}

solvers
{
    p
    {
        solver          GAMG;           // Geometric Algebraic Multigrid
        smoother        GaussSeidel;
        tolerance       1e-7;
        relTol          0.01;
    }
    
    pFinal
    {
        $p;
        relTol          0;
    }
    
    "(U|k|omega)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 1;
    consistent      yes;              // SIMPLEC algorithm
    
    residualControl
    {
        p               1e-5;
        U               1e-5;
        "(k|omega)"     1e-5;
    }
}

relaxationFactors
{
    fields
    {
        p               0.3;
    }
    equations
    {
        U               0.7;
        k               0.7;
        omega           0.7;
    }
}
```

#### 6.2.4 Boundary Conditions (0/U)

```cpp
FoamFile
{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}

dimensions      [0 1 -1 0 0 0 0];    // m/s

internalField   uniform (1.5 0 0);   // Initial velocity

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform (1.5 0 0);
    }
    
    outlet
    {
        type            zeroGradient;
    }
    
    cylinder
    {
        type            noSlip;          // Wall boundary condition
    }
    
    top
    {
        type            symmetryPlane;
    }
    
    bottom
    {
        type            symmetryPlane;
    }
    
    frontAndBack
    {
        type            empty;           // 2D simulation
    }
}
```

### 6.3 OpenFOAM Execution Sequence

```bash
#!/bin/bash
# Complete OpenFOAM execution sequence

# 1. Source OpenFOAM environment
source /opt/openfoam/etc/bashrc

# 2. Navigate to case directory
cd /data/cases/sim_20240131_143022_a1b2c3d4

# 3. Generate mesh
blockMesh
# Output: Created mesh with 45000 cells

# 4. Check mesh quality
checkMesh
# Output: Mesh OK (no failed checks)

# 5. Initialize parallel decomposition (if parallel)
decomposePar -force
# Output: Decomposed into 4 processor directories

# 6. Run solver
# Serial:
simpleFoam | tee log.simpleFoam
# Parallel:
mpirun -np 4 simpleFoam -parallel | tee log.simpleFoam

# 7. Reconstruct parallel results
reconstructPar
# Output: Reconstructed time directories

# 8. Post-process
postProcess -func forceCoeffs
# Output: Written force coefficients to postProcessing/

# 9. Convert to VTK for visualization
foamToVTK -latestTime
# Output: VTK files in VTK/ directory
```

---

## 7. Complete Workflow Example

### 7.1 Example: Flow Around a Cylinder at Re=100

Let's trace through a complete example from user input to final results.

#### Step 1: User Submits Request

```bash
# Via CLI
python cli.py run "Simulate laminar vortex shedding around a 2D cylinder at Re=100"

# Via API
curl -X POST http://localhost:8000/api/v1/simulations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Simulate laminar vortex shedding around a 2D cylinder at Re=100"}'
```

#### Step 2: LLM Conversion

The LLM receives the prompt with system instructions and generates:

```json
{
  "metadata": {
    "name": "cylinder_vortex_shedding_re100",
    "description": "2D cylinder flow with vortex shedding at Re=100",
    "version": "1.0"
  },
  "geometry": {
    "type": "cylinder_2d",
    "dimensions": {
      "characteristic_length": 0.01,
      "diameter": 0.01
    },
    "domain": {
      "upstream": 10,
      "downstream": 30,
      "lateral": 10
    }
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
    "type": "pisoFoam",
    "algorithm": "PISO",
    "max_iterations": 5000,
    "convergence_criteria": {
      "p": 1e-6,
      "U": 1e-6
    }
  },
  "time": {
    "end_time": 5.0,
    "delta_t": 0.0001,
    "adjustable_time_step": true,
    "max_courant": 0.5,
    "write_interval": 0.1
  },
  "mesh": {
    "resolution": "medium",
    "boundary_layer": {
      "enabled": true,
      "num_layers": 15,
      "expansion_ratio": 1.15
    }
  },
  "boundaries": {
    "inlet": {
      "type": "velocity_inlet",
      "velocity": [0.15, 0, 0]
    },
    "outlet": {
      "type": "pressure_outlet",
      "pressure": 0
    },
    "walls": {
      "type": "no_slip"
    },
    "symmetry": {
      "planes": ["front", "back"]
    }
  },
  "initial_conditions": {
    "velocity": [0.15, 0, 0],
    "pressure": 0
  },
  "outputs": {
    "fields": ["p", "U"],
    "derived_quantities": ["drag_coefficient", "lift_coefficient", "strouhal_number"]
  },
  "execution": {
    "parallel": false,
    "num_processors": 1
  }
}
```

#### Step 3: Validation

**Physics Validation:**
```
✓ Solver pisoFoam compatible with laminar regime
✓ Solver pisoFoam supports transient simulations
✓ Re=100 appropriate for laminar flow (Re < 2300)
✓ No turbulence model needed for laminar flow
✓ Time step dt=0.0001 gives CFL ≈ 0.15 (< max 0.5) ✓
```

**Security Validation:**
```
✓ No dangerous patterns detected in any string field
✓ Mesh cells estimate: ~50,000 (< max 10,000,000) ✓
✓ Processors: 1 (< max 8) ✓
✓ Runtime estimate: ~2 hours (< max 4 hours) ✓
```

#### Step 4: Case Generation

Generated directory structure:
```
/data/cases/sim_20240131_150000_re100/
├── system/
│   ├── controlDict      # pisoFoam, endTime=5.0, deltaT=0.0001
│   ├── fvSchemes        # Crank-Nicolson time, Gauss linear
│   ├── fvSolution       # PISO, 2 correctors
│   └── blockMeshDict    # O-grid around cylinder
├── constant/
│   ├── transportProperties  # nu = 1.5e-5
│   └── turbulenceProperties # laminar
└── 0/
    ├── p                # fixedValue inlet, fixedValue outlet
    └── U                # fixedValue inlet (0.15,0,0), noSlip cylinder
```

#### Step 5: Mesh Generation

```bash
$ blockMesh
Creating block mesh topology
--> Creating block edges
--> Creating block faces
--> Creating points with Grading
--> Creating topology blocks
--> Creating cell shapes
--> Creating cells

Creating boundary patches
--> Creating patch 'inlet' (patch)
--> Creating patch 'outlet' (patch)
--> Creating patch 'cylinder' (wall)
--> Creating patch 'top' (symmetryPlane)
--> Creating patch 'bottom' (symmetryPlane)
--> Creating patch 'frontAndBack' (empty)

Mesh Information:
  nPoints: 91202
  nCells: 45000
  nFaces: 135100
  nInternalFaces: 89100
```

#### Step 6: Solver Execution

```bash
$ pisoFoam
Starting time loop

Time = 0.0001
Courant Number mean: 0.012 max: 0.148
PISO: iteration 1
smoothSolver: Solving for Ux, Initial residual = 0.0234, Final = 1.2e-6
smoothSolver: Solving for Uy, Initial residual = 0.0189, Final = 9.8e-7
GAMG: Solving for p, Initial residual = 0.345, Final = 2.3e-7
PISO: iteration 2
...

Time = 2.5
Courant Number mean: 0.089 max: 0.42
Vortex shedding established, periodic behavior observed

Time = 5.0
Courant Number mean: 0.091 max: 0.45
End
```

#### Step 7: Post-Processing Results

```json
{
  "job_id": "sim_20240131_150000_re100",
  "status": "completed",
  "wall_time_seconds": 7234,
  "results": {
    "drag_coefficient": {
      "mean": 1.35,
      "amplitude": 0.02,
      "min": 1.33,
      "max": 1.37
    },
    "lift_coefficient": {
      "mean": 0.0,
      "amplitude": 0.34,
      "min": -0.34,
      "max": 0.34
    },
    "strouhal_number": 0.165,
    "vortex_shedding_frequency": 2.475,
    "final_residuals": {
      "p": 2.3e-7,
      "Ux": 1.2e-6,
      "Uy": 9.8e-7
    },
    "convergence": true,
    "time_steps": 50000
  },
  "files": {
    "case_directory": "/data/cases/sim_20240131_150000_re100",
    "results_json": "/data/results/sim_20240131_150000_re100/results.json",
    "vtk_files": "/data/results/sim_20240131_150000_re100/VTK/"
  },
  "validation_notes": [
    "Strouhal number St=0.165 matches literature (St≈0.16-0.17 for Re=100)",
    "Drag coefficient Cd=1.35 within expected range (1.2-1.5)",
    "Symmetric lift oscillation confirms proper vortex shedding"
  ]
}
```

---

## 8. Error Handling & Recovery

### 8.1 Error Code Reference

| Code | Name | Description | HTTP Status | Recovery |
|------|------|-------------|-------------|----------|
| E001 | INVALID_PROMPT | LLM couldn't parse input | 400 | Rephrase with specific parameters |
| E002 | SCHEMA_ERROR | JSON doesn't match schema | 400 | Check CFDSpecification format |
| E003 | PHYSICS_ERROR | Physics validation failed | 422 | Adjust solver/regime combination |
| E004 | SECURITY_ERROR | Security check failed | 403 | Remove dangerous patterns |
| E005 | GENERATION_ERROR | Case generation failed | 500 | Check template availability |
| E006 | MESH_ERROR | Meshing failed | 500 | Reduce resolution or fix geometry |
| E007 | SOLVER_ERROR | Solver diverged | 500 | Adjust relaxation/time step |
| E008 | RESOURCE_ERROR | Exceeded limits | 413 | Reduce mesh size or runtime |
| E009 | TIMEOUT_ERROR | Execution timed out | 408 | Increase timeout or simplify |
| E010 | POSTPROCESS_ERROR | Result extraction failed | 500 | Check output availability |
| E011 | LLM_ERROR | LLM API failure | 502 | Retry or use fallback |
| E012 | IO_ERROR | File system error | 500 | Check disk space/permissions |
| E013 | INTERNAL_ERROR | Unexpected error | 500 | Check logs, report bug |

### 8.2 Error Response Format

```json
{
  "error_code": "E007",
  "message": "Solver diverged after 234 iterations",
  "details": {
    "last_residual_p": 1.2e+10,
    "last_residual_U": 8.5e+8,
    "divergence_time": 0.0234,
    "log_excerpt": "FOAM FATAL ERROR: Maximum number of iterations exceeded..."
  },
  "recovery_hint": "Try reducing relaxation factors (p: 0.2, U: 0.5) or use smaller time step",
  "timestamp": "2024-01-31T15:30:45Z",
  "job_id": "sim_20240131_153000_diverge"
}
```

### 8.3 Automatic Recovery Strategies

```python
# Example: Automatic solver divergence recovery
class SolverRecovery:
    def attempt_recovery(self, error: SolverError, spec: CFDSpecification) -> CFDSpecification:
        if "divergence" in error.message.lower():
            # Strategy 1: Reduce relaxation factors
            spec.solver.relaxation_factors.p *= 0.7
            spec.solver.relaxation_factors.U *= 0.7
            
            # Strategy 2: Reduce time step (for transient)
            if spec.time:
                spec.time.delta_t *= 0.5
            
            # Strategy 3: Increase mesh resolution (if coarse)
            if spec.mesh.resolution == MeshResolution.COARSE:
                spec.mesh.resolution = MeshResolution.MEDIUM
            
        return spec
```

---

## 9. Security Considerations

### 9.1 Threat Model

| Threat | Attack Vector | Mitigation |
|--------|--------------|------------|
| Command Injection | Malicious prompt → shell command in JSON | Pattern detection, no shell execution |
| Path Traversal | `../../etc/passwd` in file paths | Path validation, sandboxed directories |
| Resource Exhaustion | Request 1 billion cells | Resource limits enforced |
| Denial of Service | Many concurrent requests | Rate limiting, job queuing |
| Data Exfiltration | Steal results from other users | User isolation, authentication |

### 9.2 Security Validation Patterns

```python
DANGEROUS_PATTERNS = [
    # Shell metacharacters
    r'[;&|`$]',
    r'\$\{',
    r'\$\(',
    
    # Path traversal
    r'\.\.',
    r'\./',
    r'/etc/',
    r'/proc/',
    
    # Command execution
    r'\b(?:rm|mv|cp|cat|curl|wget|nc|bash|sh|python|perl)\s',
    r'\b(?:exec|eval|system|popen|subprocess)\s*\(',
    
    # OpenFOAM-specific injection
    r'#include\s*[<"]',
    r'\bFoamFile\b',
    r'\b(?:code|codeInclude|codeOptions)\b',
    
    # Network access
    r'(?:http|ftp|ssh|telnet)://',
    r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    
    # Python/code injection
    r'__\w+__',
    r'\bimport\s+',
    r'\bfrom\s+\w+\s+import\b',
]
```

### 9.3 Sandboxed Execution

```yaml
# Docker security configuration
services:
  sandbox:
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp:size=1G
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    ulimits:
      nproc: 64
      nofile:
        soft: 65536
        hard: 65536
    mem_limit: 16g
    cpus: 8
```

---

## 10. Deployment Architecture

### 10.1 Docker Compose Stack

```yaml
version: '3.8'

services:
  # API Server
  api:
    build:
      context: .
      target: orchestration
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379/0
      - CELERY_BROKER_URL=redis://redis:6379/0
    volumes:
      - cases:/data/cases
      - results:/data/results
    depends_on:
      - redis
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G

  # Background Workers
  worker:
    build:
      context: .
      target: sandbox
    command: celery -A app.celery_worker worker --loglevel=info
    environment:
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - cases:/data/cases
      - results:/data/results
    depends_on:
      - redis
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '4'
          memory: 8G

  # Redis Queue
  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data

volumes:
  cases:
  results:
  redis-data:
```

### 10.2 Production Deployment Checklist

- [ ] Set all API keys via environment variables
- [ ] Configure CORS for production domains
- [ ] Enable HTTPS with valid certificates
- [ ] Set up monitoring (Prometheus/Grafana)
- [ ] Configure log aggregation (ELK/Loki)
- [ ] Set up backup for results volume
- [ ] Configure rate limiting
- [ ] Set up health check endpoints
- [ ] Document recovery procedures
- [ ] Test disaster recovery

---

## Appendix A: Quick Reference

### Common Natural Language Prompts

| Request | Key Parameters Inferred |
|---------|------------------------|
| "Flow around a cylinder at Re=100" | Laminar, transient, icoFoam/pisoFoam |
| "Turbulent flow over backward step Re=5000" | Turbulent RANS, k-omega SST, simpleFoam |
| "NACA 0012 airfoil at 5° AoA, Re=1e6" | Turbulent, steady, simpleFoam |
| "Channel flow Re=10000" | Turbulent, steady, periodic BC |
| "Supersonic flow Ma=2.0" | Compressible, rhoSimpleFoam |

### Solver Selection Guide

| Flow Characteristics | Recommended Solver |
|---------------------|-------------------|
| Laminar + Transient | icoFoam, pisoFoam |
| Laminar + Steady | simpleFoam (with laminar) |
| Turbulent + Steady | simpleFoam |
| Turbulent + Transient | pimpleFoam |
| Compressible + Steady | rhoSimpleFoam |
| Compressible + Transient | rhoPimpleFoam |

### Typical Compute Times

| Case Type | Mesh Size | Cores | Estimated Time |
|-----------|-----------|-------|----------------|
| 2D Cylinder Re=100 | 50k cells | 1 | 1-2 hours |
| 2D Cylinder Re=10000 | 100k cells | 4 | 2-4 hours |
| 3D Sphere Re=1000 | 500k cells | 8 | 4-8 hours |
| Airfoil 2D Re=1e6 | 200k cells | 4 | 1-2 hours |

---

*Document Version: 2.0*  
*Last Updated: January 31, 2026*  
*Author: LLM-OpenFOAM Orchestration System Team*
