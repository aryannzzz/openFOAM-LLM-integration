# LLM-Driven OpenFOAM Orchestration System

> **Transform natural language into production-ready CFD simulations**

A complete system that bridges the gap between human intent and computational fluid dynamics (CFD) simulations by leveraging Large Language Models (LLMs) to translate natural language descriptions into validated OpenFOAM case configurations.

## Table of Contents

1. [Overview](#overview)
2. [How It Works](#how-it-works)
3. [Quick Start](#quick-start)
4. [Input/Output Formats](#inputoutput-formats)
5. [Architecture](#architecture)
6. [Features & Novelties](#features--novelties)
7. [Use Cases](#use-cases)
8. [API Reference](#api-reference)
9. [Configuration](#configuration)
10. [Testing](#testing)
11. [Troubleshooting](#troubleshooting)

---

## Overview

### What is this?

This system allows users to describe CFD simulations in plain English (or any natural language) and automatically:

1. **Converts** the description to a structured CFD specification using LLMs
2. **Validates** the physics and security constraints
3. **Generates** complete OpenFOAM case files
4. **Executes** the simulation using OpenFOAM
5. **Returns** structured results with velocity fields, pressure data, and derived quantities

### The Problem It Solves

Traditional CFD setup requires:
- Deep knowledge of OpenFOAM dictionary syntax
- Understanding of numerical schemes and solver settings
- Manual creation of dozens of configuration files
- Trial-and-error debugging of incompatible settings

**This system eliminates these barriers** by letting you simply describe what you want to simulate.

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER INPUT                                    │
│  "Simulate laminar flow around a 2D cylinder at Reynolds 100"          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     STEP 1: LLM CONVERSION                              │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────────┐    │
│  │ Natural Lang │ → │ GPT-4 / Claude   │ → │ CFDSpecification  │     │
│  │    Prompt    │    │ + Schema Mapping │    │    (JSON IR)      │     │
│  └──────────────┘    └──────────────────┘    └───────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     STEP 2: PHYSICS VALIDATION                          │
│  • Solver-regime compatibility (icoFoam → laminar only)                │
│  • Solver-time compatibility (simpleFoam → steady only)                │
│  • Reynolds number range guidance                                       │
│  • Turbulence model selection validation                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     STEP 3: SECURITY VALIDATION                         │
│  • No shell commands or code injection                                 │
│  • Resource limits (max cells, CPUs, memory)                           │
│  • Path traversal prevention                                           │
│  • 40+ dangerous pattern checks                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     STEP 4: CASE GENERATION                             │
│  Generates OpenFOAM dictionaries:                                      │
│  • system/controlDict    • constant/transportProperties                │
│  • system/fvSchemes      • constant/momentumTransport                  │
│  • system/fvSolution     • 0/p (pressure field)                        │
│  • system/blockMeshDict  • 0/U (velocity field)                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     STEP 5: OPENFOAM EXECUTION                          │
│  ┌────────────┐    ┌────────────┐    ┌────────────────────────┐       │
│  │ blockMesh  │ → │  foamRun   │ → │ Time-stepped Results   │        │
│  │ (meshing)  │    │  (solver)  │    │ (p, U, phi fields)     │        │
│  └────────────┘    └────────────┘    └────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           OUTPUT                                        │
│  • Velocity field (U) at each time step                                │
│  • Pressure field (p) at each time step                                │
│  • Flux field (phi)                                                    │
│  • Derived quantities (drag/lift coefficients)                         │
│  • JSON specification for reproducibility                              │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- OpenFOAM v11
- OpenAI API key (or Anthropic for Claude)

### Installation

```bash
# Clone the repository
git clone https://github.com/aryannzzz/openFOAM-LLM-integration.git
cd llm-foam-orchestrator

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env with your API key
```

### Running Your First Simulation

```python
from app.llm_converter import LLMConverter
from app.validation import PhysicsValidator
from app.security import SecurityChecker

# Step 1: Convert natural language to specification
converter = LLMConverter(provider="openai", model="gpt-4o-mini")
response = converter.convert_to_cfd_specification(
    "Simulate laminar flow around a 2D cylinder at Reynolds 100"
)
spec = response.specification

# Step 2: Validate physics
validator = PhysicsValidator()
is_valid, errors = validator.validate(spec)
print(f"Physics valid: {is_valid}")

# Step 3: Validate security
checker = SecurityChecker()
is_safe, details = checker.check(spec)
print(f"Security passed: {is_safe}")

# Step 4: Generate and run (see run_real_simulation.py for full example)
```

### Command-Line Quick Test

```bash
# Run the complete E2E test
python run_real_simulation.py

# Run multiple test cases
python run_multiple_tests.py

# Run unit tests
python -m pytest tests/ -v
```

---

## Input/Output Formats

### Input: Natural Language Prompt

The system accepts natural language descriptions of CFD simulations:

```
✓ "Simulate laminar flow around a 2D cylinder at Reynolds number 100"
✓ "Model turbulent flow in a pipe at Re=50000 using k-omega SST"
✓ "Analyze transient vortex shedding behind a cylinder"
✓ "Run CFD analysis of flow over a NACA 0012 airfoil at 5 degrees angle of attack"
✓ "Compute air flow in a square cavity with moving lid"
```

### Intermediate: CFDSpecification (JSON)

The LLM converts the prompt into a structured JSON specification:

```json
{
  "metadata": {
    "name": "cylinder_2d_re100",
    "description": "Laminar flow around 2D cylinder",
    "version": "1.0"
  },
  "geometry": {
    "type": "cylinder_2d",
    "dimensions": {
      "characteristic_length": 0.01
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
    "time_dependence": "steady",
    "turbulence_model": "none"
  },
  "fluid": {
    "name": "air",
    "density": 1.225,
    "kinematic_viscosity": 1.5e-05
  },
  "solver": {
    "type": "simpleFoam",
    "algorithm": "SIMPLE",
    "max_iterations": 5000,
    "convergence_criteria": {
      "p": 1e-05,
      "U": 1e-05
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
    }
  },
  "initial_conditions": {
    "velocity": [0.15, 0, 0],
    "pressure": 0
  },
  "outputs": {
    "fields": ["p", "U"],
    "derived_quantities": ["drag_coefficient", "lift_coefficient"]
  }
}
```

### Output: OpenFOAM Results

The system produces standard OpenFOAM output:

```
case_directory/
├── 0/                          # Initial conditions
│   ├── p                       # Pressure field
│   └── U                       # Velocity field
├── 0.1/                        # Time step 0.1s
│   ├── p
│   ├── U
│   └── phi                     # Flux field
├── 0.2/                        # Time step 0.2s
│   └── ...
├── constant/
│   ├── polyMesh/               # Mesh data
│   ├── transportProperties
│   └── momentumTransport
├── system/
│   ├── controlDict
│   ├── fvSchemes
│   ├── fvSolution
│   └── blockMeshDict
└── cfd_specification.json      # Original specification
```

**Velocity Field Sample (0/U):**
```
dimensions      [0 1 -1 0 0 0 0];

internalField   nonuniform List<vector>
400
(
(-0.00018216 0.000178771 0)
(-0.000653489 0.000200626 0)
...
);
```

---

## Architecture

```
llm-foam-orchestrator/
├── app/
│   ├── __init__.py
│   ├── schemas.py          # CFDSpecification Pydantic models
│   ├── llm_converter.py    # Natural language → CFDSpecification
│   ├── validation.py       # Physics validation rules
│   ├── security.py         # Security checks (40+ patterns)
│   ├── case_generator.py   # OpenFOAM dictionary generation
│   ├── openfoam_wrapper.py # OpenFOAM execution wrapper
│   ├── postprocessing.py   # Result extraction
│   ├── errors.py           # Standardized error codes
│   └── api/
│       └── endpoints.py    # FastAPI REST endpoints
├── tests/
│   ├── test_comprehensive.py  # 37 unit tests
│   └── ...
├── docs/
│   └── PIPELINE_DOCUMENTATION.md
├── run_real_simulation.py     # Single E2E test
├── run_multiple_tests.py      # Batch E2E tests
└── requirements.txt
```

### Key Components

| Component | Purpose |
|-----------|---------|
| `schemas.py` | Pydantic models for type-safe CFD specifications |
| `llm_converter.py` | LLM integration with schema transformation |
| `validation.py` | Physics constraint matrices and validation |
| `security.py` | 40+ security patterns, resource limits |
| `case_generator.py` | OpenFOAM dictionary templates |
| `openfoam_wrapper.py` | Subprocess execution of OpenFOAM |

---

## Features & Novelties

### 1. **Intelligent Schema Transformation**

The system doesn't just rely on LLMs returning perfect JSON. It includes a robust transformation layer that:

- Maps different field names (e.g., `Re` → `reynolds_number`)
- Handles various formats (`turbulenceModel: "laminar"` → `turbulence_model: "none"`)
- Falls back to rule-based conversion when LLM output is incomplete

### 2. **Physics-Aware Validation**

Built-in compatibility matrices ensure valid configurations:

```python
SOLVER_REGIME_MATRIX = {
    "icoFoam": ["laminar"],
    "simpleFoam": ["laminar", "turbulent_rans"],
    "pimpleFoam": ["laminar", "turbulent_rans", "turbulent_les"],
}

SOLVER_TIME_MATRIX = {
    "simpleFoam": ["steady"],
    "icoFoam": ["transient"],
    "pimpleFoam": ["transient", "steady"],
}
```

### 3. **Defense-in-Depth Security**

40+ dangerous patterns are blocked:

```python
DANGEROUS_PATTERNS = [
    r'\$\(.*\)',      # Command substitution
    r'`.*`',          # Backtick execution
    r';\s*rm\s',      # File deletion
    r'&&\s*',         # Command chaining
    r'\|\s*bash',     # Pipe to shell
    # ... 35+ more patterns
]
```

### 4. **Automatic Parameter Inference**

When users don't specify parameters, the system infers reasonable defaults:

- **Reynolds number** → Compute velocity from Re, ν, L
- **Turbulence model** → Select based on flow regime
- **Time step** → Estimate from CFL number
- **Mesh resolution** → Default to "medium"

### 5. **Multi-LLM Support**

Works with multiple LLM providers:

```python
# OpenAI
converter = LLMConverter(provider="openai", model="gpt-4o-mini")

# Anthropic
converter = LLMConverter(provider="anthropic", model="claude-3-sonnet")

# Mock (for testing)
converter = LLMConverter(provider="mock")
```

---

## Use Cases

### 1. **Educational Tool**

Students can learn CFD by describing simulations in plain English and seeing the resulting configurations.

```
Input: "Simulate laminar flow in a pipe"
Output: Complete OpenFOAM case with all dictionaries explained
```

### 2. **Rapid Prototyping**

Engineers can quickly test different flow configurations without manual file editing.

```
Input: "Compare flow at Re=100 vs Re=1000"
→ Two complete simulations with comparison-ready results
```

### 3. **Automated Workflows**

Integrate CFD into larger automation pipelines:

```python
# Batch processing
prompts = [
    "Cylinder at Re=50",
    "Cylinder at Re=100",
    "Cylinder at Re=200",
]
for prompt in prompts:
    run_simulation(prompt)
    extract_drag_coefficient()
```

### 4. **Non-Expert Access**

Domain experts (e.g., aerodynamicists, chemical engineers) can run CFD without learning OpenFOAM syntax.

### 5. **AI-Assisted Design**

Combine with optimization algorithms:

```python
def objective(params):
    prompt = f"Simulate flow at Re={params['re']}, angle={params['angle']}"
    result = run_simulation(prompt)
    return result['drag_coefficient']
    
# Use scipy.optimize or similar
```

---

## API Reference

### LLMConverter

```python
from app.llm_converter import LLMConverter

converter = LLMConverter(provider="openai", model="gpt-4o-mini")
response = converter.convert_to_cfd_specification(prompt)

# Response fields
response.specification      # CFDSpecification object
response.confidence_score   # 0.0 to 1.0
response.interpretation_notes
response.inferred_parameters
```

### PhysicsValidator

```python
from app.validation import PhysicsValidator

validator = PhysicsValidator()
is_valid, errors = validator.validate(spec)

# Error object
for error in errors:
    print(f"{error.severity}: {error.code} - {error.message}")
```

### SecurityChecker

```python
from app.security import SecurityChecker

checker = SecurityChecker()
is_safe, details = checker.check(spec)

# Details dict
details['security_issues']      # List of detected issues
details['resource_violations']  # Resource limit violations
```

### REST API (FastAPI)

```bash
# Start server
uvicorn main:app --reload

# Endpoints
POST /api/v1/simulate
GET  /api/v1/jobs/{job_id}
GET  /api/v1/results/{job_id}
```

---

## Configuration

### Environment Variables (.env)

```bash
# LLM Provider
OPENAI_API_KEY=sk-...
# or
ANTHROPIC_API_KEY=sk-ant-...

# OpenFOAM
OPENFOAM_PATH=/opt/openfoam11
WORKDIR=/tmp/foam_simulations

# Resource Limits
MAX_CELLS=10000000
MAX_PROCESSORS=8
MAX_RUNTIME_HOURS=24
```

### Supported Configurations

| Geometry Types | Flow Regimes | Solvers | Turbulence Models |
|----------------|--------------|---------|-------------------|
| cylinder_2d | laminar | simpleFoam | none |
| cylinder_3d | turbulent_rans | pimpleFoam | kEpsilon |
| sphere | turbulent_les | pisoFoam | kOmega |
| flat_plate_2d | | icoFoam | kOmegaSST |
| channel_2d | | rhoSimpleFoam | SpalartAllmaras |
| channel_3d | | rhoPimpleFoam | Smagorinsky |
| pipe_3d | | | WALE |
| airfoil_naca_4digit | | | |
| backward_facing_step | | | |
| box_with_obstacle | | | |

---

## Testing

### Run All Tests

```bash
# Unit tests (37 tests)
python -m pytest tests/test_comprehensive.py -v

# E2E tests (4 test cases)
python run_multiple_tests.py

# Single E2E test
python run_real_simulation.py
```

### Test Coverage

| Category | Tests | Status |
|----------|-------|--------|
| Schema Creation | 7 | ✅ |
| Physics Validation | 6 | ✅ |
| Security Validation | 3 | ✅ |
| LLM Conversion | 8 | ✅ |
| Integration | 2 | ✅ |
| OpenFOAM Execution | 1 | ✅ |
| Prompt Variations | 10 | ✅ |
| **Total** | **37** | **✅ All Passing** |

---

## Troubleshooting

### Common Issues

**1. "keyword PIMPLE is undefined"**
```
Solution: Ensure fvSolution uses PIMPLE section with pRefCell/pRefValue
```

**2. "keyword viscosityModel is undefined"**
```
Solution: Use OpenFOAM v11 format:
viscosityModel constant;
nu [0 2 -1 0 0 0 0] 1.5e-05;
```

**3. LLM returns incompatible schema**
```
Solution: System automatically falls back to rule-based conversion
Check: response.confidence_score (0.6 = fallback, 0.85 = LLM success)
```

**4. Physics validation fails**
```
Check: Solver-regime compatibility
Example: icoFoam only works with laminar flow
```

### Getting Help

- Check [docs/PIPELINE_DOCUMENTATION.md](docs/PIPELINE_DOCUMENTATION.md) for detailed architecture
- Review test files for usage examples
- Open an issue on GitHub

---

## License

MIT License - see LICENSE file

---

## Contributors

- System designed and implemented for natural language CFD simulation interface
- OpenFOAM is a registered trademark of OpenCFD Ltd.

---

**🚀 Transform your CFD workflow with natural language!**
