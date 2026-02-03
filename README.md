# LLM-Driven OpenFOAM Orchestration System

> **Transform natural language into production-ready CFD simulations**

[![Tests](https://img.shields.io/badge/tests-37%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![OpenFOAM](https://img.shields.io/badge/OpenFOAM-v11-orange)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

A complete system that bridges the gap between human intent and computational fluid dynamics (CFD) simulations. Simply describe what you want to simulate in plain English, and the system automatically generates, validates, and executes OpenFOAM cases.

## 🚀 Quick Example

```python
# Input: Natural language
prompt = "Simulate laminar flow around a 2D cylinder at Reynolds 100"

# Output: Complete CFD simulation with results
# → Velocity field (U), Pressure field (p), Derived quantities
```

**Result:**
```
✓ LLM Conversion (confidence: 0.85)
✓ Physics Validation: PASSED
✓ Security Validation: PASSED  
✓ blockMesh: 882 points, 400 cells
✓ foamRun: 6 time steps (0 → 0.5s)
✓ Results: U, p, phi fields at /tmp/foam_test/cylinder_test/
```

## 📖 Documentation

- **[Complete User Guide](docs/USER_GUIDE.md)** - Full documentation, API reference, use cases
- **[Pipeline Architecture](docs/PIPELINE_DOCUMENTATION.md)** - Technical implementation details
- **[Test Results](TEST_RESULTS.md)** - Comprehensive test coverage

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     System Architecture                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    │
│  │   Natural   │───▶│     LLM      │───▶│  CFDSpecification│   │
│  │  Language   │    │  Converter   │    │     (JSON IR)    │   │
│  └─────────────┘    └──────────────┘    └────────┬─────────┘   │
│                                                   │              │
│  ┌────────────────────────────────────────────────▼────────┐    │
│  │                  Validation Layer                        │   │
│  │  ┌─────────────────┐  ┌──────────────────────────────┐  │   │
│  │  │ Physics Validator│  │    Security Validator        │  │   │
│  │  │ • Solver compat. │  │ • Injection detection        │  │   │
│  │  │ • Reynolds check │  │ • Resource limits            │  │   │
│  │  │ • Turbulence     │  │ • Path traversal blocking    │  │   │
│  │  └─────────────────┘  └──────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                   │                             │
│                                   ▼                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Case Generation Engine                      │   │
│  │  • Template-based OpenFOAM dictionary generation         │   │
│  │  • controlDict, fvSchemes, fvSolution, etc.             │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                   │                             │
│                                   ▼                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    OpenFOAM Execution                    │   │
│  │  blockMesh → [snappyHexMesh] → Solver → postProcess      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                   │                             │
│                                   ▼                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 Post-Processing Module                   │   │
│  │  • Field extraction • Force coefficients • Residuals     │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 🎯 Features

### Core Capabilities
- **Natural Language → CFD**: Convert plain English descriptions to validated simulations
- **Physics Validation**: Automatic solver/turbulence model selection based on Reynolds number
- **Security Enforcement**: Injection attack prevention, resource limits, sandboxed execution
- **Multiple LLM Providers**: OpenAI GPT-4, Anthropic Claude, Ollama (local)

### Supported Configurations
| Geometry Types | Solvers | Turbulence Models |
|---------------|---------|-------------------|
| cylinder_2d/3d | icoFoam | laminar |
| sphere | simpleFoam | kEpsilon |
| airfoil_naca_4digit | pimpleFoam | kOmegaSST |
| backward_facing_step | pisoFoam | SpalartAllmaras |
| channel_2d/3d | rhoSimpleFoam | |
| pipe_3d | rhoPimpleFoam | |

### Physics Validation Rules
| Reynolds Number | Regime | Recommended Solver | Turbulence |
|----------------|--------|-------------------|------------|
| Re < 2,300 | Laminar | icoFoam/pisoFoam | none |
| 2,300 ≤ Re < 10,000 | Transitional | pimpleFoam | kOmegaSST |
| Re ≥ 10,000 | Turbulent RANS | simpleFoam/pimpleFoam | kOmegaSST |

## 🚀 Quick Start

### 1. Installation

\`\`\`bash
# Clone the repository
cd /home/aryannzzz/openFOAM/llm-foam-orchestrator

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your LLM API keys
\`\`\`

### 2. Run the Server

\`\`\`bash
# Development mode
python main.py

# Production mode with Docker
docker-compose up -d
\`\`\`

### 3. Submit a Simulation

\`\`\`bash
# Using CLI
python cli.py run "Simulate flow around a cylinder at Re=100"

# Using API
curl -X POST http://localhost:8000/api/v1/simulations \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Simulate laminar flow around a 2D cylinder at Reynolds number 100"}'
\`\`\`

## 📡 API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/v1/simulations | Submit a new simulation |
| GET | /api/v1/simulations/{job_id} | Get simulation status |
| GET | /api/v1/simulations/{job_id}/result | Get simulation results |
| DELETE | /api/v1/simulations/{job_id} | Cancel a simulation |
| POST | /api/v1/convert | Convert NL to CFDSpecification |
| POST | /api/v1/validate | Validate a CFDSpecification |
| GET | /api/v1/capabilities | List system capabilities |
| GET | /health | Health check |

### Example: Submit Simulation

\`\`\`bash
curl -X POST "http://localhost:8000/api/v1/simulations" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Simulate turbulent flow over a backward-facing step at Re=5000",
    "priority": "normal"
  }'
\`\`\`

**Response:**
\`\`\`json
{
  "job_id": "sim_20240115_143022_a1b2c3d4",
  "status": "queued",
  "created_at": "2024-01-15T14:30:22Z",
  "message": "Simulation queued for processing"
}
\`\`\`

### Example: Get Results

\`\`\`bash
curl "http://localhost:8000/api/v1/simulations/sim_20240115_143022_a1b2c3d4/result"
\`\`\`

**Response:**
\`\`\`json
{
  "job_id": "sim_20240115_143022_a1b2c3d4",
  "status": "completed",
  "results": {
    "drag_coefficient": 1.24,
    "lift_coefficient": 0.02,
    "pressure_drop": 45.2,
    "strouhal_number": 0.21
  },
  "case_path": "/data/cases/sim_20240115_143022_a1b2c3d4",
  "result_path": "/data/results/sim_20240115_143022_a1b2c3d4"
}
\`\`\`

## ��️ CLI Reference

\`\`\`bash
# Submit simulation
python cli.py run "Your simulation description" [--wait] [--timeout 3600]

# Check status
python cli.py status <job_id>

# Get results
python cli.py results <job_id> [--format json|table]

# List all simulations
python cli.py list [--status pending|running|completed|failed]

# Validate a specification
python cli.py validate spec.json

# Show system capabilities
python cli.py capabilities

# Interactive mode
python cli.py interactive

# Convert NL to JSON only (no execution)
python cli.py convert "Your description" [--output spec.json]
\`\`\`

## 📦 CFDSpecification Schema

The core intermediate representation (IR) for all simulations:

\`\`\`json
{
  "metadata": {
    "name": "cylinder_re100",
    "description": "Flow around cylinder at Re=100",
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
    "type": "pisoFoam",
    "algorithm": "PISO",
    "max_iterations": 5000
  },
  "time": {
    "end_time": 2.0,
    "delta_t": 0.0001,
    "write_interval": 0.1
  },
  "mesh": {
    "resolution": "medium",
    "boundary_layer": {"enabled": true, "num_layers": 10}
  },
  "boundaries": {
    "inlet": {"type": "velocity_inlet", "velocity": [0.15, 0, 0]},
    "outlet": {"type": "pressure_outlet", "pressure": 0},
    "walls": {"type": "no_slip"}
  },
  "outputs": {
    "fields": ["p", "U"],
    "derived_quantities": ["drag_coefficient", "lift_coefficient"]
  }
}
\`\`\`

## 🐳 Docker Deployment

### docker-compose.yml Services

| Service | Description | Ports |
|---------|-------------|-------|
| api | FastAPI application | 8000 |
| worker | Celery workers (x2) | - |
| redis | Job queue & result backend | 6379 |

### Commands

\`\`\`bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api
docker-compose logs -f worker

# Scale workers
docker-compose up -d --scale worker=4

# Stop services
docker-compose down
\`\`\`

## �� Security Model

### Injection Prevention
- 40+ dangerous pattern detection (shell metacharacters, path traversal, code injection)
- Regex-based validation of all string inputs
- No shell execution of user-provided strings

### Resource Limits
| Resource | Default Limit |
|----------|---------------|
| Max Cells | 10,000,000 |
| Max CPUs | 8 |
| Max Memory | 16 GB |
| Max Runtime | 4 hours |
| Max Disk | 100 GB |

### Sandboxed Execution
- Non-root container user (cfduser:1000)
- Read-only root filesystem
- Isolated network per simulation

## 📊 Error Codes

| Code | Description | Recovery |
|------|-------------|----------|
| E001 | Invalid prompt | Rephrase with specific parameters |
| E002 | Schema validation error | Check CFDSpecification format |
| E003 | Physics validation failed | Adjust Re/solver combination |
| E004 | Security validation failed | Remove dangerous patterns |
| E005 | Case generation error | Check geometry/template availability |
| E006 | Meshing error | Reduce mesh resolution |
| E007 | Solver divergence | Adjust relaxation/time step |
| E008 | Resource limit exceeded | Reduce mesh size or runtime |
| E009 | Timeout | Increase timeout or simplify case |
| E010 | Post-processing error | Check output field availability |

## 📁 Project Structure

\`\`\`
llm-foam-orchestrator/
├── main.py                 # FastAPI application entry point
├── cli.py                  # Command-line interface
├── requirements.txt        # Python dependencies
├── Dockerfile              # Multi-stage Docker build
├── docker-compose.yml      # Docker Compose configuration
├── .env.example            # Environment configuration template
├── app/
│   ├── __init__.py
│   ├── schemas.py          # CFDSpecification Pydantic models
│   ├── validation.py       # Physics validation rules
│   ├── security.py         # Security validation & resource limits
│   ├── errors.py           # Error codes and exceptions
│   ├── llm_converter.py    # LLM integration (NL → JSON)
│   ├── case_generator.py   # OpenFOAM case generation
│   ├── postprocessing.py   # Result extraction
│   ├── celery_worker.py    # Background task worker
│   ├── api/
│   │   ├── __init__.py
│   │   ├── router.py       # Legacy API routes
│   │   └── endpoints.py    # v1 API endpoints
│   └── templates/          # OpenFOAM case templates
└── tests/
    ├── test_schemas.py
    ├── test_validation.py
    └── test_api.py
\`\`\`

## 🧪 Testing

\`\`\`bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_validation.py -v
\`\`\`

## 📝 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

---

**Built with ❤️ for the CFD community**
