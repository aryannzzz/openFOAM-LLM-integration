# Project Structure

```
llm-foam-orchestrator/
│
├── app/                          # Main application package
│   ├── __init__.py
│   ├── schemas.py               # CFDSpecification Pydantic models
│   ├── llm_converter.py         # Natural language → CFD specification
│   ├── validation.py            # Physics validation rules
│   ├── security.py              # Security checks (40+ patterns)
│   ├── case_generator.py        # OpenFOAM dictionary generation
│   ├── openfoam_wrapper.py      # OpenFOAM execution wrapper
│   ├── postprocessing.py        # Result extraction
│   ├── errors.py                # Standardized error codes
│   ├── config.py                # Configuration management
│   ├── logger.py                # Logging setup
│   ├── models.py                # Database models (if using)
│   ├── utils.py                 # Utility functions
│   └── api/
│       ├── __init__.py
│       └── endpoints.py         # FastAPI REST endpoints
│
├── tests/                        # Test suite
│   ├── test_comprehensive.py    # 37 unit tests (all modules)
│   ├── test_api.py              # API endpoint tests
│   ├── test_converter.py        # LLM converter tests
│   ├── test_system.py           # System integration tests
│   ├── run_real_simulation.py   # Single E2E test with OpenFOAM
│   └── run_multiple_tests.py    # Batch E2E tests
│
├── docs/                         # Documentation
│   ├── USER_GUIDE.md            # Complete usage guide
│   └── PIPELINE_DOCUMENTATION.md # Technical architecture
│
├── scripts/                      # Utility scripts
│   ├── cli.py                   # Command-line interface
│   ├── demo.py                  # Demo script
│   └── quickstart.sh            # Quick setup script
│
├── docker/                       # Docker configuration
│   ├── Dockerfile               # Multi-stage Docker build
│   └── docker-compose.yml       # Docker Compose for services
│
├── templates/                    # OpenFOAM case templates
│   └── ...
│
├── .env.example                  # Environment variable template
├── .gitignore                    # Git ignore rules
├── main.py                       # FastAPI application entry point
├── pyproject.toml               # Python project configuration
├── pytest.ini                    # Pytest configuration
├── requirements.txt             # Python dependencies
├── README.md                     # Project README
└── TEST_RESULTS.md              # Test results documentation
```

## Key Files

| File | Purpose |
|------|---------|
| `app/schemas.py` | CFDSpecification JSON schema with Pydantic models |
| `app/llm_converter.py` | Converts natural language to CFDSpecification |
| `app/validation.py` | Physics constraint matrices and validation |
| `app/security.py` | Security patterns and resource limits |
| `app/case_generator.py` | Generates OpenFOAM dictionaries |
| `tests/test_comprehensive.py` | 37 unit tests covering all modules |
| `tests/run_real_simulation.py` | Full E2E test with real OpenFOAM |
| `docs/USER_GUIDE.md` | Complete user documentation |

## Running the System

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Edit with your API key

# Run tests
python -m pytest tests/ -v

# Run E2E simulation
python tests/run_real_simulation.py

# Start API server
python main.py
```
