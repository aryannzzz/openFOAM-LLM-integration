"""
LLM-Driven OpenFOAM Orchestration System
Main application entry point

Based on design document: LLM-Driven OpenFOAM Orchestration System
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv

from app.api import router as api_router
from app.api.endpoints import router as endpoints_router
from app.logger import setup_logger
from app.errors import CFDException

# Load environment variables
load_dotenv()

# Setup logging
setup_logger()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    logger.info("🚀 Starting LLM-Driven OpenFOAM Orchestrator")
    logger.info(f"OpenFOAM Path: {os.getenv('OPENFOAM_PATH', '/opt/openfoam')}")
    logger.info(f"LLM Provider: {os.getenv('LLM_PROVIDER', 'openai')}")
    logger.info(f"Cases Directory: {os.getenv('CASES_DIR', '/data/cases')}")
    logger.info(f"Results Directory: {os.getenv('RESULTS_DIR', '/data/results')}")
    yield
    logger.info("🛑 Shutting down LLM-Driven OpenFOAM Orchestrator")


# Initialize FastAPI app
app = FastAPI(
    title="LLM-Driven OpenFOAM Orchestrator",
    description="""
    Natural Language Interface to OpenFOAM CFD Simulations.
    
    ## Features
    - Convert natural language to CFD specifications
    - Physics validation with solver/turbulence recommendations
    - Security validation against injection attacks
    - Automated OpenFOAM case generation
    - Post-processing and result extraction
    
    ## Workflow
    1. Submit natural language prompt to /api/v1/simulations
    2. LLM converts to CFDSpecification JSON
    3. Physics and security validation
    4. OpenFOAM case generation
    5. Meshing (blockMesh/snappyHexMesh)
    6. Solver execution
    7. Post-processing and results
    """,
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# Global exception handler for CFD errors
@app.exception_handler(CFDException)
async def cfd_exception_handler(request, exc: CFDException):
    """Handle CFD-specific exceptions"""
    return JSONResponse(
        status_code=exc.http_status,
        content={
            "error_code": exc.error.code.value,
            "message": exc.error.message,
            "details": exc.error.details,
            "recovery": exc.error.recovery_hint
        }
    )


# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
# Legacy router (v1 compatibility)
app.include_router(api_router.router, prefix="/api", tags=["legacy"])

# New endpoints (v1)
app.include_router(endpoints_router, prefix="/api/v1", tags=["simulations"])


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration"""
    return {
        "status": "healthy",
        "service": "LLM-OpenFOAM Orchestrator",
        "version": "2.0.0",
        "openfoam_available": os.path.exists(os.getenv("OPENFOAM_PATH", "/opt/openfoam"))
    }


@app.get("/")
async def root():
    """Root endpoint with API documentation links"""
    return {
        "message": "LLM-Driven OpenFOAM Orchestration System",
        "version": "2.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "simulations": "/api/v1/simulations",
            "convert": "/api/v1/convert",
            "validate": "/api/v1/validate",
            "capabilities": "/api/v1/capabilities"
        }
    }


@app.get("/api/v1/system-info")
async def system_info():
    """Get system configuration and capabilities"""
    return {
        "llm_provider": os.getenv("LLM_PROVIDER", "openai"),
        "openfoam_version": "v2312",
        "supported_solvers": [
            "icoFoam", "simpleFoam", "pimpleFoam", 
            "pisoFoam", "rhoSimpleFoam", "rhoPimpleFoam"
        ],
        "supported_turbulence_models": [
            "kEpsilon", "kOmegaSST", "SpalartAllmaras", "laminar"
        ],
        "max_cells": 10_000_000,
        "max_processors": 8,
        "max_memory_gb": 16
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=os.getenv("API_DEBUG", "true").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )
