"""
REST API Endpoints
Based on design document Section 8.1
"""
import uuid
import time
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel

from app.schemas import (
    CFDSpecification, SimulationRequest, SimulationStatusResponse,
    SimulationResult, LLMConversionResponse, CapabilitiesResponse,
    SimulationStatus, GeometryType, SolverType, TurbulenceModel, FlowRegime
)
from app.validation import validate_physics_constraints, PhysicsValidator
from app.security import SecurityChecker, validate_security
from app.errors import (
    CFDException, InvalidPromptError, SchemaValidationError,
    PhysicsValidationError, SecurityValidationError
)
from app.llm_converter import LLMConverter
from app.case_generator import CaseGenerator
from app.postprocessing import PostProcessor
from app.config import get_config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["simulations"])

# In-memory job store (replace with Redis/DB in production)
_jobs: Dict[str, Dict[str, Any]] = {}

# Initialize components
config = get_config()


# ============================================================================
# Request/Response Models
# ============================================================================

class ValidationRequest(BaseModel):
    """Request to validate a specification"""
    specification: CFDSpecification


class ValidationResponse(BaseModel):
    """Validation result"""
    valid: bool
    errors: List[str] = []
    warnings: List[str] = []


class ConvertRequest(BaseModel):
    """Request to convert NL to specification"""
    prompt: str


class JobListResponse(BaseModel):
    """List of jobs"""
    simulations: List[Dict[str, Any]]
    total: int


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/simulations", response_model=SimulationStatusResponse)
async def create_simulation(request: SimulationRequest, background_tasks: BackgroundTasks):
    """
    Submit a new simulation request.
    
    Accepts either:
    - Natural language prompt: will be converted to CFDSpecification via LLM
    - Direct CFDSpecification JSON: will be validated and used directly
    """
    job_id = str(uuid.uuid4())[:8]
    
    # Initialize job
    _jobs[job_id] = {
        "job_id": job_id,
        "status": SimulationStatus.QUEUED,
        "progress": 0.0,
        "message": "Job queued",
        "created_at": time.time(),
        "request": request.model_dump(),
    }
    
    # Queue the simulation in background
    background_tasks.add_task(run_simulation_pipeline, job_id, request)
    
    return SimulationStatusResponse(
        job_id=job_id,
        status=SimulationStatus.QUEUED,
        progress=0.0,
        message="Job queued"
    )


@router.get("/simulations/{job_id}", response_model=SimulationStatusResponse)
async def get_simulation_status(job_id: str):
    """Get the current status of a simulation."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    job = _jobs[job_id]
    
    return SimulationStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress", 0.0),
        message=job.get("message"),
        current_step=job.get("current_step"),
        time_elapsed_seconds=time.time() - job.get("created_at", time.time())
    )


@router.get("/simulations/{job_id}/result", response_model=SimulationResult)
async def get_simulation_result(job_id: str):
    """Get the complete result of a finished simulation."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    job = _jobs[job_id]
    
    if job["status"] not in [SimulationStatus.COMPLETED, SimulationStatus.FAILED]:
        raise HTTPException(
            status_code=400,
            detail=f"Simulation not finished. Current status: {job['status']}"
        )
    
    return SimulationResult(
        job_id=job_id,
        status=job["status"],
        specification=job.get("specification"),
        results=job.get("results", {}),
        summary=job.get("summary", ""),
        artifacts=job.get("artifacts", []),
        runtime_seconds=job.get("runtime_seconds", 0.0),
        error_code=job.get("error_code"),
        error_message=job.get("error_message")
    )


@router.get("/simulations", response_model=JobListResponse)
async def list_simulations(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(10, ge=1, le=100, description="Max results")
):
    """List all simulations with optional filtering."""
    simulations = []
    
    for job_id, job in _jobs.items():
        if status and job["status"].value != status:
            continue
        
        simulations.append({
            "job_id": job_id,
            "status": job["status"].value,
            "name": job.get("name", ""),
            "created_at": job.get("created_at"),
            "progress": job.get("progress", 0.0),
        })
    
    # Sort by created_at descending
    simulations.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    
    return JobListResponse(
        simulations=simulations[:limit],
        total=len(simulations)
    )


@router.post("/convert", response_model=LLMConversionResponse)
async def convert_prompt(request: ConvertRequest):
    """
    Convert a natural language prompt to CFDSpecification.
    Does not run the simulation, only returns the converted specification.
    """
    try:
        converter = LLMConverter(
            provider=config.LLM_PROVIDER,
            model=config.LLM_MODEL
        )
        
        result = converter.convert_to_cfd_specification(request.prompt)
        return result
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/validate", response_model=ValidationResponse)
async def validate_specification(request: ValidationRequest):
    """
    Validate a CFDSpecification against physics and security constraints.
    """
    errors = []
    warnings = []
    
    # Physics validation
    physics_valid, physics_errors = validate_physics_constraints(request.specification)
    for err in physics_errors:
        if err.severity == "error":
            errors.append(f"[{err.code}] {err.message}")
        else:
            warnings.append(f"[{err.code}] {err.message}")
    
    # Security validation
    security_valid, security_details = validate_security(request.specification)
    for issue in security_details.get("security_issues", []):
        errors.append(f"[{issue['code']}] {issue['message']}")
    for violation in security_details.get("resource_violations", []):
        warnings.append(f"[RESOURCE] {violation['message']}")
    
    return ValidationResponse(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )


@router.get("/capabilities", response_model=CapabilitiesResponse)
async def get_capabilities():
    """Get system capabilities and constraints."""
    return CapabilitiesResponse(
        geometries=[g.value for g in GeometryType],
        solvers=[s.value for s in SolverType],
        turbulence_models=[t.value for t in TurbulenceModel],
        flow_regimes=[r.value for r in FlowRegime],
        limits={
            "max_cells": 10_000_000,
            "max_runtime_hours": 2.0,
            "max_cpus": 8,
            "max_memory_gb": 16,
            "max_time_steps": 100_000,
        }
    )


@router.delete("/simulations/{job_id}")
async def cancel_simulation(job_id: str):
    """Cancel a running or queued simulation."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    job = _jobs[job_id]
    
    if job["status"] in [SimulationStatus.COMPLETED, SimulationStatus.FAILED]:
        raise HTTPException(
            status_code=400,
            detail="Cannot cancel finished simulation"
        )
    
    # Mark as failed/cancelled
    job["status"] = SimulationStatus.FAILED
    job["message"] = "Cancelled by user"
    job["error_code"] = "CANCELLED"
    
    return {"message": "Simulation cancelled", "job_id": job_id}


# ============================================================================
# Background Task: Simulation Pipeline
# ============================================================================

async def run_simulation_pipeline(job_id: str, request: SimulationRequest):
    """
    Execute the complete simulation pipeline.
    
    Pipeline stages:
    1. Parse (convert NL prompt to specification if needed)
    2. Validate (schema, physics, security)
    3. Generate (create OpenFOAM case)
    4. Mesh (run blockMesh/snappyHexMesh)
    5. Run (execute solver)
    6. Post-process (extract results)
    """
    job = _jobs[job_id]
    start_time = time.time()
    
    try:
        # Stage 1: Parsing
        job["status"] = SimulationStatus.PARSING
        job["progress"] = 0.1
        job["current_step"] = "Parsing request"
        
        if request.specification:
            spec = request.specification
        else:
            # Convert NL prompt to specification
            converter = LLMConverter(
                provider=config.LLM_PROVIDER,
                model=config.LLM_MODEL
            )
            result = converter.convert_to_cfd_specification(request.prompt)
            spec = result.specification
        
        job["specification"] = spec.model_dump()
        job["name"] = spec.metadata.name
        
        # Stage 2: Validation
        job["status"] = SimulationStatus.VALIDATING
        job["progress"] = 0.2
        job["current_step"] = "Validating specification"
        
        # Physics validation
        physics_valid, physics_errors = validate_physics_constraints(spec)
        errors = [e for e in physics_errors if e.severity == "error"]
        if errors:
            raise PhysicsValidationError(
                "Physics validation failed",
                details="; ".join(str(e) for e in errors)
            )
        
        # Security validation
        security_valid, security_details = validate_security(spec)
        if not security_valid:
            raise SecurityValidationError(
                "Security validation failed",
                details=str(security_details)
            )
        
        # Stage 3: Case Generation
        job["status"] = SimulationStatus.GENERATING
        job["progress"] = 0.3
        job["current_step"] = "Generating OpenFOAM case"
        
        templates_dir = Path(config.TEMPLATES_DIR)
        work_dir = Path(config.WORKDIR)
        
        generator = CaseGenerator(templates_dir, work_dir)
        case_dir = generator.generate(spec)
        
        job["case_dir"] = str(case_dir)
        
        # Stage 4: Meshing
        job["status"] = SimulationStatus.MESHING
        job["progress"] = 0.4
        job["current_step"] = "Generating mesh"
        
        # In a real implementation, run blockMesh here
        # await run_openfoam_command(case_dir, "blockMesh")
        
        # Stage 5: Running Solver
        job["status"] = SimulationStatus.RUNNING
        job["progress"] = 0.5
        job["current_step"] = f"Running {spec.solver.type.value}"
        
        # In a real implementation, run the solver here
        # await run_openfoam_command(case_dir, spec.solver.type.value)
        
        # Simulate running (remove in production)
        import asyncio
        await asyncio.sleep(2)
        job["progress"] = 0.8
        
        # Stage 6: Post-processing
        job["status"] = SimulationStatus.POSTPROCESSING
        job["progress"] = 0.9
        job["current_step"] = "Extracting results"
        
        processor = PostProcessor(case_dir)
        results = processor.extract_results(spec)
        summary = processor.generate_summary(spec, results)
        
        # Completed
        job["status"] = SimulationStatus.COMPLETED
        job["progress"] = 1.0
        job["results"] = results
        job["summary"] = summary
        job["runtime_seconds"] = time.time() - start_time
        job["artifacts"] = [str(case_dir)]
        job["message"] = "Simulation completed successfully"
        
        logger.info(f"Job {job_id} completed in {job['runtime_seconds']:.1f}s")
        
    except CFDException as e:
        job["status"] = SimulationStatus.FAILED
        job["error_code"] = e.code.value
        job["error_message"] = str(e)
        job["message"] = e.error.user_message
        job["runtime_seconds"] = time.time() - start_time
        logger.error(f"Job {job_id} failed: {e}")
        
    except Exception as e:
        job["status"] = SimulationStatus.FAILED
        job["error_code"] = "E013"
        job["error_message"] = str(e)
        job["message"] = "An unexpected error occurred"
        job["runtime_seconds"] = time.time() - start_time
        logger.exception(f"Job {job_id} failed with unexpected error")
