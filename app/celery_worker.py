"""
Celery Worker Configuration
Background task processing for OpenFOAM simulations
"""
import os
import logging
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery(
    "openfoam_worker",
    broker=os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0"),
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=int(os.getenv("MAX_RUNTIME", 14400)),  # 4 hours default
    worker_prefetch_multiplier=1,  # One task at a time for heavy simulations
    result_expires=86400,  # Results expire after 24 hours
)


@celery_app.task(bind=True, name="run_simulation")
def run_simulation_task(self, job_id: str, spec_dict: dict):
    """
    Execute a complete CFD simulation pipeline.
    
    Stages:
    1. Parsing - Convert spec to internal format
    2. Validating - Physics and security checks
    3. Generating - Create OpenFOAM case
    4. Meshing - Generate mesh (blockMesh/snappyHexMesh)
    5. Running - Execute solver
    6. Postprocessing - Extract results
    """
    from app.schemas import CFDSpecification
    from app.validation import PhysicsValidator
    from app.security import SecurityChecker
    from app.case_generator import CaseGenerator
    from app.postprocessing import PostProcessor
    from app.errors import ErrorCode, CFDError
    import subprocess
    import json
    from pathlib import Path
    
    cases_dir = Path(os.getenv("CASES_DIR", "/data/cases"))
    results_dir = Path(os.getenv("RESULTS_DIR", "/data/results"))
    
    try:
        # Stage 1: Parsing
        self.update_state(state="PROGRESS", meta={"stage": "parsing", "progress": 10})
        spec = CFDSpecification(**spec_dict)
        logger.info(f"[{job_id}] Parsed specification: {spec.metadata.name}")
        
        # Stage 2: Validation
        self.update_state(state="PROGRESS", meta={"stage": "validating", "progress": 20})
        
        # Physics validation
        physics_validator = PhysicsValidator()
        physics_result = physics_validator.validate(spec)
        if not physics_result["valid"]:
            return {
                "status": "failed",
                "stage": "validating",
                "error": CFDError(
                    code=ErrorCode.PHYSICS_ERROR,
                    message="Physics validation failed",
                    details=physics_result
                ).to_dict()
            }
        
        # Security validation
        security_checker = SecurityChecker()
        security_result = security_checker.validate_specification(spec)
        if not security_result["valid"]:
            return {
                "status": "failed",
                "stage": "validating",
                "error": CFDError(
                    code=ErrorCode.SECURITY_ERROR,
                    message="Security validation failed",
                    details=security_result
                ).to_dict()
            }
        
        logger.info(f"[{job_id}] Validation passed")
        
        # Stage 3: Case generation
        self.update_state(state="PROGRESS", meta={"stage": "generating", "progress": 40})
        case_dir = cases_dir / job_id
        
        generator = CaseGenerator(templates_dir=os.getenv("TEMPLATES_DIR", "/app/templates"))
        generator.generate_case(spec, str(case_dir))
        logger.info(f"[{job_id}] Case generated at {case_dir}")
        
        # Stage 4: Meshing
        self.update_state(state="PROGRESS", meta={"stage": "meshing", "progress": 50})
        
        # Source OpenFOAM environment and run blockMesh
        openfoam_path = os.getenv("OPENFOAM_PATH", "/opt/openfoam")
        source_cmd = f"source {openfoam_path}/etc/bashrc"
        
        mesh_result = subprocess.run(
            f"{source_cmd} && cd {case_dir} && blockMesh",
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout for meshing
        )
        
        if mesh_result.returncode != 0:
            return {
                "status": "failed",
                "stage": "meshing",
                "error": CFDError(
                    code=ErrorCode.MESH_ERROR,
                    message="Meshing failed",
                    details={"stderr": mesh_result.stderr[-2000:] if mesh_result.stderr else "No output"}
                ).to_dict()
            }
        
        logger.info(f"[{job_id}] Mesh generated successfully")
        
        # Stage 5: Running solver
        self.update_state(state="PROGRESS", meta={"stage": "running", "progress": 60})
        
        solver = spec.solver.type.value
        max_runtime = int(os.getenv("MAX_RUNTIME", 14400))
        
        solver_result = subprocess.run(
            f"{source_cmd} && cd {case_dir} && {solver}",
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=max_runtime
        )
        
        # Check for solver errors (non-zero return code doesn't always mean failure)
        if solver_result.returncode != 0 and "FOAM FATAL ERROR" in solver_result.stderr:
            return {
                "status": "failed",
                "stage": "running",
                "error": CFDError(
                    code=ErrorCode.SOLVER_ERROR,
                    message="Solver failed",
                    details={"stderr": solver_result.stderr[-2000:] if solver_result.stderr else "No output"}
                ).to_dict()
            }
        
        logger.info(f"[{job_id}] Solver completed")
        
        # Stage 6: Post-processing
        self.update_state(state="PROGRESS", meta={"stage": "postprocessing", "progress": 90})
        
        result_dir = results_dir / job_id
        result_dir.mkdir(parents=True, exist_ok=True)
        
        postprocessor = PostProcessor(str(case_dir))
        results = postprocessor.extract_results(spec.outputs)
        
        # Save results
        with open(result_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"[{job_id}] Post-processing complete, results at {result_dir}")
        
        return {
            "status": "completed",
            "stage": "completed",
            "case_path": str(case_dir),
            "result_path": str(result_dir),
            "results": results
        }
        
    except subprocess.TimeoutExpired:
        return {
            "status": "failed",
            "stage": "running",
            "error": CFDError(
                code=ErrorCode.TIMEOUT_ERROR,
                message="Simulation exceeded maximum runtime"
            ).to_dict()
        }
    except Exception as e:
        logger.error(f"[{job_id}] Simulation failed: {str(e)}")
        return {
            "status": "failed",
            "stage": "unknown",
            "error": CFDError(
                code=ErrorCode.RUNTIME_ERROR,
                message=str(e)
            ).to_dict()
        }


@celery_app.task(name="cleanup_old_jobs")
def cleanup_old_jobs(max_age_hours: int = 24):
    """Clean up old simulation directories"""
    import shutil
    from datetime import datetime, timedelta
    from pathlib import Path
    
    cases_dir = Path(os.getenv("CASES_DIR", "/data/cases"))
    results_dir = Path(os.getenv("RESULTS_DIR", "/data/results"))
    cutoff = datetime.now() - timedelta(hours=max_age_hours)
    
    cleaned = 0
    for directory in [cases_dir, results_dir]:
        if not directory.exists():
            continue
        for job_dir in directory.iterdir():
            if job_dir.is_dir():
                mtime = datetime.fromtimestamp(job_dir.stat().st_mtime)
                if mtime < cutoff:
                    shutil.rmtree(job_dir)
                    cleaned += 1
    
    logger.info(f"Cleaned up {cleaned} old job directories")
    return {"cleaned": cleaned}
