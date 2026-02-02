"""
Utilities module
"""
import logging
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


def validate_simulation_config(config: Dict[str, Any]) -> bool:
    """Validate simulation configuration"""
    required_fields = [
        'simulation_type', 'mesh_type', 'solver',
        'domain', 'boundary_conditions', 'material_properties'
    ]

    for field in required_fields:
        if field not in config:
            logger.error(f"Missing required field: {field}")
            return False

    return True


def parse_openfoam_log(log_content: str) -> Dict[str, Any]:
    """Parse OpenFOAM log output to extract key metrics"""
    metrics = {
        "convergence": False,
        "iterations": 0,
        "residuals": []
    }

    lines = log_content.split('\n')
    for line in lines:
        if 'Iteration' in line or 'Time' in line:
            # Parse iteration count
            try:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.isdigit():
                        metrics["iterations"] = int(part)
                        break
            except Exception:
                pass

        if 'SIMPLE: Converged' in line:
            metrics["convergence"] = True

    return metrics


def format_simulation_summary(case_id: str, runtime: float, config: Dict[str, Any]) -> str:
    """Format simulation summary"""
    summary = f"""
    =======================================
    SIMULATION SUMMARY
    =======================================
    Case ID: {case_id}
    Simulation Type: {config.get('simulation_type', 'Unknown')}
    Solver: {config.get('solver', 'Unknown')}
    Runtime: {runtime:.2f} seconds
    =======================================
    """
    return summary


def ensure_directory(path: Path):
    """Ensure directory exists"""
    path.mkdir(parents=True, exist_ok=True)
