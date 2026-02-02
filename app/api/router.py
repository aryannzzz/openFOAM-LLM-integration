"""
API Router for simulation endpoints
"""
import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional
import uuid
from datetime import datetime

from app.models import (
    SimulationRequest, SimulationResult, LLMConversionResponse,
    SimulationStatus, ErrorResponse
)
from app.llm_converter import get_llm_converter
from app.openfoam_wrapper import create_simulator

logger = logging.getLogger(__name__)

# In-memory storage for simulation states (in production, use database)
simulation_states = {}

router = APIRouter()


@router.post("/simulate", response_model=SimulationResult, tags=["simulation"])
async def submit_simulation(
    request: SimulationRequest,
    background_tasks: BackgroundTasks
) -> SimulationResult:
    """
    Submit a new simulation with natural language description
    
    Process:
    1. LLM converts natural language to JSON configuration
    2. OpenFOAM simulation is set up and executed
    3. Results are returned asynchronously
    """
    try:
        logger.info(f"Received simulation request: {request.case_name}")

        # Step 1: Convert natural language to JSON
        llm_converter = get_llm_converter()
        conversion_result = llm_converter.convert_to_json(request.description)

        logger.info(
            f"LLM conversion confidence: {conversion_result.confidence_score:.2%}"
        )

        # Use provided simulation type if available, otherwise use LLM result
        if request.simulation_type:
            conversion_result.json_configuration.simulation_type = request.simulation_type

        # Generate case ID
        case_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"

        # Step 2: Set up and run simulation in background
        background_tasks.add_task(
            _run_simulation,
            case_id,
            request.case_name,
            conversion_result.json_configuration,
            request.max_runtime
        )

        # Return immediate response
        result = SimulationResult(
            case_id=case_id,
            case_name=request.case_name,
            status="pending",
            simulation_type=conversion_result.json_configuration.simulation_type,
            configuration=conversion_result.json_configuration,
            output_path=""
        )

        # Store state
        simulation_states[case_id] = {
            "status": "pending",
            "result": result,
            "start_time": datetime.now()
        }

        logger.info(f"Simulation {case_id} queued for execution")

        return result

    except Exception as e:
        logger.error(f"Error in simulation submission: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{case_id}", response_model=SimulationStatus, tags=["simulation"])
async def get_simulation_status(case_id: str) -> SimulationStatus:
    """Get current status of a simulation"""
    if case_id not in simulation_states:
        raise HTTPException(status_code=404, detail=f"Simulation {case_id} not found")

    state = simulation_states[case_id]
    result = state["result"]

    return SimulationStatus(
        case_id=case_id,
        case_name=result.case_name,
        status=result.status or "pending",
        current_step=f"Simulating {result.simulation_type}",
        time_elapsed=(
            (datetime.now() - state["start_time"]).total_seconds()
            if "start_time" in state else None
        )
    )


@router.get("/results/{case_id}", response_model=SimulationResult, tags=["simulation"])
async def get_simulation_results(case_id: str) -> SimulationResult:
    """Get results of a completed simulation"""
    if case_id not in simulation_states:
        raise HTTPException(status_code=404, detail=f"Simulation {case_id} not found")

    state = simulation_states[case_id]
    result = state["result"]

    if result.status != "completed":
        raise HTTPException(
            status_code=202,
            detail=f"Simulation is {result.status or 'pending'}"
        )

    return result


@router.post("/convert", response_model=LLMConversionResponse, tags=["conversion"])
async def convert_description(description: str) -> LLMConversionResponse:
    """
    Convert natural language description to OpenFOAM JSON configuration
    (without running simulation)
    """
    try:
        logger.info(f"Converting description: {description[:50]}...")

        llm_converter = get_llm_converter()
        result = llm_converter.convert_to_json(description)

        logger.info(f"Conversion confidence: {result.confidence_score:.2%}")

        return result

    except Exception as e:
        logger.error(f"Error in conversion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/simulations", tags=["simulation"])
async def list_simulations():
    """List all simulations and their statuses"""
    simulations = []
    for case_id, state in simulation_states.items():
        result = state["result"]
        simulations.append({
            "case_id": case_id,
            "case_name": result.case_name,
            "status": result.status or "pending",
            "simulation_type": result.simulation_type,
            "created_at": state.get("start_time")
        })

    return {
        "total": len(simulations),
        "simulations": simulations
    }


@router.post("/visualize/{case_id}", tags=["visualization"])
async def generate_visualization(case_id: str):
    """Generate visualization for simulation results"""
    if case_id not in simulation_states:
        raise HTTPException(status_code=404, detail=f"Simulation {case_id} not found")

    state = simulation_states[case_id]
    result = state["result"]

    if result.status != "completed":
        raise HTTPException(
            status_code=400,
            detail="Simulation must be completed before visualization"
        )

    # Placeholder for visualization generation
    # In production, integrate with ParaView or similar
    return {
        "case_id": case_id,
        "visualization_url": f"/visualizations/{case_id}/index.html",
        "message": "Visualization would be generated here"
    }


@router.delete("/simulations/{case_id}", tags=["simulation"])
async def delete_simulation(case_id: str):
    """Delete a simulation and its data"""
    if case_id not in simulation_states:
        raise HTTPException(status_code=404, detail=f"Simulation {case_id} not found")

    try:
        state = simulation_states[case_id]
        # Cleanup would happen here
        del simulation_states[case_id]

        logger.info(f"Deleted simulation {case_id}")

        return {"message": f"Simulation {case_id} deleted successfully"}

    except Exception as e:
        logger.error(f"Error deleting simulation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background task
async def _run_simulation(case_id: str, case_name: str, config, timeout: Optional[int]):
    """Background task to run simulation"""
    try:
        logger.info(f"Executing simulation: {case_id}")

        # Create simulator
        simulator = create_simulator(case_id, case_name)

        # Setup
        if not simulator.setup_simulation(config):
            simulation_states[case_id]["result"].status = "failed"
            return

        # Run
        success, runtime_or_error = simulator.run_simulation(timeout=timeout)

        if success:
            # Get results
            result = simulator.get_results(config, float(runtime_or_error))
            result.status = "completed"
            simulation_states[case_id]["result"] = result
            logger.info(f"Simulation {case_id} completed successfully")
        else:
            simulation_states[case_id]["result"].status = "failed"
            logger.error(f"Simulation {case_id} failed: {runtime_or_error}")

    except Exception as e:
        logger.error(f"Background task error for {case_id}: {e}")
        if case_id in simulation_states:
            simulation_states[case_id]["result"].status = "failed"
