"""
LLM-Driven OpenFOAM Orchestrator
App module initialization

Based on design document: LLM-Driven OpenFOAM Orchestration System
"""

from app.schemas import (
    CFDSpecification,
    LLMConversionResponse,
    FlowRegime,
    TimeDependence,
    TurbulenceModel,
    SolverType,
    GeometryType,
    MeshResolution,
)
from app.validation import PhysicsValidator, get_recommended_solver
from app.security import SecurityChecker
from app.case_generator import CaseGenerator
from app.postprocessing import PostProcessor
from app.errors import CFDException, ErrorCode

__all__ = [
    "CFDSpecification",
    "LLMConversionResponse",
    "FlowRegime",
    "TimeDependence",
    "TurbulenceModel",
    "SolverType",
    "GeometryType",
    "MeshResolution",
    "PhysicsValidator",
    "SecurityChecker",
    "CaseGenerator",
    "PostProcessor",
    "CFDException",
    "ErrorCode",
    "get_recommended_solver",
]
