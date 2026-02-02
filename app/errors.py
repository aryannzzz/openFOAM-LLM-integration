"""
Error Codes Module
Standardized error codes based on design document Appendix A.
"""
from enum import Enum
from typing import Optional
from dataclasses import dataclass


class ErrorCode(str, Enum):
    """
    Standardized error codes for the system.
    Based on design document Appendix A.
    """
    
    # Input/Parsing Errors (E001)
    INVALID_PROMPT = "E001"
    
    # Validation Errors (E002-E004)
    SCHEMA_VALIDATION_FAILED = "E002"
    PHYSICS_VALIDATION_FAILED = "E003"
    SECURITY_VALIDATION_FAILED = "E004"
    
    # Case Generation Errors (E005-E006)
    TEMPLATE_NOT_FOUND = "E005"
    CASE_GENERATION_FAILED = "E006"
    
    # Execution Errors (E007-E011)
    MESH_GENERATION_FAILED = "E007"
    SOLVER_CRASHED = "E008"
    SOLVER_DIVERGED = "E009"
    TIMEOUT = "E010"
    RESOURCE_LIMIT = "E011"
    
    # Post-Processing Errors (E012)
    POSTPROCESSING_FAILED = "E012"
    
    # System Errors (E013)
    INTERNAL_ERROR = "E013"


# Error descriptions with user-friendly messages
ERROR_DESCRIPTIONS = {
    ErrorCode.INVALID_PROMPT: {
        "name": "INVALID_PROMPT",
        "description": "Natural language prompt could not be parsed",
        "user_message": "Unable to understand your request. Please rephrase your simulation description.",
        "http_status": 400,
    },
    ErrorCode.SCHEMA_VALIDATION_FAILED: {
        "name": "SCHEMA_VALIDATION_FAILED",
        "description": "Specification does not conform to schema",
        "user_message": "The simulation specification contains invalid fields or values.",
        "http_status": 400,
    },
    ErrorCode.PHYSICS_VALIDATION_FAILED: {
        "name": "PHYSICS_VALIDATION_FAILED",
        "description": "Specification violates physics constraints",
        "user_message": "The simulation parameters are physically inconsistent. Please check solver/regime compatibility.",
        "http_status": 400,
    },
    ErrorCode.SECURITY_VALIDATION_FAILED: {
        "name": "SECURITY_VALIDATION_FAILED",
        "description": "Potentially dangerous content detected",
        "user_message": "Your request contains disallowed content and cannot be processed.",
        "http_status": 400,
    },
    ErrorCode.TEMPLATE_NOT_FOUND: {
        "name": "TEMPLATE_NOT_FOUND",
        "description": "Requested geometry type has no template",
        "user_message": "The requested geometry type is not currently supported.",
        "http_status": 400,
    },
    ErrorCode.CASE_GENERATION_FAILED: {
        "name": "CASE_GENERATION_FAILED",
        "description": "Failed to generate OpenFOAM case",
        "user_message": "Unable to generate the simulation case files. Please try again.",
        "http_status": 500,
    },
    ErrorCode.MESH_GENERATION_FAILED: {
        "name": "MESH_GENERATION_FAILED",
        "description": "blockMesh or snappyHexMesh failed",
        "user_message": "Mesh generation failed. The geometry or mesh settings may need adjustment.",
        "http_status": 500,
    },
    ErrorCode.SOLVER_CRASHED: {
        "name": "SOLVER_CRASHED",
        "description": "Solver exited with non-zero code",
        "user_message": "The simulation solver encountered an error and stopped unexpectedly.",
        "http_status": 500,
    },
    ErrorCode.SOLVER_DIVERGED: {
        "name": "SOLVER_DIVERGED",
        "description": "Simulation diverged",
        "user_message": "The simulation became unstable. Consider adjusting time step or relaxation factors.",
        "http_status": 500,
    },
    ErrorCode.TIMEOUT: {
        "name": "TIMEOUT",
        "description": "Simulation exceeded time limit",
        "user_message": "The simulation took too long and was stopped. Consider using a coarser mesh or shorter simulation time.",
        "http_status": 408,
    },
    ErrorCode.RESOURCE_LIMIT: {
        "name": "RESOURCE_LIMIT",
        "description": "Exceeded memory or CPU limits",
        "user_message": "The simulation exceeded resource limits. Please reduce mesh resolution or problem size.",
        "http_status": 413,
    },
    ErrorCode.POSTPROCESSING_FAILED: {
        "name": "POSTPROCESSING_FAILED",
        "description": "Failed to extract results",
        "user_message": "Unable to process simulation results. The simulation may not have completed properly.",
        "http_status": 500,
    },
    ErrorCode.INTERNAL_ERROR: {
        "name": "INTERNAL_ERROR",
        "description": "Unexpected system error",
        "user_message": "An unexpected error occurred. Please try again or contact support.",
        "http_status": 500,
    },
}


@dataclass
class CFDError:
    """
    Structured error representation.
    """
    code: ErrorCode
    message: str
    details: Optional[str] = None
    context: Optional[dict] = None
    
    @property
    def name(self) -> str:
        """Get error name"""
        return ERROR_DESCRIPTIONS[self.code]["name"]
    
    @property
    def description(self) -> str:
        """Get error description"""
        return ERROR_DESCRIPTIONS[self.code]["description"]
    
    @property
    def user_message(self) -> str:
        """Get user-friendly message"""
        return ERROR_DESCRIPTIONS[self.code]["user_message"]
    
    @property
    def http_status(self) -> int:
        """Get HTTP status code"""
        return ERROR_DESCRIPTIONS[self.code]["http_status"]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API response"""
        return {
            "error_code": self.code.value,
            "error_name": self.name,
            "message": self.message,
            "description": self.description,
            "user_message": self.user_message,
            "details": self.details,
            "context": self.context,
        }
    
    def __str__(self) -> str:
        return f"[{self.code.value}] {self.name}: {self.message}"


class CFDException(Exception):
    """
    Base exception class for CFD operations.
    """
    
    def __init__(self, code: ErrorCode, message: str, 
                 details: Optional[str] = None, context: Optional[dict] = None):
        self.error = CFDError(code, message, details, context)
        super().__init__(str(self.error))
    
    @property
    def code(self) -> ErrorCode:
        return self.error.code
    
    @property
    def http_status(self) -> int:
        return self.error.http_status
    
    def to_dict(self) -> dict:
        return self.error.to_dict()


# Specific exception classes for each error type
class InvalidPromptError(CFDException):
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(ErrorCode.INVALID_PROMPT, message, details)


class SchemaValidationError(CFDException):
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(ErrorCode.SCHEMA_VALIDATION_FAILED, message, details)


class PhysicsValidationError(CFDException):
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(ErrorCode.PHYSICS_VALIDATION_FAILED, message, details)


class SecurityValidationError(CFDException):
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(ErrorCode.SECURITY_VALIDATION_FAILED, message, details)


class TemplateNotFoundError(CFDException):
    def __init__(self, template_name: str):
        super().__init__(
            ErrorCode.TEMPLATE_NOT_FOUND, 
            f"Template not found: {template_name}",
            context={"template": template_name}
        )


class CaseGenerationError(CFDException):
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(ErrorCode.CASE_GENERATION_FAILED, message, details)


class MeshGenerationError(CFDException):
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(ErrorCode.MESH_GENERATION_FAILED, message, details)


class SolverCrashedError(CFDException):
    def __init__(self, solver: str, exit_code: int, details: Optional[str] = None):
        super().__init__(
            ErrorCode.SOLVER_CRASHED,
            f"Solver {solver} crashed with exit code {exit_code}",
            details,
            context={"solver": solver, "exit_code": exit_code}
        )


class SolverDivergedError(CFDException):
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(ErrorCode.SOLVER_DIVERGED, message, details)


class TimeoutError(CFDException):
    def __init__(self, timeout_seconds: float):
        super().__init__(
            ErrorCode.TIMEOUT,
            f"Simulation exceeded timeout of {timeout_seconds} seconds",
            context={"timeout_seconds": timeout_seconds}
        )


class ResourceLimitError(CFDException):
    def __init__(self, resource: str, limit: float, requested: float):
        super().__init__(
            ErrorCode.RESOURCE_LIMIT,
            f"Resource limit exceeded: {resource} (requested: {requested}, limit: {limit})",
            context={"resource": resource, "limit": limit, "requested": requested}
        )


class PostProcessingError(CFDException):
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(ErrorCode.POSTPROCESSING_FAILED, message, details)


class InternalError(CFDException):
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(ErrorCode.INTERNAL_ERROR, message, details)
