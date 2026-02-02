"""
Security & Safety Module
Validates inputs for security concerns and enforces resource limits.
Based on design document Section 7.
"""
import re
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass
from app.schemas import CFDSpecification, MeshResolution


# ============================================================================
# SECURITY VALIDATOR
# ============================================================================

@dataclass
class SecurityIssue:
    """Represents a security issue found during validation"""
    code: str
    message: str
    path: str
    severity: str = "critical"  # critical, high, medium, low


class SecurityValidator:
    """
    Validates inputs for security concerns.
    
    Defense against:
    - Command injection
    - Path traversal
    - Code injection
    - Shell metacharacters
    """
    
    # Dangerous patterns that should NEVER appear in any input
    DANGEROUS_PATTERNS = [
        (r'[;&|`$]', "Shell metacharacter"),
        (r'\.\.[/\\]', "Path traversal"),
        (r'[<>]', "Shell redirection"),
        (r'\\x[0-9a-fA-F]{2}', "Hex escape sequence"),
        (r'%[0-9a-fA-F]{2}', "URL encoding"),
        (r'\$\{', "Variable expansion"),
        (r'\$\(', "Command substitution"),
        (r'exec\s*\(', "Exec call"),
        (r'eval\s*\(', "Eval call"),
        (r'import\s+', "Import statement"),
        (r'subprocess', "Subprocess module"),
        (r'os\.', "OS module call"),
        (r'/bin/', "Binary path"),
        (r'/etc/', "System config path"),
        (r'/usr/', "System path"),
        (r'/var/', "System path"),
        (r'/tmp/', "Temp path injection"),
        (r'rm\s+-', "Remove command"),
        (r'chmod\s+', "Permission change"),
        (r'chown\s+', "Ownership change"),
        (r'sudo\s+', "Privilege escalation"),
        (r'curl\s+', "Network command"),
        (r'wget\s+', "Network command"),
        (r'nc\s+', "Netcat command"),
        (r'bash\s+', "Shell command"),
        (r'sh\s+', "Shell command"),
        (r'python\s+', "Python command"),
        (r'perl\s+', "Perl command"),
        (r'ruby\s+', "Ruby command"),
        (r'__import__', "Python import"),
        (r'globals\s*\(', "Globals access"),
        (r'locals\s*\(', "Locals access"),
        (r'getattr\s*\(', "Getattr call"),
        (r'setattr\s*\(', "Setattr call"),
        (r'delattr\s*\(', "Delattr call"),
        (r'compile\s*\(', "Compile call"),
        (r'open\s*\(', "File open call"),
    ]
    
    # Maximum string lengths for various fields
    MAX_LENGTHS = {
        "name": 100,
        "description": 500,
        "default": 1000,
    }
    
    def __init__(self):
        """Initialize with compiled regex patterns"""
        self.compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), desc) 
            for pattern, desc in self.DANGEROUS_PATTERNS
        ]
    
    def validate_spec(self, spec: CFDSpecification) -> Tuple[bool, List[SecurityIssue]]:
        """
        Validate entire specification for security issues.
        
        Args:
            spec: CFDSpecification to validate
            
        Returns:
            Tuple of (is_safe, list of security issues)
        """
        issues: List[SecurityIssue] = []
        
        # Convert to dict for recursive checking
        spec_dict = spec.model_dump()
        
        # Recursively check all values
        self._check_recursive(spec_dict, "", issues)
        
        return len(issues) == 0, issues
    
    def validate_string(self, value: str, field_name: str = "input") -> Tuple[bool, List[SecurityIssue]]:
        """
        Validate a single string value.
        
        Args:
            value: String to validate
            field_name: Name of the field for error reporting
            
        Returns:
            Tuple of (is_safe, list of security issues)
        """
        issues: List[SecurityIssue] = []
        self._check_string(value, field_name, issues)
        return len(issues) == 0, issues
    
    def _check_recursive(self, value: Any, path: str, issues: List[SecurityIssue]):
        """Recursively check all values in a structure"""
        if isinstance(value, str):
            self._check_string(value, path, issues)
        elif isinstance(value, dict):
            for k, v in value.items():
                new_path = f"{path}.{k}" if path else k
                
                # Check key itself
                if isinstance(k, str):
                    self._check_string(k, f"{new_path}[key]", issues)
                
                # Check value
                self._check_recursive(v, new_path, issues)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                self._check_recursive(item, f"{path}[{i}]", issues)
    
    def _check_string(self, value: str, path: str, issues: List[SecurityIssue]):
        """Check a string value for dangerous patterns"""
        # Check length
        max_len = self.MAX_LENGTHS.get(path.split('.')[-1], self.MAX_LENGTHS["default"])
        if len(value) > max_len:
            issues.append(SecurityIssue(
                code="E004_SECURITY_01",
                message=f"String length ({len(value)}) exceeds maximum ({max_len})",
                path=path,
                severity="high"
            ))
        
        # Check for dangerous patterns
        for pattern, desc in self.compiled_patterns:
            if pattern.search(value):
                issues.append(SecurityIssue(
                    code="E004_SECURITY_02",
                    message=f"Dangerous pattern detected: {desc}",
                    path=path,
                    severity="critical"
                ))


# ============================================================================
# RESOURCE LIMITER
# ============================================================================

@dataclass
class ResourceLimit:
    """Represents a resource limit violation"""
    resource: str
    requested: float
    limit: float
    message: str


class ResourceLimiter:
    """
    Enforces resource limits on simulations.
    
    Protects against:
    - Memory exhaustion
    - CPU abuse
    - Disk space consumption
    - Excessive runtime
    """
    
    def __init__(
        self,
        max_cells: int = 10_000_000,
        max_time_steps: int = 100_000,
        max_runtime_hours: float = 2.0,
        max_memory_gb: float = 16.0,
        max_cpus: int = 8,
        max_write_interval_ratio: float = 0.001  # Write at most 1000 timesteps
    ):
        """
        Initialize resource limiter with limits.
        
        Args:
            max_cells: Maximum number of mesh cells
            max_time_steps: Maximum number of time steps
            max_runtime_hours: Maximum wall-clock runtime in hours
            max_memory_gb: Maximum memory in GB
            max_cpus: Maximum number of CPU cores
            max_write_interval_ratio: Minimum write interval / end_time ratio
        """
        self.max_cells = max_cells
        self.max_time_steps = max_time_steps
        self.max_runtime_hours = max_runtime_hours
        self.max_memory_gb = max_memory_gb
        self.max_cpus = max_cpus
        self.max_write_interval_ratio = max_write_interval_ratio
        
        # Cell count estimates per resolution
        self.cell_estimates = {
            MeshResolution.COARSE: 10_000,
            MeshResolution.MEDIUM: 100_000,
            MeshResolution.FINE: 1_000_000,
            MeshResolution.VERY_FINE: 5_000_000,
        }
    
    def validate_spec(self, spec: CFDSpecification) -> Tuple[bool, List[ResourceLimit]]:
        """
        Validate specification against resource limits.
        
        Args:
            spec: CFDSpecification to validate
            
        Returns:
            Tuple of (within_limits, list of violations)
        """
        violations: List[ResourceLimit] = []
        
        # 1. Estimate cell count
        self._check_cell_count(spec, violations)
        
        # 2. Check iteration count
        self._check_iterations(spec, violations)
        
        # 3. Check parallel processors
        self._check_processors(spec, violations)
        
        # 4. Check time steps for transient
        self._check_time_steps(spec, violations)
        
        # 5. Check write frequency
        self._check_write_frequency(spec, violations)
        
        return len(violations) == 0, violations
    
    def _check_cell_count(self, spec: CFDSpecification, violations: List[ResourceLimit]):
        """Estimate and check cell count"""
        resolution = spec.mesh.resolution
        base_estimate = self.cell_estimates.get(resolution, 100_000)
        
        # Multiply for 3D geometries
        if "3d" in spec.geometry.type.value.lower():
            estimated_cells = base_estimate * 10
        else:
            estimated_cells = base_estimate
        
        # Use explicit max_cells if provided
        if spec.mesh.max_cells:
            estimated_cells = spec.mesh.max_cells
        
        if estimated_cells > self.max_cells:
            violations.append(ResourceLimit(
                resource="mesh_cells",
                requested=estimated_cells,
                limit=self.max_cells,
                message=f"Estimated cell count ({estimated_cells:,}) exceeds limit ({self.max_cells:,})"
            ))
    
    def _check_iterations(self, spec: CFDSpecification, violations: List[ResourceLimit]):
        """Check iteration count for steady-state simulations"""
        max_iters = spec.solver.max_iterations
        
        if max_iters > self.max_time_steps:
            violations.append(ResourceLimit(
                resource="iterations",
                requested=max_iters,
                limit=self.max_time_steps,
                message=f"Max iterations ({max_iters:,}) exceeds limit ({self.max_time_steps:,})"
            ))
    
    def _check_processors(self, spec: CFDSpecification, violations: List[ResourceLimit]):
        """Check parallel processor count"""
        n_procs = spec.execution.num_processors
        
        if n_procs > self.max_cpus:
            violations.append(ResourceLimit(
                resource="processors",
                requested=n_procs,
                limit=self.max_cpus,
                message=f"Requested CPUs ({n_procs}) exceeds limit ({self.max_cpus})"
            ))
    
    def _check_time_steps(self, spec: CFDSpecification, violations: List[ResourceLimit]):
        """Check time step count for transient simulations"""
        if spec.time is None:
            return
        
        estimated_steps = spec.time.end_time / spec.time.delta_t
        
        if estimated_steps > self.max_time_steps:
            violations.append(ResourceLimit(
                resource="time_steps",
                requested=estimated_steps,
                limit=self.max_time_steps,
                message=f"Estimated time steps ({estimated_steps:,.0f}) exceeds limit ({self.max_time_steps:,})"
            ))
    
    def _check_write_frequency(self, spec: CFDSpecification, violations: List[ResourceLimit]):
        """Check output write frequency to prevent disk exhaustion"""
        if spec.time is None:
            return
        
        write_count = spec.time.end_time / spec.time.write_interval
        
        if write_count > 1000:
            violations.append(ResourceLimit(
                resource="write_count",
                requested=write_count,
                limit=1000,
                message=f"Estimated output writes ({write_count:,.0f}) may exhaust disk space. "
                        f"Consider increasing write_interval"
            ))
    
    def get_docker_limits(self) -> Dict[str, str]:
        """
        Get Docker resource limit flags.
        
        Returns:
            Dict of Docker flag names to values
        """
        return {
            "--memory": f"{self.max_memory_gb}g",
            "--cpus": str(self.max_cpus),
            "--pids-limit": "100",
            "--network": "none",
            "--security-opt": "no-new-privileges",
        }


# ============================================================================
# COMBINED SECURITY CHECK
# ============================================================================

class SecurityChecker:
    """
    Combined security and resource validation.
    
    This is the main security boundary that all specifications
    must pass through before case generation.
    """
    
    def __init__(
        self,
        security_validator: SecurityValidator | None = None,
        resource_limiter: ResourceLimiter | None = None
    ):
        """
        Initialize security checker.
        
        Args:
            security_validator: Optional custom SecurityValidator
            resource_limiter: Optional custom ResourceLimiter
        """
        self.security = security_validator or SecurityValidator()
        self.resources = resource_limiter or ResourceLimiter()
    
    def check(self, spec: CFDSpecification) -> Tuple[bool, Dict[str, Any]]:
        """
        Perform complete security check.
        
        Args:
            spec: CFDSpecification to check
            
        Returns:
            Tuple of (is_safe, details dict with any issues)
        """
        # Security validation
        sec_ok, sec_issues = self.security.validate_spec(spec)
        
        # Resource validation
        res_ok, res_violations = self.resources.validate_spec(spec)
        
        details = {
            "security_passed": sec_ok,
            "resources_passed": res_ok,
            "security_issues": [
                {"code": i.code, "message": i.message, "path": i.path, "severity": i.severity}
                for i in sec_issues
            ],
            "resource_violations": [
                {"resource": v.resource, "requested": v.requested, "limit": v.limit, "message": v.message}
                for v in res_violations
            ]
        }
        
        return sec_ok and res_ok, details


def validate_security(spec: CFDSpecification) -> Tuple[bool, Dict[str, Any]]:
    """
    Convenience function for security validation.
    
    Args:
        spec: CFDSpecification to validate
        
    Returns:
        Tuple of (is_safe, details)
    """
    checker = SecurityChecker()
    return checker.check(spec)
