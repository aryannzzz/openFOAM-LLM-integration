"""
Physics Validation Rules
Validates CFDSpecification against physics constraints.
Based on design document Section 2.2
"""
from typing import List, Tuple, Dict, Any
from app.schemas import CFDSpecification, FlowRegime, TimeDependence, TurbulenceModel, SolverType


# ============================================================================
# PHYSICS CONSTRAINTS MATRICES
# ============================================================================

PHYSICS_CONSTRAINTS = {
    # Solver-Regime Compatibility
    "solver_regime_matrix": {
        SolverType.ICO_FOAM: [FlowRegime.LAMINAR],
        SolverType.SIMPLE_FOAM: [FlowRegime.LAMINAR, FlowRegime.TURBULENT_RANS],
        SolverType.PIMPLE_FOAM: [FlowRegime.LAMINAR, FlowRegime.TURBULENT_RANS, FlowRegime.TURBULENT_LES],
        SolverType.PISO_FOAM: [FlowRegime.LAMINAR, FlowRegime.TURBULENT_RANS],
        SolverType.RHO_SIMPLE_FOAM: [FlowRegime.LAMINAR, FlowRegime.TURBULENT_RANS],
        SolverType.RHO_PIMPLE_FOAM: [FlowRegime.LAMINAR, FlowRegime.TURBULENT_RANS, FlowRegime.TURBULENT_LES],
    },

    # Solver-Time Compatibility
    "solver_time_matrix": {
        SolverType.SIMPLE_FOAM: [TimeDependence.STEADY],
        SolverType.RHO_SIMPLE_FOAM: [TimeDependence.STEADY],
        SolverType.PIMPLE_FOAM: [TimeDependence.TRANSIENT, TimeDependence.STEADY],
        SolverType.PISO_FOAM: [TimeDependence.TRANSIENT],
        SolverType.ICO_FOAM: [TimeDependence.TRANSIENT],
        SolverType.RHO_PIMPLE_FOAM: [TimeDependence.TRANSIENT],
    },

    # Turbulence Model Compatibility
    "turbulence_model_regime": {
        TurbulenceModel.NONE: [FlowRegime.LAMINAR],
        TurbulenceModel.K_EPSILON: [FlowRegime.TURBULENT_RANS],
        TurbulenceModel.K_OMEGA: [FlowRegime.TURBULENT_RANS],
        TurbulenceModel.K_OMEGA_SST: [FlowRegime.TURBULENT_RANS],
        TurbulenceModel.SPALART_ALLMARAS: [FlowRegime.TURBULENT_RANS],
        TurbulenceModel.SMAGORINSKY: [FlowRegime.TURBULENT_LES],
        TurbulenceModel.WALE: [FlowRegime.TURBULENT_LES],
        TurbulenceModel.DYNAMIC_K_EQN: [FlowRegime.TURBULENT_LES],
    },

    # Reynolds Number Regime Guidance
    "reynolds_regime_guidance": {
        FlowRegime.LAMINAR: {"min": 0.01, "max": 2300, "warning_max": 5000},
        FlowRegime.TURBULENT_RANS: {"min": 1000, "max": 1e9},
        FlowRegime.TURBULENT_LES: {"min": 1000, "max": 1e7},
    },
}

# Compressible solvers
COMPRESSIBLE_SOLVERS = {SolverType.RHO_SIMPLE_FOAM, SolverType.RHO_PIMPLE_FOAM}


class ValidationError:
    """Represents a single validation error"""
    
    def __init__(self, code: str, message: str, severity: str = "error"):
        self.code = code
        self.message = message
        self.severity = severity  # "error" or "warning"
    
    def __repr__(self):
        return f"[{self.severity.upper()}] {self.code}: {self.message}"


class PhysicsValidator:
    """
    Validates CFD specifications against physics constraints.
    
    This validation layer ensures:
    - Solver-regime compatibility
    - Solver-time compatibility
    - Turbulence model compatibility
    - Reynolds number range appropriateness
    - Transient simulation requirements
    - Compressibility consistency
    """
    
    def validate(self, spec: CFDSpecification) -> Tuple[bool, List[ValidationError]]:
        """
        Validate specification against all physics constraints.
        
        Returns:
            Tuple of (is_valid, list of errors/warnings)
        """
        errors: List[ValidationError] = []
        
        # Extract key parameters
        solver = spec.solver.type
        regime = spec.flow.regime
        time_dep = spec.flow.time_dependence
        turb_model = spec.flow.turbulence_model
        re = spec.flow.reynolds_number
        mach = spec.flow.mach_number
        
        # 1. Check solver-regime compatibility
        self._check_solver_regime(solver, regime, errors)
        
        # 2. Check solver-time compatibility
        self._check_solver_time(solver, time_dep, errors)
        
        # 3. Check turbulence model compatibility
        self._check_turbulence_model(turb_model, regime, errors)
        
        # 4. Check Reynolds number range
        self._check_reynolds_number(re, regime, errors)
        
        # 5. Check transient simulation requirements
        self._check_transient_requirements(spec, errors)
        
        # 6. Check compressibility consistency
        self._check_compressibility(solver, mach, errors)
        
        # 7. Check boundary condition consistency
        self._check_boundary_conditions(spec, errors)
        
        # 8. Check mesh settings
        self._check_mesh_settings(spec, errors)
        
        # Determine if valid (no errors, warnings OK)
        has_errors = any(e.severity == "error" for e in errors)
        return not has_errors, errors
    
    def _check_solver_regime(self, solver: SolverType, regime: FlowRegime, 
                             errors: List[ValidationError]):
        """Check solver-regime compatibility"""
        allowed_regimes = PHYSICS_CONSTRAINTS["solver_regime_matrix"].get(solver, [])
        if regime not in allowed_regimes:
            errors.append(ValidationError(
                code="E003_PHYSICS_01",
                message=f"Solver '{solver.value}' is incompatible with regime '{regime.value}'. "
                        f"Allowed regimes: {[r.value for r in allowed_regimes]}"
            ))
    
    def _check_solver_time(self, solver: SolverType, time_dep: TimeDependence,
                           errors: List[ValidationError]):
        """Check solver-time compatibility"""
        allowed_times = PHYSICS_CONSTRAINTS["solver_time_matrix"].get(solver, [])
        if time_dep not in allowed_times:
            errors.append(ValidationError(
                code="E003_PHYSICS_02",
                message=f"Solver '{solver.value}' is incompatible with '{time_dep.value}' simulation. "
                        f"Allowed: {[t.value for t in allowed_times]}"
            ))
    
    def _check_turbulence_model(self, turb_model: TurbulenceModel, regime: FlowRegime,
                                 errors: List[ValidationError]):
        """Check turbulence model compatibility"""
        allowed_regimes = PHYSICS_CONSTRAINTS["turbulence_model_regime"].get(turb_model, [])
        if regime not in allowed_regimes:
            errors.append(ValidationError(
                code="E003_PHYSICS_03",
                message=f"Turbulence model '{turb_model.value}' is incompatible with regime '{regime.value}'. "
                        f"Allowed regimes: {[r.value for r in allowed_regimes]}"
            ))
    
    def _check_reynolds_number(self, re: float, regime: FlowRegime,
                                errors: List[ValidationError]):
        """Check Reynolds number appropriateness"""
        bounds = PHYSICS_CONSTRAINTS["reynolds_regime_guidance"].get(regime, {})
        
        min_re = bounds.get("min", 0)
        max_re = bounds.get("max", 1e9)
        warning_max = bounds.get("warning_max")
        
        if re < min_re:
            errors.append(ValidationError(
                code="E003_PHYSICS_04",
                message=f"Reynolds number Re={re:.2e} is too low for {regime.value} regime "
                        f"(minimum: {min_re})"
            ))
        
        if re > max_re:
            errors.append(ValidationError(
                code="E003_PHYSICS_05",
                message=f"Reynolds number Re={re:.2e} is too high for {regime.value} regime "
                        f"(maximum: {max_re:.2e})"
            ))
        
        # Warning for transitional flows
        if regime == FlowRegime.LAMINAR and warning_max and re > warning_max:
            errors.append(ValidationError(
                code="W003_PHYSICS_01",
                message=f"Reynolds number Re={re:.2e} may exhibit transitional behavior. "
                        f"Consider using turbulent model for Re > {warning_max}",
                severity="warning"
            ))
    
    def _check_transient_requirements(self, spec: CFDSpecification,
                                       errors: List[ValidationError]):
        """Check transient simulation has required time settings"""
        if spec.flow.time_dependence == TimeDependence.TRANSIENT:
            if spec.time is None:
                errors.append(ValidationError(
                    code="E003_PHYSICS_06",
                    message="Transient simulation requires 'time' settings with end_time and delta_t"
                ))
            else:
                if spec.time.end_time <= 0:
                    errors.append(ValidationError(
                        code="E003_PHYSICS_07",
                        message="Transient simulation requires positive end_time"
                    ))
                if spec.time.delta_t <= 0:
                    errors.append(ValidationError(
                        code="E003_PHYSICS_08",
                        message="Transient simulation requires positive delta_t"
                    ))
                    
                # Check CFL condition (rough estimate)
                if spec.time.max_courant > 1.0 and not spec.time.adjustable_time_step:
                    errors.append(ValidationError(
                        code="W003_PHYSICS_02",
                        message=f"max_courant={spec.time.max_courant} > 1.0 without adjustable time step "
                                "may cause instability",
                        severity="warning"
                    ))
    
    def _check_compressibility(self, solver: SolverType, mach: float | None,
                                errors: List[ValidationError]):
        """Check compressibility consistency"""
        is_compressible_solver = solver in COMPRESSIBLE_SOLVERS
        
        if mach is not None:
            if mach >= 0.3 and not is_compressible_solver:
                errors.append(ValidationError(
                    code="E003_PHYSICS_09",
                    message=f"Mach number {mach} >= 0.3 requires a compressible solver "
                            f"(rhoSimpleFoam or rhoPimpleFoam), but '{solver.value}' was specified"
                ))
            
            if mach < 0.3 and is_compressible_solver:
                errors.append(ValidationError(
                    code="W003_PHYSICS_03",
                    message=f"Mach number {mach} < 0.3 suggests incompressible flow. "
                            "Consider using simpleFoam or pimpleFoam for better efficiency",
                    severity="warning"
                ))
    
    def _check_boundary_conditions(self, spec: CFDSpecification,
                                    errors: List[ValidationError]):
        """Check boundary condition consistency"""
        inlet = spec.boundaries.inlet
        
        # Velocity inlet must have velocity
        if inlet.type == "velocity_inlet" and inlet.velocity is None:
            errors.append(ValidationError(
                code="E003_PHYSICS_10",
                message="Velocity inlet boundary requires 'velocity' to be specified"
            ))
        
        # Pressure inlet must have pressure
        if inlet.type == "pressure_inlet" and inlet.pressure is None:
            errors.append(ValidationError(
                code="E003_PHYSICS_11",
                message="Pressure inlet boundary requires 'pressure' to be specified"
            ))
        
        # Check 2D geometry has symmetry planes
        if "2d" in spec.geometry.type.value.lower():
            if spec.boundaries.symmetry is None or not spec.boundaries.symmetry.planes:
                errors.append(ValidationError(
                    code="W003_PHYSICS_04",
                    message="2D geometry typically requires symmetry planes (front/back) for empty boundaries",
                    severity="warning"
                ))
    
    def _check_mesh_settings(self, spec: CFDSpecification,
                              errors: List[ValidationError]):
        """Check mesh settings for physical appropriateness"""
        mesh = spec.mesh
        regime = spec.flow.regime
        
        # Check y+ target for turbulent flows
        if regime in [FlowRegime.TURBULENT_RANS, FlowRegime.TURBULENT_LES]:
            if mesh.boundary_layer.enabled:
                y_plus = mesh.boundary_layer.target_y_plus
                
                # Wall-resolved LES/RANS typically needs y+ < 1
                if spec.flow.turbulence_model in [TurbulenceModel.K_OMEGA_SST, 
                                                   TurbulenceModel.K_OMEGA]:
                    if y_plus > 5:
                        errors.append(ValidationError(
                            code="W003_PHYSICS_05",
                            message=f"Target y+={y_plus} may be too high for wall-resolved "
                                    f"{spec.flow.turbulence_model.value}. Consider y+ < 1",
                            severity="warning"
                        ))
        
        # Check mesh resolution vs Reynolds number
        if spec.flow.reynolds_number > 1e6 and mesh.resolution in ["coarse", "medium"]:
            errors.append(ValidationError(
                code="W003_PHYSICS_06",
                message=f"High Reynolds number (Re={spec.flow.reynolds_number:.2e}) "
                        "may require finer mesh resolution",
                severity="warning"
            ))


def validate_physics_constraints(spec: CFDSpecification) -> Tuple[bool, List[ValidationError]]:
    """
    Convenience function to validate a specification.
    
    Args:
        spec: CFDSpecification to validate
        
    Returns:
        Tuple of (is_valid, list of errors/warnings)
    """
    validator = PhysicsValidator()
    return validator.validate(spec)


def get_recommended_solver(regime: FlowRegime, time_dep: TimeDependence, 
                           mach: float | None = None) -> SolverType:
    """
    Get recommended solver based on flow conditions.
    
    Args:
        regime: Flow regime
        time_dep: Time dependence
        mach: Mach number (optional)
        
    Returns:
        Recommended SolverType
    """
    is_compressible = mach is not None and mach >= 0.3
    
    if is_compressible:
        if time_dep == TimeDependence.STEADY:
            return SolverType.RHO_SIMPLE_FOAM
        else:
            return SolverType.RHO_PIMPLE_FOAM
    
    if regime == FlowRegime.LAMINAR:
        if time_dep == TimeDependence.STEADY:
            return SolverType.SIMPLE_FOAM
        else:
            return SolverType.ICO_FOAM
    
    # Turbulent
    if time_dep == TimeDependence.STEADY:
        return SolverType.SIMPLE_FOAM
    else:
        return SolverType.PIMPLE_FOAM


def get_recommended_turbulence_model(regime: FlowRegime, 
                                      re: float) -> TurbulenceModel:
    """
    Get recommended turbulence model.
    
    Args:
        regime: Flow regime
        re: Reynolds number
        
    Returns:
        Recommended TurbulenceModel
    """
    if regime == FlowRegime.LAMINAR:
        return TurbulenceModel.NONE
    
    if regime == FlowRegime.TURBULENT_RANS:
        # k-omega SST is generally robust and accurate
        return TurbulenceModel.K_OMEGA_SST
    
    if regime == FlowRegime.TURBULENT_LES:
        # WALE is good for wall-bounded flows
        return TurbulenceModel.WALE
    
    return TurbulenceModel.NONE
