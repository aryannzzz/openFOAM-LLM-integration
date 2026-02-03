"""
LLM Integration Module
Converts natural language descriptions to CFDSpecification.
Based on design document Section 3.
"""
import json
import logging
import os
from typing import Optional

from app.schemas import (
    CFDSpecification, LLMConversionResponse,
    Metadata, Geometry, GeometryType, GeometryDimensions, GeometryDomain,
    Flow, FlowRegime, TimeDependence, TurbulenceModel,
    Fluid, Solver, SolverType, SolverAlgorithm, ConvergenceCriteria,
    TimeSettings, Mesh, MeshResolution, BoundaryLayer,
    Boundaries, InletBoundary, OutletBoundary, WallBoundary, SymmetryBoundary,
    InitialConditions, Outputs, VisualizationOutputs, Execution
)
from app.validation import get_recommended_solver, get_recommended_turbulence_model

logger = logging.getLogger(__name__)


# System prompt from design document Section 3.1
SYSTEM_PROMPT = """You are a CFD (Computational Fluid Dynamics) specification generator. Your role is to translate natural language requests into structured JSON specifications for OpenFOAM simulations.

## CRITICAL CONSTRAINTS

1. You MUST output ONLY valid JSON conforming to the CFDSpecification schema
2. You MUST NEVER include:
   - Shell commands or scripts
   - Raw OpenFOAM dictionary syntax
   - File paths or system commands
   - Executable code of any kind
   - Comments outside the JSON structure

3. You MUST use ONLY these predefined geometry types:
   - cylinder_2d, cylinder_3d, sphere, flat_plate_2d
   - backward_facing_step, channel_2d, channel_3d
   - airfoil_naca_4digit, box_with_obstacle, pipe_3d

4. You MUST use ONLY these solvers:
   - simpleFoam, pimpleFoam, pisoFoam, icoFoam
   - rhoSimpleFoam, rhoPimpleFoam (compressible only)

5. You MUST select appropriate parameters based on physics:
   - Re < 2300 → laminar regime (typically)
   - Re > 4000 → turbulent regime (typically)
   - Mach < 0.3 → incompressible
   - Steady-state → simpleFoam (RANS) or pimpleFoam
   - Transient → pimpleFoam, pisoFoam, or icoFoam

## OUTPUT FORMAT

Your response must be a single JSON object with no additional text, markdown formatting, or explanation.

## PARAMETER INFERENCE

When the user does not specify certain parameters, infer reasonable defaults:
- If velocity not given but Re specified: compute U = Re * nu / L
- If time step not given for transient: estimate from CFL ≈ 0.5
- If mesh resolution not specified: use "medium"
- If turbulence model not specified for turbulent flow: use "kOmegaSST"
- If fluid not specified: assume air at STP (rho=1.225 kg/m³, nu=1.5e-5 m²/s)

## UNIT CONVENTIONS

All values in SI units:
- Length: meters (m)
- Velocity: meters per second (m/s)
- Pressure: Pascals (Pa)
- Density: kg/m³
- Viscosity: m²/s (kinematic)
- Time: seconds (s)
- Angles: degrees
"""


class LLMConverter:
    """Converts natural language descriptions to CFDSpecification"""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4"):
        self.provider = provider
        self.model = model
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize LLM client based on provider"""
        if self.provider == "openai":
            try:
                import openai
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.client = openai.OpenAI(api_key=api_key)
                else:
                    logger.warning("OPENAI_API_KEY not set, using mock mode")
            except ImportError:
                logger.warning("OpenAI package not installed")
        
        elif self.provider == "anthropic":
            try:
                import anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    self.client = anthropic.Anthropic(api_key=api_key)
                else:
                    logger.warning("ANTHROPIC_API_KEY not set, using mock mode")
            except ImportError:
                logger.warning("Anthropic package not installed")
    
    def convert_to_cfd_specification(self, prompt: str) -> LLMConversionResponse:
        """Convert natural language prompt to CFDSpecification."""
        logger.info(f"Converting prompt: {prompt[:100]}...")
        
        if self.client:
            try:
                if self.provider == "openai":
                    return self._convert_openai(prompt)
                elif self.provider == "anthropic":
                    return self._convert_anthropic(prompt)
            except Exception as e:
                logger.error(f"LLM conversion failed: {e}")
        
        return self._mock_conversion(prompt)
    
    def _convert_openai(self, prompt: str) -> LLMConversionResponse:
        """Convert using OpenAI API"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=3000,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return self._parse_llm_response(content, prompt)
    
    def _convert_anthropic(self, prompt: str) -> LLMConversionResponse:
        """Convert using Anthropic API"""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=3000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )
        content = message.content[0].text
        return self._parse_llm_response(content, prompt)
    
    def _parse_llm_response(self, content: str, original_prompt: str) -> LLMConversionResponse:
        """Parse LLM response into CFDSpecification"""
        try:
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            spec_dict = json.loads(content)
            
            # Transform LLM response to match our schema
            spec_dict = self._transform_llm_schema(spec_dict)
            
            spec = CFDSpecification(**spec_dict)
            
            return LLMConversionResponse(
                original_prompt=original_prompt,
                specification=spec,
                confidence_score=0.85,
                interpretation_notes="Converted via LLM",
                inferred_parameters=[]
            )
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return self._mock_conversion(original_prompt)
    
    def _transform_llm_schema(self, raw: dict) -> dict:
        """Transform LLM's raw JSON to match our CFDSpecification schema"""
        
        # Extract values from various formats GPT might return
        geometry = raw.get("geometry", {})
        flow = raw.get("flow", {})
        solver = raw.get("solver", "simpleFoam")
        
        # Extract Reynolds number
        re_val = flow.get("reynolds_number") or flow.get("Re") or flow.get("re") or 100
        
        # Extract geometry type
        geom_type = geometry.get("type", "cylinder_2d")
        if geom_type not in ["cylinder_2d", "cylinder_3d", "sphere", "flat_plate_2d", 
                             "backward_facing_step", "channel_2d", "channel_3d",
                             "airfoil_naca_4digit", "box_with_obstacle", "pipe_3d"]:
            geom_type = "cylinder_2d"
        
        # Extract flow regime
        regime = flow.get("regime") or flow.get("type", "laminar")
        if regime in ["laminar", "creeping"]:
            regime = "laminar"
            turb_model = "none"
        elif regime in ["turbulent", "turbulent_rans", "RANS"]:
            regime = "turbulent_rans"
            turb_model = "k_omega_sst"
        else:
            regime = "laminar" if re_val < 2300 else "turbulent_rans"
            turb_model = "none" if re_val < 2300 else "k_omega_sst"
        
        # Check for LLM turbulence model
        llm_turb = flow.get("turbulence_model") or flow.get("turbulenceModel", "")
        if llm_turb.lower() == "laminar":
            turb_model = "none"
            regime = "laminar"
        elif llm_turb.lower() in ["k-epsilon", "k_epsilon", "kepsilon"]:
            turb_model = "k_epsilon"
        elif llm_turb.lower() in ["k-omega", "k_omega", "k-omega-sst", "k_omega_sst"]:
            turb_model = "k_omega_sst"
        
        # Time dependence
        time_dep = flow.get("time_dependence", "steady")
        if time_dep.lower() in ["transient", "unsteady"]:
            time_dep = "transient"
        else:
            time_dep = "steady"
        
        # Solver type
        if isinstance(solver, str):
            solver_type = solver
        else:
            solver_type = solver.get("type", "simpleFoam")
        
        if solver_type not in ["simpleFoam", "pimpleFoam", "pisoFoam", "icoFoam", 
                               "rhoSimpleFoam", "rhoPimpleFoam"]:
            solver_type = "pimpleFoam" if time_dep == "transient" else "simpleFoam"
        
        # Fluid properties
        fluid = raw.get("fluid", {})
        nu = fluid.get("kinematic_viscosity") or fluid.get("nu") or 1.5e-5
        rho = fluid.get("density") or fluid.get("rho") or 1.225
        
        # Characteristic length
        L = geometry.get("characteristic_length") or geometry.get("diameter") or 0.01
        
        # Velocity
        velocity = flow.get("velocity") or flow.get("U")
        if velocity is None:
            velocity = re_val * nu / L
        if isinstance(velocity, (int, float)):
            velocity = [velocity, 0, 0]
        
        # Build the proper CFDSpecification structure
        name = f"{geom_type}_re{int(re_val)}"
        
        return {
            "metadata": {
                "name": name,
                "description": f"LLM generated CFD specification",
                "version": "1.0"
            },
            "geometry": {
                "type": geom_type,
                "dimensions": {
                    "characteristic_length": L
                },
                "domain": geometry.get("domain", {
                    "upstream": 10,
                    "downstream": 30,
                    "lateral": 10
                })
            },
            "flow": {
                "regime": regime,
                "reynolds_number": re_val,
                "time_dependence": time_dep,
                "turbulence_model": turb_model
            },
            "fluid": {
                "name": fluid.get("name", "air"),
                "density": rho,
                "kinematic_viscosity": nu
            },
            "solver": {
                "type": solver_type,
                "algorithm": "PIMPLE" if time_dep == "transient" else "SIMPLE",
                "max_iterations": 5000,
                "convergence_criteria": {"p": 1e-5, "U": 1e-5}
            },
            "time": {
                "end_time": 2.0,
                "delta_t": 0.0001,
                "adjustable_time_step": True,
                "max_courant": 1.0,
                "write_interval": 0.1
            } if time_dep == "transient" else None,
            "mesh": raw.get("mesh", {
                "resolution": "medium",
                "boundary_layer": {"enabled": True, "num_layers": 10}
            }),
            "boundaries": {
                "inlet": {"type": "velocity_inlet", "velocity": velocity},
                "outlet": {"type": "pressure_outlet", "pressure": 0},
                "walls": {"type": "no_slip"},
                "symmetry": {"planes": ["front", "back"]} if "2d" in geom_type else None
            },
            "initial_conditions": {
                "velocity": velocity,
                "pressure": 0
            },
            "outputs": raw.get("outputs", {
                "fields": ["p", "U"],
                "derived_quantities": ["drag_coefficient", "lift_coefficient"]
            }),
            "execution": raw.get("execution", {"parallel": False, "num_processors": 1})
        }
    
    def _mock_conversion(self, prompt: str) -> LLMConversionResponse:
        """Rule-based conversion when LLM is not available."""
        prompt_lower = prompt.lower()
        inferred = []
        
        # Extract Reynolds number
        re = self._extract_reynolds(prompt_lower)
        if re is None:
            re = 100
            inferred.append("reynolds_number (default: 100)")
        
        # Determine flow regime
        if re < 2300:
            regime = FlowRegime.LAMINAR
            turb_model = TurbulenceModel.NONE
        else:
            regime = FlowRegime.TURBULENT_RANS
            turb_model = TurbulenceModel.K_OMEGA_SST
        
        # Detect geometry
        geometry_type = self._detect_geometry(prompt_lower)
        if geometry_type is None:
            geometry_type = GeometryType.CYLINDER_2D
            inferred.append("geometry (default: cylinder_2d)")
        
        # Detect time dependence
        if any(word in prompt_lower for word in ["transient", "unsteady", "vortex"]):
            time_dep = TimeDependence.TRANSIENT
        elif geometry_type in [GeometryType.CYLINDER_2D] and re > 40:
            time_dep = TimeDependence.TRANSIENT
            inferred.append("time_dependence (transient for vortex shedding)")
        else:
            time_dep = TimeDependence.STEADY
        
        solver_type = get_recommended_solver(regime, time_dep)
        
        L = 0.01  # 1 cm default
        nu = 1.5e-5
        rho = 1.225
        U = re * nu / L
        
        name = f"{geometry_type.value}_re{int(re)}"
        
        spec = CFDSpecification(
            metadata=Metadata(name=name, description=f"From: {prompt[:100]}", version="1.0"),
            geometry=Geometry(
                type=geometry_type,
                dimensions=GeometryDimensions(characteristic_length=L),
                domain=GeometryDomain(upstream=10, downstream=30, lateral=10)
            ),
            flow=Flow(regime=regime, reynolds_number=re, time_dependence=time_dep, turbulence_model=turb_model),
            fluid=Fluid(name="air", density=rho, kinematic_viscosity=nu),
            solver=Solver(
                type=solver_type,
                algorithm=SolverAlgorithm.PIMPLE if time_dep == TimeDependence.TRANSIENT else SolverAlgorithm.SIMPLE,
                max_iterations=5000,
                convergence_criteria=ConvergenceCriteria(p=1e-5, U=1e-5)
            ),
            time=TimeSettings(end_time=2.0, delta_t=0.0001, adjustable_time_step=True, max_courant=1.0, write_interval=0.1) if time_dep == TimeDependence.TRANSIENT else None,
            mesh=Mesh(resolution=MeshResolution.MEDIUM, boundary_layer=BoundaryLayer(enabled=True, num_layers=10)),
            boundaries=Boundaries(
                inlet=InletBoundary(type="velocity_inlet", velocity=[U, 0, 0]),
                outlet=OutletBoundary(type="pressure_outlet", pressure=0),
                walls=WallBoundary(type="no_slip"),
                symmetry=SymmetryBoundary(planes=["front", "back"]) if "2d" in geometry_type.value else None
            ),
            initial_conditions=InitialConditions(velocity=[U, 0, 0], pressure=0),
            outputs=Outputs(fields=["p", "U"], derived_quantities=["drag_coefficient", "lift_coefficient"]),
            execution=Execution(parallel=False, num_processors=1)
        )
        
        return LLMConversionResponse(
            original_prompt=prompt,
            specification=spec,
            confidence_score=0.6,
            interpretation_notes=f"Rule-based: Re={re}, {geometry_type.value}, {regime.value}",
            inferred_parameters=inferred
        )
    
    def _extract_reynolds(self, prompt: str) -> Optional[float]:
        """Extract Reynolds number from prompt"""
        import re as regex
        patterns = [
            r're\s*[=:]\s*(\d+(?:\.\d+)?)',
            r'reynolds\s*(?:number)?\s*[=:of]*\s*(\d+(?:\.\d+)?)',
            r're(\d+)',
        ]
        for pattern in patterns:
            match = regex.search(pattern, prompt, regex.IGNORECASE)
            if match:
                return float(match.group(1))
        return None
    
    def _detect_geometry(self, prompt: str) -> Optional[GeometryType]:
        """Detect geometry type from prompt"""
        if "cylinder" in prompt:
            return GeometryType.CYLINDER_2D
        if "sphere" in prompt:
            return GeometryType.SPHERE
        if "airfoil" in prompt or "naca" in prompt:
            return GeometryType.AIRFOIL_NACA_4DIGIT
        if "channel" in prompt:
            return GeometryType.CHANNEL_2D
        if "pipe" in prompt:
            return GeometryType.PIPE_3D
        if "step" in prompt:
            return GeometryType.BACKWARD_FACING_STEP
        return None
