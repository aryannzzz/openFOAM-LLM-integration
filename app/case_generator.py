"""
Case Generator Module
Generates OpenFOAM cases from CFDSpecification.
Based on design document Section 4.
"""
import shutil
import uuid
import re
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from app.schemas import (
    CFDSpecification, GeometryType, FlowRegime, TimeDependence,
    TurbulenceModel, SolverType, MeshResolution
)
from app.errors import TemplateNotFoundError, CaseGenerationError

logger = logging.getLogger(__name__)


class CaseGenerator:
    """
    Generates OpenFOAM cases from CFDSpecification.
    
    This class handles:
    - Template selection based on geometry type
    - Dictionary file modification (controlDict, fvSchemes, fvSolution, etc.)
    - Boundary condition setup
    - Turbulence model file injection
    - Function objects for post-processing
    """
    
    def __init__(self, templates_dir: Path, work_dir: Path = Path("/tmp/foam_runs")):
        """
        Initialize case generator.
        
        Args:
            templates_dir: Directory containing case templates
            work_dir: Directory for generated cases
        """
        self.templates_dir = Path(templates_dir)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(self, spec: CFDSpecification) -> Path:
        """
        Generate a complete OpenFOAM case from specification.
        
        Args:
            spec: Validated CFDSpecification
            
        Returns:
            Path to the generated case directory
        """
        # Create isolated case directory
        case_id = str(uuid.uuid4())[:8]
        case_name = spec.metadata.name
        case_dir = self.work_dir / f"{case_name}_{case_id}"
        
        logger.info(f"Generating case: {case_dir}")
        
        try:
            # Copy or create base template
            template_name = spec.geometry.type.value
            self._setup_base_case(case_dir, template_name)
            
            # Apply modifications
            self._modify_control_dict(case_dir, spec)
            self._modify_fv_schemes(case_dir, spec)
            self._modify_fv_solution(case_dir, spec)
            self._modify_transport_properties(case_dir, spec)
            self._modify_boundary_conditions(case_dir, spec)
            
            # Add turbulence model files if needed
            if spec.flow.regime != FlowRegime.LAMINAR:
                self._apply_turbulence_model(case_dir, spec)
            
            # Add function objects for post-processing
            self._add_function_objects(case_dir, spec)
            
            # Save specification for reproducibility
            self._save_specification(case_dir, spec)
            
            logger.info(f"Case generation completed: {case_dir}")
            return case_dir
            
        except Exception as e:
            logger.error(f"Case generation failed: {e}")
            if case_dir.exists():
                shutil.rmtree(case_dir)
            raise CaseGenerationError(str(e))
    
    def _setup_base_case(self, case_dir: Path, template_name: str):
        """Set up base case structure from template or create new"""
        template_dir = self.templates_dir / template_name
        
        if template_dir.exists():
            # Copy existing template
            shutil.copytree(template_dir, case_dir)
            logger.info(f"Copied template: {template_name}")
        else:
            # Create minimal case structure
            logger.warning(f"Template not found: {template_name}, creating minimal structure")
            self._create_minimal_case(case_dir)
    
    def _create_minimal_case(self, case_dir: Path):
        """Create a minimal OpenFOAM case structure"""
        (case_dir / "0").mkdir(parents=True)
        (case_dir / "constant").mkdir(parents=True)
        (case_dir / "system").mkdir(parents=True)
    
    def _modify_control_dict(self, case_dir: Path, spec: CFDSpecification):
        """Generate system/controlDict based on specification"""
        control_dict_path = case_dir / "system" / "controlDict"
        
        solver_type = spec.solver.type.value
        time_dep = spec.flow.time_dependence
        
        content = self._foam_header("controlDict", "dictionary")
        content += f"\napplication     {solver_type};\n\n"
        
        if time_dep == TimeDependence.STEADY:
            content += """startFrom       startTime;
startTime       0;
stopAt          endTime;
"""
            content += f"endTime         {spec.solver.max_iterations};\n"
            content += "deltaT          1;\n"
            content += "writeControl    timeStep;\n"
            content += "writeInterval   100;\n"
        else:
            time_spec = spec.time
            content += """startFrom       startTime;
startTime       0;
stopAt          endTime;
"""
            content += f"endTime         {time_spec.end_time};\n"
            content += f"deltaT          {time_spec.delta_t};\n"
            
            if time_spec.adjustable_time_step:
                content += "writeControl    adjustableRunTime;\n"
                content += "adjustTimeStep  yes;\n"
                content += f"maxCo           {time_spec.max_courant};\n"
            else:
                content += "writeControl    runTime;\n"
            
            content += f"writeInterval   {time_spec.write_interval};\n"
        
        content += """
purgeWrite      0;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""
        
        with open(control_dict_path, 'w') as f:
            f.write(content)
        
        logger.info("Generated controlDict")
    
    def _modify_fv_schemes(self, case_dir: Path, spec: CFDSpecification):
        """Generate system/fvSchemes based on flow regime"""
        fv_schemes_path = case_dir / "system" / "fvSchemes"
        
        regime = spec.flow.regime
        time_dep = spec.flow.time_dependence
        
        # Select appropriate schemes
        if time_dep == TimeDependence.STEADY:
            ddt_scheme = "steadyState"
        else:
            ddt_scheme = "Euler"
        
        if regime == FlowRegime.LAMINAR:
            div_u_scheme = "Gauss linear"
        elif regime == FlowRegime.TURBULENT_RANS:
            div_u_scheme = "bounded Gauss linearUpwindV grad(U)"
        else:  # LES
            div_u_scheme = "Gauss linear"
        
        content = self._foam_header("fvSchemes", "dictionary")
        content += f"""
ddtSchemes
{{
    default         {ddt_scheme};
}}

gradSchemes
{{
    default         Gauss linear;
    grad(U)         cellLimited Gauss linear 1;
    grad(p)         Gauss linear;
}}

divSchemes
{{
    default         none;
    div(phi,U)      {div_u_scheme};
"""
        
        # Add turbulence-specific div schemes
        if regime == FlowRegime.TURBULENT_RANS:
            content += """    div(phi,k)      bounded Gauss limitedLinear 1;
    div(phi,omega)  bounded Gauss limitedLinear 1;
    div(phi,epsilon) bounded Gauss limitedLinear 1;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
"""
        
        content += """}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}

wallDist
{
    method          meshWave;
}
"""
        
        with open(fv_schemes_path, 'w') as f:
            f.write(content)
        
        logger.info("Generated fvSchemes")
    
    def _modify_fv_solution(self, case_dir: Path, spec: CFDSpecification):
        """Generate system/fvSolution"""
        fv_solution_path = case_dir / "system" / "fvSolution"
        
        regime = spec.flow.regime
        conv = spec.solver.convergence_criteria
        algorithm = spec.solver.algorithm.value
        
        content = self._foam_header("fvSolution", "dictionary")
        content += """
solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-7;
        relTol          0.01;
        smoother        GaussSeidel;
        nPreSweeps      0;
        nPostSweeps     2;
        cacheAgglomeration on;
        agglomerator    faceAreaPair;
        nCellsInCoarsestLevel 10;
        mergeLevels     1;
    }

    pFinal
    {
        $p;
        relTol          0;
    }

    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
    }

    UFinal
    {
        $U;
        relTol          0;
    }
"""
        
        # Add turbulence variable solvers
        if regime != FlowRegime.LAMINAR:
            content += """
    "(k|omega|epsilon|nuTilda)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-8;
        relTol          0.1;
    }

    "(k|omega|epsilon|nuTilda)Final"
    {
        $k;
        relTol          0;
    }
"""
        
        content += "}\n\n"
        
        # SIMPLE/PIMPLE/PISO settings
        content += f"{algorithm}\n{{\n"
        
        if algorithm == "SIMPLE":
            content += """    nNonOrthogonalCorrectors 0;
    consistent      yes;

    residualControl
    {
"""
            content += f"        p               {conv.p};\n"
            content += f"        U               {conv.U};\n"
            if conv.k:
                content += f"        k               {conv.k};\n"
            if conv.omega:
                content += f"        omega           {conv.omega};\n"
            content += "    }\n"
        
        elif algorithm == "PIMPLE":
            content += """    nOuterCorrectors 2;
    nCorrectors     2;
    nNonOrthogonalCorrectors 1;
    pRefCell        0;
    pRefValue       0;
"""
        
        elif algorithm == "PISO":
            content += """    nCorrectors     2;
    nNonOrthogonalCorrectors 1;
    pRefCell        0;
    pRefValue       0;
"""
        
        content += "}\n\n"
        
        # Relaxation factors
        content += """relaxationFactors
{
    fields
    {
        p               0.3;
    }
    equations
    {
        U               0.7;
"""
        if regime != FlowRegime.LAMINAR:
            content += """        k               0.7;
        omega           0.7;
        epsilon         0.7;
"""
        content += """    }
}
"""
        
        with open(fv_solution_path, 'w') as f:
            f.write(content)
        
        logger.info("Generated fvSolution")
    
    def _modify_transport_properties(self, case_dir: Path, spec: CFDSpecification):
        """Generate constant/transportProperties"""
        props_path = case_dir / "constant" / "transportProperties"
        
        nu = spec.fluid.kinematic_viscosity
        
        content = self._foam_header("transportProperties", "dictionary")
        content += f"""
transportModel  Newtonian;

nu              [0 2 -1 0 0 0 0] {nu};
"""
        
        with open(props_path, 'w') as f:
            f.write(content)
        
        logger.info("Generated transportProperties")
    
    def _modify_boundary_conditions(self, case_dir: Path, spec: CFDSpecification):
        """Generate 0/ field files for boundary conditions"""
        zero_dir = case_dir / "0"
        
        # Generate pressure field
        self._generate_p_field(zero_dir, spec)
        
        # Generate velocity field
        self._generate_u_field(zero_dir, spec)
        
        # Generate turbulence fields if needed
        if spec.flow.regime != FlowRegime.LAMINAR:
            self._generate_turbulence_fields(zero_dir, spec)
        
        logger.info("Generated boundary condition files")
    
    def _generate_p_field(self, zero_dir: Path, spec: CFDSpecification):
        """Generate pressure field file"""
        boundaries = spec.boundaries
        ic = spec.initial_conditions
        
        content = self._foam_header("p", "volScalarField")
        content += f"""
dimensions      [0 2 -2 0 0 0 0];

internalField   uniform {ic.pressure};

boundaryField
{{
    inlet
    {{
"""
        
        if boundaries.inlet.type == "velocity_inlet":
            content += "        type            zeroGradient;\n"
        else:  # pressure_inlet
            content += f"        type            fixedValue;\n        value           uniform {boundaries.inlet.pressure or 0};\n"
        
        content += """    }
    outlet
    {
"""
        content += f"        type            fixedValue;\n        value           uniform {boundaries.outlet.pressure};\n"
        content += """    }
    walls
    {
        type            zeroGradient;
    }
"""
        
        # Handle symmetry planes
        if boundaries.symmetry:
            for plane in boundaries.symmetry.planes:
                content += f"""    {plane}
    {{
        type            empty;
    }}
"""
        
        content += "}\n"
        
        with open(zero_dir / "p", 'w') as f:
            f.write(content)
    
    def _generate_u_field(self, zero_dir: Path, spec: CFDSpecification):
        """Generate velocity field file"""
        boundaries = spec.boundaries
        ic = spec.initial_conditions
        
        vel = ic.velocity
        vel_str = f"({vel[0]} {vel[1]} {vel[2]})"
        
        content = self._foam_header("U", "volVectorField")
        content += f"""
dimensions      [0 1 -1 0 0 0 0];

internalField   uniform {vel_str};

boundaryField
{{
    inlet
    {{
"""
        
        if boundaries.inlet.type in ["velocity_inlet", "freestream"]:
            inlet_vel = boundaries.inlet.velocity
            inlet_vel_str = f"({inlet_vel[0]} {inlet_vel[1]} {inlet_vel[2]})"
            content += f"        type            fixedValue;\n        value           uniform {inlet_vel_str};\n"
        else:
            content += "        type            zeroGradient;\n"
        
        content += """    }
    outlet
    {
        type            zeroGradient;
    }
    walls
    {
"""
        
        if boundaries.walls.type == "no_slip":
            content += "        type            noSlip;\n"
        else:
            content += "        type            slip;\n"
        
        content += "    }\n"
        
        # Handle symmetry planes
        if boundaries.symmetry:
            for plane in boundaries.symmetry.planes:
                content += f"""    {plane}
    {{
        type            empty;
    }}
"""
        
        content += "}\n"
        
        with open(zero_dir / "U", 'w') as f:
            f.write(content)
    
    def _generate_turbulence_fields(self, zero_dir: Path, spec: CFDSpecification):
        """Generate turbulence field files"""
        turb_model = spec.flow.turbulence_model
        
        # Estimate turbulence quantities
        U_mag = sum(v**2 for v in spec.initial_conditions.velocity) ** 0.5
        L = spec.geometry.dimensions.characteristic_length
        
        # Turbulent intensity ~5%
        I = 0.05
        k = 1.5 * (U_mag * I) ** 2
        
        # Estimate omega and epsilon
        omega = k ** 0.5 / (0.09 ** 0.25 * L)
        epsilon = 0.09 * k ** 1.5 / L
        
        # Generate k field
        if turb_model in [TurbulenceModel.K_EPSILON, TurbulenceModel.K_OMEGA, 
                          TurbulenceModel.K_OMEGA_SST]:
            self._write_scalar_field(zero_dir / "k", "k", k, spec)
        
        # Generate omega field
        if turb_model in [TurbulenceModel.K_OMEGA, TurbulenceModel.K_OMEGA_SST]:
            self._write_scalar_field(zero_dir / "omega", "omega", omega, spec)
        
        # Generate epsilon field
        if turb_model == TurbulenceModel.K_EPSILON:
            self._write_scalar_field(zero_dir / "epsilon", "epsilon", epsilon, spec)
        
        # Generate nut field
        nut = k / omega if omega > 0 else 0
        self._write_nut_field(zero_dir / "nut", nut, spec)
    
    def _write_scalar_field(self, path: Path, field_name: str, value: float, 
                            spec: CFDSpecification):
        """Write a scalar turbulence field"""
        content = self._foam_header(field_name, "volScalarField")
        
        # Dimensions for k, omega, epsilon
        dims = {
            "k": "[0 2 -2 0 0 0 0]",
            "omega": "[0 0 -1 0 0 0 0]",
            "epsilon": "[0 2 -3 0 0 0 0]",
        }
        
        content += f"""
dimensions      {dims.get(field_name, "[0 0 0 0 0 0 0]")};

internalField   uniform {value};

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform {value};
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    walls
    {{
        type            {field_name}WallFunction;
        value           uniform {value};
    }}
"""
        
        if spec.boundaries.symmetry:
            for plane in spec.boundaries.symmetry.planes:
                content += f"""    {plane}
    {{
        type            empty;
    }}
"""
        
        content += "}\n"
        
        with open(path, 'w') as f:
            f.write(content)
    
    def _write_nut_field(self, path: Path, value: float, spec: CFDSpecification):
        """Write nut (turbulent viscosity) field"""
        content = self._foam_header("nut", "volScalarField")
        content += f"""
dimensions      [0 2 -1 0 0 0 0];

internalField   uniform {value};

boundaryField
{{
    inlet
    {{
        type            calculated;
        value           uniform 0;
    }}
    outlet
    {{
        type            calculated;
        value           uniform 0;
    }}
    walls
    {{
        type            nutkWallFunction;
        value           uniform 0;
    }}
"""
        
        if spec.boundaries.symmetry:
            for plane in spec.boundaries.symmetry.planes:
                content += f"""    {plane}
    {{
        type            empty;
    }}
"""
        
        content += "}\n"
        
        with open(path, 'w') as f:
            f.write(content)
    
    def _apply_turbulence_model(self, case_dir: Path, spec: CFDSpecification):
        """Generate turbulenceProperties file"""
        turb_path = case_dir / "constant" / "turbulenceProperties"
        
        regime = spec.flow.regime
        model = spec.flow.turbulence_model.value
        
        content = self._foam_header("turbulenceProperties", "dictionary")
        
        if regime == FlowRegime.TURBULENT_RANS:
            content += f"""
simulationType  RAS;

RAS
{{
    RASModel        {model};
    turbulence      on;
    printCoeffs     on;
}}
"""
        elif regime == FlowRegime.TURBULENT_LES:
            content += f"""
simulationType  LES;

LES
{{
    LESModel        {model};
    turbulence      on;
    printCoeffs     on;
    delta           cubeRootVol;
}}
"""
        
        with open(turb_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Generated turbulenceProperties for {model}")
    
    def _add_function_objects(self, case_dir: Path, spec: CFDSpecification):
        """Add function objects for force coefficients, etc."""
        outputs = spec.outputs
        derived = outputs.derived_quantities
        
        if not any(q in derived for q in ["drag_coefficient", "lift_coefficient"]):
            return
        
        # Calculate reference values
        L = spec.geometry.dimensions.characteristic_length
        rho = spec.fluid.density
        inlet_vel = spec.boundaries.inlet.velocity or [1, 0, 0]
        U_mag = sum(v**2 for v in inlet_vel) ** 0.5
        
        # Create force coefficients function object
        force_content = f"""
forceCoeffs
{{
    type            forceCoeffs;
    libs            (forces);
    writeControl    timeStep;
    writeInterval   1;
    
    patches         (walls);
    rho             rhoInf;
    rhoInf          {rho};
    liftDir         (0 1 0);
    dragDir         (1 0 0);
    CofR            (0 0 0);
    pitchAxis       (0 0 1);
    magUInf         {U_mag};
    lRef            {L};
    Aref            {L};
}}
"""
        
        # Append to controlDict
        control_dict_path = case_dir / "system" / "controlDict"
        with open(control_dict_path, 'a') as f:
            f.write("\nfunctions\n{\n")
            f.write(force_content)
            f.write("}\n")
        
        logger.info("Added forceCoeffs function object")
    
    def _save_specification(self, case_dir: Path, spec: CFDSpecification):
        """Save specification JSON for reproducibility"""
        import json
        spec_path = case_dir / "cfd_specification.json"
        
        with open(spec_path, 'w') as f:
            json.dump(spec.model_dump(), f, indent=2)
        
        logger.info("Saved CFDSpecification")
    
    def _foam_header(self, object_name: str, class_name: str) -> str:
        """Generate standard OpenFOAM file header"""
        return f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2312                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {class_name};
    object      {object_name};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""
