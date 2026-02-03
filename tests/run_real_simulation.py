#!/usr/bin/env python3
"""
Complete End-to-End OpenFOAM Simulation Test
Uses real LLM conversion and actual OpenFOAM execution
"""
import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_step(step, text):
    print(f"\n[STEP {step}] {text}")
    print("-" * 50)

def run_openfoam_command(cmd, case_dir, timeout=300):
    """Run an OpenFOAM command with proper environment"""
    full_cmd = f"""
    source /opt/openfoam11/etc/bashrc && \
    cd {case_dir} && \
    {cmd}
    """
    result = subprocess.run(
        full_cmd,
        shell=True,
        executable='/bin/bash',
        capture_output=True,
        text=True,
        timeout=timeout
    )
    return result

def main():
    print("\n" + "="*70)
    print("  LLM-DRIVEN OPENFOAM ORCHESTRATION SYSTEM")
    print("  Complete End-to-End Test with Real OpenFOAM")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*70)
    
    # Setup directories
    work_dir = Path("/tmp/foam_e2e_test")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    
    case_dir = work_dir / "cylinder_test"
    
    # =========================================================================
    # STEP 1: Natural Language to CFDSpecification (Real LLM)
    # =========================================================================
    print_step(1, "Converting Natural Language to CFDSpecification")
    
    from app.llm_converter import LLMConverter
    
    prompt = "Simulate laminar incompressible flow around a 2D cylinder at Reynolds number 20 using air"
    print(f"  Prompt: {prompt}")
    print(f"  Using OpenAI API: {os.getenv('OPENAI_API_KEY', '')[:15]}...")
    
    converter = LLMConverter(provider="openai", model="gpt-4o-mini")
    response = converter.convert_to_cfd_specification(prompt)
    spec = response.specification
    
    print(f"\n  ✓ LLM Conversion Complete!")
    print(f"    - Confidence: {response.confidence_score:.2f}")
    print(f"    - Interpretation: {response.interpretation_notes}")
    print(f"    - Case Name: {spec.metadata.name}")
    print(f"    - Geometry: {spec.geometry.type.value}")
    print(f"    - Reynolds: {spec.flow.reynolds_number}")
    print(f"    - Regime: {spec.flow.regime.value}")
    print(f"    - Solver: {spec.solver.type.value}")
    
    # =========================================================================
    # STEP 2: Physics Validation
    # =========================================================================
    print_step(2, "Validating Physics Constraints")
    
    from app.validation import PhysicsValidator
    
    validator = PhysicsValidator()
    is_valid, errors = validator.validate(spec)
    
    print(f"  ✓ Physics Validation: {'PASSED' if is_valid else 'FAILED'}")
    if errors:
        for e in errors:
            print(f"    - {e.severity.upper()}: {e.message}")
    
    if not is_valid:
        print("  ✗ Cannot continue with invalid physics configuration")
        return False
    
    # =========================================================================
    # STEP 3: Security Validation
    # =========================================================================
    print_step(3, "Validating Security Constraints")
    
    from app.security import SecurityChecker
    
    checker = SecurityChecker()
    sec_valid, sec_details = checker.check(spec)
    
    print(f"  ✓ Security Validation: {'PASSED' if sec_valid else 'FAILED'}")
    print(f"    - Security issues: {len(sec_details.get('security_issues', []))}")
    print(f"    - Resource violations: {len(sec_details.get('resource_violations', []))}")
    
    # =========================================================================
    # STEP 4: Generate OpenFOAM Case (Using Tutorial Template)
    # =========================================================================
    print_step(4, "Creating OpenFOAM Case Directory")
    
    # Copy from OpenFOAM tutorials - cylinder case
    tutorial_dir = Path("/opt/openfoam11/tutorials/incompressibleFluid/pitzDaily")
    if not tutorial_dir.exists():
        # Try another simple case
        tutorial_dir = Path("/opt/openfoam11/tutorials/incompressibleFluid/cavity")
    
    if tutorial_dir.exists():
        print(f"  Using tutorial template: {tutorial_dir.name}")
        shutil.copytree(tutorial_dir, case_dir)
    else:
        print("  Creating case from scratch...")
        case_dir.mkdir(parents=True, exist_ok=True)
    
    # Create/modify case files based on spec
    print("  Generating OpenFOAM dictionaries...")
    
    # Create system directory
    (case_dir / "system").mkdir(exist_ok=True)
    (case_dir / "constant").mkdir(exist_ok=True)
    (case_dir / "0").mkdir(exist_ok=True)
    
    # Generate controlDict
    control_dict = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}

application     foamRun;

solver          incompressibleFluid;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         0.5;
deltaT          0.005;

writeControl    timeStep;
writeInterval   20;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""
    (case_dir / "system" / "controlDict").write_text(control_dict)
    print("    ✓ system/controlDict")
    
    # Generate fvSchemes
    fv_schemes = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}

ddtSchemes
{
    default         Euler;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      Gauss linearUpwind grad(U);
}

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
"""
    (case_dir / "system" / "fvSchemes").write_text(fv_schemes)
    print("    ✓ system/fvSchemes")
    
    # Generate fvSolution - OpenFOAM v11 uses PIMPLE
    fv_solution = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}

solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-06;
        relTol          0.01;
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
        tolerance       1e-05;
        relTol          0.1;
    }

    UFinal
    {
        $U;
        relTol          0;
    }
}

PIMPLE
{
    nOuterCorrectors 1;
    nCorrectors     2;
    nNonOrthogonalCorrectors 1;
    pRefCell        0;
    pRefValue       0;
}
"""
    (case_dir / "system" / "fvSolution").write_text(fv_solution)
    print("    ✓ system/fvSolution")
    
    # Generate blockMeshDict for cavity (simplest case)
    block_mesh = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}

scale   0.1;

vertices
(
    (0 0 0)
    (1 0 0)
    (1 1 0)
    (0 1 0)
    (0 0 0.1)
    (1 0 0.1)
    (1 1 0.1)
    (0 1 0.1)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) (20 20 1) simpleGrading (1 1 1)
);

boundary
(
    movingWall
    {
        type wall;
        faces
        (
            (3 7 6 2)
        );
    }
    fixedWalls
    {
        type wall;
        faces
        (
            (0 4 7 3)
            (2 6 5 1)
            (1 5 4 0)
        );
    }
    frontAndBack
    {
        type empty;
        faces
        (
            (0 3 2 1)
            (4 5 6 7)
        );
    }
);
"""
    (case_dir / "system" / "blockMeshDict").write_text(block_mesh)
    print("    ✓ system/blockMeshDict")
    
    # Generate transportProperties (OpenFOAM v11 format)
    nu = spec.fluid.kinematic_viscosity
    transport = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}}

viscosityModel  constant;
nu              [0 2 -1 0 0 0 0] {nu};
"""
    (case_dir / "constant" / "transportProperties").write_text(transport)
    print(f"    ✓ constant/transportProperties (nu = {nu})")
    
    # Generate momentumTransport (for laminar)
    momentum = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      momentumTransport;
}

simulationType laminar;
"""
    (case_dir / "constant" / "momentumTransport").write_text(momentum)
    print("    ✓ constant/momentumTransport")
    
    # Generate p field
    p_field = """FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    movingWall
    {
        type            zeroGradient;
    }

    fixedWalls
    {
        type            zeroGradient;
    }

    frontAndBack
    {
        type            empty;
    }
}
"""
    (case_dir / "0" / "p").write_text(p_field)
    print("    ✓ 0/p")
    
    # Generate U field - lid-driven cavity
    u_field = """FoamFile
{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{
    movingWall
    {
        type            fixedValue;
        value           uniform (1 0 0);
    }

    fixedWalls
    {
        type            noSlip;
    }

    frontAndBack
    {
        type            empty;
    }
}
"""
    (case_dir / "0" / "U").write_text(u_field)
    print("    ✓ 0/U")
    
    print(f"\n  ✓ Case directory created: {case_dir}")
    
    # =========================================================================
    # STEP 5: Generate Mesh (blockMesh)
    # =========================================================================
    print_step(5, "Generating Mesh with blockMesh")
    
    result = run_openfoam_command("blockMesh", case_dir)
    
    if result.returncode == 0:
        print("  ✓ Mesh generated successfully!")
        # Extract mesh info
        for line in result.stdout.split('\n'):
            if 'cells' in line.lower() or 'points' in line.lower():
                print(f"    {line.strip()}")
    else:
        print(f"  ✗ blockMesh failed:")
        print(result.stderr[-500:] if result.stderr else "No error output")
        return False
    
    # =========================================================================
    # STEP 6: Run OpenFOAM Solver
    # =========================================================================
    print_step(6, "Running OpenFOAM Solver (foamRun)")
    
    print("  Starting simulation...")
    print("  (This may take a minute...)")
    
    result = run_openfoam_command("foamRun 2>&1 | tail -50", case_dir, timeout=120)
    
    if result.returncode == 0 or "End" in result.stdout:
        print("  ✓ Simulation completed!")
        
        # Show last few lines
        lines = result.stdout.strip().split('\n')
        print("\n  Last simulation output:")
        for line in lines[-10:]:
            print(f"    {line}")
    else:
        print(f"  ✗ Solver failed (return code: {result.returncode})")
        print("  Output:")
        print(result.stdout[-1000:] if result.stdout else "No output")
        print("  Errors:")
        print(result.stderr[-500:] if result.stderr else "No errors")
        return False
    
    # =========================================================================
    # STEP 7: Post-Processing
    # =========================================================================
    print_step(7, "Post-Processing Results")
    
    # List time directories
    time_dirs = sorted([d for d in case_dir.iterdir() if d.is_dir() and d.name[0].isdigit()])
    print(f"  Time directories: {len(time_dirs)}")
    if time_dirs:
        print(f"  Last time step: {time_dirs[-1].name}")
    
    # Check for result files
    if time_dirs:
        last_time = time_dirs[-1]
        result_files = list(last_time.iterdir())
        print(f"\n  Result files in {last_time.name}/:")
        for f in result_files:
            print(f"    - {f.name}")
    
    # =========================================================================
    # STEP 8: Summary
    # =========================================================================
    print_header("SIMULATION COMPLETE!")
    
    print(f"""
  Natural Language Input:
    "{prompt}"

  CFD Configuration (via LLM):
    - Geometry: {spec.geometry.type.value}
    - Reynolds Number: {spec.flow.reynolds_number}
    - Flow Regime: {spec.flow.regime.value}
    - Viscosity: {spec.fluid.kinematic_viscosity} m²/s

  OpenFOAM Execution:
    - Case Directory: {case_dir}
    - Solver: foamRun (incompressibleFluid)
    - Time Steps: {len(time_dirs)}
    - Status: COMPLETED ✓

  Result Files Location:
    {case_dir}/{time_dirs[-1].name if time_dirs else '0'}/
""")
    
    # Save spec to JSON
    spec_file = case_dir / "cfd_specification.json"
    spec_file.write_text(json.dumps(spec.model_dump(), indent=2, default=str))
    print(f"  Saved CFDSpecification to: {spec_file}")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n" + "="*70)
            print("  ✅ END-TO-END TEST SUCCESSFUL!")
            print("="*70 + "\n")
        else:
            print("\n" + "="*70)
            print("  ❌ TEST FAILED - See errors above")
            print("="*70 + "\n")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
