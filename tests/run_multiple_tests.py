#!/usr/bin/env python3
"""
Multiple Test Cases for LLM-Driven OpenFOAM System
Tests various prompts and flow conditions
"""
import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

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

def create_case_files(case_dir, spec, nu):
    """Create OpenFOAM case files from specification"""
    (case_dir / "system").mkdir(exist_ok=True, parents=True)
    (case_dir / "constant").mkdir(exist_ok=True)
    (case_dir / "0").mkdir(exist_ok=True)
    
    control_dict = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}

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
    
    fv_schemes = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}

ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes { default none; div(phi,U) Gauss linearUpwind grad(U); }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
"""
    (case_dir / "system" / "fvSchemes").write_text(fv_schemes)
    
    fv_solution = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}

solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-06; relTol 0.01; }
    pFinal { $p; relTol 0; }
    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-05; relTol 0.1; }
    UFinal { $U; relTol 0; }
}

PIMPLE
{
    nOuterCorrectors 1;
    nCorrectors 2;
    nNonOrthogonalCorrectors 1;
    pRefCell 0;
    pRefValue 0;
}
"""
    (case_dir / "system" / "fvSolution").write_text(fv_solution)
    
    block_mesh = """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}

scale 0.1;

vertices
(
    (0 0 0) (1 0 0) (1 1 0) (0 1 0)
    (0 0 0.1) (1 0 0.1) (1 1 0.1) (0 1 0.1)
);

blocks ( hex (0 1 2 3 4 5 6 7) (20 20 1) simpleGrading (1 1 1) );

boundary
(
    movingWall { type wall; faces ( (3 7 6 2) ); }
    fixedWalls { type wall; faces ( (0 4 7 3) (2 6 5 1) (1 5 4 0) ); }
    frontAndBack { type empty; faces ( (0 3 2 1) (4 5 6 7) ); }
);
"""
    (case_dir / "system" / "blockMeshDict").write_text(block_mesh)
    
    transport = f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}}

viscosityModel constant;
nu [0 2 -1 0 0 0 0] {nu};
"""
    (case_dir / "constant" / "transportProperties").write_text(transport)
    
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
    
    p = """FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}

dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;

boundaryField
{
    movingWall { type zeroGradient; }
    fixedWalls { type zeroGradient; }
    frontAndBack { type empty; }
}
"""
    (case_dir / "0" / "p").write_text(p)
    
    U = """FoamFile
{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}

dimensions [0 1 -1 0 0 0 0];
internalField uniform (0 0 0);

boundaryField
{
    movingWall { type fixedValue; value uniform (1 0 0); }
    fixedWalls { type noSlip; }
    frontAndBack { type empty; }
}
"""
    (case_dir / "0" / "U").write_text(U)

def test_case(prompt: str, case_name: str):
    """Run a single test case"""
    print(f"\n{'='*70}")
    print(f"  TEST: {case_name}")
    print(f"  Prompt: {prompt}")
    print('='*70)
    
    from app.llm_converter import LLMConverter
    from app.validation import PhysicsValidator
    from app.security import SecurityChecker
    
    # Step 1: LLM Conversion
    converter = LLMConverter(provider="openai", model="gpt-4o-mini")
    response = converter.convert_to_cfd_specification(prompt)
    spec = response.specification
    
    print(f"\n  ✓ LLM Conversion (confidence: {response.confidence_score:.2f})")
    print(f"    - Geometry: {spec.geometry.type.value}")
    print(f"    - Reynolds: {spec.flow.reynolds_number}")
    print(f"    - Regime: {spec.flow.regime.value}")
    print(f"    - Solver: {spec.solver.type.value}")
    
    # Step 2: Validation
    validator = PhysicsValidator()
    is_valid, errors = validator.validate(spec)
    print(f"  ✓ Physics: {'PASSED' if is_valid else 'FAILED'}")
    
    checker = SecurityChecker()
    sec_valid, sec_details = checker.check(spec)
    print(f"  ✓ Security: {'PASSED' if sec_valid else 'FAILED'}")
    
    # Step 3: Run OpenFOAM
    work_dir = Path("/tmp/foam_multi_test")
    case_dir = work_dir / case_name
    if case_dir.exists():
        shutil.rmtree(case_dir)
    
    nu = spec.fluid.kinematic_viscosity
    create_case_files(case_dir, spec, nu)
    
    # blockMesh
    result = run_openfoam_command("blockMesh", case_dir)
    if result.returncode == 0:
        print(f"  ✓ blockMesh: SUCCESS")
    else:
        print(f"  ✗ blockMesh: FAILED")
        return False
    
    # foamRun
    result = run_openfoam_command("foamRun 2>&1", case_dir, timeout=60)
    if "End" in result.stdout:
        time_dirs = [d for d in case_dir.iterdir() if d.is_dir() and d.name.replace('.', '').isdigit()]
        print(f"  ✓ foamRun: SUCCESS ({len(time_dirs)} time steps)")
        return True
    else:
        print(f"  ✗ foamRun: FAILED")
        print(f"    Error: {result.stderr[-200:]}" if result.stderr else "")
        return False

def main():
    print("\n" + "="*70)
    print("  LLM-DRIVEN OPENFOAM - MULTIPLE TEST CASES")
    print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*70)
    
    test_cases = [
        ("Simulate laminar flow in a 2D channel at Re=100", "channel_re100"),
        ("Model incompressible flow at Reynolds 50 for a flat plate", "flatplate_re50"),
        ("Run a CFD analysis of flow around a cylinder with Re=200", "cylinder_re200"),
        ("Compute air flow in a square cavity with moving lid", "cavity_flow"),
    ]
    
    results = []
    for prompt, name in test_cases:
        try:
            success = test_case(prompt, name)
            results.append((name, success))
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            results.append((name, False))
    
    print("\n" + "="*70)
    print("  SUMMARY")
    print("="*70)
    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"\n  Tests Passed: {passed}/{total}")
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"    {status} - {name}")
    
    print("\n" + "="*70)
    print(f"  {'✅ ALL TESTS PASSED!' if passed == total else f'⚠️ {total-passed} TESTS FAILED'}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
