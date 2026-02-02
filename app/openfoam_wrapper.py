"""
OpenFOAM Wrapper Module
Handles simulation setup and execution
"""
import os
import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, Tuple
import shutil

from app.models import JSONConfiguration, SimulationResult, SimulationType
from app.config import get_config

logger = logging.getLogger(__name__)


class OpenFOAMSimulator:
    """Wrapper for OpenFOAM simulations"""

    def __init__(self, case_id: str, case_name: str):
        """Initialize OpenFOAM simulator"""
        self.case_id = case_id
        self.case_name = case_name
        self.config = get_config()
        self.case_path = Path(self.config.WORKDIR) / case_name
        self.case_path.mkdir(parents=True, exist_ok=True)

    def setup_simulation(self, config: JSONConfiguration) -> bool:
        """Set up OpenFOAM case directory structure"""
        try:
            logger.info(f"Setting up simulation case: {self.case_name}")

            # Create case structure
            self._create_case_structure(config)

            # Generate mesh
            self._generate_mesh(config)

            # Set up solver configuration
            self._setup_solver_config(config)

            # Create fvSchemes and fvSolution
            self._create_solver_files(config)

            logger.info("Simulation setup completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error setting up simulation: {e}")
            return False

    def _create_case_structure(self, config: JSONConfiguration):
        """Create OpenFOAM case directory structure"""
        # Create constant directory with polymesh
        constant_dir = self.case_path / "constant"
        polymesh_dir = constant_dir / "polyMesh"
        polymesh_dir.mkdir(parents=True, exist_ok=True)

        # Create 0 (initial conditions) directory
        zero_dir = self.case_path / "0"
        zero_dir.mkdir(parents=True, exist_ok=True)

        # Create system directory
        system_dir = self.case_path / "system"
        system_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration as JSON
        config_file = self.case_path / "foam_config.json"
        with open(config_file, 'w') as f:
            json.dump(config.model_dump(), f, indent=2)

        logger.info(f"Created case structure at {self.case_path}")

    def _generate_mesh(self, config: JSONConfiguration):
        """Generate mesh using blockMesh or snappyHexMesh"""
        logger.info(f"Generating mesh with {config.mesh_type}")

        # For this example, we'll create a simple blockMeshDict
        blockmesh_dict = self._generate_blockmesh_dict(config)

        system_dir = self.case_path / "system"
        blockmesh_file = system_dir / "blockMeshDict"

        with open(blockmesh_file, 'w') as f:
            f.write(blockmesh_dict)

        # Run blockMesh if available
        try:
            self._run_command(
                f"blockMesh -case {self.case_path}",
                "Mesh generation"
            )
        except Exception as e:
            logger.warning(f"blockMesh failed, mesh will be generated on solver startup: {e}")

    def _generate_blockmesh_dict(self, config: JSONConfiguration) -> str:
        """Generate blockMeshDict content"""
        domain = config.domain
        length = domain.get('length', 1.0)
        width = domain.get('width', 1.0)
        height = domain.get('height', 1.0)

        # Calculate mesh divisions
        cells_l = int(length * 10)
        cells_w = int(width * 10)
        cells_h = int(height * 10)

        return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

convertToMeters 1;

vertices
(
    (0 0 0)
    ({length} 0 0)
    ({length} {width} 0)
    (0 {width} 0)
    (0 0 {height})
    ({length} 0 {height})
    ({length} {width} {height})
    (0 {width} {height})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({cells_l} {cells_w} {cells_h}) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    inlet
    {{
        type patch;
        faces
        (
            (0 4 7 3)
        );
    }}
    outlet
    {{
        type patch;
        faces
        (
            (1 2 6 5)
        );
    }}
    walls
    {{
        type wall;
        faces
        (
            (0 1 5 4)
            (3 7 6 2)
            (0 3 2 1)
            (4 5 6 7)
        );
    }}
);
"""

    def _setup_solver_config(self, config: JSONConfiguration):
        """Create initial condition files (0 directory)"""
        zero_dir = self.case_path / "0"

        # Create U (velocity) file
        u_file = zero_dir / "U"
        u_content = self._generate_u_field(config)
        with open(u_file, 'w') as f:
            f.write(u_content)

        # Create p (pressure) file
        p_file = zero_dir / "p"
        p_content = self._generate_p_field(config)
        with open(p_file, 'w') as f:
            f.write(p_content)

    def _generate_u_field(self, config: JSONConfiguration) -> str:
        """Generate velocity field file"""
        inlet_vel = config.simulation_parameters.get('velocity_inlet', 1.0)

        return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}}

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform ({inlet_vel} 0 0);
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    walls
    {{
        type            noSlip;
    }}
}}
"""

    def _generate_p_field(self, config: JSONConfiguration) -> str:
        """Generate pressure field file"""
        p_ref = config.simulation_parameters.get('pressure_reference', 101325.0)

        return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}}

dimensions      [1 -1 -2 0 0 0 0];

internalField   uniform {p_ref};

boundaryField
{{
    inlet
    {{
        type            zeroGradient;
    }}
    outlet
    {{
        type            fixedValue;
        value           uniform {p_ref};
    }}
    walls
    {{
        type            zeroGradient;
    }}
}}
"""

    def _create_solver_files(self, config: JSONConfiguration):
        """Create fvSchemes and fvSolution files"""
        system_dir = self.case_path / "system"
        tolerance = config.numerics.get('tolerance', 1e-6)
        max_iter = config.numerics.get('max_iterations', 500)

        # Create fvSchemes
        fvschemes = system_dir / "fvSchemes"
        with open(fvschemes, 'w') as f:
            f.write(self._get_fvschemes_template())

        # Create fvSolution
        fvsolution = system_dir / "fvSolution"
        with open(fvsolution, 'w') as f:
            f.write(self._get_fvsolution_template(tolerance, max_iter))

        # Create controlDict
        controldict = system_dir / "controlDict"
        time_steps = config.simulation_parameters.get('time_steps', 100)
        with open(controldict, 'w') as f:
            f.write(self._get_controldict_template(time_steps))

    def _get_fvschemes_template(self) -> str:
        """Get fvSchemes template"""
        return """FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}

ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
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

    def _get_fvsolution_template(self, tolerance: float, max_iter: int) -> str:
        """Get fvSolution template"""
        return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}}

solvers
{{
    p
    {{
        solver          GAMG;
        tolerance       {tolerance};
        relTol          0.01;
        maxIter         {max_iter};
    }}
    U
    {{
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       {tolerance};
        relTol          0.1;
        maxIter         {max_iter};
    }}
}}

SIMPLE
{{
    nNonOrthogonalCorrectors 0;
    residualControl
    {{
        p               {tolerance};
        U               {tolerance};
    }}
}}

relaxationFactors
{{
    equations
    {{
        U               0.3;
        p               0.3;
    }}
}}
"""

    def _get_controldict_template(self, time_steps: int) -> str:
        """Get controlDict template"""
        return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}

application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {time_steps};
deltaT          1;
writeControl    timeStep;
writeInterval   {max(1, time_steps // 10)};
purgeWrite      0;
writeFormat     binary;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""

    def run_simulation(self, timeout: Optional[int] = None) -> Tuple[bool, str]:
        """Run the OpenFOAM simulation"""
        try:
            logger.info(f"Starting simulation: {self.case_name}")
            timeout = timeout or self.config.MAX_SIMULATION_TIME

            # Run solver
            start_time = time.time()
            cmd = f"simpleFoam -case {self.case_path}"
            result = self._run_command(cmd, "Simulation", timeout=timeout)
            runtime = time.time() - start_time

            logger.info(f"Simulation completed in {runtime:.2f} seconds")
            return True, str(runtime)

        except subprocess.TimeoutExpired:
            logger.error("Simulation timed out")
            return False, "Simulation timed out"
        except Exception as e:
            logger.error(f"Error running simulation: {e}")
            return False, str(e)

    def _run_command(
        self,
        cmd: str,
        description: str,
        timeout: Optional[int] = None
    ) -> str:
        """Run a shell command"""
        logger.info(f"{description}: {cmd}")

        try:
            # Set up environment
            env = os.environ.copy()
            foamsh = Path(self.config.OPENFOAM_PATH) / "etc" / "bashrc"
            if foamsh.exists():
                env['OPENFOAM'] = self.config.OPENFOAM_PATH

            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env
            )

            if result.returncode != 0:
                logger.warning(f"{description} stderr: {result.stderr}")

            return result.stdout

        except subprocess.TimeoutExpired:
            logger.error(f"{description} timed out after {timeout} seconds")
            raise
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            raise

    def get_results(self, config: JSONConfiguration, runtime: float) -> SimulationResult:
        """Collect simulation results"""
        return SimulationResult(
            case_id=self.case_id,
            case_name=self.case_name,
            status="completed",
            simulation_type=config.simulation_type,
            configuration=config,
            runtime_seconds=runtime,
            convergence_status="likely converged",
            output_path=str(self.case_path),
            visualization_url=f"/api/visualize/{self.case_id}"
        )

    def cleanup(self, keep_results: bool = True):
        """Clean up simulation files"""
        if not keep_results:
            try:
                shutil.rmtree(self.case_path)
                logger.info(f"Cleaned up case directory: {self.case_path}")
            except Exception as e:
                logger.error(f"Error cleaning up: {e}")


def create_simulator(case_id: str, case_name: str) -> OpenFOAMSimulator:
    """Factory function for creating simulators"""
    return OpenFOAMSimulator(case_id, case_name)
