"""
Post-Processing Module
Extracts results from OpenFOAM simulations.
Based on design document Section 6.
"""
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

from app.schemas import CFDSpecification

logger = logging.getLogger(__name__)


@dataclass
class FieldData:
    """Container for OpenFOAM field data"""
    name: str
    field_type: str  # "scalar", "vector", "tensor"
    internal_field: np.ndarray
    time: float


class OpenFOAMFieldReader:
    """Reads OpenFOAM field files"""
    
    def __init__(self, case_dir: Path):
        self.case_dir = Path(case_dir)
    
    def get_time_directories(self) -> List[float]:
        """Get list of time directories"""
        times = []
        for item in self.case_dir.iterdir():
            if item.is_dir():
                try:
                    times.append(float(item.name))
                except ValueError:
                    continue
        return sorted(times)
    
    def get_latest_time(self) -> float:
        """Get the latest simulation time"""
        times = self.get_time_directories()
        return times[-1] if times else 0.0
    
    def read_field(self, field_name: str, time: float = None) -> Optional[FieldData]:
        """Read a field at specified time (or latest)"""
        if time is None:
            time = self.get_latest_time()
        
        time_str = str(int(time)) if time == int(time) else f"{time:g}"
        field_path = self.case_dir / time_str / field_name
        
        if not field_path.exists():
            logger.warning(f"Field file not found: {field_path}")
            return None
        
        return self._parse_field_file(field_path, time)
    
    def _parse_field_file(self, path: Path, time: float) -> FieldData:
        """Parse an OpenFOAM field file"""
        with open(path, 'r') as f:
            content = f.read()
        
        # Determine field type from class
        class_match = re.search(r'class\s+(\w+);', content)
        field_class = class_match.group(1) if class_match else "volScalarField"
        
        if "Vector" in field_class:
            field_type = "vector"
        elif "Tensor" in field_class:
            field_type = "tensor"
        else:
            field_type = "scalar"
        
        # Extract internal field values
        internal_field = self._extract_internal_field(content, field_type)
        
        return FieldData(
            name=path.name,
            field_type=field_type,
            internal_field=internal_field,
            time=time
        )
    
    def _extract_internal_field(self, content: str, field_type: str) -> np.ndarray:
        """Extract internal field values from file content"""
        # Check for uniform field
        uniform_match = re.search(r'internalField\s+uniform\s+\(?([^;)]+)\)?;', content)
        if uniform_match:
            values = [float(v) for v in uniform_match.group(1).split()]
            return np.array(values)
        
        # Non-uniform field
        nonuniform_match = re.search(
            r'internalField\s+nonuniform\s+List<\w+>\s+(\d+)\s*\(\s*(.*?)\s*\)',
            content, re.DOTALL
        )
        
        if nonuniform_match:
            values_str = nonuniform_match.group(2)
            
            if field_type == "scalar":
                values = [float(v) for v in values_str.split() if v.strip()]
            else:
                vector_pattern = r'\(([\d.e+-\s]+)\)'
                vectors = re.findall(vector_pattern, values_str)
                values = []
                for v in vectors:
                    values.extend([float(x) for x in v.split() if x.strip()])
            
            return np.array(values)
        
        return np.array([])


class ForceCoefficientsReader:
    """Reads force coefficients from function object output"""
    
    def __init__(self, case_dir: Path):
        self.case_dir = Path(case_dir)
    
    def read_force_coefficients(self) -> Dict[str, np.ndarray]:
        """Read force coefficient history"""
        postproc_dir = self.case_dir / "postProcessing" / "forceCoeffs"
        
        if not postproc_dir.exists():
            logger.warning(f"Force coefficients directory not found: {postproc_dir}")
            return {}
        
        # Find coefficient file
        for time_dir in sorted(postproc_dir.iterdir(), reverse=True):
            coeff_file = time_dir / "coefficient.dat"
            if coeff_file.exists():
                return self._parse_coefficients_file(coeff_file)
            
            # Also check for forceCoeffs.dat
            force_file = time_dir / "forceCoeffs.dat"
            if force_file.exists():
                return self._parse_coefficients_file(force_file)
        
        return {}
    
    def _parse_coefficients_file(self, path: Path) -> Dict[str, np.ndarray]:
        """Parse force coefficients file"""
        data = {'time': [], 'Cd': [], 'Cl': [], 'Cm': []}
        
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        data['time'].append(float(parts[0]))
                        data['Cd'].append(float(parts[1]))
                        data['Cl'].append(float(parts[3]))
                        if len(parts) > 5:
                            data['Cm'].append(float(parts[5]))
                    except (ValueError, IndexError):
                        continue
        
        return {k: np.array(v) for k, v in data.items() if v}


class ResidualReader:
    """Reads residual history from log files"""
    
    def __init__(self, case_dir: Path):
        self.case_dir = Path(case_dir)
    
    def read_residuals(self, log_file: str = "log") -> Dict[str, List[float]]:
        """Read residual history from solver log"""
        log_path = self.case_dir / log_file
        
        if not log_path.exists():
            # Try to find any log file
            for f in self.case_dir.glob("log*"):
                log_path = f
                break
        
        if not log_path.exists():
            return {}
        
        residuals = {}
        
        with open(log_path, 'r') as f:
            for line in f:
                # Match residual output lines
                # Example: "Solving for Ux, Initial residual = 0.001, Final residual = 0.0001"
                match = re.search(
                    r'Solving for (\w+),.*Initial residual = ([\d.e+-]+)',
                    line
                )
                if match:
                    field = match.group(1)
                    residual = float(match.group(2))
                    
                    if field not in residuals:
                        residuals[field] = []
                    residuals[field].append(residual)
        
        return residuals


class PostProcessor:
    """Main post-processing orchestrator"""
    
    def __init__(self, case_dir: Path):
        self.case_dir = Path(case_dir)
        self.field_reader = OpenFOAMFieldReader(case_dir)
        self.force_reader = ForceCoefficientsReader(case_dir)
        self.residual_reader = ResidualReader(case_dir)
    
    def extract_results(self, spec: CFDSpecification) -> Dict[str, Any]:
        """Extract all requested results from simulation"""
        outputs = spec.outputs
        
        results = {
            "metadata": {
                "case_dir": str(self.case_dir),
                "final_time": self.field_reader.get_latest_time(),
            },
            "fields": {},
            "derived_quantities": {},
            "convergence": {},
        }
        
        # Extract requested fields
        for field_name in outputs.fields:
            field_data = self.field_reader.read_field(field_name)
            if field_data is not None:
                results["fields"][field_name] = self._summarize_field(field_data)
        
        # Extract force coefficients
        derived = outputs.derived_quantities
        if any(q in derived for q in ["drag_coefficient", "lift_coefficient"]):
            force_coeffs = self.force_reader.read_force_coefficients()
            
            if "Cd" in force_coeffs and "drag_coefficient" in derived:
                results["derived_quantities"]["drag_coefficient"] = {
                    "final": float(force_coeffs["Cd"][-1]),
                    "mean": float(np.mean(force_coeffs["Cd"][-min(100, len(force_coeffs["Cd"])):])),
                }
            
            if "Cl" in force_coeffs and "lift_coefficient" in derived:
                results["derived_quantities"]["lift_coefficient"] = {
                    "final": float(force_coeffs["Cl"][-1]),
                    "mean": float(np.mean(force_coeffs["Cl"][-min(100, len(force_coeffs["Cl"])):])),
                    "rms": float(np.std(force_coeffs["Cl"][-min(100, len(force_coeffs["Cl"])):])),
                }
        
        # Compute Strouhal number if requested
        if "strouhal_number" in derived:
            st = self._compute_strouhal(spec, results)
            if st is not None:
                results["derived_quantities"]["strouhal_number"] = st
        
        # Extract residuals
        residuals = self.residual_reader.read_residuals()
        if residuals:
            results["convergence"]["residuals"] = {
                k: {
                    "initial": v[0] if v else None,
                    "final": v[-1] if v else None,
                    "iterations": len(v),
                }
                for k, v in residuals.items()
            }
            
            # Check convergence
            results["convergence"]["converged"] = self._check_convergence(residuals, spec)
        
        return results
    
    def _summarize_field(self, field: FieldData) -> Dict[str, Any]:
        """Create summary statistics for a field"""
        data = field.internal_field
        
        if len(data) == 0:
            return {"type": field.field_type, "empty": True}
        
        if field.field_type == "scalar":
            return {
                "type": "scalar",
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "mean": float(np.mean(data)),
            }
        elif field.field_type == "vector":
            # Reshape if needed
            if len(data) % 3 == 0:
                data = data.reshape(-1, 3)
                magnitudes = np.linalg.norm(data, axis=1)
            else:
                magnitudes = data  # Fallback
            
            return {
                "type": "vector",
                "magnitude": {
                    "min": float(np.min(magnitudes)),
                    "max": float(np.max(magnitudes)),
                    "mean": float(np.mean(magnitudes)),
                },
            }
        
        return {"type": field.field_type}
    
    def _compute_strouhal(self, spec: CFDSpecification, results: Dict) -> Optional[float]:
        """Compute Strouhal number from lift coefficient oscillations"""
        force_coeffs = self.force_reader.read_force_coefficients()
        
        if "Cl" not in force_coeffs or len(force_coeffs["Cl"]) < 100:
            return None
        
        L = spec.geometry.dimensions.characteristic_length
        inlet_vel = spec.boundaries.inlet.velocity or [1, 0, 0]
        U = sum(v**2 for v in inlet_vel) ** 0.5
        
        if spec.time is None:
            return None
        dt = spec.time.delta_t
        
        # FFT to find dominant frequency
        cl_signal = force_coeffs["Cl"] - np.mean(force_coeffs["Cl"])
        fft = np.fft.fft(cl_signal)
        freqs = np.fft.fftfreq(len(cl_signal), dt)
        
        positive_mask = freqs > 0
        magnitudes = np.abs(fft[positive_mask])
        freqs_positive = freqs[positive_mask]
        
        if len(magnitudes) == 0:
            return None
        
        dominant_idx = np.argmax(magnitudes)
        f_shedding = freqs_positive[dominant_idx]
        
        St = f_shedding * L / U
        
        # Sanity check for reasonable Strouhal numbers
        return float(St) if 0.1 < St < 1.0 else None
    
    def _check_convergence(self, residuals: Dict[str, List[float]], 
                          spec: CFDSpecification) -> bool:
        """Check if simulation has converged"""
        conv_criteria = spec.solver.convergence_criteria
        
        # Check p and U residuals
        p_residuals = residuals.get('p', [])
        u_residuals = residuals.get('Ux', residuals.get('U', []))
        
        p_converged = p_residuals and p_residuals[-1] < conv_criteria.p
        u_converged = u_residuals and u_residuals[-1] < conv_criteria.U
        
        return p_converged and u_converged
    
    def generate_summary(self, spec: CFDSpecification, results: Dict) -> str:
        """Generate human-readable summary of results"""
        lines = [
            f"Simulation: {spec.metadata.name}",
            f"Description: {spec.metadata.description}",
            "",
            "=== Flow Conditions ===",
            f"Reynolds number: {spec.flow.reynolds_number}",
            f"Regime: {spec.flow.regime.value}",
            f"Solver: {spec.solver.type.value}",
            "",
            "=== Results ===",
        ]
        
        # Field results
        for field, data in results.get("fields", {}).items():
            if data.get("type") == "scalar":
                lines.append(f"{field}: min={data['min']:.4g}, max={data['max']:.4g}, mean={data['mean']:.4g}")
            elif data.get("type") == "vector":
                mag = data.get("magnitude", {})
                lines.append(f"|{field}|: min={mag.get('min', 0):.4g}, max={mag.get('max', 0):.4g}, mean={mag.get('mean', 0):.4g}")
        
        # Derived quantities
        derived = results.get("derived_quantities", {})
        if "drag_coefficient" in derived:
            cd = derived["drag_coefficient"]
            lines.append(f"Drag coefficient (Cd): {cd['final']:.4f} (mean: {cd['mean']:.4f})")
        
        if "lift_coefficient" in derived:
            cl = derived["lift_coefficient"]
            lines.append(f"Lift coefficient (Cl): {cl['final']:.4f} (mean: {cl['mean']:.4f})")
        
        if "strouhal_number" in derived:
            lines.append(f"Strouhal number (St): {derived['strouhal_number']:.4f}")
        
        # Convergence
        conv = results.get("convergence", {})
        if conv:
            status = "Converged" if conv.get("converged", False) else "Did not converge"
            lines.append(f"\nConvergence status: {status}")
        
        return "\n".join(lines)


def extract_simulation_results(case_dir: Path, spec: CFDSpecification) -> Dict[str, Any]:
    """
    Convenience function to extract results from a completed simulation.
    
    Args:
        case_dir: Path to OpenFOAM case directory
        spec: Original CFDSpecification
        
    Returns:
        Dictionary containing all results
    """
    processor = PostProcessor(case_dir)
    return processor.extract_results(spec)
