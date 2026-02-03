"""
Demo script showing complete workflow of LLM-OpenFOAM Orchestrator
Run this after starting the API server: python main.py
"""
import requests
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_section(title: str):
    """Print formatted section title"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}{Colors.END}\n")


def print_success(message: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")


def print_info(message: str):
    """Print info message"""
    print(f"{Colors.CYAN}ℹ {message}{Colors.END}")


def print_error(message: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {message}{Colors.END}")


def demo_health_check():
    """Demo: Check API health"""
    print_section("1. Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health")
        data = response.json()
        print_success(f"API Status: {data['status']}")
        print_info(f"Service: {data['service']}")
        print_info(f"Version: {data['version']}")
        return True
    except Exception as e:
        print_error(f"Failed to connect to API: {e}")
        return False


def demo_conversion():
    """Demo: Convert natural language to JSON configuration"""
    print_section("2. Natural Language to JSON Conversion")

    descriptions = [
        "Simulate incompressible laminar flow around a cylinder at 10 m/s with water",
        "Heat transfer problem in a rectangular channel with constant temperature wall",
        "Turbulent flow in a pipe with diameter 0.1 m and velocity 5 m/s"
    ]

    for i, desc in enumerate(descriptions, 1):
        print_info(f"Example {i}: {desc}")
        try:
            response = requests.post(
                f"{BASE_URL}/api/convert",
                params={"description": desc}
            )
            data = response.json()

            print_success(f"Conversion confidence: {data['confidence_score']:.1%}")
            config = data['json_configuration']
            print_info(f"Simulation Type: {config['simulation_type']}")
            print_info(f"Solver: {config['solver']}")
            print_info(f"Mesh Type: {config['mesh_type']}")

            print(f"\n{Colors.YELLOW}Configuration Summary:{Colors.END}")
            print(json.dumps(config, indent=2)[:300] + "...\n")

        except Exception as e:
            print_error(f"Conversion failed: {e}")


def demo_simulation_submission():
    """Demo: Submit simulations"""
    print_section("3. Submit Simulations")

    simulations = [
        {
            "description": "Laminar flow around a cylinder at 2 m/s",
            "case_name": "cylinder_laminar_2ms"
        },
        {
            "description": "Heat transfer in a channel with heat source",
            "case_name": "heat_transfer_channel"
        }
    ]

    case_ids = []

    for sim in simulations:
        print_info(f"Submitting: {sim['case_name']}")
        try:
            response = requests.post(
                f"{BASE_URL}/api/simulate",
                json={
                    "description": sim['description'],
                    "case_name": sim['case_name'],
                    "max_runtime": 600
                }
            )
            data = response.json()
            case_id = data['case_id']
            case_ids.append(case_id)

            print_success(f"Submitted successfully")
            print_info(f"Case ID: {case_id}")
            print_info(f"Status: {data['status']}")
            print_info(f"Type: {data['simulation_type']}")

        except Exception as e:
            print_error(f"Submission failed: {e}")

    return case_ids


def demo_check_status(case_ids: list):
    """Demo: Check simulation status"""
    print_section("4. Check Simulation Status")

    for case_id in case_ids:
        print_info(f"Checking status for: {case_id}")
        try:
            response = requests.get(f"{BASE_URL}/api/status/{case_id}")
            data = response.json()

            print_success(f"Case: {data['case_name']}")
            print_info(f"Status: {data['status']}")
            if data.get('progress'):
                print_info(f"Progress: {data['progress']}%")
            if data.get('time_elapsed'):
                print_info(f"Time Elapsed: {data['time_elapsed']:.1f}s")

        except Exception as e:
            print_error(f"Status check failed: {e}")

        print()


def demo_list_simulations():
    """Demo: List all simulations"""
    print_section("5. List All Simulations")

    try:
        response = requests.get(f"{BASE_URL}/api/simulations")
        data = response.json()

        print_info(f"Total Simulations: {data['total']}")
        print(f"\n{Colors.YELLOW}Simulation List:{Colors.END}")

        for sim in data['simulations']:
            print_info(
                f"{sim['case_name']} - "
                f"Status: {sim['status']} - "
                f"Type: {sim['simulation_type']}"
            )

    except Exception as e:
        print_error(f"Failed to list simulations: {e}")


def demo_monitoring():
    """Demo: Monitor simulation progress"""
    print_section("6. Real-time Monitoring (Demo)")

    print_info("Simulations are typically run in the background.")
    print_info("In a production system, you would poll /api/status/{case_id}")
    print_info("to monitor progress in real-time.")

    print(f"\n{Colors.YELLOW}Example polling loop:{Colors.END}")
    example_code = """
    import time
    
    case_id = "sim_20240131_12345678"
    while True:
        response = requests.get(f"{BASE_URL}/api/status/{case_id}")
        status = response.json()
        
        print(f"Progress: {status['progress']}%")
        
        if status['status'] == 'completed':
            break
        
        time.sleep(5)
    """
    print(example_code)


def demo_api_features():
    """Demo: Show available API features"""
    print_section("7. Available API Features")

    features = [
        ("POST /api/simulate", "Submit new simulation with natural language"),
        ("GET /api/status/{case_id}", "Get current simulation status"),
        ("GET /api/results/{case_id}", "Get completed simulation results"),
        ("POST /api/convert", "Convert description to JSON (no simulation)"),
        ("GET /api/simulations", "List all simulations and statuses"),
        ("POST /api/visualize/{case_id}", "Generate visualization"),
        ("DELETE /api/simulations/{case_id}", "Delete simulation and data"),
        ("GET /health", "Health check endpoint"),
        ("GET /", "API information"),
    ]

    for endpoint, description in features:
        print_info(f"{endpoint}")
        print(f"    → {description}\n")


def demo_supported_types():
    """Demo: Show supported simulation types"""
    print_section("8. Supported Simulation Types")

    sim_types = [
        ("incompressible_flow", "simpleFoam", "Steady-state incompressible flow"),
        ("compressible_flow", "rhoSimpleFoam", "High-speed compressible flow"),
        ("heat_transfer", "buoyantFoam", "Heat conduction/convection"),
        ("combustion", "reactingFoam", "Reactive flows and combustion"),
        ("multiphase", "interFoam", "Two-phase flows (bubbles, droplets)"),
    ]

    for sim_type, solver, description in sim_types:
        print_info(f"{sim_type} → {solver}")
        print(f"    {description}\n")


def demo_workflow():
    """Demo: Show typical workflow"""
    print_section("9. Typical Workflow")

    workflow = """
    1. DESCRIBE: "Simulate flow around a cylinder at 10 m/s"
                    ↓
    2. LLM CONVERTS: Natural language → OpenFOAM JSON configuration
                    ↓
    3. VALIDATE: Check configuration for physical validity
                    ↓
    4. SETUP: Create case directory, mesh, boundary conditions
                    ↓
    5. EXECUTE: Run OpenFOAM solver (async background task)
                    ↓
    6. MONITOR: Poll /api/status/{case_id} for progress
                    ↓
    7. RESULTS: Retrieve simulation results when complete
                    ↓
    8. VISUALIZE: Generate visualizations (ParaView, etc.)
    """
    print(workflow)


def main():
    """Run complete demo"""
    print(f"""
{Colors.BOLD}{Colors.CYAN}
╔═══════════════════════════════════════════════════════════╗
║  LLM-Driven OpenFOAM Orchestration System - DEMO          ║
║  Natural Language Interface to CFD Simulations            ║
╚═══════════════════════════════════════════════════════════╝
{Colors.END}
    """)

    # Health check
    if not demo_health_check():
        print_error("API is not running. Start it with: python main.py")
        return

    # Run demos
    demo_conversion()
    case_ids = demo_simulation_submission()
    
    if case_ids:
        demo_check_status(case_ids)
    
    demo_list_simulations()
    demo_monitoring()
    demo_api_features()
    demo_supported_types()
    demo_workflow()

    # Final message
    print_section("Demo Complete!")
    print(f"{Colors.GREEN}✓ Demo completed successfully!{Colors.END}")
    print(f"\n{Colors.YELLOW}Next Steps:{Colors.END}")
    print(f"1. Visit {Colors.BOLD}http://localhost:8000/docs{Colors.END} for interactive API docs")
    print(f"2. Check {Colors.BOLD}logs/foam_orchestrator.log{Colors.END} for detailed logs")
    print(f"3. Monitor {Colors.BOLD}/tmp/foam_simulations{Colors.END} for case data")
    print(f"\n{Colors.CYAN}For more information, see README.md{Colors.END}\n")


if __name__ == "__main__":
    main()
