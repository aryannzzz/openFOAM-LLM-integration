#!/usr/bin/env python3
"""
OpenFOAM Orchestration CLI
Command-line interface for the LLM-driven OpenFOAM system.
Based on design document Section 8.2
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="OpenFOAM Orchestration CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit simulation with natural language prompt
  foam-cli run "Simulate laminar flow over a cylinder at Re=100"
  
  # Submit with specification file
  foam-cli run --spec simulation.json
  
  # Check status
  foam-cli status abc12345
  
  # Get results
  foam-cli results abc12345 --output ./results
  
  # Interactive mode
  foam-cli interactive
        """
    )
    
    parser.add_argument(
        "--api-url", 
        default="http://localhost:8000",
        help="API server URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Submit a simulation")
    run_parser.add_argument("prompt", nargs="?", help="Natural language prompt")
    run_parser.add_argument("--spec", "-s", help="Specification JSON file")
    run_parser.add_argument("--wait", "-w", action="store_true", help="Wait for completion")
    run_parser.add_argument("--timeout", type=int, default=3600, help="Wait timeout in seconds")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check job status")
    status_parser.add_argument("job_id", help="Job ID")
    
    # Results command
    results_parser = subparsers.add_parser("results", help="Get results")
    results_parser.add_argument("job_id", help="Job ID")
    results_parser.add_argument("--output", "-o", default=".", help="Output directory")
    results_parser.add_argument("--format", choices=["json", "summary"], default="summary")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List simulations")
    list_parser.add_argument("--status", help="Filter by status")
    list_parser.add_argument("--limit", type=int, default=10, help="Max results")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate specification")
    validate_parser.add_argument("spec_file", help="Specification JSON file")
    
    # Capabilities command
    subparsers.add_parser("capabilities", help="Get system capabilities")
    
    # Interactive mode
    subparsers.add_parser("interactive", help="Interactive mode")
    
    # Convert command (NL to JSON without running)
    convert_parser = subparsers.add_parser("convert", help="Convert prompt to specification")
    convert_parser.add_argument("prompt", help="Natural language prompt")
    convert_parser.add_argument("--output", "-o", help="Output file (default: stdout)")
    
    args = parser.parse_args()
    
    # Create client
    client = CLIClient(args.api_url, verbose=args.verbose)
    
    # Dispatch command
    if args.command == "run":
        run_simulation(client, args)
    elif args.command == "status":
        check_status(client, args)
    elif args.command == "results":
        get_results(client, args)
    elif args.command == "list":
        list_simulations(client, args)
    elif args.command == "validate":
        validate_spec(client, args)
    elif args.command == "capabilities":
        show_capabilities(client, args)
    elif args.command == "interactive":
        interactive_mode(client, args)
    elif args.command == "convert":
        convert_prompt(client, args)


class CLIClient:
    """HTTP client for CLI"""
    
    def __init__(self, api_url: str, verbose: bool = False):
        self.api_url = api_url.rstrip("/")
        self.verbose = verbose
    
    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        """Make API request"""
        if not REQUESTS_AVAILABLE:
            print("Error: 'requests' package not installed. Run: pip install requests")
            sys.exit(1)
        
        url = f"{self.api_url}{endpoint}"
        
        if self.verbose:
            print(f"[DEBUG] {method.upper()} {url}")
        
        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            print(f"Error: Cannot connect to API at {self.api_url}")
            print("Make sure the server is running: python main.py")
            sys.exit(1)
        except requests.exceptions.HTTPError as e:
            error_data = {}
            try:
                error_data = e.response.json()
            except:
                pass
            
            print(f"Error: API returned {e.response.status_code}")
            if error_data:
                print(f"  {error_data.get('detail', error_data)}")
            sys.exit(1)
    
    def post(self, endpoint: str, data: dict) -> dict:
        return self._request("POST", endpoint, json=data)
    
    def get(self, endpoint: str, params: dict = None) -> dict:
        return self._request("GET", endpoint, params=params)


def run_simulation(client: CLIClient, args):
    """Submit a simulation"""
    data = {}
    
    if args.spec:
        # Load specification from file
        spec_path = Path(args.spec)
        if not spec_path.exists():
            print(f"Error: Specification file not found: {args.spec}")
            sys.exit(1)
        
        with open(spec_path) as f:
            data["specification"] = json.load(f)
        
        print(f"Loaded specification from: {args.spec}")
    elif args.prompt:
        data["prompt"] = args.prompt
        print(f"Prompt: {args.prompt}")
    else:
        print("Error: Either prompt or --spec is required")
        sys.exit(1)
    
    # Submit simulation
    print("\nSubmitting simulation...")
    result = client.post("/simulations", data)
    
    job_id = result["job_id"]
    print(f"Job submitted: {job_id}")
    print(f"Status: {result['status']}")
    
    if args.wait:
        print(f"\nWaiting for completion (timeout: {args.timeout}s)...")
        wait_for_completion(client, job_id, args.timeout)


def wait_for_completion(client: CLIClient, job_id: str, timeout: int):
    """Wait for simulation to complete"""
    start_time = time.time()
    last_status = None
    
    while True:
        result = client.get(f"/simulations/{job_id}")
        status = result["status"]
        progress = result.get("progress", 0)
        
        if status != last_status:
            print(f"  Status: {status} ({progress*100:.0f}%)")
            last_status = status
        
        if status == "completed":
            print("\n✓ Simulation completed!")
            # Show summary
            full_result = client.get(f"/simulations/{job_id}/result")
            print(f"\nSummary:\n{full_result.get('summary', 'No summary available')}")
            return
        
        if status == "failed":
            print(f"\n✗ Simulation failed: {result.get('message', 'Unknown error')}")
            sys.exit(1)
        
        if time.time() - start_time > timeout:
            print(f"\n⚠ Timeout after {timeout}s")
            sys.exit(1)
        
        time.sleep(5)


def check_status(client: CLIClient, args):
    """Check job status"""
    result = client.get(f"/simulations/{args.job_id}")
    
    print(f"Job ID:   {result['job_id']}")
    print(f"Status:   {result['status']}")
    print(f"Progress: {result.get('progress', 0)*100:.0f}%")
    
    if result.get("message"):
        print(f"Message:  {result['message']}")
    
    if result.get("time_elapsed_seconds"):
        print(f"Elapsed:  {result['time_elapsed_seconds']:.1f}s")


def get_results(client: CLIClient, args):
    """Get simulation results"""
    result = client.get(f"/simulations/{args.job_id}/result")
    
    if args.format == "json":
        print(json.dumps(result, indent=2))
    else:
        # Summary format
        print(f"Job ID: {result['job_id']}")
        print(f"Status: {result['status']}")
        print(f"Runtime: {result.get('runtime_seconds', 0):.1f}s")
        print()
        
        if result.get("summary"):
            print(result["summary"])
        
        # Save to file if output specified
        if args.output != ".":
            output_path = Path(args.output)
            output_path.mkdir(parents=True, exist_ok=True)
            
            result_file = output_path / f"{args.job_id}_results.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {result_file}")


def list_simulations(client: CLIClient, args):
    """List simulations"""
    params = {"limit": args.limit}
    if args.status:
        params["status"] = args.status
    
    result = client.get("/simulations", params)
    
    simulations = result.get("simulations", [])
    if not simulations:
        print("No simulations found")
        return
    
    print(f"{'Job ID':<12} {'Status':<15} {'Name':<30}")
    print("-" * 60)
    
    for sim in simulations:
        print(f"{sim['job_id']:<12} {sim['status']:<15} {sim.get('name', 'N/A'):<30}")


def validate_spec(client: CLIClient, args):
    """Validate a specification file"""
    spec_path = Path(args.spec_file)
    if not spec_path.exists():
        print(f"Error: File not found: {args.spec_file}")
        sys.exit(1)
    
    with open(spec_path) as f:
        spec = json.load(f)
    
    result = client.post("/validate", {"specification": spec})
    
    if result.get("valid"):
        print("✓ Specification is valid")
    else:
        print("✗ Specification has errors:")
        for error in result.get("errors", []):
            print(f"  - {error}")
    
    if result.get("warnings"):
        print("\nWarnings:")
        for warning in result["warnings"]:
            print(f"  - {warning}")


def show_capabilities(client: CLIClient, args):
    """Show system capabilities"""
    result = client.get("/capabilities")
    
    print("=== OpenFOAM Orchestration System Capabilities ===\n")
    
    print("Supported Geometries:")
    for geom in result.get("geometries", []):
        print(f"  • {geom}")
    
    print("\nSupported Solvers:")
    for solver in result.get("solvers", []):
        print(f"  • {solver}")
    
    print("\nTurbulence Models:")
    for model in result.get("turbulence_models", []):
        print(f"  • {model}")
    
    print("\nResource Limits:")
    limits = result.get("limits", {})
    for key, value in limits.items():
        print(f"  • {key}: {value}")


def convert_prompt(client: CLIClient, args):
    """Convert natural language to specification"""
    result = client.post("/convert", {"prompt": args.prompt})
    
    spec = result.get("specification", {})
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(spec, f, indent=2)
        print(f"Specification saved to: {args.output}")
    else:
        print(json.dumps(spec, indent=2))
    
    if result.get("confidence_score"):
        print(f"\nConfidence: {result['confidence_score']*100:.0f}%")
    
    if result.get("interpretation_notes"):
        print(f"Notes: {result['interpretation_notes']}")


def interactive_mode(client: CLIClient, args):
    """Interactive CLI mode"""
    print("OpenFOAM Orchestration System - Interactive Mode")
    print("Type 'help' for commands, 'quit' to exit\n")
    
    while True:
        try:
            user_input = input("foam> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        if user_input.lower() == "help":
            print("""
Commands:
  run <prompt>     - Submit simulation with natural language prompt
  status <job_id>  - Check job status
  results <job_id> - Get results
  list             - List recent simulations
  caps             - Show system capabilities
  quit             - Exit interactive mode
            """)
            continue
        
        parts = user_input.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        
        if cmd == "run" and arg:
            result = client.post("/simulations", {"prompt": arg})
            print(f"Job submitted: {result['job_id']}")
        elif cmd == "status" and arg:
            result = client.get(f"/simulations/{arg}")
            print(f"Status: {result['status']} ({result.get('progress', 0)*100:.0f}%)")
        elif cmd == "results" and arg:
            result = client.get(f"/simulations/{arg}/result")
            print(result.get("summary", "No results available"))
        elif cmd == "list":
            result = client.get("/simulations")
            for sim in result.get("simulations", [])[:5]:
                print(f"  {sim['job_id']}: {sim['status']}")
        elif cmd == "caps":
            result = client.get("/capabilities")
            print(f"Geometries: {len(result.get('geometries', []))}")
            print(f"Solvers: {result.get('solvers', [])}")
        else:
            print(f"Unknown command: {cmd}. Type 'help' for commands.")


if __name__ == "__main__":
    main()
