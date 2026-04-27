"""
Launch Streamlit dashboard for Engine Preparation solution

This script solves the problem and launches the Streamlit dashboard
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pickle
import subprocess
from part_preparation.problem import (
    EnginePreparationProblem,
    AssemblyLine,
    MSN,
    PreparationFactory,
    Engine,
    TravelInfo,
)
from part_preparation.solvers.cpsat import CPSatEngineSolver
from examples_do_usecases.part_preparation.example_complex import create_complex_example


def main():
    print("=" * 80)
    print("Solving Engine Preparation Problem...")
    print("=" * 80)

    # Create problem
    problem = create_complex_example()
    print(f"\nProblem: {len(problem.engines_list)} engines, {len(problem.factories_list)} factories")

    # Solve with CP-SAT
    cpsat_solver = CPSatEngineSolver(problem)
    cpsat_solver.init_model()
    cpsat_results = cpsat_solver.solve(time_limit=60)

    if len(cpsat_results) > 0:
        solution = cpsat_results.get_best_solution()
        kpis = problem.evaluate(solution)

        print(f"\n✓ Solution found:")
        print(f"  Feasible:         {problem.satisfy(solution)}")
        print(f"  Baseline changes: {kpis['baseline_changes']:.0f}")
        print(f"  Arrival delays:   {kpis['arrival_delays']:.1f} days")
        print(f"  Travel cost:      €{kpis['travel_cost']:.2f}")

        # Save to pickle for Streamlit to load
        data_file = Path(__file__).parent / "solution_data.pkl"
        with open(data_file, 'wb') as f:
            pickle.dump({'problem': problem, 'solution': solution}, f)

        print(f"\n✓ Data saved to: {data_file}")
        print("\n" + "=" * 80)
        print("Launching Streamlit dashboard...")
        print("=" * 80)

        # Launch Streamlit
        dashboard_app = Path(__file__).parent / "dashboard_app.py"
        subprocess.run(["streamlit", "run", str(dashboard_app)])

    else:
        print("\n✗ No solution found")


if __name__ == "__main__":
    main()
