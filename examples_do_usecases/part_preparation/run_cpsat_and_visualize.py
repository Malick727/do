"""
Run CP-SAT solver on engine preparation problem and launch visualization dashboards

Usage:
    uv run python examples_do_usecases/part_preparation/run_cpsat_and_visualize.py --instance medium --time-limit 60
    uv run python examples_do_usecases/part_preparation/run_cpsat_and_visualize.py --instance large --time-limit 300
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import pickle
import subprocess
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction, ObjectiveHandling, ModeOptim

from part_preparation.instance_generator import generate_instance, InstanceParams, FactoryDefinition
from part_preparation.solvers.cpsat import CPSatEngineSolver


def main():
    parser = argparse.ArgumentParser(description="Run CP-SAT solver and visualize results")
    parser.add_argument(
        '--instance',
        choices=['medium', 'large'],
        default='medium',
        help='Instance size to solve'
    )
    parser.add_argument(
        '--time-limit',
        type=int,
        default=20,
        help='Time limit in seconds for CP-SAT solver'
    )
    args = parser.parse_args()
    print("=" * 80)
    print(f"GENERATING {args.instance.upper()} INSTANCE")
    print("=" * 80)

    manual_factories = [
        # EngineType_1 factories
        FactoryDefinition(engine_type="EngineType_1",
                          home_fal_index=0, capacity=3,
                          preparation_duration=11, location="Derby"),
        FactoryDefinition(engine_type="EngineType_1",
                          home_fal_index=1,
                          capacity=2,
                          preparation_duration=10,
                          location="Toulouse"),  # BOTTLENECK!
        FactoryDefinition(engine_type="EngineType_1",
                          home_fal_index=2,
                          capacity=2,
                          preparation_duration=13,
                          location="Virginia"),
        FactoryDefinition(engine_type="EngineType_1",
                          home_fal_index=None,
                          capacity=1,
                          preparation_duration=14,
                          location="Singapore"),  # Distant from all

        # EngineType_2 factories
        FactoryDefinition(engine_type="EngineType_2", home_fal_index=0, capacity=4,
                          preparation_duration=15, location="Durham"),
        FactoryDefinition(engine_type="EngineType_2", home_fal_index=1, capacity=3,
                          preparation_duration=15, location="Bordeaux"),
        FactoryDefinition(engine_type="EngineType_2", home_fal_index=2, capacity=1,
                          preparation_duration=15, location="Cincinnati"),  # BOTTLENECK!
        FactoryDefinition(engine_type="EngineType_2", home_fal_index=3, capacity=2,
                          preparation_duration=12, location="Shanghai"),
    ]

    params = InstanceParams(
        n_fals=4,
        n_msns_per_fal=60,
        n_engine_types=2,
        time_horizon=300,
        fal_takt_paces=[2, 2, 5, 5],
        factory_definitions=manual_factories,  # Use manual factory definitions
        engine_to_msn_ratio={"EngineType_1": 1.1,
                             "EngineType_2": 0.95},
        supplier_regularity={"EngineType_1": 1.0, "EngineType_2": 0.8},
        supplier_production_rate={"EngineType_1": 1.0, "EngineType_2": 0.9},
        flexible_delivery_prob=0
    )
    problem = generate_instance(params)
    params_objective = ParamsObjectiveFunction(
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=[  # "baseline_changes",
            "arrival_delays",
            "travel_cost",
            "storage_cost",
            "unallocated_msns"],
        weights=[1.0, 1.0, 0., 1000000.0],  # High penalty for delays and unallocated MSNs
        sense_function=ModeOptim.MINIMIZATION
    )
    print("\nObjective weights:")
    for obj, weight in zip(params_objective.objectives, params_objective.weights):
        print(f"  - {obj}: {weight}")

    print("\n" + "=" * 80)
    print(f"SOLVING WITH CP-SAT (time limit: {args.time_limit}s)")
    print("=" * 80)

    # Create solver
    solver = CPSatEngineSolver(
        problem=problem,
        params_objective_function=params_objective
    )

    # Solve with time limit
    solver.init_model()
    from discrete_optimization.generic_tools.cp_tools import ParametersCp
    p = ParametersCp.default_cpsat()
    p.nb_process = 12
    print("\nSolving...")
    result_storage = solver.solve(
        parameters_cp=p,
        time_limit=300,
        ortools_cpsat_solver_kwargs={
            'log_search_progress': True,
        }
    )

    print("\n" + "=" * 80)
    print("SOLVER FINISHED")
    print("=" * 80)

    # Get best solution
    solution = result_storage.get_best_solution()

    if solution is None:
        print("\n❌ No solution found!")
        print("Possible reasons:")
        print("  - Problem is infeasible (constraints too tight)")
        print("  - Time limit too short")
        print("  - Model has errors")
        return

    print("\n✅ SOLUTION FOUND")

    # Evaluate solution
    kpis = problem.evaluate(solution)
    feasible = problem.satisfy(solution)

    print(f"\nFeasible: {'✅ Yes' if feasible else '❌ No'}")
    print(f"\nObjective values:")
    for obj_name, obj_value in kpis.items():
        print(f"  - {obj_name}: {obj_value:.2f}")

    # Count allocated engines and MSNs
    allocated_engines = sum(1 for e in problem.engines_list if e in solution.engine_to_msn and solution.engine_to_msn[e] is not None)
    unallocated_engines = len(problem.engines_list) - allocated_engines

    print(f"\nAllocation:")
    print(f"  - Allocated engines: {allocated_engines}/{len(problem.engines_list)}")
    print(f"  - Unallocated engines: {unallocated_engines}")
    print(f"  - Unallocated MSNs: {kpis['unallocated_msns']:.0f}/{len(problem.msns_list)}")

    # Extract all intermediate solutions for KPI evolution tracking
    print(f"\n📊 Extracting solution history...")
    all_solutions = result_storage.list_solution_fits
    print(f"   Found {len(all_solutions)} intermediate solutions")

    # Compute KPIs for all solutions
    solution_history = []
    for idx, (sol, fit) in enumerate(all_solutions):
        sol_kpis = problem.evaluate(sol)
        solution_history.append({
            'index': idx,
            'fitness': fit,
            'kpis': sol_kpis
        })
    print(f"   ✓ Computed KPIs for all {len(solution_history)} solutions")

    # Save problem and solution for dashboard
    data_dir = Path(__file__).parent / "solution_data"
    data_dir.mkdir(exist_ok=True)

    data_file = data_dir / f"{args.instance}_solution.pkl"
    with open(data_file, 'wb') as f:
        pickle.dump({
            'problem': problem,
            'solution': solution,
            'kpis': kpis,
            'feasible': feasible,
            'solution_history': solution_history,  # Add solution history for KPI evolution
        }, f)

    print(f"\n✓ Solution saved to: {data_file}")

    # Launch dashboard(s)
    print("\n" + "=" * 80)
    print("LAUNCHING DASHBOARD")
    print("=" * 80)

    # Launch unified dashboard
    print("\nLaunching unified dashboard (JIT vs Optimized + Gantt Chart)...")
    dashboard_launcher = Path(__file__).parent / "launch_unified_dashboard.py"

    launcher_code = f"""
import streamlit as st
import pickle
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load problem and solution
data_file = Path(__file__).parent / "solution_data" / "{args.instance}_solution.pkl"
with open(data_file, 'rb') as f:
    data = pickle.load(f)

st.session_state.problem = data['problem']
st.session_state.solution = data['solution']
st.session_state.solution_history = data.get('solution_history', [])

# Import and run unified dashboard
from part_preparation.unified_dashboard import main
main()
"""

    with open(dashboard_launcher, 'w') as f:
        f.write(launcher_code)

    print(f"Dashboard launcher created: {dashboard_launcher}")
    print(f"\nRun: streamlit run {dashboard_launcher}")

    # Auto-launch
    subprocess.run(["streamlit", "run", str(dashboard_launcher)])


if __name__ == "__main__":
    main()
