"""
Run Gurobi MILP solver on engine preparation problem and launch visualization dashboards

Usage:
    uv run python examples_do_usecases/part_preparation/run_milp_and_visualize.py --instance medium --time-limit 60
    uv run python examples_do_usecases/part_preparation/run_milp_and_visualize.py --instance large --time-limit 300
"""
import sys
import argparse
import pickle
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    ObjectiveHandling,
    ModeOptim,
)
from part_preparation.instance_generator import (
    generate_instance,
    InstanceParams,
    FactoryDefinition,
    generate_instance_from_csv,
)
from part_preparation.solvers.milp_solver import GurobiEngineSolver


def main():
    parser = argparse.ArgumentParser(description="Run Gurobi MILP solver and visualize results")
    parser.add_argument(
        '--instance',
        choices=['medium', 'large'],
        default='medium',
        help='Instance size to solve'
    )
    parser.add_argument(
        '--time-limit',
        type=float,
        default=300.0,
        help='Time limit in seconds for Gurobi solver'
    )
    parser.add_argument(
        '--use-csv',
        action='store_true',
        default=False,
        help='Use CSV instance file or synthetic generator'
    )
    args = parser.parse_args()

    print("=" * 80)
    print(f"GENERATING {args.instance.upper()} INSTANCE")
    print("=" * 80)

    # 1. Load or Generate Problem Instance
    if args.use_csv:
        csv_file_path = "examples_do_usecases/part_preparation/contour-export-06-07-2026.csv"
        start_filter = datetime(2026, 9, 15, 0, 0)
        end_filter = datetime(2027, 10, 1, 23, 59)

        problem = generate_instance_from_csv(
            csv_file_path=csv_file_path,
            start_date=start_filter,
            end_date=end_filter
        )
    else:
        manual_factories = [
            FactoryDefinition(engine_type="EngineType_1", home_fal_index=0, capacity=3, preparation_duration=11, location="Derby"),
            FactoryDefinition(engine_type="EngineType_1", home_fal_index=1, capacity=2, preparation_duration=10, location="Toulouse"),
            FactoryDefinition(engine_type="EngineType_1", home_fal_index=2, capacity=2, preparation_duration=13, location="Virginia"),
            FactoryDefinition(engine_type="EngineType_1", home_fal_index=None, capacity=1, preparation_duration=14, location="Singapore"),
            FactoryDefinition(engine_type="EngineType_2", home_fal_index=0, capacity=4, preparation_duration=15, location="Durham"),
            FactoryDefinition(engine_type="EngineType_2", home_fal_index=1, capacity=3, preparation_duration=15, location="Bordeaux"),
            FactoryDefinition(engine_type="EngineType_2", home_fal_index=2, capacity=1, preparation_duration=15, location="Cincinnati"),
            FactoryDefinition(engine_type="EngineType_2", home_fal_index=3, capacity=2, preparation_duration=12, location="Shanghai"),
        ]

        n_msns = 80 if args.instance == 'large' else 40
        time_horizon = 1000 if args.instance == 'large' else 300

        params = InstanceParams(
            n_fals=4,
            n_msns_per_fal=n_msns,
            n_engine_types=2,
            time_horizon=time_horizon,
            fal_takt_paces=[2, 2, 5, 5],
            factory_definitions=manual_factories,
            engine_to_msn_ratio={"EngineType_1": 1.1, "EngineType_2": 0.95},
            supplier_regularity={"EngineType_1": 1.0, "EngineType_2": 0.8},
            supplier_production_rate={"EngineType_1": 1.0, "EngineType_2": 0.9},
            flexible_delivery_prob=0
        )
        problem = generate_instance(params)

    print(f"\nProblem size:")
    print(f"  - {len(problem.assembly_lines_list)} assembly lines")
    print(f"  - {len(problem.msns_list)} MSNs")
    print(f"  - {len(problem.engines_list)} engines")
    print(f"  - {len(problem.factories_list)} factories")
    print(f"  - Time horizon: {problem.time_horizon} days")

    # 2. Configure Objective Function
    params_objective = ParamsObjectiveFunction(
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=[
            "arrival_delays",
            "travel_cost",
            "storage_cost",
            "unallocated_msns"
        ],
        weights=[1000.0, 1.0, 1.0, 100000.0],
        sense_function=ModeOptim.MINIMIZATION
    )

    print("\nObjective weights:")
    for obj, weight in zip(params_objective.objectives, params_objective.weights):
        print(f"  - {obj}: {weight}")

    # 3. Solve with Gurobi
    print("\n" + "=" * 80)
    print(f"SOLVING WITH GUROBI MILP (time limit: {args.time_limit}s)")
    print("=" * 80)

    solver = GurobiEngineSolver(
        problem=problem,
        params_objective_function=params_objective
    )

    try:
        solution = solver.solve(time_limit=args.time_limit)
    except Exception as e:
        print(f"\n❌ Execution Error during optimization: {e}")
        return

    print("\n" + "=" * 80)
    print("SOLVER FINISHED")
    print("=" * 80)

    if solution is None:
        print("\n❌ No solution found!")
        return

    print("\n✅ SOLUTION FOUND")

    # 4. Evaluate and Print Solution Metrics
    kpis = problem.evaluate(solution)
    feasible = problem.satisfy(solution)

    print(f"\nFeasible: {'✅ Yes' if feasible else '❌ No'}")
    print(f"\nObjective values:")
    for obj_name, obj_value in kpis.items():
        print(f"  - {obj_name}: {obj_value:.2f}")

    allocated_engines = sum(
        1 for e in problem.engines_list
        if e in solution.engine_to_msn and solution.engine_to_msn[e] is not None
    )
    unallocated_engines = len(problem.engines_list) - allocated_engines

    print(f"\nAllocation Summary:")
    print(f"  - Allocated engines:   {allocated_engines}/{len(problem.engines_list)}")
    print(f"  - Unallocated engines: {unallocated_engines}")
    print(f"  - Unallocated MSNs:    {kpis.get('unallocated_msns', 0):.0f}/{len(problem.msns_list)}")

    # Format solution history for the KPI dashboard tab
    solution_history = [{
        'index': 0,
        'fitness': sum(w * kpis[o] for o, w in zip(params_objective.objectives, params_objective.weights) if o in kpis),
        'kpis': kpis
    }]

    # 5. Save Data for Streamlit Dashboard
    data_dir = Path(__file__).parent / "solution_data"
    data_dir.mkdir(exist_ok=True)

    data_file = data_dir / f"{args.instance}_solution.pkl"
    with open(data_file, 'wb') as f:
        pickle.dump({
            'problem': problem,
            'solution': solution,
            'kpis': kpis,
            'feasible': feasible,
            'solution_history': solution_history,
        }, f)

    print(f"\n✓ Solution saved to: {data_file}")

    # 6. Create Launcher and Launch Dashboard
    print("\n" + "=" * 80)
    print("LAUNCHING DASHBOARD")
    print("=" * 80)

    dashboard_launcher = Path(__file__).parent / "launch_unified_dashboard.py"

    launcher_code = f"""import streamlit as st
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
    print(f"\nRun: uv run streamlit run {dashboard_launcher}")

    # Auto-launch Streamlit
    subprocess.run(["uv", "run", "streamlit", "run", str(dashboard_launcher)])


if __name__ == "__main__":
    main()
