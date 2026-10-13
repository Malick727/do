import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

import argparse
import pickle
import subprocess
import time

import didppy as dp

from part_preparation import instances
from part_preparation.paths import solution_file, ensure_dirs
from part_preparation.problem import aggregate_score
from part_preparation.solvers.bilevel.dp.adapter import solve_with_didp

SOLVERS = {
    "cabs": dp.CABS,
    "lnbs": dp.LNBS,
    "acps": dp.ACPS,
    "apps": dp.APPS,
    "cbfs": dp.CBFS,
    "dfbb": dp.DFBB,
    "caasdy": dp.CAASDy,
}


def build_problem(args):
    if args.msns_per_fal is not None:
        from part_preparation.instance_generator import generate_instance
        return generate_instance(instances.sized_params(args.msns_per_fal))
    return instances.build(args.instance, replicate_id=args.replicate)


def print_problem(problem):
    print(f"\nProblem size:")
    print(f"  - {len(problem.assembly_lines_list)} assembly lines")
    print(f"  - {len(problem.msns_list)} MSNs")
    print(f"  - {len(problem.engines_list)} engines")
    print(f"  - {len(problem.factories_list)} factories")
    print(f"  - Time horizon: {problem.time_horizon} days")
    print(f"  - Structurally uncoverable MSNs: {problem.uncoverable_msns}")


def print_results(problem, solution, results, elapsed):
    print("\nPer engine type:")
    for engine_type, result in results.items():
        print(f"  - {engine_type}: cost {result.cost}  bound {result.best_bound}  "
              f"optimal {result.is_optimal}  expanded {result.expanded}  generated {result.generated}")

    if solution is None:
        print("\nNo solution found: increase --time-limit or try another --solver")
        return None, None, None

    kpis = problem.evaluate(solution)
    feasible = problem.satisfy(solution)
    score = aggregate_score(kpis)
    didp_cost = sum(result.cost for result in results.values())

    print("\n" + "=" * 80)
    print("SOLUTION")
    print("=" * 80)
    print(f"\nFeasible: {'yes' if feasible else 'NO'}")
    print(f"\n  - aggregate score  : {score:,.2f}")
    print(f"  - DIDP cost        : {didp_cost:,.2f}   (gap {abs(score - didp_cost):,.4f})")
    print(f"  - solve time       : {elapsed:.2f}s")

    print(f"\nObjective values:")
    for name, value in kpis.items():
        print(f"  - {name}: {value:.2f}")

    allocated = sum(1 for e in problem.engines_list if solution.engine_to_msn.get(e) is not None)
    print(f"\nAllocation:")
    print(f"  - Allocated engines: {allocated}/{len(problem.engines_list)}")
    print(f"  - Unallocated MSNs: {kpis['unallocated_msns']:.0f}/{len(problem.msns_list)} "
          f"({problem.uncoverable_msns} structurally uncoverable)")
    return kpis, feasible, score


def save_solution(args, problem, solution, kpis, feasible, score):
    ensure_dirs()
    label = args.instance if args.msns_per_fal is None else f"msns{args.msns_per_fal * 4}"
    if args.replicate is not None:
        label = f"{label}_r{args.replicate}"
    label = f"{label}_didp_{args.formulation}_{args.solver}"
    data_file = solution_file(label)
    with open(data_file, 'wb') as f:
        pickle.dump({
            'problem': problem,
            'solution': solution,
            'kpis': kpis,
            'feasible': feasible,
            'solution_history': [{'index': 0, 'fitness': score, 'kpis': kpis}],
            'solver': f"DIDP {args.formulation} ({args.solver.upper()}, {args.time_limit:g}s)",
        }, f)
    print(f"\nSolution saved to: {data_file}")
    return data_file


def launch_dashboard(data_file):
    print("\n" + "=" * 80)
    print("LAUNCHING DASHBOARD")
    print("=" * 80)

    dashboard_launcher = Path(__file__).parent / "launch_didp_dashboard.py"
    with open(dashboard_launcher, 'w') as f:
        f.write(f"""
import streamlit as st
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

data_file = Path(r"{data_file}")
with open(data_file, 'rb') as f:
    data = pickle.load(f)

st.session_state.problem = data['problem']
st.session_state.solution = data['solution']
st.session_state.solution_history = data.get('solution_history', [])
st.session_state.solver = data.get('solver', 'solveur inconnu')

from part_preparation.unified_dashboard import main
main()
""")
    print(f"Dashboard launcher created: {dashboard_launcher}")
    subprocess.run(["streamlit", "run", str(dashboard_launcher)])


def main():
    parser = argparse.ArgumentParser(description="Solve the part preparation problem with DIDP and visualize results")
    parser.add_argument('--instance', choices=sorted(instances.INSTANCES), default=instances.DEFAULT_INSTANCE)
    parser.add_argument('--replicate', type=int, default=None,
                        help='Replicate id of the named instance')
    parser.add_argument('--msns-per-fal', type=int, default=None,
                        help='Override the named instance size. Total MSNs = 4 x this value')
    parser.add_argument('--solver', choices=sorted(SOLVERS), default="cabs")
    parser.add_argument('--time-limit', type=float, default=60.0,
                        help='Total seconds, shared equally between engine types')
    parser.add_argument('--threads', type=int, default=None,
                        help='Search threads (CABS, LNBS, ACPS, APPS, CBFS, CAASDy)')
    parser.add_argument('--formulation', choices=['multi', 'single'], default='multi',
                        help='multi: engine, then MSN, then factory (3 steps); single: one step per (engine, factory, MSN, station)')
    parser.add_argument('--verbose', action='store_true', help='Print the DIDP search log')
    parser.add_argument('--no-dashboard', action='store_true')
    args = parser.parse_args()

    print("=" * 80)
    print(f"GENERATING {args.instance.upper()} INSTANCE")
    print("=" * 80)
    problem = build_problem(args)
    print_problem(problem)

    print("\n" + "=" * 80)
    print(f"SOLVING WITH DIDP ({args.formulation} formulation, {args.solver.upper()}, {args.time_limit:g}s)")
    print("=" * 80)

    solver_kwargs = {} if args.threads is None else {"threads": args.threads}
    started = time.perf_counter()
    solution, results = solve_with_didp(
        problem,
        solver_class=SOLVERS[args.solver],
        time_limit=args.time_limit,
        quiet=not args.verbose,
        multi_phase=args.formulation == 'multi',
        **solver_kwargs,
    )
    elapsed = time.perf_counter() - started

    kpis, feasible, score = print_results(problem, solution, results, elapsed)
    if solution is None:
        return

    data_file = save_solution(args, problem, solution, kpis, feasible, score)
    if not args.no_dashboard:
        launch_dashboard(data_file)


if __name__ == "__main__":
    main()
