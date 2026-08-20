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

from part_preparation import instances
from part_preparation.paths import solution_file, ensure_dirs
from part_preparation.problem import default_params_objective_function
from part_preparation.solvers.cpsat import CPSatEngineSolver


def main():
    parser = argparse.ArgumentParser(description="Run CP-SAT solver and visualize results")
    parser.add_argument(
        '--instance',
        choices=sorted(instances.INSTANCES),
        default=instances.DEFAULT_INSTANCE,
        help='Instance size to solve (xlarge = 1000 MSN)'
    )
    parser.add_argument(
        '--time-limit',
        type=int,
        default=20,
        help='Time limit in seconds for CP-SAT solver'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=12,
        help='CP-SAT workers'
    )
    parser.add_argument(
        '--replicate',
        type=int,
        default=None,
        help='Replicate id of the named instance, so this run targets the SAME problem as '
             'run_rl_and_visualize.py --replicate <id>. Without it both scripts solve the '
             'base instance, and any comparison between their scores is meaningless'
    )
    parser.add_argument(
        '--msns-per-fal',
        type=int,
        default=None,
        help='Override the named instance size. Total MSNs = 4 x this value. '
             'The time horizon is derived, never given (see instances.sized_params)'
    )
    parser.add_argument(
        '--no-dashboard',
        action='store_true',
        help='Solve and save only, without launching streamlit'
    )
    args = parser.parse_args()
    print("=" * 80)
    print(f"GENERATING {args.instance.upper()} INSTANCE")
    print("=" * 80)

    # L'instance vient du registre et n'est plus redéclarée ici. Les définitions recopiées dans
    # ce script forçaient à corriger le nombre de MSN sans corriger l'horizon avec : à
    # n_msns_per_fal=1000 sur 4 FAL, on demandait 4000 MSN à caser dans 300 jours, alors que la
    # FAL la plus lente a un takt de 5. C'est exactement ce que part_preparation.instances est
    # là pour empêcher -- et c'est aussi ce qui garantit que ce script et la comparaison
    # d'approches (solvers.compare) tournent sur LA MÊME instance.
    if args.msns_per_fal is None:
        problem = instances.build(args.instance, replicate_id=args.replicate)
    else:
        from part_preparation.instance_generator import generate_instance

        problem = generate_instance(instances.sized_params(args.msns_per_fal))

    # Objectif lu depuis problem.py, source unique partagée avec la fitness du GA
    # (bilevel.lower.score_solution) : les deux solveurs minimisent la même fonction et leurs
    # valeurs sont directement comparables.
    params_objective = default_params_objective_function()

    print(f"\nProblem size:")
    print(f"  - {len(problem.assembly_lines_list)} assembly lines")
    print(f"  - {len(problem.msns_list)} MSNs")
    print(f"  - {len(problem.engines_list)} engines")
    print(f"  - {len(problem.factories_list)} factories")
    print(f"  - Time horizon: {problem.time_horizon} days")
    print(f"  - Structurally uncoverable MSNs: {problem.uncoverable_msns}")

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
    import time
    from discrete_optimization.generic_tools.cp_tools import ParametersCp

    # La construction du modèle est chronométrée à part : sur les grandes instances elle n'est
    # plus négligeable devant la recherche (le nombre de booléens both_assigned croît en
    # moteurs x usines x MSN compatibles), et un time_limit qui semble ignoré est presque
    # toujours du temps passé ici, pas dans CP-SAT.
    build_started = time.perf_counter()
    solver.init_model()
    print(f"Model built in {time.perf_counter() - build_started:.1f}s "
          f"({len(solver.cp_model.proto.variables):,} variables, "
          f"{len(solver.cp_model.proto.constraints):,} constraints)")

    p = ParametersCp.default_cpsat()
    p.nb_process = args.workers
    print("\nSolving...")
    result_storage = solver.solve(
        parameters_cp=p,
        time_limit=args.time_limit,
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

    # Borne et gap : c'est le seul chiffre qui dise si la solution est BONNE, et pas seulement
    # la meilleure trouvée dans le temps imparti. Sur les grandes instances il est fréquent que
    # CP-SAT rende une solution en quelques secondes et n'ait toujours prouvé aucune borne utile
    # à la fin -- un gap proche de 100% signale que le modèle complet ne passe pas à l'échelle,
    # ce qui est précisément l'argument en faveur de la décomposition.
    from part_preparation.problem import aggregate_score

    score = aggregate_score(kpis)
    bound = solver.solver.BestObjectiveBound()
    print(f"\nStatus: {solver.status_solver}")
    print(f"  - aggregate score : {score:,.2f}")
    print(f"  - best bound      : {bound:,.2f}")
    if score:
        print(f"  - optimality gap  : {(score - bound) / abs(score):.2%}")

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
    ensure_dirs()

    # Une taille ad hoc n'écrase pas le pickle de l'instance nommée dont elle emprunte le nom :
    # les dashboards liraient sinon un problème qui n'est pas celui qu'ils croient afficher.
    # Le réplicat entre dans le nom : deux réplicats d'une même instance nommée sont deux
    # problèmes différents, et sans ça le second écraserait le pickle du premier -- les
    # dashboards afficheraient alors une solution qui n'est pas celle qu'ils annoncent.
    label = (args.instance if args.msns_per_fal is None
             else f"msns{args.msns_per_fal * 4}")
    if args.replicate is not None:
        label = f"{label}_r{args.replicate}"
    data_file = solution_file(label)
    with open(data_file, 'wb') as f:
        pickle.dump({
            'problem': problem,
            'solution': solution,
            'kpis': kpis,
            'feasible': feasible,
            'solution_history': solution_history,  # Add solution history for KPI evolution
            # Lu par le dashboard : plusieurs solveurs ecrivent pour la meme instance.
            'solver': f"CP-SAT ({args.time_limit}s, {solver.status_solver})",
        }, f)

    print(f"\n✓ Solution saved to: {data_file}")

    if args.no_dashboard:
        return

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
data_file = Path(r"{data_file}")
with open(data_file, 'rb') as f:
    data = pickle.load(f)

st.session_state.problem = data['problem']
st.session_state.solution = data['solution']
st.session_state.solution_history = data.get('solution_history', [])
st.session_state.solver = data.get('solver', 'solveur inconnu')

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
