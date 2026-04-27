"""
Example using systematic instance generator

Demonstrates different instance sizes and difficulty levels
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from part_preparation.instance_generator import (
    generate_instance,
    InstanceParams,
    SMALL_INSTANCE,
    MEDIUM_INSTANCE,
    LARGE_INSTANCE,
)
from part_preparation.solvers.cpsat import CPSatEngineSolver
from part_preparation.solvers.greedy import GreedyEngineSolver


def solve_and_report(problem, time_limit=60):
    """Solve problem and report results"""
    print("\n" + "=" * 80)
    print("SOLVING")
    print("=" * 80)

    # Try greedy first
    print("\nGreedy heuristic:")
    greedy_solver = GreedyEngineSolver(problem)
    greedy_solver.init_model()
    greedy_results = greedy_solver.solve()

    if len(greedy_results) > 0:
        greedy_solution = greedy_results.get_best_solution()
        greedy_kpis = problem.evaluate(greedy_solution)
        greedy_feasible = problem.satisfy(greedy_solution)

        print(f"  Feasible:         {greedy_feasible}")
        print(f"  Baseline changes: {greedy_kpis['baseline_changes']:.0f}")
        print(f"  Arrival delays:   {greedy_kpis['arrival_delays']:.1f} days")
        print(f"  Travel cost:      €{greedy_kpis['travel_cost']:.2f}")
    else:
        greedy_solution = None
        print("  No solution found")

    # Solve with CP-SAT
    print(f"\nCP-SAT (time limit: {time_limit}s):")
    cpsat_solver = CPSatEngineSolver(problem)
    cpsat_solver.init_model()
    cpsat_results = cpsat_solver.solve(time_limit=time_limit)

    if len(cpsat_results) > 0:
        cpsat_solution = cpsat_results.get_best_solution()
        cpsat_kpis = problem.evaluate(cpsat_solution)
        cpsat_feasible = problem.satisfy(cpsat_solution)

        print(f"  Feasible:         {cpsat_feasible}")
        print(f"  Baseline changes: {cpsat_kpis['baseline_changes']:.0f}")
        print(f"  Arrival delays:   {cpsat_kpis['arrival_delays']:.1f} days")
        print(f"  Travel cost:      €{cpsat_kpis['travel_cost']:.2f}")

        if greedy_solution:
            print(f"\n  Improvement vs greedy:")
            print(f"    Baseline changes: {greedy_kpis['baseline_changes'] - cpsat_kpis['baseline_changes']:+.0f}")
            print(f"    Delays:           {greedy_kpis['arrival_delays'] - cpsat_kpis['arrival_delays']:+.1f} days")
            print(f"    Cost:             €{greedy_kpis['travel_cost'] - cpsat_kpis['travel_cost']:+.2f}")

        return cpsat_solution
    else:
        print("  No solution found")
        return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate and solve engine preparation instances")
    parser.add_argument(
        '--instance',
        choices=['small', 'medium', 'large', 'custom'],
        default='medium',
        help='Instance size to generate'
    )
    parser.add_argument('--time-limit', type=int, default=60, help='CP-SAT time limit (seconds)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Custom instance parameters
    parser.add_argument('--n-fals', type=int, help='Number of FALs')
    parser.add_argument('--n-msns-per-fal', type=int, help='MSNs per FAL')
    parser.add_argument('--n-engine-types', type=int, help='Number of engine types')
    parser.add_argument('--capacity-utilization', type=float, help='Target capacity utilization (0-1)')

    args = parser.parse_args()

    # Select instance parameters
    if args.instance == 'small':
        params = SMALL_INSTANCE
    elif args.instance == 'medium':
        params = MEDIUM_INSTANCE
    elif args.instance == 'large':
        params = LARGE_INSTANCE
    else:  # custom
        params = InstanceParams(
            n_fals=args.n_fals or 4,
            n_msns_per_fal=args.n_msns_per_fal or 5,
            n_engine_types=args.n_engine_types or 2,
            capacity_utilization=args.capacity_utilization or 0.95,
        )

    # Override seed if specified
    if args.seed != 42:
        params.seed = args.seed

    print("=" * 80)
    print(f"GENERATING {args.instance.upper()} INSTANCE")
    print("=" * 80)
    print(f"Parameters:")
    print(f"  FALs:                {params.n_fals}")
    print(f"  MSNs per FAL:        {params.n_msns_per_fal}")
    print(f"  Engine types:        {params.n_engine_types}")
    print(f"  Factories per type:  {params.n_factories_per_type}")
    print(f"  Capacity target:     {params.factory_capacity_utilization*100:.0f}%")
    print(f"  Seed:                {params.seed}")

    # Generate instance
    problem = generate_instance(params)

    # Solve
    solution = solve_and_report(problem, time_limit=args.time_limit)

    # Save solution for dashboard
    if solution:
        import pickle
        data_file = Path(__file__).parent / "solution_data.pkl"
        with open(data_file, 'wb') as f:
            pickle.dump({'problem': problem, 'solution': solution}, f)

        print("\n" + "=" * 80)
        print("Solution saved. To launch dashboard, run:")
        print(f"  cd {Path(__file__).parent.parent.parent}")
        print(f"  uv run streamlit run examples_do_usecases/part_preparation/dashboard_app.py")
        print("=" * 80)


if __name__ == "__main__":
    main()
