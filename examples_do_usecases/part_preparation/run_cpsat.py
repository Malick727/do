"""
Complex realistic example for Engine Preparation and Allocation Problem

Setup:
- 4 Final Assembly Lines (FALs): Hamburg, Toulouse, Mobile (USA), Tianjin (China)
- 2 Engine types: TrentXWB, GE90
- 8 Preparation factories (4 per engine type, each geographically close to 1 FAL)
- 20 MSNs (5 per FAL, mixed engine types)
- 20 Engines (10 per type)
- Baseline allocations with intentional disturbances to make problem interesting
"""
import random
from part_preparation.problem import (
    EnginePreparationProblem,
    EnginePreparationSolution,
    AssemblyLine,
    MSN,
    PreparationFactory,
    Engine,
    TravelInfo,
)
from part_preparation.solvers.greedy import GreedyEngineSolver
from part_preparation.solvers.cpsat import CPSatEngineSolver
from part_preparation.example_complex import create_complex_example
from part_preparation.example import create_simple_example



def main():
    """Run complex example with both Greedy and CP-SAT solvers"""
    print("=" * 100)
    print("Engine Preparation and Allocation - Complex Realistic Example")
    print("=" * 100)

    # Create problem
    problem: EnginePreparationProblem = create_complex_example()

    print(f"\nProblem created:")
    print(f"  - {len(problem.assembly_lines_list)} assembly lines (FALs)")
    print(f"  - {len(problem.msns_list)} MSNs (aircraft) scheduled")
    print(f"  - {len(problem.factories_list)} preparation factories")
    print(f"  - {len(problem.engines_list)} engines to allocate")
    print(f"  - 2 engine types: TrentXWB, GE90")
    print(f"  - Time horizon: {problem.time_horizon} days")

    # Show distribution
    print("\nMSN distribution by FAL:")
    for fal in problem.assembly_lines_list:
        msns_here = [m for m in problem.msns_list if m.assembly_line_id == fal.id]
        trent_count = sum(1 for m in msns_here if m.engine_type == "TrentXWB")
        ge_count = sum(1 for m in msns_here if m.engine_type == "GE90")
        print(f"  {fal.id:<15} - {len(msns_here)} MSNs (TrentXWB: {trent_count}, GE90: {ge_count})")

    print("\nFactory distribution:")
    for factory in problem.factories_list:
        types = ", ".join(factory.compatible_engine_types)
        print(f"  {factory.id:<15} ({factory.location:<12}) - {types:<10} - capacity: {factory.capacity}/week")

    # ==========================================
    # Solve with Greedy Heuristic
    # ==========================================
    print("\n" + "=" * 100)
    print("GREEDY HEURISTIC SOLVER")
    print("=" * 100)

    greedy_solver = GreedyEngineSolver(problem)
    greedy_solver.init_model()
    greedy_results = greedy_solver.solve()

    if len(greedy_results) > 0:
        greedy_solution = greedy_results.get_best_solution()
        greedy_kpis = problem.evaluate(greedy_solution)
        greedy_feasible = problem.satisfy(greedy_solution)

        print(f"\n✓ Greedy solution found")
        print(f"  Feasible:          {greedy_feasible}")
        print(f"  Baseline changes:  {greedy_kpis['baseline_changes']:.0f}")
        print(f"  Arrival delays:    {greedy_kpis['arrival_delays']:.1f} days")
        print(f"  Travel cost:       €{greedy_kpis['travel_cost']:.2f}")
    else:
        print("\n✗ No greedy solution found")
        greedy_solution = None

    # ==========================================
    # Solve with CP-SAT
    # ==========================================
    print("\n" + "=" * 100)
    print("CP-SAT SOLVER (OR-Tools)")
    print("=" * 100)

    cpsat_solver = CPSatEngineSolver(problem)
    cpsat_solver.init_model()
    cpsat_results = cpsat_solver.solve(time_limit=60,
                                       ortools_cpsat_solver_kwargs=dict(log_search_progress=True))

    if len(cpsat_results) > 0:
        cpsat_solution: EnginePreparationSolution = cpsat_results.get_best_solution()
        cpsat_kpis = problem.evaluate(cpsat_solution)
        cpsat_feasible = problem.satisfy(cpsat_solution)

        print(f"\n✓ CP-SAT solution found")
        print(f"  Feasible:          {cpsat_feasible}")
        print(f"  Baseline changes:  {cpsat_kpis['baseline_changes']:.0f}")
        print(f"  Arrival delays:    {cpsat_kpis['arrival_delays']:.1f} days")
        print(f"  Travel cost:       €{cpsat_kpis['travel_cost']:.2f}")

        if greedy_solution is not None:
            print(f"\n  Improvement over greedy:")
            print(f"    Baseline changes:  {greedy_kpis['baseline_changes'] - cpsat_kpis['baseline_changes']:.0f} fewer changes")
            print(f"    Arrival delays:    {greedy_kpis['arrival_delays'] - cpsat_kpis['arrival_delays']:.1f} days reduced")
            print(f"    Travel cost:       €{greedy_kpis['travel_cost'] - cpsat_kpis['travel_cost']:.2f} saved")

        # Show some allocation details
        print(f"\n" + "-" * 100)
        print("Sample allocations (first 5 engines):")
        print(f"{'Engine':<12} {'→ MSN':<12} {'→ Factory':<18} {'Prep':<15} {'Arrival':<12} {'Delay':<10} {'Baseline':<15}")
        print("-" * 100)

        for i, engine in enumerate(problem.engines_list[:5]):
            if engine in cpsat_solution.engine_to_msn:
                msn = cpsat_solution.engine_to_msn[engine]
                factory = cpsat_solution.engine_to_factory[engine]
                prep_start = cpsat_solution.prep_start_times[engine]
                prep_end = cpsat_solution.prep_end_times[engine]
                arrival = cpsat_solution.fal_arrival_times[engine]

                delay = max(0, arrival - msn.engine_installation_date)
                delay_str = f"{delay:.0f} days" if delay > 0 else "On time"

                baseline_str = "✓" if (engine.baseline_msn_id == msn.id and
                                       engine.baseline_factory_id == factory.id) else "✗ Changed"

                print(f"{engine.id:<12} {msn.id:<12} {factory.id:<18} {prep_start}-{prep_end:<10} {arrival:<12} {delay_str:<10} {baseline_str:<15}")

    else:
        print("\n✗ No CP-SAT solution found")


if __name__ == "__main__":
    main()
