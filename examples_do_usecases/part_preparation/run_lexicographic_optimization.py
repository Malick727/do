"""
Lexicographic optimization for Engine Preparation Problem

Optimizes objectives in priority order:
1. unallocated_msns (minimize MSNs without engines) - HIGHEST PRIORITY
2. arrival_delays (minimize delays to installation dates)
3. storage_cost (minimize storage costs) - LOWEST PRIORITY

Usage:
    uv run python examples_do_usecases/part_preparation/run_lexicographic_optimization.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import logging
from discrete_optimization.generic_tools.lexico_tools import LexicoSolver
from discrete_optimization.generic_tools.cp_tools import ParametersCp

from part_preparation.instance_generator import generate_instance, InstanceParams, FactoryDefinition
from part_preparation.solvers.cpsat import CPSatEngineSolver

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    print("=" * 80)
    print("LEXICOGRAPHIC OPTIMIZATION - ENGINE PREPARATION PROBLEM")
    print("=" * 80)
    print("\nObjective priority order:")
    print("  1. unallocated_msns  (HIGHEST PRIORITY - must minimize unmet demand)")
    print("  2. arrival_delays    (MEDIUM PRIORITY  - minimize late deliveries)")
    print("  3. storage_cost      (LOWEST PRIORITY  - minimize holding costs)")
    print("=" * 80)

    # =====================================================================
    # Generate Instance with Intentional Shortage (to test unallocated_msns)
    # =====================================================================

    manual_factories = [
        # EngineType_1 factories (deliberately under-capacity)
        FactoryDefinition(
            engine_type="EngineType_1",
            home_fal_index=0,
            capacity=2,  # Limited capacity
            preparation_duration=10,
            location="Derby"
        ),
        FactoryDefinition(
            engine_type="EngineType_1",
            home_fal_index=1,
            capacity=1,  # BOTTLENECK!
            preparation_duration=12,
            location="Toulouse"
        ),

        # EngineType_2 factories
        FactoryDefinition(
            engine_type="EngineType_2",
            home_fal_index=0,
            capacity=3,
            preparation_duration=12,
            location="Durham"
        ),
        FactoryDefinition(
            engine_type="EngineType_2",
            home_fal_index=1,
            capacity=2,
            preparation_duration=14,
            location="Bordeaux"
        ),
    ]

    params = InstanceParams(
        n_fals=2,
        n_msns_per_fal=15,  # 30 MSNs total
        n_engine_types=2,
        time_horizon=100,
        fal_takt_paces=[3, 4],
        factory_definitions=manual_factories,
        # Create shortage for EngineType_1 (some MSNs won't get engines)
        engine_to_msn_ratio={
            "EngineType_1": 0.8,  # 20% shortage → unallocated MSNs
            "EngineType_2": 1.0   # Perfect balance
        },
        supplier_regularity={
            "EngineType_1": 0.7,  # Noisy deliveries → potential delays
            "EngineType_2": 0.9
        },
        supplier_production_rate={
            "EngineType_1": 0.9,  # Under-production → late deliveries → delays
            "EngineType_2": 1.0
        },
        flexible_delivery_prob=0.1,
        seed=42
    )

    problem = generate_instance(params)

    print(f"\n📊 Instance Summary:")
    print(f"   - {len(problem.engines_list)} engines")
    print(f"   - {len(problem.msns_list)} MSNs")
    print(f"   - {len(problem.factories_list)} factories")
    print(f"   - Time horizon: {problem.time_horizon} days")

    # Count expected shortage
    from collections import Counter
    engines_by_type = Counter(e.engine_type for e in problem.engines_list)
    msns_by_type = Counter(m.engine_type for m in problem.msns_list)

    print(f"\n📦 Supply vs Demand:")
    for etype in engines_by_type.keys():
        supply = engines_by_type[etype]
        demand = msns_by_type[etype]
        shortage = max(0, demand - supply)
        print(f"   {etype}: {supply} engines for {demand} MSNs (shortage: {shortage})")

    # =====================================================================
    # Setup Base Solver (CP-SAT)
    # =====================================================================

    print("\n" + "=" * 80)
    print("SETTING UP LEXICOGRAPHIC SOLVER")
    print("=" * 80)

    # Create base CP-SAT solver (no objectives yet - lexico solver will set them)
    base_solver = CPSatEngineSolver(problem=problem)
    base_solver.init_model()

    # Check if solver supports lexicographic optimization
    if not base_solver.implements_lexico_api():
        print("❌ Error: Solver does not implement lexicographic API!")
        sys.exit(1)

    available_objectives = base_solver.get_lexico_objectives_available()
    print(f"\n✅ Available objectives: {available_objectives}")

    # =====================================================================
    # Create Lexicographic Solver
    # =====================================================================

    lexico_solver = LexicoSolver(
        subsolver=base_solver,
        problem=problem
    )

    # Define objective priority order
    objectives_in_order = [
        "unallocated_msns",  # 1st priority: minimize unmet demand
        "arrival_delays",     # 2nd priority: minimize delays
        "travel_cost",
        "storage_cost",       # 3rd priority: minimize storage costs
    ]

    print(f"\n🎯 Optimizing in lexicographic order: {' → '.join(objectives_in_order)}")

    # =====================================================================
    # Solve with Lexicographic Optimization
    # =====================================================================

    print("\n" + "=" * 80)
    print("SOLVING WITH LEXICOGRAPHIC OPTIMIZATION")
    print("=" * 80)

    # Configure CP-SAT parameters
    cp_params = ParametersCp.default_cpsat()
    cp_params.nb_process = 8  # Use multiple workers

    # Solve!
    result_storage = lexico_solver.solve(
        objectives=objectives_in_order,
        parameters_cp=cp_params,
        time_limit=20,  # 20 seconds per objective
        ortools_cpsat_solver_kwargs={
            'log_search_progress': True,
        }
    )

    # =====================================================================
    # Analyze Results
    # =====================================================================

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    if len(result_storage) == 0:
        print("❌ No solution found!")
        return

    best_solution = result_storage[-1][0]
    kpis = problem.evaluate(best_solution)
    feasible = problem.satisfy(best_solution)

    print(f"\n{'Feasible:':<25} {'✅ Yes' if feasible else '❌ No'}")
    print(f"\n{'Objective':<25} {'Value':<15} {'Priority'}")
    print("-" * 60)
    print(f"{'unallocated_msns':<25} {kpis['unallocated_msns']:<15.0f} 1st (HIGHEST)")
    print(f"{'arrival_delays':<25} {kpis['arrival_delays']:<15.2f} 2nd")
    print(f"{'storage_cost':<25} {kpis['storage_cost']:<15.2f} 3rd (LOWEST)")

    if "travel_cost" in kpis:
        print(f"{'travel_cost':<25} {kpis['travel_cost']:<15.2f} (not optimized)")
    if "baseline_changes" in kpis:
        print(f"{'baseline_changes':<25} {kpis['baseline_changes']:<15.0f} (not optimized)")

    # Allocation details
    allocated_engines = sum(
        1 for e in problem.engines_list
        if e in best_solution.engine_to_msn and best_solution.engine_to_msn[e] is not None
    )
    unallocated_engines = len(problem.engines_list) - allocated_engines

    print(f"\n{'📊 Allocation Summary':}")
    print(f"   Allocated engines:    {allocated_engines}/{len(problem.engines_list)}")
    print(f"   Unallocated engines:  {unallocated_engines}")
    print(f"   Unallocated MSNs:     {kpis['unallocated_msns']:.0f}/{len(problem.msns_list)}")

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("""
Lexicographic optimization solves objectives SEQUENTIALLY:

1. First, minimize unallocated_msns to the MINIMUM possible value
   → This is the BEST we can do for meeting demand

2. Then, FIX unallocated_msns at that value and minimize arrival_delays
   → Among all solutions with MINIMUM unallocated MSNs, find the one with MINIMUM delays

3. Finally, FIX both and minimize storage_cost
   → Among all solutions with MIN unallocated_msns AND MIN delays, find MIN storage cost

This guarantees:
✅ No solution exists with FEWER unallocated MSNs
✅ No solution with SAME unallocated MSNs has LOWER delays
✅ No solution with SAME (unallocated, delays) has LOWER storage cost

Compare to weighted sum: would trade off objectives (e.g., might accept more unallocated
MSNs to get lower storage cost). Lexicographic NEVER makes such tradeoffs!
    """)

    print("=" * 80)
    print("✅ LEXICOGRAPHIC OPTIMIZATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
