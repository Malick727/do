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
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import random
from part_preparation.problem import (
    EnginePreparationProblem,
    AssemblyLine,
    MSN,
    PreparationFactory,
    Engine,
    TravelInfo,
)
from part_preparation.solvers.greedy import GreedyEngineSolver
from part_preparation.solvers.cpsat import CPSatEngineSolver
from part_preparation.dashboard_v2 import visualize_solution


def create_complex_example():
    """Create a complex realistic problem instance"""

    # ==========================================
    # 1. Create 4 Final Assembly Lines
    # ==========================================
    assembly_lines = [
        AssemblyLine(id="FAL_Hamburg", takt_pace=7, location="Hamburg"),
        AssemblyLine(id="FAL_Toulouse", takt_pace=5, location="Toulouse"),
        AssemblyLine(id="FAL_Mobile", takt_pace=6, location="Mobile"),
        AssemblyLine(id="FAL_Tianjin", takt_pace=8, location="Tianjin"),
    ]

    # ==========================================
    # 2. Create 8 Preparation Factories
    # (4 for TrentXWB, 4 for GE90, geographically distributed)
    # ==========================================
    factories = [
        # TrentXWB factories (Rolls-Royce)
        PreparationFactory(
            id="Derby_UK",
            location="Derby",
            compatible_engine_types={"TrentXWB"},
            preparation_durations={"TrentXWB": 12},
            capacity=3,
        ),
        PreparationFactory(
            id="Toulouse_FR",
            location="Toulouse",
            compatible_engine_types={"TrentXWB"},
            preparation_durations={"TrentXWB": 10},
            capacity=4,
        ),
        PreparationFactory(
            id="Virginia_US",
            location="Virginia",
            compatible_engine_types={"TrentXWB"},
            preparation_durations={"TrentXWB": 11},
            capacity=2,
        ),
        PreparationFactory(
            id="Singapore_SG",
            location="Singapore",
            compatible_engine_types={"TrentXWB"},
            preparation_durations={"TrentXWB": 13},
            capacity=3,
        ),
        # GE90 factories (GE Aviation)
        PreparationFactory(
            id="Durham_UK",
            location="Durham",
            compatible_engine_types={"GE90"},
            preparation_durations={"GE90": 14},
            capacity=3,
        ),
        PreparationFactory(
            id="Bordeaux_FR",
            location="Bordeaux",
            compatible_engine_types={"GE90"},
            preparation_durations={"GE90": 12},
            capacity=4,
        ),
        PreparationFactory(
            id="Cincinnati_US",
            location="Cincinnati",
            compatible_engine_types={"GE90"},
            preparation_durations={"GE90": 13},
            capacity=4,
        ),
        PreparationFactory(
            id="Shanghai_CN",
            location="Shanghai",
            compatible_engine_types={"GE90"},
            preparation_durations={"GE90": 15},
            capacity=2,
        ),
    ]

    # ==========================================
    # 3. Create 20 MSNs (5 per FAL, mixed engine types)
    # ==========================================
    msns = []
    msn_counter = 1

    # Distribution: Each FAL gets mix of both engine types
    fal_configs = [
        ("FAL_Hamburg", [("TrentXWB", 10, 50, 30), ("TrentXWB", 17, 57, 37), ("GE90", 25, 65, 45),
                         ("GE90", 32, 72, 52), ("TrentXWB", 40, 80, 60)]),
        ("FAL_Toulouse", [("GE90", 5, 45, 25), ("TrentXWB", 12, 52, 32), ("GE90", 20, 60, 40),
                          ("TrentXWB", 28, 68, 48), ("GE90", 35, 75, 55)]),
        ("FAL_Mobile", [("TrentXWB", 8, 48, 28), ("GE90", 15, 55, 35), ("TrentXWB", 22, 62, 42),
                        ("GE90", 30, 70, 50), ("TrentXWB", 38, 78, 58)]),
        ("FAL_Tianjin", [("GE90", 6, 46, 26), ("TrentXWB", 14, 54, 34), ("GE90", 21, 61, 41),
                         ("TrentXWB", 29, 69, 49), ("GE90", 37, 77, 57)]),
    ]

    for fal_id, configs in fal_configs:
        for engine_type, entry, end, install in configs:
            msns.append(MSN(
                id=f"MSN{msn_counter:03d}",
                assembly_line_id=fal_id,
                engine_type=engine_type,
                entry_time=entry,
                end_time=end,
                engine_installation_date=install,
            ))
            msn_counter += 1

    # ==========================================
    # 4. Create 20 Engines (10 per type)
    # WITH BASELINE DISTURBANCES
    # ==========================================
    engines = []
    random.seed(42)  # For reproducible disturbances

    # TrentXWB engines (10)
    trent_msns = [m for m in msns if m.engine_type == "TrentXWB"]
    trent_factories = [f for f in factories if "TrentXWB" in f.compatible_engine_types]

    for i in range(10):
        msn = trent_msns[i]

        # Baseline allocation (may not be optimal!)
        # Add disturbances:
        # - 30% chance: factory far from FAL
        # - 20% chance: different MSN than intended
        # - 40% chance: early/late start time

        if random.random() < 0.3:
            # Assign to factory FAR from the MSN's FAL (creates travel cost problem)
            baseline_factory = random.choice(trent_factories)
        else:
            # Assign to closest factory (reasonable)
            baseline_factory = trent_factories[i % len(trent_factories)]

        if random.random() < 0.2 and i < len(trent_msns) - 1:
            # Assign to WRONG MSN (creates allocation change)
            baseline_msn = trent_msns[i + 1]
        else:
            baseline_msn = msn

        # Start time: could be too early (wasted inventory) or too late (creates delay)
        ideal_start = msn.engine_installation_date - baseline_factory.get_preparation_duration("TrentXWB") - 5  # 5 days for travel

        if random.random() < 0.4:
            # Add disturbance: ±5 days
            baseline_start = ideal_start + random.randint(-5, 5)
        else:
            baseline_start = ideal_start

        engines.append(Engine(
            id=f"ENG_T{i+1:02d}",
            engine_type="TrentXWB",
            delivery_time=max(0, baseline_start - 3),  # Delivered 3 days before baseline start
            baseline_msn_id=baseline_msn.id,
            baseline_factory_id=baseline_factory.id,
            baseline_prep_start=baseline_start,
            baseline_prep_end=baseline_start + baseline_factory.get_preparation_duration("TrentXWB"),
        ))

    # GE90 engines (10)
    ge_msns = [m for m in msns if m.engine_type == "GE90"]
    ge_factories = [f for f in factories if "GE90" in f.compatible_engine_types]

    for i in range(10):
        msn = ge_msns[i]

        # Similar disturbances for GE90
        if random.random() < 0.3:
            baseline_factory = random.choice(ge_factories)
        else:
            baseline_factory = ge_factories[i % len(ge_factories)]

        if random.random() < 0.2 and i < len(ge_msns) - 1:
            baseline_msn = ge_msns[i + 1]
        else:
            baseline_msn = msn

        ideal_start = msn.engine_installation_date - baseline_factory.get_preparation_duration("GE90") - 5

        if random.random() < 0.4:
            baseline_start = ideal_start + random.randint(-5, 5)
        else:
            baseline_start = ideal_start

        engines.append(Engine(
            id=f"ENG_G{i+1:02d}",
            engine_type="GE90",
            delivery_time=max(0, baseline_start - 3),
            baseline_msn_id=baseline_msn.id,
            baseline_factory_id=baseline_factory.id,
            baseline_prep_start=baseline_start,
            baseline_prep_end=baseline_start + baseline_factory.get_preparation_duration("GE90"),
        ))

    # ==========================================
    # 5. Define Travel Info (geographically realistic)
    # ==========================================
    # Factories are "close" to certain FALs
    # - Derby_UK, Durham_UK → close to FAL_Hamburg
    # - Toulouse_FR, Bordeaux_FR → close to FAL_Toulouse
    # - Virginia_US, Cincinnati_US → close to FAL_Mobile
    # - Singapore_SG, Shanghai_CN → close to FAL_Tianjin

    travel_info = []

    # Define base travel times and costs (days, euros)
    # Format: (factory_location, fal_location): (time, cost)
    travel_matrix = {
        # Derby/Durham (UK) to FALs
        ("Derby", "Hamburg"): (2, 400),
        ("Derby", "Toulouse"): (3, 700),
        ("Derby", "Mobile"): (5, 2000),
        ("Derby", "Tianjin"): (7, 3000),
        ("Durham", "Hamburg"): (2, 450),
        ("Durham", "Toulouse"): (3, 750),
        ("Durham", "Mobile"): (5, 2100),
        ("Durham", "Tianjin"): (7, 3100),

        # Toulouse/Bordeaux (France) to FALs
        ("Toulouse", "Hamburg"): (3, 600),
        ("Toulouse", "Toulouse"): (1, 100),  # Same location!
        ("Toulouse", "Mobile"): (5, 1900),
        ("Toulouse", "Tianjin"): (7, 2900),
        ("Bordeaux", "Hamburg"): (3, 650),
        ("Bordeaux", "Toulouse"): (2, 300),
        ("Bordeaux", "Mobile"): (5, 2000),
        ("Bordeaux", "Tianjin"): (7, 3000),

        # Virginia/Cincinnati (USA) to FALs
        ("Virginia", "Hamburg"): (5, 1900),
        ("Virginia", "Toulouse"): (5, 1800),
        ("Virginia", "Mobile"): (2, 500),  # Close
        ("Virginia", "Tianjin"): (8, 3500),
        ("Cincinnati", "Hamburg"): (5, 2000),
        ("Cincinnati", "Toulouse"): (5, 1900),
        ("Cincinnati", "Mobile"): (2, 450),  # Close
        ("Cincinnati", "Tianjin"): (8, 3600),

        # Singapore/Shanghai (Asia) to FALs
        ("Singapore", "Hamburg"): (7, 3200),
        ("Singapore", "Toulouse"): (7, 3100),
        ("Singapore", "Mobile"): (8, 3700),
        ("Singapore", "Tianjin"): (3, 800),  # Close
        ("Shanghai", "Hamburg"): (7, 3000),
        ("Shanghai", "Toulouse"): (7, 2900),
        ("Shanghai", "Mobile"): (8, 3500),
        ("Shanghai", "Tianjin"): (2, 400),  # Close
    }

    for factory in factories:
        for assembly_line in assembly_lines:
            key = (factory.location, assembly_line.location)
            if key in travel_matrix:
                travel_time, travel_cost = travel_matrix[key]
                travel_info.append(TravelInfo(
                    factory_id=factory.id,
                    assembly_line_id=assembly_line.id,
                    travel_time=travel_time,
                    travel_cost=travel_cost,
                ))

    # ==========================================
    # 6. Create Problem
    # ==========================================
    problem = EnginePreparationProblem(
        assembly_lines=assembly_lines,
        msns=msns,
        factories=factories,
        engines=engines,
        travel_info=travel_info,
        time_horizon=100,
    )

    return problem


def main():
    """Run complex example with both Greedy and CP-SAT solvers"""
    print("=" * 100)
    print("Engine Preparation and Allocation - Complex Realistic Example")
    print("=" * 100)

    # Create problem
    problem = create_complex_example()

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
    cpsat_results = cpsat_solver.solve(time_limit=60)

    if len(cpsat_results) > 0:
        cpsat_solution = cpsat_results.get_best_solution()
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

        # ==========================================
        # Visualize CP-SAT solution
        # ==========================================
        print("\n" + "=" * 100)
        print("INTERACTIVE VISUALIZATION")
        print("=" * 100)
        print("\nGenerating interactive charts...")
        print("  - Gantt chart: Engine preparation timeline by factory")
        print("  - Workload: Factory capacity utilization over time")
        print("  - Arrivals: Engine arrivals at FALs with delays")
        print("  - Cumulative delays: Delay accumulation at each FAL")

        visualize_solution(
            problem=problem,
            solution=cpsat_solution,
            save_html="part_preparation_solution.html"
        )

    else:
        print("\n✗ No CP-SAT solution found")


if __name__ == "__main__":
    main()
