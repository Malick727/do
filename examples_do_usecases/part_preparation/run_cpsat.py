"""
Run CP-SAT solver on engine preparation problem and launch visualization dashboards

Usage:
    uv run python examples_do_usecases/part_preparation/run_cpsat_and_visualize.py --instance medium --time-limit 60
    uv run python examples_do_usecases/part_preparation/run_cpsat_and_visualize.py --instance large --time-limit 300
"""
import logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import pickle
import subprocess
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction, ObjectiveHandling, ModeOptim
from part_preparation.instance_generator import generate_instance, InstanceParams, FactoryDefinition
from part_preparation.solvers.cpsat import CPSatEngineSolver
logging.basicConfig(level=logging.INFO)


def main():
    # Example: Manual factory definitions for precise control over bottlenecks
    # FAL indices: 0=Hamburg, 1=Toulouse, 2=Mobile, 3=Tianjin
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
        n_msns_per_fal=80,
        n_engine_types=2,
        time_horizon=1000,
        fal_takt_paces=[2, 2, 5, 5],
        factory_definitions=manual_factories,  # Use manual factory definitions
        engine_to_msn_ratio={"EngineType_1": 1.1,
                             "EngineType_2": 0.95},
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
    # Configure objective function
    # Use all objectives with custom weights
    params_objective = ParamsObjectiveFunction(
        objective_handling=ObjectiveHandling.AGGREGATE,
        objectives=[#"baseline_changes",
                    "arrival_delays",
                    "travel_cost",
                    "storage_cost",
                    "unallocated_msns"],
        weights=[1000.0, 1.0, 1.0, 1000.0],  # High penalty for delays and unallocated MSNs
        sense_function=ModeOptim.MINIMIZATION
    )
    from discrete_optimization.generic_tools.callbacks.loggers import ProblemEvaluateLogger



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
        callbacks=[ProblemEvaluateLogger(step_verbosity_level=logging.INFO,
                                         end_verbosity_level=logging.INFO)],
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

if __name__ == "__main__":
    main()
