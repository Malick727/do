"""
Minimal test to debug CP-SAT model
"""
from part_preparation.problem import *
from part_preparation.solvers.cpsat import CPSatEngineSolver
from ortools.sat.python import cp_model

# Create minimal problem
assembly_lines = [AssemblyLine(id='FAL1', takt_pace=7, location='Hamburg')]
msns = [
    MSN(id='MSN1', assembly_line_id='FAL1', engine_type='T1',
        entry_time=10, end_time=50, engine_installation_date=30),
    MSN(id='MSN2', assembly_line_id='FAL1', engine_type='T1',
        entry_time=20, end_time=60, engine_installation_date=40),
]
factories = [
    PreparationFactory(
        id='F1', location='Hamburg',
        compatible_engine_types={'T1'},
        preparation_durations={'T1': 10},
        capacity=None  # NO CAPACITY for simplicity
    ),
]
engines = [
    Engine(id='E1', engine_type='T1', delivery_time=5),
    Engine(id='E2', engine_type='T1', delivery_time=10),
]
travel_info = [
    TravelInfo(factory_id='F1', assembly_line_id='FAL1', travel_time=2, travel_cost=100),
]

problem = EnginePreparationProblem(
    assembly_lines=assembly_lines,
    msns=msns,
    factories=factories,
    engines=engines,
    travel_info=travel_info,
    time_horizon=60,
)

print(f"Problem: {len(engines)} engines, {len(msns)} MSNs, {len(factories)} factories")

# Try to solve
solver = CPSatEngineSolver(problem)
solver.init_model()

print(f"Model has {len(solver.cp_model.proto.variables)} variables")
print(f"Model has {len(solver.cp_model.proto.constraints)} constraints")

# Solve with verbose output
from ortools.sat.python.cp_model import CpSolver
cp_solver = CpSolver()
cp_solver.parameters.max_time_in_seconds = 10.0
cp_solver.parameters.log_search_progress = True

status = cp_solver.Solve(solver.cp_model)

print(f"\nStatus: {cp_solver.status_name(status)}")
print(f"Solve time: {cp_solver.wall_time}s")

if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
    print("\n✓ Solution found!")
else:
    print("\n✗ No solution found")
    if status == cp_model.INFEASIBLE:
        print("Problem is INFEASIBLE")
    elif status == cp_model.UNKNOWN:
        print("Solver timeout or other issue")