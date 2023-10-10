import os
os.environ["DO_SKIP_MZN_CHECK"] = "1"
from ortools.linear_solver import pywraplp
from discrete_optimization.generic_tools.do_problem import ObjectiveHandling
from aircraft_loading.solvers.local_search_solver import LocalSolverAircraftLoading
from aircraft_loading.solvers.ortools_lp_solver import OrtoolsLpTeamAllocationSolver
from aircraft_loading.aircraft_loading_problem import AircraftLoadingSolution, AircraftLoadingProblem
import logging
logging.basicConfig(level=logging.INFO)


def create_problem(j=2):
    return AircraftLoadingProblem(container_types=[2, 2, 3, 1] * j,
                                  weights=[4, 2, 5, 2] * j,
                                  slot_distances=list(range(-3, 4)),
                                  max_loading_weight=9*j,
                                  max_shear=4*j,
                                  length=8,
                                  cog_bounds=[-1, 1])


def create_problem_toy_ionq():
    return AircraftLoadingProblem(container_types=[2, 2, 1, 1],
                                  weights=[4, 2, 5, 2],
                                  slot_distances=[-3, 1],
                                  max_loading_weight=9,
                                  max_shear=4,
                                  length=8,
                                  cog_bounds=[-1, 1])


def run_local_solver():
    problem = create_problem(j=10)
    solver = LocalSolverAircraftLoading(problem=problem)
    result = solver.solve(nb_iteration_max=20000)
    solution, fit = result.get_best_solution_fit()
    print("Solution : ", solution)
    print("Sat ", problem.satisfy(solution))
    print("Eval ", problem.evaluate(solution))


def run_ga():
    from discrete_optimization.generic_tools.ea.ga import Ga
    problem = create_problem(10)
    solver = Ga(problem=problem,
                objectives=["value", "penalty_violation"],
                objective_weights=[1, -100],
                objective_handling=ObjectiveHandling.AGGREGATE,
                pop_size=25,
                max_evals=100000,
                mut_rate=0.1,
                encoding="allocation")
    res = solver.solve()
    solution, fit = res.get_best_solution_fit()
    print("Solution : ", solution)
    print("Sat ", problem.satisfy(solution))
    print("Eval ", problem.evaluate(solution))


def run_ortools(size_instance: int = 4, toy: bool = False):
    if not toy:
        problem = create_problem(size_instance)
    else:
        problem = create_problem_toy_ionq()
    solver = OrtoolsLpTeamAllocationSolver(problem=problem,
                                           solver_name=
                                           "gurobi")
    solver.init_model(overlap=False)
    sol = [4, -1, -1, -1, 0, -1, -1, -1, 1, -1, -1, 6, -1, 3, 6, 6, -1, 5, -1, -1, 2, 3,
           6, -1, 2, 5, -1, -1, -1, -1, -1, -1, 6, 4, -1, -1, 1, 0, -1, -1]
    # print([problem.slot_distances[x] if x != -1 else -20 for x in sol])
    for i in range(len(sol)):
        if sol[i] != -1:
            solver.model.Add(solver.variables["allocation_binary"][i][sol[i]] == 1)
        else:
            for j in solver.variables["allocation_binary"][i]:
                solver.model.Add(solver.variables["allocation_binary"][i][j] == 0)
    result = solver.solve()
    solution, fit = result.get_best_solution_fit()
    print("Solution : ", solution)
    print("Sat ", problem.satisfy(solution))
    print("Eval ", problem.evaluate(solution))


def debug():
    problem = create_problem(4)
    sol = [3, 2, -1, 5, 1, 4, 6, -1, 3, 4, -1, 0, 2, 1, -1, 6]
    sol = AircraftLoadingSolution(problem, sol)
    problem.sat_packing(sol)


if __name__ == "__main__":
    run_local_solver()
