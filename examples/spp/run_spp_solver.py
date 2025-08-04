import os

from satellite_path_planning.problem import SppProblem, SppSolution
from satellite_path_planning.parser import parse_spp_problem
from satellite_path_planning.solver.dp import DpSppSolver, DpModeling, dp
from satellite_path_planning.solver.solver_bf import (SppBackwardSolver, SppForwardSolver,
                                                      SppBackwardForwardSolver, SppSolverRandomPick)
from satellite_path_planning.solver.astar_based import AStarBasedSppSolver
from discrete_optimization.generic_tools.callbacks.stats_retrievers import BasicStatsCallback

folder_data = os.environ["SPP_PATH"]
instance_medium = os.path.join(folder_data, "eosp_376_medium_cases/")


def get_instances_path(folder: str = None):
    if folder is None:
        folder = instance_medium
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if "json" in f]


def run_dp():
    file = get_instances_path()[2]
    problem = parse_spp_problem(file)
    solver = SppForwardSolver(problem)
    res = solver.solve(nb_way_memory=0)
    sol = res[-1][0]
    print(problem.satisfy(sol))
    print(problem.evaluate(sol))
    solver = DpSppSolver(problem)
    solver.init_model(modeling=DpModeling.GOTO)
    # solver.set_warm_start(sol)
    res = solver.solve(time_limit=20,
                       solver=dp.LNBS,
                       retrieve_intermediate_solutions=True,
                       threads=10)
    sol = res[-1][0]
    print(problem.solution)
    print(problem.satisfy(sol))
    print(problem.evaluate(sol))


def run_bf():
    file = [f for f in get_instances_path()
            if "eosp_80acqr_22ramp_13greedy_aZZKEGng.json" in f][0]
    problem = parse_spp_problem(file)
    solver = SppBackwardForwardSolver(problem)
    res = solver.solve(nb_way_memory=3)
    sol: SppSolution = res[-1][0]
    print(problem.solution)
    print(sol.solution_path)
    print(problem.satisfy(sol))
    print(problem.evaluate(sol))


def run_random_bf():
    file = get_instances_path()[5]
    problem = parse_spp_problem(file)
    solver = SppSolverRandomPick(problem)
    cb = BasicStatsCallback()
    res = solver.solve(callbacks=[cb],
                       nb_way_memory=2,
                       random_pick=0.2,
                       nb_iterations=10,
                       forward=False)
    sol: SppSolution = res[-1][0]
    print(sol.solution_value)
    print(problem.solution)
    print(problem.satisfy(sol))
    print(problem.evaluate(sol))
    print(cb.stats)


def run_astar_based():
    file = get_instances_path()[7]
    #{'value': 311159553.9}
    problem = parse_spp_problem(file)
    solver = AStarBasedSppSolver(problem)
    cb = BasicStatsCallback()
    res = solver.solve(time_limit=10, callbacks=[cb])
    for sol, fit in res:
        print(problem.satisfy(sol))
    sol: SppSolution = res[-1][0]
    print(res.list_solution_fits)
    print(sol.solution_value)
    print(problem.solution)
    print(problem.satisfy(sol))
    print(problem.evaluate(sol))
    print(cb.stats)


if __name__ == "__main__":
    run_bf()