from discrete_optimization.pickup_vrp.gpdp import ProxyClass
from discrete_optimization.tsp.tsp_parser import get_data_available, parse_file
from cpsat_solvers.tsp.tsp_cpsat_solver import CpSatTspSolver
from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.tsp.solver.solver_ortools import TSP_ORtools
from discrete_optimization.pickup_vrp.solver.ortools_solver import ORToolsGPDP, ParametersCost
from discrete_optimization.tsp.plots.plot_tsp import plot_tsp_solution, plt
import logging
logging.basicConfig(level=logging.INFO)


def cpsat_tsp():
    files = get_data_available()
    file = [f for f in files if "tsp_1323_1" in f][0]
    problem = parse_file(file)
    solver = CpSatTspSolver(problem)
    solver.init_model()
    p = ParametersCP.default_cpsat()
    p.time_limit = 200
    res = solver.solve(parameters_cp=p)
    sol = res.get_best_solution()
    assert problem.satisfy(sol)
    print(problem.evaluate(sol))
    plot_tsp_solution(tsp_model=problem, solution=sol)
    plt.show()


if __name__ == "__main__":
    cpsat_tsp()