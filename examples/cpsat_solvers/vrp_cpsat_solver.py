
from matplotlib import pyplot as plt
from typing import Any

from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.vrp.vrp_parser import get_data_available, parse_file
from discrete_optimization.vrp.vrp_model import VrpProblem2D, VrpSolution
from cpsat_solvers.vrp.vrp_cpsat_solver import CpSatVrpSolver
from examples.hexaly_solvers.vrp_solver_hexaly import plot_vrp_solution
import logging
logging.basicConfig(level=logging.INFO)

def run():
    file = [f for f in get_data_available() if "vrp_26_8_1" in f][0]
    problem = parse_file(file_path=file)
    problem.vehicle_capacities = [problem.vehicle_capacities[i] for i in range(problem.vehicle_count)]
    print(problem)
    solver = CpSatVrpSolver(problem=problem)
    solver.init_model()
    p = ParametersCP.default_cpsat()
    p.nb_process = 10
    p.time_limit = 100
    res = solver.solve(parameters_cp=p)
    sol, fit = res.get_best_solution_fit()
    sol: VrpSolution
    print(problem.evaluate(sol))
    plot_vrp_solution(vrp_model=problem, solution=sol)
    plt.show()


if __name__ == "__main__":
    run()