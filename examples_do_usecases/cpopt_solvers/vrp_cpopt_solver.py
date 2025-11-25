
from matplotlib import pyplot as plt
from typing import Any

from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.vrp.vrp_parser import get_data_available, parse_file
from discrete_optimization.vrp.vrp_model import VrpProblem2D
from cpopt_solvers.vrp_cpopt_solver import CPOptSolverVRP, VrpSolution
from examples.hexaly_solvers.vrp_solver_hexaly import plot_vrp_solution


def run():
    file = [f for f in get_data_available() if "vrp_76_9_3" in f][0]
    problem = parse_file(file_path=file)
    # problem.vehicle_capacities = [problem.vehicle_capacities[i] * 2 for i in range(problem.vehicle_count)]
    print(problem)
    solver = CPOptSolverVRP(problem=problem)

    solver.init_model()
    p = ParametersCP.default()
    res = solver.solve(parameters_cp=p, time_limit=20)
    sol, fit = res.get_best_solution_fit()
    sol: VrpSolution
    print(sol.lengths)
    print(fit)
    print(problem.satisfy(sol), problem.evaluate(sol))
    plot_vrp_solution(vrp_model=problem, solution=sol)
    plt.show()


if __name__ == "__main__":
    run()