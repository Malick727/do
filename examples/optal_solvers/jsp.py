#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import sys
from discrete_optimization.jsp.parser import get_data_available, parse_file
from discrete_optimization.jsp.parser import JobShopProblem
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from optalcp_solvers.solvers.jsp_optal import OptalJspSolver

logging.basicConfig(level=logging.INFO)


def run_optal_jsp():
    file_path = get_data_available()[2]
    problem = parse_file(file_path)
    solver = OptalJspSolver(problem=problem)
    solver.init_model()
    p = ParametersCp.default_cpsat()
    p.nb_process = 16
    res = solver.solve(parameters_cp=p, time_limit=10)
    sol = res.get_best_solution()
    print(problem.satisfy(sol), problem.evaluate(sol))
    stats = solver.get_output_stats()
    obj_hist = [x["objective"] for x in stats["objectiveHistory"]]
    time_obj = [x["solveTime"] for x in stats["objectiveHistory"]]
    bound_hist = [x["value"] for x in stats["lowerBoundHistory"]]
    time_bound = [x["solveTime"] for x in stats["lowerBoundHistory"]]
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1)
    ax.plot(time_obj, obj_hist, color="g", label="obj")
    ax.plot(time_bound, bound_hist, color="r", label="bound")
    ax.set_xlabel("Time computation")
    ax.legend()
    ax.semilogx()
    plt.show()
    print(stats)
    print("hey")

if __name__ == "__main__":
    run_optal_jsp()
