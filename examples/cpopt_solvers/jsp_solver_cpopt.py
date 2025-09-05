#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
import sys
from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.jsp.job_shop_parser import get_data_available, parse_file
from discrete_optimization.jsp.job_shop_problem import JobShopProblem
from cpopt_solvers.jsp_cpopt_solver import CPOJspSolver

logging.basicConfig(level=logging.INFO)


def run_cpopt_jsp():
    file_path = get_data_available()[1]
    # file_path = [f for f in get_data_available() if "abz6" in f][0]
    problem = parse_file(file_path)
    print("File path ", file_path)
    solver = CPOJspSolver(problem=problem)
    p = ParametersCP.default_cpsat()
    res = solver.solve(parameters_cp=p,
                       time_limit=100)
    sol = res.get_best_solution_fit()[0]
    assert problem.satisfy(sol)
    print(problem.evaluate(sol))


if __name__ == "__main__":
    run_cpopt_jsp()
