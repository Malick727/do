#  Copyright (c) 2024-2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging

import matplotlib.pyplot as plt

from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.tsp.parser import get_data_available, parse_file
from discrete_optimization.tsp.plot import plot_tsp_solution
from optal_py_solvers.tsp_optalcp_solver import TspOptalCpSolver, cp
logging.basicConfig(level=logging.INFO)


def run_optal_solver():
    files = get_data_available()
    files = [f for f in files if "tsp_200_1" in f]
    model = parse_file(files[0], start_index=0, end_index=0)
    params_objective_function = get_default_objective_setup(problem=model)
    solver = TspOptalCpSolver(model, params_objective_function=params_objective_function)
    solver.init_model()
    parameters_cp = ParametersCp.default_cpsat()
    parameters_cp.nb_process = 4
    fds_worker = cp.WorkerParameters(
        searchType="FDS",
        noOverlapPropagationLevel=4,
    )
    fdslb_worker = cp.WorkerParameters(
        searchType="FDSLB",
        noOverlapPropagationLevel=4
    )
    res = solver.solve(parameters_cp=parameters_cp,
                       time_limit=30,
                       workers=[fds_worker,
                                fdslb_worker])
    sol, fitness = res.get_best_solution_fit()
    assert model.satisfy(sol)
    fig, ax = plt.subplots(1)
    for sol, fit in res.list_solution_fits:
        ax.clear()
        plot_tsp_solution(tsp_model=model, solution=sol, ax=ax)
        ax.set_title(f"Length ={fit}")
        plt.pause(0.1)
    plt.show()


if __name__ == "__main__":
    run_optal_solver()
