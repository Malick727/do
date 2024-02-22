import os

os.environ["DO_SKIP_MZN_CHECK"] = "1"
import logging

import matplotlib.pyplot as plt
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file
from discrete_optimization.rcpsp.rcpsp_solvers import CP_RCPSP_MZN, ParametersCP
from discrete_optimization.rcpsp.solver.cpsat_solver import CPSatRCPSPSolver
from discrete_optimization.rcpsp.rcpsp_utils import plot_ressource_view, plot_task_gantt
from hexaly_solvers.rcpsp.hexaly_rcpsp_solver import LSolverMonomodeRCPSP, ModelisationLSolver
from hexaly_solvers.rcpsp.hexaly_mrcpsp_solver import LSolverMultiModeRCPSP, ModelisationLSolver
logging.basicConfig(level=logging.DEBUG)

def run_rcpsp_local_solver():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "j1010_8.mm" in f][0]
    rcpsp_model = parse_file(file)
    solver = LSolverMultiModeRCPSP(rcpsp_model=rcpsp_model)
    solver.init_model(modelisation_type=ModelisationLSolver.INTERVAL,
                      )
    res = solver.solve(timeout_seconds=5)
    sol, fit = res.get_best_solution_fit()
    assert rcpsp_model.satisfy(sol)
    plot_task_gantt(rcpsp_model=rcpsp_model, rcpsp_sol=sol)
    plt.show()


def run_rcpsp_cp():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "j1201_10.sm" in f][0]
    rcpsp_model = parse_file(file)
    solver = CP_RCPSP_MZN(rcpsp_model=rcpsp_model)
    solver.init_model(output_type=True)
    params_cp = ParametersCP.default()
    params_cp.time_limit = 50
    res = solver.solve(parameters_cp=params_cp)
    print("Status solver : ", solver.get_status_solver())
    sol, fit = res.get_best_solution_fit()
    plot_task_gantt(rcpsp_model=rcpsp_model, rcpsp_sol=sol)
    plt.show()


def run_rcpsp_ortools():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "j1201_2.sm" in f][0]
    rcpsp_model = parse_file(file)
    solver = CPSatRCPSPSolver(problem=rcpsp_model)
    solver.init_model()
    params_cp = ParametersCP.default()
    params_cp.time_limit = 50
    res = solver.solve(parameters_cp=params_cp)
    print("Status solver : ", solver.get_status_solver())
    sol, fit = res.get_best_solution_fit()
    plot_task_gantt(rcpsp_model=rcpsp_model, rcpsp_sol=sol)
    plt.show()


if __name__ == "__main__":
    # run_rcpsp_ortools()
    run_rcpsp_local_solver()