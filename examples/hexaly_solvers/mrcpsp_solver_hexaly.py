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


def run_mrcpsp_local_solver():
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


def run_all():
    logging.basicConfig(level=logging.INFO)
    files = [f for f in get_data_available() if "mm" in f]
    print("all files ", files)
    for file in files:
        rcpsp_model = parse_file(file)
        solver = LSolverMultiModeRCPSP(rcpsp_model=rcpsp_model)
        solver.init_model(modelisation_type=ModelisationLSolver.INTERVAL)
        res = solver.solve(timeout_seconds=5)
        sol, fit = res.get_best_solution_fit()
        assert rcpsp_model.satisfy(sol)
        print("Sol Hexaly", rcpsp_model.evaluate(sol))

        solver = CPSatRCPSPSolver(problem=rcpsp_model)
        solver.init_model(modelisation_type=ModelisationLSolver.INTERVAL)
        p = ParametersCP.default()
        p.time_limit = 5
        res = solver.solve(parameters_cp=p)
        sol, fit = res.get_best_solution_fit()
        print("Sol CP ", rcpsp_model.evaluate(sol))
        assert rcpsp_model.satisfy(sol)


if __name__ == "__main__":
    # run_rcpsp_ortools()
    run_all()