import os
os.environ["DO_SKIP_MZN_CHECK"] = "1"
import logging
import matplotlib.pyplot as plt
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.utils import plot_ressource_view, plot_task_gantt
from hexaly_solvers.rcpsp.hexaly_rcpsp_solver import LSolverMonomodeRCPSP, ModelisationLSolver
from hexaly_solvers.rcpsp.hexaly_mrcpsp_solver import HexalyMultiModeRCPSP, ModelisationLSolver
logging.basicConfig(level=logging.INFO)


def run_mrcpsp_local_solver():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "j1201_8.sm" in f][0]
    rcpsp_model = parse_file(file)
    solver = HexalyMultiModeRCPSP(problem=rcpsp_model)
    solver.init_model(modelisation_type=ModelisationLSolver.OPT_INTERVAL)
    res = solver.solve(time_limit=100)
    sol, fit = res.get_best_solution_fit()
    assert rcpsp_model.satisfy(sol)
    plot_task_gantt(rcpsp_problem=rcpsp_model, rcpsp_sol=sol)
    plt.show()


def run_all():
    logging.basicConfig(level=logging.INFO)
    files = [f for f in get_data_available() if "mm" in f]
    print("all files ", files)
    for file in files:
        rcpsp_model = parse_file(file)
        solver = HexalyMultiModeRCPSP(problem=rcpsp_model)
        solver.init_model(modelisation_type=ModelisationLSolver.INTERVAL)
        res = solver.solve(time_limit=5)
        sol, fit = res.get_best_solution_fit()
        assert rcpsp_model.satisfy(sol)
        print("Sol Hexaly", rcpsp_model.evaluate(sol))


if __name__ == "__main__":
    run_mrcpsp_local_solver()
    # run_rcpsp_ortools()
    # run_all()