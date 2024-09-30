import os
from docplex.cp.modeler import end_of
from docplex.cp.solution import CpoRefineConflictResult

os.environ["DO_SKIP_MZN_CHECK"] = "1"
from discrete_optimization.datasets import get_data_home
import logging
import matplotlib.pyplot as plt
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file
from discrete_optimization.rcpsp.rcpsp_solvers import CP_RCPSP_MZN, ParametersCP
from discrete_optimization.rcpsp.solver.cpsat_solver import CPSatRCPSPSolver
from discrete_optimization.rcpsp.rcpsp_utils import plot_ressource_view, plot_task_gantt
from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from cpopt_solvers.rcpsp_cpopt_solver import CPOptRCPSPSolver


def run_rcpsp_cpo_solver():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "j1201_2.sm" in f][0]
    data_folder_rcp = f"{get_data_home()}/rcpsp/RG300/"
    files_patterson = get_data_available(data_folder=data_folder_rcp)
    file = [f for f in files_patterson if "RG300_185.rcp" in f][0]
    rcpsp_model = parse_file(file)
    solver = CPOptRCPSPSolver(problem=rcpsp_model)
    solver.init_model()
    res = solver.solve(callbacks=[ObjectiveLogger(step_verbosity_level=logging.INFO,
                                                  end_verbosity_level=logging.INFO)],
                       timeout_seconds=50)
    sol, fit = res.get_best_solution_fit()
    assert rcpsp_model.satisfy(sol)
    plot_task_gantt(rcpsp_model=rcpsp_model, rcpsp_sol=sol)
    plt.show()


def run_rcpsp_explainable():
    logging.basicConfig(level=logging.INFO)
    file = [f for f in get_data_available() if "j301_1.sm" in f][0]
    # data_folder_rcp = f"{get_data_home()}/rcpsp/RG300/"
    # files_patterson = get_data_available(data_folder=data_folder_rcp)
    # file = [f for f in files_patterson if "RG300_185.rcp" in f][0]
    rcpsp_model = parse_file(file)
    solver = CPOptRCPSPSolver(problem=rcpsp_model)
    solver.init_model()
    solver.cp_model.add(end_of(solver.variables["intervals"][rcpsp_model.sink_task]) <= 42)
    p = ParametersCP.default()
    p.time_limit = 30
    res = solver.explain_infeasibility(parameters_cp=p,
                                       explain_failure=False,
                                       refine_conflict=True)
    c: CpoRefineConflictResult = res[0]
    print(c.print_conflict())


if __name__ == "__main__":
    run_rcpsp_explainable()