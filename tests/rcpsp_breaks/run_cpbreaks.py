import logging
import os
os.environ["DO_SKIP_MZN_CHECK"] = "1"
from typing import Optional
from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.rcpsp.rcpsp_parser import get_data_available, parse_file
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPSolution
from rcpsp_breaks.rcpsp_breaks_cpopt import CPOptRCPSPBreakSolver
from rcpsp_breaks.rcpsp_breaks_ortools import OrtoolsRCPSPBreakSolver
from discrete_optimization.rcpsp.rcpsp_utils import plot_task_gantt, plt, plot_ressource_view
import numpy as np
import random


def create_variable_resource(rcpsp_model: RCPSPModel,
                             new_horizon: Optional[int],
                             nb_breaks: int = 20):
    resource_availability = {r: np.full(new_horizon,
                                        rcpsp_model.get_max_resource_capacity(r))
                             for r in rcpsp_model.resources_list}
    for r in resource_availability:
        for j in range(nb_breaks):
            t = random.randint(0, len(resource_availability[r])-1)
            len_break = random.randint(1, 5)
            resource_availability[r][t:t+len_break] = 0
    return RCPSPModel(resources=resource_availability,
                      non_renewable_resources=rcpsp_model.non_renewable_resources,
                      mode_details=rcpsp_model.mode_details,
                      successors=rcpsp_model.successors,
                      horizon=new_horizon-1,
                      tasks_list=rcpsp_model.tasks_list,
                      source_task=rcpsp_model.source_task,
                      sink_task=rcpsp_model.sink_task)


def run_calendar_experiments():
    file = [f for f in get_data_available() if "j601_1.sm" in f][0]
    print(file)
    model: RCPSPModel = parse_file(file_path=file)
    model = create_variable_resource(model,
                                     new_horizon=300,
                                     nb_breaks=20)
    solver = CPOptRCPSPBreakSolver(rcpsp_model=model)
    parameters_cp = ParametersCP.default()
    solver.init_model()
    results = solver.solve(parameters_cp=parameters_cp)
    sol, fit = results.get_best_solution_fit()
    print(fit, " cpopt makespan ")

    sol: RCPSPSolution = sol
    for task in sol.rcpsp_schedule:
        duration = sol.rcpsp_schedule[task]["end_time"]-sol.rcpsp_schedule[task]["start_time"]
        exp_dur = model.mode_details[task][1]["duration"]
        if exp_dur != duration:
            logging.debug(f"Length of size {task} different to its 'execution size' : {duration}!={exp_dur}")
    plot_task_gantt(rcpsp_model=model, rcpsp_sol=sol)
    plot_ressource_view(rcpsp_model=model, rcpsp_sol=sol)

    solver2 = OrtoolsRCPSPBreakSolver(problem=model)
    parameters_cp = ParametersCP.default()
    # Method with element : Inspired by https://groups.google.com/g/or-tools-discuss/c/-HhchJcMTV4
    # solver2.init_model_durations_element_constraint()
    # Method with indicators : Inspired by https://groups.google.com/g/or-tools-discuss/c/DA_4Pniwhn8?pli=1
    solver2.init_model_durations_indicator_method()

    results = solver2.solve(parameters_cp=parameters_cp,
                            add_solution_callback=True)
    sol2, fit2 = results.get_best_solution_fit()
    print(fit2, " ortools makespan ")
    for task in sol2.rcpsp_schedule:
        duration = sol2.rcpsp_schedule[task]["end_time"] - sol2.rcpsp_schedule[task]["start_time"]
        exp_dur = model.mode_details[task][1]["duration"]
        if exp_dur != duration:
            logging.debug(f"Length of size {task} different to its 'execution size' : {duration}!={exp_dur}")
    plot_task_gantt(rcpsp_model=model, rcpsp_sol=sol2)
    plot_ressource_view(rcpsp_model=model, rcpsp_sol=sol2)
    plt.show()


if __name__ == "__main__":
    run_calendar_experiments()

