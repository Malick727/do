from functools import partial

import numpy as np
from typing import Union, Callable

from discrete_optimization.rcpsp import RCPSPModel, RCPSPModelCalendar
from discrete_optimization.rcpsp.rcpsp_model import RCPSPSolution
from discrete_optimization.rcpsp.fast_function_rcpsp import sgs_fast, \
    sgs_fast_partial_schedule_incomplete_permutation_tasks, compute_mean_ressource #, compute_ressource_consumption


def compute_resource_consumption_function(
    rcpsp_problem: Union[RCPSPModel, RCPSPModelCalendar]
):
    consumption_array = np.zeros(
        (
            rcpsp_problem.n_jobs,
            rcpsp_problem.max_number_of_mode,
            len(rcpsp_problem.resources_list),
        ),
        dtype=np.int32,
    )
    duration_array = np.zeros(
        (rcpsp_problem.n_jobs, rcpsp_problem.max_number_of_mode), dtype=np.int32
    )
    predecessors = np.zeros(
        (rcpsp_problem.n_jobs, rcpsp_problem.n_jobs), dtype=np.int32
    )
    successors = np.zeros((rcpsp_problem.n_jobs, rcpsp_problem.n_jobs), dtype=np.int32)
    horizon = rcpsp_problem.horizon
    ressource_available = np.zeros(
        (len(rcpsp_problem.resources_list), horizon), dtype=np.int32
    )
    ressource_renewable = np.ones((len(rcpsp_problem.resources_list)), dtype=bool)

    for i in range(len(rcpsp_problem.tasks_list)):
        task = rcpsp_problem.tasks_list[i]
        index_mode = 0
        for mode in sorted(
            rcpsp_problem.mode_details[rcpsp_problem.tasks_list[i]].keys()
        ):
            for k in range(len(rcpsp_problem.resources_list)):
                consumption_array[i, index_mode, k] = rcpsp_problem.mode_details[task][
                    mode
                ].get(rcpsp_problem.resources_list[k], 0)
            duration_array[i, index_mode] = rcpsp_problem.mode_details[task][mode][
                "duration"
            ]
            index_mode += 1

    task_index = {rcpsp_problem.tasks_list[i]: i for i in range(rcpsp_problem.n_jobs)}
    for k in range(len(rcpsp_problem.resources_list)):
        if rcpsp_problem.is_varying_resource():
            ressource_available[k, :] = rcpsp_problem.resources[
                rcpsp_problem.resources_list[k]
            ][: ressource_available.shape[1]]
        else:
            ressource_available[k, :] = np.full(
                ressource_available.shape[1],
                rcpsp_problem.resources[rcpsp_problem.resources_list[k]],
                dtype=int,
            )
        if rcpsp_problem.resources_list[k] in rcpsp_problem.non_renewable_resources:
            ressource_renewable[k] = False

    for i in range(len(rcpsp_problem.tasks_list)):
        task = rcpsp_problem.tasks_list[i]
        for s in rcpsp_problem.successors[task]:
            index_s = task_index[s]
            predecessors[index_s, i] = 1
            successors[i, index_s] = 1
    minimum_starting_time_array = np.zeros(rcpsp_problem.n_jobs, dtype=int)
    if "special_constraints" in rcpsp_problem.__dict__.keys():
        for t in rcpsp_problem.special_constraints.start_times_window:
            if rcpsp_problem.special_constraints.start_times_window[t][0] is not None:
                minimum_starting_time_array[
                    rcpsp_problem.index_task[t]
                ] = rcpsp_problem.special_constraints.start_times_window[t][0]
    func_sgs = partial(
        sgs_fast,
        consumption_array=consumption_array,
        duration_array=duration_array,
        predecessors=predecessors,
        successors=successors,
        horizon=horizon,
        ressource_available=ressource_available,
        ressource_renewable=ressource_renewable,
        minimum_starting_time_array=minimum_starting_time_array,
    )
    func_sgs_2 = partial(
        sgs_fast_partial_schedule_incomplete_permutation_tasks,
        consumption_array=consumption_array,
        duration_array=duration_array,
        predecessors=predecessors,
        successors=successors,
        horizon=horizon,
        ressource_available=ressource_available,
        ressource_renewable=ressource_renewable,
        minimum_starting_time_array=minimum_starting_time_array,
    )
    func_compute_mean_resource = partial(
        compute_ressource_consumption,
        consumption_array=consumption_array,
        ressource_available=ressource_available,
        ressource_renewable=ressource_renewable,
    )
    return func_compute_mean_resource


def compute_res_consumption(problem: RCPSPModel,
                            solution: RCPSPSolution,
                            function_numba: Callable,):
    last_activity = problem.sink_task
    makespan = solution.rcpsp_schedule[last_activity]["end_time"]
    return function_numba(
                        horizon=makespan,
                        modes_array=np.array(
                            problem.build_mode_array(solution.rcpsp_modes)
                        )
                        - 1,  # permutation_task=array(task)->task index
                        start_array=np.array(
                            [
                                solution.rcpsp_schedule[t]["start_time"]
                                for t in problem.tasks_list
                            ]
                        ),
                        end_array=np.array(
                            [
                                solution.rcpsp_schedule[t]["end_time"]
                                for t in problem.tasks_list
                            ]
                        ),
                    )