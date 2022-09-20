from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from typing import Dict, Set, List, Hashable
import logging
# logging.basicConfig(level=logging.DEBUG)
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPSolution
from custom_utils import compute_resource_consumption_function, compute_res_consumption
import pymzn
from discrete_optimization.generic_tools.graph_api import Graph
from discrete_optimization.rcpsp.solver.cp_solvers import CP_MRCPSP_MZN, CP_RCPSP_MZN, \
    CPSolverName, RCPSPSolCP, ParametersCP
import numpy as np
from cpo_solver_res_allocation import CPOptSolver
from scheduling_dmz.dmz_script import custom_plot_resource_individual_gantt
import os

this_dir = os.path.dirname(os.path.abspath(__file__))
default_data = os.path.join(this_dir, "data/data_example.dzn")


def compute_individual_worker_schedule(rcpsp_sol: RCPSPSolution,
                                       rcpsp_model: RCPSPModel,
                                       resource_needed: Dict[int, Dict[str, Set[int]]]):
    resources = rcpsp_model.resources_list
    sorted_task_by_start = sorted(
        rcpsp_sol.rcpsp_schedule,
        key=lambda x: 100000 * rcpsp_sol.get_start_time(x) + rcpsp_model.index_task[x],
    )
    sorted_task_by_end = sorted(
        rcpsp_sol.rcpsp_schedule,
        key=lambda x: 100000 * rcpsp_sol.get_end_time(x) + rcpsp_model.index_task[x],
    )
    max_time = rcpsp_sol.get_end_time(sorted_task_by_end[-1])
    min_time = rcpsp_sol.get_start_time(sorted_task_by_end[0])
    with_calendar = rcpsp_model.is_varying_resource()
    array_ressource_usage = {
        resources[i]: {
            "activity": np.zeros(
                (
                    max_time - min_time + 1,
                    rcpsp_model.get_max_resource_capacity(resources[i]),
                )
            ),
            "binary_activity": np.zeros(
                (
                    max_time - min_time + 1,
                    rcpsp_model.get_max_resource_capacity(resources[i]),
                )
            ),
            "total_activity": np.zeros(
                rcpsp_model.get_max_resource_capacity(resources[i])
            ),
            "activity_last_n_hours": np.zeros(
                (
                    max_time - min_time + 1,
                    rcpsp_model.get_max_resource_capacity(resources[i]),
                )
            ),
            "boxes_time": [],
        }
        for i in range(len(resources))}
    done_task = set()
    for task in resource_needed:
        for resource in resource_needed[task]:
            for index_ressource in resource_needed[task][resource]:
                st = rcpsp_sol.get_start_time(task)
                end = rcpsp_sol.get_end_time(task)
                array_ressource_usage[resource]["activity"][st:end, index_ressource] = task
                array_ressource_usage[resource]["binary_activity"][st:end, index_ressource] = 1
                array_ressource_usage[resource]["total_activity"][index_ressource] += 1
                array_ressource_usage[resource]["boxes_time"] += [
                    [
                        (index_ressource - 0.25, st + 0.01, task),
                        (index_ressource - 0.25, end - 0.01, task),
                        (index_ressource + 0.25, end - 0.01, task),
                        (index_ressource + 0.25, st + 0.01, task),
                        (index_ressource - 0.25, st + 0.01, task),
                    ]
                ]
        if sum([len(resource_needed[task][r]) for r in resource_needed[task]]) \
                == sum([rcpsp_model.mode_details[task][1][r] for r in resources]):
            done_task.add(task)

    for task in sorted_task_by_start:
        if task in done_task:
            continue
        else:
            start = rcpsp_sol.get_start_time(task)
            end = rcpsp_sol.get_end_time(task)
            for r in resources:
                nb_need = rcpsp_model.mode_details[task][1].get(r, 0)
                if nb_need>0:
                    if task in resource_needed and (r not in resource_needed[task] or len(resource_needed[task][r])<nb_need):
                        nb_res_to_add = nb_need - len(resource_needed[task].get(r, []))
                        rneeded = nb_res_to_add
                        if rneeded > 0:
                            availables_people_r = [
                                i
                                for i in range(array_ressource_usage[r]["activity"].shape[1])
                                if np.max(np.maximum(0, array_ressource_usage[r]["activity"][
                                       start:end, i
                                   ])) == 0
                                ]
                            sorted_people = sorted(
                                availables_people_r,
                                key=lambda x: array_ressource_usage[r]["total_activity"][x],
                                reverse=True
                            )
                            assert len(sorted_people)>=rneeded
                            to_add = sorted_people[:rneeded]
                            for index_ressource in to_add:
                                array_ressource_usage[r]["activity"][start:end, index_ressource] = task
                                array_ressource_usage[r]["binary_activity"][start:end, index_ressource] = 1
                                array_ressource_usage[r]["total_activity"][index_ressource] += 1
                                array_ressource_usage[r]["boxes_time"] += [
                                    [
                                        (index_ressource - 0.25, start + 0.01, task),
                                        (index_ressource - 0.25, end - 0.01, task),
                                        (index_ressource + 0.25, end - 0.01, task),
                                        (index_ressource + 0.25, start + 0.01, task),
                                        (index_ressource - 0.25, start + 0.01, task),
                                    ]
                                ]
    return array_ressource_usage


def create_rcpsp_model(file_data=default_data,
                       compute_no_overlap=False):
    data = pymzn.dzn2dict(file_data)
    nb_tasks = data["n_tasks"] + 2
    source = 0
    sink = nb_tasks - 1
    # successors = {source: [i for i in range(1, sink+1)]}
    successors = {i: [] for i in range(sink + 1)}
    successors[source] = [i for i in range(1, sink + 1)]
    mode_details = {t: {1: {}} for t in range(sink + 1)}

    needed_res = {i: [] for i in range(1, data["n_resources"] + 1)}

    needed_right_form = {t: {"R1": set(), "R2": set()}
                         for t in range(sink+1)}
    forbidden_right_form = {t: {"R1": set(), "R2": set()}
                            for t in range(sink + 1)}
    for k in range(len(data["resource_needed"])):
        id_task = k+1
        if len(data["resource_needed"][k]) > 0:
            if max(data["resource_needed"][k]) < 29:
                r = "R1"
                offset = -1
            else:
                r = "R2"
                offset = -29
            needed_right_form[id_task][r].update([x+offset for x in data["resource_needed"][k]])
        if len(data["resource_forbidden"][k]) > 0:
            if max(data["resource_forbidden"][k]) < 29:
                r = "R1"
                offset = -1
            else:
                r = "R2"
                offset = -29
            forbidden_right_form[id_task][r].update([x+offset for x in data["resource_forbidden"][k]])

    for k in range(len(data["duration"])):
        for nr in data["resource_needed"][k]:
            needed_res[nr].append(k + 1)

    for k in range(1, sink):
        successors[k] += [sink]
    mode_details[source][1]["duration"] = 0
    mode_details[sink][1]["duration"] = 0

    for k in range(len(data["duration"])):
        mode_details[k + 1][1]["duration"] = data["duration"][k]

    for k in range(len(data["predecessor"])):
        for pred in data["predecessor"][k]:
            successors[pred] += [k + 1]
    ressource = ["R1", "R2"]
    for k in mode_details:
        for r in ressource:
            mode_details[k][1][r] = 0
    corrected_ressource_needed = {t: {} for t in range(sink + 1)}
    for k in range(len(data["resource_forbidden"])):
        if len(data["resource_forbidden"][k]) == 0:
            print("HEY")
        else:
            min_res_forbidden = min(data["resource_forbidden"][k])
            if min_res_forbidden < 29:
                # print("R2", data["number_resource_needed"][k])
                mode_details[k + 1][1]["R2"] = data["number_resource_needed"][k]
                mode_details[k + 1][1]["R1"] = 0
                corrected_ressource_needed[k + 1]["R2"] = [x - 29 for x in data["resource_needed"][k]]
            else:
                # print("R1", data["number_resource_needed"][k])
                mode_details[k + 1][1]["R1"] = data["number_resource_needed"][k]
                mode_details[k + 1][1]["R2"] = 0
                corrected_ressource_needed[k + 1]["R1"] = [x - 1 for x in data["resource_needed"][k]]
    model = RCPSPModel(resources={"R1": 28, "R2": 7},
                       non_renewable_resources=[], mode_details=mode_details,
                       successors=successors,
                       horizon=18700,
                       tasks_list=list(range(sink + 1)),
                       source_task=source,
                       sink_task=sink)
    model.extra = (needed_right_form, forbidden_right_form)
    if compute_no_overlap:
        graph = model.compute_graph(compute_predecessors=True)
        full_succ = graph.full_successors
        not_overlap = set()
        for res in needed_res:
            if len(needed_res[res]) > 0:
                for j in range(len(needed_res[res])):
                    for k in range(j + 1, len(needed_res[res])):
                        tup = (needed_res[res][j], needed_res[res][k])
                        if tup in not_overlap:
                            continue
                        if tup[0] in full_succ[tup[1]] or tup[1] in full_succ[tup[0]]:
                            continue
                        not_overlap.add(tup)
        return model, not_overlap, needed_res
    return model


def run():
    model = create_rcpsp_model(default_data)
    solver = CPOptSolver(rcpsp_model=model)
    solver.init_model_with_alternative(resource_needed=model.extra[0],
                                       resource_forbidden=model.extra[1],
                                       include_instantaneous_resource_consumption=False,
                                       resource_allocation_break=True)
    # solver.init_model(resource_needed=model.extra[0],
    #                   resource_forbidden=model.extra[1])
    params_cp = ParametersCP.default()
    res = solver.solve(parameters_cp=params_cp)
    best_sol = res.list_solution_fits[0][0]
    print(best_sol.alloc)

    def check_solution(rcpsp_model: RCPSPModel,
                       solution: RCPSPSolution,
                       resource_forbidden,
                       resource_needed):
        alloc: Dict[str, Dict[int, List[Hashable]]] = solution.alloc
        max_time = solution.get_end_time(rcpsp_model.sink_task)
        min_time = 0
        resources = rcpsp_model.resources_list
        with_calendar = rcpsp_model.is_varying_resource()
        array_ressource_usage = {
            resources[i]: {
                "activity": np.zeros(
                    (
                        max_time - min_time + 1,
                        rcpsp_model.get_max_resource_capacity(resources[i]),
                    )
                ),
                "binary_activity": np.zeros(
                    (
                        max_time - min_time + 1,
                        rcpsp_model.get_max_resource_capacity(resources[i]),
                    )
                ),
                "total_activity": np.zeros(
                    rcpsp_model.get_max_resource_capacity(resources[i])
                ),
                "activity_last_n_hours": np.zeros(
                    (
                        max_time - min_time + 1,
                        rcpsp_model.get_max_resource_capacity(resources[i]),
                    )
                ),
                "boxes_time": [],
            }
            for i in range(len(resources))}
        for res in alloc:
            for index in alloc[res]:
                for task in alloc[res][index]:
                    st = solution.get_start_time(task)
                    end = solution.get_end_time(task)
                    array_ressource_usage[res]["activity"][st:end, index] = rcpsp_model.index_task[task]
                    array_ressource_usage[res]["binary_activity"][st:end, index] += 1
                    if end>st:
                        assert np.max(array_ressource_usage[res]["binary_activity"][st:end, index])<=1
                    array_ressource_usage[res]["total_activity"][index] += end-st
                    array_ressource_usage[res]["boxes_time"] += [
                    [
                        (index - 0.25, st + 0.01, task),
                        (index - 0.25, end - 0.01, task),
                        (index + 0.25, end - 0.01, task),
                        (index + 0.25, st + 0.01, task),
                        (index - 0.25, st + 0.01, task),
                    ]
                    ]
        for task in resource_needed:
            for r in resource_needed[task]:
                for index in resource_needed[task][r]:
                    assert task in alloc[r][index]
        for task in resource_forbidden:
            for r in resource_forbidden[task]:
                for index in resource_forbidden[task][r]:
                    assert task not in alloc[r][index]
        nb_ressource_used = sum([len(alloc[r][i])>0 for r in alloc for i in alloc[r]])
        print(nb_ressource_used)
        return array_ressource_usage

    array_ressource_usage = check_solution(rcpsp_model=model,
                                           solution=best_sol,
                                           resource_needed=model.extra[0],
                                           resource_forbidden=model.extra[1])
    nb_ressource_used = 0
    for res in array_ressource_usage:
        nb_non_zeros = np.sum(array_ressource_usage[res]["total_activity"] > 0)
        print(f"Working on {res}", np.where(array_ressource_usage[res]["total_activity"] > 0))
        nb_ressource_used += nb_non_zeros
        print(res, nb_non_zeros)
    print("Overall, ", nb_ressource_used)
    # plot_ressource_view(rcpsp_model=model, rcpsp_sol=best_sol)
    custom_plot_resource_individual_gantt(rcpsp_model=model,
                                          rcpsp_sol=best_sol,
                                          array_ressource_usage=array_ressource_usage)
    plt.show()

if __name__ == "__main__":
    run()