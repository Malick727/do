from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from typing import Dict, Set, List
import logging
# logging.basicConfig(level=logging.DEBUG)
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPSolution
from custom_utils import compute_resource_consumption_function, compute_res_consumption
import pymzn
from discrete_optimization.generic_tools.graph_api import Graph
from discrete_optimization.rcpsp.solver.cp_solvers import CP_MRCPSP_MZN, CP_RCPSP_MZN, \
    CPSolverName, RCPSPSolCP, ParametersCP
import numpy as np
import os

this_dir = os.path.dirname(os.path.abspath(__file__))
default_data = os.path.join(this_dir, "data/data_example.dzn")

def create_rcpsp_model(file_data=default_data,
                       compute_no_overlap=False):
    data = pymzn.dzn2dict(file_data)  # "/Users/poveda_g/Downloads/data (3).dzn")
    nb_tasks = data["n_tasks"] + 2
    source = 0
    sink = nb_tasks - 1
    # successors = {source: [i for i in range(1, sink+1)]}
    successors = {i: [] for i in range(sink + 1)}
    successors[source] = [i for i in range(1, sink + 1)]
    mode_details = {t: {1: {}} for t in range(sink + 1)}

    needed_res = {i: [] for i in range(1, data["n_resources"] + 1)}

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


def custom_plot_resource_individual_gantt(
    rcpsp_model: RCPSPModel,
    rcpsp_sol: RCPSPSolution,
    array_ressource_usage,
    resource_types_to_consider: List[str] = None,
    title_figure="",
    x_lim=None,
    fig=None,
    ax=None,
    current_t=None,
):
    from matplotlib.patches import Polygon as pp
    from shapely.geometry import Polygon
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
    resources_list = list(array_ressource_usage.keys())
    if fig is None or ax is None:
        fig, ax = plt.subplots(len(array_ressource_usage), figsize=(10, 5))
        fig.suptitle(title_figure)
        if len(array_ressource_usage) == 1:
            ax = [ax]

    for i in range(len(resources_list)):
        patches = []
        nb_colors = len(sorted_task_by_start) // 2
        colors = plt.cm.get_cmap("hsv", nb_colors)
        for boxe in array_ressource_usage[resources_list[i]]["boxes_time"]:
            polygon = Polygon([(b[1], b[0]) for b in boxe])
            activity = boxe[0][2]
            x, y = polygon.exterior.xy
            ax[i].plot(x, y, zorder=-1, color="b")
            patches.append(
                pp(
                    xy=polygon.exterior.coords,
                    facecolor=colors((activity - 1) % nb_colors),
                )
            )
        p = PatchCollection(
            patches,
            match_original=True,
            alpha=0.4,
        )
        ax[i].add_collection(p)
        ax[i].set_title(resources_list[i])
        if x_lim is None:
            ax[i].set_xlim((min_time, max_time))
        else:
            ax[i].set_xlim(x_lim)
        try:
            ax[i].set_ylim((-0.5, rcpsp_model.resources[resources_list[i]]))
            ax[i].set_yticks(range(rcpsp_model.resources[resources_list[i]]))
            ax[i].set_yticklabels(
                tuple([j for j in range(rcpsp_model.resources[resources_list[i]])]),
                fontdict={"size": 7},
            )
        except:
            m = max(rcpsp_model.resources[resources_list[i]])
            ax[i].set_ylim((-0.5, m))
            ax[i].set_yticks(range(m))
            ax[i].set_yticklabels(tuple([j for j in range(m)]), fontdict={"size": 7})

        ax[i].grid(True)
        if current_t is not None:
            ax[i].axvline(x=current_t, label="pyplot vertical line", color="r", ls="--")
    ax[-1].set_xlabel("Timestep")
    return fig


def run(do_makespan_optim=False,
        do_resource_optim=False,
        file_data=default_data,
        file_output="best_sol.pk"):

    data = pymzn.dzn2dict(file_data)
    nb_tasks = data["n_tasks"]+2
    source = 0
    sink = nb_tasks - 1
    # successors = {source: [i for i in range(1, sink+1)]}
    successors = {i: [] for i in range(sink+1)}
    successors[source] = [i for i in range(1, sink+1)]
    mode_details = {t: {1: {}} for t in range(sink+1)}

    needed_res = {i: [] for i in range(1, data["n_resources"]+1)}

    for k in range(len(data["duration"])):
        for nr in data["resource_needed"][k]:
            needed_res[nr].append(k+1)

    for k in range(1, sink):
        successors[k] += [sink]
    mode_details[source][1]["duration"] = 0
    mode_details[sink][1]["duration"] = 0

    for k in range(len(data["duration"])):
        mode_details[k+1][1]["duration"] = data["duration"][k]

    for k in range(len(data["predecessor"])):
        for pred in data["predecessor"][k]:
            successors[pred] += [k+1]
    ressource = ["R1", "R2"]
    for k in mode_details:
        for r in ressource:
            mode_details[k][1][r] = 0
    corrected_ressource_needed = {t: {} for t in range(sink+1)}
    for k in range(len(data["resource_forbidden"])):
        if len(data["resource_forbidden"][k]) == 0:
            print("HEY")
        else:
            min_res_forbidden = min(data["resource_forbidden"][k])
            if min_res_forbidden < 29:
                # print("R2", data["number_resource_needed"][k])
                mode_details[k+1][1]["R2"] = data["number_resource_needed"][k]
                mode_details[k+1][1]["R1"] = 0
                corrected_ressource_needed[k+1]["R2"] = [x-29 for x in data["resource_needed"][k]]
            else:
                # print("R1", data["number_resource_needed"][k])
                mode_details[k + 1][1]["R1"] = data["number_resource_needed"][k]
                mode_details[k + 1][1]["R2"] = 0
                corrected_ressource_needed[k+1]["R1"] = [x-1 for x in data["resource_needed"][k]]

    model = RCPSPModel(resources={"R1": 28, "R2": 7},
                       non_renewable_resources=[], mode_details=mode_details,
                       successors=successors,
                       horizon=18700,
                       tasks_list=list(range(sink+1)),
                       source_task=source,
                       sink_task=sink)

    func_resource_consumption = compute_resource_consumption_function(rcpsp_problem=model)

    graph = model.compute_graph(compute_predecessors=True)
    full_pred = graph.full_predecessors
    full_succ = graph.full_successors
    not_overlap = set()
    for res in needed_res:
        if len(needed_res[res]) > 0:
            for j in range(len(needed_res[res])):
                for k in range(j+1, len(needed_res[res])):
                    tup = (needed_res[res][j], needed_res[res][k])
                    if tup in not_overlap:
                        continue
                    if tup[0] in full_succ[tup[1]] or tup[1] in full_succ[tup[0]]:
                        continue
                    not_overlap.add(tup)
        print("Resource ", res)

    print("nb not overlap ", len(not_overlap))
    dummy_sol = model.get_dummy_solution()
    # print(dummy_sol)
    print(model.evaluate(dummy_sol))

    if do_makespan_optim:
        cp_solver = CP_RCPSP_MZN(rcpsp_model=model,
                                 cp_solver_name=CPSolverName.CPOPT)
        cp_solver.init_model(output_type=True,
                             model_type="single",
                             max_time=model.horizon)
        for t1, t2 in not_overlap:
            cp_solver.instance.add_string(f"constraint disjunctive([s[{t1+1}],s[{t2+1}]], [d[{t1+1}], d[{t2+1}]]);\n")
        params_cp = ParametersCP.default()
        params_cp.time_limit = 1000
        params_cp.free_search = False
        params_cp.multiprocess = True
        params_cp.nb_process = 4
        params_cp.optimisation_level = 1
        logging.basicConfig(level=logging.DEBUG)
        res = cp_solver.solve(parameters_cp=params_cp)
    if do_resource_optim:
        cp_solver = CP_RCPSP_MZN(rcpsp_model=model,
                                 cp_solver_name=CPSolverName.CHUFFED)
        cp_solver.init_model(output_type=True,
                             model_type="single_resource",
                             max_time=model.horizon)
        for t1, t2 in not_overlap:
            cp_solver.instance.add_string(f"constraint disjunctive([s[{t1+1}],s[{t2+1}]], [d[{t1+1}], d[{t2+1}]]);\n")
        params_cp = ParametersCP.default()
        params_cp.time_limit = 10000
        params_cp.free_search = False
        params_cp.multiprocess = True
        params_cp.nb_process = 4
        params_cp.optimisation_level = 1
        logging.basicConfig(level=logging.DEBUG)
        res = cp_solver.solve(parameters_cp=params_cp)
        sol_fits = res.list_solution_fits
        res_consumption = []
        max_res_consumption = []
        for s, f in sol_fits:
            rr = compute_res_consumption(problem=model,
                                         solution=s,
                                         function_numba=func_resource_consumption)
            m = sum([max(rr[k]) for k in rr])
            print("nb ressource used : ", m)
            max_res_consumption += [m]

        best_sol, f = res.list_solution_fits[-1]
        best_sol: RCPSPSolution = best_sol
        from discrete_optimization.rcpsp.rcpsp_utils import plot_resource_individual_gantt, \
            plt, plot_ressource_view
        fig, ax = plt.subplots(1)
        for r in rr:
            ax.plot(rr[r], label=str(r))
        ax.legend()
        import json
        import pickle
        pickle.dump(best_sol, open(file_output, "wb"))
        json.dump(best_sol.rcpsp_schedule, open(file_output+".json", "w"), indent=2)
        array_ressource_usage = compute_individual_worker_schedule(rcpsp_sol=best_sol, rcpsp_model=model,
                                                                   resource_needed=corrected_ressource_needed)
        plot_resource_individual_gantt(rcpsp_model=model,
                                       rcpsp_sol=best_sol)
        custom_plot_resource_individual_gantt(rcpsp_model=model, rcpsp_sol=best_sol,
                                              array_ressource_usage=array_ressource_usage)
        nb_ressource_used = 0
        for res in array_ressource_usage:
            nb_non_zeros = np.sum(array_ressource_usage[res]["total_activity"] > 0)
            nb_ressource_used += nb_non_zeros
            print(res, nb_non_zeros)
        print("Overall, ", nb_ressource_used)
        # plot_ressource_view(rcpsp_model=model, rcpsp_sol=best_sol)
        print(max_res_consumption)
        plt.show()

    if False:
        cp_solver = CP_MRCPSP_MZN(rcpsp_model=model,
                                  cp_solver_name=CPSolverName.CPOPT)
        cp_solver.init_model(output_type=True,
                             model_type="multi-resource-feasibility",
                             max_time=model.horizon)
        params_cp = ParametersCP.default()
        params_cp.time_limit = 10000
        params_cp.free_search = True
        params_cp.multiprocess = True
        params_cp.nb_process = 4
        params_cp.optimisation_level = 1
        res = cp_solver.solve(parameters_cp=params_cp)
    return corrected_ressource_needed


def postpro(file_data=default_data):
    import pickle
    corrected_ressource_needed = run(file_data=file_data)
    best_sol: RCPSPSolution = pickle.load(open("best_sol.pk", "rb"))
    model = best_sol.problem
    array_ressource_usage = compute_individual_worker_schedule(rcpsp_sol=best_sol, rcpsp_model=model,
                                                               resource_needed=corrected_ressource_needed)
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


def run_cpopt():
    model, no_overlaps, needed_res = create_rcpsp_model(compute_no_overlap=True)
    from cpo_solver import CPOptSolver, no_overlap
    solver = CPOptSolver(rcpsp_model=model)
    solver.init_resource()
    for res in needed_res:
        if len(needed_res[res]) > 0:
            solver.model.add(no_overlap([solver.variables["tasks"][t]
                                         for t in needed_res[res]]))
    #for t1, t2 in no_overlaps:
    #    solver.model.add(no_overlap([solver.variables["tasks"][t1], solver.variables["tasks"][t2]]))
    params_cp = ParametersCP.default()
    solver.solve(parameters_cp=params_cp)


if __name__ == "__main__":
    run_cpopt()


