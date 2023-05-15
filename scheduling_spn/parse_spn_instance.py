import pandas as pd
import os
os.environ["DO_SKIP_MZN_CHECK"] = "1"
from discrete_optimization.rcpsp.rcpsp_parser import parse_file, parse_psplib
this_folder = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(this_folder, "data/")
file_rcpsp = os.path.join(data_folder, "Tablada_FLS.SM")
file_calendar = os.path.join(data_folder, "Availability Matrix.xlsx")
import logging
#logging.basicConfig(level=logging.DEBUG)

def parse_calendar():
    import logging
    logging.basicConfig(level=logging.INFO)
    # Parse Excel file as dictionary
    # Create a Pandas Excel file object
    excel_file = pd.ExcelFile(file_calendar)
    # Parse each sheet into a dictionary where the key is the sheet name and the value is a DataFrame
    data_dict = {}
    for sheet_name in excel_file.sheet_names:
        data_dict[sheet_name] = excel_file.parse(sheet_name)
    name_sheet = list(data_dict.keys())[0]
    dataf = data_dict[name_sheet]
    #print(dataf.T)
    d = dataf.T
    #print(d[0][1:])
    model = parse_file(file_path=file_rcpsp)
    print(model)
    print(model.resources)
    #model.horizon = len(d[0][1:])
    import numpy as np
    calendars = {i: np.array(d[i][1:]) for i in range(len(model.resources_list))}
    binary_calendars = {i: np.array(calendars[i] > 0, dtype=int) for i in calendars}
    cumulated_calendars = {i: np.cumsum(binary_calendars[i]) for i in calendars}
    #for k in cumulated_calendars:
    #    jj = next(j for j in range(len(cumulated_calendars[k])) if cumulated_calendars[k][j] >= 776)
    #    print("jj", jj)
    for j in binary_calendars:
        for k in binary_calendars:
            pass
            #print("Max : ", max(binary_calendars[k]-binary_calendars[j]))
    from discrete_optimization.rcpsp.solver.cpm import CPM
    from rcpsp_breaks.rcpsp_breaks_cpopt import CPOptRCPSPBreakSolver, ParametersCP
    from rcpsp_breaks.rcpsp_breaks_ortools import OrtoolsRCPSPBreakSolver
    cpm = CPM(model)
    cpm.run_classic_cpm()
    print(cpm.map_node[model.sink_task])
    for i in range(len(model.resources_list)):
        r = model.resources_list[i]
        model.resources[r] = np.empty((model.horizon))
        #model.resources[r][:] = max(d[i][1:])
        model.resources[r][:len(d[i][1:])] = np.array(d[i][1:], dtype=int)
        model.resources[r][len(d[i][1:]):2*len(d[i][1:])] = np.array(d[i][1:], dtype=int)
        model.resources[r][2*len(d[i][1:]):3*len(d[i][1:])] = np.array(d[i][1:], dtype=int)
        model.resources[r][3*len(d[i][1:]):4*len(d[i][1:])] = np.array(d[i][1:], dtype=int)
        model.resources[r][4*len(d[i][1:]):5*len(d[i][1:])] = np.array(d[i][1:], dtype=int)
        model.resources[r][5*len(d[i][1:]):] = int(max(d[i][1:]))

    # sanity check
    for t in model.mode_details:
        if model.mode_details[t][1]["duration"] == 0:
            for r in model.mode_details[t][1]:
                model.mode_details[t][1][r] = 0
        res = [r for r in model.resources_list if model.mode_details[t][1][r]>0]
        print(len(res))
        print("task ", t,  res)
        #assert len(res) <= 1
        #print(max(d[i][1:]))
    model.is_calendar = True
    model.update_functions()

    cpo_solver = CPOptRCPSPBreakSolver(model)
    params_cp = ParametersCP.default()
    result = cpo_solver.solve(params_cp)
    solution = result.get_best_solution()
    if False:
        ortools_solver = OrtoolsRCPSPBreakSolver(model)
        params_cp = ParametersCP.default()
        ortools_solver.init_model_durations_indicator_method()
        result = ortools_solver.solve(params_cp, add_solution_callback=True)
        solution = result.get_best_solution()

    from discrete_optimization.rcpsp.rcpsp_utils import plot_task_gantt, plt, plot_ressource_view
    plot_task_gantt(rcpsp_model=model, rcpsp_sol=solution)
    plot_ressource_view(rcpsp_model=model, rcpsp_sol=solution)
    #plt.show()
    from discrete_optimization.rcpsp.rcpsp_model_preemptive import get_rcpsp_modelp_preemptive
    from discrete_optimization.rcpsp.plots.rcpsp_utils_preemptive import plot_task_gantt, plt
    dummy_solution = model.get_dummy_solution()
    preemptive_model = get_rcpsp_modelp_preemptive(model)

    #print(dummy_solution)
    #print(model.evaluate(dummy_solution))

    dummy_solution = preemptive_model.get_dummy_solution()
    print(dummy_solution)
    print(preemptive_model.evaluate(dummy_solution))
    from cpopt_solver import CPOSolverRCPSP

    cp = CPOSolverRCPSP(preemptive_model)
    cp.init_model(add_resource_constraint=True,
                  add_worker_availability_constraint=True,
                  add_alternative_worker_constraint=True,
                  same_worker_by_task=False,
                  create_worker_meta_task=False,
                  nb_preemption=40,
                  min_duration_preemption=None,
                  symettry_breaking_subtasks=True,
                  strictly_disjunctive_subtasks=True,
                  force_minimal_preemption=False,
                  add_penalty_on_preemption=True,
                  include_generalized_precedence_constraints=False,
                  soft_generalized_precedence_constraints=False,
                  add_partially_preemptive=True,
                  add_skills_constraints=True,
                  include_search=False)
    params_cp.time_limit = 10000
    #cp.set_warm_start(None, nb_iteration=10)
    cp.solve(parameters_cp=params_cp)
    #import networkx as nx
    #import matplotlib.pyplot as plt
    #if False:
    #    for r in preemptive_model.resources:
    #        fig, ax = plt.subplots(1)
    #        ax.plot(preemptive_model.resources[r])
    #        ax.set_title(f"Res {r}")
    plt.show()
    #pos = nx.spring_layout(model.graph.graph_nx)
    #nx.draw(model.graph.graph_nx, pos=pos, with_labels=True)
    #plot_task_gantt(rcpsp_model=preemptive_model, rcpsp_sol=dummy_solution)
    #plt.show()
    #from discrete_optimization.rcpsp.solver.ls_solver import LS_RCPSP_Solver
    #solver = LS_RCPSP_Solver(preemptive_model)
    #res = solver.solve(nb_iteration_max=10000, nb_iteration_no_improvement=100)
    #print(preemptive_model.evaluate(res.get_best_solution()))



if __name__ == "__main__":
    parse_calendar()