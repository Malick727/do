# Trying to solve the latest problem provided by EAD :

from multibatching.parse_config_data import parse_json_to_problem
from multibatching.problem import compute_graph_for_product, Product
from multibatching.solvers.cpsat import (CpsatMultibatchingSolver, CpsatBendersMultibatchingSolver,
                                         ModelingMultiBatch, CpsatPackingSubproblem)
from discrete_optimization.generic_tools.cp_tools import ParametersCp, StatusSolver
import json
import os

from multibatching.utils import basic_sanity_check, plot_solution_graph
import logging
logging.basicConfig(level=logging.INFO)


def parse_problem():
    this_folder = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(this_folder,
                               "../../multibatching/datalocal/config-local.json")
    d = json.load(open(config_file, "r"))
    problem = parse_json_to_problem(d)
    # Temporary fix = section2 transportable by any means...
    problem.products[3].valid_transports = set(problem.transport_types)

    return problem


def run_cp():
    problem = parse_problem()
    solver = CpsatMultibatchingSolver(problem=problem, scaling_factor=1)
    solver.init_model(modeling=ModelingMultiBatch.FLOW,
                      detailed_capacity_constraint=False)
    p = ParametersCp.default_cpsat()
    p.nb_process = 16
    result = solver.solve(callbacks=[],
                          time_limit=30,
                          parameters_cp=p,
                          ortools_cpsat_solver_kwargs=dict(log_search_progress=True))
    subsolver = CpsatPackingSubproblem(problem=problem, scaling_factor=1)
    subsolver.init_model(solution=result[-1][0])
    res_ = subsolver.solve(parameters_cp=p,
                           time_limit=5,
                           ortools_cpsat_solver_kwargs={"log_search_progress": False})
    sol = res_[-1][0]
    print(problem.evaluate(sol))
    print(sum(problem.evaluate(sol).values()))
    print(problem.evaluate(sol), problem.satisfy(sol))
    plot_solution_graph(problem=problem, solution=sol)



def run_cp_debug():
    this_folder = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(this_folder,
                               "../../multibatching/datalocal/config-local.json")
    d = json.load(open(config_file, "r"))
    problem = parse_json_to_problem(d)
    # basic_sanity_check(problem)
    for i in range(len(problem.products)):
        solver = CpsatMultibatchingSolver(problem=problem)

        solver.init_model(modeling=ModelingMultiBatch.FLOW,
                          detailed_capacity_constraint=False,
                          debug_mode=True,
                          debug_settings=dict(index_products=[i]))
        p = ParametersCp.default_cpsat()
        p.nb_process = 16
        result = solver.solve(callbacks=[],
                              time_limit=3,
                              parameters_cp=p,
                              ortools_cpsat_solver_kwargs=dict(log_search_progress=False))
        if solver.status_solver == StatusSolver.UNSATISFIABLE:
            print("UNSAT with product : ", problem.products[i])
        else:
            print("SAT with product : ", problem.products[i])
        # sol = result[-1][0]
        # print(problem.evaluate(sol), problem.satisfy(sol))


def run_cp_debug_section_2():
    this_folder = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(this_folder,
                               "../../multibatching/datalocal/config-local.json")
    d = json.load(open(config_file, "r"))
    problem = parse_json_to_problem(d)
    # basic_sanity_check(problem)
    i = 2
    solver = CpsatMultibatchingSolver(problem=problem)

    solver.init_model(modeling=ModelingMultiBatch.FLOW,
                      detailed_capacity_constraint=False,
                      debug_mode=True,
                      debug_settings=dict(index_products=[3]))
    p = ParametersCp.default_cpsat()
    p.nb_process = 16
    result = solver.solve(callbacks=[],
                          time_limit=3,
                          parameters_cp=p,
                          ortools_cpsat_solver_kwargs=dict(log_search_progress=False))
    if solver.status_solver == StatusSolver.UNSATISFIABLE:
        print("UNSAT with product : ", problem.products[i])
    else:
        print("SAT with product : ", problem.products[i])
        # sol = result[-1][0]
        # print(problem.evaluate(sol), problem.satisfy(sol))


def run_cp_debug_flows():
    problem = parse_problem()
    graph, flows = compute_graph_for_product(problem=problem,
                                             product=problem.products[3])
    print(graph)
    locations = problem.get_supplier_location(problem.products[3])

    available_transport_type = set(problem.products[3].valid_transports)
    print(locations, " supply")
    for location, capa in locations:
        print(location)
        tls = set([tl.transport_type for tl in problem.transport_links if tl.location_l1 == location])
        print(tls)
        intersect = tls.intersection(available_transport_type)
        print(intersect, " possible transport type to go out from ", problem.products[3], " from ", location)
        if len(intersect) == 0:
            print("Problem, you won't be able to build an outgoing flow...")
            print(problem.get_outgoing_tl_from_location(location=location),
                  " here is the outgoing link from the location")





    locations = problem.get_consumer_location(problem.products[3])
    print(locations, " consume")


if __name__ == "__main__":
    run_cp()

