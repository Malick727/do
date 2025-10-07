# Trying to solve the latest problem provided by EAD :
import sys
import os
this_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(this_folder, "../../"))

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick

os.environ["SDKROOT"] = "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
from multibatching.parse_config_data import parse_json_to_problem
from multibatching.problem import compute_graph_for_product, Product
from multibatching.solvers.cpsat import (CpsatMultibatchingSolver,
                                         CpsatBendersMultibatchingSolver,
                                         ModelingMultiBatch,
                                         CpsatPackingSubproblem)
from multibatching.solvers.packing_subproblem import (CpsatPackingSubproblem,
                                                      PackingViaBinPacking,
                                                      GreedyPackingForMultibatching)

from multibatching.solvers.netx import NetxMultibatchingSolver
from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
from multibatching.solvers.cpmpy import CpmpyMultibatchingSolver
from discrete_optimization.generic_tools.cp_tools import ParametersCp, StatusSolver
import json
import os
from examples.multibatching.instance2 import init_instance

from multibatching.utils import basic_sanity_check, plot_solution_graph, plotly_solution_graph
import logging
logging.basicConfig(level=logging.INFO)


def parse_problem(fix_problem: bool = False):
    this_folder = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(this_folder,
                               "../../multibatching/datalocal/config-local.json")
    d = json.load(open(config_file, "r"))
    problem = parse_json_to_problem(d)

    # Temporary fix = section2 transportable by any means..
    if fix_problem:
        problem.products[3].valid_transports = set(problem.transport_types)

    return problem

def run_networkx():
    problem = init_instance()
    solver = NetxMultibatchingSolver(problem)
    res = solver.solve()
    sol = res[-1][0]
    print(problem.evaluate(sol))
    print(sum(problem.evaluate(sol).values()))
    print(problem.evaluate(sol), problem.satisfy(sol))
    plot_solution_graph(problem=problem, solution=sol)
def run_oe():
    from multibatching.solvers.oe.solver_2 import OpenEvolveMultibatchingSolver
    problem = parse_problem()
    solver = OpenEvolveMultibatchingSolver(problem)
    res = solver.solve()
    sol = res[-1][0]
    print(problem.evaluate(sol))
    print(sum(problem.evaluate(sol).values()))
    print(problem.evaluate(sol), problem.satisfy(sol))

def run_cp_2steps():
    problem = parse_problem()
    # problem = init_instance()
    solver = CpsatMultibatchingSolver(problem=problem,
                                      scaling_factor=1)
    solver.init_model(modeling=ModelingMultiBatch.FLOW,
                      detailed_capacity_constraint=True)
    p = ParametersCp.default_cpsat()
    p.nb_process = 16
    result = solver.solve(callbacks=[],
                          time_limit=100,
                          parameters_cp=p,
                          ortools_cpsat_solver_kwargs=dict(log_search_progress=True))
    subsolver = CpsatPackingSubproblem(problem=problem, scaling_factor=1)
    subsolver.init_model(solution=result[-1][0])
    res_ = subsolver.solve(parameters_cp=p,
                           time_limit=10,
                           ortools_cpsat_solver_kwargs={"log_search_progress": False})
    sol = res_[-1][0]
    print(problem.evaluate(sol))
    print(sum(problem.evaluate(sol).values()))
    print(problem.evaluate(sol), problem.satisfy(sol))
    # plot_solution_graph(problem=problem, solution=sol)
    for i in range(len(problem.products)):
        plotly_solution_graph(solution=sol,
                              problem=problem,
                              use_logical_layout=False,
                              products_to_plot=[problem.products[i]],
                              explode_flows=True)


def run_cp_benders():
    problem = parse_problem(fix_problem=False)
    solver = CpsatBendersMultibatchingSolver(problem=problem,
                                             scaling_factor=1)
    solver.init_model(detailed_capacity_constraint=False)
    p = ParametersCp.default_cpsat()
    p.nb_process = 32
    subbrick = SubBrick(cls=CpsatPackingSubproblem,
                        kwargs={"time_limit": 20, "parameters_cp": p,
                                "ortools_cpsat_solver_kwargs":
                                {"log_search_progress": True}})
    # subbrick = SubBrick(cls=PackingViaBinPacking,
    #                     kwargs={"time_limit_per_link": 3})
    subbrick = SubBrick(cls=GreedyPackingForMultibatching,
                        kwargs={})
    result = solver.solve(callbacks=[],
                          nb_iteration=12,
                          params_solver_master=
                          {"time_limit": 50,
                           "parameters_cp": p,
                           "ortools_cpsat_solver_kwargs": {"log_search_progress": True}},
                          packing_solver=subbrick)
    sol, _ = result.get_best_solution_fit(problem)
    print(problem.evaluate(sol))
    print(sum(problem.evaluate(sol).values()))
    print(problem.evaluate(sol), problem.satisfy(sol))
    if True:
        for i in range(min(5, len(problem.products))):
            plotly_solution_graph(solution=sol,
                                  problem=problem,
                                  use_logical_layout=False,
                                  products_to_plot=[problem.products[i]],
                                  explode_flows=True)


def run_cpmpy():
    problem = parse_problem(fix_problem=True)
    solver = CpmpyMultibatchingSolver(problem=problem,
                                      scaling_factor=1,
                                      solver_name="ortools")
    print("start init")
    solver.init_model(modeling=ModelingMultiBatch.FLOW,
                      detailed_capacity_constraint=False)
    print("end init")
    p = ParametersCp.default_cpsat()
    p.nb_process = 16
    result = solver.solve(callbacks=[ObjectiveLogger(step_verbosity_level=logging.INFO,
                                                     end_verbosity_level=logging.INFO)],
                          time_limit=30,
                          parameters_cp=p,
                          #verbose=True
                          )


def run_explanation():
    from discrete_optimization.generic_tools.cpmpy_tools import CpmpyExplainUnsatMethod
    problem = parse_problem(fix_problem=False)
    solver = CpmpyMultibatchingSolver(problem=problem, scaling_factor=1,
                                      solver_name="ortools")
    print("start init")
    solver.init_model(modeling=ModelingMultiBatch.FLOW,
                      detailed_capacity_constraint=False)
    solver.solve(parameters_cp=ParametersCp.default_cpsat(),
                 time_limit=10)
    soft = [mc for mc in solver.meta_constraints if mc.metadata["type"] == "flow"]
    hard = [mc for mc in solver.meta_constraints if mc.metadata["type"] == "background"]
    print(len(soft), len(hard))
    # Obvious stuff
    for s in soft:
        for c in s.constraints:
            if isinstance(c, bool) and not c:
                print(c)
                print(s.metadata)
    explanation = solver.explain_unsat_meta(soft=soft,
                                            hard=hard,
                                            cpmpy_method=CpmpyExplainUnsatMethod.mus,
                                            solver="ortools")
    for e in explanation:
        print(e.metadata)


def run_cp_detailed_trips():
    problem = parse_problem()
    solver = CpsatMultibatchingSolver(problem=problem,
                                      scaling_factor=1)
    solver.init_model(modeling=ModelingMultiBatch.UNIT_FLOW,
                      detailed_capacity_constraint=False,
                      max_trips=5)
    p = ParametersCp.default_cpsat()
    p.nb_process = 16
    result = solver.solve(callbacks=[],
                          time_limit=10000,
                          parameters_cp=p,
                          ortools_cpsat_solver_kwargs=dict(log_search_progress=True))
    sol = result[-1][0]
    print(problem.evaluate(sol))
    print(sum(problem.evaluate(sol).values()))
    print(problem.evaluate(sol), problem.satisfy(sol))
    # plot_solution_graph(problem=problem, solution=sol)
    for i in range(len(problem.products)):
        plotly_solution_graph(solution=sol,
                              problem=problem,
                              use_logical_layout=False,
                              products_to_plot=[problem.products[i]],
                              explode_flows=True)


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
    result = solver.solve(callbacks=[ObjectiveLogger(step_verbosity_level=logging.INFO,
                                                     end_verbosity_level=logging.INFO)],
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
                                             product=problem.products[26])
    print(graph)
    locations = problem.get_supplier_location(problem.products[26])

    available_transport_type = set(problem.products[26].valid_transports)
    print(locations, " supply")
    for location, capa in locations:
        print(location)
        tls = set([tl.transport_type for tl in problem.transport_links if tl.location_l1 == location])
        print(tls)
        intersect = tls.intersection(available_transport_type)
        print(intersect, " possible transport type to go out from ", problem.products[26], " from ", location)
        if len(intersect) == 0:
            print("Problem, you won't be able to build an outgoing flow...")
            print(problem.get_outgoing_tl_from_location(location=location),
                  " here is the outgoing link from the location")
    locations = problem.get_consumer_location(problem.products[26])
    print(locations, " consume")


if __name__ == "__main__":
    run_oe()
    # run_cpmpy()
    # run_explanation()
    # run_cp_2steps()
    # 2637923900.2783985
    # 2686538384.2949004
    run_cp_2steps()
    # 2,637,923,900.2783985
    # 2,686,538,384.2949004
    # Greedy : 2676972529 2647669849.170104
    # Binpacking : 2,637,614,464.8357964 2653396240.1278048 680530479.6346035 2693366369.0273995
    # 2695909540.234205, 2643758582.8569036
    # GlobalPack : 2677042356.087301
    # TotalBinpacking :

    #run_cp_benders()
    # run_cp_2steps()
    # run_cp_benders()
    # run_networkx()
    # run_cp_benders()

