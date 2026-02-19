# Trying to solve the latest problem provided by EAD :
import pickle
import random

import sys
import os
this_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(this_folder, "../../"))

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
import plotly.io as pio
pio.renderers.default = "browser"
os.environ["SDKROOT"] = "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk"
from multibatching.parse_config_data import parse_json_to_problem
from multibatching.problem import compute_graph_for_product, Product, PackingTransport, MultibatchingSolution, \
    MultibatchingProblem
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
#from examples_do_usecases.multibatching.instance import init_instance
from examples_do_usecases.multibatching.instance2 import init_instance


from multibatching.utils import basic_sanity_check, plot_solution_graph, plotly_solution_graph
import logging
logging.basicConfig(level=logging.INFO)


def parse_problem(fix_problem: bool = False,
                  harbor_constraint: bool = False,
                  scale_co2: float = 1):
    this_folder = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(this_folder,
                               "../../multibatching/datalocal/config-local.json")
    d = json.load(open(config_file, "r"))
    problem = parse_json_to_problem(d, harbor_constraint=harbor_constraint,
                                    scale_co2=scale_co2)

    # Temporary fix = section2 transportable by any means..
    if fix_problem:
        problem.products[3].valid_transports = set(problem.transport_types)


    for product in problem.products:
        size = product.size
        product.valid_transports = set({t for t in product.valid_transports
                                        if t.capacity >= size})
    return problem

def run_networkx():
    problem = init_instance()
    solver = NetxMultibatchingSolver(problem)
    res = solver.solve()
    sol = res[-1][0]
    print(problem.evaluate(sol))
    print(sum(problem.evaluate(sol).values()))
    print(problem.evaluate(sol), problem.satisfy(sol))

    subsolver = CpsatPackingSubproblem(problem=problem, scaling_factor=1)
    subsolver.init_from_solution(solution=sol)
    subsolver.init_model(solution=sol)
    p = ParametersCp.default_cpsat()
    res_ = subsolver.solve(parameters_cp=p,
                           time_limit=10,
                           ortools_cpsat_solver_kwargs={"log_search_progress": True})
    sol = res_[-1][0]
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

def run_cpoptal_2steps():
    # problem = parse_problem()
    # problem = init_instance()
    from multibatching.solvers.optal import OptalMultibatchingSolver
    import optal_py_solvers as cp
    problem: MultibatchingProblem = parse_problem()
    solver = OptalMultibatchingSolver(problem=problem,
                                      scaling_factor=1/100000)
    solver.init_model(modeling=ModelingMultiBatch.FLOW,
                      detailed_capacity_constraint=True)
    p = ParametersCp.default_cpsat()
    p.nb_process = 16
    fds_worker = cp.WorkerParameters(
        searchType="FDS",
        noOverlapPropagationLevel=4,
    )
    fdslb_worker = cp.WorkerParameters(
        searchType="FDSLB",
        noOverlapPropagationLevel=4
    )
    result = solver.solve(callbacks=[],
                          time_limit=100,
                          parameters_cp=p,
                          workers=[fds_worker, fdslb_worker])
    subsolver = CpsatPackingSubproblem(problem=problem, scaling_factor=1)
    subsolver.init_from_solution(solution=result[-1][0])
    subsolver.init_model(solution=result[-1][0])
    res_ = subsolver.solve(parameters_cp=p,
                           time_limit=10,
                           ortools_cpsat_solver_kwargs={"log_search_progress": True})
    sol = res_[-1][0]

    #print(problem.evaluate(sol))
    #print(sum(problem.evaluate(sol).values()))
    #print(problem.evaluate(sol), problem.satisfy(sol))
    # plot_solution_graph(problem=problem, solution=sol)
    #for i in range(len(problem.products)):
    #    plotly_solution_graph(solution=sol,
    #                          problem=problem,
    #                          use_logical_layout=False,
    #                          products_to_plot=[problem.products[i]],
    #                          explode_flows=True)



def run_cp_2steps():
    # problem = parse_problem()
    # problem = init_instance()
    problem: MultibatchingProblem = parse_problem()
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
    subsolver.init_from_solution(solution=result[-1][0])
    subsolver.init_model(solution=result[-1][0])
    res_ = subsolver.solve(parameters_cp=p,
                           time_limit=10,
                           ortools_cpsat_solver_kwargs={"log_search_progress": True})
    sol = res_[-1][0]
    print("sol", problem.evaluate(sol).values())
    print(problem.evaluate(sol))
    print(sum(problem.evaluate(sol).values()))
    print(problem.evaluate(sol), problem.satisfy(sol))
    plot_solution_graph(problem=problem, solution=sol)
    for i in range(len(problem.products)):
        plotly_solution_graph(solution=sol,
                              problem=problem,
                              use_logical_layout=False,
                              products_to_plot=[problem.products[i]],
                              explode_flows=True)

def run_cp_2steps_and_complete():
    # problem = parse_problem()
    # problem = init_instance()
    problem = parse_problem()
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
    subsolver.init_from_solution(solution=result[-1][0])
    subsolver.init_model(solution=result[-1][0])
    res_ = subsolver.solve(parameters_cp=p,
                           time_limit=10,
                           ortools_cpsat_solver_kwargs={"log_search_progress": True})
    sol = res_[-1][0]

    complete_solver = CpsatMultibatchingSolver(problem=problem)
    complete_solver.init_model(modeling=ModelingMultiBatch.UNIT_FLOW)
    complete_solver.set_warm_start(sol)
    result = complete_solver.solve(parameters_cp=p,
                                   time_limit=300,
                                   ortools_cpsat_solver_kwargs={"log_search_progress": True})
    sol, _ = result.get_best_solution_fit()
    print(problem.evaluate(sol))
    print(sum(problem.evaluate(sol).values()))
    print(problem.evaluate(sol), problem.satisfy(sol))


def run_cp_singlebatch():
    problem = parse_problem()
    solver = CpsatMultibatchingSolver(problem=problem,
                                      scaling_factor=1)
    solver.init_model(modeling=ModelingMultiBatch.FLOW,
                      single_batching=True)
    p = ParametersCp.default_cpsat()
    p.nb_process = 32
    result = solver.solve(callbacks=[],
                          time_limit=1000,
                          parameters_cp=p,
                          ortools_cpsat_solver_kwargs=
                          dict(log_search_progress=True))
    sol = result.get_best_solution()
    print(problem.evaluate(sol),
          "\nSatisfy ", problem.satisfy(sol), "\n",
          sum(problem.evaluate(sol).values()))

    greedy_ = GreedyPackingForMultibatching(problem=problem)
    greedy_.init_from_solution(sol)
    res = greedy_.solve(single_batching=True)
    sol_postpro = res[-1][0]
    print(problem.evaluate(sol),
          "\nSatisfy ", problem.satisfy(sol), "\n",
          sum(problem.evaluate(sol).values()))
    print(problem.evaluate(sol_postpro),
          "\nSatisfy ", problem.satisfy(sol_postpro), "\n",
          sum(problem.evaluate(sol_postpro).values()))
    print('hey')
    import pickle
    pickle.dump(sol, open("res_data/solution_single_batch.pk", "wb"))

def run_cp_singlebatch_ws():
    import pickle
    problem = parse_problem()
    solver = CpsatMultibatchingSolver(problem=problem,
                                      scaling_factor=1)
    solver.init_model(modeling=ModelingMultiBatch.FLOW,
                      single_batching=True)
    sol = pickle.load(open("solution_multibatching.pk", 'rb'))
    solver.set_warm_start_flow(sol)
    p = ParametersCp.default_cpsat()
    p.nb_process = 32
    result = solver.solve(callbacks=[],
                          time_limit=1000,
                          parameters_cp=p,
                          ortools_cpsat_solver_kwargs=
                          dict(log_search_progress=True))
    sol = result.get_best_solution()
    print(problem.evaluate(sol),
          "\nSatisfy ", problem.satisfy(sol), "\n",
          sum(problem.evaluate(sol).values()))

    greedy_ = GreedyPackingForMultibatching(problem=problem)
    greedy_.init_from_solution(sol)
    res = greedy_.solve(single_batching=True)
    sol_postpro = res[-1][0]
    print(problem.evaluate(sol),
          "\nSatisfy ", problem.satisfy(sol), "\n",
          sum(problem.evaluate(sol).values()))
    print(problem.evaluate(sol_postpro),
          "\nSatisfy ", problem.satisfy(sol_postpro), "\n",
          sum(problem.evaluate(sol_postpro).values()))
    print('hey')
    # import pickle
    # pickle.dump(sol, open("res_data/solution_single_batch.pk", "wb"))


def run_cp_benders_single_batch():
    problem = parse_problem(fix_problem=False)
    solver = CpsatBendersMultibatchingSolver(problem=problem,
                                             scaling_factor=1)
    solver.init_model(detailed_capacity_constraint=False,
                      single_batching=True)
    p = ParametersCp.default_cpsat()
    p.nb_process = 32
    subbrick = SubBrick(cls=GreedyPackingForMultibatching,
                        kwargs=dict(single_batching=True))
    result = solver.solve(callbacks=[],
                          nb_iteration=12,
                          params_solver_master=
                          {"time_limit": 10,
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


def run_cpopt_2steps():
    import config.env
    # problem = parse_problem()
    # problem = init_instance()
    problem = parse_problem()
    from multibatching.solvers.cpopt import CpOptMultibatchingSolver
    from multibatching.solvers.hexaly import MultibatchingHexalySolver, ModelingMultibatching
    solver = MultibatchingHexalySolver(problem=problem)
    solver.init_model(modeling=ModelingMultibatching.FLOW, single_batching=True)
    result = solver.solve(time_limit=300,
                          nb_threads=16)
    print(problem.evaluate(result[-1][0]))
    print(sum(problem.evaluate(result[-1][0]).values()))
    print(problem.evaluate(result[-1][0]), problem.satisfy(result[-1][0]))
    p = ParametersCp.default_cpsat()
    p.nb_process = 16

    # plot_solution_graph(problem=problem, solution=sol)
    # for i in range(len(problem.products)):
    #     plotly_solution_graph(solution=sol,
    #                           problem=problem,
    #                           use_logical_layout=False,
    #                           products_to_plot=[problem.products[i]],
    #                           explode_flows=True)

def run_milp_single():
    problem = parse_problem(False)
    from multibatching.solvers.milp_flow import GurobiMultibatchingSolver
    from discrete_optimization.generic_tools.lp_tools import ParametersMilp
    solver = GurobiMultibatchingSolver(problem)
    p = ParametersMilp.default()
    solver.init_model(single_batching=True)
    res = solver.solve(parameters_milp=p, time_limit=100)
    sol = res.get_best_solution()
    print(problem.evaluate(sol), problem.satisfy(sol))

def run_milp_2step():
    problem = parse_problem(False)
    from multibatching.solvers.milp_flow import GurobiMultibatchingSolver
    from discrete_optimization.generic_tools.lp_tools import ParametersMilp
    solver = GurobiMultibatchingSolver(problem)
    p = ParametersMilp.default()
    solver.init_model(single_batching=False)
    res = solver.solve(parameters_milp=p, time_limit=1000)
    sol = res.get_best_solution()
    print(problem.evaluate(sol), problem.satisfy(sol))
    pack = PackingViaBinPacking(problem)
    pack.init_from_solution(sol)
    res = pack.solve(time_limit_per_link=1)
    sol_ = res.get_best_solution()
    print(problem.evaluate(sol_), problem.satisfy(sol_), sum(problem.evaluate(sol_).values()))
    sol_: MultibatchingSolution
    sol_.list_flows = [PackingTransport(transport_link=pt.transport_link,
                                        product_packing=dict(pt.product_packing),
                                        nb_packing=pt.nb_packing)
                       for pt in sol_.list_flows]
    pickle.dump(sol_, open("solution_multibatching_milp.pk", "wb"))

def run_milp_direct():
    problem = parse_problem(False)
    from multibatching.solvers.milp_flow import GurobiMultibatchingSolver, GurobiMultibatchingSolverUnitFlow
    from discrete_optimization.generic_tools.lp_tools import ParametersMilp
    solver = GurobiMultibatchingSolverUnitFlow(problem)
    p = ParametersMilp.default()
    solver.init_model(max_trips_per_link=20)
    res = solver.solve(parameters_milp=p,
                       time_limit=1000)
    sol = res.get_best_solution()
    print(problem.evaluate(sol), problem.satisfy(sol), sum(problem.evaluate(sol).values()))
    sol: MultibatchingSolution
    sol.list_flows = [PackingTransport(transport_link=pt.transport_link,
                                        product_packing=dict(pt.product_packing),
                                        nb_packing=pt.nb_packing)
                       for pt in sol.list_flows]
    pickle.dump(sol, open("solution_multibatching_milp_direct.pk", "wb"))

def run_milp_lazy():
    problem = parse_problem(False)
    from multibatching.solvers.milp_flow import GurobiMultibatchingSolver, GurobiMultibatchingSolverWithLazyConstraint
    from discrete_optimization.generic_tools.lp_tools import ParametersMilp
    solver = GurobiMultibatchingSolverWithLazyConstraint(problem)
    p = ParametersMilp.default()
    solver.init_model()
    res = solver.solve(parameters_milp=p, time_limit=300)
    sol = res.get_best_solution()
    print(problem.evaluate(sol), problem.satisfy(sol), sum(problem.evaluate(sol).values()))
    sol_: MultibatchingSolution = sol
    sol_.list_flows = [PackingTransport(transport_link=pt.transport_link,
                                        product_packing=dict(pt.product_packing),
                                        nb_packing=pt.nb_packing)
                       for pt in sol_.list_flows]
    pickle.dump(sol_, open("solution_multibatching_lazy.pk", "wb"))


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
    subbrick = SubBrick(cls=PackingViaBinPacking,
                        kwargs={"time_limit_per_link": 1})
    #subbrick = SubBrick(cls=GreedyPackingForMultibatching,
    #                    kwargs={})
    result = solver.solve(callbacks=[],
                          nb_iteration=5,
                          params_solver_master=
                          {"time_limit": 100,
                           "parameters_cp": p,
                           "ortools_cpsat_solver_kwargs": {"log_search_progress": True}},
                          packing_solver=subbrick)
    sol, _ = result.get_best_solution_fit()
    print(problem.evaluate(sol))
    print(sum(problem.evaluate(sol).values()))
    print(problem.evaluate(sol), problem.satisfy(sol))
    from multibatching.problem import MultibatchingSolution, MultibatchingProblem
    sol: MultibatchingSolution
    sol.list_flows = [PackingTransport(transport_link=pt.transport_link,
                                       product_packing=dict(pt.product_packing),
                                       nb_packing=pt.nb_packing)
                      for pt in sol.list_flows]
    pickle.dump(sol, open("solution_multibatching.pk", "wb"))
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
    print("Sol", sol)
    print(problem.evaluate(sol))
    print(sum(problem.evaluate(sol).values()))
    print(problem.evaluate(sol), problem.satisfy(sol))
    plot_solution_graph(problem=problem, solution=sol)
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
    # run_cpoptal_2steps()
    run_cp_2steps()
    # run_milp_2step()
    # run_milp_lazy()
    # run_milp_2step()
    # run_milp_lazy()
    # run_cpopt_2steps()
    # run_cp_singlebatch_ws()
    # run_cp_benders()
    # run_cp_benders_single_batch()
    # run_cp_benders_single_batch()
    # run_cp_singlebatch()
    # run_cpopt_2steps()
    # run_cp_2steps()
    # run_networkx()
    # run_cp_2steps()
    # run_oe()
    # run_cpmpy()
    # run_explanation()

