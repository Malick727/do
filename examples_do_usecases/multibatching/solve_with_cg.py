import os
import sys
import json
import pickle

this_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(this_folder, "../../"))


from examples_do_usecases.multibatching.instance4 import init_instance
from multibatching.utils import plot_solution_graph, plotly_solution_graph, save_solution, read_solution, plot_solution_on_map, plot_solution_on_map_grouped, plot_solution_on_map_by_transport, getFALsLocation, formatCoordinates
from multibatching.parse_config_data import parse_json_to_problem

from multibatching.solvers.branch_and_price_copy import BranchAndPriceSolver
from multibatching.solvers.column_generation_cp_parallel import ColumnGenerationSolver
# from multibatching.solvers.column_generation_cp import ColumnGenerationSolver
from multibatching.solvers.milp import MilpSolver
from multibatching.solvers.milp_flow import GurobiMultibatchingSolverUnitFlow
import time


def parse_problem(fix_problem: bool = False):
    # this_folder = os.path.dirname(os.path.abspath(__file__))
    # config_file = os.path.join(this_folder,
    #                            "../../multibatching/datalocal/config_local.json")
    
    # print(f"DEBUG: Looking for config at: {os.path.abspath(config_file)}")
    # print(f"DEBUG: Does it exist? {os.path.exists(config_file)}")

    # 1. Get the directory of the script: .../examples_do_usecases/multibatching/
    this_folder = os.path.dirname(os.path.abspath(__file__))

    # 2. Go up TWO levels to get to: .../discrete-optimisation-use-cases/
    project_root = os.path.dirname(os.path.dirname(this_folder))
    print(project_root)

    # 3. Construct the path to the data
    config_file = os.path.join(project_root, "multibatching", "datalocal", "config-local.json")

    print(f"Checking path: {config_file}")

    if os.path.exists(config_file):
        with open(config_file, "r") as f:
            d = json.load(f)
        print("Success: File loaded!")
    else:
        print("Error: Still not found. Check if 'multibatching' is spelled correctly at the root.")

    d = json.load(open(config_file, "r"))
    problem = parse_json_to_problem(d, harbor_constraint=True)


    # Temporary fix = section2 transportable by any means..
    if fix_problem:
        problem.products[3].valid_transports = set(problem.transport_types)

    return problem


def run():

    # problem = init_instance()
    problem = parse_problem()
    with open('examples_do_usecases/multibatching/res_data/best_flow_solver_sol.pickle', 'rb') as handle:
        init_solution = pickle.load(handle)



    # #### Branch-and-price solver
    solver = BranchAndPriceSolver(problem, init_solution)


    # solver = GurobiMultibatchingSolverUnitFlow(problem)
    

    ##### Column generation solver
    # df.to_csv("multibatching/datalocal/locations.csv", index=True)
    # print("Products :", len(problem.products))
    # print("Locations :", len(problem.locations))
    # print("Transport links", len(problem.transport_links))
    # print("Transport types :", len(problem.transport_types))
    # solver = ColumnGenerationSolver(problem)

    # solver.init_model(init_solution, flows_on_demand = True)
    start = time.time()
    # res = solver.solve()
    # print(res.get_best_solution_fit())
    # print(res)
    res, best_value = solver.solve()
    # print(res.list_flows, best_value)

    # print(res[-1][0].list_flows)
    for packing_transport in res.list_flows:
        print(packing_transport.transport_link.location_l1.id, packing_transport.transport_link.location_l2.id, packing_transport.product_packing, packing_transport.nb_packing)
    
    end = time.time()
    print("computation time", end-start)

    # sol = solver.get_final_solution()
    


    
if __name__== "__main__":
    run()