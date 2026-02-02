import os
import sys
import json

this_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(this_folder, "../../"))

from examples_do_usecases.multibatching.instance4 import init_instance
from multibatching.utils import plot_solution_graph, plotly_solution_graph, save_solution, read_solution, plot_solution_on_map, plot_solution_on_map_grouped, plot_solution_on_map_by_transport, getFALsLocation, formatCoordinates
from multibatching.parse_config_data import parse_json_to_problem

from multibatching.solvers.branch_and_price_copy import BranchAndPriceSolver
from multibatching.solvers.column_generation_cp_parallel import ColumnGenerationSolver
from multibatching.solvers.milp import MilpSolver
from multibatching.solvers.milp_flow import GurobiMultibatchingSolverUnitFlow
import time


def parse_problem(fix_problem: bool = False):
    this_folder = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(this_folder,
                               "../../multibatching/datalocal/config-local.json")
    d = json.load(open(config_file, "r"))
    problem = parse_json_to_problem(d, harbor_constraint=True)

    # Temporary fix = section2 transportable by any means..
    if fix_problem:
        problem.products[3].valid_transports = set(problem.transport_types)

    return problem


def run():

    # problem = init_instance()
    problem = parse_problem()



    # #### Branch-and-price solver
    # solver = BranchAndPriceSolver(problem)


    # solver = GurobiMultibatchingSolverUnitFlow(problem)
    

    ##### Column generation solver
    # df.to_csv("multibatching/datalocal/locations.csv", index=True)
    # print("Products :", len(problem.products))
    # print("Locations :", len(problem.locations))
    # print("Transport links", len(problem.transport_links))
    # print("Transport types :", len(problem.transport_types))
    solver = ColumnGenerationSolver(problem)

    solver.init_model()
    start = time.time()
    res = solver.solve()
    # res, best_value = solver.solve()
    # print(res.list_flows, best_value)

    # print(res[-1][0].list_flows)
    end = time.time()
    print("computation time", end-start)
    # sol = solver.get_final_solution()
    


    
if __name__== "__main__":
    run()