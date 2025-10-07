import os
import sys
import json

this_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(this_folder, "../../"))

from examples.multibatching.instance2 import init_instance
from multibatching.utils import plot_solution_graph
from multibatching.parse_config_data import parse_json_to_problem

from multibatching.solvers.milp import MilpSolver
from multibatching.solvers.column_generation import ColumnGenerationSolver


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


def run():
    problem = init_instance()
    solver = ColumnGenerationSolver(problem)
    solver.init_model()
    res = solver.solve()
    sol = solver.get_final_solution()
    print(sol.list_flows)
    print(problem.evaluate(sol), problem.satisfy(sol))
    plot_solution_graph(problem=problem, solution=sol)
    # print(problem.satisfy(sol))
    # # print(res.list_solution_fits)
    # # sol = res.get_best_solution_fit()
    # # print(sol)

    # # c = solver.explain_unsat_fine()
    # # print(c)
    
    
    

if __name__== "__main__":
    run()