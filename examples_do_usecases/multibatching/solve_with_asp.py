from multibatching.solvers.asp import ClingconMultibatchingSolver
from multibatching.solvers.cpsat import CpsatMultibatchingSolver
from multibatching.utils import generate_multibatching_problem
import logging
logging.basicConfig(level=logging.INFO)
import os
import json
from multibatching.parse_config_data import parse_json_to_problem

def parse_problem(fix_problem: bool = False,
                  harbor_constraint: bool = False,
                  filter_transport_types:bool = False):
    this_folder = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(this_folder,
                               "../../multibatching/datalocal/config-local.json")
    d = json.load(open(config_file, "r"))
    problem = parse_json_to_problem(d, harbor_constraint=harbor_constraint)

    # Temporary fix = section2 transportable by any means..
    if fix_problem:
        problem.products[3].valid_transports = set(problem.transport_types)
    if filter_transport_types:
        for product in problem.products:
            size = product.size
            product.valid_transports = set({t for t in product.valid_transports
                                            if t.capacity >= size})
    return problem

def script():
    # 1. Generate the problem instance
    from discrete_optimization.generic_tools.callbacks.stats_retrievers import BasicStatsCallback
    problem = generate_multibatching_problem(num_locations=20,
                                             num_transport_types=5,
                                             num_products=3,
                                             min_dist=1,
                                             max_dist=5,
                                             seed=51)
    problem = parse_problem(harbor_constraint=False, filter_transport_types=True)
    # 2. Initialize the solver
    solver = ClingconMultibatchingSolver(problem)
    solver.init_model()  # Initialize the ASP/Clingcon model
    # 3. Solve with a time limit (e.g., 30 seconds)
    time_limit=1000
    print(f"Starting solve with {time_limit}s timeout...")
    result_storage = solver.solve(callbacks=[BasicStatsCallback()],
                                  time_limit=time_limit)
    # 4. Retrieve and display results
    solution, fitness = result_storage.get_best_solution_fit()
    if solution == None:
        print("UNSAT")
    else:
        print(problem.satisfy(solution), problem.evaluate(solution))

    print(f"Solve Status: {solver.status_solver}")
    if solution:
        print(f"Best Solution Found: {fitness}")
        # You can process 'solution' here (e.g. print flows)
    else:
        print("No solution found within the time limit.")


if __name__ == "__main__":
    script()