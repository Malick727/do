from multibatching.solvers.asp import ClingconMultibatchingSolver
from multibatching.solvers.cpsat import CpsatMultibatchingSolver
from multibatching.utils import generate_multibatching_problem
import logging
logging.basicConfig(level=logging.INFO)
import os
import json
from multibatching.parse_config_data import parse_json_to_problem

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

def script():
    # 1. Generate the problem instance
    
    problem = generate_multibatching_problem(num_locations=20,
                                             num_transport_types=5,
                                             num_products=3,
                                             min_dist=1,
                                            max_dist=5,
                                             seed=51)
    
    problem = parse_problem()
    # 2. Initialize the solver
    solver = ClingconMultibatchingSolver(problem)
    solver.init_model()  # Initialize the ASP/Clingcon model

    # 3. Solve with a time limit (e.g., 30 seconds)

    time_limit=6000
    print(f"Starting solve with {time_limit}s timeout...")
    result_storage = solver.solve(time_limit=time_limit)

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