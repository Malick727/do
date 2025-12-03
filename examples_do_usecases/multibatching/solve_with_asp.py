from multibatching.solvers.asp import ClingconMultibatchingSolver
from multibatching.solvers.cpsat import CpsatMultibatchingSolver
from multibatching.utils import generate_multibatching_problem
import logging
logging.basicConfig(level=logging.INFO)

def script():
    # 1. Generate the problem instance
    problem = generate_multibatching_problem(num_locations=20,
                                             num_transport_types=2,
                                             num_products=3,
                                             min_dist=1,
                                             max_dist=5,
                                             seed=51)

    # 2. Initialize the solver
    solver = ClingconMultibatchingSolver(problem)
    solver.init_model()  # Initialize the ASP/Clingcon model

    # 3. Solve with a time limit (e.g., 30 seconds)
    print("Starting solve with 30s timeout...")
    result_storage = solver.solve(time_limit=30)

    # 4. Retrieve and display results
    solution, fitness = result_storage.get_best_solution_fit()
    print(problem.satisfy(solution), problem.evaluate(solution))

    print(f"Solve Status: {solver.status_solver}")
    if solution:
        print(f"Best Solution Found: {fitness}")
        # You can process 'solution' here (e.g. print flows)
    else:
        print("No solution found within the time limit.")


if __name__ == "__main__":
    script()