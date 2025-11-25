from discrete_optimization.fjsp.parser import get_data_available, parse_file
from cpopt_solvers.fjsp_cpopt_solver import CPOFJspSolver
import logging

logging.basicConfig(level=logging.INFO)


def run_cpopt_fjsp():
    files = get_data_available()
    print(files)
    file = [f for f in files if "Behnke60.fjs" in f][0]
    problem = parse_file(file)
    solver = CPOFJspSolver(problem=problem)
    solver.solve(time_limit=100)


if __name__ == "__main__":
    run_cpopt_fjsp()