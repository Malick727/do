from discrete_optimization.fjsp.flex_job_shop_parser import get_data_available_fjsp, parse_fjs_file
from cpopt_solvers.fjsp_cpopt_solver import CPOFJspSolver
import logging

logging.basicConfig(level=logging.INFO)


def run_cpopt_fjsp():
    files = get_data_available_fjsp()
    print(files)
    file = [f for f in files if "Behnke60.fjs" in f][0]
    problem = parse_fjs_file(file)
    solver = CPOFJspSolver(problem=problem)
    solver.solve(time_limit=100)


if __name__ == "__main__":
    run_cpopt_fjsp()