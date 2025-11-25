from discrete_optimization.fjsp.parser import get_data_available, parse_file
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from optalcp_solvers.solvers.fjsp_optal import OptalFJspSolver
import logging
logging.basicConfig(level=logging.INFO)


def run_optal_fjsp():
    files = get_data_available()
    print(files)
    file = [f for f in files if "Behnke16.fjs" in f][0]
    problem = parse_file(file)
    solver = OptalFJspSolver(problem=problem)
    solver.init_model()
    p = ParametersCp.default_cpsat()
    p.nb_process = 16
    res = solver.solve(time_limit=3,
                       parameters_cp=p)
    sol = res.get_best_solution()
    print(problem.satisfy(sol))


if __name__ == "__main__":
    run_optal_fjsp()