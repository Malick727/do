import os

import config.env
from discrete_optimization.datasets import get_data_home
from discrete_optimization.tsptw.parser import get_data_available, parse_tsptw_file
from cpopt_solvers.tsptw_cpopt_solver import CPOptSolverTSPTw

def run():
    problem = parse_tsptw_file(get_data_available()[0])
    data_available_ohlmann = get_data_available(
        data_folder=os.path.join(get_data_home(), "tsptw", "OhlmannThomas")
    )
    problem = parse_tsptw_file(data_available_ohlmann[2])
    solver = CPOptSolverTSPTw(problem)
    solver.init_model()
    res = solver.solve(time_limit=5)
    sol = res.get_best_solution()
    print(problem.satisfy(sol), problem.evaluate(sol))

if __name__ == "__main__":
    run()
