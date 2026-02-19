import config.env
from discrete_optimization.top.parser import get_data_available, parse_file
from hexaly_solvers.top.hexaly_top_solver import TopHexalySolver
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.callbacks.loggers import ProblemEvaluateLogger
import logging
logging.basicConfig(level=logging.INFO)
def run():
    files, files_dict = get_data_available()
    print(files)
    print(files[0])
    problem = parse_file(files[4])
    solver = TopHexalySolver(problem)
    solver.init_model()
    res = solver.solve(parameters_cp=ParametersCp.default_cpsat(),
                       callbacks=[ProblemEvaluateLogger(step_verbosity_level=logging.INFO,
                                                        end_verbosity_level=logging.INFO)],
                       time_limit=30)
    sol = res[-1][0]
    print(problem.satisfy(sol), problem.evaluate(sol))

if __name__ == "__main__":
    run()