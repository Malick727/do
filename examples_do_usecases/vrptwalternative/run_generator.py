import logging

from vrp_tw_alternative_delivery.generator import generate_from_vrptw, VRPTWProblem, VrpTwAlternativeDeliveryProblem
from vrp_tw_alternative_delivery.solvers.cpsat import CpSatVrpTwAlternative
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.vrptw.parser import parse_solomon
from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
logging.basicConfig(level=logging.INFO)

def run_generator():
    problem = parse_solomon("homberger_200_customer_instances/R2_2_2.TXT")
    problem = generate_from_vrptw(vrptw_problem=problem,
                                  nb_nodes_target=50,
                                  nb_vehicles_target=20)
    return problem
    # print(problem)
    # print("Done")

def run_cp():
    problem = run_generator()
    solver = CpSatVrpTwAlternative(problem=problem)
    solver.init_model()
    params = ParametersCp.default_cpsat()
    params.nb_process = 16
    res = solver.solve(callbacks=[ObjectiveLogger(step_verbosity_level=logging.INFO,
                                                  end_verbosity_level=logging.INFO)],
                       time_limit=100,
                       parameters_cp=params,
                       ortools_cpsat_solver_kwargs={"log_search_progress": True})
    sol = res[-1][0]
    print(problem.satisfy(sol))


if __name__ == "__main__":
    run_cp()