from multibatching.problem import MultibatchingProblem
from multibatching.solvers.cpsat import (CpsatMultibatchingSolver, ModelingMultiBatch,
                                         CpsatPackingSubproblem, CpsatBendersMultibatchingSolver)
from examples.multibatching.instance import init_instance
from discrete_optimization.generic_tools.cp_tools import ParametersCp
import logging

from multibatching.utils import plot_solution_graph, generate_multibatching_problem


# logging.basicConfig(level=logging.INFO)
def run():
    problem = init_instance()
    problem = generate_multibatching_problem(num_locations=20,
                                             num_transport_types=2,
                                             num_products=3,
                                             min_dist=1,
                                             max_dist=5,
                                             seed=51)
    solver = CpsatMultibatchingSolver(problem)
    solver.init_model(modeling=ModelingMultiBatch.UNIT_FLOW)
    p = ParametersCp.default_cpsat()
    res = solver.solve(parameters_cp=p,
                       time_limit=60,
                       ortools_cpsat_solver_kwargs={"log_search_progress": True})
    sol = res[-1][0]
    print(problem.evaluate(sol), problem.satisfy(sol))
    # plot_solution_graph(problem=problem, solution=res[-1][0])

def run_flow_postpro():
    problem = init_instance()
    solver = CpsatMultibatchingSolver(problem)
    solver.init_model(modeling=ModelingMultiBatch.FLOW)
    p = ParametersCp.default_cpsat()
    res = solver.solve(parameters_cp=p,
                       time_limit=5,
                       ortools_cpsat_solver_kwargs={"log_search_progress": True})
    subsolver = CpsatPackingSubproblem(problem=problem)
    subsolver.init_model(solution=res[-1][0])
    res_ = subsolver.solve(parameters_cp=p,
                           time_limit=5,
                           ortools_cpsat_solver_kwargs={"log_search_progress": True})
    sol = res_[-1][0]
    print(problem.evaluate(sol))
    plot_solution_graph(problem=problem, solution=res_[-1][0])


def run_benders():
    problem = generate_multibatching_problem(num_locations=20,
                                             num_transport_types=2,
                                             num_products=3,
                                             min_dist=1,
                                             max_dist=5,
                                             seed=51)
    solver = CpsatBendersMultibatchingSolver(problem)
    solver.init_model()
    p = ParametersCp.default_cpsat()
    p.nb_process = 16
    params_solver_sub = dict(parameters_cp=p,
                             time_limit=10,
                             ortools_cpsat_solver_kwargs={"log_search_progress": False})
    params_solver_master = dict(parameters_cp=p,
                                time_limit=5,
                                ortools_cpsat_solver_kwargs={"log_search_progress": True})
    res = solver.solve(params_solver_sub=params_solver_sub,
                       params_solver_master=params_solver_master,
                       nb_iteration=100)
    sol = res[-1][0]
    best_sol, _ = res.get_best_solution_fit(satisfying=problem)
    print(problem.evaluate(best_sol), problem.satisfy(best_sol))


if __name__ == "__main__":
    # {'transport': 10238.799999999997, 'emission': 7679.100000000002, 'capital': 109530.22792234425}
    # {'transport': 12812.499999999998, 'emission': 6397.3, 'capital': 168393.1352487149}
    # {'transport': 4146.799999999999, 'emission': 3295.8999999999996, 'capital': 111147.34371688118}
    # {'transport': 3606.4000000000005, 'emission': 2774.7, 'capital': 114390.67257492928}
    # {'transport': 5100.799999999999, 'emission': 4837.6, 'capital': 30471.807403013085} True
    # {'transport': 2550.2, 'emission': 3536.9000000000005, 'capital': 15422.914494694925} True
    # {'transport': 2968.3999999999996, 'emission': 3168.0, 'capital': 22787.58998410206} True
    # {'transport': 4481.299999999999, 'emission': 6312.7, 'capital': 37508.84347690008} True
    # {'transport': 4497.7, 'emission': 6147.900000000001, 'capital': 37619.24737521801} True
    run_benders()
    # {'transport': 12948.999999999998, 'emission': 6464.8, 'capital': 170786.9171569645}
    # {'transport': 12948.999999999998, 'emission': 6464.8, 'capital': 170786.9171569645}
    # {'transport': 4050.0, 'emission': 3203.5, 'capital': 110786.37716757182}
    # {'transport': 5100.799999999999, 'emission': 4837.6, 'capital': 30471.807403013085} True
    # {'transport': 2550.2000000000003, 'emission': 3536.9000000000005, 'capital': 15422.914494694925} True
    # {'transport': 3025.7999999999997, 'emission': 3206.2000000000003, 'capital': 22986.095839876634} True
    # {'transport': 2968.3999999999996, 'emission': 3168.0, 'capital': 22787.58998410206} True
    # {'transport': 4829.8, 'emission': 7283.8, 'capital': 36738.52938620086} True
    # run()
    # {'transport': 1654.0,
    #  'emission': 1770.0,
    #  'capital': 26150.627509687314}
    # run_flow_postpro()
    # {'transport': 1542.0,
    # 'emission': 1650.0,
    # 'capital': 26150.62750968731}
    # run()
