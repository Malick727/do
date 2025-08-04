from multibatching.problem import MultibatchingProblem
from multibatching.solvers.cpsat import (CpsatMultibatchingSolver, ModelingMultiBatch,
                                         CpsatPackingSubproblem, CpsatBendersMultibatchingSolver)
from examples.multibatching.instance import init_instance
from discrete_optimization.generic_tools.cp_tools import ParametersCp
import logging

from multibatching.utils import plot_solution_graph, generate_multibatching_problem, plotly_solution_graph
from multibatching.parsing_airbus_data_ import main_parse

#logging.basicConfig(level=logging.INFO)
def run():
    problem = main_parse()
    # problem = generate_multibatching_problem(num_locations=20,
    #                                          num_transport_types=2,
    #                                          num_products=3,
    #                                          min_dist=1,
    #                                          max_dist=5,
    #                                          seed=51)
    solver = CpsatMultibatchingSolver(problem)
    solver.init_model(modeling=ModelingMultiBatch.UNIT_FLOW)
    p = ParametersCp.default_cpsat()
    p.nb_process = 16
    res = solver.solve(parameters_cp=p,
                       time_limit=300,
                       ortools_cpsat_solver_kwargs={"log_search_progress": True})
    sol = res[-1][0]
    print(problem.evaluate(sol), problem.satisfy(sol))
    plot_solution_graph(problem=problem, solution=res[-1][0])


def run_flow():
    problem = main_parse()
    # problem = generate_multibatching_problem(num_locations=20,
    #                                          num_transport_types=2,
    #                                          num_products=3,
    #                                          min_dist=1,
    #                                          max_dist=5,
    #                                          seed=51)
    solver = CpsatMultibatchingSolver(problem, scaling_factor=10**3)
    solver.init_model(modeling=ModelingMultiBatch.FLOW,
                      detailed_capacity_constraint=False)
    p = ParametersCp.default_cpsat()
    p.nb_process = 16
    res = solver.solve(parameters_cp=p,
                       time_limit=60,
                       ortools_cpsat_solver_kwargs={"log_search_progress": True})
    sol = res[-1][0]
    print(problem.evaluate(sol), problem.satisfy(sol))
    for product in problem.products:
        fig = plotly_solution_graph(problem=problem,
                                    solution=res[-1][0],
                                    products_to_plot=[product],
                                    explode_flows=True,
                                    use_logical_layout=False,
                                    png_path=f"flow_{product.id}.png")

    #plot_solution_graph(problem=problem, solution=res[-1][0])


def run_flow_postpro():
    problem = main_parse()
    solver = CpsatMultibatchingSolver(problem)
    solver.init_model(modeling=ModelingMultiBatch.FLOW,
                      detailed_capacity_constraint=True)
    p = ParametersCp.default_cpsat()
    # False : 47491131
    # True : 47510661, 47719922
    res = solver.solve(parameters_cp=p,
                       time_limit=30,
                       ortools_cpsat_solver_kwargs={"log_search_progress": True})
    subsolver = CpsatPackingSubproblem(problem=problem)
    subsolver.init_model(solution=res[-1][0])
    res_ = subsolver.solve(parameters_cp=p,
                           time_limit=5,
                           ortools_cpsat_solver_kwargs={"log_search_progress": False})
    sol = res_[-1][0]
    print(problem.evaluate(sol))
    print(sum(problem.evaluate(sol).values()))
    plot_solution_graph(problem=problem, solution=res_[-1][0])


def run_benders():
    problem = main_parse()
    # problem = init_instance()
    solver = CpsatBendersMultibatchingSolver(problem)
    solver.init_model(detailed_capacity_constraint=True)
    p = ParametersCp.default_cpsat()
    p.nb_process = 16
    params_solver_sub = dict(parameters_cp=p,
                             time_limit=5,
                             ortools_cpsat_solver_kwargs={"log_search_progress": False})
    params_solver_master = dict(parameters_cp=p,
                                time_limit=20,
                                ortools_cpsat_solver_kwargs={"log_search_progress": False})
    res = solver.solve(params_solver_sub=params_solver_sub,
                       params_solver_master=params_solver_master,
                       nb_iteration=100)
    sol = res[-1][0]
    best_sol, _ = res.get_best_solution_fit(satisfying=problem)
    print(problem.evaluate(best_sol), problem.satisfy(best_sol))


if __name__ == "__main__":
    run_flow()