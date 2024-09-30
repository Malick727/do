from discrete_optimization.pickup_vrp.gpdp import ProxyClass
from discrete_optimization.tsp.tsp_parser import get_data_available, parse_file
from hexaly_solvers.tsp.hexaly_tsp_solver import TspHexalySolver
from discrete_optimization.tsp.solver.solver_ortools import TSP_ORtools
from discrete_optimization.pickup_vrp.solver.ortools_solver import ORToolsGPDP, ParametersCost
from discrete_optimization.tsp.plots.plot_tsp import plot_tsp_solution, plt
import logging
logging.basicConfig(level=logging.INFO)


def example_hexaly():
    files = get_data_available()
    file = [f for f in files if "tsp_18512_1" in f][0]
    problem = parse_file(file)
    solver = TspHexalySolver(problem)
    solver.init_model()
    res = solver.solve(timeout_seconds=40)
    sol = res.get_best_solution()
    assert problem.satisfy(sol)
    print(problem.evaluate(sol))
    plot_tsp_solution(tsp_model=problem, solution=sol)
    plt.show()


def example_ortools():
    files = get_data_available()
    file = [f for f in files if "tsp_1323_1" in f][0]
    problem = parse_file(file)
    solver = TSP_ORtools(problem)
    solver.init_model()
    res = solver.solve()
    sol = res.get_best_solution()
    assert problem.satisfy(sol)
    print(problem.evaluate(sol))
    plot_tsp_solution(tsp_model=problem, solution=sol)
    plt.show()


def example_gpdp():
    files = get_data_available()
    file = [f for f in files if "tsp_200_1" in f][0]
    problem = parse_file(file)
    gpdp = ProxyClass.from_tsp_model_gpdp(tsp_model=problem)
    print(gpdp, " done ")
    solver = ORToolsGPDP(problem=gpdp, factor_multiplier_distance=1)
    p_cost = [ParametersCost(dimension_name="Distance", global_span=False,
                             sum_over_vehicles=True, coefficient_vehicles=[1])]
    solver.init_model(include_time_dimension=False,
                      include_demand=False,
                      include_cumulative=False,
                      parameters_cost=p_cost,
                      time_limit=30)
    res = solver.solve()
    sol = res.get_best_solution()
    print(gpdp.evaluate(sol))


if __name__ == "__main__":
    example_hexaly()