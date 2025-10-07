import sys
import os
this_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(this_folder, "../../"))


from discrete_optimization.generic_tools.cp_tools import ParametersCp
from multibatching.solvers.packing_subproblem import GreedyPackingForMultibatching, PackingViaBinPacking
from multibatching.solvers.netx import NetxMultibatchingSolver
from multibatching.solvers.cpsat import CpsatPackingSubproblem
from multibatching.parse_config_data import parse_json_to_problem, parse_config_file_to_problem
import os

from multibatching.utils import plotly_solution_graph


def run_netw():
    this_folder = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(this_folder,
                               "../../multibatching/datalocal/config-local.json")
    problem = parse_config_file_to_problem(config_file)
    solver = NetxMultibatchingSolver(problem=problem)
    result = solver.solve()
    sol = result[0][0]
    print(problem.evaluate(result[0][0]), problem.satisfy(result[0][0]))
    problem.check_flows(sol)
    packing_solver = PackingViaBinPacking(problem=problem)
    res = packing_solver.solve(solution=sol, time_limit_per_link=1)
    # packing_solver = GreedyPackingForMultibatching(problem=problem)
    # res = packing_solver.solve(solution=sol)
    sol = res[-1][0]
    print(problem.evaluate(sol), problem.satisfy(sol))
    print(sum(problem.evaluate(sol).values()), " binpack")

    packing_solver = GreedyPackingForMultibatching(problem=problem)
    res = packing_solver.solve(solution=sol)
    sol = res[-1][0]
    print(sum(problem.evaluate(sol).values()), " by hand ")

    if True:
        packing_solver = CpsatPackingSubproblem(problem=problem, scaling_factor=1)
        packing_solver.init_model(solution=result[0][0])
        p = ParametersCp.default_cpsat()
        p.nb_process = 16
        res = packing_solver.solve(parameters_cp=p,
                                   time_limit=10)
        sol = res[-1][0]
        print(problem.evaluate(sol), problem.satisfy(sol))
        print(sum(problem.evaluate(sol).values()), " via full opti ")

        problem.check_flows(sol)
    if True:
        for i in range(1):
            plotly_solution_graph(solution=sol,
                                  problem=problem,
                                  use_logical_layout=False,
                                  products_to_plot=[problem.products[i]],
                                  explode_flows=True)


if __name__ == "__main__":
    run_netw()
