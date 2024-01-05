import os
os.environ["DO_SKIP_MZN_CHECK"] = "1"
from discrete_optimization.generic_tools.graph_api import Graph, from_networkx
import networkx as nx
from maximum_independent_set.mis_problem import MisProblem
from maximum_independent_set.solvers.mis_asp import MisASPSolver
from maximum_independent_set.solvers.mis_networkx import MisNXSolver
from maximum_independent_set.solvers.mis_ortools import MisOrtoolsSolver, ParametersCP
import logging
logging.basicConfig(level=logging.INFO)


def run_asp():
    graph = nx.Graph()
    graph.add_nodes_from([i for i in range(5)])
    graph.add_edge(0, 1)
    graph.add_edge(2, 3)
    g = from_networkx(graph)
    problem = MisProblem(g, attribute_aggregate="size")
    solver = MisASPSolver(problem)
    solver.init_model(max_models=1000)
    res = solver.solve()
    sol, fit = res.get_best_solution_fit()
    print(problem.evaluate(sol))
    print(problem.satisfy(sol))


def run_networkx():
    graph = nx.Graph()
    graph.add_nodes_from([i for i in range(5)])
    graph.add_edge(0, 1)
    graph.add_edge(2, 3)
    g = from_networkx(graph)
    problem = MisProblem(g, attribute_aggregate="size")
    solver = MisNXSolver(problem)
    res = solver.solve()
    sol, fit = res.get_best_solution_fit()
    print(problem.evaluate(sol))
    print(problem.satisfy(sol))

def run_ortools():
    graph = nx.Graph()
    graph.add_nodes_from([i for i in range(5)])
    graph.add_edge(0, 1)
    graph.add_edge(2, 3)
    g = from_networkx(graph)
    problem = MisProblem(g, attribute_aggregate="size")
    solver = MisOrtoolsSolver(problem)
    solver.init_model()
    res = solver.solve(parameters_cp=ParametersCP.default())
    sol, fit = res.get_best_solution_fit()
    print(problem.evaluate(sol))
    print(problem.satisfy(sol))


if __name__ == "__main__":
    run_ortools()