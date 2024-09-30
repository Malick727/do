import os
os.environ["DO_SKIP_MZN_CHECK"] = "1"
from discrete_optimization.generic_tools.graph_api import Graph, from_networkx
import networkx as nx
from maximum_independent_set.mis_problem import MisProblem
from maximum_independent_set.solvers.mis_asp import MisASPSolver
from maximum_independent_set.solvers.mis_networkx import MisNXSolver
from maximum_independent_set.solvers.mis_ortools import MisOrtoolsSolver, ParametersCP
from maximum_independent_set.solvers.mis_kamis import MisKamisSolver
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


def run_kamis():
    graph = nx.Graph()
    graph.add_nodes_from([i for i in range(5)])
    graph.add_edge(0, 1)
    graph.add_edge(2, 3)
    g = from_networkx(graph)
    problem = MisProblem(g, attribute_aggregate="size")
    solver = MisKamisSolver(problem)
    solver.init_model()
    res = solver.solve(time_limit=20, method="weighted_local_search")
    sol, fit = res.get_best_solution_fit()
    print(problem.evaluate(sol))
    print(problem.satisfy(sol))

def run_one_exemple():
    from maximum_independent_set.solvers.mis_kamis import methods
    import subprocess
    file = "file_1712838486005060503_true.graph"
    method = "redumis"
    command = (f"{methods[method]} {file} "
               f"--output test.txt --time_limit=50 --console_log")
    # Launch the command and wait for it to finish
    print("Launching command")
    result = subprocess.run(command, shell=True, check=True, timeout=100)


if __name__ == "__main__":
    run_one_exemple()