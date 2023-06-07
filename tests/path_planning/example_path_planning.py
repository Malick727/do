import os

os.environ["DO_SKIP_MZN_CHECK"] = "1"
import logging

from discrete_optimization.generic_tools.graph_api import Graph

from path_planning.path_planning import (
    Comparison,
    NodeConstraints,
    PathConstraints,
    PathPlanning,
    PathSolution,
)
from path_planning.solvers.path_planning_columngen import PathPlanningColumnGen
from path_planning.solvers.path_planning_milp import (
    LPPathPlanning_Gurobi,
    ParametersMilp,
)
from path_planning.solvers.path_planning_networkx import (
    PathPlanningNX,
    ShortestPathMethod,
)
from path_planning.solvers.path_planning_ortools import PathPlanningOrtools


def create_small_instance():
    nodes = [(0, {}), (1, {}), (2, {}), (3, {}), (4, {})]
    edges = [
        (0, 1, {"w": 1}),
        (1, 2, {"w": 1}),
        (2, 3, {"w": 1}),
        (3, 4, {"w": 1}),
        (2, 4, {"w": 5}),
        (1, 4, {"w": 3}),
    ]
    graph = Graph(nodes=nodes, edges=edges, undirected=False)
    problem = PathPlanning(
        graph=graph,
        source=0,
        sink=4,
        length_attribute="w",
        # I want to visit node 3.
        # node_constraints=NodeConstraints(list_constraint=[([3], Comparison.UEQ, 1)]),
        # I want to have a path longer than 4.2
        path_constraints=PathConstraints(list_constraint=[("w", Comparison.UP, 4.2)]),
    )
    return problem


def run_gurobi_1():
    problem = create_small_instance()
    solver = LPPathPlanning_Gurobi(problem=problem)
    solver.init_model()
    result = solver.solve(parameters_milp=ParametersMilp.default())
    sol, fit = result.get_best_solution_fit()
    print("Fitness ", fit)
    print(sol.path)


def create_duplicate_node_instance(nb_clients: int = 20, integer=False):
    import random

    nb_clients = nb_clients
    nb_duplicates = 5
    delta = 2
    source = "source"
    sink = "sink"
    nodes = (
        [(source, {})]
        + [
            ((client, dup), {})
            for client in range(nb_clients)
            for dup in range(nb_duplicates)
        ]
        + [(sink, {})]
    )
    # links between source and first client
    edges = [
        (
            source,
            (0, dup),
            {"w": int(-10 * random.random()) if integer else -10 * random.random()},
        )
        for dup in range(nb_duplicates)
    ]
    # links between last client and sink
    edges += [
        (
            (nb_clients - 1, dup),
            sink,
            {"w": int(-10 * random.random()) if integer else -10 * random.random()},
        )
        for dup in range(nb_duplicates)
    ]
    for i in range(nb_clients - delta):
        # links "consecutive clients" client
        edges += [
            (
                (i, d),
                (i + dd, d2),
                {"w": int(-2 * random.random()) if integer else -2 * random.random()},
            )
            for dd in range(1, delta + 1)
            for d in range(nb_duplicates)
            for d2 in range(nb_duplicates)
        ]

    for e in edges:
        e[2]["w_pos"] = abs(e[2]["w"])
    graph = Graph(nodes=nodes, edges=edges, undirected=False)
    constraints = NodeConstraints(
        list_constraint=[
            ([(client, d) for d in range(nb_duplicates)], Comparison.LEQ, 1)
            for client in range(nb_clients)
        ]
    )
    # No more than 1 visit to this "client"
    problem = PathPlanning(
        graph=graph,
        source=source,
        sink=sink,
        length_attribute="w",
        node_constraints=constraints,
    )
    return problem


def run_gurobi_duplicate():
    problem = create_duplicate_node_instance()
    solver = LPPathPlanning_Gurobi(problem=problem)
    solver.init_model()
    result = solver.solve(parameters_milp=ParametersMilp.default())
    sol, fit = result.get_best_solution_fit()
    print("Fitness ", fit)
    print(sol.path)


def run_networkx():
    problem = create_duplicate_node_instance(nb_clients=100, integer=True)
    solver = PathPlanningNX(problem=problem)
    result = solver.solve(
        shortest_path_method=ShortestPathMethod.DAG_LONGEST_PATH, weight="w_pos"
    )
    sol, fit = result.get_best_solution_fit()
    print("Fitness Dijkstra", fit)
    print(sol.path)

    solver = PathPlanningNX(problem=problem)
    result = solver.solve(shortest_path_method=ShortestPathMethod.MIN_COST_FLOW)
    sol, fit = result.get_best_solution_fit()
    print("Fitness Min cost flow", fit)
    print(sol.path)

    solver = PathPlanningOrtools(problem=problem)
    result = solver.solve()
    sol, fit = result.get_best_solution_fit()
    print("Fitness Min cost flow ortools", fit)
    print(sol.path)


def run_column_gen():
    logging.basicConfig(level=logging.INFO)
    problem = create_small_instance()
    solver = PathPlanningColumnGen(problem=problem)
    solver.init_subsolver(
        subsolver=PathPlanningNX, shortest_path_method=ShortestPathMethod.ASTAR_PATH
    )
    result = solver.solve()
    sol, fit = result.get_best_solution_fit()
    print("Fitness ", fit)
    print(sol.path)


if __name__ == "__main__":
    run_column_gen()
