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
                (max(0, i + dd), d2),
                {"w": int(-2 * random.random()) if integer else -2 * random.random()},
            )
            for dd in range(-1, delta + 1)
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


def create_duplicate_node_instance_ads(nb_clients: int = 20, integer=False):
    import random
    nb_clients = nb_clients
    nb_duplicates = 5
    delta = 2
    source = "source"
    sink = "sink"
    times = list(range(nb_clients*nb_duplicates))
    times = list(range(120))  # 2 hours for example
    nodes = [(source, {"time": -1})]
    for client in range(nb_clients):
        min_reward = int(-2 * random.random()) if integer else -2 * random.random()
        time = random.choice(times)
        tt = [t for t in range(max(0, time-nb_duplicates//2), max(0, time-nb_duplicates//2)+nb_duplicates)]
        nodes += [((client, j), {"time": tt[j], "w": min_reward}) for j in range(len(tt))]
    nodes += [(sink, {"w": 0})]
    # links between source and first client
    edges = []
    for node1 in nodes[:-1]:
        edges += [(node1[0], sink, {"w": 0})]
        for node2 in nodes[:-1]:
            if node1[1]["time"] < node2[1]["time"] < node1[1]["time"]+5:
                edges += [(node1[0], node2[0], {"w": node2[1]["w"]})]
    for e in edges:
        e[2]["w_pos"] = abs(e[2]["w"])
    graph = Graph(nodes=nodes, edges=edges, undirected=False, compute_predecessors=False)
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
    problem = create_duplicate_node_instance_ads(nb_clients=500)
    print("Problem initialized")
    solver = LPPathPlanning_Gurobi(problem=problem)
    solver.init_model(include_constraints=True)
    parameters_milp = ParametersMilp.default()
    parameters_milp.time_limit = 300
    result = solver.solve(parameters_milp=parameters_milp)
    sol, fit = result.get_best_solution_fit()
    print("Fitness ", fit)
    print(problem.evaluate(sol))
    print("Satisfy ", problem.satisfy(sol))
    print(sol.path)
    print("Len path : ", len(sol.path))


def run_networkx():
    problem = create_duplicate_node_instance(nb_clients=100, integer=True)
    solver = PathPlanningNX(problem=problem)
    result = solver.solve(
        shortest_path_method=ShortestPathMethod.DAG_LONGEST_PATH, weight="w_pos"
    )
    sol, fit = result.get_best_solution_fit()
    print("Fitness Dag longest path", fit)
    print(sol.path)
    print([problem.index_to_nodes[x] for x in sol.path])
    print(problem.satisfy(sol))

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
    run_gurobi_duplicate()
