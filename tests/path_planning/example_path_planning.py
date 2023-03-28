import os
os.environ["DO_SKIP_MZN_CHECK"] = "1"
from discrete_optimization.generic_tools.graph_api import Graph
from path_planning.path_planning import PathPlanning, PathSolution, NodeConstraints,\
    Comparison, PathConstraints
from path_planning.solvers.path_planning_milp import LPPathPlanning_Gurobi, ParametersMilp
import logging


def create_small_instance():
    nodes = [(0, {}),
             (1, {}),
             (2, {}),
             (3, {}),
             (4, {})]
    edges = [(0, 1, {"w": 1}),
             (1, 2, {"w": 1}),
             (2, 3, {"w": 1}),
             (3, 4, {"w": 1}),
             (2, 4, {"w": 5}),
             (1, 4, {"w": 3})]
    graph = Graph(nodes=nodes,
                  edges=edges,
                  undirected=False)
    problem = PathPlanning(graph=graph,
                           source=0,
                           sink=4,
                           length_attribute="w",
                           # I want to visit node 3.
                           node_constraints=NodeConstraints(list_constraint=[([3], Comparison.UEQ, 1)]),
                           # I want to have a path longer than 4.2
                           path_constraints=PathConstraints(list_constraint=[("w", Comparison.UP, 4.2)]))
    solver = LPPathPlanning_Gurobi(problem=problem)
    solver.init_model()
    result = solver.solve(parameters_milp=ParametersMilp.default())
    sol, fit = result.get_best_solution_fit()
    print("Fitness ", fit)
    print(sol.path)


if __name__ == "__main__":
    create_small_instance()
