import os
from typing import Any, Dict, Hashable
from discrete_optimization.generic_tools.do_problem import build_aggreg_function_and_params_objective, \
    ParamsObjectiveFunction
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from path_planning.path_planning import PathPlanning, PathSolution,\
    NodeConstraints, Comparison
from discrete_optimization.generic_tools.do_solver import SolverDO
import networkx as nx
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ShortestPathMethod(Enum):
    DIJKSTRA_PATH = 0
    ASTAR_PATH = 1
    BELLMAN_FORD = 2
    MIN_COST_FLOW = 3
    DAG_LONGEST_PATH = 4


def build_path_from_flow(flow_dict: Dict[Hashable,
                                         Dict[Hashable, float]],
                         source: Hashable, target: Hashable):
    edges = set([(u, v) for u in flow_dict for v in flow_dict[u] if flow_dict[u][v] > 0])
    path = [source]
    while True:
        element = next((e for e in edges if e[0] == path[-1]), None)
        path += [element[1]]
        if element[1] == target:
            break
    return path


class PathPlanningNX(SolverDO):
    def __init__(self, problem: PathPlanning, params_objective_function: ParamsObjectiveFunction = None):
        self.problem = problem
        self.aggreg_sol, self.aggreg_from_dict_values, self.params_objective_function = \
            build_aggreg_function_and_params_objective(self.problem,
                                                       params_objective_function=params_objective_function)
        self.graph_nx = self.problem.graph.graph_nx
        self.default_attribute = self.problem.length_attribute
        self.has_negative_weights = False
        if self.problem.only_negative_length() or self.problem.mixed_positive_negative_length():
            self.has_negative_weights = True
            # Trick for dijkstra -> simply doesn't work :)
            # mini = self.problem.min_cost
            # new_attribute = f"{self.problem.length_attribute}_shift"
            # for edge in self.graph_nx.edges:
            #     self.graph_nx.edges[edge][new_attribute] = \
            #         self.graph_nx.edges[edge][self.problem.length_attribute]+abs(mini)
            # self.default_attribute = new_attribute

    def solve(self, shortest_path_method: ShortestPathMethod = ShortestPathMethod.ASTAR_PATH, **kwargs: Any) \
            -> ResultStorage:
        path = []
        weight = kwargs.get("weight", self.default_attribute)
        if shortest_path_method == ShortestPathMethod.ASTAR_PATH:
            heuristic = kwargs.get("heuristic", None)
            if self.problem.only_negative_length() or self.problem.mixed_positive_negative_length():
                heuristic = lambda x, y: -100 if x != self.problem.sink else 0
            path = nx.astar_path(G=self.graph_nx,
                                 source=self.problem.source,
                                 target=self.problem.sink,
                                 heuristic=heuristic,
                                 weight=weight)
        if shortest_path_method == ShortestPathMethod.DIJKSTRA_PATH:
            if self.has_negative_weights:
                logger.error(f"Algo {shortest_path_method} is not suitable with graph with negative weights.")
                raise Exception(f"Algo {shortest_path_method} is not suitable "
                                f"with graph with negative weights.")
            path = nx.dijkstra_path(G=self.graph_nx,
                                    source=self.problem.source,
                                    target=self.problem.sink,
                                    weight=weight)
        if shortest_path_method == ShortestPathMethod.BELLMAN_FORD:
            path = nx.bellman_ford_path(G=self.graph_nx,
                                        source=self.problem.source,
                                        target=self.problem.sink,
                                        weight=weight)
        if shortest_path_method == ShortestPathMethod.MIN_COST_FLOW:
            attr_demand = "demand"
            for node in self.graph_nx.nodes():
                self.graph_nx.nodes[node][attr_demand] = 0
            self.graph_nx.nodes[self.problem.source][attr_demand] = -1
            self.graph_nx.nodes[self.problem.sink][attr_demand] = 1
            flow_cost, flow_dict = nx.network_simplex(self.graph_nx,
                                                      demand='demand',
                                                      capacity='capacity',
                                                      weight=weight)
            path = build_path_from_flow(flow_dict=flow_dict,
                                        source=self.problem.source,
                                        target=self.problem.sink)
        if shortest_path_method == ShortestPathMethod.DAG_LONGEST_PATH:
            if not nx.is_directed_acyclic_graph(self.graph_nx):
                logger.error(f"Graph is not a dag, don't use dag method")
                raise Exception("Graph is not a dag, don't use dag method")
            path = nx.dag_longest_path(G=self.graph_nx,
                                       weight=weight)
        logger.info(f"Solver {shortest_path_method} found a solution")
        path_index = [self.problem.index_nodes[p] for p in path]
        sol = PathSolution(problem=self.problem, path=path_index)
        fit = self.aggreg_sol(sol)
        logger.info(f"Fitness = {fit}")
        return ResultStorage(list_solution_fits=[(sol, fit)],
                             mode_optim=self.params_objective_function.sense_function)








