from copy import deepcopy
from typing import Optional, List, Any, Iterable, Tuple

import networkx as nx
import numpy as np

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import IntegerHyperparameter
from discrete_optimization.generic_tools.hyperparameters.hyperparametrizable import Hyperparametrizable
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage

from satellite_path_planning.solver.spp_solver import SppSolver
from satellite_path_planning.spp_model import SppProblem, SppSolution


def long_path_DAG(graph: nx.DiGraph, list_nodes: Iterable = None) -> Tuple[dict[int, float], dict[int, list[int]], dict[int, set]]:
    """
    @param graph: DAG corresponding to the problem to solve
    @param list_nodes: list of nodes of DAG in topological order
    @return: the better value for each node corresponding to the better path to this node, the set of acquisition captured by this path
    """
    if list_nodes is None:
        list_nodes = nx.topological_sort(graph)
    acquisition_request_id = nx.get_node_attributes(graph, "acquisition_request_id")
    value = nx.get_node_attributes(graph, "value")

    chemin = nx.get_node_attributes(graph, "chemin")
    acquisition_set = nx.get_node_attributes(graph, "chemin_set")
    value_chemin = nx.get_node_attributes(graph, "value_chemin")

    for node in list_nodes:
        best_value = 0

        if acquisition_request_id[node] == "first":
            best_parent = graph.number_of_nodes()
            for parent in graph.predecessors(node):
                if value_chemin[parent] > best_value:
                    best_parent = parent
                    best_value = value_chemin[parent]
            chemin[node] = chemin[best_parent]
            value_chemin[node] = best_value

        elif acquisition_request_id[node] == "final":
            best_parent = 1
            for parent in graph.predecessors(node):
                if value_chemin[parent] > best_value:
                    best_parent = parent
                    best_value = value_chemin[parent]
            chemin[node] = chemin[best_parent]
            value_chemin[node] = best_value

        else:
            best_parent = None
            for parent in graph.predecessors(node):
                if acquisition_request_id[node] not in acquisition_set[parent]:
                    if value_chemin[parent] + value[node] > best_value:
                        best_value = value_chemin[parent] + value[node]
                        best_parent = parent
                """
                else:
                    if value_chemin[parent] > best_value:
                        best_value = value_chemin[parent]
                        best_parent = parent
                """
            if best_parent is not None:
                chemin[node] = deepcopy(chemin[best_parent])
                chemin[node].append(node)
                acquisition_set[node] = deepcopy(acquisition_set[best_parent])
                acquisition_set[node].add(acquisition_request_id[node])
                value_chemin[node] = best_value

    return value_chemin, chemin, acquisition_set


def long_path_DAG_with_memory_way(graph: nx.DiGraph, nb_way_memory: int, list_nodes: Iterable = None,) -> Tuple[dict[int, int], list[list[list[int]]], list[list[set[int]]]]:
    """
    @param nb_way_memory: the number of better path we want to keep in memory
    @param graph: DAG corresponding to the problem to solve
    @param list_nodes: list of nodes of DAG in topological order
    @return: the better value for each node corresponding to the better path to this node, the nb_way_memory better path, the nb_way_memory set of acquisition captured by the nb_way_memory better path
    """
    if list_nodes is None:
        list_nodes = nx.topological_sort(graph)
    acquisition_request_id = nx.get_node_attributes(graph, "acquisition_request_id")
    value = nx.get_node_attributes(graph, "value")

    chemin = nx.get_node_attributes(graph, "chemin")
    acquisition_set = nx.get_node_attributes(graph, "chemin_set")
    value_chemin = nx.get_node_attributes(graph, "value_chemin")

    all_acquisition_set = []
    all_chemin = []

    for node in list_nodes:
        best_value = 0
        if acquisition_request_id[node] == "first":
            best_parent = graph.number_of_nodes()
            for parent in graph.predecessors(node):
                if value_chemin[parent] > best_value:
                    best_parent = parent
                    best_value = value_chemin[parent]
            chemin[node] = chemin[best_parent]
            value_chemin[node] = best_value
            all_acquisition_set.append([])
            all_chemin.append([])

        elif acquisition_request_id[node] == "final":
            best_parent = 1
            for parent in graph.predecessors(node):
                if value_chemin[parent] > best_value:
                    best_parent = parent
                    best_value = value_chemin[parent]
            chemin[node] = chemin[best_parent]
            value_chemin[node] = best_value
            all_acquisition_set.append([])
            all_chemin.append([])

        else:
            best_parent = None
            value_better_parent = np.zeros(nb_way_memory)
            list_better_parent = np.zeros(nb_way_memory, dtype=int)
            min_parent = 0
            for parent in graph.predecessors(node):
                if acquisition_request_id[node] not in acquisition_set[parent]:
                    if value_chemin[parent] + value[node] > best_value:
                        best_value = value_chemin[parent] + value[node]
                        best_parent = parent
                        index = np.argmin(value_better_parent)
                        value_better_parent[index] = value_chemin[parent] + value[node]
                        list_better_parent[index] = parent
                        min_parent = np.min(value_better_parent)

                    elif value_chemin[parent] + value[node] > min_parent:
                        index = np.argmin(value_better_parent)
                        value_better_parent[index] = value_chemin[parent] + value[node]
                        list_better_parent[index] = parent
                        min_parent = np.min(value_better_parent)

            acq = []
            ch = []
            for x in list_better_parent:
                if x != 0:
                    temp_way = deepcopy(acquisition_set[x])
                    temp_way.add(acquisition_request_id[node])
                    acq.append(temp_way)
                    temp_way = deepcopy(chemin[x])
                    temp_way.append(node)
                    ch.append(temp_way)
            all_acquisition_set.append(acq)
            all_chemin.append(ch)

            if best_parent is not None:
                chemin[node] = deepcopy(chemin[best_parent])
                chemin[node].append(node)
                acquisition_set[node] = deepcopy(acquisition_set[best_parent])
                acquisition_set[node].add(acquisition_request_id[node])
                value_chemin[node] = best_value

    return value_chemin, all_chemin, all_acquisition_set


def long_path_DAG_backward_forward(graph: nx.DiGraph) -> Tuple[float, list[int]]:
    """
    @param graph: DAG who represent the problem
    @return: path and path's value of the better path find
    """

    value_request = {graph.nodes[n]["acquisition_request_id"]: graph.nodes[n]["value"] for n in graph.nodes()}
    nb_nodes = len(graph.nodes())
    list_nodes = graph.nodes()

    value_chemin, chemin, acquisition = long_path_DAG(graph, list_nodes)
    graph_reverse = graph.reverse(copy=False)
    value_chemin_reverse, chemin_reverse, acquisition_reverse = long_path_DAG(graph_reverse, reversed(list(graph.nodes())))

    best_value = max(value_chemin_reverse[1], value_chemin[nb_nodes])
    if value_chemin_reverse[1] > value_chemin[nb_nodes]:
        chemin_reverse[1].reverse()
        best_chemin = chemin_reverse[1]
    else:
        best_chemin = chemin[nb_nodes]

    del value_chemin
    del value_chemin_reverse

    for node in list_nodes:
        """
        intersect = acquisition[node].intersection(acquisition_reverse[node])
        s = sum([value_request[x] for x in intersect])
        new_eval = value_chemin[node]+value_chemin_reverse[node]-s
        """
        u = acquisition[node].union(acquisition_reverse[node])
        new_eval = sum([value_request[x] for x in u])
        if new_eval > best_value:
            best_value = new_eval
            if len(chemin_reverse[node]) > 1 and chemin_reverse[node][0] > chemin_reverse[node][1]:
                chemin_reverse[node].reverse()
            if len(chemin_reverse[node]) >= 1 and chemin_reverse[node][0] == node:
                chemin_reverse[node].pop(0)
            best_chemin = chemin[node] + chemin_reverse[node]
    return best_value, best_chemin


def long_path_DAG_backward_forward_with_memory_way(graph: nx.DiGraph, nb_way_memory: int) -> Tuple[float, list[int]]:
    """
    @param nb_way_memory: the number of better path we want to keep in memory
    @param graph: DAG who represent the problem
    @return: path and path's value of the better path find
    """

    value_request = {graph.nodes[n]["acquisition_request_id"]: graph.nodes[n]["value"] for n in graph.nodes()}
    nb_nodes = len(graph.nodes())
    list_nodes = graph.nodes()

    value_chemin, all_chemin, all_acquisition_set = long_path_DAG_with_memory_way(graph, nb_way_memory, list_nodes)
    graph_reverse = graph.reverse(copy=False)
    value_chemin_reverse, all_chemin_reverse, all_acquisition_set_reverse = long_path_DAG_with_memory_way(graph_reverse, nb_way_memory, reversed(list(graph.nodes())))

    all_acquisition_set_reverse.reverse()
    all_chemin_reverse.reverse()

    best_value = max(value_chemin_reverse[1], value_chemin[nb_nodes])
    best_chemin = []

    for node in list_nodes:
        j = 0
        for acquisition_set in all_acquisition_set[node-1]:
            i = 0
            for acquisition_set_reverse in all_acquisition_set_reverse[node-1]:
                u = acquisition_set.union(acquisition_set_reverse)
                new_eval = sum([value_request[x] for x in u])
                if new_eval >= best_value:
                    best_value = new_eval
                    if len(all_chemin_reverse[node-1][i]) > 1 and all_chemin_reverse[node-1][i][0] > all_chemin_reverse[node-1][i][1]:
                        all_chemin_reverse[node-1][i].reverse()
                    if len(all_chemin_reverse[node-1][i]) >= 1 and all_chemin_reverse[node-1][i][0] == node:
                        all_chemin_reverse[node-1][i].pop(0)
                    best_chemin = all_chemin[node-1][j] + all_chemin_reverse[node-1][i]
                i += 1
        j += 1
    return best_value, best_chemin


class SppBackwardForwardSolver(SppSolver, Hyperparametrizable):
    hyperparameters = [
        IntegerHyperparameter(name="nb_way_memory", low=1, high=10, default=1),
    ]

    def __init__(self, problem: SppProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)

    def solve(self, callbacks: Optional[List[Callback]] = None, **kwargs: Any) -> ResultStorage:

        kwargs = self.complete_with_default_hyperparameters(kwargs)

        if kwargs["nb_way_memory"] > 1:
            best_value, best_chemin = long_path_DAG_backward_forward_with_memory_way(self.problem.graph, nb_way_memory=kwargs["nb_way_memory"])
        else:
            best_value, best_chemin = long_path_DAG_backward_forward(self.problem.graph)

        sol = SppSolution(problem=self.problem, solution_path=best_chemin, solution_value=best_value)
        fit = self.aggreg_from_sol(sol)
        return self.create_result_storage(
            [(sol, fit)],
        )


class SppForwardSolver(SppSolver, Hyperparametrizable):
    hyperparameters = [
        IntegerHyperparameter(name="nb_way_memory", low=1, high=10, default=1),
    ]

    def __init__(self, problem: SppProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)

    def solve(self, callbacks: Optional[List[Callback]] = None, **kwargs: Any) -> ResultStorage:

        kwargs = self.complete_with_default_hyperparameters(kwargs)

        if kwargs["nb_way_memory"] > 1:
            value_chemin, all_chemin, all_acquisition_set = long_path_DAG_with_memory_way(
                self.problem.graph, kwargs["nb_way_memory"], self.problem.graph.nodes())
        else:
            value_chemin, all_chemin, all_acquisition_set = long_path_DAG(
                self.problem.graph, self.problem.graph.nodes())

        sol = SppSolution(problem=self.problem, solution_path=all_chemin[len(self.problem.graph.nodes())], solution_value=value_chemin[len(self.problem.graph.nodes())])
        fit = self.aggreg_from_sol(sol)
        return self.create_result_storage(
            [(sol, fit)],
        )


class SppBackwardSolver(SppSolver, Hyperparametrizable):
    hyperparameters = [
        IntegerHyperparameter(name="nb_way_memory", low=1, high=10, default=1),
    ]

    def __init__(self, problem: SppProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)

    def solve(self, callbacks: Optional[List[Callback]] = None, **kwargs: Any) -> ResultStorage:

        kwargs = self.complete_with_default_hyperparameters(kwargs)

        graph_reverse = self.problem.graph.reverse(copy=False)
        if kwargs["nb_way_memory"] > 1:
            value_chemin_reverse, all_chemin_reverse, all_acquisition_set_reverse = long_path_DAG_with_memory_way(
                graph_reverse, kwargs["nb_way_memory"], reversed(list(self.problem.graph.nodes())))
        else:
            value_chemin_reverse, all_chemin_reverse, all_acquisition_set_reverse = long_path_DAG(
                graph_reverse, reversed(list(self.problem.graph.nodes())))

        sol = SppSolution(problem=self.problem, solution_path=all_chemin_reverse[1], solution_value=value_chemin_reverse[1])
        fit = self.aggreg_from_sol(sol)
        return self.create_result_storage(
            [(sol, fit)],
        )
