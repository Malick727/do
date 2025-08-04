import time
from copy import deepcopy
import random
from typing import Optional, List, Any, Iterable, Tuple, Hashable, Callable
import networkx as nx
import numpy as np
from discrete_optimization.generic_tools.callbacks.callback import Callback, CallbackList
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (IntegerHyperparameter,
                                                                                CategoricalHyperparameter,
                                                                                FloatHyperparameter)
from discrete_optimization.generic_tools.hyperparameters.hyperparametrizable import Hyperparametrizable
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from satellite_path_planning.solver.spp_solver import SppSolver
from satellite_path_planning.problem import SppProblem, SppSolution


def long_path_DAG(graph: nx.DiGraph, list_nodes: Iterable = None,
                  random_pick: float = 0) -> Tuple[dict[int, float], dict[int, list[int]], dict[int, set]]:
    """
    @param graph: DAG corresponding to the problem to solve
    @param list_nodes: list of nodes of DAG in topological order
    @param random_pick: some proba where to pick random best parent
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
                if random_pick == 0:
                    if acquisition_request_id[node] not in acquisition_set[parent]:
                        if value_chemin[parent] + value[node] > best_value:
                            best_value = value_chemin[parent] + value[node]
                            best_parent = parent
                else:
                    if acquisition_request_id[node] not in acquisition_set[parent]:
                        if value_chemin[parent] + value[node] > best_value or random.random() < random_pick:
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


def long_path_DAG_with_memory_way(graph: nx.DiGraph, nb_way_memory: int, list_nodes: Iterable = None,
                                  random_pick: float = 0) -> Tuple[dict[int, int], list[list[list[int]]],
                                                                   list[list[set[int]]]]:
    """
    @param nb_way_memory: the number of better path we want to keep in memory
    @param graph: DAG corresponding to the problem to solve
    @param list_nodes: list of nodes of DAG in topological order
    @param random_pick: some proba where to pick random best parent
    @return: the better value for each node corresponding to the better path to this node,
    the nb_way_memory better path, the nb_way_memory set of acquisition captured by the nb_way_memory better path
    """
    if list_nodes is None:
        list_nodes = nx.topological_sort(graph)
    acquisition_request_id = nx.get_node_attributes(graph, "acquisition_request_id")
    value = nx.get_node_attributes(graph, "value")

    chemin = deepcopy(nx.get_node_attributes(graph, "chemin"))
    acquisition_set = deepcopy(nx.get_node_attributes(graph, "chemin_set"))
    value_chemin = nx.get_node_attributes(graph, "value_chemin")

    all_acquisition_set = {}
    all_chemin = {}

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
            all_acquisition_set[node] = []
            all_chemin[node] = [chemin[best_parent]]

        elif acquisition_request_id[node] == "final":
            best_parent = 1
            for parent in graph.predecessors(node):
                if value_chemin[parent] > best_value:
                    best_parent = parent
                    best_value = value_chemin[parent]
            chemin[node] = chemin[best_parent]
            value_chemin[node] = best_value
            all_acquisition_set[node] = []
            all_chemin[node] = [chemin[best_parent]]
        else:
            best_parent = None
            value_better_parent = np.zeros(nb_way_memory)
            list_better_parent = np.zeros(nb_way_memory, dtype=int)
            min_parent = 0
            for parent in graph.predecessors(node):
                if acquisition_request_id[node] not in acquisition_set[parent]:
                    if value_chemin[parent] + value[node] > best_value or random.random() < random_pick:
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
            all_acquisition_set[node] = acq
            all_chemin[node] = ch

            if best_parent is not None:
                chemin[node] = deepcopy(chemin[best_parent])
                chemin[node].append(node)
                acquisition_set[node] = deepcopy(acquisition_set[best_parent])
                acquisition_set[node].add(acquisition_request_id[node])
                value_chemin[node] = best_value

    return value_chemin, all_chemin, all_acquisition_set


import networkx as nx
from typing import Dict, List, Set, Tuple, Any


def long_path_dag_with_memory_corrected(graph: nx.DiGraph,
                                        nb_way_memory: int,
                                        list_nodes: Iterable = None,
                                        random_pick = 0) -> Tuple[
    Dict[Any, List[float]], Dict[Any, List[List[Any]]], Dict[Any, List[Set[Any]]]]:
    """
    Computes the N-best paths to each node in a Directed Acyclic Graph (DAG).

    The value of a path is the sum of its nodes' values, with the constraint that
    the value of a node is only counted once per unique 'acquisition_request_id'.

    Args:
        graph (nx.DiGraph): The input graph. Nodes must have 'acquisition_request_id' and 'value' attributes.
        nb_way_memory (int): The number of best paths to store for each node.

    Returns:
        A tuple containing three dictionaries, mapping each node to:
        - A list of the values of its N-best paths.
        - A list of its N-best paths (as lists of nodes).
        - A list of its N-best acquisition sets.
    """
    # Helper dictionary to store the computed N-best solutions for each node.
    # Format: {node_id: [(value, path, acquisition_set), ...]}
    node_solutions: Dict[Any, List[Tuple[float, List[Any], Set[Any]]]] = {}

    # Extract node attributes for quick access
    acquisition_ids = nx.get_node_attributes(graph, "acquisition_request_id")
    node_values = nx.get_node_attributes(graph, "value")
    if list_nodes is None:
        list_nodes = nx.topological_sort(graph)
    # Process nodes in topological order to ensure predecessors are computed first.
    for node in list_nodes:
        candidate_solutions = []

        # Get predecessors of the current node
        predecessors = list(graph.predecessors(node))

        if not predecessors:
            # This is a source node (no predecessors)
            # Its path starts with itself.
            acq_id = acquisition_ids.get(node)
            initial_value = node_values.get(node, 0)
            candidate_solutions.append((initial_value, [node], {acq_id}))
        else:
            # This node has predecessors, extend their paths.
            for pred in predecessors:
                # Retrieve the stored N-best solutions for the predecessor.
                if pred not in node_solutions:
                    continue

                for p_value, p_path, p_acq_set in node_solutions[pred]:
                    acq_id = acquisition_ids.get(node)

                    # Calculate new value: add node's value only if its acquisition is new.
                    new_value = p_value
                    if acq_id not in p_acq_set:
                        new_value += node_values.get(node, 0)

                    # Create the new path and acquisition set
                    new_path = p_path + [node]
                    new_acq_set = p_acq_set.union({acq_id})

                    candidate_solutions.append((new_value, new_path, new_acq_set))

        # --- Selection Step ---
        # Sort all generated candidate solutions for the current node by value, descending.
        candidate_solutions.sort(key=lambda x: x[0], reverse=True)

        # Filter out solutions with duplicate acquisition sets, keeping the highest-valued one.
        unique_solutions = []
        seen_acq_sets = set()
        for value, path, acq_set in candidate_solutions:
            # Use a frozenset as a key because sets are not hashable
            frozen_acq_set = frozenset(acq_set)
            if frozen_acq_set not in seen_acq_sets:
                unique_solutions.append((value, path, acq_set))
                seen_acq_sets.add(frozen_acq_set)

        # Store the top 'nb_way_memory' unique solutions for the current node.
        node_solutions[node] = unique_solutions[:nb_way_memory]
        if random.random() <= random_pick:
            node_solutions[node] = random.sample(unique_solutions, min(nb_way_memory,
                                                                       len(unique_solutions)))

    # --- Formatting the Output ---
    # Deconstruct the 'node_solutions' dictionary into the three separate
    # dictionaries required by the original solver structure.
    final_values: Dict[Any, List[float]] = {}
    final_paths: Dict[Any, List[List[Any]]] = {}
    final_acq_sets: Dict[Any, List[Set[Any]]] = {}

    for node, solutions in node_solutions.items():
        if solutions:
            # Unzip the list of tuples into separate lists
            values, paths, acq_sets = zip(*solutions)
            final_values[node] = list(values)
            final_paths[node] = list(paths)
            final_acq_sets[node] = list(acq_sets)
        else:
            # Handle nodes that might not have paths if the graph is disconnected
            final_values[node] = []
            final_paths[node] = []
            final_acq_sets[node] = []
    return final_values, final_paths, final_acq_sets


def long_path_DAG_backward_forward(graph: nx.DiGraph,
                                   random_pick: float = 0) -> Tuple[float, list[int]]:
    """
    @param graph: DAG who represent the problem
    @param random_pick: random pick proba
    @return: path and path's value of the better path find
    """

    value_request = {graph.nodes[n]["acquisition_request_id"]: graph.nodes[n]["value"] for n in graph.nodes()}
    nb_nodes = len(graph.nodes())
    list_nodes = graph.nodes()

    value_chemin, chemin, acquisition = long_path_DAG(graph=graph, list_nodes=list_nodes,
                                                      random_pick=random_pick)
    graph_reverse = graph.reverse(copy=False)
    value_chemin_reverse, chemin_reverse, acquisition_reverse = long_path_DAG(graph=graph_reverse,
                                                                              list_nodes=reversed(list(graph.nodes())),
                                                                              random_pick=random_pick)

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


def long_path_DAG_backward_forward_with_memory_way(graph: nx.DiGraph, nb_way_memory: int,
                                                   random_pick: float = 0) -> Tuple[float, list[int]]:
    """
    @param nb_way_memory: the number of better path we want to keep in memory
    @param graph: DAG who represent the problem
    @param random_pick: some proba where to pick random best parent
    @return: path and path's value of the better path find
    """

    value_request = {graph.nodes[n]["acquisition_request_id"]: graph.nodes[n]["value"] for n in graph.nodes()}
    nb_nodes = len(graph.nodes())
    list_nodes = graph.nodes()

    value_chemin, all_chemin, all_acquisition_set = long_path_dag_with_memory_corrected(graph=graph,
                                                                                        nb_way_memory=nb_way_memory,
                                                                                        list_nodes=list_nodes,
                                                                                        random_pick=random_pick)
    graph_reverse = graph.reverse(copy=False)
    value_chemin_reverse, all_chemin_reverse, all_acquisition_set_reverse = (
        long_path_dag_with_memory_corrected(graph=graph_reverse,
                                            nb_way_memory=nb_way_memory,
                                            list_nodes=reversed(list(graph.nodes())),
                                            random_pick=random_pick))
    best_value = max(value_chemin_reverse[1][0], value_chemin[nb_nodes][0])
    best_chemin = []

    for node in list_nodes:
        j = 0
        for acquisition_set in all_acquisition_set[node]:
            i = 0
            for acquisition_set_reverse in all_acquisition_set_reverse[node]:
                u = acquisition_set.union(acquisition_set_reverse)
                new_eval = sum([value_request[x] for x in u])
                if new_eval >= best_value:
                    best_value = new_eval
                    if (len(all_chemin_reverse[node][i]) > 1
                            and all_chemin_reverse[node][i][0] > all_chemin_reverse[node][i][1]):
                        all_chemin_reverse[node][i].reverse()
                    if len(all_chemin_reverse[node][i]) >= 1 and all_chemin_reverse[node][i][0] == node:
                        all_chemin_reverse[node][i].pop(0)
                    best_chemin = all_chemin[node][j] + all_chemin_reverse[node][i]
                i += 1
        j += 1
    return best_value, best_chemin


def find_best_bidirectional_path(graph: nx.DiGraph, nb_way_memory: int,
                                 random_pick: float = 0) -> Tuple[float, List[Any]]:
    """
    Finds the highest-value path in a DAG by running a forward and backward pass
    and combining the best sub-paths at each node.

    This function serves as a corrected and clearer replacement for
    'long_path_DAG_backward_forward_with_memory_way'.

    Args:
        graph (nx.DiGraph): The problem graph. Nodes must have 'value' and
                            'acquisition_request_id' attributes.
        nb_way_memory (int): The number of best paths to store at each node
                             during the forward and backward passes.

    Returns:
        A tuple containing:
        - The value of the best path found.
        - The best path as a list of node IDs.
    """
    # --- 1. Pre-computation ---
    # Create a mapping from acquisition ID to its value for quick lookups.
    value_per_request = {
        data["acquisition_request_id"]: data["value"]
        for _, data in graph.nodes(data=True)
        if "acquisition_request_id" in data
    }
    list_nodes = list(graph.nodes())
    # --- 2. Forward Pass ---
    # Calculate the N-best paths from the start to every node in the graph.
    _, forward_paths, forward_sets = long_path_dag_with_memory_corrected(
        graph, nb_way_memory, list_nodes=list_nodes, random_pick=random_pick
    )

    # --- 3. Backward Pass ---
    # Calculate the N-best paths from the end to every node by using a reversed graph.
    reversed_graph = graph.reverse(copy=True)
    _, backward_paths_rev, backward_sets = long_path_dag_with_memory_corrected(
        reversed_graph, nb_way_memory, list_nodes=list_nodes[::-1],
        random_pick=random_pick
    )

    # The paths from the backward pass are reversed; fix them.
    backward_paths = {
        node: [path[::-1] for path in paths]
        for node, paths in backward_paths_rev.items()
    }

    # --- 4. Combination Step ---
    best_value = -1.0
    best_path = []

    # Iterate through every node, treating it as a potential "meeting point"
    # for a forward and a backward path.
    for node in graph.nodes():
        if node not in forward_paths or node not in backward_paths:
            continue

        # Get all stored forward and backward solutions for this meeting node.
        paths_ending_at_node = forward_paths[node]
        sets_ending_at_node = forward_sets[node]
        paths_starting_at_node = backward_paths[node]
        sets_starting_at_node = backward_sets[node]

        # Check every combination of a forward and a backward path.
        for i in range(len(paths_ending_at_node)):
            path_fwd, set_fwd = paths_ending_at_node[i], sets_ending_at_node[i]

            for j in range(len(paths_starting_at_node)):
                path_bwd, set_bwd = paths_starting_at_node[j], sets_starting_at_node[j]

                # Combine the acquisition sets and calculate the total value.
                combined_set = set_fwd.union(set_bwd)
                current_value = sum(value_per_request.get(req, 0) for req in combined_set)

                if current_value > best_value:
                    best_value = current_value
                    # Join the paths, removing the duplicated meeting node from the backward path.
                    best_path = path_fwd + path_bwd[1:]

    # As a final check, evaluate the simple forward-only path to the end node.
    # This covers cases where the best path doesn't benefit from a backward segment.
    final_node = max(graph.nodes())
    if final_node in forward_paths and forward_paths[final_node]:
        path_fwd_only = forward_paths[final_node][0]
        set_fwd_only = forward_sets[final_node][0]
        value_fwd_only = sum(value_per_request.get(req, 0) for req in set_fwd_only)
        if value_fwd_only > best_value:
            best_value = value_fwd_only
            best_path = path_fwd_only

    return best_value, best_path

def repeated_call_to_function(func: Callable,
                              graph: nx.DiGraph,
                              forward: bool,
                              nb_iteration: int,
                              time_limit: float = 10,
                              callback: Optional[Callback] = None,
                              res: ResultStorage = None,
                              problem: SppProblem = None,
                              solver: SppSolver = None):
    t = time.perf_counter()
    start = min(graph.nodes())
    end = max(graph.nodes())
    if not forward:
        start = max(graph.nodes())
        end = min(graph.nodes())
    esd, path_d, _ = func(graph,
                          random_pick=0)
    esd_best = (deepcopy(esd), path_d)

    def get_sol(esd: tuple):
        if callback is not None:
            if forward:
                path = esd[1][len(problem.graph.nodes())]
            else:
                path = esd[1][1]
            if isinstance(path[0], list):
                path = path[0]
            if not forward:
                path = path[::-1]
            sol = SppSolution(problem=problem,
                              solution_path=path)
            fit = solver.aggreg_from_sol(sol)
            res.append((sol, fit))
            callback.on_step_end(0, res, solver)
    get_sol(esd_best)
    best_sol = esd[end]
    print("Current sol", best_sol)
    for k in range(nb_iteration):
        if time_limit is not None and time.perf_counter() - t > time_limit:
            break
        esd, path_d, _ = func(graph)
        if esd[end] > best_sol:
            best_sol = esd[end]
            esd_best = (esd, path_d)
            print("Improved ", best_sol)
            get_sol(esd_best)
    return best_sol, esd_best


class SppBackwardForwardSolver(SppSolver, Hyperparametrizable):
    hyperparameters = [
        IntegerHyperparameter(name="nb_way_memory", low=1, high=10, default=1),
    ]

    def __init__(self, problem: SppProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)

    def solve(self, callbacks: Optional[List[Callback]] = None, **kwargs: Any) -> ResultStorage:
        callback = CallbackList(callbacks)
        callback.on_solve_start(self)
        kwargs = self.complete_with_default_hyperparameters(kwargs)

        if kwargs["nb_way_memory"] > 1:
            best_value, best_chemin = find_best_bidirectional_path(self.problem.graph,
                                                                   nb_way_memory=kwargs["nb_way_memory"])
        else:
            best_value, best_chemin = long_path_DAG_backward_forward(self.problem.graph)

        sol = SppSolution(problem=self.problem, solution_path=best_chemin, solution_value=best_value)
        fit = self.aggreg_from_sol(sol)
        res = self.create_result_storage([(sol, fit)])
        callback.on_solve_end(res, self)
        return res


class SppSolverRandomPick(SppSolver, Hyperparametrizable):
    hyperparameters = [
        IntegerHyperparameter(name="nb_way_memory", low=1, high=10, default=1),
        CategoricalHyperparameter(name="forward", choices=[True, False], default=True),
        FloatHyperparameter(name="random_pick", low=0, high=1,
                            step=0.1, default=0.1),
    ]

    def __init__(self, problem: SppProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)

    def solve(self, callbacks: Optional[List[Callback]] = None,
              time_limit: int = 10,
              nb_iterations: int = 100,
              **kwargs: Any) -> ResultStorage:
        from functools import partial
        callback = CallbackList(callbacks)
        callback.on_solve_start(self)
        res = self.create_result_storage([])
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        random_pick = kwargs['random_pick']
        forward = kwargs["forward"]
        nb_way_memory = kwargs['nb_way_memory']
        if forward:
            list_nodes = list(self.problem.graph.nodes())
            gr = self.problem.graph
        else:
            list_nodes = list(self.problem.graph.nodes())[::-1]
            gr = self.problem.graph.reverse()
        if nb_way_memory > 1:
            func = partial(long_path_dag_with_memory_corrected,
                           random_pick=random_pick,
                           list_nodes=list_nodes,
                           nb_way_memory=nb_way_memory)
        else:
            func = partial(long_path_DAG, random_pick=random_pick,
                           list_nodes=list_nodes)
        best_value, best_data = repeated_call_to_function(func,
                                                          graph=gr,
                                                          forward=forward,
                                                          callback=callback,
                                                          res=res,
                                                          problem=self.problem,
                                                          solver=self,
                                                          nb_iteration=nb_iterations,
                                                          time_limit=time_limit)
        callback.on_solve_end(res, self)
        return res


class SppForwardSolver(SppSolver, Hyperparametrizable):
    hyperparameters = [
        IntegerHyperparameter(name="nb_way_memory", low=1, high=10, default=1),
    ]

    def __init__(self, problem: SppProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)

    def solve(self, callbacks: Optional[List[Callback]] = None, **kwargs: Any) -> ResultStorage:
        callback = CallbackList(callbacks)
        callback.on_solve_start(self)
        kwargs = self.complete_with_default_hyperparameters(kwargs)

        if kwargs["nb_way_memory"] > 1:
            value_chemin, all_chemin, all_acquisition_set = long_path_dag_with_memory_corrected(
                self.problem.graph, nb_way_memory=kwargs["nb_way_memory"],
                list_nodes=self.problem.graph.nodes())
            chemin = all_chemin[len(self.problem.graph.nodes())][0]
        else:
            value_chemin, all_chemin, all_acquisition_set = long_path_DAG(
                self.problem.graph, self.problem.graph.nodes())
            chemin = all_chemin[len(self.problem.graph.nodes())]

        sol = SppSolution(problem=self.problem, solution_path=chemin)
        fit = self.aggreg_from_sol(sol)
        res = self.create_result_storage([(sol, fit)])
        callback.on_solve_end(res=res, solver=self)
        return res


class SppBackwardSolver(SppSolver, Hyperparametrizable):
    hyperparameters = [
        IntegerHyperparameter(name="nb_way_memory", low=1, high=10, default=1),
    ]

    def __init__(self, problem: SppProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)

    def solve(self, callbacks: Optional[List[Callback]] = None, **kwargs: Any) -> ResultStorage:
        callback = CallbackList(callbacks)
        callback.on_solve_start(self)
        kwargs = self.complete_with_default_hyperparameters(kwargs)

        graph_reverse = self.problem.graph.reverse(copy=False)
        if kwargs["nb_way_memory"] > 1:
            value_chemin_reverse, all_chemin_reverse, all_acquisition_set_reverse = long_path_dag_with_memory_corrected(
                graph=graph_reverse,
                nb_way_memory=kwargs["nb_way_memory"],
                list_nodes=list(self.problem.graph.nodes())[::-1])
            chemin = all_chemin_reverse[1][0][::-1]
        else:
            value_chemin_reverse, all_chemin_reverse, all_acquisition_set_reverse = long_path_DAG(
                graph_reverse, reversed(list(self.problem.graph.nodes())))
            chemin = all_chemin_reverse[1][::-1]

        sol = SppSolution(problem=self.problem,
                          solution_path=chemin)
        fit = self.aggreg_from_sol(sol)
        res = self.create_result_storage([(sol, fit)])
        callback.on_solve_end(res=res, solver=self)
        return res
