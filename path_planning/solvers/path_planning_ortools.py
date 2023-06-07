import logging
from typing import Any, Hashable, Set, Tuple

import numpy as np
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from ortools.graph.python import min_cost_flow

from path_planning.path_planning import PathPlanning, PathSolution

logger = logging.getLogger(__name__)


def build_path_from_set_edges(
    set_edges: Set[Tuple[int, int]], source: Hashable, target: Hashable
):
    edges = set(set_edges)
    path = [source]
    while True:
        element = next((e for e in edges if e[0] == path[-1]), None)
        path += [element[1]]
        if element[1] == target:
            break
        print(element, path)
    return path


class PathPlanningOrtools(SolverDO):
    def __init__(
        self,
        problem: PathPlanning,
        params_objective_function: ParamsObjectiveFunction = None,
    ):
        self.problem = problem
        (
            self.aggreg_sol,
            self.aggreg_from_dict_values,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            self.problem, params_objective_function=params_objective_function
        )

    def solve(self, **kwargs: Any) -> ResultStorage:
        # Instantiate a SimpleMinCostFlow solver.
        smcf = min_cost_flow.SimpleMinCostFlow()
        # Define an array of supplies at each node.
        supplies = [0 for i in range(self.problem.nb_nodes)]
        supplies[self.problem.index_source] = 1
        supplies[self.problem.index_sink] = -1
        edges = self.problem.graph.get_edges()
        edges_index = [
            (self.problem.index_nodes[e[0]], self.problem.index_nodes[e[1]])
            for e in edges
        ]
        start_nodes = np.array([e[0] for e in edges_index])
        end_nodes = np.array([e[1] for e in edges_index])
        unit_costs = np.array(
            [
                max(1, int(self.problem.matrix_length[s, e] * 10000))
                for s, e in zip(start_nodes, end_nodes)
            ]
        )
        capacities = np.array([1 for i in range(start_nodes.shape[0])])
        # Add arcs, capacities and costs in bulk using numpy.
        all_arcs = smcf.add_arcs_with_capacity_and_unit_cost(
            start_nodes, end_nodes, capacities, unit_costs
        )
        # Add supply for each nodes.
        for i in range(len(supplies)):
            smcf.set_node_supply(i, supplies[i])
        status = smcf.solve()
        if status == smcf.INFEASIBLE:
            logger.info("Min flow solver failed.")
            raise Exception("Min flow solver failed.")
        logger.info(f"Minimum cost: {smcf.optimal_cost()}")
        solution_flows = smcf.flows(all_arcs)
        edges = set()
        for j in range(len(all_arcs)):
            flow = solution_flows[j]
            if flow > 0:
                edges.add((int(start_nodes[j]), int(end_nodes[j])))
                logger.info(
                    f"Edge {(start_nodes[j], end_nodes[j])}, flow={flow}, cost={unit_costs[j]*flow}"
                )
        path = build_path_from_set_edges(
            set_edges=edges,
            source=self.problem.index_source,
            target=self.problem.index_sink,
        )
        sol = PathSolution(problem=self.problem, path=path)
        fit = self.aggreg_sol(sol)
        logger.info(f"Fitness={fit}")
        return ResultStorage(
            list_solution_fits=[(sol, fit)],
            mode_optim=self.params_objective_function.sense_function,
        )
