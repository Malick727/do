from collections import defaultdict

import networkx as nx
from copy import deepcopy
from typing import Dict, Type

from networkx import DiGraph

from discrete_optimization.generic_tools.do_problem import (
    EncodingRegister,
    ObjectiveRegister,
    Problem,
    Solution, ModeOptim, ObjectiveHandling, ObjectiveDoc, TypeObjective
)
from discrete_optimization.generic_tools.encoding_register import EncodingRegister, ListInteger


class SppSolution(Solution):

    def __init__(self, problem: "SppProblem", solution_path: list[int], solution_value: float = None):
        self.problem = problem
        self.solution_path = solution_path
        self.solution_value = solution_value
        if self.solution_value is None:
            problem.func(solution_path=self.solution_path)

    def __str__(self) -> str:
        pass

    def copy(self) -> "Solution":
        return SppSolution(problem=self.problem, solution_path=deepcopy(self.solution_path), solution_value=self.solution_value)

    def change_problem(self, new_problem: "Problem") -> None:
        self.problem = new_problem


class SppProblem(Problem):

    def __init__(self, graph: DiGraph, acquisition_dict: Dict):
        self.graph = graph
        self.acquisition_dict = acquisition_dict
        self.nb_nodes = graph.number_of_nodes()
        self.ind_to_id = {}
        self.id_to_ind = {}
        ind = 0
        for node in graph.nodes:
            self.ind_to_id[ind] = node
            self.id_to_ind[node] = ind
            ind += 1

    def compute_predecessors_map(self):
        pred = defaultdict(lambda: set())
        for n0, n1 in self.graph.edges():
            pred[n1].add(n0)
        self.pred_maps = pred
        return pred


    def compute_transitive_closure_graph(self):
        graph = nx.transitive_closure_dag(self.graph,
                                          topo_order=sorted(list(self.graph.nodes)))
        self.closure_graph: nx.DiGraph = graph
        dict_acquisition_descendants = {node: {self.graph.nodes[s]["acquisition_request_id"]
                                               for s in nbrdict}
                                        for node, nbrdict in graph.adjacency()}
        self.node_to_acquisition_descendants = dict_acquisition_descendants


    def evaluate(self, variable: SppSolution) -> Dict[str, float]:
        return {
            "value": self.func(variable.solution_path),
        }

    def satisfy(self, variable: SppSolution) -> bool:
        return self.valid_path(variable)

    def get_attribute_register(self) -> EncodingRegister:
        return EncodingRegister(
            dict_attribute_to_type={
                "solution_path": ListInteger(length=self.nb_nodes)
            }
        )

    def get_solution_type(self) -> Type[Solution]:
        return SppSolution

    def get_objective_register(self) -> ObjectiveRegister:
        return ObjectiveRegister(
            objective_sense=ModeOptim.MAXIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc={
                "value": ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=1),
            },
        )

    def valid_path(self, variable) -> bool:
        for i in range(len(variable.solution_path) - 1):
            if (variable.solution_path[i], variable.solution_path[i+1]) not in self.graph.edges:
                print((variable.solution_path[i], variable.solution_path[i+1]), " not in graph")
                return False
        return True

    def func(self, solution_path) -> float:
        val = 0
        done_request = set()
        for node in solution_path:
            req = self.graph.nodes[node]["acquisition_request_id"]
            if req not in done_request:
                val += self.acquisition_dict.get(req, 0)
                done_request.add(req)
        return val

