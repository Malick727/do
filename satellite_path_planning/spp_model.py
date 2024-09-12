from copy import deepcopy
from typing import Dict, Type

from networkx import DiGraph

from discrete_optimization.generic_tools.do_problem import (
    EncodingRegister,
    ObjectiveRegister,
    Problem,
    Solution, ModeOptim, ObjectiveHandling, ObjectiveDoc, TypeObjective, TypeAttribute
)


class SppSolution(Solution):

    def __init__(self, problem: "SppProblem", solution_path: list[int]):
        self.problem = problem
        self.solution_path = solution_path

    def __str__(self) -> str:
        pass

    def copy(self) -> "Solution":
        return SppSolution(problem=self.problem, solution_path=deepcopy(self.solution_path))

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

    def evaluate(self, variable: SppSolution) -> Dict[str, float]:
        return {
            "value": self.func(variable.solution_path),
        }

    def satisfy(self, variable: SppSolution) -> bool:
        return self.valid_path(variable)

    def get_attribute_register(self) -> EncodingRegister:
        return EncodingRegister(
            dict_attribute_to_type={
                "chosen": {
                    "name": "solution_path",
                    "type": [TypeAttribute.LIST_INTEGER],
                }
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
        for i in range(len(variable.solution_path - 1)):
            if (variable.solution_path[i], variable.solution_path[i+1]) not in self.graph.edges:
                return False
        return True

    def func(self, solution_path) -> float:
        val = 0
        for node in solution_path:
            val += self.acquisition_dict[self.graph.nodes[node]["acquisition_request"]]
        return val

