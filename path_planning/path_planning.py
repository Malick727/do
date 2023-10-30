import operator
from copy import deepcopy
from enum import Enum
from typing import Dict, Hashable, List, Optional, Tuple, Type

import numpy as np
from discrete_optimization.generic_tools.do_problem import (
    EncodingRegister,
    ModeOptim,
    ObjectiveDoc,
    ObjectiveHandling,
    ObjectiveRegister,
    Problem,
    Solution,
    TypeObjective,
)
from discrete_optimization.generic_tools.graph_api import Graph


class Comparison(Enum):
    EQ = "="
    LEQ = "<="
    LE = "<"
    UP = ">"
    UEQ = ">="


operator_map = {
    Comparison.EQ: operator.eq,
    Comparison.LEQ: operator.le,
    Comparison.LE: operator.lt,
    Comparison.UEQ: operator.ge,
    Comparison.UP: operator.gt,
}

# Return positive values where the constraint is broken
delta_map = {
    Comparison.EQ: lambda x, y: abs(x - y),
    Comparison.LEQ: lambda x, y: x - y,
    Comparison.LE: lambda x, y: x - y,
    Comparison.UEQ: lambda x, y: y - x,
    Comparison.UP: lambda x, y: y - x,
}


class PathSolution(Solution):
    def __init__(self, problem: "PathPlanning", path: List[int]):
        self.problem = problem
        self.path = path

    def copy(self) -> "PathSolution":
        return deepcopy(self)

    def change_problem(self, new_problem: "PathPlanning") -> None:
        self.problem = new_problem


class NodeConstraints:
    def __init__(
        self,
        list_constraint: Optional[List[Tuple[List[Hashable], Comparison, int]]] = None,
    ):
        self.list_constraint = list_constraint
        if list_constraint is None:
            self.list_constraint = []
        self.nb_constraints = len(self.list_constraint)
        self.node_to_index_constraint = {}
        for i in range(self.nb_constraints):
            for index in self.list_constraint[i][0]:
                if index not in self.node_to_index_constraint:
                    self.node_to_index_constraint[index] = set()
                self.node_to_index_constraint[index].add(i)

    def is_empty(self):
        return self.nb_constraints == 0


def reindex_constraint(
    node_constraints: NodeConstraints, index_nodes: Dict[Hashable, int]
):
    list_constraints = []
    for i in range(len(node_constraints.list_constraint)):
        lc = node_constraints.list_constraint[i]
        list_constraints += [([index_nodes[x] for x in lc[0]], lc[1], lc[2])]
    return NodeConstraints(list_constraint=list_constraints)


class PathConstraints:
    # Object to specify constraints on the aggregated value along the chosen path.
    def __init__(
        self, list_constraint: Optional[List[Tuple[str, Comparison, float]]] = None
    ):
        self.list_constraint = list_constraint
        if list_constraint is None:
            self.list_constraint = []
        self.nb_constraints = len(self.list_constraint)
        self.attributes = set([x[0] for x in self.list_constraint])

    def is_empty(self):
        return self.nb_constraints == 0


class PathPlanning(Problem):
    def __init__(
        self,
        graph: Graph,
        source: Hashable,
        sink: Hashable,
        length_attribute: str,
        node_constraints: Optional[NodeConstraints] = None,
        path_constraints: Optional[PathConstraints] = None,
        reindex_node_constraints: bool = True
    ):
        self.graph = graph
        self.source = source
        self.sink = sink
        self.length_attribute = length_attribute
        self.nb_nodes = len(self.graph.nodes)
        self.index_nodes = {self.graph.nodes[i][0]: i for i in range(self.nb_nodes)}
        self.index_to_nodes = {i: self.graph.nodes[i][0] for i in range(self.nb_nodes)}
        self.index_source = self.index_nodes[self.source]
        self.index_sink = self.index_nodes[self.sink]
        (
            self.matrix_length,
            self.min_cost,
            self.max_cost,
        ) = self.compute_matrix_attribute(attribute=self.length_attribute)
        self.node_constraints = node_constraints
        if node_constraints is None:
            self.node_constraints = NodeConstraints()
        if reindex_node_constraints:
            self.node_constraints = reindex_constraint(
                node_constraints=self.node_constraints, index_nodes=self.index_nodes
            )
        else:
            self.node_constraints = node_constraints
        self.path_constraints = path_constraints
        if path_constraints is None:
            self.path_constraints = PathConstraints()
        self.matrix_length_attr = {}
        self.min_cost_attr = {}
        self.max_cost_attr = {}
        if not self.path_constraints.is_empty():
            for attr in self.path_constraints.attributes:
                (
                    self.matrix_length_attr[attr],
                    self.min_cost_attr[attr],
                    self.max_cost_attr[attr],
                ) = self.compute_matrix_attribute(attribute=attr)

    def change_source(self, new_source):
        self.source = new_source
        self.index_source = self.index_nodes[new_source]

    def change_sink(self, new_sink):
        self.sink = new_sink
        self.index_sink = self.index_nodes[new_sink]

    def change_length_attribute(self, length_attribute):
        if length_attribute != self.length_attribute:
            self.length_attribute = length_attribute
            attr = self.length_attribute
            if attr not in self.matrix_length_attr:
                (
                    self.matrix_length_attr[attr],
                    self.min_cost_attr[attr],
                    self.max_cost_attr[attr],
                ) = self.compute_matrix_attribute(attribute=attr)
            self.matrix_length, self.min_cost, self.max_cost = (
                self.matrix_length_attr[attr],
                self.min_cost_attr[attr],
                self.max_cost_attr[attr],
            )

    def only_negative_length(self):
        return self.max_cost <= 0

    def only_positive_length(self):
        return self.min_cost >= 0

    def mixed_positive_negative_length(self):
        return self.min_cost < 0 < self.max_cost

    def compute_matrix_attribute(self, attribute: Optional[str] = None):
        if attribute is None:
            attribute = self.length_attribute
        matrix = 1000 * np.ones((self.nb_nodes, self.nb_nodes))
        min_ = float("inf")
        max_ = -float("inf")
        for e in self.graph.edges:
            i1 = self.index_nodes[e[0]]
            i2 = self.index_nodes[e[1]]
            v = e[2].get(attribute, None)
            if v is not None:
                matrix[i1, i2] = v
                min_ = min(v, min_)
                max_ = max(v, max_)
            if self.graph.undirected:
                matrix[i2, i1] = v
        return matrix, min_, max_

    def evaluate(self, variable: PathSolution) -> Dict[str, float]:
        length = sum(
            self.matrix_length[i, j]
            for i, j in zip(variable.path[:-1], variable.path[1:])
        )
        cnstr_violated, _ = self.evaluate_node_constraints(variable)
        cnstr_violated_path, _ = self.evaluate_path_constraints(variable)
        return {
            "length": length,
            "constraint_node": cnstr_violated,
            "constraint_path": cnstr_violated_path,
        }

    def evaluate_node_constraints(self, variable: PathSolution):
        if self.node_constraints.is_empty():
            return 0, None
        values_constraints = {
            index: 0 for index in range(self.node_constraints.nb_constraints)
        }
        for i in variable.path:
            if i in self.node_constraints.node_to_index_constraint:
                for index in self.node_constraints.node_to_index_constraint[i]:
                    if index not in values_constraints:
                        values_constraints[index] = 0
                    values_constraints[index] += 1
        cnstr_violated = 0
        info = []
        for index in values_constraints:
            constr = self.node_constraints.list_constraint[index]
            if not operator_map[constr[1]](values_constraints[index], constr[2]):
                cnstr_violated += 1
                info += [index]
        return cnstr_violated, info

    def compute_cumulated_value_path(self, variable: PathSolution, attribute: str):
        if attribute not in self.matrix_length_attr:
            self.matrix_length_attr[attribute], _, _ = self.compute_matrix_attribute(
                attribute
            )
        return sum(
            self.matrix_length_attr[attribute][i, j]
            for i, j in zip(variable.path[:-1], variable.path[1:])
        )

    def compute_value_path(self, variable: PathSolution):
        return sum(
            self.matrix_length[i, j]
            for i, j in zip(variable.path[:-1], variable.path[1:])
        )

    def evaluate_path_constraints(self, variable: PathSolution):
        if self.path_constraints.is_empty():
            return 0, None
        info = []
        absolute_violation = 0
        vals_attribute = {}
        for attribute in self.path_constraints.attributes:
            vals_attribute[attribute] = self.compute_cumulated_value_path(
                variable, attribute
            )
        for index in range(len(self.path_constraints.list_constraint)):
            constr = self.path_constraints.list_constraint[index]
            if not operator_map[constr[1]](vals_attribute[constr[0]], constr[2]):
                delta = delta_map[constr[1]](vals_attribute[constr[0]], constr[2])
                absolute_violation += delta
                info += [(index, delta)]
        return absolute_violation, info

    def satisfy(self, variable: PathSolution) -> bool:
        if variable.path[0] != self.index_source:
            return False
        if variable.path[-1] != self.index_sink:
            return False
        cnt, _ = self.evaluate_node_constraints(variable)
        if cnt > 0:
            return False
        return True

    def get_attribute_register(self) -> EncodingRegister:
        return EncodingRegister(
            dict_attribute_to_type={"path": {"name": "path", "type": [None]}}
        )

    def get_solution_type(self) -> Type[Solution]:
        return PathSolution

    def get_objective_register(self) -> ObjectiveRegister:
        return ObjectiveRegister(
            objective_sense=ModeOptim.MINIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc={
                "length": ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=1),
                "constraint_node": ObjectiveDoc(
                    type=TypeObjective.PENALTY, default_weight=100
                ),
                "constraint_path": ObjectiveDoc(
                    type=TypeObjective.PENALTY, default_weight=100
                ),
            },
        )
