from typing import Type, Dict, Union, List, Tuple, Optional
import numpy as np
from discrete_optimization.generic_tools.do_problem import Problem, ParamsObjectiveFunction, \
    EncodingRegister, ObjectiveDoc, TypeObjective, ObjectiveRegister, Solution, TypeAttribute, ModeOptim, \
    ObjectiveHandling
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


class AircraftLoadingSolution(Solution):
    def __init__(self, problem: "AircraftLoadingProblem",
                 allocation: Union[List[int], np.ndarray]):
        self.problem = problem
        self.allocation = allocation

    def copy(self) -> "Solution":
        return AircraftLoadingSolution(problem=self.problem,
                                       allocation=deepcopy(self.allocation))

    def lazy_copy(self) -> "AircraftLoadingSolution":
        return AircraftLoadingSolution(problem=self.problem, allocation=self.allocation)

    def change_problem(self, new_problem: "Problem") -> None:
        self.problem = new_problem

    def __str__(self):
        return str(self.allocation)


class AircraftLoadingProblem(Problem):
    def __init__(self, container_types: List[int],
                 weights: List[float],
                 slot_distances: List[int],
                 max_loading_weight: float,
                 max_shear: float,
                 length: float,
                 edge_list: Optional[List[Tuple[int, int]]] = None,
                 cog_bounds: Optional[Tuple[float, float]] = None,
                 length_container: Optional[List[float]] = None,
                 heigth_container: Optional[List[float]] = None):
        self.container_types = container_types
        self.number_of_containers = len(self.container_types)
        self.length_container = length_container
        self.heigth_container = heigth_container
        if self.length_container is None:
            self.length_container = [1 for i in range(self.number_of_containers)]
            for i in range(len(self.container_types)):
                if self.container_types[i] in {1, 2}:
                    self.length_container[i] = 1
                if self.container_types[i] == 3:
                    self.length_container[i] = 2
        if self.heigth_container is None:
            self.heigth_container = [1 for i in range(self.number_of_containers)]
            for i in range(len(self.container_types)):
                if self.container_types[i] in {1, 3}:
                    self.heigth_container[i] = 1
                if self.container_types[i] == 2:
                    self.heigth_container[i] = 0.5
        self.weights = weights
        self.slot_distances = slot_distances
        self.max_loading_weight = max_loading_weight
        self.max_shear = max_shear
        self.length = length
        self.edge_list = edge_list
        self.cog_bounds = cog_bounds
        self.number_of_slots = len(slot_distances)
        self.index_slot_to_distance = {i: self.slot_distances[i] for i in range(self.number_of_slots)}
        self.dummy_alloc = -1
        self.shear_profile = lambda p: self.max_shear * (1 - 2 * abs(p) / self.length)

    def evaluate(self, variable: Solution) -> Dict[str, float]:
        sat, margin, total_weight = self.sat_weight(variable)
        kpis = {}
        kpis["value"] = total_weight
        kpis["penalty_violation"] = max(0, margin)
        sat, margin = self.sat_packing(variable)
        kpis["penalty_violation"] += max(0, margin)
        sat, margin = self.sat_shear(variable)
        kpis["penalty_violation"] += max(0, margin)
        sat, margin = self.sat_cog(variable, total_weight)
        kpis["penalty_violation"] += max(0, margin)
        return kpis

    def evaluate_from_encoding(self, int_vector, encoding):
        return self.evaluate(AircraftLoadingSolution(self, int_vector))

    def sat_weight(self, variable: AircraftLoadingSolution) -> Tuple[bool, float, float]:
        total_weight = sum([(variable.allocation[i] != self.dummy_alloc) * self.weights[i]
                            for i in range(self.number_of_containers)])
        if total_weight > self.max_loading_weight:
            # logger.info("Solution not satisfied, because of max weight constraint")
            return False, total_weight-self.max_loading_weight, total_weight
        return True, total_weight-self.max_loading_weight, total_weight

    def sat_packing(self, variable: AircraftLoadingSolution):
        min_distance = min(self.slot_distances)
        max_distance = max(self.slot_distances)
        length_array = max_distance - min_distance
        width_array = np.zeros(length_array)
        height_array = np.zeros(length_array)
        for i in range(self.number_of_containers):
            if variable.allocation[i] != -1:
                ind = variable.allocation[i]
                width = self.length_container[i]
                d = self.slot_distances[ind]
                d_i = d - min_distance
                if d_i + width > length_array - 1:
                    width_array[d_i:] += 1
                    height_array[d_i:] += self.heigth_container[i]
                else:
                    width_array[d_i:d_i + width] += 1
                    height_array[d_i:d_i + width] += self.heigth_container[i]
        if max(height_array) > 1:
            # logger.info("Height constrained ")
            return False, max(height_array)-1

        return True, max(height_array)-1

    def sat_shear(self, variable: AircraftLoadingSolution):
        for j in range(self.number_of_slots):
            weight_in_this = sum([self.weights[i]*(variable.allocation[i] == j)
                                  for i in range(self.number_of_containers)])
            dist = self.slot_distances[j]
            indexes = set()
            if dist < 0:
                indexes = set(range(j))
            if dist > 0:
                indexes = set(range(j+1, self.number_of_slots))
            other_weights = 0
            if len(indexes) > 0:
                other_weights = sum([self.weights[i]*(variable.allocation[i] in indexes)
                                     for i in range(self.number_of_containers)])
            shear = 1/2*weight_in_this+other_weights
            if shear > self.shear_profile(self.slot_distances[j]):
                return False, shear-self.shear_profile(self.slot_distances[j])
        return True, 0

    def sat_cog(self, variable, total_weight):
        centor_gravity = sum([self.slot_distances[variable.allocation[i]] * self.weights[i]
                              for i in range(self.number_of_containers)
                              if variable.allocation[i] != self.dummy_alloc]) / max(0.001,total_weight)
        if not (self.cog_bounds[0] <= centor_gravity <= self.cog_bounds[1]):
            return False, max(self.cog_bounds[0]-centor_gravity, centor_gravity-self.cog_bounds[1], 0)
        return True, 0

    def satisfy(self, variable: Solution) -> bool:
        variable: AircraftLoadingSolution
        sat, margin, total_weight = self.sat_weight(variable)
        if not sat:
            logger.info("Solution not satisfied, because of max weight constraint")
            return False
        sat, margin = self.sat_cog(variable, total_weight)
        if not sat:
            logger.info("Solution not satisfied, because of cog")
            return False
        sat, margin = self.sat_packing(variable)
        if not sat:
            logger.info("Solution not satisfied, because of packing")
            return sat
        sat, _ = self.sat_shear(variable)
        if not sat:
            logger.info("Solution not satisfied, because of shear")
        return sat

    def get_attribute_register(self) -> EncodingRegister:
        dict_register = {"allocation": {"name": "allocation",
                                        "type": [TypeAttribute.LIST_INTEGER_SPECIFIC_ARITY],
                                        "n": self.number_of_containers,
                                        "low": -1,
                                        "up": self.number_of_slots-1}}
        return EncodingRegister(dict_register)

    def get_solution_type(self) -> Type[Solution]:
        return AircraftLoadingSolution

    def get_objective_register(self) -> ObjectiveRegister:
        dict_objective = {
            "value": ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=-1),
            "penalty_violation": ObjectiveDoc(type=TypeObjective.PENALTY,
                                              default_weight=100)
        }
        return ObjectiveRegister(
            objective_sense=ModeOptim.MINIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc=dict_objective,
        )

    def get_dummy_solution(self) -> AircraftLoadingSolution:
        return AircraftLoadingSolution(problem=self, allocation=[-1 for i in range(self.number_of_containers)])