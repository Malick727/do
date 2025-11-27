from collections import defaultdict
from copy import deepcopy

import numpy as np

from discrete_optimization.generic_tools.do_problem import (Problem, Solution, EncodingRegister, ModeOptim,
                                                            ObjectiveDoc, ObjectiveHandling, ObjectiveRegister,
                                                            TypeObjective, TypeAttribute)
from discrete_optimization.vrptw.problem import VRPTWProblem, VRPTWSolution
from dataclasses import dataclass, field
from typing import Hashable, Union


@dataclass(frozen=True)
class ProductType:
    id: Hashable
    name: str
    is_renewable: bool

@dataclass
class ProductInstance:
    id: Hashable
    name: str
    product_type: ProductType

    def __hash__(self):
        return hash((self.id, self.name))


@dataclass
class RequirementRelation:
    """If product_type_1 is delivered some vehicle should also deliver some product of type 2 to the
        client"""
    product_type_1: ProductType
    product_type_2: ProductType

@dataclass
class RequiredProductRelation:
    """If product_type_1 is delivered some vehicle should also deliver some product of type 2 to the
        client"""
    product_type_1: ProductType
    product_types_2: set[ProductType]


@dataclass
class Node:
    id: Hashable
    name: str
    is_client: bool
    is_depot: bool
    time_window: tuple[int, int]
    service_time: int
    group_id: Hashable
    mandatory: bool


    def __post_init__(self):
        if self.is_client:
            assert not self.is_depot
        if self.is_depot:
            assert not self.is_client

    def __hash__(self):
        return hash((self.id, self.name))

TeamId = Hashable

@dataclass
class Vehicle:
    id: Hashable
    name: str
    load: list[ProductInstance]
    min_speed: float
    max_speed: float
    start_node: Node
    end_node: Node
    team_id: TeamId = field(default=0)
    load_per_type: list[tuple[ProductType, int]] = field(init=False)
    instance_dict_per_type: dict[ProductType, list[ProductInstance]] = field(init=False)

    def __post_init__(self):
        count_per_product_type = {}
        instance_dict_per_type = {}
        for product_instance in self.load:
            type_ = product_instance.product_type
            if type_ not in count_per_product_type:
                count_per_product_type[type_] = 0
                instance_dict_per_type[type_] = []
            count_per_product_type[type_] += 1
            instance_dict_per_type[type_].append(product_instance)
        self.load_per_type = [(type_, count_per_product_type[type_])
                              for type_ in count_per_product_type]
        self.instance_dict_per_type = instance_dict_per_type

    def __hash__(self):
        return hash((self.id, self.name))

class VrpTwAltSolution(Solution):
    problem: "VrpTwAlternativeDeliveryProblem"

    def __init__(self, problem: "VrpTwAlternativeDeliveryProblem",
                 # for each vehicle List of node_index, time, set_of_product_delivered
                 trajectories: list[list[tuple[int,
                                               int,
                                               set[ProductInstance]]]]):
        self.problem = problem
        self.trajectories = trajectories

    def copy(self) -> "Solution":
        return deepcopy(self)

    def change_problem(self, new_problem: "Problem") -> None:
        self.problem = new_problem


class VrpTwAlternativeDeliveryProblem(Problem):
    def __init__(self, nodes: list[Node],
                 vehicles: list[Vehicle],
                 # Graph ??
                 distance_matrix: np.ndarray,
                 travel_times: np.ndarray,
                 travel_costs: np.ndarray,
                 # To store different kpi on edges.
                 edge_cost_dict: dict[tuple[Vehicle, Node, Node],
                                      dict[str, float]] = None,
                 # For each vehicle, product type and node,
                 # different kpi on delivery of this product to the Node.
                 product_delivery_reward: dict[tuple[Vehicle, ProductType, Node],
                                               dict[str, float]] = None,
                 requirement_relations: list[Union[RequirementRelation, RequiredProductRelation]] = None,
                 other_data: dict = None):
        self.nodes = nodes
        self.vehicles = vehicles
        self.distance_matrix = distance_matrix
        self.travel_times = travel_times
        self.travel_costs = travel_costs

        self.edge_cost_dict = edge_cost_dict
        if self.edge_cost_dict is None:
            self.edge_cost_dict = {}
        self.product_delivery_reward = product_delivery_reward
        if self.product_delivery_reward is None:
            self.product_delivery_reward = {}

        self.requirement_relations = requirement_relations
        if self.requirement_relations is None:
            self.requirement_relations = []

        self.number_vehicles = len(self.vehicles)
        self.number_nodes = len(self.nodes)
        self.nb_requirement_relations = len(self.requirement_relations)
        # Pre-compute maps for faster validation
        self.id_to_index_map = {n.id: i for i, n in enumerate(self.nodes)}
        self.product_to_vehicle_map = {
            p: k
            for k, v in enumerate(self.vehicles)
            for p in v.load
        }
        self.client_indices = {i for i, n in enumerate(self.nodes) if n.is_client}

        self.other_data = other_data
        self.index_of_nodes_per_group = self.build_group_data()

    def build_group_data(self):
        index_of_nodes_per_group = defaultdict(lambda: [])
        for i in range(self.number_nodes):
            index_of_nodes_per_group[self.nodes[i].group_id].append(i)
        return index_of_nodes_per_group

    def build_team_data(self):
        index_of_vehicle_per_team = defaultdict(lambda: [])
        for i in range(self.number_vehicles):
            index_of_vehicle_per_team[self.vehicles[i].team_id].append(i)
        return index_of_vehicle_per_team


    def evaluate(self, variable: VrpTwAltSolution) -> dict[str, float]:
        return {"dummy": 0}

    def satisfy(self, variable: VrpTwAltSolution) -> bool:
        if len(variable.trajectories) != self.number_vehicles:
            print("Mismatch")
            return False  # Solution structure mismatch

        all_delivered_non_renewable: set[ProductInstance] = set()
        deliveries_at_node_by_type: dict[int, set[ProductType]] = {}
        visited_customer_indices: set[int] = set()

        # 1. Check each vehicle's route validity and collect delivery info
        for k in range(self.number_vehicles):
            vehicle = self.vehicles[k]
            vehicle_trajectory = variable.trajectories[k]

            if not vehicle_trajectory:
                continue  # Vehicle not used

            vehicle_load_by_type = dict(vehicle.load_per_type)
            delivered_count_by_type: dict[ProductType, int] = {}

            start_node_idx = self.id_to_index_map[vehicle.start_node.id]
            current_node_idx = start_node_idx
            # Departure time from start node
            last_departure_time = self.nodes[start_node_idx].time_window[0]

            for visit in vehicle_trajectory:
                node_index, arrival_time, delivered_products = visit

                if node_index not in self.client_indices:
                    print("Trajectories should only contain client visits")
                    return False  # Trajectories should only contain client visits

                node = self.nodes[node_index]

                # Check travel time
                travel_time = self.travel_times[current_node_idx, node_index]
                if arrival_time < last_departure_time + travel_time:
                    print("Arrived before travel was possible, ", node_index,
                          last_departure_time,
                          travel_time)
                    return False  # Arrived before travel was possible

                # Check node time window (assuming single TW)
                if not (node.time_window[0] <= arrival_time <= node.time_window[1]):
                    print("Arrive outside time-window")
                    return False  # Arrived outside time window

                # Update departure time from this node
                service_start_time = max(arrival_time, node.time_window[0])
                last_departure_time = service_start_time + node.service_time
                current_node_idx = node_index

                # --- Process Deliveries ---
                if node_index not in deliveries_at_node_by_type:
                    deliveries_at_node_by_type[node_index] = set()
                visited_customer_indices.add(node_index)

                if not delivered_products:
                    print("nothing delivered")
                    return False  # A visit must deliver at least one product

                for p_instance in delivered_products:
                    # Check if product belongs to this vehicle
                    if p_instance not in self.product_to_vehicle_map or self.product_to_vehicle_map[p_instance] != k:
                        print("wrong vehicle")
                        return False  # Product delivered by wrong vehicle

                    deliveries_at_node_by_type[node_index].add(p_instance.product_type)

                    # Check non-renewable and capacity
                    if not p_instance.product_type.is_renewable:
                        if p_instance in all_delivered_non_renewable:
                            print("already delivered")
                            return False  # Non-renewable product delivered more than once
                        all_delivered_non_renewable.add(p_instance)

                        p_type = p_instance.product_type
                        delivered_count_by_type[p_type] = delivered_count_by_type.get(p_type, 0) + 1

            # Check capacity totals for this vehicle
            for p_type, count in delivered_count_by_type.items():
                if p_type not in vehicle_load_by_type or count > vehicle_load_by_type[p_type]:
                    print("Capacity exceeded")
                    return False  # Exceeded capacity for a non-renewable product

            # Check travel to end node
            end_node_idx = self.id_to_index_map[vehicle.end_node.id]
            end_node = self.nodes[end_node_idx]
            travel_time = self.travel_times[current_node_idx, end_node_idx]
            end_arrival_time = last_departure_time + travel_time

            if end_arrival_time > end_node.time_window[1]:
                print("Arrive too late")
                return False  # Arrived at end depot too late

        # 2. Check customer coverage (after checking all vehicles)
        if visited_customer_indices != self.client_indices:
            print("Missing customer")
            return False  # Not all customers were visited

        # 3. Check requirement relations for each customer (after checking all vehicles)
        for node_index in self.client_indices:
            delivered_types = deliveries_at_node_by_type.get(node_index, set())
            for req in self.requirement_relations:
                if isinstance(req, RequirementRelation):
                    if req.product_type_1 in delivered_types and req.product_type_2 not in delivered_types:
                        print("Requirement not met")
                        return False  # Requirement not met
                if isinstance(req, RequiredProductRelation):
                    if req.product_type_1 in delivered_types and not(any(r in delivered_types for r in req.product_types_2)):
                        print("Requirement not met")
                        return False  # Requirement not met
            if not any(not t.is_renewable for t in delivered_types):
                print(delivered_types)
                print(f"Missing delivery for node index {node_index}")
        return True

    def get_attribute_register(self) -> EncodingRegister:
        return EncodingRegister(
            {
                "routes": {
                    "name": "trajectories",
                    "type": [],
                }
            }
        )

    def get_solution_type(self) -> type[Solution]:
        return VrpTwAltSolution

    def get_objective_register(self) -> ObjectiveRegister:
        return ObjectiveRegister(objective_sense=ModeOptim.MINIMIZATION,
                                 objective_handling=ObjectiveHandling.AGGREGATE,
                                 dict_objective_to_doc={"dummy": ObjectiveDoc(type=TypeObjective.OBJECTIVE,
                                                                              default_weight=1)})