from collections import defaultdict
from typing import Hashable

from discrete_optimization.generic_tools.do_problem import *
from discrete_optimization.generic_tools.graph_api import Graph, from_networkx
import networkx as nx
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TransportType:
    id: int = field(hash=True)
    cost: int = field(hash=False)
    speed: int = field(hash=False)
    emissions: int = field(hash=False)
    capacity: int = field(hash=False)


@dataclass(frozen=True)
class Product:
    id: int
    size: int
    value: int
    valid_transports: frozenset[TransportType] = field(default_factory=set, hash=False)


@dataclass(frozen=True)
class Location:
    id: Hashable = field(repr=True, hash=True)
    net_supply: dict[Product, int] = field(repr=False, hash=False)


@dataclass(frozen=True)
class TransportLink:
    location_l1: Location
    location_l2: Location
    distance: int
    transport_type: TransportType


@dataclass
class PackingTransport:
    transport_link: TransportLink
    product_packing: dict[Product, int]
    nb_packing: int


class MultibatchingSolution(Solution):
    def __init__(self, problem: "MultibatchingProblem",
                 list_flows: list[PackingTransport]):
        self.problem = problem
        self.list_flows = list_flows

    def change_problem(self, new_problem: "Problem") -> None:
        self.problem = new_problem

    def copy(self) -> "Solution":
        return MultibatchingSolution(problem=self.problem,
                                     list_flows=list(self.list_flows))


class MultibatchingProblem(Problem):
    def __init__(self,
                 transport_types: list[TransportType],
                 products: list[Product],
                 locations: list[Location],
                 transport_links: list[TransportLink],
                 capital_factor: float = 0.1):
        self.transport_types = transport_types
        self.products = products
        self.locations = locations
        self.transport_links = transport_links
        self.capital_factor = capital_factor
        self.nb_products = len(self.products)
        self.nb_locations = len(self.locations)
        self.nb_transport_types = len(self.transport_types)
        self.nb_transport_links = len(self.transport_links)

        self.product_to_index = {self.products[i]: i for i in range(self.nb_products)}
        self.transport_types_to_index = {self.transport_types[i]: i for i in range(self.nb_transport_types)}
        self.locations_to_index = {self.locations[i]: i for i in range(self.nb_locations)}
        self.transport_links_to_index = {self.transport_links[i]: i for i in range(self.nb_transport_links)}

    def evaluate(self, variable: "MultibatchingSolution") -> dict[str, float]:
        transport_cost = sum([flow.transport_link.transport_type.cost
                              * flow.transport_link.distance*flow.nb_packing
                              for flow in variable.list_flows])
        emission_co2 = sum([flow.transport_link.transport_type.emissions
                            * flow.transport_link.distance
                            * flow.nb_packing
                            for flow in variable.list_flows])
        compound = 0
        for flow in variable.list_flows:
            value = sum([p.value*flow.product_packing[p]
                         for p in flow.product_packing])
            time = flow.transport_link.distance/flow.transport_link.transport_type.speed
            coef = value*((1+self.capital_factor)**time-1)
            compound += flow.nb_packing*coef
        return {'transport': transport_cost,
                'emission': emission_co2,
                'capital': compound}

    def satisfy(self, variable: MultibatchingSolution) -> bool:
        """
        Checks if a given MultibatchingSolution is feasible.

        Feasibility conditions checked:
        1.  **Flow Conservation**: For each location and product, the total incoming flow
            plus local supply must equal total outgoing flow plus local demand.
        2.  **Packing Capacity**: For each PackingTransport, the total size of products
            packed must not exceed the capacity of its associated TransportType.
        3.  **Product-Transport Compatibility**: Each product in a PackingTransport must
            be valid for the TransportType of its TransportLink.
        4.  **Non-negative Frequencies**: nb_packing must be non-negative.
        5.  **Non-negative Packed Units**: Units in product_packing must be non-negative.
        6.  **Integer Packing Units**: Units in product_packing must be integers (as per problem definition).
        7.  **Positive Frequency for Non-empty Packing**: If nb_packing is > 0, then product_packing cannot be empty.
            (This is usually implicitly handled by solver, but explicit check for robustness).
        """
        epsilon = 1e-6  # Tolerance for floating-point comparisons

        # --- 1. Initialize net flows for each location and product ---
        # { (location_id, product_id): net_flow }
        # Start with initial net_supply/demand from the problem definition
        current_net_flows = defaultdict(float)
        for loc in self.locations:
            for product in self.products:
                current_net_flows[(loc.id, product.id)] = loc.net_supply.get(product, 0)

        # --- 2. Process all PackingTransport flows ---
        for flow_item in variable.list_flows:
            link = flow_item.transport_link

            # Check 4: Non-negative Frequencies
            if flow_item.nb_packing < -epsilon:
                # print(f"Validation Error: Negative nb_packing for link {link.location_l1.id}->{link.location_l2.id}: {flow_item.nb_packing}")
                return False

            # Calculate packed volume for this specific packing
            current_packing_volume = 0.0

            # Check 7: If nb_packing > 0, packing must not be empty.
            if flow_item.nb_packing > epsilon and not flow_item.product_packing:
                # print(f"Validation Error: Non-zero nb_packing with empty product_packing for link {link.location_l1.id}->{link.location_l2.id}")
                return False

            for product, units in flow_item.product_packing.items():
                # Check 5: Non-negative Packed Units
                if units < -epsilon:
                    # print(f"Validation Error: Negative packed units for P{product.id} in packing {flow_item.product_packing}: {units}")
                    return False

                # Check 6: Integer Packing Units
                # If units are not integers, the packing itself is invalid
                if abs(units - round(units)) > epsilon:
                    # print(f"Validation Error: Non-integer packed units for P{product.id} in packing {flow_item.product_packing}: {units}")
                    return False

                # Check 3: Product-Transport Compatibility
                if link.transport_type not in product.valid_transports:
                    # print(f"Validation Error: Product P{product.id} not compatible with TR{link.transport_type.id} on link {link.location_l1.id}->{link.location_l2.id}")
                    return False

                current_packing_volume += product.size * units

                # Update net flows for flow conservation check
                # Outgoing flow from link.location_l1
                current_net_flows[(link.location_l1.id, product.id)] -= units * flow_item.nb_packing
                # Incoming flow to link.location_l2
                current_net_flows[(link.location_l2.id, product.id)] += units * flow_item.nb_packing

            # Check 2: Packing Capacity
            if current_packing_volume > link.transport_type.capacity + epsilon:
                # print(f"Validation Error: Packing volume {current_packing_volume} exceeds capacity {link.transport_type.capacity} for link {link.location_l1.id}->{link.location_l2.id}")
                return False

        # --- 3. Final Flow Conservation Check ---
        # After processing all flows, all current_net_flows should be close to zero
        for (loc_id, prod_id), net_flow in current_net_flows.items():
            if abs(net_flow) > epsilon:
                # print(f"Validation Error: Flow imbalance for Location {loc_id}, Product P{prod_id}: {net_flow}")
                return False

        return True  # All checks passed

    def get_attribute_register(self) -> EncodingRegister:
        return EncodingRegister(dict_attribute_to_type={})

    def get_solution_type(self) -> type[Solution]:
        return MultibatchingSolution

    def get_objective_register(self) -> ObjectiveRegister:
        return ObjectiveRegister(objective_sense=ModeOptim.MINIMIZATION,
                                 objective_handling=ObjectiveHandling.AGGREGATE,
                                 dict_objective_to_doc=
                                 {cost: ObjectiveDoc(type=TypeObjective.OBJECTIVE,
                                                     default_weight=1)
                                  for cost in ["transport",
                                               "emission",
                                               "capital"]})
