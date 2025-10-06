import sys
from typing import Any, Optional
from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from multibatching.problem import MultibatchingProblem, MultibatchingSolution
from multibatching.problem import TransportType, PackingTransport, Product, TransportLink, Location
from discrete_optimization.generic_tools.do_solver import SolverDO, ResultStorage
import heapq
import collections
from collections import defaultdict
import math
import networkx as nx  # can be useful for flow optimisation.

info = """
Some API for networkx :-) : 
min_cost_flow(G, demand="demand", capacity="capacity", weight="weight")
"""
var = ("\n"
       "Here is the code of the MultibatchingProblem and MultibatchingSolution. Only use the available stuff !!! \n"
       "@dataclass(frozen=True)\n"
       "class TransportType:\n"
       "    id: int = field(hash=True, repr=False)\n"
       "    cost: int = field(hash=False)\n"
       "    speed: int = field(hash=False)\n"
       "    emissions: int = field(hash=False)\n"
       "    capacity: int = field(hash=False, repr=True)\n"
       "    name: str = field(hash=False, repr=True, default=None)\n"
       "\n"
       "@dataclass\n"
       "class Product:\n"
       "    id: int = field(repr=False)\n"
       "    name: str = field(repr=True, hash=False, default=\"\")\n"
       "    size: int = field(repr=True, default=0)\n"
       "    value: int = field(repr=False, default=0)\n"
       "    valid_transports: frozenset[TransportType] = field(default_factory=set, hash=False, repr=False)\n"
       "\n"
       "    def __hash__(self):\n"
       "        return hash(self.id)\n"
       "\n"
       "\n"
       "@dataclass(frozen=True)\n"
       "class Location:\n"
       "    id: Hashable = field(repr=False, hash=True)\n"
       "    name: str = field(repr=True, hash=False, default=\"\")\n"
       "    net_supply: dict[Product, int] = field(repr=False, hash=False, default_factory=lambda: {})\n"
       "\n"
       "\n"
       "@dataclass(frozen=True)\n"
       "class TransportLink:\n"
       "    location_l1: Location\n"
       "    location_l2: Location\n"
       "    distance: int\n"
       "    transport_type: TransportType\n"
       "    max_trips: Optional[int] = field(repr=True, hash=False, default=None)\n"
       "\n"
       "\n"
       "@dataclass\n"
       "class PackingTransport:\n"
       "    transport_link: TransportLink\n"
       "    product_packing: dict[Product, int]\n"
       "    nb_packing: int\n"
       "\n"
       "\n"
       "class MultibatchingSolution(Solution):\n"
       "    def __init__(self, problem: \"MultibatchingProblem\",\n"
       "                 list_flows: list[PackingTransport]):\n"
       "        self.problem = problem\n"
       "        self.list_flows = list_flows\n"
       "\n"
       "    def change_problem(self, new_problem: \"Problem\") -> None:\n"
       "        self.problem = new_problem\n"
       "\n"
       "    def copy(self) -> \"Solution\":\n"
       "        return MultibatchingSolution(problem=self.problem,\n"
       "                                     list_flows=list(self.list_flows))\n"
       "\n"
       "\n"
       "class MultibatchingProblem(Problem):\n"
       "    def __init__(self,\n"
       "                 transport_types: list[TransportType],\n"
       "                 products: list[Product],\n"
       "                 locations: list[Location],\n"
       "                 transport_links: list[TransportLink],\n"
       "                 capital_factor: float = 0.1):\n"
       "        self.transport_types = transport_types\n"
       "        self.products = products\n"
       "        self.locations = locations\n"
       "        self.transport_links = transport_links\n"
       "        self.capital_factor = capital_factor\n"
       "        self.nb_products = len(self.products)\n"
       "        self.nb_locations = len(self.locations)\n"
       "        self.nb_transport_types = len(self.transport_types)\n"
       "        self.nb_transport_links = len(self.transport_links)\n"
       "\n"
       "        self.product_to_index = {self.products[i]: i for i in range(self.nb_products)}\n"
       "        self.transport_types_to_index = {self.transport_types[i]: i for i in range(self.nb_transport_types)}\n"
       "        self.locations_to_index = {self.locations[i]: i for i in range(self.nb_locations)}\n"
       "        self.transport_links_to_index = {self.transport_links[i]: i for i in range(self.nb_transport_links)}\n"
       "\n"
       "        self.transport_type_id_to_transport_type = {tt.id: tt for tt in self.transport_types}\n"
       "        self.location_id_to_location = {loc.id: loc for loc in self.locations}\n"
       "\n"
       "        self.loc_and_transport_type_to_transport_link = {(tl.location_l1.id, tl.location_l2.id,\n"
       "                                                          tl.transport_type.id): tl\n"
       "                                                         for tl in self.transport_links}\n"
       "\n"
       "    def get_supplier_location(self, product: Product):\n"
       "        'Returns the location where a given product is done (i.e net-supply>0)'\n"
       "        return [(l, l.net_supply.get(product, 0)) for l in self.locations if l.net_supply.get(product, 0) > 0]\n"
       "\n"
       "    def get_consumer_location(self, product: Product):\n"
       "        'Returns the location where a given product is consumed (i.e net-supply<0)'\n"
       "        return [(l, l.net_supply.get(product, 0)) for l in self.locations if l.net_supply.get(product, 0) < 0]\n"
       "\n"
       "    def get_outgoing_tl_from_location(self, location: Location):\n"
       "        'Returns all transport link that goes out of a given location'\n"
       "        return [tl for tl in self.transport_links if tl.location_l1 == location]\n"
       "\n"
       "    def get_total_supply(self, product: Product):\n"
       "        return sum([max(loc.net_supply.get(product, 0), 0) for loc in self.locations])\n"
       "\n"
       "    def get_total_demand(self, product: Product):\n"
       "        return -sum([min(loc.net_supply.get(product, 0), 0) for loc in self.locations])\n"
       "\n")


class OpenEvolveMultibatchingSolver(SolverDO):
    problem: MultibatchingProblem

    def __init__(self, problem: MultibatchingProblem,
                 params_objective_function: Optional[ParamsObjectiveFunction] = None,
                 **kwargs: Any):
        super().__init__(problem, params_objective_function=params_objective_function,
                         **kwargs)

    # EVOLVE-BLOCK-START
    def solve(self, callbacks: Optional[list[Callback]] = None,
              time_limit: int = 5, **kwargs: Any) -> ResultStorage:
        """You should limit yourself to 5 seconds, or try to take the time_limit into consideration in your solver.
        """
        problem = self.problem
        all_flows_per_link_product = defaultdict(lambda: defaultdict(int))  # tl -> product -> total_quantity

        # Step 1: Solve min-cost flow for each product independently
        for p in problem.products:
            G_p = nx.DiGraph()
            source_p_node = f"source_{p.id}"
            sink_p_node = f"sink_{p.id}"

            # Nodes for locations and source/sink
            G_p.add_node(source_p_node)
            G_p.add_node(sink_p_node)
            for loc in problem.locations:
                G_p.add_node(loc.id)

            # Add demand/supply to nodes
            # networkx min_cost_flow uses node attributes for demand.
            # positive demand for sink, negative for source.
            node_demands = {}
            total_supply_p = 0
            total_demand_p = 0

            for loc in problem.locations:
                net_qty = loc.net_supply.get(p, 0)
                if net_qty > 0:  # Supplier
                    G_p.add_edge(source_p_node, loc.id, capacity=net_qty, weight=0)
                    total_supply_p += net_qty
                elif net_qty < 0:  # Consumer
                    G_p.add_edge(loc.id, sink_p_node, capacity=abs(net_qty), weight=0)
                    total_demand_p += abs(net_qty)

            # If no demand or supply for this product, skip
            if total_demand_p == 0 or total_supply_p == 0:
                continue

            # Set demand for source and sink (min_cost_flow needs balanced demand)
            node_demands[source_p_node] = -total_supply_p
            node_demands[sink_p_node] = total_demand_p
            # For intermediate nodes, demand is 0 by default, but explicitly set for clarity
            for loc in problem.locations:
                if loc.id not in node_demands:  # Only if not already source/sink
                    node_demands[loc.id] = 0

            nx.set_node_attributes(G_p, node_demands, "demand")

            # Add transport links as edges
            for tl in problem.transport_links:
                # Check if product can be transported by this type
                if tl.transport_type not in p.valid_transports:
                    continue

                # Avoid division by zero or invalid product sizes
                if tl.transport_type.capacity == 0:
                    continue  # Cannot transport anything if the transport type has 0 capacity

                max_units_p_per_trip = 0
                cost_per_unit_p_adjusted = 0.0

                if p.size == 0:
                    # Products with size 0 don't consume volumetric capacity.
                    # Assume a very large number can fit in a trip (sys.maxsize represents effectively infinite).
                    max_units_p_per_trip = sys.maxsize
                    # Assign a very small but non-zero cost per unit.
                    # This ensures they are not "free" but are still routed if cheaper than unmet demand penalty.
                    # Use a large divisor to make the per-unit cost minimal, only if there's a base cost.
                    cost_per_unit_p_adjusted = (tl.distance * tl.transport_type.cost) / (
                                sys.maxsize / 2) if tl.transport_type.cost > 0 else 0.0
                else:  # p.size > 0
                    max_units_p_per_trip = tl.transport_type.capacity // p.size
                    if max_units_p_per_trip == 0:  # Product too large to fit in a single trip
                        continue
                    cost_per_unit_p_adjusted = (tl.distance * tl.transport_type.cost) / max_units_p_per_trip

                # Multiply by a factor to keep weights as integers for networkx, preserving precision.
                # Ensure non-negative cost. Use max(1, ...) for positive costs to avoid values becoming 0 after int conversion,
                # especially for very small 'cost_per_unit_p_adjusted'.
                int_cost = max(1, int(cost_per_unit_p_adjusted * 1000)) if cost_per_unit_p_adjusted > 0 else 0

                # Total capacity for product p on this link in the min-cost flow graph.
                # We set this to the total demand for product p, allowing the solver to attempt to route
                # all of it if economically viable. The actual shared link capacity (max_trips and volume)
                # will be handled in the subsequent capping and packing step.
                total_demand_or_supply_p = max(problem.get_total_demand(p), problem.get_total_supply(p))
                total_capacity_for_p_on_link = total_demand_or_supply_p
                # Cap by sys.maxsize to avoid overflow with extremely large demands or if demand is 0.
                if total_capacity_for_p_on_link > 0:
                    total_capacity_for_p_on_link = min(total_capacity_for_p_on_link, sys.maxsize)

                # Add edge to the graph for product p
                G_p.add_edge(tl.location_l1.id, tl.location_l2.id,
                             capacity=total_capacity_for_p_on_link,
                             weight=int_cost,
                             _transport_link=tl)  # Store original tl for later

            # Solve min-cost flow for product p
            try:
                flow_solution_p = nx.min_cost_flow(G_p, demand="demand", capacity="capacity", weight="weight")

                # Accumulate flows for all products on actual transport links
                for u, v_flows in flow_solution_p.items():
                    for v, flow_amount in v_flows.items():
                        if flow_amount > 0:
                            edge_data = G_p.get_edge_data(u, v)
                            # Only accumulate flow if it's on an actual transport link.
                            # Skip flows on source/sink edges, balancing edges, or unmet demand edges.
                            if '_transport_link' in edge_data:
                                original_tl = edge_data['_transport_link']
                                all_flows_per_link_product[original_tl][p] += flow_amount
            except nx.NetworkXUnfeasible:
                # This should theoretically not happen with the balancing node, as a feasible solution is guaranteed.
                # However, it's good practice to catch it for unexpected graph issues.
                pass
            except Exception as e:
                # Catch any other exceptions during flow calculation.
                # print(f"Error solving min-cost flow for product {p.id}: {e}") # For debugging purposes
                pass

        # Step 2: Capping flows based on actual link capacity (max_trips)
        # This addresses the shared capacity issue that min-cost flow per product couldn't model.
        for tl in problem.transport_links:
            products_flow_on_this_link = all_flows_per_link_product[tl]

            # Calculate total volume requested on this link, ignoring zero-sized products for volume calculation
            total_volume_requested = sum(qty * p.size for p, qty in products_flow_on_this_link.items() if p.size > 0)

            max_trips_for_tl = tl.max_trips if tl.max_trips is not None else sys.maxsize
            max_volume_capacity = tl.transport_type.capacity * max_trips_for_tl

            # If there's an over-request for volume on this link, scale down proportionally
            if total_volume_requested > max_volume_capacity and max_volume_capacity > 0:
                scaling_factor = max_volume_capacity / total_volume_requested
                for p in products_flow_on_this_link:
                    # Only scale down products that consume volume
                    if p.size > 0:
                        products_flow_on_this_link[p] = math.floor(products_flow_on_this_link[p] * scaling_factor)
            elif max_volume_capacity == 0 and total_volume_requested > 0:
                # If capacity is 0 but flow is requested for volume-consuming products, set those flows to 0
                for p in products_flow_on_this_link:
                    if p.size > 0:
                        products_flow_on_this_link[p] = 0

        # Step 3: Consolidate product flows into PackingTransport objects
        final_packing_transports = []
        link_trips_taken = defaultdict(int)  # Track trips per TransportLink

        # Process links, trying to pack products into trips
        for tl in problem.transport_links:
            products_flow_on_this_link = all_flows_per_link_product[tl]

            # Filter out products with 0 flow after capping
            current_products_to_pack = {p: qty for p, qty in products_flow_on_this_link.items() if qty > 0}
            if not current_products_to_pack:
                continue

            # Sort products by size (largest first for denser packing)
            sorted_products = sorted(current_products_to_pack.keys(), key=lambda prod: prod.size, reverse=True)

            max_trips_for_tl = tl.max_trips if tl.max_trips is not None else sys.maxsize

            while any(qty > 0 for qty in current_products_to_pack.values()) and link_trips_taken[tl] < max_trips_for_tl:
                current_packing_config = {}
                current_trip_volume = 0

                # Attempt to fill a trip
                for p in sorted_products:
                    if current_products_to_pack[p] > 0:
                        space_left_in_current_trip = tl.transport_type.capacity - current_trip_volume

                        if p.size == 0:
                            units_of_p_that_fit = current_products_to_pack[p]  # Takes no volume, so all remaining fit
                        else:
                            units_of_p_that_fit = space_left_in_current_trip // p.size

                        qty_to_add = min(current_products_to_pack[p], units_of_p_that_fit)

                        if qty_to_add > 0:
                            current_packing_config[p] = current_packing_config.get(p, 0) + qty_to_add
                            if p.size > 0:  # Only add to volume if product has size
                                current_trip_volume += qty_to_add * p.size
                            current_products_to_pack[p] -= qty_to_add

                if current_packing_config:  # If we managed to pack something into this trip
                    final_packing_transports.append(PackingTransport(tl, dict(current_packing_config), 1))
                    link_trips_taken[tl] += 1
                else:  # No more products fit in current trip or no products left to pack for this link
                    break  # Exit while loop for this tl

        solution = MultibatchingSolution(problem=problem, list_flows=final_packing_transports)
        fit = self.aggreg_from_sol(solution)
        result_storage = self.create_result_storage([(solution, fit)])
        return result_storage
    # EVOLVE-BLOCK-END
