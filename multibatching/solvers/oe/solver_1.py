import math
import networkx as nx

import sys
from collections import defaultdict
from typing import Any, Optional

import heapq

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from multibatching.problem import PackingTransport, MultibatchingSolution, MultibatchingProblem
from multibatching.solvers import MultibatchingSolver


class OpenEvolveMultibatchingSolver(MultibatchingSolver):
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

                # Cannot transport anything if transport type capacity is 0, regardless of product size
                if tl.transport_type.capacity == 0:
                    continue

                max_units_p_per_trip: int
                cost_per_unit_p_raw: float

                if p.size == 0:
                    # Product takes no physical volume, so effectively infinite units can be carried per trip
                    # Use sys.maxsize to represent effectively infinite capacity for this product on a trip.
                    max_units_p_per_trip = sys.maxsize
                    # For size-0 products, assign a base cost per unit.
                    # This is a heuristic: we assume 1000 units of size-0 product incur the full trip cost.
                    cost_per_unit_p_raw = (tl.distance * (
                                tl.transport_type.cost + tl.transport_type.emissions)) / 1000.0
                else:
                    # Product takes physical volume
                    max_units_p_per_trip = tl.transport_type.capacity // p.size
                    if max_units_p_per_trip == 0:  # Product too large for transport type
                        continue  # Cannot transport this product with this transport type

                    # Cost per unit of product p, considering trip cost and emissions
                    cost_per_unit_p_raw = (tl.distance * (
                                tl.transport_type.cost + tl.transport_type.emissions)) / max_units_p_per_trip

                # Scale cost to integer for networkx, and ensure it's at least 1 to avoid free edges.
                int_cost = max(1, int(cost_per_unit_p_raw * 1000))  # Multiply by 1000 for precision, then cast to int

                # Total capacity for product p on this link.
                # If max_units_p_per_trip is sys.maxsize, ensure total capacity is also sys.maxsize
                # to represent truly infinite total capacity for this product.
                max_available_trips_for_link = tl.max_trips if tl.max_trips is not None else sys.maxsize

                total_capacity_for_p_on_link: int
                if max_units_p_per_trip == sys.maxsize or max_available_trips_for_link == sys.maxsize:
                    total_capacity_for_p_on_link = sys.maxsize  # Effectively infinite total capacity
                else:
                    total_capacity_for_p_on_link = max_units_p_per_trip * max_available_trips_for_link

                # Add edge to the graph for product p
                G_p.add_edge(tl.location_l1.id, tl.location_l2.id,
                             capacity=total_capacity_for_p_on_link,
                             weight=int_cost,
                             _transport_link=tl)  # Store original tl for later

            # Solve min-cost flow for product p
            try:
                flow_solution_p = nx.min_cost_flow(G_p, demand="demand", capacity="capacity", weight="weight")

                # Accumulate flows for all products on each transport link
                for u, v_flows in flow_solution_p.items():
                    for v, flow_amount in v_flows.items():
                        # Only consider flows on actual transport links, not source/sink edges
                        if flow_amount > 0 and u != source_p_node and v != sink_p_node:
                            edge_data = G_p.get_edge_data(u, v)
                            original_tl = edge_data['_transport_link']
                            all_flows_per_link_product[original_tl][p] += flow_amount
            except nx.NetworkXUnfeasible:
                # This product's demand cannot be met, continue to next product
                pass
            except Exception as e:
                # Other potential errors during flow calculation
                # print(f"Error solving min-cost flow for product {p.id}: {e}") # For debugging
                pass

        # Step 2: Consolidate product flows into PackingTransport objects
        final_packing_transports = []
        link_trips_taken = defaultdict(int)  # Track trips per TransportLink

        # Process links, trying to pack products into trips
        # Iterate over all transport links to ensure they are considered for packing,
        # even if no product flows were assigned in the min-cost step (unlikely for active links)
        for tl in problem.transport_links:
            products_flow_on_this_link = all_flows_per_link_product[tl]
            if not products_flow_on_this_link:
                continue

            # Sort products by size (largest first for denser packing)
            # Make a copy of flows to modify as we pack
            current_products_to_pack = {p: qty for p, qty in products_flow_on_this_link.items() if qty > 0}
            sorted_products = sorted(current_products_to_pack.keys(), key=lambda prod: prod.size, reverse=True)

            # Keep packing until no more products or no more trips
            max_trips_for_tl = tl.max_trips if tl.max_trips is not None else sys.maxsize

            while any(qty > 0 for qty in current_products_to_pack.values()) and link_trips_taken[tl] < max_trips_for_tl:
                current_packing_config = {}
                current_trip_volume = 0

                for p in sorted_products:
                    if current_products_to_pack[p] > 0:
                        # Calculate how much of 'p' can fit in the remaining capacity of the current trip
                        space_left_in_current_trip = tl.transport_type.capacity - current_trip_volume

                        if p.size == 0:  # If product has no size, it takes no volume, assume all remaining fit
                            units_of_p_that_fit = current_products_to_pack[p]
                        else:
                            units_of_p_that_fit = space_left_in_current_trip // p.size

                        qty_to_add = min(current_products_to_pack[p], units_of_p_that_fit)

                        if qty_to_add > 0:
                            current_packing_config[p] = current_packing_config.get(p, 0) + qty_to_add
                            current_trip_volume += qty_to_add * p.size
                            current_products_to_pack[p] -= qty_to_add

                if current_packing_config:  # If we managed to pack something into this trip
                    final_packing_transports.append(PackingTransport(tl, dict(current_packing_config), 1))
                    link_trips_taken[tl] += 1
                else:  # No more products fit in current trip or no products left to pack
                    break  # Exit while loop for this tl

        solution = MultibatchingSolution(problem=problem, list_flows=final_packing_transports)
        fit = self.aggreg_from_sol(solution)
        result_storage = self.create_result_storage([(solution, fit)])
        return result_storage
    # EVOLVE-BLOCK-END


