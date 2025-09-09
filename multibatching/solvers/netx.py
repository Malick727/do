from collections import defaultdict
from typing import Optional, Any

import networkx as nx

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from multibatching.problem import MultibatchingProblem, MultibatchingSolution, Product, PackingTransport, TransportType, \
    Location, TransportLink
from discrete_optimization.generic_tools.do_solver import SolverDO


class NetxMultibatchingSolver(SolverDO):

    problem: MultibatchingProblem

    def compute_optimal_flow_for_product(self, product: Product):
        problem = self.problem
        G = nx.DiGraph()
        # source = 'source'
        # sink = 'sink'
        # G.add_nodes_from([source, sink])

        total_supply = 0
        total_demand = 0
        # 2. Add nodes for each location and connect source/sink
        demand_per_location = {}
        for loc in problem.locations:
            supply = loc.net_supply.get(product, 0)
            G.add_node(loc.id, demand=0,
                       type="location")
            demand_per_location[loc.id] = supply
            if supply > 0:
                # Add edge from super-source to supplier locations
                # G.add_edge(source, loc.id, capacity=supply)
                total_supply += supply
            elif supply < 0:
                # Add edge from demander locations to super-sink
                demand = -supply
                # G.add_edge(loc.id, sink, capacity=demand)
                total_demand += demand

        G.add_node("source", demand=-total_supply, type="source")
        G.add_node("sink", demand=total_demand, type="sink")
        for loc in demand_per_location:
            if demand_per_location[loc] > 0:
                G.add_edge("source",
                           loc,
                           capacity=demand_per_location[loc],
                           weight=0)
                #print(f"adding link from source to " + loc)
            if demand_per_location[loc] < 0:
                G.add_edge(loc,
                           "sink",
                           capacity=-demand_per_location[loc],
                           weight=0)
                #print(f"adding link from {loc} to sink")

        for link in problem.transport_links:
            if link.transport_type in product.valid_transports:
                if link.transport_type.capacity == 0:
                    continue
                if link.transport_type.capacity < product.size:
                    continue
                coefficient = (max(1, product.size / link.transport_type.capacity) *
                               (link.transport_type.cost + link.transport_type.emissions) * link.distance)
                intermediary_node = (link.location_l1.id,
                                     link.location_l2.id,
                                     link.transport_type.id)
                G.add_node(intermediary_node, demand=0,
                           type="intermediary")
                G.add_edge(link.location_l1.id,
                           intermediary_node,
                           weight=int(coefficient),
                           capacity=total_supply)
                G.add_edge(intermediary_node,
                           link.location_l2.id,
                           weight=0,
                           capacity=total_supply)
        #dict_flow = nx.max_flow_min_cost(G,
        #                                  s="source",
        #                                  t="sink",
        #                                  capacity="capacity",
        #                                  weight="weight")
        dict_flow = nx.min_cost_flow(G,
                                     demand="demand",
                                     capacity="capacity",
                                     weight="weight")
        cost_of_flow = nx.cost_of_flow(G, flowDict=dict_flow, weight="weight")
        # print("cost of flow", cost_of_flow)
        if False:
            for n0 in dict_flow:
                for n1 in dict_flow[n0]:
                    if dict_flow[n0][n1] > 0:
                        print("flow between ", n0, " and ", n1, " : ", dict_flow[n0][n1])
                        print("Cost : ", dict_flow[n0][n1] * G[n0][n1]["weight"])
        #print(cost_of_flow)
        return G, dict_flow, cost_of_flow

    def solve(self, callbacks: Optional[list[Callback]] = None, **kwargs: Any) -> ResultStorage:
        supply_per_product = {p: self.problem.get_total_supply(p) for p in self.problem.products}
        demand_per_product = {p: self.problem.get_total_demand(p) for p in self.problem.products}
        flow_per_product = defaultdict(lambda: set())
        packs: dict[TransportLink, PackingTransport] = {}
        total_cost_of_flow = 0
        for product in supply_per_product:
            size_product = product.size
            if supply_per_product[product] == 0:
                continue
            graph, flow, cost_of_flow = self.compute_optimal_flow_for_product(product=product)
            print(product.name, " cost of flow : ", cost_of_flow)
            total_cost_of_flow += cost_of_flow
            for node_1 in flow:
                type_node_1 = graph.nodes[node_1]["type"]
                for node_2 in flow[node_1]:
                    type_node_2 = graph.nodes[node_2]["type"]
                    if flow[node_1][node_2] > 0:
                        value = flow[node_1][node_2]
                        if type_node_2 == "intermediary":
                            loc1_id, loc2_id, transport_id = node_2
                            transport: TransportType = self.problem.transport_type_id_to_transport_type[transport_id]
                            loc1: Location = self.problem.location_id_to_location[loc1_id]
                            loc2: Location = self.problem.location_id_to_location[loc2_id]
                            capacity_transport = transport.capacity
                            transport_link = self.problem.loc_and_transport_type_to_transport_link[node_2]
                            if transport_link in packs:
                                packs[transport_link].product_packing[product] = value
                            else:
                                packing = PackingTransport(transport_link=transport_link,
                                                           product_packing={product: value},
                                                           nb_packing=1)
                                packs[transport_link] = packing
        print(total_cost_of_flow, " total cost flow")
        sol = MultibatchingSolution(problem=self.problem,
                                    list_flows=list(packs.values()))
        fit = self.aggreg_from_sol(sol)
        return self.create_result_storage([(sol, fit)])
