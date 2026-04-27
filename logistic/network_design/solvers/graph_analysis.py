from collections import defaultdict
import networkx as nx
from logistic.network_design.preprocessing import PrecomputedRoute
from logistic.network_design.problem import Location, Part, TransportOption, GlobalLogisticProblem
import logging
logger = logging.getLogger(__name__)

class LogisticProblemGraphAnalysis:
    problem: GlobalLogisticProblem
    def __init__(self, problem: GlobalLogisticProblem):
        self.problem = problem
        self.graph_per_parts = {part: self.build_transport_graph(part)
                                for part in self.problem.parts}

    def build_transport_graph(self, part: Part) -> 'nx.DiGraph':
        """
        Build a directed transport graph for a specific part.

        Nodes: Locations
        Edges: Transport options for this part

        Returns a NetworkX DiGraph where paths represent feasible transport routes.
        """

        G = nx.DiGraph()
        # Add all locations as nodes
        for loc in self.problem.locations:
            G.add_node(loc)
        # Add transport edges for this part
        t_options_per_origin_dest = defaultdict(list)
        for t_opt in self.problem.transport_options:
            if t_opt.part == part:
                t_options_per_origin_dest[(t_opt.origin,
                                           t_opt.destination)].append(t_opt)
        for t_opt in self.problem.transport_options:
            if t_opt.part == part:
                cheapest = min(t_options_per_origin_dest[(t_opt.origin, t_opt.destination)],
                               key=lambda x: x.total_cost)
                cost = cheapest.total_cost
                G.add_edge(t_opt.origin,
                           t_opt.destination,
                           transport_option=cheapest,
                           weight=cost)
                           #t_options_per_origin_dest[(t_opt.origin,
                           #                           t_opt.destination)])
        return G

    def get_all_necessary_part(self, part: Part):
        """Recursively get all children and their children."""
        descendants = set()
        for child in part.child_parts:
            descendants.add(child)
            descendants.update(self.get_all_necessary_part(child))
        return descendants

    def get_all_next_parts(self, part: Part):
        """Recursively get all children and their children."""
        next_parts = set()
        if part.parent_part is not None:
            next_parts.add(part.parent_part)
            next_parts.add(self.get_all_next_parts(part.parent_part))
        return next_parts

    # def get_all_descendants(self, part: Part):

    def get_part_and_location_graph(self):
        """
        Build the (part, location) dependency graph with cached shortest paths.

        Edges include precomputed transport routes stored in 'route' attribute.
        This avoids recomputing shortest paths during solution reconstruction.
        """
        graph = nx.DiGraph()
        for part in self.problem.parts:
            for loc in part.produceable_at:
                graph.add_node((part, loc))

        for part in self.problem.parts:
            graph_of_part = self.graph_per_parts[part]
            parent = part.parent_part
            if parent:
                locs_current_parts = part.produceable_at
                locs_next_part = parent.produceable_at
                for loc_cp in locs_current_parts:
                    next_locs = nx.descendants(graph_of_part, loc_cp)
                    next_locs = [loc for loc in next_locs
                                 if loc in locs_next_part]
                    if len(next_locs) > 0:
                        for nl in next_locs:
                            # Cache the shortest path for this edge
                            route = self.get_shortest_path(part, loc_cp, nl)
                            graph.add_edge(
                                (part, loc_cp),
                                (parent, nl),
                                weight=1,
                                route=route  # Cache precomputed route
                            )

                    if loc_cp in locs_next_part and loc_cp not in next_locs:
                        # Local consumption - no transport needed
                        graph.add_edge(
                            (part, loc_cp),
                            (parent, loc_cp),
                            weight=0,
                            route=None  # No transport needed for local consumption
                        )
        return graph

    def get_shortest_path(self,
                          part: Part,
                          loc_1: Location,
                          loc_2: Location):
        try:
            # Use Dijkstra's algorithm
            g = self.graph_per_parts[part]
            path_nodes = nx.shortest_path(g,
                                          loc_1, loc_2,
                                          weight='weight')
            cost = [g[e0][e1]["weight"]
                    for e0, e1 in zip(path_nodes[:-1], path_nodes[1:])]
            path_cost = sum(cost)
            # Reconstruct transport sequence
            transport_sequence = []
            transport_means_used = set()

            for i in range(len(path_nodes) - 1):
                src = path_nodes[i]
                dst = path_nodes[i + 1]
                edge_data = g[src][dst]
                t_opt: TransportOption = edge_data['transport_option']
                transport_sequence.append(t_opt)
                transport_means_used.add(t_opt.mean.name)

            return PrecomputedRoute(
                part=part,
                origin=loc_1,
                destination=loc_2,
                transport_sequence=tuple(transport_sequence),
                total_cost=path_cost,
                transport_means_used=frozenset(transport_means_used),
                is_feasible=True
            )

        except nx.NetworkXNoPath:
            # No path exists
            logger.debug(f"  No path: {part.name} from {loc_1.name} to {loc_2.name}")
            return PrecomputedRoute(
                part=part,
                origin=loc_1,
                destination=loc_2,
                transport_sequence=tuple(),
                total_cost=float('inf'),
                transport_means_used=frozenset(),
                is_feasible=False
            )

