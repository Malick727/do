import logging
import os
from dataclasses import dataclass
from typing import Optional, Set, List, Dict, Any

import networkx as nx
import numpy as np

from discrete_optimization.generic_tasks_tools.allocation import AllocationSolution, AllocationProblem
from discrete_optimization.generic_tools.do_problem import Solution, ObjectiveRegister, ModeOptim, ObjectiveHandling, \
    ObjectiveDoc, TypeObjective
from discrete_optimization.generic_tools.encoding_register import EncodingRegister

logger = logging.getLogger(__name__)


@dataclass
class CountryConstraint:
    """Minimum and/or maximum value-added constraint for a country."""
    country: str
    min_value_added: Optional[float] = None
    max_value_added: Optional[float] = None

    def __repr__(self):
        bounds = []
        if self.min_value_added is not None:
            bounds.append(f"min={self.min_value_added}")
        if self.max_value_added is not None:
            bounds.append(f"max={self.max_value_added}")
        return f"CountryConstraint({self.country}: {', '.join(bounds)})"


class Location:
    """Represents a production site or warehouse in the network."""
    def __init__(self, name: str):
        self.name = name
        self.country: str = "unknown"
        self.factory_activation_cost: float = 0.0  # Fixed cost to activate this factory
        self.is_production: bool = False
        self.is_warehouse: bool = False
        self.min_value_added: Optional[float] = None
        self.max_value_added: Optional[float] = None

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Location) and self.name == other.name

    def __repr__(self):
        roles = []
        if self.is_production: roles.append("Prod")
        if self.is_warehouse: roles.append("WH")
        role_str = "/".join(roles) if roles else "Unknown"
        return f"Location({self.name} - {role_str})"


class Part:
    """Represents a component, sub-assembly, or final product."""

    def __init__(self, name: str):
        self.name = name
        self.value_added: float = 0.0
        self.produceable_at: Set[Location] = set()
        self.parent_part: Optional['Part'] = None  # A part has only ONE successor
        self.child_parts: List['Part'] = []  # A part can have SEVERAL predecessors
        self.sourcing_requirement: int = 1  # Minimum active sites required

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Part) and self.name == other.name

    def __repr__(self):
        return f"Part({self.name})"


class TransportMean:
    """Represents the transport vehicle (e.g., truck, ship)."""

    def __init__(self, name: str):
        self.name = name
        self.nrc_cost: float = 0.0

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, TransportMean) and self.name == other.name

    def __repr__(self):
        return f"Mean({self.name})"


Task = Part
UnaryResource = Location


@dataclass(frozen=True)
class TransportOption:
    """An immutable, object-oriented representation of a routing edge."""
    origin: Location
    destination: Location
    part: Part
    mean: TransportMean
    duration: float
    total_cost: float

    def __repr__(self):
        return f"Route({self.part.name}: {self.origin.name} -> {self.destination.name} via {self.mean.name})"


class GlobalLogisticSolution(AllocationSolution[Task, UnaryResource]):

    def __init__(self, problem: "GlobalLogisticProblem",
                 part_to_locations: Dict[Part, Set[Location]],
                 transport_plan: Set[TransportOption], **kwargs: Any):
        """
        part_to_locations: Maps a Part object to a SET of activated Location objects.
        transport_plan: A Set of chosen TransportOption objects representing the active network.
        """
        super().__init__(problem)
        self.problem = problem
        self.part_to_locations = part_to_locations
        self.transport_plan = transport_plan
        self.kwargs = kwargs

    def is_allocated(self, task: Task, unary_resource: UnaryResource) -> bool:
        return unary_resource in self.part_to_locations.get(task, set())

    def __str__(self) -> str:
        lines = ["\n" + "=" * 60]
        lines.append(" GLOBAL LOGISTICS SOLUTION (SMC + ROUTING)")
        lines.append("=" * 60)

        lines.append("\n🏭 PRODUCTION ASSIGNMENTS:")
        lines.append("-" * 40)
        for part, locs in sorted(self.part_to_locations.items(), key=lambda p: p[0].name):
            loc_names = ", ".join([loc.name for loc in locs])
            lines.append(f"  • {part.name:<15} -> [ {loc_names} ]")

        lines.append("\n🚢 TRANSPORTATION PLAN:")
        lines.append("-" * 40)
        if not self.transport_plan:
            lines.append("  (No inter-site transport required)")
        else:
            total_transport_cost = 0.0
            for t_opt in sorted(self.transport_plan, key=lambda t: (t.part.name, t.origin.name)):
                total_transport_cost += t_opt.total_cost
                lines.append(
                    f"  • Ship '{t_opt.part.name}'\n"
                    f"      Route: {t_opt.origin.name} -> {t_opt.destination.name} via {t_opt.mean.name}\n"
                    f"      Cost : €{t_opt.total_cost:,.2f}"
                )
            lines.append("-" * 40)
            lines.append(f"  Total Routing Cost: €{total_transport_cost:,.2f}")

        lines.append("=" * 60 + "\n")
        return "\n".join(lines)

    def copy(self) -> "GlobalLogisticSolution":
        return GlobalLogisticSolution(
            self.problem,
            {k: set(v) for k, v in self.part_to_locations.items()},
            self.transport_plan.copy(),
            **self.kwargs
        )

    def lazy_copy(self) -> "GlobalLogisticSolution":
        return self.copy()


class GlobalLogisticProblem(AllocationProblem[Task, UnaryResource]):
    @property
    def unary_resources_list(self) -> list[UnaryResource]:
        return self.locations

    @property
    def tasks_list(self) -> list[Task]:
        return self.parts

    def is_compatible_task_unary_resource(self, task: Task, unary_resource: UnaryResource) -> bool:
        return unary_resource in task.produceable_at

    def get_attribute_register(self) -> EncodingRegister:
        return EncodingRegister({})

    def get_solution_type(self) -> type[Solution]:
        return GlobalLogisticSolution

    def get_objective_register(self) -> ObjectiveRegister:
        return ObjectiveRegister(
            objective_sense=ModeOptim.MINIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc={
                "factory_activation_cost": ObjectiveDoc(type=TypeObjective.OBJECTIVE,
                                                        default_weight=0),
                "part_production_cost": ObjectiveDoc(type=TypeObjective.OBJECTIVE,
                                                     default_weight=0),
                "production_cost": ObjectiveDoc(type=TypeObjective.OBJECTIVE,
                                                default_weight=1.0),  # Aggregated
                "transport_cost": ObjectiveDoc(type=TypeObjective.OBJECTIVE,
                                               default_weight=0.0),
                "nrc_cost": ObjectiveDoc(type=TypeObjective.OBJECTIVE,
                                         default_weight=0.0),
                "total_transport_cost": ObjectiveDoc(type=TypeObjective.OBJECTIVE,
                                                     default_weight=1.0) # Aggregated
            }
        )

    def __init__(self, data_folder: str):
        self.data_folder = data_folder

        # String registries used primarily during parsing
        self._parts_registry: Dict[str, Part] = {}
        self._locations_registry: Dict[str, Location] = {}
        self._means_registry: Dict[str, TransportMean] = {}

        # Public lists of domain objects
        self.parts: List[Part] = []
        self.locations: List[Location] = []
        self.transport_means: List[TransportMean] = []
        self.transport_options: List[TransportOption] = []

        self.country_constraints: List[CountryConstraint] = []

        # Part-location specific production costs (variable cost per part produced)
        # This is separate from factory_activation_cost (fixed cost per active location)
        self.part_location_production_costs: Dict[tuple[Part, Location], int] = {}

        self._parse_data()
        self._initialize_part_location_costs()

    def _extract_facts(self, filename: str) -> List[tuple[str, List[str]]]:
        """Reads ASP facts safely, stripping comments and quotes."""
        facts = []
        filepath = os.path.join(self.data_folder, filename)
        if not os.path.exists(filepath):
            return facts

        with open(filepath, "r") as f:
            for line in f:
                # Remove inline ASP comments (%) and whitespace
                line = line.split("%")[0].strip()
                if not line:
                    continue

                # Match standard ASP fact format: predicate(arg1, arg2).
                if "(" in line and line.endswith("."):
                    pred = line[:line.index("(")].strip()
                    args_str = line[line.index("(") + 1: line.rindex(")")]
                    # Split by comma and strip whitespace and quotes from each argument
                    args = [a.strip().strip("'\"") for a in args_str.split(",")]
                    facts.append((pred, args))
        return facts

    def _parse_data(self) -> None:
        """Two-pass parsing with permissive predicate matching."""

        # PASS 1: Object Instantiation
        for pred, args in self._extract_facts("product"):
            if pred == "part":
                self._parts_registry[args[0]] = Part(args[0])

        for pred, args in self._extract_facts("location"):
            loc_name = args[0]
            # Ensure the object exists
            if loc_name not in self._locations_registry:
                self._locations_registry[loc_name] = Location(loc_name)
            # Flag its specific role based on the predicate
            if pred in ("productionLoc", "productionLocation"):
                self._locations_registry[loc_name].is_production = True
            elif pred in ("warehouseLoc", "warehouseLocation"):
                self._locations_registry[loc_name].is_warehouse = True

        for pred, args in self._extract_facts("transportmean"):
            if pred in ("transportMean", "truck", "ship", "shortDistanceTransportMean", "longDistanceTransportMean"):
                if args[0] not in self._means_registry:
                    self._means_registry[args[0]] = TransportMean(args[0])

        # PASS 2: Data Population and Object Linking
        for pred, args in self._extract_facts("product"):
            # Matches paper's valAdded and dummy's valueAdded
            if pred in ("valueAdded", "valAdded") and args[0] in self._parts_registry:
                self._parts_registry[args[0]].value_added = float(args[1])

        for pred, args in self._extract_facts("location"):
            # Matches paper's costLoc and dummy's costLocation
            if pred in ("costLocation", "costLoc") and args[0] in self._locations_registry:
                self._locations_registry[args[0]].factory_activation_cost = float(args[1])

        for pred, args in self._extract_facts("located-in"):
            if pred in ("locatedIn", "located-in") and args[0] in self._locations_registry:
                self._locations_registry[args[0]].country = args[1]

        for pred, args in self._extract_facts("transportmean"):
            if pred in ("transportMeanNRC", "nrc") and args[0] in self._means_registry:
                self._means_registry[args[0]].nrc_cost = float(args[1])

        # The BOM (Bill of Materials) & Capabilities
        for pred, args in self._extract_facts("produceable-at"):
            # Permissive check for both notations
            if pred in ("produceableAt", "partProduceableAt") and len(args) >= 2:
                if args[0] in self._parts_registry and args[1] in self._locations_registry:
                    part_obj = self._parts_registry[args[0]]
                    loc_obj = self._locations_registry[args[1]]
                    part_obj.produceable_at.add(loc_obj)

        for pred, args in self._extract_facts("production-plan"):
            if pred == "productionPlan" and len(args) >= 2:
                # productionPlan(parent, child) [cite: 1]
                parent = self._parts_registry.get(args[0])
                child = self._parts_registry.get(args[1])
                if parent and child:
                    child.parent_part = parent
                    parent.child_parts.append(child)

        # Routing edges
        for pred, args in self._extract_facts("can-be-transported-from-to"):
            # Matches paper's canTransport and dummy's canBeTransportedFromTo [cite: 1]
            if pred in ("canTransport", "canBeTransportedFromTo") and len(args) >= 8:
                origin = self._locations_registry.get(args[0])
                dest = self._locations_registry.get(args[1])
                part = self._parts_registry.get(args[2])
                mean = self._means_registry.get(args[3])

                if origin and dest and part and mean:
                    self.transport_options.append(TransportOption(
                        origin=origin, destination=dest, part=part, mean=mean,
                        duration=float(args[5]), total_cost=float(args[7])
                    ))

        # Expose lists publicly
        self.parts = list(self._parts_registry.values())
        self.locations = list(self._locations_registry.values())
        self.transport_means = list(self._means_registry.values())

    def _initialize_part_location_costs(self) -> None:
        """
        Initialize part-location specific production costs.

        Costs are correlated with factory_activation_cost:
        - Cheap locations (low factory cost) → cheaper production costs
        - Expensive locations (high factory cost) → more expensive production costs

        This creates interesting Pareto trade-offs between:
        - Opening few expensive factories with high production costs
        - Opening many cheap factories with low production costs

        Uses deterministic random generation for reproducibility.
        """
        import hashlib

        # Find min/max factory activation costs
        factory_costs = [loc.factory_activation_cost for loc in self.locations if loc.factory_activation_cost > 0]
        if not factory_costs:
            # Fallback if no factory costs defined
            min_factory_cost, max_factory_cost = 0, 100
        else:
            min_factory_cost = min(factory_costs)
            max_factory_cost = max(factory_costs)

        for part in self.parts:
            for loc in part.produceable_at:
                # Create deterministic seed from part and location names
                seed_string = f"{part.name}_{loc.name}"
                seed_hash = int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16)
                rng = np.random.RandomState(seed_hash % (2**32))

                # Normalize factory cost to [0, 1]
                if max_factory_cost > min_factory_cost:
                    factory_cost_ratio = (loc.factory_activation_cost - min_factory_cost) / (max_factory_cost - min_factory_cost)
                else:
                    factory_cost_ratio = 0.5

                # Correlate production cost with factory cost
                # Cheap factory (ratio=0) → cost in [10, 50]
                # Expensive factory (ratio=1) → cost in [60, 100]
                min_cost = int(10 + factory_cost_ratio * 300)  # 10 to 60
                max_cost = int(50 + factory_cost_ratio * 300)  # 50 to 100

                # Generate cost in the correlated range
                cost = int(rng.randint(min_cost, max_cost + 1))

                self.part_location_production_costs[(part, loc)] = cost

    # =====================================================================
    # 4. Evaluation & Constraint Satisfaction
    # =====================================================================

    def get_all_countries(self) -> Set[str]:
        """Utility to get all unique, known countries in the network."""
        return {loc.country for loc in self.locations if loc.country and loc.country != "unknown"}

    def build_transport_graph(self, part: Part) -> nx.DiGraph:
        """
        Build a directed transport graph for a specific part.

        Nodes: Locations
        Edges: Transport options for this part

        Returns a NetworkX DiGraph where paths represent feasible transport routes.
        """
        graph_part = nx.DiGraph()
        # Add all locations as nodes
        for loc in self.locations:
            graph_part.add_node(loc)
        # Add transport edges for this part
        for t_opt in self.transport_options:
            if t_opt.part == part:
                graph_part.add_edge(t_opt.origin, t_opt.destination, transport_option=t_opt)
        return graph_part

    def compute_value_added_by_country(self, solution: GlobalLogisticSolution) -> Dict[str, float]:
        """Compute cumulative value-added per country for a solution."""
        value_added_by_country = {c: 0.0 for c in self.get_all_countries()}
        for part, locs in solution.part_to_locations.items():
            for loc in locs:
                if loc.country in value_added_by_country:
                    value_added_by_country[loc.country] += part.value_added
        return value_added_by_country

    def compute_value_added_by_location(self, solution: GlobalLogisticSolution) -> Dict[Location, float]:
        """Compute cumulative value-added per location for a solution."""
        value_added_by_location = {loc: 0.0 for loc in self.locations}
        for part, locs in solution.part_to_locations.items():
            for loc in locs:
                value_added_by_location[loc] += part.value_added
        return value_added_by_location

    def get_all_necessary_part(self, part: Part):
        """Recursively get all necessary parts for a given part"""
        descendants = set()
        for child in part.child_parts:
            descendants.add(child)
            descendants.update(self.get_all_necessary_part(child))
        return descendants

    def evaluate(self, solution: GlobalLogisticSolution) -> Dict[str, float]:
        factory_activation_cost = 0
        part_production_cost = 0
        transport_cost = 0.0
        nrc_cost = 0.0

        # 1. Factory Activation Cost (fixed cost per active location)
        active_locations = {loc for locs in solution.part_to_locations.values() for loc in locs}
        for loc in active_locations:
            factory_activation_cost += int(loc.factory_activation_cost)

        # 2. Part Production Cost (variable cost per part-location assignment)
        for part, locs in solution.part_to_locations.items():
            for loc in locs:
                part_production_cost += self.part_location_production_costs.get((part, loc), 0)

        # 3. Transport & Fleet NRC Cost
        active_means = set()
        for t_opt in solution.transport_plan:
            transport_cost += t_opt.total_cost
            active_means.add(t_opt.mean)

        for mean in active_means:
            nrc_cost += mean.nrc_cost

        return {
            "factory_activation_cost": float(factory_activation_cost),
            "part_production_cost": float(part_production_cost),
            "production_cost": float(factory_activation_cost + part_production_cost),  # Aggregated
            "transport_cost": float(transport_cost),
            "nrc_cost": float(nrc_cost),
            "total_transport_cost": float(transport_cost+nrc_cost)
        }

    def satisfy(self, solution: GlobalLogisticSolution, verbose: bool = False) -> bool:
        """
        Validates the Set Multicover and multi-hop Graph Flow constraints.
        Uses a local flow conservation logic at every node to allow intermediate warehouses.
        """
        for part in self.parts:
            # 1. Source requirement check
            locs = solution.part_to_locations.get(part, set())
            if len(locs) < part.sourcing_requirement:
                if verbose:
                    logger.warning(f"❌ VIOLATED: Sourcing requirement for {part.name}")
                    logger.warning(f"   Required: {part.sourcing_requirement}, Found: {len(locs)}")
                    logger.warning(f"   Locations: {[l.name for l in locs]}")
                return False
            # 2. Eligibility check
            if not locs.issubset(part.produceable_at):
                if verbose:
                    invalid_locs = locs - part.produceable_at
                    logger.warning(f"❌ VIOLATED: Eligibility for {part.name}")
                    logger.warning(f"   Invalid locations: {[l.name for l in invalid_locs]}")
                return False

            # 3. Multi-Hop Flow Conservation Check
            # We check the local balance for this part at EVERY location in the network
            for loc in self.locations:
                # Local Activity States
                produced = loc in locs
                consumed = part.parent_part and (loc in solution.part_to_locations.get(part.parent_part, set()))
                # Transport Edge States
                inbound = any(t.part == part and t.destination == loc for t in solution.transport_plan)
                outbound = any(t.part == part and t.origin == loc for t in solution.transport_plan)
                # Rule A: No Starvation
                # If the factory builds the parent, the child MUST be available locally.
                if consumed and not (produced or inbound):
                    if verbose:
                        logger.warning(f"❌ VIOLATED: Rule A (No Starvation)")
                        logger.warning(f"   Part: {part.name}, Location: {loc.name}")
                        logger.warning(f"   Parent {part.parent_part.name} is produced here but {part.name} is NOT available")
                        logger.warning(f"   produced={produced}, inbound={inbound}")
                    return False

                # Rule B: No Dead-Ends (Orphans)
                # If the part exists here (built or received) and isn't the final product,
                # it MUST be used locally or shipped out.
                if part.parent_part and (produced or inbound):
                    if not (consumed or outbound):
                        if verbose:
                            logger.warning(f"❌ VIOLATED: Rule B (No Dead-Ends)")
                            logger.warning(f"   Part: {part.name}, Location: {loc.name}")
                            logger.warning(f"   Part exists here (produced={produced}, inbound={inbound})")
                            logger.warning(f"   But is neither consumed nor shipped out (consumed={consumed}, outbound={outbound})")
                        return False

                # Rule C: No Spontaneous Generation
                # If a node ships the part out, it MUST have built it or received it.
                if outbound and not (produced or inbound):
                    if verbose:
                        logger.warning(f"❌ VIOLATED: Rule C (No Spontaneous Generation)")
                        logger.warning(f"   Part: {part.name}, Location: {loc.name}")
                        logger.warning(f"   Part is shipped out but was NOT produced or received here")
                        logger.warning(f"   produced={produced}, inbound={inbound}")
                    return False

        # 4. Country Work-Share Constraints
        if self.country_constraints:
            value_added_by_country = self.compute_value_added_by_country(solution)
            for constraint in self.country_constraints:
                country_value = value_added_by_country.get(constraint.country, 0.0)

                if constraint.min_value_added is not None and country_value < constraint.min_value_added:
                    if verbose:
                        logger.warning(f"❌ VIOLATED: Minimum work-share for {constraint.country}")
                        logger.warning(f"   Required: >={constraint.min_value_added}, Found: {country_value}")
                    return False

                if constraint.max_value_added is not None and country_value > constraint.max_value_added:
                    if verbose:
                        logger.warning(f"❌ VIOLATED: Maximum work-share for {constraint.country}")
                        logger.warning(f"   Required: <={constraint.max_value_added}, Found: {country_value}")
                    return False

        # 5. Location-Level Value-Added Constraints
        for loc in self.locations:
            if loc.min_value_added is not None or loc.max_value_added is not None:
                # Compute value-added for this location
                loc_value = sum(
                    part.value_added
                    for part, locs in solution.part_to_locations.items()
                    if loc in locs
                )

                if loc.min_value_added is not None and loc_value < loc.min_value_added:
                    if verbose:
                        logger.warning(f"❌ VIOLATED: Minimum value-added for location {loc.name}")
                        logger.warning(f"   Required: >={loc.min_value_added}, Found: {loc_value}")
                    return False

                if loc.max_value_added is not None and loc_value > loc.max_value_added:
                    if verbose:
                        logger.warning(f"❌ VIOLATED: Maximum value-added for location {loc.name}")
                        logger.warning(f"   Required: <={loc.max_value_added}, Found: {loc_value}")
                    return False

        return True

    def get_dummy_solution(self) -> GlobalLogisticSolution:
        """
        Creates a feasible baseline dummy solution using a top-down greedy graph traversal.
        Ensures selected child locations have routes to parent locations AND can actually produce the part.
        """
        part_to_locations = {}
        transport_plan = set()

        # 1. Separate parts by top-level (no parent) vs sub-components
        top_level_parts = [p for p in self.parts if not p.parent_part]

        # 2. Assign top-level parts first
        for part in top_level_parts:
            req = part.sourcing_requirement
            valid_locs = list(part.produceable_at)
            part_to_locations[part] = set(valid_locs[:req])

        # 3. Top-down greedy assignment
        queue = list(top_level_parts)

        while queue:
            current_parent = queue.pop(0)
            p_locs = part_to_locations.get(current_parent, set())

            for child in current_parent.child_parts:
                if child not in part_to_locations:
                    part_to_locations[child] = set()

                c_locs = part_to_locations[child]
                req = child.sourcing_requirement

                for p_loc in p_locs:
                    # Hub logic: Is it built locally?
                    if p_loc in child.produceable_at:
                        c_locs.add(p_loc)
                        continue

                    # FIXED: Find routes where the origin is ACTUALLY capable of producing the child
                    valid_routes = [
                        t for t in self.transport_options
                        if t.part == child
                           and t.destination == p_loc
                           and t.origin in child.produceable_at
                    ]

                    if valid_routes:
                        chosen_route = valid_routes[0]
                        c_loc = chosen_route.origin
                        c_locs.add(c_loc)
                        transport_plan.add(chosen_route)
                    else:
                        import logging
                        logging.warning(f"Dummy Gen Failed: No valid inbound route for {child.name} to {p_loc.name}")

                # FIXED: Ensure fallback redundancy options are also valid production sites
                if len(c_locs) < req:
                    all_valid_routes = [
                        t for t in self.transport_options
                        if t.part == child
                           and t.destination in p_locs
                           and t.origin not in c_locs
                           and t.origin in child.produceable_at  # <-- THE CRITICAL FIX
                    ]
                    for route in all_valid_routes:
                        c_locs.add(route.origin)
                        transport_plan.add(route)
                        if len(c_locs) >= req:
                            break

                queue.append(child)

        return GlobalLogisticSolution(self, part_to_locations, transport_plan)


def display_solution_statistics(problem: 'GlobalLogisticProblem', solution: 'GlobalLogisticSolution') -> None:
    """
    Display comprehensive statistics per location for a given solution.

    Shows:
    - Number of parts produced
    - Total value-added
    - Parts transiting through (incoming + outgoing without consumption)
    - Parts imported and consumed
    - Parts produced and exported
    - Parts produced and consumed locally
    """
    import pandas as pd
    from collections import defaultdict

    # Initialize stats per location
    location_stats = {loc: {
        'parts_produced': [],
        'value_added': 0.0,
        'parts_imported_consumed': [],
        'parts_produced_exported': [],
        'parts_produced_consumed_local': [],
        'parts_transiting': [],
        'inbound_flows': 0,
        'outbound_flows': 0,
    } for loc in problem.locations}

    # Collect production information
    for part, locs in solution.part_to_locations.items():
        for loc in locs:
            location_stats[loc]['parts_produced'].append(part.name)
            location_stats[loc]['value_added'] += part.value_added

    # Build a detailed flow graph per part per location
    location_part_flows = defaultdict(lambda: defaultdict(lambda: {
        'inbound': [], 'outbound': [],
        'produced': False, 'consumed': False
    }))

    for t_opt in solution.transport_plan:
        part = t_opt.part
        origin = t_opt.origin
        dest = t_opt.destination

        # Count flows
        location_stats[origin]['outbound_flows'] += 1
        location_stats[dest]['inbound_flows'] += 1

        # Track flows per part per location
        location_part_flows[dest][part]['inbound'].append(t_opt)
        location_part_flows[origin][part]['outbound'].append(t_opt)

    # Mark production and consumption
    for part, locs in solution.part_to_locations.items():
        for loc in locs:
            location_part_flows[loc][part]['produced'] = True

    for part in problem.parts:
        if part.parent_part:
            parent_locs = solution.part_to_locations.get(part.parent_part, set())
            for loc in parent_locs:
                location_part_flows[loc][part]['consumed'] = True

    # Analyze each location's flow patterns
    for loc in problem.locations:
        for part, flows in location_part_flows[loc].items():
            has_inbound = len(flows['inbound']) > 0
            has_outbound = len(flows['outbound']) > 0
            is_produced = flows['produced']
            is_consumed = flows['consumed']

            # Build description with flow pattern
            flow_desc = f"{part.name}"
            flags = []
            if is_produced:
                flags.append("PROD")
            if is_consumed:
                flags.append("CONS")
            if has_inbound:
                flags.append(f"IN:{len(flows['inbound'])}")
            if has_outbound:
                flags.append(f"OUT:{len(flows['outbound'])}")

            if flags:
                flow_desc += f" [{'+'.join(flags)}]"

            # Categorize (can be in multiple categories if flows are suspicious)
            if is_produced and has_outbound:
                location_stats[loc]['parts_produced_exported'].append(flow_desc)

            if has_inbound and is_consumed:
                location_stats[loc]['parts_imported_consumed'].append(flow_desc)

            if has_inbound and has_outbound and not is_produced:
                # Transiting through
                location_stats[loc]['parts_transiting'].append(flow_desc)

                # Flag suspicious patterns
                if is_consumed:
                    logger.warning(
                        f"⚠️  SUSPICIOUS: {part.name} at {loc.name} has inbound AND outbound "
                        f"flows but is consumed (parent={part.parent_part.name if part.parent_part else 'none'})"
                    )

    # Check for locally produced and consumed parts
    for part in problem.parts:
        if not part.parent_part:
            continue

        parent = part.parent_part
        for loc in problem.locations:
            part_produced_here = loc in solution.part_to_locations.get(part, set())
            parent_produced_here = loc in solution.part_to_locations.get(parent, set())

            if part_produced_here and parent_produced_here:
                # Part is produced and consumed at same location
                part_exported = any(
                    t.part == part and t.origin == loc
                    for t in solution.transport_plan
                )
                if not part_exported:
                    # Only consumed locally (not exported)
                    if f"{part.name}" not in location_stats[loc]['parts_produced_consumed_local']:
                        location_stats[loc]['parts_produced_consumed_local'].append(f"{part.name}")

    # Build summary table
    rows = []
    for loc in sorted(problem.locations, key=lambda l: l.name):
        stats = location_stats[loc]
        rows.append({
            'Location': loc.name,
            'Country': loc.country,
            'Cost': f"{loc.factory_activation_cost:.0f}",
            'Parts Produced': len(stats['parts_produced']),
            'Parts List': ', '.join(stats['parts_produced'][:3]) + ('...' if len(stats['parts_produced']) > 3 else ''),
            'Value Added': f"{stats['value_added']:.0f}",
            'Inbound Flows': stats['inbound_flows'],
            'Outbound Flows': stats['outbound_flows'],
            'Imported & Consumed': len(stats['parts_imported_consumed']),
            'Produced & Exported': len(stats['parts_produced_exported']),
            'Produced & Local': len(stats['parts_produced_consumed_local']),
            'Transiting': len(stats['parts_transiting']),
        })

    df = pd.DataFrame(rows)

    # Display summary
    print("\n" + "="*120)
    print("LOCATION STATISTICS SUMMARY")
    print("="*120)
    print(df.to_string(index=False))

    # Detailed breakdown for active locations
    print("\n" + "="*120)
    print("DETAILED BREAKDOWN (Active Locations Only)")
    print("="*120)

    for loc in sorted(problem.locations, key=lambda l: l.name):
        stats = location_stats[loc]

        # Skip inactive locations
        if not stats['parts_produced'] and stats['inbound_flows'] == 0 and stats['outbound_flows'] == 0:
            continue

        print(f"\n📍 {loc.name} ({loc.country}) - Cost: {loc.factory_activation_cost:.0f}")
        print("-" * 100)

        if stats['parts_produced']:
            print(f"  🏭 PRODUCTION ({len(stats['parts_produced'])} parts, {stats['value_added']:.0f} value-added):")
            for part_name in stats['parts_produced']:
                # Find the part object
                part_obj = next((p for p in problem.parts if p.name == part_name), None)
                if part_obj:
                    print(f"     • {part_name} (value: {part_obj.value_added:.0f})")

        if stats['parts_imported_consumed']:
            print(f"  📥 IMPORTED & CONSUMED ({len(stats['parts_imported_consumed'])} flows):")
            for flow_desc in set(stats['parts_imported_consumed']):
                count = stats['parts_imported_consumed'].count(flow_desc)
                print(f"     • {flow_desc} ({count} route{'s' if count > 1 else ''})")

        if stats['parts_produced_exported']:
            print(f"  📤 PRODUCED & EXPORTED ({len(stats['parts_produced_exported'])} flows):")
            for flow_desc in set(stats['parts_produced_exported']):
                count = stats['parts_produced_exported'].count(flow_desc)
                print(f"     • {flow_desc} ({count} route{'s' if count > 1 else ''})")

        if stats['parts_produced_consumed_local']:
            print(f"  🔄 PRODUCED & CONSUMED LOCALLY ({len(stats['parts_produced_consumed_local'])} parts):")
            for part_name in stats['parts_produced_consumed_local']:
                print(f"     • {part_name}")

        if stats['parts_transiting']:
            print(f"  🚚 TRANSITING ({len(stats['parts_transiting'])} parts):")
            for part_name in stats['parts_transiting']:
                print(f"     • {part_name}")

    # Overall statistics
    print("\n" + "="*120)
    print("OVERALL SOLUTION STATISTICS")
    print("="*120)
    total_value_added = sum(stats['value_added'] for stats in location_stats.values())
    active_locations = sum(1 for stats in location_stats.values() if stats['parts_produced'])
    total_transport_routes = len(solution.transport_plan)

    print(f"  Active Locations: {active_locations}/{len(problem.locations)}")
    print(f"  Total Value-Added: {total_value_added:.0f}")
    print(f"  Total Transport Routes: {total_transport_routes}")

    # Country-level statistics
    country_stats = defaultdict(lambda: {'value_added': 0.0, 'locations': 0})
    for loc in problem.locations:
        stats = location_stats[loc]
        if stats['parts_produced']:
            country_stats[loc.country]['value_added'] += stats['value_added']
            country_stats[loc.country]['locations'] += 1

    if country_stats:
        print(f"\n  Country Distribution:")
        for country in sorted(country_stats.keys()):
            stats = country_stats[country]
            print(f"     • {country}: {stats['locations']} location(s), {stats['value_added']:.0f} value-added")

    print("="*120 + "\n")


def analyze_part_reachability(
    problem: 'GlobalLogisticProblem',
    verbose: bool = True
) -> Dict[str, Dict[str, Dict[str, bool]]]:
    """
    Analyze which parts can be feasibly produced at which locations.

    For each part at each location, checks if:
    1. All DIRECT children can reach that location (produced there or transported)
    2. Each child is recursively feasible at where it's produced

    This is different from checking if ALL deep descendants can reach the location -
    it accounts for intermediate subassemblies at different locations.

    Args:
        problem: The GlobalLogisticProblem instance
        verbose: If True, prints detailed analysis

    Returns:
        Dictionary: {part_name: {location_name: {'feasible': bool, 'children': {...}}}}
    """
    import networkx as nx

    # Cache transport graphs per part (expensive to build)
    transport_graphs = {}
    # Cache feasibility results to avoid recomputation
    feasibility_cache = {}

    def get_transport_graph(part: Part):
        """Get or build transport graph for a part."""
        if part not in transport_graphs:
            transport_graphs[part] = problem.build_transport_graph(part)
        return transport_graphs[part]

    def can_part_reach_location(part, target_loc):
        """Check if a part can reach target_loc through multi-hop transport network."""
        # Option 1: Produced at target
        if target_loc in part.produceable_at:
            return True, f"produced at {target_loc.name}", None

        # Option 2: Can be transported there from any valid production site
        G = get_transport_graph(part)
        production_sites = list(part.produceable_at)

        # Check if any production site has a path to target_loc
        shortest_path = None
        shortest_length = float('inf')

        for prod_site in production_sites:
            if nx.has_path(G, prod_site, target_loc):
                path = nx.shortest_path(G, prod_site, target_loc)
                if len(path) < shortest_length:
                    shortest_path = path
                    shortest_length = len(path)

        if shortest_path:
            # Build path description
            if len(shortest_path) == 2:
                path_desc = f"via {shortest_path[0].name} → {shortest_path[-1].name}"
            else:
                path_desc = f"via {' → '.join(loc.name for loc in shortest_path)}"
            return True, path_desc, shortest_path

        return False, "no path found", None

    def is_part_feasible_at_location(part, loc):
        """
        Recursively check if a part can be feasibly produced at a location.

        A part is feasible at location L if:
        1. Each direct child can reach L (either produced at L or transported)
        2. For transported children, they must be feasible at their production location
        """
        # Use cache to avoid recomputation
        cache_key = (part.name, loc.name)
        if cache_key in feasibility_cache:
            return feasibility_cache[cache_key]

        # Leaf parts (no children) are always feasible if produceable
        if not part.child_parts:
            result = (True, {})
            feasibility_cache[cache_key] = result
            return result

        # Check each direct child
        children_info = {}
        all_feasible = True

        for child in part.child_parts:
            reachable, path_desc, path_locs = can_part_reach_location(child, loc)

            if not reachable:
                # Child cannot reach this location at all
                children_info[child.name] = {
                    'reachable': False,
                    'path_desc': path_desc,
                    'feasible_at_source': None
                }
                all_feasible = False
            else:
                # Child can reach - but check if it's feasible at its production location
                if loc in child.produceable_at:
                    # Produced at same location - recursively check feasibility
                    child_feasible, _ = is_part_feasible_at_location(child, loc)
                    children_info[child.name] = {
                        'reachable': True,
                        'path_desc': path_desc,
                        'feasible_at_source': child_feasible
                    }
                    if not child_feasible:
                        all_feasible = False
                else:
                    # Transported from elsewhere - check feasibility at source locations
                    # For simplicity, check if ANY production location is feasible
                    any_source_feasible = False
                    for child_loc in child.produceable_at:
                        child_at_source, _ = is_part_feasible_at_location(child, child_loc)
                        if child_at_source:
                            any_source_feasible = True
                            break

                    children_info[child.name] = {
                        'reachable': True,
                        'path_desc': path_desc,
                        'feasible_at_source': any_source_feasible
                    }
                    if not any_source_feasible:
                        all_feasible = False

        result = (all_feasible, children_info)
        feasibility_cache[cache_key] = result
        return result

    analysis = {}
    issues_found = False

    for part in problem.parts:
        if not part.child_parts:
            continue  # Leaf parts have no dependencies

        part_analysis = {}

        for loc in part.produceable_at:
            feasible, children_info = is_part_feasible_at_location(part, loc)

            if not feasible:
                issues_found = True

            part_analysis[loc.name] = {
                'feasible': feasible,
                'children': children_info
            }

        if part_analysis:
            analysis[part.name] = part_analysis

    # Display analysis
    if verbose:
        logger.info(f"\n{'='*100}")
        logger.info("PART FEASIBILITY ANALYSIS (Recursive BOM Check)")
        logger.info(f"{'='*100}")

        for part_name, loc_data in analysis.items():
            part_obj = next((p for p in problem.parts if p.name == part_name), None)
            if not part_obj:
                continue

            logger.info(f"\n🔧 Part: {part_name}")
            logger.info(f"   Produceable at: {', '.join(loc_data.keys())}")

            for loc_name, loc_info in loc_data.items():
                feasible = loc_info['feasible']
                status = "✓ FEASIBLE" if feasible else "❌ INFEASIBLE"
                logger.info(f"\n   {status} at {loc_name}:")

                children = loc_info['children']
                for child_name, child_info in children.items():
                    reachable = child_info['reachable']
                    path_desc = child_info['path_desc']
                    feasible_at_source = child_info['feasible_at_source']

                    if not reachable:
                        logger.info(f"      ✗ {child_name}: {path_desc}")
                    elif feasible_at_source is None:
                        # Produced locally, no source check needed
                        logger.info(f"      ✓ {child_name}: {path_desc}")
                    elif feasible_at_source:
                        logger.info(f"      ✓ {child_name}: {path_desc} (feasible at source)")
                    else:
                        logger.info(f"      ✗ {child_name}: {path_desc} (NOT feasible at source)")

        logger.info(f"\n{'='*100}")
        if issues_found:
            logger.warning("⚠️  Found locations where parts cannot be feasibly assembled!")
            logger.warning("   These locations should not be used in solutions.")
        else:
            logger.info("✓ All production sites are feasible for BOM assembly")
        logger.info(f"{'='*100}\n")

    return analysis


def setup_location_value_added_constraints(
    problem: GlobalLogisticProblem,
    min_utilization_pct: float = 0.2,
    max_utilization_pct: Optional[float] = None,
    top_n_locations: Optional[int] = None,
) -> None:
    """
    Automatically sets min/max value-added constraints for capable locations.

    This heuristic identifies the most capable production sites and ensures they
    contribute a minimum value-added to the supply chain, preventing underutilization.

    Args:
        problem: The GlobalLogisticProblem instance to configure
        min_utilization_pct: Minimum percentage of potential value-added (0.0-1.0)
                            For example, 0.2 means location must produce at least 20%
                            of its total capability
        max_utilization_pct: Optional maximum percentage of potential value-added
        top_n_locations: If set, only apply to the N most capable locations.
                        If None, applies to all locations with non-zero capability.

    Example:
        # Ensure top 5 factories operate at least at 30% capacity
        setup_location_value_added_constraints(problem, min_utilization_pct=0.3, top_n_locations=5)
    """
    # Compute potential value-added for each location
    location_potential = {}
    for loc in problem.locations:
        # Sum the value-added of all parts that can be produced at this location
        potential = sum(
            part.value_added
            for part in problem.parts
            if loc in part.produceable_at
        )
        if potential > 0:
            location_potential[loc] = potential

    if not location_potential:
        logger.warning("No locations have production capability. No constraints added.")
        return

    # Sort locations by potential (descending)
    sorted_locations = sorted(
        location_potential.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Select which locations to constrain
    if top_n_locations is not None:
        selected = sorted_locations[:top_n_locations]
    else:
        selected = sorted_locations

    # Apply constraints
    num_constrained = 0
    for loc, potential in selected:
        if min_utilization_pct is not None and min_utilization_pct > 0:
            loc.min_value_added = potential * min_utilization_pct
            num_constrained += 1

        if max_utilization_pct is not None and max_utilization_pct > 0:
            loc.max_value_added = potential * max_utilization_pct

    logger.info(f"Applied value-added constraints to {num_constrained} location(s):")
    for loc, potential in selected:
        constraints = []
        if loc.min_value_added is not None:
            constraints.append(f"min={loc.min_value_added:.2f}")
        if loc.max_value_added is not None:
            constraints.append(f"max={loc.max_value_added:.2f}")
        logger.info(f"  {loc.name}: potential={potential:.2f}, {', '.join(constraints)}")
