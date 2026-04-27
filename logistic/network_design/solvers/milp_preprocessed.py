"""
Preprocessed MILP solver for Global Logistic Problem.

Uses precomputed shortest paths to reduce the number of variables dramatically.
Follows discrete-optimization pattern with base class and specific solver implementations.
"""

import logging
from typing import Any, Dict, Optional
from collections.abc import Callable

from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.lp_tools import (
    GurobiMilpSolver,
    MilpSolver,
    OrtoolsMathOptMilpSolver,
)

from logistic.network_design.problem import GlobalLogisticSolution, GlobalLogisticProblem
from logistic.network_design.preprocessing.shortest_paths import (
    ShortestPathPreprocessor,
)

try:
    from gurobipy import Var
    from ortools.math_opt.python import mathopt
except ImportError:
    pass

logger = logging.getLogger(__name__)


class _BaseLpGlobalLogisticPreprocessedSolver(MilpSolver):
    """
    Base MILP solver using precomputed shortest paths.

    Variables:
    - x[(part, loc)]: binary, whether part is produced at location
    - z[route]: binary, whether precomputed route is used
    - u[loc]: binary, whether location is active
    - v[mean_name]: binary, whether transport mean is used
    """

    def __init__(
        self,
        problem: GlobalLogisticProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        preprocessor: Optional[ShortestPathPreprocessor] = None,
        **kwargs: Any,
    ):
        super().__init__(
            problem=problem,
            params_objective_function=params_objective_function,
            **kwargs
        )
        self.problem: GlobalLogisticProblem = problem

        # Preprocessor
        if preprocessor is None:
            logger.info("No preprocessor provided, creating one...")
            preprocessor = ShortestPathPreprocessor(problem)
            preprocessor.precompute()
        self.preprocessor = preprocessor

        # Variables
        self.var_x: Dict = {}
        self.var_z: Dict = {}
        self.var_u: Dict = {}
        self.var_v: Dict = {}

    def init_model(
        self,
        forbid_transit_at_consumer_sites: bool = False,
        relax_default_sourcing: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Initialize the preprocessed MILP model.

        Args:
            forbid_transit_at_consumer_sites: If True, forbid transit at consumer-only sites
            relax_default_sourcing: If True, relax sourcing_requirement=1 constraints to allow
                                   0 or more locations (useful when other parts have higher requirements)
        """
        logger.info("="*60)
        logger.info("Initializing Preprocessed MILP Model")
        logger.info("="*60)

        if relax_default_sourcing:
            logger.info("  relax_default_sourcing=True: Allowing flexible sourcing for parts with sourcing_requirement=1")

        # Create model
        self.model = self.create_empty_model("GlobalLogisticMILP_Preprocessed")

        # ==========================================
        # 1. Variables
        # ==========================================
        logger.info("\nCreating variables...")

        # Production variables
        for part in self.problem.parts:
            for loc in part.produceable_at:
                self.var_x[(part, loc)] = self.add_binary_variable(
                    name=f"x_{part.name}_{loc.name}"
                )
            if not relax_default_sourcing:
                nb_source = part.sourcing_requirement
                self.add_linear_constraint(
                    self.construct_linear_sum([self.var_x[(part, loc)]
                                               for loc in part.produceable_at])
                    ==nb_source)

        # Route variables (only feasible routes, excluding same-location)
        for route in self.preprocessor.routes.values():
            if route.is_feasible and route.origin != route.destination:
                self.var_z[route] = self.add_binary_variable(
                    name=f"z_{route.part.name}_{route.origin.name}_{route.destination.name}"
                )

        # Location activation
        for loc in self.problem.locations:
            self.var_u[loc] = self.add_binary_variable(name=f"u_{loc.name}")

        # Transport mean activation
        for mean in self.problem.transport_means:
            self.var_v[mean.name] = self.add_binary_variable(name=f"v_{mean.name}")

        logger.info(f"  Production variables (x): {len(self.var_x)}")
        logger.info(f"  Route variables (z): {len(self.var_z)}")
        logger.info(f"  Location variables (u): {len(self.var_u)}")
        logger.info(f"  Transport mean variables (v): {len(self.var_v)}")

        # Pre-index routes for fast lookup
        logger.info("\nIndexing routes...")
        routes_by_part_dest = {}  # (part, destination) -> [routes]
        routes_by_part_origin = {}  # (part, origin) -> [routes]
        for route in self.var_z.keys():
            key_dest = (route.part, route.destination)
            if key_dest not in routes_by_part_dest:
                routes_by_part_dest[key_dest] = []
            routes_by_part_dest[key_dest].append(route)

            key_origin = (route.part, route.origin)
            if key_origin not in routes_by_part_origin:
                routes_by_part_origin[key_origin] = []
            routes_by_part_origin[key_origin].append(route)

        # ==========================================
        # 2. Constraints
        # ==========================================
        logger.info("\nAdding constraints...")

        # 2a. Sourcing requirements
        for part in self.problem.parts:
            valid_x_vars = [self.var_x[(part, loc)] for loc in part.produceable_at]
            if valid_x_vars:
                self.add_linear_constraint(
                    self.construct_linear_sum(valid_x_vars) >= part.sourcing_requirement
                )

        # 2b. Location activation
        for loc in self.problem.locations:
            production_at_loc = [
                self.var_x[(part, l)] for (part, l) in self.var_x.keys() if l == loc
            ]
            if production_at_loc:
                for x_var in production_at_loc:
                    self.add_linear_constraint(self.var_u[loc] >= x_var)
                self.add_linear_constraint(
                    self.var_u[loc] <= self.construct_linear_sum(production_at_loc)
                )
            else:
                self.add_linear_constraint(self.var_u[loc] == 0)

        # 2c. Assembly requirements
        for part in self.problem.parts:
            if part.child_parts:
                for loc in part.produceable_at:
                    if (part, loc) in self.var_x:
                        for child_part in part.child_parts:
                            child_sources = []
                            if (child_part, loc) in self.var_x:
                                child_sources.append(self.var_x[(child_part, loc)])
                            # Use indexed lookup instead of scanning all routes
                            for route in routes_by_part_dest.get((child_part, loc), []):
                                child_sources.append(self.var_z[route])
                            if child_sources:
                                self.add_linear_constraint(
                                    self.construct_linear_sum(child_sources) >= self.var_x[(part, loc)]
                                )

        # 2d. Flow Rules (A, B, C) - NO FLOW CONSERVATION!
        # Parts can be produced, consumed, AND shipped simultaneously
        logger.info("Adding flow rules (A, B, C)...")

        for part in self.problem.parts:
            for loc in self.problem.locations:
                # Check if production is possible at this location
                production_possible = (part, loc) in self.var_x
                produced = self.var_x[(part, loc)] if production_possible else None

                # Check if consumption is possible at this location
                consumption_possible = part.parent_part and (part.parent_part, loc) in self.var_x
                consumed = self.var_x[(part.parent_part, loc)] if consumption_possible else None

                # Get incoming/outgoing routes
                incoming_routes = routes_by_part_dest.get((part, loc), [])
                outgoing_routes = routes_by_part_origin.get((part, loc), [])

                incoming_vars = [self.var_z[r] for r in incoming_routes]
                outgoing_vars = [self.var_z[r] for r in outgoing_routes]

                # Rule A (No Starvation): If consumed here, must be produced or received
                if consumption_possible:
                    sources = []
                    if production_possible:
                        sources.append(produced)
                    sources.extend(incoming_vars)

                    if sources:
                        self.add_linear_constraint(
                            self.construct_linear_sum(sources) >= consumed
                        )

                # Rule B (No Dead-Ends): If produced/received here, must be consumed or shipped
                if part.parent_part:
                    # If produced locally, must go somewhere
                    if production_possible:
                        sinks = []
                        if consumption_possible:
                            sinks.append(consumed)
                        sinks.extend(outgoing_vars)

                        if sinks:
                            self.add_linear_constraint(
                                self.construct_linear_sum(sinks) >= produced
                            )
                        else:
                            # No sinks available - this is a dead-end location, forbid production
                            self.add_linear_constraint(produced == 0)

                    # If received here, must go somewhere
                    for inbound_route in incoming_routes:
                        sinks = []
                        if consumption_possible:
                            sinks.append(consumed)
                        sinks.extend(outgoing_vars)

                        if sinks:
                            self.add_linear_constraint(
                                self.construct_linear_sum(sinks) >= self.var_z[inbound_route]
                            )
                        else:
                            # No sinks available - forbid receiving at this dead-end location
                            self.add_linear_constraint(self.var_z[inbound_route] == 0)

                # Rule C (No Spontaneous Generation): If shipped from here, must be produced or received
                for outbound_route in outgoing_routes:
                    sources = []
                    if production_possible:
                        sources.append(produced)
                    sources.extend(incoming_vars)

                    if sources:
                        self.add_linear_constraint(
                            self.construct_linear_sum(sources) >= self.var_z[outbound_route]
                        )

        # 2e. Transport mean activation
        mean_to_routes = {mean.name: [] for mean in self.problem.transport_means}
        for route in self.var_z.keys():
            for mean_name in route.transport_means_used:
                if mean_name in mean_to_routes:
                    mean_to_routes[mean_name].append(self.var_z[route])

        for mean in self.problem.transport_means:
            routes_using_mean = mean_to_routes.get(mean.name, [])
            if routes_using_mean:
                for z_var in routes_using_mean:
                    self.add_linear_constraint(self.var_v[mean.name] >= z_var)
                self.add_linear_constraint(
                    self.var_v[mean.name] <= self.construct_linear_sum(routes_using_mean)
                )
            else:
                self.add_linear_constraint(self.var_v[mean.name] == 0)

        # 2f. Heuristic: Forbid Transit at Consumer Sites
        if forbid_transit_at_consumer_sites:
            logger.info("Adding transit-forbidding constraints...")
            for part in self.problem.parts:
                if not part.parent_part:
                    continue

                parent = part.parent_part
                # Find locations where the parent can be produced (potential consumers)
                for loc in parent.produceable_at:
                    if (parent, loc) not in self.var_x:
                        continue

                    inbound_routes = [
                        self.var_z[route] for route in self.var_z
                        if route.part == part and route.destination == loc
                    ]

                    if inbound_routes:
                        # If part arrives at a location where parent can be produced,
                        # then parent MUST be produced there (forbids pure transit)
                        has_inbound = self.add_binary_variable(name=f"has_inbound_{part.name}_{loc.name}")

                        # has_inbound = 1 iff sum(inbound_routes) >= 1
                        M = len(inbound_routes)
                        self.add_linear_constraint(self.construct_linear_sum(inbound_routes) <= M * has_inbound)
                        self.add_linear_constraint(self.construct_linear_sum(inbound_routes) >= has_inbound)

                        # If part arrives, parent must be produced
                        self.add_linear_constraint(self.var_x[(parent, loc)] >= has_inbound)

        # 2g. Country work-share constraints
        if self.problem.country_constraints:
            logger.info(f"Adding {len(self.problem.country_constraints)} country work-share constraints...")
            # Pre-compute value-added terms per country
            country_value_added = {}
            for country in {c.country for c in self.problem.country_constraints}:
                terms = [
                    self.var_x[(part, loc)] * part.value_added
                    for (part, loc) in self.var_x.keys()
                    if loc.country == country
                ]
                if terms:
                    country_value_added[country] = self.construct_linear_sum(terms)

            # Add min/max constraints per country
            for constraint in self.problem.country_constraints:
                if constraint.country in country_value_added:
                    country_val = country_value_added[constraint.country]
                    if constraint.min_value_added is not None:
                        self.add_linear_constraint(country_val >= constraint.min_value_added)
                    if constraint.max_value_added is not None:
                        self.add_linear_constraint(country_val <= constraint.max_value_added)

        # 2h. Location-level value-added constraints
        locations_with_constraints = [loc for loc in self.problem.locations
                                      if loc.min_value_added is not None or loc.max_value_added is not None]
        if locations_with_constraints:
            logger.info(f"Adding value-added constraints for {len(locations_with_constraints)} locations...")
            for loc in locations_with_constraints:
                # Sum value-added for all parts produced at this location
                terms = [
                    self.var_x[(part, l)] * part.value_added
                    for (part, l) in self.var_x.keys()
                    if l == loc
                ]
                if terms:
                    loc_val = self.construct_linear_sum(terms)
                    if loc.min_value_added is not None:
                        self.add_linear_constraint(loc_val >= loc.min_value_added)
                    if loc.max_value_added is not None:
                        self.add_linear_constraint(loc_val <= loc.max_value_added)

        logger.info("✓ Constraints added")

        # ==========================================
        # 3. Objective
        # ==========================================
        logger.info("\nCreating objective...")

        # Factory activation cost (fixed cost per active location)
        factory_cost = self.construct_linear_sum([
            self.var_u[loc] * loc.factory_activation_cost for loc in self.problem.locations
        ])

        # Part production cost (variable cost per part-location assignment)
        part_cost = self.construct_linear_sum([
            self.var_x[(part, loc)] * self.problem.part_location_production_costs[(part, loc)]
            for (part, loc) in self.var_x.keys()
        ])

        # Transport cost (from precomputed routes)
        trans_cost = self.construct_linear_sum([
            self.var_z[route] * route.total_cost for route in self.var_z.keys()
        ])

        # NRC cost
        nrc_cost = self.construct_linear_sum([
            self.var_v[mean.name] * mean.nrc_cost for mean in self.problem.transport_means
        ])

        # Aggregate objective
        obj_terms = []
        weights = dict(zip(
            self.params_objective_function.objectives,
            self.params_objective_function.weights
        ))

        if "factory_activation_cost" in weights:
            obj_terms.append(weights["factory_activation_cost"] * factory_cost)
        if "part_production_cost" in weights:
            obj_terms.append(weights["part_production_cost"] * part_cost)
        if "transport_cost" in weights:
            obj_terms.append(weights["transport_cost"] * trans_cost)
        if "nrc_cost" in weights:
            obj_terms.append(weights["nrc_cost"] * nrc_cost)

        self.set_model_objective(
            self.construct_linear_sum(obj_terms) if obj_terms else 0,
            minimize=True
        )

        logger.info("✓ Model initialized")
        logger.info("="*60 + "\n")

    def retrieve_current_solution(
        self,
        get_var_value_for_current_solution: Callable[[Any], float],
        get_obj_value_for_current_solution: Callable[[], float],
    ) -> GlobalLogisticSolution:
        """Extract solution from model."""
        part_to_locations = {part: set() for part in self.problem.parts}
        for (part, loc), x_var in self.var_x.items():
            if get_var_value_for_current_solution(x_var) > 0.5:
                part_to_locations[part].add(loc)

        transport_plan = set()
        for route, z_var in self.var_z.items():
            if get_var_value_for_current_solution(z_var) > 0.5:
                # Add all transport options in this route's sequence
                transport_plan.update(route.transport_sequence)

        return GlobalLogisticSolution(
            problem=self.problem,
            part_to_locations=part_to_locations,
            transport_plan=transport_plan
        )

    def convert_to_variable_values(
        self, solution: GlobalLogisticSolution
    ) -> Dict[Any, float]:
        """Convert a solution to a mapping between model variables and their values.

        Will be used by set_warm_start().
        """
        var_values = {}

        # Production variables
        for (part, loc), x_var in self.var_x.items():
            var_values[x_var] = 1.0 if loc in solution.part_to_locations.get(part, set()) else 0.0

        # Route variables
        for route, z_var in self.var_z.items():
            # Check if all transport options in this route are used
            all_used = all(opt in solution.transport_plan for opt in route.transport_sequence)
            var_values[z_var] = 1.0 if all_used else 0.0

        # Location activation
        active_locs = {loc for locs in solution.part_to_locations.values() for loc in locs}
        for loc, u_var in self.var_u.items():
            var_values[u_var] = 1.0 if loc in active_locs else 0.0

        # Transport mean activation
        used_means = {opt.mean_name for opt in solution.transport_plan}
        for mean_name, v_var in self.var_v.items():
            var_values[v_var] = 1.0 if mean_name in used_means else 0.0

        return var_values


class GlobalLogisticGurobiPreprocessedSolver(GurobiMilpSolver, _BaseLpGlobalLogisticPreprocessedSolver):
    """Gurobi implementation of Preprocessed Global Logistic MILP solver."""

    def init_model(self, **kwargs: Any) -> None:
        _BaseLpGlobalLogisticPreprocessedSolver.init_model(self, **kwargs)
        self.model.update()

    def convert_to_variable_values(
        self, solution: GlobalLogisticSolution
    ) -> Dict[Var, float]:
        """Convert a solution to a mapping between model variables and their values.

        Will be used by set_warm_start().
        """
        return _BaseLpGlobalLogisticPreprocessedSolver.convert_to_variable_values(self, solution)


class GlobalLogisticMathOptPreprocessedSolver(OrtoolsMathOptMilpSolver, _BaseLpGlobalLogisticPreprocessedSolver):
    """OR-Tools MathOpt implementation of Preprocessed Global Logistic MILP solver."""

    def convert_to_variable_values(
        self, solution: GlobalLogisticSolution
    ) -> Dict[mathopt.Variable, float]:
        """Convert a solution to a mapping between model variables and their values.

        Will be used by set_warm_start() to provide a suitable SolutionHint.variable_values.
        """
        return _BaseLpGlobalLogisticPreprocessedSolver.convert_to_variable_values(self, solution)


# Backward compatibility alias
GlobalLogisticMilpPreprocessedSolver = GlobalLogisticGurobiPreprocessedSolver