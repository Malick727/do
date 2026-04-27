"""
Efficient CP-SAT solver for Global Logistic Problem using preprocessed shortest paths.

This solver eliminates flow variables and flow conservation constraints by using
precomputed shortest paths for transport routing. Much faster than the full model.
"""

import logging
from typing import Any, Dict, Optional, Tuple

from ortools.sat.python.cp_model import CpModel, CpSolverSolutionCallback

from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage

from logistic.network_design.problem import Location, Part, GlobalLogisticSolution, GlobalLogisticProblem
from logistic.network_design.preprocessing.shortest_paths import (
    ShortestPathPreprocessor,
    PrecomputedRoute,
)

logger = logging.getLogger(__name__)


class GlobalLogisticCPSatPreprocessedSolver(OrtoolsCpSatSolver):
    """
    CP-SAT solver using precomputed shortest paths.

    Variables:
    - x[part, loc]: binary, whether part is produced at location
    - z[route]: binary, whether precomputed route is used
    - u[loc]: binary, whether location is active
    - v[mean]: binary, whether transport mean is used

    Much simpler than full model - no flow variables or flow conservation!
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

        # CP-SAT Variables
        self.var_x: Dict[Tuple[Part, Location], Any] = {}  # Production decisions
        self.var_z: Dict[PrecomputedRoute, Any] = {}  # Route usage decisions
        self.var_u: Dict[Location, Any] = {}  # Location activation
        self.var_v: Dict[str, Any] = {}  # Transport mean activation

        # Objective variables
        self.obj_vars: Dict[str, Any] = {}

        # Scale float costs to integers
        self.scaling_factor = 1

    def init_model(
        self,
        forbid_transit_at_consumer_sites: bool = False,
        relax_default_sourcing: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Initialize the simplified CP-SAT model.

        Args:
            forbid_transit_at_consumer_sites: If True, forbid transit at consumer-only sites
            relax_default_sourcing: If True, relax sourcing_requirement=1 constraints to allow
                                   0 or more locations (useful when other parts have higher requirements)
        """
        logger.info("="*60)
        logger.info("Initializing Preprocessed CP-SAT Model")
        logger.info("="*60)

        if relax_default_sourcing:
            logger.info("  relax_default_sourcing=True: Allowing flexible sourcing for parts with sourcing_requirement=1")

        self.cp_model = CpModel()

        # ==========================================
        # 1. Variables
        # ==========================================
        logger.info("\nCreating variables...")

        # Production variables: x[part, location]
        for part in self.problem.parts:
            for loc in part.produceable_at:
                self.var_x[part, loc] = self.cp_model.NewBoolVar(f"x_{part.name}_{loc.name}")

        # Location activation variables
        for loc in self.problem.locations:
            self.var_u[loc] = self.cp_model.NewBoolVar(f"u_{loc.name}")

        # Transport mean activation variables
        for mean in self.problem.transport_means:
            self.var_v[mean.name] = self.cp_model.NewBoolVar(f"v_{mean.name}")

        # Route usage variables: z[route]
        # Only create variables for feasible routes
        for route in self.preprocessor.routes.values():
            if route.origin != route.destination and route.is_feasible:
                self.var_z[route] = self.cp_model.NewBoolVar(
                    f"z_{route.part.name}_{route.origin.name}_{route.destination.name}"
                )

        logger.info(f"  Production variables (x): {len(self.var_x)}")
        logger.info(f"  Route variables (z): {len(self.var_z)}")
        logger.info(f"  Location variables (u): {len(self.var_u)}")
        logger.info(f"  Transport mean variables (v): {len(self.var_v)}")

        # ==========================================
        # 2. Constraints
        # ==========================================
        logger.info("\nAdding constraints...")

        # 2a. Production assignment constraints
        for part in self.problem.parts:
            valid_x_vars = [self.var_x[part, loc] for loc in part.produceable_at]
            if valid_x_vars:
                self.cp_model.Add(sum(valid_x_vars) >= part.sourcing_requirement)
                if part.sourcing_requirement == 1 and not relax_default_sourcing:
                    # Enforce exactly one location only if not relaxing
                    self.cp_model.AddExactlyOne(valid_x_vars)

        # 2b. Link production to location activation (using maximum)
        # Location is active if and only if at least one part is produced there
        loc_to_production = {loc: [] for loc in self.problem.locations}
        for (part, loc), x_var in self.var_x.items():
            loc_to_production[loc].append(x_var)

        for loc in self.problem.locations:
            u_var = self.var_u[loc]
            production_vars = loc_to_production[loc]
            if production_vars:
                # u_var = max(production_vars)
                self.cp_model.AddMaxEquality(u_var, production_vars)
            else:
                # No production possible at this location
                self.cp_model.Add(u_var == 0)

        # 2c. Route selection constraints
        # For each part: if produced at origin and consumed at destination,
        # must use the corresponding route
        for part in self.problem.parts:
            for loc in self.problem.locations:
                incoming_routes = [self.var_z[route]
                                   for route in self.var_z
                                   if route.destination == loc and
                                   route.part == part]
                outgoing_routes = [self.var_z[route]
                                   for route in self.var_z
                                   if route.origin == loc and
                                   route.part == part]
                if part.parent_part is not None and (part.parent_part, loc) in self.var_x:
                    consumed = self.var_x[(part.parent_part, loc)]
                else:
                    consumed = 0
                if (part, loc) in self.var_x:
                    produced = self.var_x[(part, loc)]
                else:
                    produced = 0
                if part.parent_part is not None:
                    # NO FLOW CONSERVATION! Parts can be produced, consumed, AND shipped simultaneously
                    # We only enforce logical rules A, B, C (without flow balance):

                    # Rule A (No Starvation): If consumed here → must be produced or received
                    if (part.parent_part, loc) in self.var_x:  # If consumption is possible
                        self.cp_model.Add(produced + sum(incoming_routes) >= 1).OnlyEnforceIf(consumed)

                    # Rule B (No Dead-Ends): If produced/received here → must be consumed or shipped
                    if (part, loc) in self.var_x:  # If production is possible
                        self.cp_model.Add(consumed + sum(outgoing_routes) >= 1).OnlyEnforceIf(produced)
                    for inbound_y in incoming_routes:
                        self.cp_model.Add(consumed + sum(outgoing_routes) >= 1).OnlyEnforceIf(inbound_y)

                    # Rule C (No Spontaneous Generation): If shipped from here → must be produced or received
                    for outbound_y in outgoing_routes:
                        self.cp_model.Add(produced + sum(incoming_routes) >= 1).OnlyEnforceIf(outbound_y)

                if (part, loc) not in self.var_x:
                    continue
                if (part.child_parts is not None
                    and len(part.child_parts) > 0):
                    # To produce this part in the current site, we need
                    # each child part to be available (produced locally or transported in)
                    for child_part in part.child_parts:
                        incoming_and_produced = []
                        if (child_part, loc) in self.var_x:
                            incoming_and_produced.append(self.var_x[(child_part, loc)])
                        incoming_and_produced.extend([self.var_z[route]
                                                      for route in self.var_z
                                                      if route.destination == loc and
                                                      route.part == child_part])
                        # If we produce part at loc, we need at least one source for each child part
                        if incoming_and_produced:
                            self.cp_model.Add(sum(incoming_and_produced) >= 1).OnlyEnforceIf(self.var_x[(part, loc)])


        # 2e. Link routes to transport means (using maximum)
        # Build mapping: transport_mean -> list of routes using it
        mean_to_routes = {mean.name: [] for mean in self.problem.transport_means}
        for route, z_var in self.var_z.items():
            for mean_name in route.transport_means_used:
                if mean_name in mean_to_routes:
                    mean_to_routes[mean_name].append(z_var)

        for mean in self.problem.transport_means:
            v_var = self.var_v.get(mean.name)
            if v_var is not None:
                routes_using_mean = mean_to_routes.get(mean.name, [])
                if routes_using_mean:
                    # Transport mean is active if and only if at least one route using it is active
                    # v_var = max(routes_using_mean)
                    self.cp_model.AddMaxEquality(v_var, routes_using_mean)
                else:
                    # No routes use this transport mean, so it should never be active
                    self.cp_model.Add(v_var == 0)

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
                        has_inbound = self.cp_model.NewBoolVar(f"has_inbound_{part.name}_{loc.name}")

                        # has_inbound = 1 iff sum(inbound_routes) >= 1
                        self.cp_model.Add(sum(inbound_routes) >= 1).OnlyEnforceIf(has_inbound)
                        self.cp_model.Add(sum(inbound_routes) == 0).OnlyEnforceIf(has_inbound.Not())

                        # If part arrives, parent must be produced
                        self.cp_model.AddImplication(has_inbound, self.var_x[(parent, loc)])

        logger.info("✓ Constraints added")
        # ==========================================
        # 3. Objective
        # ==========================================
        logger.info("\nCreating objective...")

        # Factory activation cost (fixed cost per active location)
        max_factory_cost = sum(int(loc.factory_activation_cost * self.scaling_factor) for loc in self.problem.locations)
        factory_cost_var = self.cp_model.NewIntVar(0, max_factory_cost, "factory_activation_cost")
        self.cp_model.Add(factory_cost_var == sum(
            self.var_u[loc] * int(loc.factory_activation_cost * self.scaling_factor)
            for loc in self.problem.locations
        ))

        # Part production cost (variable cost per part-location assignment)
        max_part_cost = sum(
            int(cost * self.scaling_factor)
            for cost in self.problem.part_location_production_costs.values()
        )
        part_cost_var = self.cp_model.NewIntVar(0, max_part_cost, "part_production_cost")
        self.cp_model.Add(part_cost_var == sum(
            self.var_x[(part, loc)] * int(self.problem.part_location_production_costs[(part, loc)] * self.scaling_factor)
            for (part, loc) in self.var_x.keys()
        ))
        # Transport cost (from precomputed routes)
        max_trans_cost = sum(
            int(route.total_cost * self.scaling_factor)
            for route in self.preprocessor.routes.values()
            if route.is_feasible
        )
        trans_cost_var = self.cp_model.NewIntVar(0, max_trans_cost, "transport_cost")
        self.cp_model.Add(trans_cost_var == sum(
            self.var_z[route] * int(route.total_cost * self.scaling_factor)
            for route in self.var_z.keys()
        ))

        # NRC cost
        max_nrc_cost = sum(int(m.nrc_cost * self.scaling_factor) for m in self.problem.transport_means)
        nrc_cost_var = self.cp_model.NewIntVar(0, max_nrc_cost, "nrc_cost")
        self.cp_model.Add(nrc_cost_var == sum(
            self.var_v[m.name] * int(m.nrc_cost * self.scaling_factor)
            for m in self.problem.transport_means
        ))

        # Aggregated production cost (factory + part)
        max_aggregated_prod_cost = max_factory_cost + max_part_cost
        aggregated_prod_cost_var = self.cp_model.NewIntVar(0, max_aggregated_prod_cost, "production_cost")
        self.cp_model.Add(aggregated_prod_cost_var == factory_cost_var + part_cost_var)

        # Country work-share constraints (icWorkshare from the paper)
        if self.problem.country_constraints:
            logger.info(f"Adding {len(self.problem.country_constraints)} country work-share constraints...")
            max_val_added = sum(int(part.value_added * self.scaling_factor) for part in self.problem.parts)

            # Compute value-added per country
            country_value_vars = {}
            for constraint in self.problem.country_constraints:
                terms = [
                    self.var_x[(part, loc)] * int(part.value_added * self.scaling_factor)
                    for (part, loc) in self.var_x.keys()
                    if loc.country == constraint.country
                ]
                if terms:
                    c_var = self.cp_model.NewIntVar(0, max_val_added, f"val_added_{constraint.country}")
                    self.cp_model.Add(c_var == sum(terms))
                    country_value_vars[constraint.country] = c_var

            # Add min/max constraints
            for constraint in self.problem.country_constraints:
                if constraint.country in country_value_vars:
                    c_var = country_value_vars[constraint.country]
                    if constraint.min_value_added is not None:
                        self.cp_model.Add(c_var >= int(constraint.min_value_added * self.scaling_factor))
                    if constraint.max_value_added is not None:
                        self.cp_model.Add(c_var <= int(constraint.max_value_added * self.scaling_factor))

        # Location-level value-added constraints
        locations_with_constraints = [loc for loc in self.problem.locations
                                      if loc.min_value_added is not None or loc.max_value_added is not None]
        if locations_with_constraints:
            logger.info(f"Adding value-added constraints for {len(locations_with_constraints)} locations...")
            max_val_added = sum(int(part.value_added * self.scaling_factor) for part in self.problem.parts)

            for loc in locations_with_constraints:
                # Sum value-added for all parts produced at this location
                terms = [
                    self.var_x[(part, l)] * int(part.value_added * self.scaling_factor)
                    for (part, l) in self.var_x.keys()
                    if l == loc
                ]
                if terms:
                    loc_val_var = self.cp_model.NewIntVar(0, max_val_added, f"val_added_loc_{loc.name}")
                    self.cp_model.Add(loc_val_var == sum(terms))

                    if loc.min_value_added is not None:
                        self.cp_model.Add(loc_val_var >= int(loc.min_value_added * self.scaling_factor))
                    if loc.max_value_added is not None:
                        self.cp_model.Add(loc_val_var <= int(loc.max_value_added * self.scaling_factor))

        # Store objective variables
        self.obj_vars = {
            "factory_activation_cost": factory_cost_var,
            "part_production_cost": part_cost_var,
            "production_cost": aggregated_prod_cost_var,  # Aggregated (factory + part)
            "transport_cost": trans_cost_var,
            "nrc_cost": nrc_cost_var,
        }

        # Set default objective (aggregate)
        objective_list = []
        for obj, weight in zip(self.params_objective_function.objectives,
                               self.params_objective_function.weights):
            if obj in self.obj_vars:
                objective_list.append(weight * self.obj_vars[obj])

        self.cp_model.Minimize(sum(objective_list))

        logger.info("✓ Model initialized")
        logger.info("="*60 + "\n")

    def implements_lexico_api(self) -> bool:
        return True

    def add_lexico_constraint(self, obj: str, value: float):
        return [self.cp_model.Add(self.obj_vars[obj] <= value)]

    def get_lexico_objectives_available(self) -> list[str]:
        return list(self.obj_vars.keys())

    def set_lexico_objective(self, obj: str) -> None:
        self.cp_model.Minimize(self.obj_vars[obj])

    def get_lexico_objective_value(self, obj: str, res: ResultStorage) -> float:
        eval = self.problem.evaluate(res[-1][0])
        return self.scaling_factor * eval[obj]

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> GlobalLogisticSolution:
        """Reconstruct solution from CP-SAT variables."""
        # Log objective values
        for obj in self.obj_vars:
            logger.info(f"{obj}: {cpsolvercb.Value(self.obj_vars[obj])}")

        # Extract production decisions
        part_to_locations = {part: set() for part in self.problem.parts}
        for (part, loc), x_var in self.var_x.items():
            if cpsolvercb.Value(x_var) == 1:
                part_to_locations[part].add(loc)

        # Extract transport plan from active routes
        transport_plan = set()
        for route, z_var in self.var_z.items():
            if cpsolvercb.Value(z_var) == 1:
                # Add all transport options in this route's sequence
                transport_plan.update(route.transport_sequence)

        return GlobalLogisticSolution(
            problem=self.problem,
            part_to_locations=part_to_locations,
            transport_plan=transport_plan
        )