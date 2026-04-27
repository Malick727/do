"""
Efficient CP-SAT solver for Quantified Global Logistic Problem using preprocessed shortest paths.

This solver eliminates flow variables by using precomputed shortest paths.
Instead of tracking flow on each edge, we track quantity on each preprocessed route.

Much simpler and faster than the full flow model!
"""

import logging
from typing import Any, Dict, Optional, Tuple

from ortools.sat.python.cp_model import CpModel, CpSolverSolutionCallback

from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver

from logistic.production_design.problem_quantified import (
    QuantifiedGlobalLogisticProblem,
    QuantifiedGlobalLogisticSolution,
    QuantifiedPart,
    QuantifiedLocation,
)
from logistic.network_design.preprocessing.shortest_paths import (
    ShortestPathPreprocessor,
    PrecomputedRoute,
)

logger = logging.getLogger(__name__)


class QuantifiedGlobalLogisticCPSatPreprocessedSolver(OrtoolsCpSatSolver):
    """
    CP-SAT solver using precomputed shortest paths for quantified problem.

    Variables (all integers):
    - prod[part, loc]: quantity of part produced at location
    - route_qty[route]: quantity shipped on this precomputed route
    - active_mean[mean]: binary, whether transport mean is used (for NRC)

    Key simplification: No flow conservation! Routes directly connect production to consumption.

    Constraints:
    - Demand: sum(prod[part, *]) == part.demand
    - Part availability: For each parent at location, ensure children available
      - incoming_route_qty + local_production >= parent_production (for each child)
    - Sourcing: at least K locations active per part
    """

    def __init__(
        self,
        problem: QuantifiedGlobalLogisticProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        preprocessor: Optional[ShortestPathPreprocessor] = None,
        **kwargs: Any,
    ):
        super().__init__(
            problem=problem,
            params_objective_function=params_objective_function,
            **kwargs
        )
        self.problem: QuantifiedGlobalLogisticProblem = problem

        # Preprocessor (reuse from base problem structure)
        if preprocessor is None:
            logger.info("No preprocessor provided, creating one...")
            # The preprocessor works on any problem with the same structure
            # We can create a temporary base problem or just use it directly
            # Since the quantified problem has the same graph structure, preprocessing works!
            preprocessor = ShortestPathPreprocessor(problem)
            preprocessor.precompute()
        self.preprocessor = preprocessor

        # CP-SAT Variables
        self.var_prod: Dict[Tuple[QuantifiedPart, QuantifiedLocation], Any] = {}  # Production quantities
        self.var_route_qty: Dict[PrecomputedRoute, Any] = {}  # Route quantities
        self.var_active_mean: Dict[str, Any] = {}  # Transport mean activation (binary)
        self.var_active_location: Dict[QuantifiedLocation, Any] = {}  # Factory activation (binary)

        # Auxiliary variables
        self.var_prod_binary: Dict[Tuple[QuantifiedPart, QuantifiedLocation], Any] = {}  # Is production > 0?

        # Cost scaling
        self.cost_scale = 100

    def init_model(
        self,
        max_quantity_bound: Optional[int] = None,
        relax_default_sourcing: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Initialize the preprocessed CP-SAT model.

        Args:
            max_quantity_bound: Upper bound on quantities (default: 10 * max demand)
            relax_default_sourcing: If True, allow parts with sourcing_requirement=1 to have 0+ locations
        """
        logger.info("="*80)
        logger.info("Initializing Quantified Global Logistics CP-SAT Model (Preprocessed)")
        logger.info("="*80)

        self.cp_model = CpModel()

        # Determine max quantity bound
        if max_quantity_bound is None:
            max_demand = max((p.demand for p in self.problem.parts if p.demand > 0), default=1)
            max_quantity_bound = 10 * max_demand

        logger.info(f"Max quantity bound: {max_quantity_bound}")
        logger.info(f"Relax default sourcing: {relax_default_sourcing}")

        # ==========================================
        # 1. Create Variables
        # ==========================================
        logger.info("\n[1] Creating variables...")

        # Production quantity variables
        for part in self.problem.parts:
            for loc in part.produceable_at:
                var_name = f"prod_{part.name}_{loc.name}"
                self.var_prod[(part, loc)] = self.cp_model.NewIntVar(0, max_quantity_bound, var_name)

                # Binary indicator
                binary_name = f"prod_bin_{part.name}_{loc.name}"
                self.var_prod_binary[(part, loc)] = self.cp_model.NewBoolVar(binary_name)

                # Link binary to production
                self.cp_model.Add(self.var_prod[(part, loc)] > 0).OnlyEnforceIf(self.var_prod_binary[(part, loc)])
                self.cp_model.Add(self.var_prod[(part, loc)] == 0).OnlyEnforceIf(self.var_prod_binary[(part, loc)].Not())

        logger.info(f"  Created {len(self.var_prod)} production quantity variables")

        # Route quantity variables (only for feasible routes)
        for route_key, route in self.preprocessor.routes.items():
            if route.is_feasible:
                part, origin, dest = route_key
                var_name = f"route_{part.name}_{origin.name}_{dest.name}"
                self.var_route_qty[route] = self.cp_model.NewIntVar(0, max_quantity_bound, var_name)

        logger.info(f"  Created {len(self.var_route_qty)} route quantity variables")

        # Location activation (for factory activation cost)
        for loc in self.problem.locations.values():
            var_name = f"active_loc_{loc.name}"
            self.var_active_location[loc] = self.cp_model.NewBoolVar(var_name)

        logger.info(f"  Created {len(self.var_active_location)} location activation variables")

        # Transport mean activation
        for mean in self.problem.transport_means:
            var_name = f"active_mean_{mean.name}"
            self.var_active_mean[mean.name] = self.cp_model.NewBoolVar(var_name)

        logger.info(f"  Created {len(self.var_active_mean)} transport mean activation variables")

        # ==========================================
        # 2. Demand Constraints
        # ==========================================
        logger.info("\n[2] Adding demand constraints...")

        n_demand_constraints = 0
        for part in self.problem.parts:
            if part.demand > 0:
                prod_vars = [self.var_prod[(part, loc)] for loc in part.produceable_at if (part, loc) in self.var_prod]
                if prod_vars:
                    self.cp_model.Add(sum(prod_vars) == part.demand)
                    n_demand_constraints += 1

        logger.info(f"  Added {n_demand_constraints} demand constraints")

        # ==========================================
        # 3. Part Availability Constraints
        # ==========================================
        logger.info("\n[3] Adding part availability constraints...")

        # For each parent produced at a location, ensure children are available
        # available[child, loc] = local_production + incoming_routes
        # consumed[child, loc] = sum(parent_production for parents using child)
        # Constraint: available >= consumed

        n_availability_constraints = 0
        for part in self.problem.parts:
            for loc in self.problem.locations.values():
                # Calculate how much of this part is consumed at this location
                consumed_by_parents = []
                for parent in self.problem.parts:
                    if part in parent.child_parts:
                        if (parent, loc) in self.var_prod:
                            consumed_by_parents.append(self.var_prod[(parent, loc)])

                if not consumed_by_parents:
                    continue  # Not consumed here

                # Calculate availability
                # 1. Local production
                local_prod_var = self.var_prod.get((part, loc), None)

                # 2. Incoming routes (from any origin to this destination)
                incoming_routes = []
                for route_key, route in self.preprocessor.routes.items():
                    r_part, r_origin, r_dest = route_key
                    if r_part == part and r_dest == loc and route.is_feasible:
                        incoming_routes.append(self.var_route_qty[route])

                # Availability >= Consumption
                available_terms = ([local_prod_var] if local_prod_var is not None else []) + incoming_routes
                consumed_terms = consumed_by_parents

                if available_terms:
                    self.cp_model.Add(sum(available_terms) >= sum(consumed_terms))
                    n_availability_constraints += 1

        logger.info(f"  Added {n_availability_constraints} part availability constraints")

        # ==========================================
        # 4. Route Feasibility (link routes to production)
        # ==========================================
        logger.info("\n[4] Linking routes to production...")

        # A route can only be used if the part is produced at the origin
        n_route_production_links = 0
        for route in self.var_route_qty.keys():
            if route.origin == route.destination:
                continue  # No transport needed

            # route_qty <= production at origin
            if (route.part, route.origin) in self.var_prod:
                self.cp_model.Add(self.var_route_qty[route] <= self.var_prod[(route.part, route.origin)])
                n_route_production_links += 1

        logger.info(f"  Added {n_route_production_links} route-production links")

        # ==========================================
        # 5. Production Balance (outgoing routes)
        # ==========================================
        logger.info("\n[5] Adding production balance constraints...")

        # Total outgoing routes from a location <= production at that location
        # (already handled by route feasibility, but we can add explicit constraint for clarity)
        n_balance_constraints = 0
        for part in self.problem.parts:
            for loc in part.produceable_at:
                if (part, loc) not in self.var_prod:
                    continue

                # Find all outgoing routes from this (part, location)
                outgoing_routes = []
                for route_key, route in self.preprocessor.routes.items():
                    r_part, r_origin, r_dest = route_key
                    if r_part == part and r_origin == loc and route.is_feasible:
                        if r_origin != r_dest:  # Exclude self-loops
                            outgoing_routes.append(self.var_route_qty[route])

                # Also count consumption at this location
                consumed_here = []
                for parent in self.problem.parts:
                    if part in parent.child_parts:
                        if (parent, loc) in self.var_prod:
                            consumed_here.append(self.var_prod[(parent, loc)])

                # NO DEAD-ENDS: production must be fully used
                # For intermediate parts: production == outgoing + consumed
                # For final products with demand: production >= outgoing + consumed (rest satisfies external demand)
                if part.demand > 0:
                    # Final product: can produce more than local usage (satisfies external demand)
                    self.cp_model.Add(
                        self.var_prod[(part, loc)] >= sum(outgoing_routes + consumed_here)
                    )
                else:
                    # Intermediate part: must be fully used (no dead-ends)
                    self.cp_model.Add(
                        self.var_prod[(part, loc)] == sum(outgoing_routes + consumed_here)
                    )
                n_balance_constraints += 1

        logger.info(f"  Added {n_balance_constraints} production balance constraints")

        # ==========================================
        # 6. Sourcing Requirements
        # ==========================================
        logger.info("\n[6] Adding sourcing requirement constraints...")

        n_sourcing_constraints = 0
        for part in self.problem.parts:
            if part.sourcing_requirement > 1 or (part.sourcing_requirement == 1 and not relax_default_sourcing):
                active_locs = [
                    self.var_prod_binary[(part, loc)] for loc in part.produceable_at
                    if (part, loc) in self.var_prod_binary
                ]

                if active_locs:
                    if part.sourcing_requirement == 1 and not relax_default_sourcing:
                        self.cp_model.Add(sum(active_locs) == 1)
                    else:
                        self.cp_model.Add(sum(active_locs) >= part.sourcing_requirement)
                    n_sourcing_constraints += 1

        logger.info(f"  Added {n_sourcing_constraints} sourcing requirement constraints")

        # ==========================================
        # 7. Location Activation (for factory cost)
        # ==========================================
        logger.info("\n[7] Linking location activation to production...")

        # Location is active if and only if any part is produced there
        # Use the binary production indicators (var_prod_binary) that track if production > 0
        n_location_links = 0
        for loc in self.problem.locations.values():
            # Find all binary production indicators for this location
            prod_binary_at_loc = [
                self.var_prod_binary[(part, l)]
                for (part, l) in self.var_prod_binary.keys()
                if l == loc
            ]

            if prod_binary_at_loc:
                # active_location = max(prod_binary[part, loc] for all parts)
                # This means: location is active iff at least one part has production > 0
                self.cp_model.AddMaxEquality(self.var_active_location[loc], prod_binary_at_loc)
                n_location_links += 1
            else:
                # No production possible at this location
                self.cp_model.Add(self.var_active_location[loc] == 0)

        logger.info(f"  Linked {n_location_links} locations to production")

        # ==========================================
        # 8. Transport Mean Activation (for NRC)
        # ==========================================
        logger.info("\n[8] Linking transport mean activation...")

        for mean in self.problem.transport_means:
            # Mean is active if any route using this mean has quantity > 0
            routes_using_mean = []
            for route in self.var_route_qty.keys():
                if mean.name in route.transport_means_used:
                    routes_using_mean.append(self.var_route_qty[route])

            if routes_using_mean:
                # If any route > 0, then active_mean = 1
                for route_var in routes_using_mean:
                    self.cp_model.Add(route_var <= max_quantity_bound * self.var_active_mean[mean.name])

        logger.info(f"  Linked {len(self.var_active_mean)} transport means")

        # ==========================================
        # 9. Country Work-Share Constraints
        # ==========================================
        if self.problem.country_constraints:
            logger.info(f"\n[9] Adding {len(self.problem.country_constraints)} country work-share constraints...")

            # Max possible value-added (quantity * unit_value_added)
            max_val_added = max_quantity_bound * sum(
                int(part.unit_value_added * self.cost_scale)
                for part in self.problem.parts
            )

            # Compute value-added per country
            country_value_vars = {}
            for constraint in self.problem.country_constraints:
                # Sum: quantity * unit_value_added for all parts produced in this country
                terms = [
                    self.var_prod[(part, loc)] * int(part.unit_value_added * self.cost_scale)
                    for (part, loc) in self.var_prod.keys()
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
                        min_scaled = int(constraint.min_value_added * self.cost_scale)
                        self.cp_model.Add(c_var >= min_scaled)
                        logger.info(f"  {constraint.country}: min value-added >= €{constraint.min_value_added:,.0f}")
                    if constraint.max_value_added is not None:
                        max_scaled = int(constraint.max_value_added * self.cost_scale)
                        self.cp_model.Add(c_var <= max_scaled)
                        logger.info(f"  {constraint.country}: max value-added <= €{constraint.max_value_added:,.0f}")

        # Location-level value-added constraints
        locations_with_constraints = [
            loc for loc in self.problem.locations.values()
            if loc.min_value_added is not None or loc.max_value_added is not None
        ]
        if locations_with_constraints:
            logger.info(f"\n[9b] Adding value-added constraints for {len(locations_with_constraints)} locations...")

            max_val_added = max_quantity_bound * sum(
                int(part.unit_value_added * self.cost_scale)
                for part in self.problem.parts
            )

            for loc in locations_with_constraints:
                # Sum value-added for all parts produced at this location
                terms = [
                    self.var_prod[(part, l)] * int(part.unit_value_added * self.cost_scale)
                    for (part, l) in self.var_prod.keys()
                    if l == loc
                ]
                if terms:
                    loc_val_var = self.cp_model.NewIntVar(0, max_val_added, f"val_added_loc_{loc.name}")
                    self.cp_model.Add(loc_val_var == sum(terms))

                    if loc.min_value_added is not None:
                        min_scaled = int(loc.min_value_added * self.cost_scale)
                        self.cp_model.Add(loc_val_var >= min_scaled)
                        logger.info(f"  {loc.name}: min value-added >= €{loc.min_value_added:,.0f}")
                    if loc.max_value_added is not None:
                        max_scaled = int(loc.max_value_added * self.cost_scale)
                        self.cp_model.Add(loc_val_var <= max_scaled)
                        logger.info(f"  {loc.name}: max value-added <= €{loc.max_value_added:,.0f}")

        # ==========================================
        # 10. Objective Function
        # ==========================================
        logger.info("\n[10] Building objective function...")

        # Factory activation cost
        factory_terms = []
        for loc in self.problem.locations.values():
            if loc.factory_activation_cost > 0:
                scaled_cost = int(loc.factory_activation_cost * self.cost_scale)
                factory_terms.append(scaled_cost * self.var_active_location[loc])

        factory_cost_var = self.cp_model.NewIntVar(
            0,
            sum(int(loc.factory_activation_cost * self.cost_scale)
                for loc in self.problem.locations.values()),
            "factory_activation_cost"
        )
        if factory_terms:
            self.cp_model.Add(factory_cost_var == sum(factory_terms))
        else:
            self.cp_model.Add(factory_cost_var == 0)

        # Production cost (per-unit)
        production_terms = []
        for (part, loc), var in self.var_prod.items():
            unit_cost = loc.unit_production_cost.get(part, 0.0)
            scaled_cost = int(unit_cost * self.cost_scale)
            if scaled_cost > 0:
                production_terms.append(scaled_cost * var)

        production_cost_var = self.cp_model.NewIntVar(
            0,
            sum(max_quantity_bound * int(loc.unit_production_cost.get(part, 0) * self.cost_scale)
                for (part, loc) in self.var_prod.keys()),
            "production_cost"
        )
        if production_terms:
            self.cp_model.Add(production_cost_var == sum(production_terms))
        else:
            self.cp_model.Add(production_cost_var == 0)

        # Transport cost (using precomputed route costs)
        transport_terms = []
        for route, var in self.var_route_qty.items():
            # route.total_cost is the total cost of the route (sum of all hops)
            scaled_cost = int(route.total_cost * self.cost_scale)
            if scaled_cost > 0:
                transport_terms.append(scaled_cost * var)

        transport_cost_var = self.cp_model.NewIntVar(
            0,
            sum(max_quantity_bound * int(route.total_cost * self.cost_scale)
                for route in self.var_route_qty.keys()),
            "transport_cost"
        )
        if transport_terms:
            self.cp_model.Add(transport_cost_var == sum(transport_terms))
        else:
            self.cp_model.Add(transport_cost_var == 0)

        # NRC cost
        nrc_terms = []
        for mean in self.problem.transport_means:
            nrc_cost = mean.nrc_cost
            scaled_cost = int(nrc_cost * self.cost_scale)
            if scaled_cost > 0:
                nrc_terms.append(scaled_cost * self.var_active_mean[mean.name])

        nrc_cost_var = self.cp_model.NewIntVar(
            0,
            sum(int(mean.nrc_cost * self.cost_scale) for mean in self.problem.transport_means),
            "nrc_cost"
        )
        if nrc_terms:
            self.cp_model.Add(nrc_cost_var == sum(nrc_terms))
        else:
            self.cp_model.Add(nrc_cost_var == 0)

        # Store objective variables
        self.obj_vars = {
            "factory_activation_cost": factory_cost_var,
            "production_cost": production_cost_var,
            "transport_cost": transport_cost_var,
            "nrc_cost": nrc_cost_var,
        }

        # Build objective using ParamsObjectiveFunction
        objective_list = []
        for obj, weight in zip(self.params_objective_function.objectives,
                              self.params_objective_function.weights):
            if obj in self.obj_vars:
                objective_list.append(weight * self.obj_vars[obj])
                logger.info(f"  {obj}: weight={weight}")

        if objective_list:
            self.cp_model.Minimize(sum(objective_list))
        else:
            # Fallback: minimize all costs equally
            logger.warning("  No matching objectives found, using default (all costs)")
            total_cost = production_cost_var + transport_cost_var + nrc_cost_var
            self.cp_model.Minimize(total_cost)

        logger.info(f"  Factory activation terms: {len(factory_terms)}")
        logger.info(f"  Production (per-unit) terms: {len(production_terms)}")
        logger.info(f"  Transport terms: {len(transport_terms)}")
        logger.info(f"  NRC terms: {len(nrc_terms)}")
        logger.info(f"  Cost scaling factor: {self.cost_scale}")

        logger.info("\n" + "="*80)
        logger.info("Model initialized successfully")
        logger.info("="*80)

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> QuantifiedGlobalLogisticSolution:
        """Extract solution from CP-SAT solver."""
        production_quantities = {}
        flow_quantities = {}

        # Extract production quantities
        for (part, loc), var in self.var_prod.items():
            qty = cpsolvercb.Value(var)
            if qty > 0:
                production_quantities[(part, loc)] = qty

        # Reconstruct flow_quantities from route quantities
        # For each active route, decompose it into individual transport options
        for route, var in self.var_route_qty.items():
            qty = cpsolvercb.Value(var)
            if qty > 0:
                # Decompose route into individual hops
                for transport_option in route.transport_sequence:
                    # Accumulate flow on each transport option
                    current_qty = flow_quantities.get(transport_option, 0)
                    flow_quantities[transport_option] = current_qty + qty

        return QuantifiedGlobalLogisticSolution(
            problem=self.problem,
            production_quantities=production_quantities,
            flow_quantities=flow_quantities
        )