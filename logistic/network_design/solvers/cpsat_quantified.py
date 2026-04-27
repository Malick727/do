"""
CP-SAT solver for Quantified Global Logistic Problem (full flow model).

This solver models the problem with integer quantities:
- Production quantities: how much of each part at each location
- Flow quantities: how much transported on each route

Constraints:
- Demand satisfaction
- Flow conservation at each location
- BOM consumption (N parents need N children)
- Non-negativity
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
    QuantifiedTransportOption,
)

logger = logging.getLogger(__name__)


class QuantifiedGlobalLogisticCPSatSolver(OrtoolsCpSatSolver):
    """
    CP-SAT solver for quantified global logistics problem.

    Variables (all integers):
    - prod[part, loc]: quantity of part produced at location
    - flow[transport_option]: quantity shipped on this route
    - active_mean[mean]: binary, whether transport mean is used (for NRC)

    Constraints:
    - Demand: sum(prod[part, *]) == part.demand for final products
    - Flow conservation: incoming + produced == outgoing + consumed
    - BOM: consumed[child] == sum(prod[parent] for parents using child)
    - Sourcing: at least K locations active per part (if sourcing_requirement > 1)
    """

    def __init__(
        self,
        problem: QuantifiedGlobalLogisticProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(
            problem=problem,
            params_objective_function=params_objective_function,
            **kwargs
        )
        self.problem: QuantifiedGlobalLogisticProblem = problem

        # CP-SAT Variables
        self.var_prod: Dict[Tuple[QuantifiedPart, QuantifiedLocation], Any] = {}  # Production quantities
        self.var_flow: Dict[QuantifiedTransportOption, Any] = {}  # Flow quantities
        self.var_active_mean: Dict[str, Any] = {}  # Transport mean activation (binary)
        self.var_active_location: Dict[QuantifiedLocation, Any] = {}  # Factory activation (binary)

        # Auxiliary variables
        self.var_prod_binary: Dict[Tuple[QuantifiedPart, QuantifiedLocation], Any] = {}  # Is production > 0?

        # Objective components
        self.obj_vars: Dict[str, Any] = {}

        # Cost scaling for integer CP-SAT (costs are floats, need to scale)
        self.cost_scale = 100  # Scale costs by 100 to preserve 2 decimal places

    def init_model(
        self,
        max_quantity_bound: Optional[int] = None,
        relax_default_sourcing: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Initialize the CP-SAT model.

        Args:
            max_quantity_bound: Upper bound on production quantities (default: 10 * max demand)
            relax_default_sourcing: If True, allow parts with sourcing_requirement=1 to have 0+ locations
        """
        logger.info("="*70)
        logger.info("Initializing Quantified Global Logistics CP-SAT Model (Full Flow)")
        logger.info("="*70)

        self.cp_model = CpModel()

        # Determine max quantity bound
        if max_quantity_bound is None:
            max_demand = max((p.demand for p in self.problem.parts if p.demand > 0), default=1)
            max_quantity_bound = 10 * max_demand  # Reasonable upper bound

        logger.info(f"Max quantity bound: {max_quantity_bound}")

        # ==========================================
        # 1. Create Variables
        # ==========================================
        logger.info("\n[1] Creating variables...")

        # Production quantity variables
        for part in self.problem.parts:
            for loc in part.produceable_at:
                var_name = f"prod_{part.name}_{loc.name}"
                self.var_prod[(part, loc)] = self.cp_model.NewIntVar(0, max_quantity_bound, var_name)

                # Binary indicator: is production > 0?
                binary_name = f"prod_bin_{part.name}_{loc.name}"
                self.var_prod_binary[(part, loc)] = self.cp_model.NewBoolVar(binary_name)

                # Link binary to production
                # prod > 0 <==> prod_binary == 1
                self.cp_model.Add(self.var_prod[(part, loc)] > 0).OnlyEnforceIf(self.var_prod_binary[(part, loc)])
                self.cp_model.Add(self.var_prod[(part, loc)] == 0).OnlyEnforceIf(self.var_prod_binary[(part, loc)].Not())

        logger.info(f"  Created {len(self.var_prod)} production quantity variables")

        # Flow quantity variables
        for transport_option in self.problem.transport_options:
            var_name = f"flow_{transport_option.part.name}_{transport_option.origin.name}_{transport_option.destination.name}"
            self.var_flow[transport_option] = self.cp_model.NewIntVar(0, max_quantity_bound, var_name)

        logger.info(f"  Created {len(self.var_flow)} flow quantity variables")

        # Location activation (for factory activation cost)
        for loc in self.problem.locations.values():
            var_name = f"active_loc_{loc.name}"
            self.var_active_location[loc] = self.cp_model.NewBoolVar(var_name)

        logger.info(f"  Created {len(self.var_active_location)} location activation variables")

        # Transport mean activation (binary for NRC cost)
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
                # Total production must equal demand
                prod_vars = [self.var_prod[(part, loc)] for loc in part.produceable_at if (part, loc) in self.var_prod]
                if prod_vars:
                    self.cp_model.Add(sum(prod_vars) == part.demand)
                    n_demand_constraints += 1

        logger.info(f"  Added {n_demand_constraints} demand constraints")

        # ==========================================
        # 3. Flow Conservation Constraints
        # ==========================================
        logger.info("\n[3] Adding flow conservation constraints...")

        n_flow_constraints = 0
        for part in self.problem.parts:
            # Skip final products (they satisfy external demand, not flow conservation)
            if part.demand > 0 and part.parent_part is None:
                continue

            for loc in self.problem.locations.values():
                # Incoming flow
                incoming_flows = [
                    self.var_flow[t_opt] for t_opt in self.problem.transport_options
                    if t_opt.part == part and t_opt.destination == loc
                ]

                # Production at this location
                prod_at_loc = self.var_prod.get((part, loc), 0)

                # Outgoing flow
                outgoing_flows = [
                    self.var_flow[t_opt] for t_opt in self.problem.transport_options
                    if t_opt.part == part and t_opt.origin == loc
                ]

                # Consumption by parents assembled at this location
                consumed = []
                for parent in self.problem.parts:
                    if part in parent.child_parts:
                        if (parent, loc) in self.var_prod:
                            consumed.append(self.var_prod[(parent, loc)])

                # Flow conservation: incoming + produced == outgoing + consumed
                # For final products with demand: >= (excess satisfies external demand)
                # For intermediate parts: == (no dead-ends allowed)
                has_activity = (
                    incoming_flows or
                    (part, loc) in self.var_prod or  # Production possible here
                    outgoing_flows or
                    consumed
                )

                if has_activity:
                    lhs_terms = incoming_flows + ([self.var_prod[(part, loc)]] if (part, loc) in self.var_prod else [])
                    rhs_terms = outgoing_flows + consumed

                    if lhs_terms or rhs_terms:
                        if part.demand > 0:
                            # Final product: allow surplus for external demand
                            self.cp_model.Add(sum(lhs_terms) >= sum(rhs_terms))
                        else:
                            # Intermediate part: strict equality (no dead-ends)
                            self.cp_model.Add(sum(lhs_terms) == sum(rhs_terms))
                        n_flow_constraints += 1

        logger.info(f"  Added {n_flow_constraints} flow conservation constraints")

        # ==========================================
        # 4. Sourcing Requirements
        # ==========================================
        logger.info("\n[4] Adding sourcing requirement constraints...")

        n_sourcing_constraints = 0
        for part in self.problem.parts:
            if part.sourcing_requirement > 1 or (part.sourcing_requirement == 1 and not relax_default_sourcing):
                # Count active locations (where production > 0)
                active_locs = [
                    self.var_prod_binary[(part, loc)] for loc in part.produceable_at
                    if (part, loc) in self.var_prod_binary
                ]

                if active_locs:
                    if part.sourcing_requirement == 1 and not relax_default_sourcing:
                        # Exactly 1 location
                        self.cp_model.Add(sum(active_locs) == 1)
                    else:
                        # At least K locations
                        self.cp_model.Add(sum(active_locs) >= part.sourcing_requirement)
                    n_sourcing_constraints += 1

        logger.info(f"  Added {n_sourcing_constraints} sourcing requirement constraints")

        # ==========================================
        # 5. Location Activation (for factory cost)
        # ==========================================
        logger.info("\n[5] Linking location activation to production...")

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
        # 6. Transport Mean Activation (for NRC)
        # ==========================================
        logger.info("\n[6] Linking transport mean activation...")

        for mean in self.problem.transport_means:
            # Mean is active if any flow using this mean > 0
            flows_using_mean = [
                self.var_flow[t_opt] for t_opt in self.problem.transport_options
                if t_opt.mean.name == mean.name
            ]

            if flows_using_mean:
                # active_mean == 1 if any flow > 0
                # Use big-M constraint
                for flow_var in flows_using_mean:
                    # If flow > 0, then active_mean must be 1
                    self.cp_model.Add(flow_var <= max_quantity_bound * self.var_active_mean[mean.name])

        logger.info(f"  Linked {len(self.var_active_mean)} transport means")

        # ==========================================
        # 7. Country Work-Share Constraints
        # ==========================================
        if self.problem.country_constraints:
            logger.info(f"\n[7] Adding {len(self.problem.country_constraints)} country work-share constraints...")

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
            logger.info(f"\n[7b] Adding value-added constraints for {len(locations_with_constraints)} locations...")

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
        # 8. Objective Function
        # ==========================================
        logger.info("\n[8] Building objective function...")

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

        # Scale costs to integers and create objective variables
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

        transport_terms = []
        for t_opt, var in self.var_flow.items():
            unit_cost = t_opt.total_cost  # Already per-unit cost in quantified version
            scaled_cost = int(unit_cost * self.cost_scale)
            if scaled_cost > 0:
                transport_terms.append(scaled_cost * var)

        transport_cost_var = self.cp_model.NewIntVar(
            0,
            sum(max_quantity_bound * int(t_opt.total_cost * self.cost_scale)
                for t_opt in self.var_flow.keys()),
            "transport_cost"
        )
        if transport_terms:
            self.cp_model.Add(transport_cost_var == sum(transport_terms))
        else:
            self.cp_model.Add(transport_cost_var == 0)

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

        logger.info("\n" + "="*70)
        logger.info("Model initialized successfully")
        logger.info("="*70)

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> QuantifiedGlobalLogisticSolution:
        """Extract solution from CP-SAT solver."""
        production_quantities = {}
        flow_quantities = {}

        # Extract production quantities
        for (part, loc), var in self.var_prod.items():
            qty = cpsolvercb.Value(var)
            if qty > 0:
                production_quantities[(part, loc)] = qty

        # Extract flow quantities
        for t_opt, var in self.var_flow.items():
            qty = cpsolvercb.Value(var)
            if qty > 0:
                flow_quantities[t_opt] = qty

        return QuantifiedGlobalLogisticSolution(
            problem=self.problem,
            production_quantities=production_quantities,
            flow_quantities=flow_quantities
        )
