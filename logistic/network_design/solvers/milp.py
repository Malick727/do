"""
MILP solver for Global Logistic Problem (full model with flow variables).

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

try:
    from gurobipy import Var
    from ortools.math_opt.python import mathopt
except ImportError:
    pass

logger = logging.getLogger(__name__)


class _BaseLpGlobalLogisticSolver(MilpSolver):
    """
    Base MILP solver for Global Logistic Problem using full flow model.

    Variables:
    - x[(part, loc)]: binary, whether part is produced at location
    - y[transport_option]: binary, whether transport option is used
    - u[loc]: binary, whether location is active
    - v[mean_name]: binary, whether transport mean is used
    """

    def __init__(
        self,
        problem: GlobalLogisticProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs: Any,
    ):
        super().__init__(
            problem=problem,
            params_objective_function=params_objective_function,
            **kwargs
        )
        self.problem: GlobalLogisticProblem = problem
        self.var_x: Dict = {}
        self.var_y: Dict = {}
        self.var_u: Dict = {}
        self.var_v: Dict = {}

    def init_model(
        self,
        forbid_transit_at_consumer_sites: bool = False,
        relax_default_sourcing: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Initialize the MILP model using generic interface.

        Args:
            forbid_transit_at_consumer_sites: If True, forbid transit at consumer-only sites
            relax_default_sourcing: If True, relax sourcing_requirement=1 constraints to allow
                                   0 or more locations (useful when other parts have higher requirements)
        """
        logger.info("="*60)
        logger.info("Initializing MILP Model (Full Flow)")
        logger.info("="*60)

        if relax_default_sourcing:
            logger.info("  relax_default_sourcing=True: Allowing flexible sourcing for parts with sourcing_requirement=1")

        # Create model
        self.model = self.create_empty_model("GlobalLogisticMILP")

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

        # Transport variables
        for t_opt in self.problem.transport_options:
            self.var_y[t_opt] = self.add_binary_variable(
                name=f"y_{t_opt.part.name}_{t_opt.origin.name}_{t_opt.destination.name}"
            )

        # Location activation
        for loc in self.problem.locations:
            self.var_u[loc] = self.add_binary_variable(name=f"u_{loc.name}")

        # Transport mean activation
        for mean in self.problem.transport_means:
            self.var_v[mean.name] = self.add_binary_variable(name=f"v_{mean.name}")

        logger.info(f"  Production variables (x): {len(self.var_x)}")
        logger.info(f"  Transport variables (y): {len(self.var_y)}")
        logger.info(f"  Location variables (u): {len(self.var_u)}")
        logger.info(f"  Transport mean variables (v): {len(self.var_v)}")

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

        # 2b. Location activation (max constraints)
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

        # 2c. Transport mean activation (max constraints)
        for mean in self.problem.transport_means:
            routes_using_mean = [
                self.var_y[t] for t in self.problem.transport_options
                if t.mean.name == mean.name
            ]
            if routes_using_mean:
                for y_var in routes_using_mean:
                    self.add_linear_constraint(self.var_v[mean.name] >= y_var)
                self.add_linear_constraint(
                    self.var_v[mean.name] <= self.construct_linear_sum(routes_using_mean)
                )
            else:
                self.add_linear_constraint(self.var_v[mean.name] == 0)

        # 2d. Flow Rules (A, B, C) - NO FLOW CONSERVATION!
        # Parts can be produced, consumed, AND shipped simultaneously
        logger.info("Adding flow rules (A, B, C)...")
        for part in self.problem.parts:
            for loc in self.problem.locations:
                produced = self.var_x.get((part, loc), 0)
                consumed = self.var_x.get((part.parent_part, loc), 0) if part.parent_part else 0

                inbound_vars = [
                    self.var_y[t] for t in self.problem.transport_options
                    if t.part == part and t.destination == loc
                ]
                outbound_vars = [
                    self.var_y[t] for t in self.problem.transport_options
                    if t.part == part and t.origin == loc
                ]

                inbound = self.construct_linear_sum(inbound_vars) if inbound_vars else 0
                outbound = self.construct_linear_sum(outbound_vars) if outbound_vars else 0

                # Rule A (No Starvation): If consumed here, must be produced or received
                if part.parent_part and (part.parent_part, loc) in self.var_x:
                    self.add_linear_constraint(produced + inbound >= consumed)

                # Rule B (No Dead-Ends): If produced/received here, must be consumed or shipped
                # Note: When consumed=0 and outbound=0, this forces produced=0 (prevents dead-ends)
                if part.parent_part:
                    if (part, loc) in self.var_x:
                        self.add_linear_constraint(consumed + outbound >= produced)
                    for y_var in inbound_vars:
                        self.add_linear_constraint(consumed + outbound >= y_var)

                # Rule C (No Spontaneous Generation): If shipped from here, must be produced or received
                for y_var in outbound_vars:
                    self.add_linear_constraint(produced + inbound >= y_var)

                # Assembly requirements
                if (part, loc) in self.var_x and part.child_parts:
                    for child_part in part.child_parts:
                        child_sources = []
                        if (child_part, loc) in self.var_x:
                            child_sources.append(self.var_x[(child_part, loc)])
                        child_sources.extend([
                            self.var_y[t] for t in self.problem.transport_options
                            if t.part == child_part and t.destination == loc
                        ])
                        if child_sources:
                            self.add_linear_constraint(
                                self.construct_linear_sum(child_sources) >= self.var_x[(part, loc)]
                            )

        # 2e. Heuristic: Forbid Transit at Consumer Sites
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

                    inbound_vars = [
                        self.var_y[t] for t in self.problem.transport_options
                        if t.part == part and t.destination == loc
                    ]

                    if inbound_vars:
                        # If part arrives at a location where parent can be produced,
                        # then parent MUST be produced there (forbids pure transit)
                        has_inbound = self.add_binary_variable(name=f"has_inbound_{part.name}_{loc.name}")

                        # has_inbound = 1 iff sum(inbound_vars) >= 1
                        M = len(inbound_vars)
                        self.add_linear_constraint(self.construct_linear_sum(inbound_vars) <= M * has_inbound)
                        self.add_linear_constraint(self.construct_linear_sum(inbound_vars) >= has_inbound)

                        # If part arrives, parent must be produced
                        self.add_linear_constraint(self.var_x[(parent, loc)] >= has_inbound)

        # 2f. Country work-share constraints (icWorkshare from the paper)
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

        # 2g. Location-level value-added constraints
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

        # Transport cost
        trans_cost = self.construct_linear_sum([
            self.var_y[t] * t.total_cost for t in self.problem.transport_options
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
        for t_opt, y_var in self.var_y.items():
            if get_var_value_for_current_solution(y_var) > 0.5:
                transport_plan.add(t_opt)

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

        # Transport variables
        for t_opt, y_var in self.var_y.items():
            var_values[y_var] = 1.0 if t_opt in solution.transport_plan else 0.0

        # Location activation
        active_locs = {loc for locs in solution.part_to_locations.values() for loc in locs}
        for loc, u_var in self.var_u.items():
            var_values[u_var] = 1.0 if loc in active_locs else 0.0

        # Transport mean activation
        used_means = {opt.mean_name for opt in solution.transport_plan}
        for mean_name, v_var in self.var_v.items():
            var_values[v_var] = 1.0 if mean_name in used_means else 0.0

        return var_values


class GlobalLogisticGurobiSolver(GurobiMilpSolver, _BaseLpGlobalLogisticSolver):
    """Gurobi implementation of Global Logistic MILP solver."""

    def init_model(self, **kwargs: Any) -> None:
        _BaseLpGlobalLogisticSolver.init_model(self, **kwargs)
        self.model.update()

    def convert_to_variable_values(
        self, solution: GlobalLogisticSolution
    ) -> Dict[Var, float]:
        """Convert a solution to a mapping between model variables and their values.

        Will be used by set_warm_start().
        """
        return _BaseLpGlobalLogisticSolver.convert_to_variable_values(self, solution)


class GlobalLogisticMathOptSolver(OrtoolsMathOptMilpSolver, _BaseLpGlobalLogisticSolver):
    """OR-Tools MathOpt implementation of Global Logistic MILP solver."""

    def convert_to_variable_values(
        self, solution: GlobalLogisticSolution
    ) -> Dict[mathopt.Variable, float]:
        """Convert a solution to a mapping between model variables and their values.

        Will be used by set_warm_start() to provide a suitable SolutionHint.variable_values.
        """
        return _BaseLpGlobalLogisticSolver.convert_to_variable_values(self, solution)


# Backward compatibility alias
GlobalLogisticMilpSolver = GlobalLogisticGurobiSolver