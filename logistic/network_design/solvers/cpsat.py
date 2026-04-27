#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import logging
from typing import Any, Dict, Optional, Tuple, Iterable

from ortools.sat.python.cp_model import CpModel, CpSolverSolutionCallback

from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from logistic.network_design.problem import Location, Part, TransportOption, GlobalLogisticSolution, \
    GlobalLogisticProblem
from discrete_optimization.generic_tasks_tools.solvers.cpsat import AllocationCpSatSolver
logger = logging.getLogger(__name__)


class GlobalLogisticCPSatSolver(AllocationCpSatSolver[]):
    def __init__(
            self,
            problem: GlobalLogisticProblem,
            params_objective_function: Optional[ParamsObjectiveFunction] = None,
            **kwargs: Any,
    ):
        super().__init__(
            problem=problem, params_objective_function=params_objective_function, **kwargs
        )
        self.problem: GlobalLogisticProblem = problem
        # Mappings for CP-SAT Variables
        self.allocation_variables: Dict[Tuple[Part, Location], Any] = {}
        self.edge_variables: Dict[TransportOption, Any] = {}
        self.location_variable: Dict[Location, Any] = {}
        self.transport_mean_variable: Dict[str, Any] = {}
        # Will store our objective variables for lexicographic optimization
        self.obj_vars: Dict[str, Any] = {}
        # Scale float costs to integers for the CP-SAT engine
        self.scaling_factor = 1

    def init_model(self,
                   scaling_factor: int = 1,
                   use_classic_flow_constraints: bool = True,
                   max_flows_per_part_multiplier: int = 2,
                   forbid_transit_at_consumer_sites: bool = False,
                   relax_default_sourcing: bool = False,
                   **kwargs: Any) -> None:
        """
        Initialize the CP-SAT model.

        Args:
            scaling_factor (int, optional): Scaling factor. Defaults to 1.
            use_classic_flow_constraints: If True, uses classic flow conservation constraints
            max_flows_per_part_multiplier: Multiplier for max transport routes per part
            forbid_transit_at_consumer_sites: If True, forbid transit at consumer-only sites
            relax_default_sourcing: If True, relax sourcing_requirement=1 constraints to allow
                                   0 or more locations (useful when other parts have higher requirements)
        """
        self.cp_model = CpModel()

        # ==========================================
        # 1. Variables
        # ==========================================
        for part in self.problem.parts:
            for loc in part.produceable_at:
                self.allocation_variables[part, loc] = self.cp_model.NewBoolVar(f"x_{part.name}_{loc.name}")

        for loc in self.problem.locations:
            self.location_variable[loc] = self.cp_model.NewBoolVar(f"u_{loc.name}")

        for mean in self.problem.transport_means:
            self.transport_mean_variable[mean.name] = self.cp_model.NewBoolVar(f"v_{mean.name}")

        for t_opt in self.problem.transport_options:
            self.edge_variables[t_opt] = self.cp_model.NewBoolVar(
                f"y_{t_opt.part.name}_{t_opt.origin.name}_{t_opt.destination.name}_{t_opt.mean.name}"
            )

        # ==========================================
        # 2. Base Constraints
        # ==========================================
        for part in self.problem.parts:
            valid_x_vars = [self.allocation_variables[part, loc] for loc in part.produceable_at]
            if valid_x_vars:
                self.cp_model.Add(sum(valid_x_vars) >= part.sourcing_requirement)
            if part.sourcing_requirement == 1 and not relax_default_sourcing:
                # Enforce exactly one location only if not relaxing
                self.cp_model.AddExactlyOne(valid_x_vars)

        for (part, loc), x_var in self.allocation_variables.items():
            #self.cp_model.AddImplication(x_var, self.var_u[loc])
            self.cp_model.add(self.location_variable[loc] == 1).only_enforce_if(x_var)

        for t_opt, y_var in self.edge_variables.items():
            self.cp_model.AddImplication(y_var, self.transport_mean_variable[t_opt.mean.name])

        # ==========================================
        # 2b. Limit Transport Flows Per Part (Search Space Reduction)
        # ==========================================
        # Constraint: Each part can only use a limited number of transport routes
        # This significantly reduces the search space while maintaining feasibility
        if max_flows_per_part_multiplier is not None and max_flows_per_part_multiplier > 0:
            for part in self.problem.parts:
                transport_vars_for_part = [
                    self.edge_variables[t] for t in self.problem.transport_options
                    if t.part == part
                ]
                if transport_vars_for_part:
                    # Heuristic: Allow at most sourcing_requirement * multiplier transport routes
                    # This is usually sufficient for reasonable supply chain configurations
                    max_flows = max(2, part.sourcing_requirement * max_flows_per_part_multiplier)
                    self.cp_model.Add(sum(transport_vars_for_part) <= max_flows)
                    logger.debug(f"Part {part.name}: limited to {max_flows} transport routes (out of {len(transport_vars_for_part)} possible)")

        # ==========================================
        # 3. Multi-Hop Flow Conservation
        # ==========================================
        for part in self.problem.parts:
            for loc in self.problem.locations:
                produced_var = self.allocation_variables.get((part, loc), 0)
                consumed_var = self.allocation_variables.get((part.parent_part, loc), 0) if part.parent_part else 0

                inbound_vars = [
                    self.edge_variables[t] for t in self.problem.transport_options
                    if t.part == part and t.destination == loc
                ]
                outbound_vars = [
                    self.edge_variables[t] for t in self.problem.transport_options
                    if t.part == part and t.origin == loc
                ]

                if use_classic_flow_constraints:
                    # ---------------------------------------------------
                    # CLASSIC LINEAR FLOW INEQUALITIES (Rules A, B, C)
                    # NO FLOW CONSERVATION! Parts can be produced, consumed, AND shipped simultaneously
                    # ---------------------------------------------------

                    # Rule A: No Starvation (If consumed here, it must be produced or imported)
                    self.cp_model.Add(produced_var + sum(inbound_vars) >= consumed_var)

                    # Rule B: No Dead-Ends (If it exists here, it must be consumed or exported)
                    if part.parent_part:
                        # If produced locally, it must go somewhere
                        self.cp_model.Add(consumed_var + sum(outbound_vars) >= produced_var)
                        # If imported via a specific route, it must go somewhere
                        for inbound_y in inbound_vars:
                            self.cp_model.Add(consumed_var + sum(outbound_vars) >= inbound_y)

                    # Rule C: No Spontaneous Generation (If exported, it must be produced or imported)
                    for outbound_y in outbound_vars:
                        self.cp_model.Add(produced_var + sum(inbound_vars) >= outbound_y)

                else:
                    # ---------------------------------------------------
                    # INDICATOR CONSTRAINTS (OnlyEnforceIf)
                    # NO FLOW CONSERVATION! Parts can be produced, consumed, AND shipped simultaneously
                    # ---------------------------------------------------

                    # Rule A: No Starvation
                    if (part.parent_part, loc) in self.allocation_variables:
                        self.cp_model.Add(produced_var + sum(inbound_vars) >= 1).OnlyEnforceIf(consumed_var)

                    # Rule B: No Dead-Ends (Orphans)
                    if part.parent_part:
                        if (part, loc) in self.allocation_variables:
                            self.cp_model.Add(consumed_var + sum(outbound_vars) >= 1).OnlyEnforceIf(produced_var)
                        for inbound_y in inbound_vars:
                            self.cp_model.Add(consumed_var + sum(outbound_vars) >= 1).OnlyEnforceIf(inbound_y)

                    # Rule C: No Spontaneous Generation
                    for outbound_y in outbound_vars:
                        if (part, loc) not in self.allocation_variables: # Lieu de transport
                            self.cp_model.Add(sum(inbound_vars) >= 1).OnlyEnforceIf(outbound_y)
                        else:
                            self.cp_model.Add(produced_var + sum(inbound_vars) >= 1).OnlyEnforceIf(outbound_y)

        # ==========================================
        # 3b. Heuristic: Forbid Transit at Consumer Sites
        # ==========================================
        if forbid_transit_at_consumer_sites:
            logger.info("Adding transit-forbidding constraints...")
            for part in self.problem.parts:
                if not part.parent_part:
                    continue

                parent = part.parent_part
                # Find locations where the parent can be produced (potential consumers)
                for loc in parent.produceable_at:
                    if (parent, loc) not in self.allocation_variables:
                        continue

                    inbound_vars = [
                        self.edge_variables[t] for t in self.problem.transport_options
                        if t.part == part and t.destination == loc
                    ]

                    if inbound_vars:
                        # If part arrives at a location where parent can be produced,
                        # then parent MUST be produced there (forbids pure transit)
                        has_inbound = self.cp_model.NewBoolVar(f"has_inbound_{part.name}_{loc.name}")

                        # has_inbound = 1 iff sum(inbound_vars) >= 1
                        self.cp_model.Add(sum(inbound_vars) >= 1).OnlyEnforceIf(has_inbound)
                        self.cp_model.Add(sum(inbound_vars) == 0).OnlyEnforceIf(has_inbound.Not())

                        # If part arrives, parent must be produced
                        self.cp_model.AddImplication(has_inbound, self.allocation_variables[(parent, loc)])

        # ==========================================
        # 4. Metrics & Objective Tracking
        # ==========================================

        # A. Calculate Base Costs

        # Factory activation cost (fixed cost per active location)
        max_factory_cost = sum(int(loc.factory_activation_cost * self.scaling_factor) for loc in self.problem.locations)
        factory_cost_var = self.cp_model.NewIntVar(0, max_factory_cost, "factory_activation_cost")
        self.cp_model.Add(factory_cost_var == sum(
            self.location_variable[loc] * int(loc.factory_activation_cost * self.scaling_factor) for loc in self.problem.locations
        ))

        # Part production cost (variable cost per part-location assignment)
        max_part_cost = sum(
            int(cost * self.scaling_factor)
            for cost in self.problem.part_location_production_costs.values()
        )
        part_cost_var = self.cp_model.NewIntVar(0, max_part_cost, "part_production_cost")
        self.cp_model.Add(part_cost_var == sum(
            self.allocation_variables[(part, loc)] * int(self.problem.part_location_production_costs[(part, loc)] * self.scaling_factor)
            for (part, loc) in self.allocation_variables.keys()
        ))

        max_trans_cost = sum(int(t.total_cost * self.scaling_factor) for t in self.problem.transport_options)
        trans_cost_var = self.cp_model.NewIntVar(0, max_trans_cost, "transport_cost")
        self.cp_model.Add(trans_cost_var == sum(
            self.edge_variables[t] * int(t.total_cost * self.scaling_factor) for t in self.problem.transport_options
        ))

        max_nrc_cost = sum(int(m.nrc_cost * self.scaling_factor) for m in self.problem.transport_means)
        nrc_cost_var = self.cp_model.NewIntVar(0, max_nrc_cost, "nrc_cost")
        self.cp_model.Add(nrc_cost_var == sum(
            self.transport_mean_variable[m.name] * int(m.nrc_cost * self.scaling_factor) for m in self.problem.transport_means
        ))

        # Aggregated production cost (factory + part)
        max_aggregated_prod_cost = max_factory_cost + max_part_cost
        aggregated_prod_cost_var = self.cp_model.NewIntVar(0, max_aggregated_prod_cost, "production_cost")
        self.cp_model.Add(aggregated_prod_cost_var == factory_cost_var + part_cost_var)

        total_cost_var = self.cp_model.NewIntVar(
            0, max_factory_cost + max_part_cost + max_trans_cost + max_nrc_cost, "total_cost"
        )
        self.cp_model.Add(total_cost_var == factory_cost_var + part_cost_var + trans_cost_var + nrc_cost_var)

        # B. Country work-share constraints (icWorkshare from the paper)
        if self.problem.country_constraints:
            logger.info(f"Adding {len(self.problem.country_constraints)} country work-share constraints...")
            max_val_added = sum(int(part.value_added * self.scaling_factor) for part in self.problem.parts)

            # Compute value-added per country
            country_value_vars = {}
            for constraint in self.problem.country_constraints:
                terms = [
                    self.allocation_variables[(part, loc)] * int(part.value_added * self.scaling_factor)
                    for (part, loc) in self.allocation_variables.keys()
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

        # C. Location-level value-added constraints
        locations_with_constraints = [loc for loc in self.problem.locations
                                      if loc.min_value_added is not None or loc.max_value_added is not None]
        if locations_with_constraints:
            logger.info(f"Adding value-added constraints for {len(locations_with_constraints)} locations...")
            max_val_added = sum(int(part.value_added * self.scaling_factor) for part in self.problem.parts)

            for loc in locations_with_constraints:
                # Sum value-added for all parts produced at this location
                terms = [
                    self.allocation_variables[(part, l)] * int(part.value_added * self.scaling_factor)
                    for (part, l) in self.allocation_variables.keys()
                    if l == loc
                ]
                if terms:
                    loc_val_var = self.cp_model.NewIntVar(0, max_val_added, f"val_added_loc_{loc.name}")
                    self.cp_model.Add(loc_val_var == sum(terms))

                    if loc.min_value_added is not None:
                        self.cp_model.Add(loc_val_var >= int(loc.min_value_added * self.scaling_factor))
                    if loc.max_value_added is not None:
                        self.cp_model.Add(loc_val_var <= int(loc.max_value_added * self.scaling_factor))

        # Store for lexico
        self.obj_vars = {
            "factory_activation_cost": factory_cost_var,
            "part_production_cost": part_cost_var,
            "production_cost": aggregated_prod_cost_var,  # Aggregated (factory + part)
            "transport_cost": trans_cost_var,
            "nrc_cost": nrc_cost_var,
        }

        # C. Apply ParamsObjectiveFunction Weights
        objective_list = []
        # Float weights need to be converted to int for CP-SAT
        for obj, weight in zip(self.params_objective_function.objectives,
                               self.params_objective_function.weights):
            if obj in self.obj_vars:
                objective_list.append(weight*self.obj_vars[obj])

        self.cp_model.Minimize(sum(objective_list))

    def create_allocation_variables(self):

    def implements_lexico_api(self) -> bool:
        return True
    def add_lexico_constraint(self, obj: str, value: float) -> Iterable[Any]:
        return [self.cp_model.add(self.obj_vars[obj])<=value]
    def get_lexico_objectives_available(self) -> list[str]:
        return list(self.obj_vars.keys())
    def set_lexico_objective(self, obj: str) -> None:
        self.cp_model.minimize(self.obj_vars[obj])
    def get_lexico_objective_value(self, obj: str, res: ResultStorage) -> float:
        eval = self.problem.evaluate(res[-1][0])
        return self.scaling_factor*eval[obj]
    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> GlobalLogisticSolution:
        for obj in self.obj_vars:
            logger.info(f"{obj} : {cpsolvercb.value(self.obj_vars[obj])}")
        part_to_locations = {part: set() for part in self.problem.parts}
        transport_plan = set()

        for (part, loc), x_var in self.allocation_variables.items():
            if cpsolvercb.Value(x_var) == 1:
                part_to_locations[part].add(loc)

        for t_opt, y_var in self.edge_variables.items():
            if cpsolvercb.Value(y_var) == 1:
                transport_plan.add(t_opt)

        return GlobalLogisticSolution(
            problem=self.problem,
            part_to_locations=part_to_locations,
            transport_plan=transport_plan
        )