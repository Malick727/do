"""
Feasibility Analysis Problem for Global Logistics.

This is a pure constraint satisfaction problem (CSP) that checks if a valid
production allocation exists, WITHOUT considering costs or optimization.

Useful for:
- Quickly testing if sourcing_requirement configurations are feasible
- Analyzing country_constraint scenarios
- Understanding dependency constraints
"""

import logging
from typing import Any, Dict, Optional, Set
from dataclasses import dataclass

from ortools.sat.python.cp_model import CpModel, CpSolverSolutionCallback

from discrete_optimization.generic_tools.do_problem import Problem, Solution
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver

from logistic.network_design.problem import Location, Part, GlobalLogisticSolution, GlobalLogisticProblem

logger = logging.getLogger(__name__)


@dataclass
class FeasibilityResult:
    """Result of a feasibility check."""
    is_feasible: bool
    solution: Optional[GlobalLogisticSolution] = None
    violated_constraints: list = None
    message: str = ""

    def __post_init__(self):
        if self.violated_constraints is None:
            self.violated_constraints = []


class FeasibilitySolver(OrtoolsCpSatSolver):
    """
    CP-SAT based feasibility solver.

    Only checks if a valid production allocation exists - no optimization.
    Much faster than the full model for scenario analysis.
    """

    def __init__(
        self,
        problem: GlobalLogisticProblem,
        **kwargs: Any,
    ):
        # Use the base problem's get_objective_register
        super().__init__(problem=problem, params_objective_function=None, **kwargs)
        self.problem: GlobalLogisticProblem = problem

        # Variables: only production allocation (no transport)
        self.var_x: Dict = {}

    def init_model(self, require_reachability: bool = True, **kwargs: Any) -> None:
        """
        Initialize the feasibility model.

        Args:
            require_reachability: If True, enforces that all child parts must be
                                 reachable at locations where parent is produced
        """
        logger.info("="*80)
        logger.info("Initializing Feasibility Model (Pure CSP)")
        logger.info("="*80)
        graph_per_part = {part: self.problem.build_transport_graph(part)
                          for part in self.problem.parts}

        self.cp_model = CpModel()

        # ==========================================
        # 1. Variables (Production Allocation Only)
        # ==========================================
        logger.info("\nCreating variables...")

        for part in self.problem.parts:
            for loc in part.produceable_at:
                self.var_x[(part, loc)] = self.cp_model.NewBoolVar(
                    f"x_{part.name}_{loc.name}"
                )

        logger.info(f"  Production variables: {len(self.var_x)}")

        # ==========================================
        # 2. Constraints
        # ==========================================
        logger.info("\nAdding constraints...")

        # 2a. Sourcing requirements
        for part in self.problem.parts:
            valid_x_vars = [self.var_x[(part, loc)] for loc in part.produceable_at]
            if valid_x_vars:
                self.cp_model.Add(sum(valid_x_vars) >= part.sourcing_requirement)
                logger.debug(f"  {part.name}: sourcing >= {part.sourcing_requirement}")

        # 2b. Reachability constraints (if enabled)
        if require_reachability:
            logger.info("  Adding reachability constraints...")
            reachability_count = 0

            for part in self.problem.parts:
                if not part.child_parts:
                    continue

                # Build transport graph for each child
                child_graphs = {}
                for child in part.child_parts:
                    child_graphs[child] = graph_per_part[child]

                # For each location where part can be produced
                for loc in part.produceable_at:
                    if (part, loc) not in self.var_x:
                        continue

                    # Check each child can reach this location
                    for child in part.child_parts:
                        # Get all production sites for this child
                        child_prod_sites = list(child.produceable_at)

                        # Check if ANY child production site can reach loc
                        G = child_graphs[child]
                        reachable_sites = []

                        for prod_site in child_prod_sites:
                            # Child can be at loc if:
                            # 1. Produced at loc directly
                            if prod_site == loc:
                                reachable_sites.append(prod_site)
                            # 2. Has a transport path from prod_site to loc
                            elif G.has_node(prod_site) and G.has_node(loc):
                                import networkx as nx
                                if nx.has_path(G, prod_site, loc):
                                    reachable_sites.append(prod_site)

                        if not reachable_sites:
                            # No way for child to reach loc - forbid producing part there
                            self.cp_model.Add(self.var_x[(part, loc)] == 0)
                            logger.debug(
                                f"    Forbid {part.name} at {loc.name}: "
                                f"{child.name} unreachable"
                            )
                            reachability_count += 1

            logger.info(f"  Added {reachability_count} reachability constraints")

        # 2c. Country work-share constraints
        if self.problem.country_constraints:
            logger.info(f"  Adding {len(self.problem.country_constraints)} country constraints...")
            max_val_added = sum(int(part.value_added) for part in self.problem.parts)

            for constraint in self.problem.country_constraints:
                terms = [
                    self.var_x[(part, loc)] * int(part.value_added)
                    for (part, loc) in self.var_x.keys()
                    if loc.country == constraint.country
                ]
                if terms:
                    if constraint.min_value_added is not None:
                        self.cp_model.Add(sum(terms) >= int(constraint.min_value_added))
                    if constraint.max_value_added is not None:
                        self.cp_model.Add(sum(terms) <= int(constraint.max_value_added))

        # 2d. Location-level value-added constraints
        locations_with_constraints = [
            loc for loc in self.problem.locations
            if loc.min_value_added is not None or loc.max_value_added is not None
        ]
        if locations_with_constraints:
            logger.info(f"  Adding {len(locations_with_constraints)} location constraints...")

            for loc in locations_with_constraints:
                terms = [
                    self.var_x[(part, l)] * int(part.value_added)
                    for (part, l) in self.var_x.keys()
                    if l == loc
                ]
                if terms:
                    if loc.min_value_added is not None:
                        self.cp_model.Add(sum(terms) >= int(loc.min_value_added))
                    if loc.max_value_added is not None:
                        self.cp_model.Add(sum(terms) <= int(loc.max_value_added))

        logger.info("✓ Feasibility model initialized")
        logger.info("="*80 + "\n")

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> GlobalLogisticSolution:
        """Extract the feasible production allocation (no transport plan)."""
        part_to_locations = {part: set() for part in self.problem.parts}

        for (part, loc), x_var in self.var_x.items():
            if cpsolvercb.Value(x_var) == 1:
                part_to_locations[part].add(loc)

        # No transport plan in this simplified problem
        return GlobalLogisticSolution(
            problem=self.problem,
            part_to_locations=part_to_locations,
            transport_plan=set()
        )


def check_feasibility(
    problem: GlobalLogisticProblem,
    time_limit: int = 10,
    require_reachability: bool = True,
    verbose: bool = True
) -> FeasibilityResult:
    """
    Quick feasibility check for a production allocation configuration.

    Args:
        problem: The GlobalLogisticProblem instance
        time_limit: Maximum time in seconds
        require_reachability: If True, enforces child part reachability
        verbose: If True, prints detailed logs

    Returns:
        FeasibilityResult with solution if feasible
    """
    if verbose:
        logger.info("\n" + "="*80)
        logger.info("FEASIBILITY ANALYSIS")
        logger.info("="*80)

    solver = FeasibilitySolver(problem)
    solver.init_model(require_reachability=require_reachability)

    # Solve
    results = solver.solve(time_limit=time_limit)

    if len(results) > 0:
        solution = results[0][0]
        is_feasible = problem.satisfy(solution, verbose=verbose)

        if is_feasible:
            if verbose:
                logger.info("\n✓ FEASIBLE - Found valid production allocation")
                logger.info(f"  Active locations: {sum(1 for locs in solution.part_to_locations.values() if locs)}")
                logger.info(f"  Total value-added: {sum(part.value_added for part, locs in solution.part_to_locations.items() for loc in locs):.0f}")

            return FeasibilityResult(
                is_feasible=True,
                solution=solution,
                message="Found feasible allocation"
            )
        else:
            if verbose:
                logger.warning("\n⚠️  SOLVER FOUND SOLUTION BUT VIOLATED CONSTRAINTS")

            return FeasibilityResult(
                is_feasible=False,
                solution=solution,
                message="Solution violates problem constraints"
            )
    else:
        if verbose:
            logger.error("\n❌ INFEASIBLE - No valid production allocation exists")
            logger.error("  Possible causes:")
            logger.error("    - Sourcing requirements too high")
            logger.error("    - Country constraints too restrictive")
            logger.error("    - Child parts unreachable at production sites")

        return FeasibilityResult(
            is_feasible=False,
            message="No feasible allocation found within time limit"
        )
