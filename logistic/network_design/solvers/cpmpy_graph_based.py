"""
CPMpy-Based Graph Feasibility Solver for Global Logistics with Explainability.

This solver uses CPMpy instead of OR-Tools CP-SAT to enable:
- MUS (Minimal Unsatisfiable Subset) computation for infeasibility debugging
- MCS (Minimal Correction Set) computation
- Constraint explainability through MetaConstraints

Key features:
- Same graph-based modeling as GraphBasedFeasibilitySolver
- MetaConstraints for tracking different constraint types
- Ability to explain WHY a problem is infeasible
- Categorizes constraints: sourcing requirements, dependencies, dead-ends, etc.
"""

import logging
from typing import Any, Optional

import cpmpy
from cpmpy import Model

from discrete_optimization.generic_tools.cpmpy_tools import (
    CpmpySolver,
    MetaCpmpyConstraint,
)
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
)
from logistic.network_design.problem import GlobalLogisticSolution, GlobalLogisticProblem
from logistic.network_design.solvers.graph_analysis import LogisticProblemGraphAnalysis

logger = logging.getLogger(__name__)


class CpmpyGraphBasedFeasibilitySolver(CpmpySolver):
    """
    CPMpy solver using the (part, location) dependency graph.

    Enables MUS/MCS computation for debugging infeasibilities.

    Instead of modeling individual transport flows, models the problem as:
    - Select nodes in the (part, location) graph
    - Ensure dependency constraints: if (parent, loc) is selected,
      then for each child, at least one (child, child_loc) predecessor must be selected

    All constraints are wrapped in MetaCpmpyConstraint for explainability.
    """

    def __init__(
        self,
        problem: GlobalLogisticProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        enforce_no_dead_ends: bool = True,
        minimize_active_nodes: bool = False,
        enforce_exact_sourcing: bool = True,
        solver_name: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            problem=problem,
            params_objective_function=params_objective_function,
            solver_name=solver_name,
            **kwargs,
        )
        self.problem: GlobalLogisticProblem = problem
        self.enforce_no_dead_ends = enforce_no_dead_ends
        self.minimize_active_nodes = minimize_active_nodes
        self.enforce_exact_sourcing = enforce_exact_sourcing

        # Build the dependency graph
        logger.info("Building (part, location) dependency graph...")
        self.graph_analyzer = LogisticProblemGraphAnalysis(problem)
        self.dependency_graph = self.graph_analyzer.get_part_and_location_graph()

        logger.info(f"  Nodes: {self.dependency_graph.number_of_nodes()}")
        logger.info(f"  Edges: {self.dependency_graph.number_of_edges()}")

        # Variables and meta-constraints
        self.var_x = {}
        self.meta_constraints = []

    def init_model(self, enforce_exact_sourcing: Optional[bool] = None, **kwargs: Any) -> None:
        """Initialize the graph-based feasibility model with MetaConstraints."""
        logger.info("=" * 80)
        logger.info("Initializing CPMpy Graph-Based Feasibility Model")
        logger.info("=" * 80)

        # Use parameter from init_model if provided, otherwise use constructor value
        if enforce_exact_sourcing is None:
            enforce_exact_sourcing = self.enforce_exact_sourcing

        self.model = Model()
        self.meta_constraints = []  # Reset meta-constraints

        # ==========================================
        # 1. Variables
        # ==========================================
        logger.info("\nCreating variables...")

        # Create variable for each node in the graph
        for (part, loc) in self.dependency_graph.nodes():
            self.var_x[(part, loc)] = cpmpy.boolvar(name=f"x_{part.name}_{loc.name}")

        logger.info(f"  Variables: {len(self.var_x)}")

        # ==========================================
        # 2. Constraints
        # ==========================================
        logger.info("\nAdding constraints...")

        # 2a. Sourcing requirements
        # Each part must be produced at >= sourcing_requirement locations
        exact_sourcing_count = 0
        for part in self.problem.parts:
            part_vars = [
                self.var_x[(p, loc)] for (p, loc) in self.var_x.keys()
                if p == part
            ]
            if part_vars:
                # Minimum requirement (>= constraint)
                constraint_min = cpmpy.sum(part_vars) >= part.sourcing_requirement
                self.model += [constraint_min]
                # Store as meta-constraint
                self.meta_constraints.append(
                    MetaCpmpyConstraint(
                        name=f"sourcing_min_{part.name}",
                        constraints=[constraint_min],
                        metadata={
                            "type": "sourcing_requirement_min",
                            "part": part.name,
                            "requirement": part.sourcing_requirement,
                        },
                    )
                )

                # Exact requirement (== constraint) - optional
                if enforce_exact_sourcing:
                    constraint_exact = cpmpy.sum(part_vars) == part.sourcing_requirement
                    self.model += [constraint_exact]
                    # Store as separate meta-constraint for MUS/MCS analysis
                    self.meta_constraints.append(
                        MetaCpmpyConstraint(
                            name=f"sourcing_exact_{part.name}",
                            constraints=[constraint_exact],
                            metadata={
                                "type": "sourcing_requirement_exact",
                                "part": part.name,
                                "requirement": part.sourcing_requirement,
                            },
                        )
                    )
                    exact_sourcing_count += 1

        if enforce_exact_sourcing:
            logger.info(f"  Added sourcing requirement constraints for {len(self.problem.parts)} parts (MIN + EXACT)")
        else:
            logger.info(f"  Added sourcing requirement constraints for {len(self.problem.parts)} parts (MIN only)")

        # 2b. Dependency constraints (graph-based)
        # If (parent, loc) is selected, then for each child, at least one
        # predecessor (child, child_loc) must be selected
        constraint_count = 0

        for part in self.problem.parts:
            if not part.child_parts:
                continue  # Leaf parts have no dependencies

            for loc in part.produceable_at:
                if (part, loc) not in self.var_x:
                    continue

                parent_var = self.var_x[(part, loc)]

                # For each child of this part
                for child in part.child_parts:
                    # Find all (child, child_loc) nodes that have an edge to (part, loc)
                    predecessors = [
                        self.var_x[(p, l)]
                        for (p, l) in self.dependency_graph.predecessors((part, loc))
                        if p == child
                    ]

                    if predecessors:
                        # If parent is produced at loc, at least one child source must be active
                        # parent_var => sum(predecessors) >= 1
                        constraint = (cpmpy.sum(predecessors) >= 1) | (~parent_var)
                        # Equivalent to: parent_var.implies(sum(predecessors) >= 1)
                        self.model += [constraint]

                        # Store as meta-constraint
                        self.meta_constraints.append(
                            MetaCpmpyConstraint(
                                name=f"dependency_{part.name}_{loc.name}_{child.name}",
                                constraints=[constraint],
                                metadata={
                                    "type": "dependency",
                                    "parent_part": part.name,
                                    "parent_location": loc.name,
                                    "child_part": child.name,
                                },
                            )
                        )
                        constraint_count += 1
                    else:
                        # No valid predecessor for this child - forbid producing parent here
                        constraint = parent_var == 0
                        self.model += [constraint]

                        # Store as meta-constraint (hard constraint - infeasibility indicator)
                        self.meta_constraints.append(
                            MetaCpmpyConstraint(
                                name=f"forbid_{part.name}_{loc.name}_{child.name}",
                                constraints=[constraint],
                                metadata={
                                    "type": "impossible_dependency",
                                    "parent_part": part.name,
                                    "parent_location": loc.name,
                                    "child_part": child.name,
                                    "reason": "no valid path for child",
                                },
                            )
                        )
                        logger.debug(
                            f"  Forbid {part.name} at {loc.name}: "
                            f"child {child.name} has no valid path"
                        )

        logger.info(f"  Added {constraint_count} dependency constraints")

        # 2c. No dead-ends constraints (forward constraints)
        # If a non-final part is produced at a location, it must be consumed or shipped out
        if self.enforce_no_dead_ends:
            no_dead_end_count = 0

            for part in self.problem.parts:
                # Skip final products (parts with no parent)
                if not part.parent_part:
                    continue

                for loc in part.produceable_at:
                    if (part, loc) not in self.var_x:
                        continue

                    child_var = self.var_x[(part, loc)]

                    # Find all possible uses for this (part, loc):
                    # 1. Consumed locally: parent produced at same location
                    local_consumption_vars = []
                    if part.parent_part and (part.parent_part, loc) in self.var_x:
                        local_consumption_vars.append(self.var_x[(part.parent_part, loc)])

                    # 2. Shipped out: any successor in dependency graph
                    successor_vars = [
                        self.var_x[(p, l)]
                        for (p, l) in self.dependency_graph.successors((part, loc))
                        if (p, l) in self.var_x
                    ]

                    # Combine all possible uses
                    usage_vars = local_consumption_vars + successor_vars

                    if usage_vars:
                        # If child produced here => at least one use must be active
                        # child_var => sum(usage_vars) >= 1
                        constraint = (cpmpy.sum(usage_vars) >= 1) | (~child_var)
                        self.model += [constraint]

                        # Store as meta-constraint
                        self.meta_constraints.append(
                            MetaCpmpyConstraint(
                                name=f"no_dead_end_{part.name}_{loc.name}",
                                constraints=[constraint],
                                metadata={
                                    "type": "no_dead_ends",
                                    "part": part.name,
                                    "location": loc.name,
                                },
                            )
                        )
                        no_dead_end_count += 1
                    else:
                        # No possible use - forbid producing here
                        constraint = child_var == 0
                        self.model += [constraint]

                        # Store as meta-constraint (hard constraint)
                        self.meta_constraints.append(
                            MetaCpmpyConstraint(
                                name=f"forbid_dead_end_{part.name}_{loc.name}",
                                constraints=[constraint],
                                metadata={
                                    "type": "impossible_dead_end",
                                    "part": part.name,
                                    "location": loc.name,
                                    "reason": "no parent consumes it and no outbound path",
                                },
                            )
                        )
                        logger.debug(
                            f"  Forbid {part.name} at {loc.name}: "
                            f"no parent consumes it and no outbound path"
                        )

            logger.info(f"  Added {no_dead_end_count} no-dead-ends constraints")

        # 2d. Country work-share constraints
        if self.problem.country_constraints:
            logger.info(f"  Adding {len(self.problem.country_constraints)} country constraints...")

            for constraint_obj in self.problem.country_constraints:
                terms = [
                    self.var_x[(part, loc)] * int(part.value_added)
                    for (part, loc) in self.var_x.keys()
                    if loc.country == constraint_obj.country
                ]
                if terms:
                    constraints_list = []
                    if constraint_obj.min_value_added is not None:
                        c_min = cpmpy.sum(terms) >= int(constraint_obj.min_value_added)
                        self.model += [c_min]
                        constraints_list.append(c_min)
                    if constraint_obj.max_value_added is not None:
                        c_max = cpmpy.sum(terms) <= int(constraint_obj.max_value_added)
                        self.model += [c_max]
                        constraints_list.append(c_max)

                    if constraints_list:
                        self.meta_constraints.append(
                            MetaCpmpyConstraint(
                                name=f"country_{constraint_obj.country}",
                                constraints=constraints_list,
                                metadata={
                                    "type": "country_constraint",
                                    "country": constraint_obj.country,
                                    "min_value_added": constraint_obj.min_value_added,
                                    "max_value_added": constraint_obj.max_value_added,
                                },
                            )
                        )

        # 2e. Location-level value-added constraints
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
                    constraints_list = []
                    if loc.min_value_added is not None:
                        c_min = cpmpy.sum(terms) >= int(loc.min_value_added)
                        self.model += [c_min]
                        constraints_list.append(c_min)
                    if loc.max_value_added is not None:
                        c_max = cpmpy.sum(terms) <= int(loc.max_value_added)
                        self.model += [c_max]
                        constraints_list.append(c_max)

                    if constraints_list:
                        self.meta_constraints.append(
                            MetaCpmpyConstraint(
                                name=f"location_{loc.name}",
                                constraints=constraints_list,
                                metadata={
                                    "type": "location_constraint",
                                    "location": loc.name,
                                    "min_value_added": loc.min_value_added,
                                    "max_value_added": loc.max_value_added,
                                },
                            )
                        )

        # 3. Objective (optional)
        # Minimize number of active (part, location) nodes to avoid orphans
        if self.minimize_active_nodes:
            logger.info("\nAdding objective: minimize active nodes...")
            objective_expr = cpmpy.sum(list(self.var_x.values()))
            self.model.minimize(objective_expr)
            logger.info("  Objective: minimize sum of x[(part, loc)] variables")

        logger.info(f"\n✓ CPMpy graph-based model initialized")
        logger.info(f"  Total meta-constraints: {len(self.meta_constraints)}")
        logger.info("=" * 80 + "\n")

    def retrieve_current_solution(self) -> GlobalLogisticSolution:
        """Extract the production allocation from selected graph nodes."""
        part_to_locations = {part: set() for part in self.problem.parts}

        for (part, loc), x_var in self.var_x.items():
            if x_var.value() == 1:
                part_to_locations[part].add(loc)

        # Reconstruct transport plan from graph edges using cached routes
        transport_plan = set()
        transport_edges_added = set()

        for part in self.problem.parts:
            if not part.parent_part:
                continue

            # For each active child location
            for child_loc in part_to_locations.get(part, set()):
                child_node = (part, child_loc)

                # Check if this node has any active successors in the dependency graph
                for successor_node in self.dependency_graph.successors(child_node):
                    # Check if the successor is active
                    if successor_node in self.var_x and self.var_x[successor_node].value() == 1:
                        successor_part, successor_loc = successor_node

                        # Get the cached route from the edge attribute
                        edge_data = self.dependency_graph[child_node][successor_node]
                        cached_route = edge_data.get('route')

                        # Add transport if locations differ and route is cached
                        if child_loc != successor_loc:
                            edge_key = (part, child_loc, successor_loc)
                            if edge_key not in transport_edges_added:
                                if cached_route and cached_route.is_feasible:
                                    # Use the precomputed route from cache
                                    transport_plan.update(cached_route.transport_sequence)
                                    transport_edges_added.add(edge_key)
                                else:
                                    logger.warning(
                                        f"No cached route for {part.name}: "
                                        f"{child_loc.name} -> {successor_loc.name}"
                                    )

        return GlobalLogisticSolution(
            problem=self.problem,
            part_to_locations=part_to_locations,
            transport_plan=transport_plan
        )

    def get_soft_meta_constraints(self) -> list[MetaCpmpyConstraint]:
        """
        Get soft meta-constraints (can be relaxed to find MUS/MCS).

        Soft constraints are those that could be removed or relaxed:
        - Sourcing requirements (could be reduced)
        - Sourcing exact requirements (could be relaxed to >=)
        - Country constraints (business requirements)
        - Location constraints (business requirements)

        Returns:
            List of soft meta-constraints
        """
        soft_types = {
            "sourcing_requirement_min",
            "sourcing_requirement_exact",
            "country_constraint",
            "location_constraint"
        }
        return [
            mc for mc in self.meta_constraints
            if mc.metadata.get("type") in soft_types
        ]

    def get_hard_meta_constraints(self) -> list[MetaCpmpyConstraint]:
        """
        Get hard meta-constraints (structural, cannot be relaxed).

        Hard constraints are physical/structural facts:
        - Dependency constraints (BOM structure)
        - Dead-ends prevention (physical flow)
        - Impossible constraints (no valid paths)

        Returns:
            List of hard meta-constraints
        """
        hard_types = {
            "dependency",
            "no_dead_ends",
            "impossible_dependency",
            "impossible_dead_end",
        }
        return [
            mc for mc in self.meta_constraints
            if mc.metadata.get("type") in hard_types
        ]

    def get_meta_constraints_by_type(self, constraint_type: str) -> list[MetaCpmpyConstraint]:
        """Get all meta-constraints of a specific type."""
        return [
            mc for mc in self.meta_constraints
            if mc.metadata.get("type") == constraint_type
        ]

    def print_constraint_summary(self) -> None:
        """Print a summary of all constraint types."""
        from collections import Counter

        type_counts = Counter(mc.metadata.get("type") for mc in self.meta_constraints)

        print("\n" + "=" * 80)
        print("CONSTRAINT SUMMARY")
        print("=" * 80)
        for ctype, count in sorted(type_counts.items()):
            print(f"  {ctype:30s}: {count:4d} constraints")
        print(f"  {'TOTAL':30s}: {len(self.meta_constraints):4d} constraints")
        print("=" * 80 + "\n")


def check_feasibility_cpmpy(
    problem: GlobalLogisticProblem,
    time_limit: int = 10,
    verbose: bool = True,
    enforce_no_dead_ends: bool = True,
    minimize_active_nodes: bool = False,
    enforce_exact_sourcing: bool = True,
) -> dict:
    """
    Quick feasibility check using CPMpy with explainability support.

    Args:
        problem: The GlobalLogisticProblem instance
        time_limit: Maximum time in seconds
        verbose: If True, prints detailed logs
        enforce_no_dead_ends: If True, adds constraints to prevent orphaned parts
        minimize_active_nodes: If True, minimizes the number of active (part, loc) nodes
        enforce_exact_sourcing: If True, enforces EXACT sourcing requirement (==), not just minimum (>=)

    Returns:
        Dictionary with is_feasible, solution, solver, and message
    """
    if verbose:
        logger.info("\n" + "=" * 80)
        logger.info("CPMPY GRAPH-BASED FEASIBILITY ANALYSIS")
        logger.info("=" * 80)

    solver = CpmpyGraphBasedFeasibilitySolver(
        problem,
        enforce_no_dead_ends=enforce_no_dead_ends,
        minimize_active_nodes=minimize_active_nodes,
        enforce_exact_sourcing=enforce_exact_sourcing
    )
    solver.init_model()

    if verbose:
        solver.print_constraint_summary()

    # Solve
    results = solver.solve(time_limit=time_limit)

    if len(results) > 0:
        solution = results[0][0]
        is_feasible = problem.satisfy(solution, verbose=verbose)

        if is_feasible:
            if verbose:
                logger.info("\n✓ FEASIBLE - Found valid production allocation")

                # Count active nodes
                active_parts = sum(1 for locs in solution.part_to_locations.values() if locs)
                total_value = sum(
                    part.value_added
                    for part, locs in solution.part_to_locations.items()
                    for loc in locs
                )

                logger.info(f"  Active (part, location) assignments: {active_parts}")
                logger.info(f"  Total value-added: {total_value:.0f}")
                logger.info(f"  Transport routes: {len(solution.transport_plan)}")

            return {
                'is_feasible': True,
                'solution': solution,
                'solver': solver,
                'message': "Found feasible allocation"
            }
        else:
            if verbose:
                logger.warning("\n⚠️  SOLVER FOUND SOLUTION BUT VIOLATED CONSTRAINTS")

            return {
                'is_feasible': False,
                'solution': solution,
                'solver': solver,
                'message': "Solution violates problem constraints"
            }
    else:
        if verbose:
            logger.error("\n❌ INFEASIBLE - No valid production allocation exists")
            logger.info("\nTo debug infeasibility, use:")
            logger.info("  solver.explain_unsat_meta() - Get MUS (Minimal Unsatisfiable Subset)")
            logger.info("  solver.correct_unsat_meta() - Get MCS (Minimal Correction Set)")

        return {
            'is_feasible': False,
            'solution': None,
            'solver': solver,
            'message': "No feasible allocation found within time limit"
        }


def explain_infeasibility(
    solver: CpmpyGraphBasedFeasibilitySolver,
    method: str = "mus",
    verbose: bool = True,
) -> list[MetaCpmpyConstraint]:
    """
    Explain why a problem is infeasible by computing MUS.

    Args:
        solver: The solver instance (after calling solve() with INFEASIBLE result)
        method: Method to use ("mus", "quickxplain", "optimal_mus", etc.)
        verbose: If True, prints detailed explanation

    Returns:
        List of meta-constraints causing infeasibility
    """
    from discrete_optimization.generic_tools.cpmpy_tools import CpmpyExplainUnsatMethod

    method_map = {
        "mus": CpmpyExplainUnsatMethod.mus,
        "quickxplain": CpmpyExplainUnsatMethod.quickxplain,
        "optimal_mus": CpmpyExplainUnsatMethod.optimal_mus,
        "core": CpmpyExplainUnsatMethod.core,
    }
    cpmpy_method = method_map.get(method, CpmpyExplainUnsatMethod.mus)
    minimal_unsat = solver.explain_unsat_meta(cpmpy_method=cpmpy_method,
                                              solver="ortools")
    if verbose:
        print("\n" + "=" * 80)
        print(f"INFEASIBILITY EXPLANATION ({method.upper()})")
        print("=" * 80)
        print(f"Found {len(minimal_unsat)} conflicting constraints:\n")

        for i, mc in enumerate(minimal_unsat, 1):
            print(f"{i}. {mc.name}")
            print(f"   Type: {mc.metadata.get('type', 'unknown')}")
            for key, value in mc.metadata.items():
                if key != 'type':
                    print(f"   {key}: {value}")
            print()

        print("=" * 80)

    return minimal_unsat