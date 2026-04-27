"""
Graph-Based Feasibility Solver for Global Logistics.

Uses the (part, location) dependency graph to model production allocation
as a path selection problem. Much simpler than flow-based models.

Key idea:
- Nodes = (part, location) pairs
- Edges = dependency + reachability (child can reach parent via transport)
- Variables = select which (part, location) nodes are active
- Constraints = if node is active, at least one predecessor must be active
"""

import logging
from typing import Any, Dict, Tuple

from ortools.sat.python.cp_model import CpModel, CpSolverSolutionCallback

from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver
from logistic.network_design.problem import Location, Part, GlobalLogisticSolution, GlobalLogisticProblem
from logistic.network_design.solvers.graph_analysis import LogisticProblemGraphAnalysis

logger = logging.getLogger(__name__)


class GraphBasedFeasibilitySolver(OrtoolsCpSatSolver):
    """
    CP-SAT solver using the (part, location) dependency graph.

    Instead of modeling individual transport flows, models the problem as:
    - Select nodes in the (part, location) graph
    - Ensure dependency constraints: if (parent, loc) is selected,
      then for each child, at least one (child, child_loc) predecessor must be selected

    This automatically handles:
    - Production capability
    - Transport reachability (encoded in graph edges)
    - Assembly dependencies
    """

    def __init__(
        self,
        problem: GlobalLogisticProblem,
        enforce_no_dead_ends: bool = True,
        minimize_active_nodes: bool = False,
        **kwargs: Any,
    ):
        super().__init__(problem=problem, params_objective_function=None, **kwargs)
        self.problem: GlobalLogisticProblem = problem
        self.enforce_no_dead_ends = enforce_no_dead_ends
        self.minimize_active_nodes = minimize_active_nodes

        # Build the dependency graph
        logger.info("Building (part, location) dependency graph...")
        self.graph_analyzer = LogisticProblemGraphAnalysis(problem)
        self.dependency_graph = self.graph_analyzer.get_part_and_location_graph()

        logger.info(f"  Nodes: {self.dependency_graph.number_of_nodes()}")
        logger.info(f"  Edges: {self.dependency_graph.number_of_edges()}")

        # Variables: x[(part, loc)] = whether this (part, loc) is selected
        self.var_x: Dict[Tuple[Part, Location], Any] = {}

    def init_model(self, **kwargs: Any) -> None:
        """Initialize the graph-based feasibility model."""
        logger.info("="*80)
        logger.info("Initializing Graph-Based Feasibility Model")
        logger.info("="*80)

        self.cp_model = CpModel()

        # ==========================================
        # 1. Variables
        # ==========================================
        logger.info("\nCreating variables...")

        # Create variable for each node in the graph
        for (part, loc) in self.dependency_graph.nodes():
            self.var_x[(part, loc)] = self.cp_model.NewBoolVar(f"x_{part.name}_{loc.name}")

        logger.info(f"  Variables: {len(self.var_x)}")

        # ==========================================
        # 2. Constraints
        # ==========================================
        logger.info("\nAdding constraints...")

        # 2a. Sourcing requirements
        # Each part must be produced at >= sourcing_requirement locations
        for part in self.problem.parts:
            part_vars = [
                self.var_x[(p, loc)] for (p, loc) in self.var_x.keys()
                if p == part
            ]
            if part_vars:
                self.cp_model.Add(sum(part_vars) >= part.sourcing_requirement)

        logger.info(f"  Added sourcing requirement constraints for {len(self.problem.parts)} parts")

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
                        self.cp_model.Add(sum(predecessors) >= 1).OnlyEnforceIf(parent_var)
                        constraint_count += 1
                    else:
                        # No valid predecessor for this child - forbid producing parent here
                        self.cp_model.Add(parent_var == 0)
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
                        self.cp_model.Add(sum(usage_vars) >= 1).OnlyEnforceIf(child_var)
                        no_dead_end_count += 1
                    else:
                        # No possible use - forbid producing here
                        self.cp_model.Add(child_var == 0)
                        logger.debug(
                            f"  Forbid {part.name} at {loc.name}: "
                            f"no parent consumes it and no outbound path"
                        )

            logger.info(f"  Added {no_dead_end_count} no-dead-ends constraints")

        # 2d. Country work-share constraints
        if self.problem.country_constraints:
            logger.info(f"  Adding {len(self.problem.country_constraints)} country constraints...")

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
                    if loc.min_value_added is not None:
                        self.cp_model.Add(sum(terms) >= int(loc.min_value_added))
                    if loc.max_value_added is not None:
                        self.cp_model.Add(sum(terms) <= int(loc.max_value_added))

        # 3. Objective (optional)
        # Minimize number of active (part, location) nodes to avoid orphans
        if self.minimize_active_nodes:
            logger.info("\nAdding objective: minimize active nodes...")
            objective_expr = sum(self.var_x.values())
            self.cp_model.Minimize(objective_expr)
            logger.info("  Objective: minimize sum of x[(part, loc)] variables")

        logger.info("✓ Graph-based model initialized")
        logger.info("="*80 + "\n")

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> GlobalLogisticSolution:
        """Extract the production allocation from selected graph nodes."""
        part_to_locations = {part: set() for part in self.problem.parts}

        for (part, loc), x_var in self.var_x.items():
            if cpsolvercb.Value(x_var) == 1:
                part_to_locations[part].add(loc)

        # Reconstruct transport plan from graph edges using cached routes
        # Strategy: for each active edge in the dependency graph, add its precomputed transport
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
                    if successor_node in self.var_x and cpsolvercb.Value(self.var_x[successor_node]) == 1:
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


def check_feasibility_graph_based(
    problem: GlobalLogisticProblem,
    time_limit: int = 10,
    verbose: bool = True,
    enforce_no_dead_ends: bool = True,
    minimize_active_nodes: bool = False,
) -> Dict:
    """
    Quick feasibility check using the graph-based approach.

    Args:
        problem: The GlobalLogisticProblem instance
        time_limit: Maximum time in seconds
        verbose: If True, prints detailed logs
        enforce_no_dead_ends: If True, adds constraints to prevent orphaned parts
        minimize_active_nodes: If True, minimizes the number of active (part, loc) nodes

    Returns:
        Dictionary with is_feasible, solution, and message
    """
    if verbose:
        logger.info("\n" + "="*80)
        logger.info("GRAPH-BASED FEASIBILITY ANALYSIS")
        logger.info("="*80)

    solver = GraphBasedFeasibilitySolver(
        problem,
        enforce_no_dead_ends=enforce_no_dead_ends,
        minimize_active_nodes=minimize_active_nodes
    )
    solver.init_model()

    # Solve
    results = solver.solve(time_limit=time_limit,
                           ortools_cpsat_solver_kwargs=dict(log_search_progress=True))

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
                'message': "Found feasible allocation"
            }
        else:
            if verbose:
                logger.warning("\n⚠️  SOLVER FOUND SOLUTION BUT VIOLATED CONSTRAINTS")

            return {
                'is_feasible': False,
                'solution': solution,
                'message': "Solution violates problem constraints"
            }
    else:
        if verbose:
            logger.error("\n❌ INFEASIBLE - No valid production allocation exists")

        return {
            'is_feasible': False,
            'solution': None,
            'message': "No feasible allocation found within time limit"
        }
