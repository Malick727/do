import logging
from typing import Dict, List, Any

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.sequential_metasolver import SequentialMetasolver
from discrete_optimization.generic_tools.study import SolverConfig
from multibatching.solvers.kh.src.datamodel import Route, Product
from multibatching.solvers.kh.src.pricers.base_pricer import BasePricer
from multibatching.solvers.milp_flow import GurobiMultibatchingSolver, GurobiMultibatchingSolverUnitFlow
from multibatching.problem import MultibatchingProblem, MultibatchingSolution
from multibatching.solvers.packing_subproblem import GreedyPackingForMultibatching
from multibatching.solvers.two_steps import TwoStepMultibatchingSolver

logger = logging.getLogger(__name__)


@BasePricer.register("heuristic_mip")
class HeuristicMIPPricer(BasePricer):
    """
    A heuristic pricer that runs a full MIP solver (Gurobi) ONCE at the beginning
    to find a high-quality integer solution, extracts the routes/batches used,
    and feeds them as columns to the Master Problem.
    """

    # Static injection point for the original problem
    # Set this before calling solve(): HeuristicMIPPricer.problem_instance = my_problem
    problem_instance: MultibatchingProblem = None

    def __init__(self, constraints, products, links, routes, settings, mode):
        super().__init__(constraints, products, links, routes, settings, mode)
        self.already_run = False

        # Pre-index routes for fast lookup: (OriginName, DestName, ResourceName) -> Route
        self.route_map = {}
        for r in self.routes:
            key = (r.origin.name, r.destination.name, r.transport_resource.name)
            self.route_map[key] = r

        # Pre-index products: ID -> Product
        self.product_map = {p.id: p for p in self.products}
        from multibatching.solvers.kh.src.pricers.pricer_scip import ScipKnapsackPricer
        from multibatching.solvers.kh.src.pricers.pricer_greedy import GreedyPricer

        self.other_pricer = GreedyPricer(constraints, products, links, routes, settings,
                                         mode)

    def find_patterns(self, duals: Dict) -> List[Dict[str, Any]]:
        """
        Runs the MIP solver once, extracts patterns, and returns them.
        Subsequent calls return an empty list.
        """
        if self.already_run:
            return self.other_pricer.find_patterns(duals)

        if self.problem_instance is None:
            logger.error("HeuristicMIPPricer.problem_instance is not set! Cannot run Gurobi.")
            self.already_run = True
            return []
        logger.info(">>> Running Heuristic MIP (Gurobi) to generate initial columns...")
        found_patterns = []
        # 1. Solve the full problem using Gurobi
        # You can adjust time_limit or other kwargs here
        config = SolverConfig(cls=SequentialMetasolver,
                                          kwargs=dict(list_subbricks=[
                                              SubBrick(cls=TwoStepMultibatchingSolver,
                                                       kwargs={"flow_solver":
                                                                   SubBrick(cls=GurobiMultibatchingSolver,
                                                                            kwargs={"time_limit": 20}),
                                                               "packing_solver":
                                                                   SubBrick(cls=GreedyPackingForMultibatching,
                                                                            kwargs={})}),
                                              SubBrick(GurobiMultibatchingSolverUnitFlow,
                                                       dict(time_limit=30,
                                                            delta_to_solution=2),
                                                       kwargs_from_solution={"solution": lambda sol: sol}
                                                       )]
                                          )
                                          )
        solver = config.cls(self.problem_instance, **config.kwargs)
        solver.init_model(**config.kwargs)
        # solve
        result_storage = solver.solve(**config.kwargs)
        if not result_storage:
            logger.warning("Gurobi found no solution.")
            self.already_run = True
            return []
        for sol, _ in result_storage:
            best_sol: MultibatchingSolution = sol
            logger.info(f"Gurobi found solution with objective: {self.problem_instance.evaluate(best_sol)}")
            # 2. Extract Patterns (Batches) from the Solution
            # best_sol.list_flows contains PackingTransport objects
            for flow in best_sol.list_flows:
                # Skip empty flows
                if flow.nb_packing <= 1e-6:
                    continue
                # A. Match to Student Route
                # DO Link -> Student Route
                l1 = flow.transport_link.location_l1.name  # or id
                l2 = flow.transport_link.location_l2.name  # or id
                res_name = flow.transport_link.transport_type.name

                # Note: Ensure naming consistency between problem.py and datamodel.py conversion
                # If names are reliable:
                route_key = (str(l1), str(l2), str(res_name))
                route = self.route_map.get(route_key)

                if not route:
                    logger.warning(f"Could not map flow {route_key} to a Student Route.")
                    continue

                # B. Convert Packing
                # DO Packing: {DO_Product: qty} -> Student Packing: {Student_Product: qty}
                student_pattern = {}
                for do_prod, qty in flow.product_packing.items():
                    if qty > 1e-6:
                        s_prod = self.product_map.get(do_prod.id)
                        if s_prod:
                            student_pattern[s_prod] = float(qty)

                if not student_pattern:
                    continue

                # C. Calculate Costs (Helper method from BasePricer logic usually needed)
                # We can use the route data directly
                # duration = distance / speed
                duration = route.distance / route.transport_resource.speed if route.transport_resource.speed > 0 else 0

                # Fixed cost = distance * cost
                # (Note: Logic might vary, check CostCalculator in logic.py if available)
                fixed_cost = 0.0
                if self.settings.enable_transport:
                    fixed_cost += route.distance * route.transport_resource.cost
                if self.settings.enable_co2:
                    # co2_costs is usually per ton/unit? adjusting based on wrapper logic
                    # wrapper: settings.co2_costs passed to StudentSettings
                    # StudentSettings stores it.
                    # CostCalculator logic: emission_cost = dist * emissions * (co2_costs / 1e6)
                    fixed_cost += route.distance * route.transport_resource.emissions * (
                                self.settings.co2_costs)

                # D. Add Pattern
                # Note: 'nb_packing' from Gurobi indicates how many times this truck runs.
                # The Pricer returns unique *column types*. The Master Problem decides the quantity (lambda).
                # We just add the pattern definition.

                # Check for duplicates to avoid adding the same truck config multiple times
                # (Simple check, exact dict match might be needed)

                pattern_entry = {
                    'route': route,
                    'pattern': student_pattern,
                    'duration': duration,
                    'fixed_cost': fixed_cost
                }

                found_patterns.append(pattern_entry)

        self.already_run = True
        logger.info(f"Injecting {len(found_patterns)} columns from Gurobi solution.")
        if self.options.get("add_single_batch", True):
            found_patterns.extend(self.generate_initial_patterns())
        return found_patterns