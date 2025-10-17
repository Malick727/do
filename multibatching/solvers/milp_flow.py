# In a new file, e.g., multibatching/solvers/gurobi_solver.py

import logging
import math
from collections import defaultdict, Counter
from typing import Any, Callable, Dict

import gurobipy
from discrete_optimization.generic_tools.callbacks.callback import Callback, CallbackList
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.lp_tools import GurobiMilpSolver, ParametersMilp
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage

from multibatching.problem import (
    MultibatchingProblem,
    MultibatchingSolution,
    PackingTransport,
    Product,
    TransportLink,
)
from multibatching.solvers.packing_subproblem import GreedyPackingForMultibatching

logger = logging.getLogger(__name__)


class GurobiMultibatchingSolver(GurobiMilpSolver):
    """Gurobi solver for the Multibatching problem, based on a flow formulation.

    This solver supports two main modeling variants:
    1.  **Standard Flow Model**: Allows multiple different products to be packed
        into the same transport (multi-batching).
    2.  **Single Batching Flow Model**: A more constrained version where each transport
        can only carry a single type of product.

    Attributes:
        problem (MultibatchingProblem): The multibatching problem instance to solve.
        variables (Dict): A dictionary to store Gurobi variables created for the model.
        single_batching (bool): Flag indicating which model variant to use.
        scaling_factor (int): A factor to scale floating-point costs into integers for the model.
    """

    problem: MultibatchingProblem

    def __init__(self, problem: MultibatchingProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)
        self.variables: Dict[str, Any] = {}
        self.single_batching: bool = False
        self.scaling_factor = kwargs.get("scaling_factor", 1000)

    def init_model(self, single_batching: bool = False, **kwargs: Any) -> None:
        """
        Initializes the Gurobi model by dispatching to the appropriate
        model-building method based on the `single_batching` flag.
        """
        self.single_batching = single_batching
        if self.single_batching:
            self._init_model_flows_single_batching(**kwargs)
        else:
            self._init_model_flows_multi_batching(**kwargs)

    def _init_model_flows_multi_batching(self, **kwargs: Any) -> None:
        """Builds the standard flow model where multiple products can share a transport."""
        logger.info("Initializing Gurobi model (Multi-Batching Flow)...")
        model = gurobipy.Model("multibatching-flow")

        # --- Pre-computation ---
        supply_per_product = {p: self.problem.get_total_supply(p) for p in self.problem.products}

        # --- Variables ---
        nb_trips = {
            i: model.addVar(vtype=gurobipy.GRB.INTEGER, name=f"nb_trips_{i}")
            for i in range(self.problem.nb_transport_links)
        }
        flows = {
            i: {
                p_idx: model.addVar(
                    vtype=gurobipy.GRB.INTEGER,
                    ub=supply_per_product[p],
                    name=f"flow_{i}_{p_idx}",
                )
                for p_idx, p in enumerate(self.problem.products)
                if tl.transport_type in p.valid_transports and tl.transport_type.capacity > p.size
            }
            for i, tl in enumerate(self.problem.transport_links)
        }

        # --- Constraints ---
        # Capacity Constraint
        for i, tl in enumerate(self.problem.transport_links):
            total_volume = gurobipy.quicksum(
                flows[i][p_idx] * p.size
                for p_idx, p in enumerate(self.problem.products)
                if p_idx in flows[i]
            )
            model.addLConstr(total_volume <= nb_trips[i] * tl.transport_type.capacity)

        # Flow Conservation
        for loc in self.problem.locations:
            for p_idx, p in enumerate(self.problem.products):
                in_flow = gurobipy.quicksum(
                    flows[i][p_idx]
                    for i, tl in enumerate(self.problem.transport_links)
                    if tl.location_l2 == loc and p_idx in flows[i]
                )
                out_flow = gurobipy.quicksum(
                    flows[i][p_idx]
                    for i, tl in enumerate(self.problem.transport_links)
                    if tl.location_l1 == loc and p_idx in flows[i]
                )
                model.addLConstr(in_flow + loc.net_supply.get(p, 0) == out_flow)

        # --- Objective ---
        self._set_objective(model, nb_trips, flows)

        self.model = model
        self.variables = {"nb_trips": nb_trips, "flows": flows}

    def _init_model_flows_single_batching(self, **kwargs: Any) -> None:
        """Builds the constrained flow model where only one product type is allowed per transport."""
        logger.info("Initializing Gurobi model (Single-Batching Flow)...")
        model = gurobipy.Model("multibatching-single-flow")

        # --- Pre-computation ---
        supply_per_product = {p: self.problem.get_total_supply(p) for p in self.problem.products}

        # --- Variables ---
        nb_trips = {i: model.addVar(vtype=gurobipy.GRB.INTEGER, name=f"nb_trips_{i}") for i in
                    range(self.problem.nb_transport_links)}
        flows = {
            i: {
                p_idx: model.addVar(vtype=gurobipy.GRB.INTEGER, ub=supply_per_product[p], name=f"flow_{i}_{p_idx}")
                for p_idx, p in enumerate(self.problem.products) if tl.transport_type in p.valid_transports
                and tl.transport_type.capacity > p.size
            } for i, tl in enumerate(self.problem.transport_links)
        }
        # Trips dedicated to a specific product on a specific link
        nb_trips_per_product = {
            (i, p_idx): model.addVar(vtype=gurobipy.GRB.INTEGER, name=f"nb_trips_{i}_{p_idx}")
            for i in range(self.problem.nb_transport_links) for p_idx in flows[i]
        }

        # --- Constraints ---
        for i, tl in enumerate(self.problem.transport_links):
            # Total trips on a link is the sum of product-dedicated trips
            model.addLConstr(gurobipy.quicksum(nb_trips_per_product[(i, p_idx)] for p_idx in flows[i]) == nb_trips[i])

            for p_idx, p in enumerate(self.problem.products):
                if p_idx in flows[i]:
                    # Capacity constraint for single-product trips
                    products_per_trip = tl.transport_type.capacity // p.size if p.size > 0 else 0
                    if products_per_trip > 0:
                        model.addLConstr(flows[i][p_idx] <= nb_trips_per_product[(i, p_idx)] * products_per_trip)
                    else:  # Cannot transport this product
                        model.addLConstr(flows[i][p_idx] == 0)

        # Flow conservation (same as multi-batching)
        for loc in self.problem.locations:
            for p_idx, p in enumerate(self.problem.products):
                in_flow = gurobipy.quicksum(flows[i][p_idx] for i, tl in enumerate(self.problem.transport_links) if
                                            tl.location_l2 == loc and p_idx in flows[i])
                out_flow = gurobipy.quicksum(flows[i][p_idx] for i, tl in enumerate(self.problem.transport_links) if
                                             tl.location_l1 == loc and p_idx in flows[i])
                model.addLConstr(in_flow + loc.net_supply.get(p, 0) == out_flow)

        # --- Objective ---
        self._set_objective(model, nb_trips, flows)

        self.model = model
        self.variables = {"nb_trips": nb_trips, "flows": flows, "nb_trips_per_product": nb_trips_per_product}

    def _set_objective(self, model, nb_trips, flows):
        """Helper function to build and set the objective expression."""
        transport_cost = gurobipy.quicksum(
            nb_trips[i] * int(tl.distance * tl.transport_type.cost * self.scaling_factor)
            for i, tl in enumerate(self.problem.transport_links)
        )
        emission_cost = gurobipy.quicksum(
            nb_trips[i] * int(tl.distance * tl.transport_type.emissions * self.scaling_factor)
            for i, tl in enumerate(self.problem.transport_links)
        )
        capital_cost = gurobipy.quicksum(
            flows[i][p_idx] * p.value * ((1 + self.problem.capital_factor) ** (
                        tl.distance / tl.transport_type.speed) - 1) * self.scaling_factor
            for i, tl in enumerate(self.problem.transport_links) for p_idx, p in enumerate(self.problem.products) if
            p_idx in flows[i]
        )
        model.setObjective(transport_cost + emission_cost + capital_cost, sense=gurobipy.GRB.MINIMIZE)

    def retrieve_current_solution(
            self,
            get_var_value_for_current_solution: Callable[[Any], float],
            get_obj_value_for_current_solution: Callable[[], float],
    ) -> Solution:
        """
        Converts a Gurobi solution into a `MultibatchingSolution`.
        If `single_batching` is used, it reconstructs the exact packings. Otherwise, it creates
        an "average" packing for multi-batch flows.
        """
        list_flows = []
        nb_trips_vals = {i: get_var_value_for_current_solution(var) for i, var in self.variables["nb_trips"].items()}

        for i, tl in enumerate(self.problem.transport_links):
            if nb_trips_vals.get(i, 0) > 1e-6:
                if self.single_batching:
                    # Reconstruct exact packings for single-batching
                    for p_idx, p in enumerate(self.problem.products):
                        if (i, p_idx) in self.variables["nb_trips_per_product"]:
                            flow_val = get_var_value_for_current_solution(self.variables["flows"][i][p_idx])
                            if flow_val > 1e-6:
                                products_per_trip = tl.transport_type.capacity // p.size
                                num_full_trips = int(flow_val // products_per_trip)
                                remainder = flow_val % products_per_trip

                                if num_full_trips > 0:
                                    list_flows.append(PackingTransport(
                                        transport_link=tl, product_packing={p: products_per_trip},
                                        nb_packing=num_full_trips
                                    ))
                                if remainder > 1e-6:
                                    list_flows.append(PackingTransport(
                                        transport_link=tl, product_packing={p: remainder}, nb_packing=1
                                    ))
                else:
                    # Create average packing for multi-batching
                    product_packing = {}
                    for p_idx, p in enumerate(self.problem.products):
                        if p_idx in self.variables["flows"][i]:
                            flow_val = get_var_value_for_current_solution(self.variables["flows"][i][p_idx])
                            if flow_val > 1e-6:
                                product_packing[p] = flow_val / nb_trips_vals[i]
                    if product_packing:
                        list_flows.append(PackingTransport(
                            transport_link=tl, product_packing=product_packing, nb_packing=int(round(nb_trips_vals[i]))
                        ))
        return MultibatchingSolution(problem=self.problem, list_flows=list_flows)

    def convert_to_variable_values(self, solution: MultibatchingSolution) -> Dict[gurobipy.Var, float]:
        """Converts a `MultibatchingSolution` to a dictionary of Gurobi variable values for warm start."""
        variable_values = {}

        # Aggregate flows and trips from the solution
        agg_flows = defaultdict(lambda: defaultdict(float))
        agg_trips = defaultdict(float)
        agg_trips_per_product = defaultdict(float)

        for flow in solution.list_flows:
            link_idx = self.problem.transport_links_to_index[flow.transport_link]
            agg_trips[link_idx] += flow.nb_packing
            for product, units in flow.product_packing.items():
                prod_idx = self.problem.product_to_index[product]
                agg_flows[link_idx][prod_idx] += units * flow.nb_packing
                if self.single_batching:
                    agg_trips_per_product[(link_idx, prod_idx)] += flow.nb_packing

        # Map aggregated values to Gurobi variables
        for i, var in self.variables["nb_trips"].items():
            variable_values[var] = agg_trips.get(i, 0)
        for i, flow_map in self.variables["flows"].items():
            for p_idx, var in flow_map.items():
                variable_values[var] = agg_flows.get(i, {}).get(p_idx, 0)

        if self.single_batching:
            for (i, p_idx), var in self.variables["nb_trips_per_product"].items():
                variable_values[var] = agg_trips_per_product.get((i, p_idx), 0)

        return variable_values


# In your milp_flow.py file or a new one

# ... (keep existing imports and the GurobiMultibatchingSolver class) ...

class GurobiMultibatchingSolverUnitFlow(GurobiMilpSolver):
    """
    Gurobi solver for the Multibatching problem using a detailed "unit flow" formulation.

    This model creates variables for each potential trip on each transport link and
    decides the exact content of each trip. It is more precise than an aggregated
    flow model but can be more computationally expensive due to the large number
    of binary and integer variables.

    This formulation is analogous to the `UNIT_FLOW` model in the Cpsat solver.
    """
    problem: MultibatchingProblem

    def __init__(self, problem: MultibatchingProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)
        self.variables: Dict[str, Any] = {}
        self.scaling_factor = kwargs.get("scaling_factor", 1)

    def init_model(self, max_trips_per_link: int = 30, **kwargs: Any) -> None:
        """
        Builds the Gurobi model based on a detailed, per-trip ("unit flow") formulation.

        Args:
            max_trips_per_link (int): A hard upper bound on the number of trips that can be
                                      used on any single transport link. This is necessary
                                      to define the variable space.
        """
        logger.info("Initializing Gurobi model (Unit Flow)...")
        model = gurobipy.Model("multibatching-unit-flow")

        # --- Decision Variables ---
        # `used_trips[link_idx][trip_idx]` is a binary var, 1 if the trip is used
        used_trips = {
            i: {
                j: model.addVar(vtype=gurobipy.GRB.BINARY, name=f"used_{i}_{j}")
                for j in range(max_trips_per_link)
            } for i in range(self.problem.nb_transport_links)
        }

        # `trip_contents[link_idx][trip_idx][prod_idx]` = quantity of a product on a specific trip
        trip_contents = {
            i: {
                j: {
                    p_idx: model.addVar(
                        vtype=gurobipy.GRB.INTEGER,
                        name=f"cont_{i}_{j}_{p_idx}",
                        ub=int(tl.transport_type.capacity / p.size) if p.size > 0 else 0
                    )
                    for p_idx, p in enumerate(self.problem.products)
                    if tl.transport_type in p.valid_transports and tl.transport_type.capacity > p.size
                } for j in range(max_trips_per_link)
            } for i, tl in enumerate(self.problem.transport_links)
        }

        # --- Constraints ---
        for i, tl in enumerate(self.problem.transport_links):
            # Symmetry breaking: force trips to be used in order (trip j+1 can be used only if trip j is)
            #for j in range(max_trips_per_link - 1):
            #    model.addLConstr(used_trips[i][j+1] <= used_trips[i][j])

            for j in range(max_trips_per_link):
                # Capacity constraint for each individual trip
                trip_volume = gurobipy.quicksum(
                    trip_contents[i][j][p_idx] * p.size
                    for p_idx, p in enumerate(self.problem.products)
                    if p_idx in trip_contents[i][j]
                )
                model.addConstr(trip_volume <= tl.transport_type.capacity)
                # Link `used_trips` to trip contents. A trip is used if it carries something.
                total_items_in_trip = gurobipy.quicksum(
                    trip_contents[i][j][p_idx] for p_idx in trip_contents[i][j]
                )
                # IF used_trips[i][j] is 1, THEN total_items > 0 (use >= 1 for integers)

                model.addGenConstrIndicator(
                    used_trips[i][j], True, total_items_in_trip >= 1
                )
                # IF used_trips[i][j] is 0, THEN total_items == 0
                model.addGenConstrIndicator(
                    used_trips[i][j], False, total_items_in_trip == 0
                )

        # Aggregate expressions for total flow, linking trip contents to flow conservation
        total_flows = {
            i: {
                p_idx: gurobipy.quicksum(
                    trip_contents[i][j][p_idx] for j in range(max_trips_per_link)
                )
                for p_idx, p in enumerate(self.problem.products) if p_idx in trip_contents[i][0]
            } for i in range(self.problem.nb_transport_links)
        }

        # Flow conservation constraints (using the aggregated total_flows)
        for loc in self.problem.locations:
            for p_idx, p in enumerate(self.problem.products):
                in_flow = gurobipy.quicksum(
                    total_flows[i].get(p_idx, 0)
                    for i, tl in enumerate(self.problem.transport_links) if tl.location_l2 == loc
                )
                out_flow = gurobipy.quicksum(
                    total_flows[i].get(p_idx, 0)
                    for i, tl in enumerate(self.problem.transport_links) if tl.location_l1 == loc
                )
                model.addLConstr(in_flow + loc.net_supply.get(p, 0) == out_flow)

        # --- Objective Function ---
        nb_trips_per_link = {
            i: gurobipy.quicksum(used_trips[i][j] for j in range(max_trips_per_link))
            for i in range(self.problem.nb_transport_links)
        }
        transport_cost = gurobipy.quicksum(
            nb_trips_per_link[i] * int(tl.distance * tl.transport_type.cost * self.scaling_factor)
            for i, tl in enumerate(self.problem.transport_links)
        )
        emission_cost = gurobipy.quicksum(
            nb_trips_per_link[i] * int(tl.distance * tl.transport_type.emissions * self.scaling_factor)
            for i, tl in enumerate(self.problem.transport_links)
        )

        # Capital cost is quadratic (binary * integer). Gurobi handles this.
        capital_cost = gurobipy.quicksum(
            used_trips[i][j] * gurobipy.quicksum(
                trip_contents[i][j][p_idx] * p.value
                for p_idx, p in enumerate(self.problem.products) if p_idx in trip_contents[i][j]
            ) * ((1 + self.problem.capital_factor)**(tl.distance / tl.transport_type.speed) - 1) * self.scaling_factor
            for i, tl in enumerate(self.problem.transport_links) for j in range(max_trips_per_link)
        )
        model.setObjective(transport_cost + emission_cost + capital_cost, sense=gurobipy.GRB.MINIMIZE)

        self.model = model
        self.variables = {
            "used_trips": used_trips,
            "trip_contents": trip_contents,
        }

    def retrieve_current_solution(
        self,
        get_var_value_for_current_solution: Callable[[Any], float],
        get_obj_value_for_current_solution: Callable[[], float],
    ) -> Solution:
        """
        Retrieves the solution by reconstructing packings from the detailed trip variables.
        It groups identical packings to create a concise solution.
        """
        list_flows = []
        for i, link in enumerate(self.problem.transport_links):
            packings_on_link = []
            for j in range(len(self.variables["used_trips"][i])):
                if get_var_value_for_current_solution(self.variables["used_trips"][i][j]) > 0.5:
                    current_packing = {
                        self.problem.products[p_idx]: round(
                            get_var_value_for_current_solution(content_var)
                        )
                        for p_idx, content_var in self.variables["trip_contents"][i][j].items()
                        if get_var_value_for_current_solution(content_var) > 0.5
                    }
                    if current_packing:
                        # Use a frozenset of items for a hashable representation of the packing
                        packings_on_link.append(frozenset(current_packing.items()))

            if packings_on_link:
                # Count occurrences of each unique packing configuration
                for unique_packing_items, count in Counter(packings_on_link).items():
                    list_flows.append(PackingTransport(
                        transport_link=link,
                        product_packing=dict(unique_packing_items),
                        nb_packing=count,
                    ))
        return MultibatchingSolution(problem=self.problem, list_flows=list_flows)

    def convert_to_variable_values(self, solution: MultibatchingSolution) -> Dict[gurobipy.Var, float]:
        # Implementation for warm start would be complex and is omitted for brevity,
        # but would involve un-grouping packings and assigning them to trip variables.
        logger.warning("Warm start for the Unit Flow Gurobi model is not yet implemented.")
        return {}

class GurobiMultibatchingSolverWithLazyConstraint(GurobiMultibatchingSolver):
    """
    A Gurobi solver for the multi-batching problem that uses lazy constraints
    to iteratively refine the packing model.
    """

    def init_model(self, **kwargs: Any) -> None:
        """
        Initializes the standard multi-batching flow model and enables lazy constraints.
        """
        # We only apply this logic to the multi-batching case
        super().init_model(single_batching=False, **kwargs)
        # Tell Gurobi to enable lazy constraints
        self.model.setParam("LazyConstraints", 1)
        logger.info("Gurobi model with Lazy Constraints enabled.")

    def solve(
            self,
            callbacks: list[Callback] = None,
            parameters_milp: ParametersMilp = None,
            time_limit: float = 30,
            **kwargs: Any,
    ) -> ResultStorage:
        """
        Overrides the default solve method to use a custom callback for generating
        lazy constraints.
        """
        self.early_stopping_exception = None
        self._current_internal_objective_best_value = None
        self._current_internal_objective_best_bound = None

        callbacks_list = CallbackList(callbacks=callbacks)
        callbacks_list.on_solve_start(solver=self)

        # Use our specialized callback for lazy constraints
        gurobi_lazy_callback = GurobiLazyCallback(do_solver=self, callback=callbacks_list)

        self.optimize_model(
            parameters_milp=parameters_milp,
            time_limit=time_limit,
            gurobi_callback=gurobi_lazy_callback,
            **kwargs,
        )

        if hasattr(self.model, "ObjVal"):
            self._current_internal_objective_best_value = self.model.ObjVal
        if hasattr(self.model, "ObjBound"):
            self._current_internal_objective_best_bound = self.model.ObjBound

        res = gurobi_lazy_callback.res
        callbacks_list.on_solve_end(res=res, solver=self)

        return res



class GurobiLazyCallback:
    """
    Custom Gurobi callback to generate valid, flow-dependent cuts for the multibatching problem.
    """

    def __init__(self, do_solver: GurobiMultibatchingSolver, callback: Callback):
        self.do_solver = do_solver
        self.callback = callback
        self.res = do_solver.create_result_storage()
        self.nb_solutions = 0
        self.nb_cuts_added = 0

    def __call__(self, model: gurobipy.Model, where: int) -> None:
        if where == gurobipy.GRB.Callback.MIPSOL:
            try:
                # 1. Retrieve the candidate solution from the MILP
                candidate_solution = self.do_solver.retrieve_current_solution(
                    get_var_value_for_current_solution=model.cbGetSolution,
                    get_obj_value_for_current_solution=lambda: model.cbGet(gurobipy.GRB.Callback.MIPSOL_OBJ),
                )

                # 2. Run the greedy packing subproblem to get the "true" number of trips
                packer = GreedyPackingForMultibatching(problem=self.do_solver.problem)
                packer.init_from_solution(solution=candidate_solution)
                packing_result = packer.solve()

                if len(packing_result) == 0:
                    return

                packed_solution = packing_result.get_best_solution_fit()[0]

                # 3. Check for violations and add valid, flow-dependent lazy constraints
                milp_trips = defaultdict(int)
                for flow in candidate_solution.list_flows:
                    milp_trips[flow.transport_link] += flow.nb_packing

                greedy_trips = defaultdict(int)
                for flow in packed_solution.list_flows:
                    greedy_trips[flow.transport_link] += flow.nb_packing

                violation_found = False
                for link, milp_trip_count_float in milp_trips.items():
                    milp_trip_count = round(milp_trip_count_float)
                    greedy_trip_count = greedy_trips.get(link, 0)

                    if greedy_trip_count > milp_trip_count:
                        violation_found = True
                        self.nb_cuts_added += 1
                        link_idx = self.do_solver.problem.transport_links_to_index[link]
                        nb_trips_var = self.do_solver.variables["nb_trips"][link_idx]

                        # Calculate the total volume in the candidate solution for this link
                        candidate_volume = sum(
                            p.size * round(flow.product_packing.get(p, 0) * flow.nb_packing)
                            for p in self.do_solver.problem.products
                            for flow in candidate_solution.list_flows if flow.transport_link == link
                        )

                        if candidate_volume < 1e-6:
                            continue  # Avoid division by zero

                        # Define the total_volume_var as a Gurobi linear expression
                        total_volume_var = gurobipy.quicksum(
                            p.size * self.do_solver.variables["flows"][link_idx][p_idx]
                            for p_idx, p in enumerate(self.do_solver.problem.products)
                            if p_idx in self.do_solver.variables["flows"][link_idx]
                        )

                        # Add the valid, flow-dependent cut
                        # nb_trips >= (N_greedy / V_candidate) * total_volume_var
                        cut_coefficient = greedy_trip_count / candidate_volume
                        model.cbLazy(nb_trips_var >= cut_coefficient * total_volume_var)

                        logger.info(f"CUT #{self.nb_cuts_added} on link {link.id}: "
                                    f"MILP trips={milp_trip_count}, Greedy trips={greedy_trip_count}, Volume={candidate_volume:.2f}. "
                                    f"Adding lazy cut.")

                # 4. If no violation was found, the solution is valid from a packing perspective.
                if not violation_found:
                    fit = self.do_solver.aggreg_from_sol(packed_solution)
                    self.res.append((packed_solution, fit))
                    self.nb_solutions += 1

                    stopping = self.callback.on_step_end(
                        step=self.nb_solutions, res=self.res, solver=self.do_solver
                    )
                    if stopping:
                        model.terminate()

            except Exception as e:
                logger.error(f"Error in Gurobi lazy callback: {e}", exc_info=True)
                self.do_solver.early_stopping_exception = e
                model.terminate()