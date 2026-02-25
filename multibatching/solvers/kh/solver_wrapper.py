import logging
from collections import defaultdict, Counter
from typing import Any, Optional, List, Dict, Type

# discrete-optimization library imports
from discrete_optimization.generic_tools.callbacks.callback import Callback, CallbackList
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from multibatching.solvers.kh.src.pricers.pricer_template import TemplatePricer
from multibatching.utils import generate_multibatching_problem

# Fallback if running from package root
from multibatching.solvers.kh.src.master_problem import MasterProblem
from multibatching.solvers.kh.src.datamodel import (
    Settings as StudentSettings,
    Location as StudentLocation,
    Product as StudentProduct,
    TransportLink as StudentLink,
    TransportResource as StudentResource,
    Route as StudentRoute
)
from multibatching.solvers.kh.src.pricers.base_pricer import BasePricer
from multibatching.solvers.kh.src.pricers.pricer_greedy import GreedyPricer
from multibatching.problem import MultibatchingProblem, MultibatchingSolution, PackingTransport

logger = logging.getLogger(__name__)

class MultibatchingWrapper(SolverDO):

    def solve(self, callbacks: Optional[list[Callback]] = None, **kwargs: Any) -> ResultStorage:
        res = self.create_result_storage([])
        sol = MultibatchingSolution()
        fit = self.aggreg_from_sol(sol)
        res.append((sol, fit))
        return ResultStorage


class MultibatchingMasterWrapper(SolverDO):
    """
    Wrapper for the MasterProblem (Column Generation) solver.
    Reconstructs valid MultibatchingSolutions by reconciling
    generated patterns (capacity) with balanced flows (conservation).
    """
    problem: MultibatchingProblem

    def __init__(self,
                 problem: MultibatchingProblem,
                 params_objective_function=None,
                 **kwargs):
        super().__init__(problem, params_objective_function, **kwargs)
        self.model: Optional[MasterProblem] = None
        self.student_settings: Optional[StudentSettings] = None

        # Internal Mappings
        self.loc_id_to_student: Dict[Any, StudentLocation] = {}
        self.prod_id_to_student: Dict[int, StudentProduct] = {}
        self.student_prod_to_orig: Dict[StudentProduct, Any] = {}
        self.link_to_student_link: Dict[Any, StudentLink] = {}
        self.student_route_to_orig_link: Dict[StudentRoute, Any] = {}

    def init_model(self, **kwargs: Any) -> None:
        """
        Translates MultibatchingProblem data into the student's datamodel.
        """
        # 1. Settings
        self.student_settings = StudentSettings(
            co2_costs=kwargs.get("co2_costs", 1),
            capital_costs=1, #self.problem.capital_factor,
            enable_transport=kwargs.get("enable_transport", True),
            enable_co2=kwargs.get("enable_co2", True),
            enable_interest=kwargs.get("enable_interest", False),
            big_m=kwargs.get("big_m", 10**9),
            epsilon=1e-06,
            print_cutoff=1e-5
        )
        # 2. Locations
        for loc in self.problem.locations:
            self.loc_id_to_student[loc.id] = StudentLocation(name=str(loc.name))
        # 3. Products
        student_products = []
        for p in self.problem.products:
            # Map net supply using student location objects
            net_flow = {self.loc_id_to_student[loc.id]: loc.net_supply[p]
                        for loc in self.problem.locations if p in loc.net_supply}
            sp = StudentProduct(
                id=p.id,
                name=p.name,
                size=float(p.size),
                value=float(p.value),
                valid_transport_resources=[str(tt.name) for tt in p.valid_transports
                                           if p.size <= tt.capacity],
                net_supply_demand=net_flow
            )
            self.prod_id_to_student[p.id] = sp
            self.student_prod_to_orig[sp] = p
            student_products.append(sp)

        # 4. Links and Routes
        student_links = []
        student_routes = []
        tr_map = {}
        sl_map = {}
        for tl in self.problem.transport_links:
            tt = tl.transport_type
            if tt.id not in tr_map:
                tr_map[tt.id] = StudentResource(
                    name=str(tt.name),
                    capacity=float(tt.capacity),
                    cost=float(tt.cost),
                    emissions=float(tt.emissions),
                    speed=float(tt.speed)
                )
            if (self.loc_id_to_student[tl.location_l1.id],
                self.loc_id_to_student[tl.location_l2.id]) in sl_map:
                sl = sl_map[(self.loc_id_to_student[tl.location_l1.id],
                             self.loc_id_to_student[tl.location_l2.id])]
            else:
                sl = StudentLink(
                    origin=self.loc_id_to_student[tl.location_l1.id],
                    destination=self.loc_id_to_student[tl.location_l2.id]
                )
                student_links.append(sl)
                sl_map[(self.loc_id_to_student[tl.location_l1.id],
                        self.loc_id_to_student[tl.location_l2.id])] = sl
            self.link_to_student_link[tl] = sl
            sr = StudentRoute(
                transport_link=sl,
                transport_resource=tr_map[tt.id],
                distance=float(tl.distance)
            )
            student_routes.append(sr)
            self.student_route_to_orig_link[sr] = tl
        # 5. Build Master Problem
        self.model = MasterProblem(
            settings=self.student_settings,
            locations=list(self.loc_id_to_student.values()),
            products=student_products,
            links=student_links,
            routes=student_routes
        )
        self.model.build_model()

    def solve(self,
              callbacks: Optional[List[Callback]] = None,
              pricer_class: Optional[Type[BasePricer]] = None,
              mode: str = "ip",
              time_limit: int = 300,
              **kwargs: Any) -> ResultStorage:

        if self.model is None:
            self.init_model(**kwargs)

        if pricer_class is None:
            raise ValueError("A pricer_class must be provided for Column Generation.")

        callbacks_list = CallbackList(callbacks=callbacks)
        callbacks_list.on_solve_start(solver=self)

        # Execute student solve
        self.model.solve(
            pricer_class=pricer_class,
            original_problem=self.problem,
            mode=mode,
            time_limit=time_limit,
            export_lp=kwargs.get("export_lp", True)
        )
        self.model.print_results()
        # Retrieve result using the exact logic from 'print_results'
        result_storage = self.create_result_storage([])
        if self.model.model.getNSols() > 0:
            sol = self.retrieve_solution_like_print_results(mode)
            fit = self.aggreg_from_sol(sol)
            result_storage.append((sol, fit))

        callbacks_list.on_solve_end(res=result_storage, solver=self)
        return result_storage

    def retrieve_solution_like_print_results(self, mode="ip") -> MultibatchingSolution:
        """
        Reconstructs the solution by greedily filling the active patterns (capacity)
        with the exact amount of balanced flow (demand) found by the solver.
        This avoids overestimation by trimming 'full' patterns down to what is actually needed.
        """
        list_flows = []
        epsilon = self.student_settings.epsilon

        # --- 1. Identify Available Capacity (Patterns) ---
        # Key: Original TransportLink -> List of {'nb': int, 'cap': dict}
        link_capacity_groups = defaultdict(list)

        if self.model.pricer and hasattr(self.model.pricer, 'pattern_registry'):
            for var_name, data in self.model.pricer.pattern_registry.items():
                val = self.model.model.getVal(data['variable'])
                if val > epsilon:
                    # Number of trucks of this type
                    nb_packing = int(round(val))

                    route = data['route']
                    orig_link = self.student_route_to_orig_link.get(route)

                    if orig_link:
                        # Map student products to original products for the capacity map
                        capacity_map = {}
                        for sp, qty in data['pattern'].items():
                            if qty > 0:
                                capacity_map[self.student_prod_to_orig[sp]] = int(qty)

                        link_capacity_groups[orig_link].append({
                            'nb': nb_packing,
                            'cap': capacity_map
                        })

        # --- 2. Distribute Flow into Capacity ---
        for tl, groups in link_capacity_groups.items():
            student_link = self.link_to_student_link[tl]

            # A. Retrieve Target Flow (The Truth from Flow Vars)
            target_flows = {}
            for p in self.problem.products:
                sp = self.prod_id_to_student[p.id]
                f_var = self.model.flow_vars.get((student_link, sp))
                if f_var:
                    f_val = self.model.model.getVal(f_var)
                    if f_val > epsilon:
                        target_flows[p] = int(round(f_val))
            if not target_flows:
                continue
            # B. Greedy Fill
            # Iterate through each group of identical trucks
            for group in groups:
                nb_trucks = group['nb']
                pattern_cap = group['cap']
                # Continue while we have trucks in this group and flow to assign
                while nb_trucks > 0:
                    # Determine the load for the next truck(s)
                    # It cannot exceed pattern capacity OR remaining demand
                    current_load = {}
                    has_load = False
                    for p, cap in pattern_cap.items():
                        needed = target_flows.get(p, 0)
                        if needed > 0:
                            take = min(cap, needed)
                            current_load[p] = take
                            has_load = True
                    if not has_load:
                        # This truck group cannot carry any remaining needed product
                        # (e.g., it carries P2 but we only need P1 now)
                        break
                        # How many trucks can strictly follow this 'current_load' configuration?
                    # We are limited by the number of available trucks (nb_trucks)
                    # AND by how many times we can repeat this load before exhausting a product's demand.
                    n_repeats = nb_trucks
                    for p, amt in current_load.items():
                        needed = target_flows[p]
                        # How many times does 'amt' fit into 'needed'?
                        # e.g. needed=25, amt=10 -> can do 2 times (remaining 5)
                        can_do = needed // amt
                        n_repeats = min(n_repeats, can_do)
                    # Edge case: if n_repeats becomes 0 (e.g. needed < amt?), strictly it implies amt > needed.
                    # But 'take' = min(cap, needed), so 'amt' <= 'needed'.
                    # Thus 'can_do' >= 1. n_repeats should be >= 1.
                    if n_repeats < 1:
                        n_repeats = 1
                    # Register this batch
                    list_flows.append(PackingTransport(
                        transport_link=tl,
                        product_packing=current_load,
                        nb_packing=n_repeats
                    ))
                    # Update State
                    nb_trucks -= n_repeats
                    for p, amt in current_load.items():
                        target_flows[p] -= amt * n_repeats

        return MultibatchingSolution(problem=self.problem, list_flows=list_flows)

def main():
    #problem = generate_multibatching_problem(num_locations=5,
    #                                         num_transport_types=5,
    #                                         num_products=3,
    #                                         min_dist=1,
    #                                         max_dist=5,
    #                                         seed=51)
    #multibatching_problem_to_json(problem, 80, "example_3.json")
    from examples_do_usecases.multibatching.solve_industrial_use_case import parse_problem
    problem = parse_problem(False, False)
    solver = MultibatchingMasterWrapper(problem=problem)
    from multibatching.solvers.kh.src.pricers.pricer_scip import ScipKnapsackPricer
    from multibatching.solvers.kh.src.pricers.pricer_do import HeuristicMIPPricer
    results = solver.solve(pricer_class=HeuristicMIPPricer, mode="ip", time_limit=600)
    if results:
        sol: MultibatchingSolution = results[-1][0]
        print(f"Satisfy : {problem.satisfy(sol)}")
        print(f"Objective: {problem.evaluate(sol)}")
        print(problem.check_flows(sol))
        for loc in problem.locations:
            print(loc.id, loc.net_supply)
        for packing in sol.list_flows:
            print(packing.transport_link.location_l1.name, "->",
                  packing.transport_link.location_l2.name,
                  "pack", packing.product_packing, "frequency", packing.nb_packing)

import json
def multibatching_problem_to_json(problem: "MultibatchingProblem",
                                  co2_costs: float = 0.0,
                                  output_path: str = None) -> Dict[str, Any]:
    """
    Translates a MultibatchingProblem instance into the specified JSON format.

    Args:
        problem: The MultibatchingProblem instance.
        co2_costs: The cost of CO2 to be put in settings (not present in problem object).
        output_path: If provided, writes the JSON to this file path.

    Returns:
        The dictionary representing the JSON structure.
    """

    # 1. Settings
    # capital_factor corresponds to capitalCosts
    data = {
        "settings": {
            "co2Costs": co2_costs,
            "capitalCosts": problem.capital_factor
        },
        "locations": {},
        "transportResources": {},
        "products": {},
        "routes": {}
    }

    # Helper map to ensure consistent ID usage (using name if available, else str(id))
    loc_id_map = {loc.id: (loc.name if loc.name else str(loc.id)) for loc in problem.locations}
    # For Transport Types, the example uses "TR1", "TR2". We use name or fallback to ID.
    tr_id_map = {tt.id: (tt.name if tt.name else str(tt.id)) for tt in problem.transport_types}
    prod_id_map = {p.id: (p.name if p.name else str(p.id)) for p in problem.products}

    # 2. Locations
    for loc in problem.locations:
        l_key = loc_id_map[loc.id]
        data["locations"][l_key] = {
            "name": loc.name if loc.name else str(loc.id)
        }

    # 3. Transport Resources (from TransportTypes)
    for tt in problem.transport_types:
        t_key = tr_id_map[tt.id]
        data["transportResources"][t_key] = {
            "name": tt.name if tt.name else str(tt.id),
            "capacity": tt.capacity,
            "co2Emissions": tt.emissions,
            "cost": tt.cost,
            "speed": tt.speed
        }

    # 4. Products
    for p in problem.products:
        p_key = prod_id_map[p.id]

        # Valid Transport Resources
        valid_tr_ids = [tr_id_map[tt.id] for tt in p.valid_transports]

        # Invert Net Supply: Location -> Product  ==>  Product -> Location map
        net_supply_demand = {}
        for loc in problem.locations:
            val = loc.net_supply.get(p, 0)
            if val != 0:
                net_supply_demand[loc_id_map[loc.id]] = val

        data["products"][p_key] = {
            "name": p.name if p.name else str(p.id),
            "validTR": valid_tr_ids,
            "size": p.size,
            "value": p.value,
            "netSupplyDemand": net_supply_demand
        }

    # 5. Routes (Aggregating TransportLinks by Origin-Destination)
    # The JSON structure groups multiple transport modes under one O-D route entry.
    routes_agg = defaultdict(dict)  # (from_id, to_id) -> { tr_id: {distance: x} }

    for link in problem.transport_links:
        from_id = loc_id_map[link.location_l1.id]
        to_id = loc_id_map[link.location_l2.id]
        tr_id = tr_id_map[link.transport_type.id]

        # Store distance for this specific resource on this route
        routes_agg[(from_id, to_id)][tr_id] = {
            "distance": link.distance
        }

    for (from_id, to_id), resources in routes_agg.items():
        # Construct a route key, e.g., "L1-L2"
        route_key = f"{from_id}-{to_id}"

        data["routes"][route_key] = {
            "from": from_id,
            "to": to_id,
            "transportResources": resources
        }

    # Optional: Write to file
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)

    return data

# flow milp : {'transport': 2424916.1399999997,
# 'emission': 2861.429459214601,
# 'capital': 0.0}
#+9.79899074338331e+06
#Objective: {'transport': 3932708.1593999993, 'emission': 5865244.560799302, 'capital': 0.0}

if __name__ == "__main__":
    main()