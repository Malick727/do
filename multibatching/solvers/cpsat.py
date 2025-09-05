import math
from collections import defaultdict, Counter
from typing import Any, Optional
from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import CpSolverSolutionCallback
from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.callbacks.callback import CallbackList
from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver, CpSolver, CpModel
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from discrete_optimization.generic_tools.do_solver import WarmstartMixin, BoundsProviderMixin
from multibatching.problem import (MultibatchingProblem, Location, TransportLink, TransportType, PackingTransport,
Product, MultibatchingSolution)
import logging
from enum import Enum
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import EnumHyperparameter, CategoricalHyperparameter
logger = logging.getLogger(__name__)


class ModelingMultiBatch(Enum):
    FLOW = 0
    UNIT_FLOW = 1


class CpsatMultibatchingSolver(OrtoolsCpSatSolver, WarmstartMixin):

    hyperparameters = [EnumHyperparameter(name="modeling", enum=ModelingMultiBatch,
                                          default=ModelingMultiBatch.FLOW),
                       CategoricalHyperparameter(name="detailed_capacity_constraint",
                                                 choices=[True, False],
                                                 default=False,
                                                 depends_on=[("modeling", [ModelingMultiBatch.FLOW])])]
    problem: MultibatchingProblem

    def __init__(self, problem: MultibatchingProblem, **kwargs: Any):

        super().__init__(problem, **kwargs)
        self.variables = {}
        self.modeling: ModelingMultiBatch = None
        self.verbose_logs = kwargs.get("verbose", 0)
        self.scaling_factor = kwargs.get("scaling_factor", 10**3)

    def set_warm_start_from_prev_solve(self):
        if self.solver is not None:
            self.cp_model.ClearHints()
            response = self.solver.ResponseProto()  # Get the raw response
            for i in range(len(response.solution)):
                var = self.cp_model.GetIntVarFromProtoIndex(i)
                # print(f"Variable {var} = {response.solution[i]}")
                self.cp_model.AddHint(var, response.solution[i])

    def set_warm_start(self, solution: MultibatchingSolution) -> None:
        if self.modeling == ModelingMultiBatch.FLOW:
            self.set_warm_start_flow(solution=solution)

    def set_warm_start_flow(self, solution: MultibatchingSolution) -> None:
        aggreg_flows_solution = {}
        nb_trips = {}
        self.cp_model.ClearHints()
        for flow in solution.list_flows:
            ind = self.problem.transport_links_to_index[flow.transport_link]
            if ind not in aggreg_flows_solution:
                aggreg_flows_solution[ind] = defaultdict(lambda: 0)
                nb_trips[ind] = 0
            for p in flow.product_packing:
                aggreg_flows_solution[ind][self.problem.product_to_index[p]] += (
                    int(flow.product_packing[p]*flow.nb_packing))
            nb_trips[ind] += flow.nb_packing
        for index_link in self.variables["flows"]:
            for index_product in self.variables["flows"][index_link]:
                self.cp_model.AddHint(self.variables["flows"][index_link][index_product],
                                      aggreg_flows_solution.get(index_link, {}).get(index_product, 0))
            self.cp_model.AddHint(self.variables["nb_trips"][index_link], nb_trips.get(index_link, 0))

    def logs_solution(self, cpsolvercb: CpSolverSolutionCallback) -> Solution:
        if 'objs' in self.variables:
            for obj in self.variables["objs"]:
                logger.info(f"{obj}: {cpsolvercb.value(self.variables['objs'][obj])}")
        for index_tl in self.variables["nb_trips"]:
            tl = self.problem.transport_links[index_tl]
            nb_trips = cpsolvercb.value(self.variables['nb_trips'][index_tl])
            if nb_trips > 0:
                logger.info(f"Nb trips on {tl.location_l1, tl.location_l2, tl.transport_type.id}, :"
                            f"{cpsolvercb.value(self.variables['nb_trips'][index_tl])}")
                for index_product in self.variables["flows"][index_tl]:
                    val = cpsolvercb.value(self.variables['flows'][index_tl][index_product])
                    if val > 0:
                        logger.info(f"Nb product {self.problem.products[index_product].id} "
                                    f"on {tl.location_l1, tl.location_l2, tl.transport_type.id}, :"
                                    f"{cpsolvercb.value(self.variables['flows'][index_tl][index_product])}")
            if self.modeling == ModelingMultiBatch.UNIT_FLOW:
                for i in self.variables["trips_variables"][index_tl]:
                    used = self.variables["used_trips"][index_tl][i]
                    if cpsolvercb.value(used) == 1:
                        logger.info(f"{i}-th trip used on route {tl.location_l1, tl.location_l2, tl.transport_type.id}")
                        for index_product in self.variables["trips_variables"][index_tl][i]:
                            val = cpsolvercb.value(self.variables['trips_variables'][index_tl][i][index_product])
                            if val > 0:
                                logger.info(f"Nb product {self.problem.products[index_product].id} "
                                            f"on {tl.location_l1, tl.location_l2, tl.transport_type.id}, :"
                                            f"{val}")
                        logger.info(f"------")
            if nb_trips > 0:
                logger.info(f"------End of transport link------")

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> Solution:
        if self.verbose_logs > 0:
            self.logs_solution(cpsolvercb)
        packing_solution = []
        products = self.problem.products
        for index_tl in self.variables["flows"]:
            nb = cpsolvercb.value(self.variables["nb_trips"][index_tl])
            if nb > 0:
                packings = []
                if self.modeling == ModelingMultiBatch.UNIT_FLOW:
                    for i in sorted(self.variables["trips_variables"][index_tl]):
                        used = self.variables["used_trips"][index_tl][i]
                        if cpsolvercb.value(used) > 0:
                            current_packing = set()
                            for index_product in self.variables["trips_variables"][index_tl][i]:
                                val = cpsolvercb.value(self.variables['trips_variables'][index_tl][i][index_product])
                                if val > 0:
                                    current_packing.add((index_product, int(val)))
                            cpack = frozenset(current_packing)
                            packings.append(cpack)
                    if len(packings) > 0:
                        c = Counter(packings)
                        for key, val in c.items():
                            packing_transport = PackingTransport(transport_link=self.problem.transport_links[index_tl],
                                                                 product_packing={products[x[0]]: x[1] for x in key},
                                                                 nb_packing=val)
                            packing_solution.append(packing_transport)
                else:
                    cur_packing = {}
                    for index_product in self.variables["flows"][index_tl]:
                        val = cpsolvercb.value(self.variables['flows'][index_tl][index_product])
                        if val > 0:
                            cur_packing[products[index_product]] = val/nb
                    if len(cur_packing) > 0:
                        packing_solution.append(PackingTransport(transport_link=self.problem.transport_links[index_tl],
                                                                 product_packing=cur_packing,
                                                                 nb_packing=nb))
        return MultibatchingSolution(problem=self.problem, list_flows=packing_solution)

    def init_model(self, **args: Any) -> None:
        args = self.complete_with_default_hyperparameters(args)
        modeling = args["modeling"]
        if modeling == ModelingMultiBatch.FLOW:
            self.init_model_flows(**args)
        if modeling == ModelingMultiBatch.UNIT_FLOW:
            self.init_model_detailed_trips(**args)
        self.modeling = modeling

    def init_model_flows(self, **args: Any) -> None:
        model = cp_model.CpModel()
        transport_links = self.problem.transport_links
        flows_variables = {}
        nb_trips_per_link = {}
        keys_inflow_location_product = dict()
        keys_outflow_location_product = dict()
        for index_tl in range(self.problem.nb_transport_links):
            flows_variables[index_tl] = {}
            tl = transport_links[index_tl]
            loc1 = tl.location_l1
            loc2 = tl.location_l2
            if loc1.id not in keys_outflow_location_product:
                keys_outflow_location_product[loc1.id] = defaultdict(lambda: set())
            if loc2.id not in keys_inflow_location_product:
                keys_inflow_location_product[loc2.id] = defaultdict(lambda: set())
            tr: TransportType = tl.transport_type
            nb_trips_per_link[index_tl] = model.NewIntVar(lb=0, ub=10000000,
                                                          name=f'num_trips_{index_tl}')
            list_var_weight = []
            # for index_product in range(self.problem.nb_products):
            for index_product in range(self.problem.nb_products):
                product = self.problem.products[index_product]
                if tr in product.valid_transports:
                    flows_variables[index_tl][index_product] = model.NewIntVar(lb=0,
                                                                               ub=10000000,
                                                                               name=f"flow_{index_tl}_{index_product}")
                    keys_outflow_location_product[loc1.id][product.id].add((index_tl, index_product))
                    keys_inflow_location_product[loc2.id][product.id].add((index_tl, index_product))
                    list_var_weight.append((flows_variables[index_tl][index_product],
                                            int(product.size)))

            # Global capacity constraint
            model.Add(cp_model.LinearExpr.weighted_sum([x[0] for x in list_var_weight],
                                                       [x[1] for x in list_var_weight])
                      <= nb_trips_per_link[index_tl]*tl.transport_type.capacity)
        if args["detailed_capacity_constraint"]:
            nb_trips_min_for_product_flows = {}
            for index_product in range(self.problem.nb_products):
                product = self.problem.products[index_product]
                size_product = product.size
                for tl in range(self.problem.nb_transport_types):
                    tt: TransportType = self.problem.transport_types[tl]
                    capa_transport = tt.capacity
                    nb_trips_min_for_product_flows[(index_product, tt)] = [0]
                    current_greedy_bins = defaultdict(lambda: 0)
                    current_bin = 0
                    current_greedy_bins[0] = 0
                    for i in range(30):
                        if current_greedy_bins[current_bin] + size_product <= capa_transport:
                            current_greedy_bins[current_bin] += size_product
                        else:
                            current_bin += 1
                            current_greedy_bins[current_bin] = size_product
                        nb_trips_min_for_product_flows[(index_product, tt)].append(current_bin+1)
            for index_tl in flows_variables:
                transport_link = self.problem.transport_links[index_tl]
                transport_type = transport_link.transport_type
                for index_product in flows_variables[index_tl]:
                    nb_trips_min = nb_trips_min_for_product_flows[(index_product, transport_type)]
                    binary_indicators = [model.NewBoolVar(f'flow_{index_tl}_{index_product}_equal_{k}')
                                         for k in range(len(nb_trips_min))]
                    model.add_map_domain(flows_variables[index_tl][index_product],
                                         binary_indicators)
                    for j in range(1, len(binary_indicators)):
                        (model.add(nb_trips_per_link[index_tl] >= nb_trips_min[j])
                         .OnlyEnforceIf(binary_indicators[j]))

        # Flows constraint
        for location in self.problem.locations:
            loc_id = location.id
            debug_mode = args.get("debug_mode", False)
            index_products = args.get("debug_settings", {}).get("index_products", range(len(self.problem.products)))
            for index_product in index_products:
                product = self.problem.products[index_product]
                net_supply = location.net_supply.get(product, 0)
                flow_in = cp_model.LinearExpr.Sum([flows_variables[x[0]][x[1]]
                                                   for x in
                                                   keys_inflow_location_product.get(loc_id, {}).get(product.id,
                                                                                                    set())])
                flow_out = cp_model.LinearExpr.Sum([flows_variables[x[0]][x[1]]
                                                    for x in
                                                    keys_outflow_location_product.get(loc_id, {}).get(product.id,
                                                                                                      set())])
                model.Add(flow_in+net_supply-flow_out == 0)
        transport_cost = self.scaling_factor*sum([nb_trips_per_link[index_tl]
                                                  * int(transport_links[index_tl].distance
                                                        * transport_links[index_tl].transport_type.cost)
                                                  for index_tl in range(self.problem.nb_transport_links)])
        emission_cost = self.scaling_factor*sum([nb_trips_per_link[index_tl]
                                                 * int(transport_links[index_tl].distance
                                                       * transport_links[index_tl].transport_type.emissions)
                                                 for index_tl in range(self.problem.nb_transport_links)])
        estimated_compound = []
        for index_tl in range(self.problem.nb_transport_links):
            tl = self.problem.transport_links[index_tl]
            time = tl.distance/tl.transport_type.speed
            value = sum(flows_variables[index_tl][index_product]*int(self.problem.products[index_product].value)
                        for index_product in flows_variables[index_tl])
            # estimated_compound.append(value*int(self.scaling_factor*((1+self.problem.capital_factor)**time-1)))
            estimated_compound.append(0)
        self.variables["total_obj"] = transport_cost+emission_cost+sum(estimated_compound)
        model.Minimize(self.variables["total_obj"])
        self.variables["objs"] = {"transport": transport_cost,
                                  "emission": emission_cost,
                                  "capital": sum(estimated_compound)}
        self.cp_model = model
        self.variables["flows"] = flows_variables
        self.variables["nb_trips"] = nb_trips_per_link

    def init_model_flows_relaxed(self, **args: Any) -> None:
        model = cp_model.CpModel()
        transport_links = self.problem.transport_links
        flows_variables = {}
        nb_trips_per_link = {}
        keys_inflow_location_product = dict()
        keys_outflow_location_product = dict()
        for index_tl in range(self.problem.nb_transport_links):
            flows_variables[index_tl] = {}
            tl = transport_links[index_tl]
            loc1 = tl.location_l1
            loc2 = tl.location_l2
            if loc1.id not in keys_outflow_location_product:
                keys_outflow_location_product[loc1.id] = defaultdict(lambda: set())
            if loc2.id not in keys_inflow_location_product:
                keys_inflow_location_product[loc2.id] = defaultdict(lambda: set())
            tr: TransportType = tl.transport_type
            nb_trips_per_link[index_tl] = model.NewIntVar(lb=0, ub=100,
                                                          name=f'num_trips_{index_tl}')
            list_var_weight = []
            # for index_product in range(self.problem.nb_products):
            for index_product in range(self.problem.nb_products):
                product = self.problem.products[index_product]
                if tr in product.valid_transports:
                    flows_variables[index_tl][index_product] = model.NewIntVar(lb=0,
                                                                               ub=100,
                                                                               name=f"flow_{index_tl}_{index_product}")
                    keys_outflow_location_product[loc1.id][product.id].add((index_tl, index_product))
                    keys_inflow_location_product[loc2.id][product.id].add((index_tl, index_product))
                    list_var_weight.append((flows_variables[index_tl][index_product],
                                            int(product.size)))

            # Global capacity constraint
            model.Add(cp_model.LinearExpr.weighted_sum([x[0] for x in list_var_weight],
                                                       [x[1] for x in list_var_weight])
                      <= nb_trips_per_link[index_tl]*tl.transport_type.capacity)
            used = model.NewBoolVar(f"tl_{index_tl}")
            model.add(sum([x[0] for x in list_var_weight])>0).OnlyEnforceIf(used)
            model.add(sum([x[0] for x in list_var_weight])==0).OnlyEnforceIf(used.Not())
            model.add(nb_trips_per_link[index_tl]>=1).OnlyEnforceIf(used)

        if args["detailed_capacity_constraint"]:
            nb_trips_min_for_product_flows = {}
            for index_product in range(self.problem.nb_products):
                product = self.problem.products[index_product]
                size_product = product.size
                for tl in range(self.problem.nb_transport_types):
                    tt: TransportType = self.problem.transport_types[tl]
                    capa_transport = tt.capacity
                    nb_trips_min_for_product_flows[(index_product, tt)] = [0]
                    current_greedy_bins = defaultdict(lambda: 0)
                    current_bin = 0
                    current_greedy_bins[0] = 0
                    for i in range(30):
                        if current_greedy_bins[current_bin] + size_product <= capa_transport:
                            current_greedy_bins[current_bin] += size_product
                        else:
                            current_bin += 1
                            current_greedy_bins[current_bin] = size_product
                        nb_trips_min_for_product_flows[(index_product, tt)].append(current_bin+1)
            for index_tl in flows_variables:
                transport_link = self.problem.transport_links[index_tl]
                transport_type = transport_link.transport_type
                for index_product in flows_variables[index_tl]:
                    nb_trips_min = nb_trips_min_for_product_flows[(index_product, transport_type)]
                    binary_indicators = [model.NewBoolVar(f'flow_{index_tl}_{index_product}_equal_{k}')
                                         for k in range(len(nb_trips_min))]
                    model.add_map_domain(flows_variables[index_tl][index_product],
                                         binary_indicators)
                    for j in range(1, len(binary_indicators)):
                        (model.add(nb_trips_per_link[index_tl] >= nb_trips_min[j])
                         .OnlyEnforceIf(binary_indicators[j]))

        # Flows constraint
        for location in self.problem.locations:
            loc_id = location.id
            for product in self.problem.products:
                net_supply = location.net_supply.get(product, 0)
                flow_in = cp_model.LinearExpr.Sum([flows_variables[x[0]][x[1]]
                                                   for x in
                                                   keys_inflow_location_product.get(loc_id, {}).get(product.id, set())])
                flow_out = cp_model.LinearExpr.Sum([flows_variables[x[0]][x[1]]
                                                   for x in
                                                   keys_outflow_location_product.get(loc_id, {}).get(product.id, set())])
                print(net_supply, flow_in, flow_out)
                model.Add(flow_in+net_supply-flow_out == 0)

        transport_cost = sum([nb_trips_per_link[index_tl]*int(self.scaling_factor*transport_links[index_tl].distance
                                                              * transport_links[index_tl].transport_type.cost)
                             for index_tl in range(self.problem.nb_transport_links)])
        emission_cost = sum([nb_trips_per_link[index_tl]*int(self.scaling_factor*transport_links[index_tl].distance
                                                             * transport_links[index_tl].transport_type.emissions)
                             for index_tl in range(self.problem.nb_transport_links)])
        estimated_compound = []
        for index_tl in range(self.problem.nb_transport_links):
            tl = self.problem.transport_links[index_tl]
            time = tl.distance/tl.transport_type.speed
            value = sum(flows_variables[index_tl][index_product]*int(self.problem.products[index_product].value)
                        for index_product in flows_variables[index_tl])
            estimated_compound.append(value*int(self.scaling_factor*((1+self.problem.capital_factor)**time-1)))
        self.variables["total_obj"] = transport_cost+emission_cost+sum(estimated_compound)
        model.Minimize(self.variables["total_obj"])
        self.cp_model = model
        self.variables["flows"] = flows_variables
        self.variables["nb_trips"] = nb_trips_per_link
        self.variables["objs"] = {"transport": transport_cost,
                                  "emission": emission_cost,
                                  "capital": sum(estimated_compound)}

    def init_model_detailed_trips(self, **args: Any) -> None:
        model = cp_model.CpModel()
        transport_links = self.problem.transport_links
        flows_variables = {}
        used_trips_variables = {}
        trips_variables = {}
        nb_trips_per_link = {}
        keys_inflow_location_product = dict()
        keys_outflow_location_product = dict()
        max_nb_trips_per_link = args.get("max_trips", 10)
        for index_tl in range(self.problem.nb_transport_links):
            flows_variables[index_tl] = {}
            trips_variables[index_tl] = {}
            used_trips_variables[index_tl] = {}
            tl = transport_links[index_tl]
            loc1 = tl.location_l1
            loc2 = tl.location_l2
            if loc1.id not in keys_outflow_location_product:
                keys_outflow_location_product[loc1.id] = defaultdict(lambda: set())
            if loc2.id not in keys_inflow_location_product:
                keys_inflow_location_product[loc2.id] = defaultdict(lambda: set())
            tr: TransportType = tl.transport_type
            max_trips_for_this_link = tl.max_trips
            if max_trips_for_this_link is None:
                max_trips_for_this_link = 10
            nb_trips_per_link[index_tl] = model.NewIntVar(lb=0, ub=max_trips_for_this_link,
                                                          name=f'num_trips_{index_tl}')
            list_var_weight = []
            for nb_trip in range(max_trips_for_this_link):
                trips_variables[index_tl][nb_trip] = {}
                # means "nb_trip-th" vehicle is used.
                used_trips_variables[index_tl][nb_trip] = model.NewBoolVar(name=f"used_{index_tl}_{nb_trip}")
            # Symmetry breaking
            for i in range(1, max_trips_for_this_link):
                # ensure that if used_trip[i] == 0 then used_trip[i+1] == 0
                model.Add(used_trips_variables[index_tl][i] <= used_trips_variables[index_tl][i-1])
            for index_product in range(self.problem.nb_products):
                product = self.problem.products[index_product]
                if tr in product.valid_transports:
                    flows_variables[index_tl][index_product] = model.NewIntVar(lb=0,
                                                                               ub=100,
                                                                               name=f"flow_{index_tl}_{index_product}")
                    keys_outflow_location_product[loc1.id][product.id].add((index_tl, index_product))
                    keys_inflow_location_product[loc2.id][product.id].add((index_tl, index_product))
                    list_var_weight.append((flows_variables[index_tl][index_product],
                                            int(product.size)))
                    for nb_trip in trips_variables[index_tl]:
                        trips_variables[index_tl][nb_trip][index_product] = (
                            model.NewIntVar(lb=0,
                                            ub=math.floor(tl.transport_type.capacity/product.size),
                                            name=f"flow_{index_tl}_{nb_trip}_{index_product}"))
                    # Total flows of product in this edge ==
                    # the sum of transport product on individual trips on this edge.
                    model.Add(sum([trips_variables[index_tl][nb_trip][index_product]
                                   for nb_trip in trips_variables[index_tl]]) ==
                              flows_variables[index_tl][index_product])
            for i in used_trips_variables[index_tl]:
                model.Add(sum([trips_variables[index_tl][i][p]
                              for p in trips_variables[index_tl][i]]) > 0).OnlyEnforceIf(
                    used_trips_variables[index_tl][i])

                model.Add(sum([trips_variables[index_tl][i][p]
                              for p in trips_variables[index_tl][i]]) == 0).OnlyEnforceIf(
                    used_trips_variables[index_tl][i].Not())
                (model.Add(sum([trips_variables[index_tl][i][p]*int(self.problem.products[p].size)
                                for p in trips_variables[index_tl][i]]) <= tl.transport_type.capacity)
                 .OnlyEnforceIf(used_trips_variables[index_tl][i]))

            # Global capacity constraint
            model.Add(cp_model.LinearExpr.weighted_sum([x[0] for x in list_var_weight],
                                                       [x[1] for x in list_var_weight])
                      <= nb_trips_per_link[index_tl] * tl.transport_type.capacity)
            model.Add(nb_trips_per_link[index_tl] == sum(used_trips_variables[index_tl][i]
                                                         for i in used_trips_variables[index_tl]))

        # Flows constraint
        for location in self.problem.locations:
            loc_id = location.id
            for product in self.problem.products:
                net_supply = location.net_supply.get(product, 0)
                flow_in = cp_model.LinearExpr.Sum([flows_variables[x[0]][x[1]]
                                                   for x in
                                                   keys_inflow_location_product.get(loc_id, {}).get(product.id, set())])
                flow_out = cp_model.LinearExpr.Sum([flows_variables[x[0]][x[1]]
                                                    for x in
                                                    keys_outflow_location_product.get(loc_id, {}).get(product.id,
                                                                                                      set())])
                model.Add(flow_in + net_supply - flow_out == 0)

        transport_cost = self.scaling_factor*sum([nb_trips_per_link[index_tl] * transport_links[index_tl].distance
                              * transport_links[index_tl].transport_type.cost
                              for index_tl in range(self.problem.nb_transport_links)])

        emission_cost = self.scaling_factor*sum([nb_trips_per_link[index_tl] * transport_links[index_tl].distance
                             * transport_links[index_tl].transport_type.emissions
                             for index_tl in range(self.problem.nb_transport_links)])

        nb_trips_cost = sum([used_trips_variables[index_tl][nb_trip]
                             for index_tl in used_trips_variables
                             for nb_trip in used_trips_variables[index_tl]])
        compound_cost = []
        for index_tl in range(self.problem.nb_transport_links):
            tl = transport_links[index_tl]
            time = tl.distance/tl.transport_type.speed
            for i in trips_variables[index_tl]:
                value = sum([trips_variables[index_tl][i][index_p]*self.problem.products[index_p].value
                             for index_p in trips_variables[index_tl][i]])
                compound_cost.append(value*int(self.scaling_factor*((1+self.problem.capital_factor)**time-1)))
        self.variables["total_obj"] = transport_cost + emission_cost + sum(compound_cost)
        model.Minimize(self.variables["total_obj"])
        self.cp_model = model
        self.variables["flows"] = flows_variables
        self.variables["nb_trips"] = nb_trips_per_link
        self.variables["used_trips"] = used_trips_variables
        self.variables["trips_variables"] = trips_variables


class CpsatPackingSubproblem(OrtoolsCpSatSolver):
    problem: MultibatchingProblem

    def __init__(self, problem: MultibatchingProblem, **kwargs: Any):

        super().__init__(problem, **kwargs)
        self.variables = {}
        # self.modeling: ModelingMultiBatch = None
        self.base_solution = None
        self.scaling_factor = kwargs.get("scaling_factor", 10**3)

    def init_model(self, solution: MultibatchingSolution, **args: Any) -> None:
        model = cp_model.CpModel()
        self.base_solution = solution
        transport_links = [x.transport_link for x in solution.list_flows]
        num_trips = [x.nb_packing for x in solution.list_flows]
        transport_link_to_index = {transport_links[i]: i for i in range(len(transport_links))}
        num_trips_dict = {x.transport_link: x.nb_packing for x in solution.list_flows}
        flow_per_link = {x.transport_link: {p: int(x.product_packing[p]*x.nb_packing) for p in x.product_packing}
                         for x in solution.list_flows}
        # --- Determine Max Trips per Link for Variable Indexing ---
        # This ensures we create enough variables for the maximum number of trips any link might have.
        max_trips_overall = 30
        # --- Decision Variables ---
        # packed_on_trip[link_obj][trip_idx][product_id]: Amount of product packed on this trip
        packed_on_trip = defaultdict(lambda: defaultdict(dict))
        # is_trip_used[link_obj][trip_idx]: Binary variable, 1 if trip is used, 0 otherwise
        is_trip_used = defaultdict(lambda: {})

        for link in transport_links:
            # Number of trips to actually attempt packing into for this specific link
            # Use the number of trips allocated by the main model.
            num_trips_for_this_link = num_trips_dict.get(link, 0)+10
            # Only create variables if trips are allocated to avoid unnecessary complexity
            if num_trips_for_this_link > 0:
                for t_idx in range(num_trips_for_this_link):
                    is_trip_used[link][t_idx] = model.NewBoolVar(f'is_trip_used_{transport_link_to_index[link]}_{t_idx}')
                    for product in flow_per_link[link]:
                        # Upper bound for product units in a single trip
                        ub_per_product = link.transport_type.capacity // product.size
                        packed_on_trip[link][t_idx][product] = model.NewIntVar(
                            0, ub_per_product,
                            f'packed_{transport_link_to_index[link]}_trip{t_idx}_P{product.id}'
                        )

        # --- Constraints ---

        for link in transport_links:
            # 1. Total Flow Fulfillment (per link, per product):
            for product in flow_per_link[link]:
                total_product_flow_on_link = flow_per_link[link][product]
                model.Add(sum(packed_on_trip[link][t_idx][product]
                              for t_idx in packed_on_trip[link]) == total_product_flow_on_link)
            # 2. Trip Capacity Constraint (per individual trip for this link):
            for t_idx in packed_on_trip[link]:
                # Total volume packed on this specific trip
                packed_volume_on_this_trip = sum(
                    product.size * packed_on_trip[link][t_idx][product] for product in packed_on_trip[link][t_idx]
                )
                # If trip is used, its volume must respect capacity. If not used, its volume must be 0.
                model.Add(packed_volume_on_this_trip <= link.transport_type.capacity)
                model.Add(packed_volume_on_this_trip > 0).OnlyEnforceIf(is_trip_used[link][t_idx])
                model.Add(packed_volume_on_this_trip == 0).OnlyEnforceIf(is_trip_used[link][t_idx].Not())

                # .OnlyEnforceIf(
                #    is_trip_used[link][t_idx]))
                # model.Add(packed_volume_on_this_trip == 0).OnlyEnforceIf(is_trip_used[link][t_idx].Not())
            # 4. Symmetry Breaking (per link): Force trips to be used sequentially
            # index_trip = sorted(packed_on_trip[link].keys())
            # for j in range(1, len(index_trip)):
            #    model.Add(is_trip_used[link][index_trip[j-1]] >= is_trip_used[link][index_trip[j]])

        # --- Objective (Feasibility Problem) ---
        # No explicit objective needed; solver just finds a feasible solution.
        # model.Minimize(0)
        compound_cost = []
        for link in transport_links:
            time = link.distance / link.transport_type.speed
            for i in packed_on_trip[link]:
                value = sum([packed_on_trip[link][i][p] * p.value
                             for p in packed_on_trip[link][i]])
                compound_cost.append(value * int(self.scaling_factor*((1 + self.problem.capital_factor) ** time - 1)))

        nb_trips_link = {link: sum(is_trip_used[link][i]
                                   for i in is_trip_used[link]) for link in is_trip_used}
        transport_cost = self.scaling_factor*sum([nb_trips_link[link] * link.distance
                                                  * link.transport_type.cost
                                                  for link in is_trip_used])

        emission_cost = self.scaling_factor*sum([nb_trips_link[link] * link.distance
                                                 * link.transport_type.emissions
                                                 for link in is_trip_used])
        model.Minimize(transport_cost+emission_cost+sum(compound_cost))
        self.cp_model = model
        self.variables["packed_on_trip"] = packed_on_trip
        self.variables["trip_used"] = is_trip_used
        self.variables["objs"] = {"transport": transport_cost,
                                  "emission": emission_cost,
                                  "capital": sum(compound_cost)}

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> Solution:
        logger.info(f"Obj {cpsolvercb.ObjectiveValue()}")
        if 'objs' in self.variables:
            for obj in self.variables["objs"]:
                if obj == "capital":
                    continue
                try:
                    logger.info(f"{obj}: {cpsolvercb.value(self.variables['objs'][obj])}")
                except Exception as e:
                    pass
        list_flows = []
        for link in self.variables["packed_on_trip"]:
            packings = []
            for i in sorted(self.variables["trip_used"][link]):
                used = self.variables["trip_used"][link][i]
                if cpsolvercb.value(used) > 0:
                    current_packing = set()
                    for product in self.variables["packed_on_trip"][link][i]:
                        val = cpsolvercb.value(self.variables["packed_on_trip"][link][i][product])
                        if val > 0:
                            current_packing.add((product, int(val)))
                    cpack = frozenset(current_packing)
                    packings.append(cpack)
            if len(packings) > 0:
                c = Counter(packings)
                for key, val in c.items():
                    packing_transport = PackingTransport(transport_link=link,
                                                         product_packing={x[0]: x[1] for x in key},
                                                         nb_packing=val)
                    list_flows.append(packing_transport)
        return MultibatchingSolution(problem=self.problem, list_flows=list_flows)

    def analyse_solution(self, new_solution: MultibatchingSolution):
        nb_trips_per_link_base_solution = {x.transport_link: x.nb_packing for x in self.base_solution.list_flows}
        flow_per_link = {x.transport_link: {p: int(x.product_packing[p] * x.nb_packing) for p in x.product_packing}
                         for x in self.base_solution.list_flows}
        nb_trips_per_link_new = defaultdict(lambda: 0)
        for flow in new_solution.list_flows:
            link = flow.transport_link
            nb_trips_per_link_new[link] += flow.nb_packing
        changes = []
        for link in nb_trips_per_link_new:
            if nb_trips_per_link_base_solution[link] != nb_trips_per_link_new[link]:
                changes.append((link,
                                flow_per_link[link],
                                nb_trips_per_link_new[link],
                                nb_trips_per_link_base_solution[link]))
        return changes


class CpsatBendersMultibatchingSolver(SolverDO):
    problem: MultibatchingProblem

    def __init__(self, problem: MultibatchingProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)
        self.variables = {}
        # self.modeling: ModelingMultiBatch = None
        self.base_solution = None
        self.solver_master: CpsatMultibatchingSolver = None
        self.solver_subproblem: CpsatPackingSubproblem = None
        self.nb_cuts = 0

    def init_model(self, **kwargs: Any) -> None:
        self.solver_master = CpsatMultibatchingSolver(problem=self.problem)
        self.solver_master.init_model(modeling=ModelingMultiBatch.FLOW, **kwargs)

    def update_model_master(self, changed: list[tuple[TransportLink, dict[Product, int],
                                                      int, int]], best_lb: int):
        for link, flows, new_nb_trip, prev_nb_trip in changed:
            index_link = self.problem.transport_links_to_index[link]
            indicators = []
            for all_index_link in [index_link]:
                for product in flows:
                    ind = self.solver_master.cp_model.NewBoolVar(f"flow_{all_index_link}_{product.id}_equal_{flows[product]}")
                    var = self.solver_master.variables["flows"][all_index_link][self.problem.product_to_index[product]]
                    self.solver_master.cp_model.Add(var == flows[product]).OnlyEnforceIf(ind)
                    self.solver_master.cp_model.Add(var != flows[product]).OnlyEnforceIf(ind.Not())
                    indicators.append(ind)
                all_equal = self.solver_master.cp_model.NewBoolVar(f"indicator_cut_{all_index_link}_{self.nb_cuts}")
                self.solver_master.cp_model.Add(all_equal == 1).OnlyEnforceIf(*indicators)

                (self.solver_master.cp_model.Add(self.solver_master.variables["nb_trips"][all_index_link] == new_nb_trip)
                 .OnlyEnforceIf(all_equal))
                self.nb_cuts += 1
        self.solver_master.cp_model.Add(self.solver_master.variables["total_obj"]
                                        >= int(math.floor(best_lb)))

    def solve(
        self, callbacks: Optional[list[Callback]] = None,
        params_solver_master: dict = None,
        params_solver_sub: dict = None,
        nb_iteration: int = 10,
        **kwargs: Any
    ) -> ResultStorage:
        callback = CallbackList()
        callback.on_solve_start(solver=self)
        if params_solver_master is None:
            params_solver_master = dict(parameters_cp=ParametersCp.default_cpsat(),
                                        time_limit=5,
                                        ortools_cpsat_solver_kwargs={"log_search_progress": True})
        if params_solver_sub is None:
            params_solver_sub = dict(parameters_cp=ParametersCp.default_cpsat(),
                                     time_limit=5,
                                     ortools_cpsat_solver_kwargs={"log_search_progress": True})
        res_merged = self.create_result_storage([])
        current_lbs = []
        for i in range(nb_iteration):
            if i >= 1:
                bs, _ = res_merged.get_best_solution_fit(satisfying=self.problem)
                self.solver_master.set_warm_start(bs)
            res = self.solver_master.solve(callbacks=callbacks,
                                           **params_solver_master)
            current_lbs.append(self.solver_master.get_current_best_internal_objective_bound())
            base_solution = res[-1][0]
            print("Status master : ", self.solver_master.status_solver)
            print("Cur solution", self.problem.evaluate(base_solution), self.problem.satisfy(base_solution))
            print("Obj ", self.aggreg_from_sol(base_solution))
            subsolver = CpsatPackingSubproblem(problem=self.problem)
            subsolver.init_model(solution=base_solution)
            self.solver_subproblem = subsolver
            res_sub = subsolver.solve(**params_solver_sub)
            res_merged.extend(res_sub)
            print("Status subsolver : ", subsolver.status_solver)
            print("Solution post pro", self.problem.evaluate(res_sub[-1][0]), self.problem.satisfy(res_sub[-1][0]))
            print("Obj ", self.aggreg_from_sol(res_sub[-1][0]))
            callback.on_step_end(i, res=res_sub, solver=self)
            changes = subsolver.analyse_solution(new_solution=res_sub[-1][0])
            if len(changes) == 0:
                print("stop")
                break
            print("Nb changes :", len(changes))
            best_lb = max(current_lbs)
            self.update_model_master(changed=changes, best_lb=best_lb)
            bs, best_ub = res_merged.get_best_solution_fit(satisfying=self.problem)
            print("Best lb", best_lb)
            print("Best Ub", best_ub)
            # self.solver_master.cp_model.Add(sum(self.solver_master.variables["objs"][obj]
            #                                     for obj in self.solver_master.variables["objs"])
            #                                 >= int(math.floor(best_lb)))
        return res_merged






