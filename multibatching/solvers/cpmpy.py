import math
from collections import defaultdict, Counter

from ortools.sat.python.cp_model import CpSolverSolutionCallback
from typing import Any, Optional
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.cpmpy_tools import CpmpySolver, Model
import cpmpy as cp
from multibatching.problem import (MultibatchingProblem, Location,
                                   TransportLink, TransportType,
                                   PackingTransport, Product, MultibatchingSolution)
import logging
from enum import Enum
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (EnumHyperparameter,
                                                                                CategoricalHyperparameter)
from multibatching.solvers.cpsat import ModelingMultiBatch
from discrete_optimization.generic_tools.cpmpy_tools import MetaCpmpyConstraint
logger = logging.getLogger(__name__)


class CpmpyMultibatchingSolver(CpmpySolver):
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
        self.meta_constraints: list[MetaCpmpyConstraint] = []

    def retrieve_current_solution(self) -> MultibatchingSolution:
        print(self.cpm_solver.objective_value())
        packing_solution = []
        products = self.problem.products
        for index_tl in self.variables["flows"]:
            nb = self.variables["nb_trips"][index_tl].value()
            if nb > 0:
                packings = []
                if self.modeling == ModelingMultiBatch.UNIT_FLOW:
                    for i in sorted(self.variables["trips_variables"][index_tl]):
                        used = self.variables["used_trips"][index_tl][i]
                        if used.value() > 0:
                            current_packing = set()
                            for index_product in self.variables["trips_variables"][index_tl][i]:
                                val = self.variables['trips_variables'][index_tl][i][index_product].value()
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
                        val = self.variables['flows'][index_tl][index_product].value()
                        if val > 0:
                            cur_packing[products[index_product]] = val / nb
                    if len(cur_packing) > 0:
                        packing_solution.append(PackingTransport(transport_link=self.problem.transport_links[index_tl],
                                                                 product_packing=cur_packing,
                                                                 nb_packing=nb))
        return MultibatchingSolution(problem=self.problem, list_flows=packing_solution)

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

    def init_model(self, **args: Any) -> None:
        args = self.complete_with_default_hyperparameters(args)
        modeling = args["modeling"]
        if modeling == ModelingMultiBatch.FLOW:
            self.init_model_flows(**args)
        self.modeling = modeling

    def init_model_flows(self, **args: Any) -> None:
        supply_per_product = {p: self.problem.get_total_supply(p) for p in self.problem.products}
        demand_per_product = {p: self.problem.get_total_demand(p) for p in self.problem.products}
        total_size_per_product = {p: p.size*supply_per_product[p] for p in self.problem.products}
        total_size_in_problem = sum(total_size_per_product.values())
        # TODO : this is not totally correct, but fine.
        max_nb_trip = int(total_size_in_problem/min([tt.capacity for tt in self.problem.transport_types
                                                     if tt.capacity > 0]))
        # Could provide non-hardcoded upper bound for our variables...
        model = Model()
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
            nb_trips_per_link[index_tl] = cp.intvar(lb=0, ub=max_nb_trip,
                                                    shape=1,
                                                    name=f'num_trips_{index_tl}')
            list_var_weight = []
            # for index_product in range(self.problem.nb_products):
            for index_product in range(self.problem.nb_products):
                product = self.problem.products[index_product]
                max_flow_product = int(max(supply_per_product[product],
                                       demand_per_product[product]))
                if tr in product.valid_transports:
                    flows_variables[index_tl][index_product] = cp.intvar(lb=0, ub=max_flow_product,
                                                                         shape=1,
                                                                         name=f"flow_{index_tl}_{index_product}")
                    keys_outflow_location_product[loc1.id][product.id].add((index_tl, index_product))
                    keys_inflow_location_product[loc2.id][product.id].add((index_tl, index_product))
                    list_var_weight.append((flows_variables[index_tl][index_product],
                                            int(product.size)))

            # Global capacity constraint
            c = (cp.sum([x[1]*x[0] for x in list_var_weight]) <=
                 nb_trips_per_link[index_tl]*int(tl.transport_type.capacity))
            capacity_meta = MetaCpmpyConstraint(name=f"link_flow_to_nb_trips_{index_tl}",
                                                constraints=[c],
                                                metadata={"type": "background",
                                                          "tl_index": index_tl,
                                                          "location1": tl.location_l1.name,
                                                          "location2": tl.location_l2.name,
                                                          "transport_type": tl.transport_type.name})
            self.meta_constraints.append(capacity_meta)
            model += capacity_meta.constraints
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
                    binary_indicators = [cp.boolvar(shape=1,
                                                    name=f'flow_{index_tl}_{index_product}_equal_{k}')
                                         for k in range(len(nb_trips_min))]
                    for k in range(len(nb_trips_min)):
                        model += [(binary_indicators[k] == (flows_variables[index_tl][index_product] == k))]
                    for j in range(1, len(binary_indicators)):
                        model += [binary_indicators[j].implies(nb_trips_per_link[index_tl] >= nb_trips_min[j])]

        # Flows constraint
        for location in self.problem.locations:
            loc_id = location.id
            debug_mode = args.get("debug_mode", False)
            index_products = args.get("debug_settings", {}).get("index_products", range(len(self.problem.products)))
            for index_product in index_products:
                product = self.problem.products[index_product]
                net_supply = int(location.net_supply.get(product, 0))
                flow_in = cp.sum([flows_variables[x[0]][x[1]]
                                  for x in keys_inflow_location_product.get(loc_id, {}).get(product.id,
                                                                                            set())])
                flow_out = cp.sum([flows_variables[x[0]][x[1]]
                                   for x in keys_outflow_location_product.get(loc_id, {}).get(product.id,
                                                                                              set())])
                c_flow = (flow_in+net_supply-flow_out == 0)
                flow_meta = MetaCpmpyConstraint(name=f"flow_{product.name}_{location.name}",
                                                constraints=[c_flow],
                                                metadata={"type": "flow",
                                                          "product": product.name,
                                                          "location": location.name})
                self.meta_constraints.append(flow_meta)
                model += flow_meta.constraints
        if False:
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
            self.variables["objs"] = {"transport": transport_cost,
                                      "emission": emission_cost,
                                      "capital": sum(estimated_compound)}
        model.minimize(12)

        self.model = model
        self.variables["flows"] = flows_variables
        self.variables["nb_trips"] = nb_trips_per_link







