import random
from typing import Optional, Any

from discrete_optimization.generic_tools.ortools_cpsat_tools import OrtoolsCpSatSolver
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction, Solution
from pydantic.version import version_short

from vrp_tw_alternative_delivery.problem import (VrpTwAlternativeDeliveryProblem,
                                                 VRPTWSolution,
                                                 VrpTwAltSolution,
                                                 RequirementRelation,
                                                 ProductInstance,
                                                 ProductType, Vehicle, RequiredProductRelation)
from ortools.sat.python.cp_model import CpModel, CpSolverSolutionCallback, IntVar, LinearExpr
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (CategoricalHyperparameter,
                                                                                EnumHyperparameter)

#TODO :

# - write a version with relaxed visits.
# - contrainte "cluster de client", quand un vehicle visite un ensemble de client C, alors quand il
# part visiter un client en dehors de C, tu ne reviens pas dans C ensuite.
# - Contraintes de type patrouille :
# Lorsque les patrouilles sont "autonomes" i.e have all "RequirementRelations" inside, we should use them.
#
#
from enum import Enum
class GroupRuleConstraint(Enum):
    FLOW_REASONING = 0
    TIME_REASONING = 1


class CpSatVrpTwAlternative(OrtoolsCpSatSolver):
    hyperparameters = [CategoricalHyperparameter(name="symmetry_breaking_vehicle",
                                                 choices=[True, False],
                                                 default=False),
                       CategoricalHyperparameter(name="create_time_per_vehicle",
                                                 choices=[True, False],
                                                 default=False),
                       EnumHyperparameter(name="group_constraint", enum=GroupRuleConstraint,
                                          default=GroupRuleConstraint.FLOW_REASONING),
                       ]
    problem: VrpTwAlternativeDeliveryProblem

    def __init__(self, problem: VrpTwAlternativeDeliveryProblem,
                 params_objective_function: Optional[ParamsObjectiveFunction] = None, **kwargs):
        super().__init__(problem=problem, params_objective_function=params_objective_function, **kwargs)
        self.variables = {}
        self.nodes = self.problem.nodes
        self.id_nodes_list = [n.id for n in self.nodes]
        self.id_to_index = {self.id_nodes_list[i]: i for i in range(len(self.id_nodes_list))}

    def init_model(self, **kwargs: Any) -> None:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        relaxed = kwargs.get("relaxed", False)
        self.init_model_default(**kwargs)
        # self.init_model_relaxed(**kwargs)
        pass

    def compute_equivalent_vehicle_classes(self):
        dict_vehicle_per_feature = {}
        for i in range(self.problem.number_vehicles):
            vehicle = self.problem.vehicles[i]
            features = (tuple(vehicle.load_per_type), vehicle.min_speed, vehicle.max_speed)
            if features not in dict_vehicle_per_feature:
                dict_vehicle_per_feature[features] = []
            dict_vehicle_per_feature[features].append(i)
        return dict_vehicle_per_feature

    def init_model_relaxed(self, **kwargs):
        pass

    def init_model_default(self, optional: bool = False, **kwargs: Any) -> None:
        self.cp_model = CpModel()
        arcs_dict = {}
        used_vehicle_dict = {}
        for k in range(self.problem.number_vehicles):
            arcs_dict_k, used_vehicle_k = self.create_flow_variables_per_vehicle(index_vehicle=k)
            arcs_dict.update(arcs_dict_k)
            used_vehicle_dict[k] = used_vehicle_k

        if kwargs["symmetry_breaking_vehicle"]:
            dict_vehicle_per_feature = self.compute_equivalent_vehicle_classes()
            not_used_per_class_key = {}
            for class_key in dict_vehicle_per_feature:
                list_ = dict_vehicle_per_feature[class_key]

                for i in range(len(list_)-1):
                    self.cp_model.AddImplication(used_vehicle_dict[list_[i]].Not(),
                                                 used_vehicle_dict[list_[i+1]].Not())
                    self.cp_model.add(used_vehicle_dict[list_[i+1]] <= used_vehicle_dict[list_[i]])
                not_used_per_class_key[class_key] = sum([used_vehicle_dict[x].Not()
                                                         for x in list_])
            not_used_per_total = sum([not_used_per_class_key[x] for x in not_used_per_class_key])
            keys = list(not_used_per_class_key.keys())
            self.cp_model.add(not_used_per_class_key[keys[0]]-not_used_per_class_key[keys[1]] <= 2)
            self.cp_model.add(not_used_per_class_key[keys[1]]-not_used_per_class_key[keys[0]] <= 2)
        self.create_many_constraints(arcs_dict=arcs_dict,
                                     optional=optional,
                                     **kwargs)
        self.variables["used"] = used_vehicle_dict
        self.variables["arcs_dict"] = arcs_dict
        self.create_objectives()

    def create_flow_variables_per_vehicle(self, index_vehicle: int = 0):
        arcs_vehicle = []
        arcs_dict = {}
        vehicle: Vehicle = self.problem.vehicles[index_vehicle]
        start_node = vehicle.start_node
        index_start_node = self.id_to_index[start_node.id]
        end_node = vehicle.end_node
        index_end_node = self.id_to_index[end_node.id]
        for i in range(self.problem.number_nodes):
            for j in range(self.problem.number_nodes):
                # We don't create self loop or transition between start and end here.
                if i == index_start_node and j == index_end_node:
                    continue
                if i == index_start_node and i == j:
                    continue
                if i == index_end_node and i == j:
                    continue
                flow = self.cp_model.NewBoolVar(name=f"flow_{i, j, index_vehicle}")
                arcs_dict[(i, j, index_vehicle)] = flow
                arcs_vehicle.append((i, j, flow))
        dummy_literal = self.cp_model.NewBoolVar(name=f"flow_{index_start_node, index_end_node, index_vehicle}")
        arcs_vehicle.append((index_start_node, index_end_node, dummy_literal))
        used_vehicle = self.cp_model.NewBoolVar(name=f"used_{index_vehicle}")
        self.cp_model.Add(used_vehicle == dummy_literal.Not())
        # self.cp_model.AddCircuit(arcs=arcs_vehicle)
        if True:
            for i in range(self.problem.number_nodes):
                in_ = {x for x in arcs_dict if x[1] == i}
                out_ = {x for x in arcs_dict if x[0] == i}
                if i == index_start_node:
                    self.cp_model.add(sum(arcs_dict[x] for x in out_)+dummy_literal == 1)
                elif i == index_end_node:
                    self.cp_model.add(sum(arcs_dict[x] for x in in_)+dummy_literal == 1)
                else:
                    sum_in = sum(arcs_dict[x] for x in in_)
                    sum_out = sum(arcs_dict[x] for x in in_)
                    self.cp_model.add(sum_in == sum_out)
        return arcs_dict, used_vehicle

    def create_many_constraints(self, arcs_dict: dict[tuple[int, int, int], IntVar],
                                optional: bool = False, **kwargs):
        create_time_per_vehicle = kwargs["create_time_per_vehicle"]
        group_constraint: GroupRuleConstraint = kwargs["group_constraint"]
        visited_node_per_vehicle = {}
        delivered_to_node_per_vehicle = {}
        time_per_index = {}
        index_customers = [i for i in range(self.problem.number_nodes)
                           if self.problem.nodes[i].is_client]
        incoming_arc_to_i = {i: {x for x in arcs_dict if x[1] == i
                                 and x[0] != x[1]} for i in index_customers}
        outgoing_arc_from_i = {i: {x for x in arcs_dict
                                 if x[0] == i
                                 and x[0] != x[1]} for i in index_customers}
        # Creating the time.
        for index in incoming_arc_to_i:
            node = self.problem.nodes[index]
            time_per_index[index] = self.cp_model.NewIntVar(lb=node.time_window[0],
                                                            ub=node.time_window[1],
                                                            name=f"arrival_time_{index}")
        if create_time_per_vehicle:
            time_per_index_per_vehicle = {}
            horizon = max([n.time_window[1] for n in self.problem.nodes])
            for index in incoming_arc_to_i:
                #node = self.problem.nodes[index]
                for index_vehicle in range(self.problem.number_vehicles):
                    time_per_index_per_vehicle[(index, index_vehicle)] = (
                        self.cp_model.NewIntVar(lb=0,
                                                ub=horizon,
                                                name=f"arrival_time_{index}_{index_vehicle}"))


        for index in incoming_arc_to_i:
            visited_node_per_vehicle[index] = {}

            for index_vehicle in range(self.problem.number_vehicles):
                inc_ = {x for x in incoming_arc_to_i[index] if x[2] == index_vehicle}
                out_ = {x for x in outgoing_arc_from_i[index] if x[2] == index_vehicle}
                visited_index_per_vehicle = self.cp_model.NewBoolVar(name=f"{index}_visited_by_{index_vehicle}")
                self.cp_model.add(visited_index_per_vehicle == sum(arcs_dict[c] for c in inc_))
                self.cp_model.add(visited_index_per_vehicle == sum(arcs_dict[c] for c in out_))
                visited_node_per_vehicle[index][index_vehicle] = visited_index_per_vehicle
                vehicle = self.problem.vehicles[index_vehicle]
                keys_product_by_vehicle = []
                for (type_product, _) in vehicle.load_per_type:
                    type_product: ProductType
                    delivered_prod_type = self.cp_model.NewBoolVar(name=f"{type_product.name}_delivered_by_"
                                                                        f"{index_vehicle}_to_{index}")
                    delivered_to_node_per_vehicle[(index, index_vehicle, type_product)] = delivered_prod_type
                    keys_product_by_vehicle.append((index, index_vehicle, type_product))
                # Index is visited per the vehicle <-> at least one product is delivered.
                (self.cp_model.add(sum([delivered_to_node_per_vehicle[x] for x in keys_product_by_vehicle]) >= 1)
                 .OnlyEnforceIf(visited_index_per_vehicle))
                (self.cp_model.add(sum([delivered_to_node_per_vehicle[x] for x in keys_product_by_vehicle]) == 0)
                 .OnlyEnforceIf(visited_index_per_vehicle.Not()))
        # Initial and return time for each vehicle
        for index_vehicle in range(self.problem.number_vehicles):
            vehicle: Vehicle = self.problem.vehicles[index_vehicle]
            start_node = vehicle.start_node
            index_start_node = self.id_to_index[start_node.id]
            end_node = vehicle.end_node
            index_end_node = self.id_to_index[end_node.id]
            time_per_index[(index_start_node, index_vehicle)] = self.nodes[index_start_node].time_window[0]
            time_per_index[("end", index_vehicle)] = self.cp_model.NewIntVar(lb=0,
                                                                             ub=self.nodes[index_end_node].time_window[1],
                                                                             name=f"end_{index_vehicle}")
        if optional:
            is_visited = {}
            for index in range(self.problem.number_nodes):
                node = self.problem.nodes[index]
                if node.mandatory:
                    is_visited[index] = 1
                else:
                    is_visited[index] = self.cp_model.NewBoolVar(name=f"is_visited_{index}")
        else:
            is_visited = {i: 1 for i in range(self.problem.number_nodes)}
        # Timings constraints
        if create_time_per_vehicle:
            # Synchronize...
            for (index, index_vehicle) in time_per_index_per_vehicle:
                (self.cp_model.Add(time_per_index_per_vehicle[(index, index_vehicle)] == time_per_index[index])
                 .OnlyEnforceIf(visited_node_per_vehicle[index][index_vehicle]))

            for i in range(self.problem.number_nodes):
                incoming_arc_to_i = {x for x in arcs_dict if x[1] == i}
                for arc in incoming_arc_to_i:
                    if arc[0] == arc[1]:
                        # Ignore loops
                        continue
                    vehicle_index = arc[2]
                    vehicle: Vehicle = self.problem.vehicles[vehicle_index]
                    start_node = vehicle.start_node
                    index_start_node = self.id_to_index[start_node.id]
                    end_node = vehicle.end_node
                    index_end_node = self.id_to_index[end_node.id]
                    if arc[0] == index_start_node:
                        t0 = time_per_index[(index_start_node, vehicle_index)]
                    else:
                        t0 = time_per_index_per_vehicle[(arc[0], vehicle_index)]
                    if arc[1] == index_end_node:
                        t1 = time_per_index[("end", vehicle_index)]
                    else:
                        t1 = time_per_index_per_vehicle[(arc[1], vehicle_index)]
                    # TODO : put specific vehicle speed constraint here.
                    self.cp_model.add(t1 >= t0 + int(self.problem.distance_matrix[arc[0], arc[1]])
                                      + int(self.problem.nodes[arc[0]].service_time)).OnlyEnforceIf(arcs_dict[arc])
        for i in range(self.problem.number_nodes):
            incoming_arc_to_i = {x for x in arcs_dict if x[1] == i}
            for arc in incoming_arc_to_i:
                if arc[0] == arc[1]:
                    # Ignore loops
                    continue
                vehicle_index = arc[2]
                vehicle: Vehicle = self.problem.vehicles[vehicle_index]
                start_node = vehicle.start_node
                index_start_node = self.id_to_index[start_node.id]
                end_node = vehicle.end_node
                index_end_node = self.id_to_index[end_node.id]
                if arc[0] == index_start_node:
                    t0 = time_per_index[(index_start_node, vehicle_index)]
                else:
                    t0 = time_per_index[arc[0]]
                if arc[1] == index_end_node:
                    t1 = time_per_index[("end", vehicle_index)]
                else:
                    t1 = time_per_index[arc[1]]
                # TODO : put specific vehicle speed constraint here.
                self.cp_model.add(t1 >= t0 + int(self.problem.distance_matrix[arc[0], arc[1]])
                                  + int(self.problem.nodes[arc[0]].service_time)).OnlyEnforceIf(arcs_dict[arc])
                # TODO : Minimum Speed ??
                #self.cp_model.add(t1 <= t0 + int(self.problem.distance_matrix[arc[0], arc[1]])
                #                  + int(self.problem.nodes[arc[0]].service_time)).OnlyEnforceIf(arcs_dict[arc])


        if group_constraint == GroupRuleConstraint.FLOW_REASONING:
            for group in self.problem.index_of_nodes_per_group:
                indexes_in_group = self.problem.index_of_nodes_per_group[group]
                other_indexes = {i for i in range(self.problem.number_nodes) if i not in indexes_in_group}
                incoming_i_j = {(i, j) for i in other_indexes for j in indexes_in_group}
                outcoming_j_i = {(j, i) for i in other_indexes for j in indexes_in_group}
                for index_vehicle in range(self.problem.number_vehicles):
                    arcs_in = {(i, j, index_vehicle) for (i, j) in incoming_i_j
                               if (i, j, index_vehicle) in arcs_dict}
                    self.cp_model.add_at_most_one([arcs_dict[arc] for arc in arcs_in])
                    arcs_out = {(i, j, index_vehicle) for (i, j) in outcoming_j_i
                                if (i, j, index_vehicle) in arcs_dict}
                    self.cp_model.add_at_most_one([arcs_dict[arc] for arc in arcs_out])

        if group_constraint == GroupRuleConstraint.TIME_REASONING:
            # Group constraint : This seems rather complicated, should we maybe include them lazily
            # #Group*#Vehicle*(nbnodesPerGroup=#Nodes/#group)*#Nodes*#Nodes/#Group
            # = #Vehicle*#Nodes^3/#Group constraint.
            if create_time_per_vehicle:
                for group in self.problem.index_of_nodes_per_group:
                    indexes = self.problem.index_of_nodes_per_group[group]
                    for vehicle in range(self.problem.number_vehicles):
                        for i in indexes:
                            outgoing_outside = {x for x in outgoing_arc_from_i[i]
                                                if x[2] == vehicle
                                                and x[1] not in indexes}
                            for arc in outgoing_outside:
                                for j in indexes:
                                    if i == j:
                                        continue
                                    (self.cp_model.Add(time_per_index_per_vehicle[(j, vehicle)]<=time_per_index[i])
                                     .only_enforce_if(self.variables["arcs_dict"][arc]))

        # Requirements constraints
        for index in index_customers:
            keys = {key for key in delivered_to_node_per_vehicle if key[0] == index}
            # Index in visited by at least one vehicle
            # For the classical vrp : it would be add_exactly_one instead.
            # For the optional visit version : no constraint at all or something like
            # sum(is_node_visited[node])>=0.5*len(nodes)
            consumable_delivery_keys = {key for key in keys
                                        if not key[2].is_renewable}
            if not optional: # all nodes are visited..
                self.cp_model.add_at_least_one([visited_node_per_vehicle[index][vehicle]
                                                for vehicle in visited_node_per_vehicle[index]])
                self.cp_model.add_at_least_one([delivered_to_node_per_vehicle[key]
                                                for key in consumable_delivery_keys])
            else:
                self.cp_model.add(sum([visited_node_per_vehicle[index][vehicle]
                                       for vehicle in visited_node_per_vehicle[index]])>=1).OnlyEnforceIf(
                    is_visited[index])
                self.cp_model.add(sum([delivered_to_node_per_vehicle[key]
                                       for key in consumable_delivery_keys])>=1).OnlyEnforceIf(is_visited[index])
            for req in self.problem.requirement_relations:
                keys_prod1 = set()
                keys_prod2 = set()
                if isinstance(req, RequirementRelation):
                    req: RequirementRelation
                    prod1 = req.product_type_1
                    prod2 = req.product_type_2
                    keys_prod1 = {key for key in keys if key[2] == prod1}
                    keys_prod2 = {key for key in keys if key[2] == prod2}
                elif isinstance(req, RequiredProductRelation):
                    req: RequiredProductRelation
                    prod1 = req.product_type_1
                    prod2 = req.product_types_2
                    keys_prod1 = {key for key in keys if key[2] == prod1}
                    keys_prod2 = {key for key in keys if key[2] in prod2}
                if len(keys_prod1) > 0 and len(keys_prod2) > 0:
                    sum_prod1 = sum([delivered_to_node_per_vehicle[key] for key in keys_prod1])
                    sum_prod2 = sum([delivered_to_node_per_vehicle[key] for key in keys_prod2])
                    self.cp_model.add(sum_prod2 >= sum_prod1)  # If prod1 is delivered, prod2 is needed too.
        # Capacity constraint per vehicle
        for index_vehicle in range(self.problem.number_vehicles):
            vehicle = self.problem.vehicles[index_vehicle]
            for (type_product, capacity) in vehicle.load_per_type:
                type_product: ProductType
                if not type_product.is_renewable:  # This is consumable, so we need to put some limit.
                    keys = {key for key in delivered_to_node_per_vehicle if key[1] == index_vehicle
                            and key[2] == type_product}
                    self.cp_model.add(sum([delivered_to_node_per_vehicle[x] for x in keys]) <= capacity)
        self.variables["delivered_to_node_per_vehicle"] = delivered_to_node_per_vehicle
        self.variables["visited_node_per_vehicle"] = visited_node_per_vehicle
        self.variables["time_per_index"] = time_per_index

    def create_objectives(self):
        random_values_per_index_and_tp = {}
        coefs = []
        expr = []
        for index, vehic, type_prod in self.variables["delivered_to_node_per_vehicle"]:
            if (index, type_prod) not in random_values_per_index_and_tp:
                random_values_per_index_and_tp[(index, type_prod)] = 10*random.random()
            coefs.append(random_values_per_index_and_tp[(index, type_prod)])
            expr.append(self.variables["delivered_to_node_per_vehicle"][(index, vehic, type_prod)])
        coefs = []
        expr = []
        for i, j, index_vehicle in self.variables["arcs_dict"]:
            coefs.append(self.problem.travel_costs[i, j])
            expr.append(self.variables["arcs_dict"][i, j, index_vehicle])
        obj = LinearExpr.weighted_sum(expr, coefs)
        self.cp_model.Minimize(obj)
        # self.cp_model.Minimize(obj+1000*sum([self.variables["used"][x] for x in self.variables["used"]]))


    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> VrpTwAltSolution:
        # Initialize one empty list for each vehicle's trajectory
        trajectories: list[list[tuple[int, int, set[ProductInstance]]]] = [[] for _ in
                                                                           range(self.problem.number_vehicles)]

        delivered_vars = self.variables["delivered_to_node_per_vehicle"]
        visited_node_per_vehicle = self.variables["visited_node_per_vehicle"]
        time_vars = self.variables["time_per_index"]
        print([cpsolvercb.Value(self.variables["used"][x])
               for x in self.variables["used"]])
        print('arcs')
        for (i, j, index_vehicle) in self.variables["arcs_dict"]:
            if cpsolvercb.Value(self.variables["arcs_dict"][(i,j,index_vehicle)]):
                pass
                #print(i, j, index_vehicle)

        # Keep track of non-renewable instances used by each vehicle
        product_instances_used_count: dict[tuple[int, ProductType], int] = {}

        # Group deliveries by (node_index, vehicle_index)
        # This ensures all products delivered by the same vehicle to the same node
        # are grouped into a single visit tuple.
        visits_data: dict[tuple[int, int], (int, set[ProductInstance])] = {}

        for (node_index, vehicle_index, product_type), var in delivered_vars.items():
            if cpsolvercb.Value(var):
                # print("Visited node per vehicle ?",
                #       cpsolvercb.Value(visited_node_per_vehicle[node_index][vehicle_index]))
                key = (node_index, vehicle_index)

                # Get arrival time (once per (node, vehicle) visit)
                if key not in visits_data:
                    arrival_time = cpsolvercb.Value(time_vars[node_index])
                    # print("Time", node_index, arrival_time)
                    visits_data[key] = (arrival_time, set())

                # Find the specific ProductInstance to deliver
                vehicle = self.problem.vehicles[vehicle_index]
                available_instances = vehicle.instance_dict_per_type.get(product_type, [])

                if not available_instances:
                    raise RuntimeError(
                        f"Model error: Vehicle {vehicle_index} delivered {product_type.name} but has none.")

                product_instance_to_add: Optional[ProductInstance] = None
                if product_type.is_renewable:
                    product_instance_to_add = available_instances[0]
                else:
                    instance_key = (vehicle_index, product_type)
                    instance_idx_to_use = product_instances_used_count.get(instance_key, 0)

                    if instance_idx_to_use >= len(available_instances):
                        raise RuntimeError(
                            f"Model error: Vehicle {vehicle_index} delivered more non-renewable "
                            f"{product_type.name} than available.")

                    product_instance_to_add = available_instances[instance_idx_to_use]
                    product_instances_used_count[instance_key] = instance_idx_to_use + 1

                if product_instance_to_add:
                    visits_data[key][1].add(product_instance_to_add)

        # Convert the dictionary of visits into the list[list] structure
        for (node_index, vehicle_index), (arrival_time, product_set) in visits_data.items():
            if product_set:  # Only add visits that actually deliver something
                visit_tuple = (node_index, arrival_time, product_set)
                trajectories[vehicle_index].append(visit_tuple)

        # Sort each vehicle's trajectory by arrival time
        for k in range(self.problem.number_vehicles):
            trajectories[k].sort(key=lambda x: x[1])

        return VrpTwAltSolution(problem=self.problem, trajectories=trajectories)



