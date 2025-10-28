import math
import random
import re

from discrete_optimization.generic_tools.do_problem import Solution

from satellite_path_planning.problem import SppProblem, SppSolution
from discrete_optimization.generic_tools.dyn_prog_tools import DpSolver, dp
from typing import Any
import networkx as nx
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (CategoricalHyperparameter,
                                                                                EnumHyperparameter,
                                                                                IntegerHyperparameter)
from enum import Enum


class DpModeling(Enum):
    PICK_IGNORE = 0
    GOTO = 1
    GOTO_WITH_DOM = 2


class DpSppSolver(DpSolver, WarmstartMixin):
    problem: SppProblem
    transitions: dict[str, dp.Transition]
    hyperparameters = DpSolver.hyperparameters+[EnumHyperparameter(name="modeling", enum=DpModeling,
                                                                   default=DpModeling.PICK_IGNORE),
                                                CategoricalHyperparameter(name="allow_revisit", choices=[True, False],
                                                                          default=False),
                                                CategoricalHyperparameter(name="forced_transition", choices=[True, False],
                                                                          default=False,
                                                                          depends_on=('modeling', DpModeling.PICK_IGNORE)),
                                                CategoricalHyperparameter(name="forced_transition_threshold",
                                                                          choices=[10**i for i in range(4, 9)],
                                                                          depends_on=[("forced_transition", True)],
                                                                          default=10**5)
                                                ]
    modeling: DpModeling = None

    def init_model(self, **kwargs):
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        modeling: DpModeling = kwargs["modeling"]
        if modeling == DpModeling.GOTO:
            self.init_model_goto(**kwargs)
            self.modeling = modeling
        if modeling == DpModeling.GOTO_WITH_DOM:
            self.init_model_with_domination(**kwargs)
            self.modeling = modeling
        if modeling == DpModeling.PICK_IGNORE:
            self.init_model_pick_ignore_improved(**kwargs)
            self.modeling = modeling

    def init_model_goto(self, **kwargs: Any) -> None:
        graph = self.problem.graph
        # graph = nx.transitive_closure_dag(graph, topo_order=sorted(list(self.problem.graph.nodes)))
        # print("Transitive closure done : ")
        nodes = sorted(list(graph.nodes))
        nb_nodes = len(nodes)
        index_of_node = {nodes[i]: i for i in range(nb_nodes)}
        acquisition_request_id = nx.get_node_attributes(graph, "acquisition_request_id")
        nodes_per_acquisition = {}
        for n in acquisition_request_id:
            if acquisition_request_id[n] == "first":
                nodes_per_acquisition[-1] = {n}
                acquisition_request_id[n] = -1
            elif acquisition_request_id[n] == "last":
                nodes_per_acquisition[-2] = {n}
                acquisition_request_id[n] = -2
            else:
                if acquisition_request_id[n] not in nodes_per_acquisition:
                    nodes_per_acquisition[acquisition_request_id[n]] = set()
                nodes_per_acquisition[acquisition_request_id[n]].add(n)
        acquisitions = list(nodes_per_acquisition.keys())
        values = [int(self.problem.acquisition_dict.get(acq, 0)) for acq in acquisitions]
        print("values ", values)
        acq_to_index = {acquisitions[i]: i for i in range(len(acquisitions))}
        nb_acquisitions = len(acquisitions)
        source = 0
        target_ = len(nodes) - 1
        neighs = {index_of_node[n]: {index_of_node[s] for s in nbrdict}
                  for n, nbrdict in graph.adjacency()}
        neighs_table = [neighs.get(i, set()) for i in range(nb_nodes)]
        acquisitions_indexed = [{index_of_node[node]
                                 for node in nodes_per_acquisition[acq]}
                                for acq in acquisitions]
        acquisition_of_node = [acq_to_index[acquisition_request_id[nodes[i]]]
                               for i in range(nb_nodes)]
        model = dp.Model(maximize=True)
        node_object = model.add_object_type(number=nb_nodes)
        acq_object = model.add_object_type(number=nb_acquisitions)

        neighs_table = model.add_set_table(neighs_table, object_type=node_object)
        nodes_per_acquisition = model.add_set_table(acquisitions_indexed,
                                                    object_type=node_object)
        values_ = model.add_int_table(values)
        acquisition_object = model.add_object_type(number=nb_acquisitions)
        acquisitions_done = model.add_set_var(object_type=acquisition_object,
                                              target=set())
        closed = model.add_set_var(object_type=node_object,
                                   target=set())
        cur_node = model.add_element_var(object_type=node_object,
                                         target=source)
        model.add_base_case([cur_node == target_])
        self.transitions = {}
        for i in range(nb_nodes):
            s_j = model.create_set_const(object_type=node_object, value={j for j in range(i+1)})
            val = (acquisitions_done.contains(acquisition_of_node[i])).if_then_else(
                0, values_[acquisition_of_node[i]])
            tr = dp.Transition(name=f"goto_{i}",
                               cost=val+dp.IntExpr.state_cost(),
                               # values_[acquisition_of_node[i]]+dp.IntExpr.state_cost(),
                               effects=[(cur_node, i),
                                        (closed,
                                         closed
                                         - nodes_per_acquisition[acquisition_of_node[i]]
                                         -
                                         s_j
                                         ),
                                        (acquisitions_done,
                                         acquisitions_done.add(acquisition_of_node[i]))],
                               preconditions=[i > cur_node,
                                              ~closed.contains(i),
                                              neighs_table[cur_node].contains(i)])
            model.add_transition(tr)
            self.transitions[f"goto_{i}"] = tr
        acq_of_node = model.add_set_table([{acq_to_index[acquisition_request_id[nodes[i]]]}
                                           for i in range(nb_nodes)],
                                          object_type=acq_object)
        acq_in_the_future = [{acquisition_of_node[-1]}]

        for i in range(len(acquisition_of_node)-2, -1, -1):
            acq_in_the_future.append(acq_in_the_future[-1].union({acquisition_of_node[i]}))
        acq_in_the_future = acq_in_the_future[::-1]
        acq_in_future_table = model.add_set_table(acq_in_the_future, object_type=acq_object)
        model.add_dual_bound(values_[acq_in_future_table[cur_node].difference(acquisitions_done)])
        model.add_dual_bound(values_[acq_of_node.union(closed.complement())])
        if "expected_values" in kwargs:
            values = [int((kwargs["expected_values"][n]-graph.nodes[n]["value"])) for n in nodes]
            values_table = model.add_int_table(values)
            model.add_dual_bound(values_table[cur_node])
        self.model = model
        self.nodes = nodes

    def init_model_with_domination(self, **kwargs: Any) -> None:
        graph = self.problem.graph
        self.problem.compute_transitive_closure_graph()
        node_to_acquisition_descendants = self.problem.node_to_acquisition_descendants
        nodes = sorted(list(graph.nodes))
        nb_nodes = len(nodes)
        index_of_node = {nodes[i]: i for i in range(nb_nodes)}
        acquisition_request_id = nx.get_node_attributes(graph, "acquisition_request_id")
        nodes_per_acquisition = {}
        for n in acquisition_request_id:
            if acquisition_request_id[n] == "first":
                nodes_per_acquisition[-1] = {n}
                acquisition_request_id[n] = -1
            elif acquisition_request_id[n] == "last":
                nodes_per_acquisition[-2] = {n}
                acquisition_request_id[n] = -2
            else:
                if acquisition_request_id[n] not in nodes_per_acquisition:
                    nodes_per_acquisition[acquisition_request_id[n]] = set()
                nodes_per_acquisition[acquisition_request_id[n]].add(n)
        acquisitions = list(nodes_per_acquisition.keys())
        values = [int(self.problem.acquisition_dict.get(acq, 0)) for acq in acquisitions]
        print("values ", values)
        acq_to_index = {acquisitions[i]: i for i in range(len(acquisitions))}

        node_to_acquisition_descendants_recomputed = []
        for i in range(len(nodes)):
            node = nodes[i]
            acqs = [n for n in node_to_acquisition_descendants.get(node, [])
                    if n not in {"first", 'last'} and n != acquisition_request_id[node]]
            node_to_acquisition_descendants_recomputed.append(set([acq_to_index[x] for x in acqs]))

        nb_acquisitions = len(acquisitions)
        source = 0
        target_ = len(nodes) - 1
        neighs = {index_of_node[n]: {index_of_node[s] for s in nbrdict}
                  for n, nbrdict in graph.adjacency()}
        neighs_table = [neighs.get(i, set()) for i in range(nb_nodes)]
        acquisitions_indexed = [{index_of_node[node]
                                 for node in nodes_per_acquisition[acq]}
                                for acq in acquisitions]
        acquisition_of_node = [acq_to_index[acquisition_request_id[nodes[i]]]
                               for i in range(nb_nodes)]
        model = dp.Model(maximize=True)
        node_object = model.add_object_type(number=nb_nodes)
        acq_object = model.add_object_type(number=nb_acquisitions)
        node_to_acquisition_desc = model.add_set_table(node_to_acquisition_descendants_recomputed,
                                                       object_type=acq_object)
        neighs_table = model.add_set_table(neighs_table, object_type=node_object)
        nodes_per_acquisition = model.add_set_table(acquisitions_indexed,
                                                    object_type=node_object)
        values_ = model.add_int_table(values)
        acquisition_object = model.add_object_type(number=nb_acquisitions)
        acquisitions_done = model.add_set_var(object_type=acquisition_object,
                                              target=set())
        closed = model.add_set_var(object_type=node_object,
                                   target=set())
        cur_node = model.add_element_var(object_type=node_object,
                                         target=source)
        potential_next_acquisition_from_node = [model.add_set_state_fun(node_to_acquisition_desc[i]
                                                                        - acquisitions_done)
                                                for i in range(nb_nodes)]
        potential_value_from_node = [model.add_int_state_fun(values_[potential_next_acquisition_from_node[i]])
                                     for i in range(nb_nodes)]
        value_of_acquisition = [model.add_int_state_fun(acquisitions_done.contains(i).if_then_else(
                                                        0, values_[i]))
                                for i in range(len(acquisitions))]

        model.add_base_case([cur_node == target_])
        self.transitions = {}
        self.transitions_id = {}
        for i in range(nb_nodes):
            s_j = model.create_set_const(object_type=node_object, value={j for j in range(i + 1)})
            tr = dp.Transition(name=f"goto_{i}",
                               cost=value_of_acquisition[acquisition_of_node[i]] + dp.IntExpr.state_cost(),
                               # values_[acquisition_of_node[i]]+dp.IntExpr.state_cost(),
                               effects=[(cur_node, i),
                                        (closed,
                                         closed
                                         | nodes_per_acquisition[acquisition_of_node[i]]
                                         | s_j),
                                        (acquisitions_done,
                                         acquisitions_done.add(acquisition_of_node[i]))],
                               preconditions=[i > cur_node,
                                              ~closed.contains(i),
                                              neighs_table[cur_node].contains(i)])
            id_t = model.add_transition(tr)
            self.transitions[f"goto_{i}"] = tr
            self.transitions_id[i] = id_t
        preds = self.problem.compute_predecessors_map()
        if True:
            for i in range(nb_nodes):
                print(i)
                node_i = nodes[i]
                for j in range(nb_nodes):
                    if i == j:
                        continue
                    node_j = nodes[j]
                    if len(preds[node_i].intersection(preds[node_j])) > 0:
                        if values[acquisition_of_node[j]] <= values[acquisition_of_node[i]]:
                            cond1 = (potential_value_from_node[j]+values[acquisition_of_node[j]] <
                                     potential_value_from_node[i]+values[acquisition_of_node[i]])
                            cond2 = potential_next_acquisition_from_node[j].issubset(potential_next_acquisition_from_node[i])
                            model.add_transition_dominance(self.transitions_id[i],
                                                           self.transitions_id[j],
                                                               [cond2, cond1])

        acq_of_node = model.add_set_table([{acq_to_index[acquisition_request_id[nodes[i]]]}
                                           for i in range(nb_nodes)],
                                          object_type=acq_object)
        acq_in_the_future = [{acquisition_of_node[-1]}]
        for i in range(len(acquisition_of_node) - 2, -1, -1):
            acq_in_the_future.append(acq_in_the_future[-1].union({acquisition_of_node[i]}))
        acq_in_the_future = acq_in_the_future[::-1]
        acq_in_future_table = model.add_set_table(acq_in_the_future, object_type=acq_object)
        model.add_dual_bound(values_[acq_in_future_table[cur_node].difference(acquisitions_done)])
        model.add_dual_bound(values_[node_to_acquisition_desc[cur_node].difference(acquisitions_done)])
        model.add_dual_bound(values_[acq_of_node.union(closed.complement())])
        if "expected_values" in kwargs:
            values = [int((kwargs["expected_values"][n] - graph.nodes[n]["value"])) for n in nodes]
            values_table = model.add_int_table(values)
            model.add_dual_bound(values_table[cur_node])

        self.model = model
        self.nodes = nodes










        # print("Transitive closure done : ")
        nodes = sorted(list(graph.nodes))
        nb_nodes = len(nodes)
        index_of_node = {nodes[i]: i for i in range(nb_nodes)}
        acquisition_request_id = nx.get_node_attributes(graph, "acquisition_request_id")
        nodes_per_acquisition = {}
        for n in acquisition_request_id:
            if acquisition_request_id[n] == "first":
                nodes_per_acquisition[-1] = {n}
                acquisition_request_id[n] = -1
            elif acquisition_request_id[n] == "last":
                nodes_per_acquisition[-2] = {n}
                acquisition_request_id[n] = -2
            else:
                if acquisition_request_id[n] not in nodes_per_acquisition:
                    nodes_per_acquisition[acquisition_request_id[n]] = set()
                nodes_per_acquisition[acquisition_request_id[n]].add(n)
        acquisitions = list(nodes_per_acquisition.keys())
        values = [int(self.problem.acquisition_dict.get(acq, 0)) for acq in acquisitions]
        print("values ", values)
        acq_to_index = {acquisitions[i]: i for i in range(len(acquisitions))}
        nb_acquisitions = len(acquisitions)
        source = 0
        target_ = len(nodes) - 1
        neighs = {index_of_node[n]: {index_of_node[s] for s in nbrdict}
                  for n, nbrdict in graph.adjacency()}




        neighs_table = [neighs.get(i, set()) for i in range(nb_nodes)]
        acquisitions_indexed = [{index_of_node[node]
                                 for node in nodes_per_acquisition[acq]}
                                for acq in acquisitions]
        acquisition_of_node = [acq_to_index[acquisition_request_id[nodes[i]]]
                               for i in range(nb_nodes)]
        model = dp.Model(maximize=True)

        node_object = model.add_object_type(number=nb_nodes)
        acq_object = model.add_object_type(number=nb_acquisitions)
        neighs_table = model.add_set_table(neighs_table, object_type=node_object)
        nodes_per_acquisition = model.add_set_table(acquisitions_indexed,
                                                    object_type=node_object)
        values_ = model.add_int_table(values)
        acquisition_object = model.add_object_type(number=nb_acquisitions)
        acquisitions_done = model.add_set_var(object_type=acquisition_object,
                                              target=set())
        closed = model.add_set_var(object_type=node_object,
                                   target=set())
        cur_node = model.add_element_var(object_type=node_object,
                                         target=source)
        model.add_base_case([cur_node == target_])
        self.transitions = {}
        for i in range(nb_nodes):
            s_j = model.create_set_const(object_type=node_object, value={j for j in range(i+1)})
            val = (acquisitions_done.contains(acquisition_of_node[i])).if_then_else(
                0, values_[acquisition_of_node[i]])
            tr = dp.Transition(name=f"goto_{i}",
                               cost=val+dp.IntExpr.state_cost(),
                               # values_[acquisition_of_node[i]]+dp.IntExpr.state_cost(),
                               effects=[(cur_node, i),
                                        (closed,
                                         closed
                                         - nodes_per_acquisition[acquisition_of_node[i]]
                                         -
                                         s_j
                                         ),
                                        (acquisitions_done,
                                         acquisitions_done.add(acquisition_of_node[i]))],
                               preconditions=[i > cur_node,
                                              ~closed.contains(i),
                                              neighs_table[cur_node].contains(i)])
            model.add_transition(tr)
            self.transitions[f"goto_{i}"] = tr
        acq_of_node = model.add_set_table([{acq_to_index[acquisition_request_id[nodes[i]]]}
                                           for i in range(nb_nodes)],
                                          object_type=acq_object)
        acq_in_the_future = [{acquisition_of_node[-1]}]

        for i in range(len(acquisition_of_node)-2, -1, -1):
            acq_in_the_future.append(acq_in_the_future[-1].union({acquisition_of_node[i]}))
        acq_in_the_future = acq_in_the_future[::-1]
        acq_in_future_table = model.add_set_table(acq_in_the_future, object_type=acq_object)
        model.add_dual_bound(values_[acq_in_future_table[cur_node].difference(acquisitions_done)])
        model.add_dual_bound(values_[acq_of_node.union(closed.complement())])
        if "expected_values" in kwargs:
            values = [int((kwargs["expected_values"][n]-graph.nodes[n]["value"])) for n in nodes]
            values_table = model.add_int_table(values)
            model.add_dual_bound(values_table[cur_node])
        self.model = model
        self.nodes = nodes

    def init_model_pick_ignore(self, **kwargs: Any) -> None:
        graph = self.problem.graph
        nodes = sorted(list(graph.nodes))
        nb_nodes = len(nodes)
        index_of_node = {nodes[i]: i for i in range(nb_nodes)}
        acquisition_request_id = nx.get_node_attributes(graph, "acquisition_request_id")
        times = nx.get_node_attributes(graph, "start")
        timespan = {t: times[t].timestamp() for t in times}
        min_ts = min(timespan.values())
        timespan = {t: int(timespan[t]-min_ts) for t in timespan}
        times_array = [timespan[n] for n in nodes]
        nodes_per_acquisition = {}
        for n in acquisition_request_id:
            if acquisition_request_id[n] == "first":
                nodes_per_acquisition[-1] = {n}
                acquisition_request_id[n] = -1
            elif acquisition_request_id[n] == "last":
                nodes_per_acquisition[-2] = {n}
                acquisition_request_id[n] = -2
            else:
                if acquisition_request_id[n] not in nodes_per_acquisition:
                    nodes_per_acquisition[acquisition_request_id[n]] = set()
                nodes_per_acquisition[acquisition_request_id[n]].add(n)
        acquisitions = list(nodes_per_acquisition.keys())
        values = [int(self.problem.acquisition_dict.get(acq, 0)) for acq in acquisitions]
        print("values ", values)
        acq_to_index = {acquisitions[i]: i for i in range(len(acquisitions))}
        nb_acquisitions = len(acquisitions)
        source = 0
        target_ = len(nodes) - 1
        neighs = {index_of_node[n]: {index_of_node[s] for s in nbrdict}
                  for n, nbrdict in graph.adjacency()}
        neighs_table = [neighs.get(i, set()) for i in range(nb_nodes)]
        acquisitions_indexed = [{index_of_node[node]
                                 for node in nodes_per_acquisition[acq]}
                                for acq in acquisitions]
        acquisition_of_node = [acq_to_index[acquisition_request_id[nodes[i]]]
                               for i in range(nb_nodes)]
        model = dp.Model(maximize=True)
        node_object = model.add_object_type(number=nb_nodes)
        acq_object = model.add_object_type(number=nb_acquisitions)
        neighs_table = model.add_set_table(neighs_table, object_type=node_object)
        nodes_per_acquisition = model.add_set_table(acquisitions_indexed,
                                                    object_type=node_object)
        values_acq = model.add_int_table(values)
        acquisition_object = model.add_object_type(number=nb_acquisitions)
        acquisitions_done = model.add_set_var(object_type=acquisition_object,
                                              target=set())
        # acquisition_of_node_tb = model.add_int_table(acquisition_of_node)
        acquisition_of_node_tb = model.add_set_table(
            [{x} for x in acquisition_of_node],
            object_type=acquisition_object)
        acquisition_of_node_table = model.add_element_table(acquisition_of_node)
        closed = model.add_set_var(object_type=node_object,
                                   target=set())
        cur_node = model.add_element_var(object_type=node_object,
                                         target=0)
        last_node = model.add_element_var(object_type=node_object,
                                          target=0)
        values_node = model.add_int_table([int(self.problem.acquisition_dict.get(acquisition_request_id[n], 0))
                                           for n in nodes])
        times_table = model.add_int_table(times_array)
        # cur_node = model.add_int_resource_var(target=0)
        model.add_base_case([cur_node == nb_nodes-1])
        self.transitions = {}
        if kwargs["allow_revisit"]:
            pick = dp.Transition(name=f"pick_",
                                 cost=acquisitions_done.contains(acquisition_of_node_table[cur_node])
                                 .if_then_else(0, values_node[cur_node])
                                      + dp.IntExpr.state_cost(),
                                 effects=[(cur_node, cur_node + 1),
                                          (acquisitions_done,
                                           acquisitions_done.add(acquisition_of_node_table[cur_node])),
                                          (last_node, cur_node)],
                                 preconditions=[cur_node <= nb_nodes - 1,
                                                (neighs_table[last_node].contains(cur_node)) |
                                                (times_table[cur_node] >= times_table[last_node] + 30),
                                                ~closed.contains(cur_node)])
        else:
            pick = dp.Transition(name=f"pick_",
                                 cost=values_node[cur_node]+dp.IntExpr.state_cost(),
                                 effects=[(cur_node, cur_node+1),
                                          (acquisitions_done, acquisitions_done.union(acquisition_of_node_tb[cur_node])),
                                          # (closed, closed | nodes_per_acquisition[acquisition_of_node_tb[cur_node]]),
                                          (last_node, cur_node)],
                                 preconditions=[cur_node <= nb_nodes-1,
                                                (neighs_table[last_node].contains(cur_node)) |
                                                (times_table[cur_node] >= times_table[last_node] + 30),
                                                # acquisitions_done.intersection(acquisition_of_node_tb[cur_node]).is_empty(),
                                                ~acquisitions_done.contains(acquisition_of_node_table[cur_node]),
                                                ~closed.contains(cur_node)])
        model.add_transition(pick)
        self.transitions["pick"] = pick
        if kwargs["forced_transition"]:
            pick_force = dp.Transition(name=f"pickforce_",
                                       cost=values_node[cur_node] + dp.IntExpr.state_cost(),
                                       effects=[(cur_node, cur_node + 1),
                                                (acquisitions_done, acquisitions_done.union(acquisition_of_node_tb[cur_node])),
                                                (last_node, cur_node)],
                                       preconditions=[cur_node <= nb_nodes - 1,
                                                      values_node[cur_node] >= kwargs["forced_transition_threshold"],
                                                      (neighs_table[last_node].contains(cur_node)) |
                                                      (times_table[cur_node] >= times_table[last_node] + 30),
                                                      # acquisitions_done.intersection(acquisition_of_node_tb[cur_node]).is_empty(),
                                                      ~acquisitions_done.contains(acquisition_of_node_table[cur_node]),
                                                      ~closed.contains(cur_node)])
            model.add_transition(pick_force,
                                 forced=True)
            self.transitions["pick_force"] = pick_force
        ignore = dp.Transition(name=f"ignore_",
                               cost=dp.IntExpr.state_cost(),
                               effects=[(cur_node, cur_node + 1),
                                        (closed, closed.add(cur_node))],
                               preconditions=[cur_node <= nb_nodes - 1])
        model.add_transition(ignore)
        self.transitions["ignore"] = ignore
        if "expected_values" in kwargs:
            values = [int(kwargs["expected_values"][n]) for n in nodes]
            max_v = [max(values[i:]) for i in range(len(nodes))]
            values_table = model.add_int_table(max_v)
            model.add_dual_bound(values_table[cur_node])
        # acq_of_node = model.add_set_table([{acq_to_index[acquisition_request_id[nodes[i]]]}
        #                                   for i in range(nb_nodes)],
        #                                   object_type=acq_object)
        # model.add_dual_bound(values_acq[acquisitions_done.complement()]//2)
        model.add_dual_bound(values_acq[acquisitions_done.complement()])
        self.model = model
        self.nodes = nodes
        self.cur_node = cur_node

    def init_model_pick_ignore_2(self, **kwargs: Any) -> None:
        graph = self.problem.graph
        nodes = sorted(list(graph.nodes))
        nb_nodes = len(nodes)
        index_of_node = {nodes[i]: i for i in range(nb_nodes)}
        acquisition_request_id = nx.get_node_attributes(graph, "acquisition_request_id")
        times = nx.get_node_attributes(graph, "start")
        timespan = {t: times[t].timestamp() for t in times}
        min_ts = min(timespan.values())
        timespan = {t: int(timespan[t]-min_ts) for t in timespan}
        times_array = [timespan[n] for n in nodes]
        nodes_per_acquisition = {}
        for n in acquisition_request_id:
            if acquisition_request_id[n] == "first":
                nodes_per_acquisition[-1] = {n}
                acquisition_request_id[n] = -1
            elif acquisition_request_id[n] == "last":
                nodes_per_acquisition[-2] = {n}
                acquisition_request_id[n] = -2
            else:
                if acquisition_request_id[n] not in nodes_per_acquisition:
                    nodes_per_acquisition[acquisition_request_id[n]] = set()
                nodes_per_acquisition[acquisition_request_id[n]].add(n)
        acquisitions = list(nodes_per_acquisition.keys())
        values = [int(self.problem.acquisition_dict.get(acq, 0)) for acq in acquisitions]
        print("values ", values)
        acq_to_index = {acquisitions[i]: i for i in range(len(acquisitions))}
        nb_acquisitions = len(acquisitions)
        source = 0
        target_ = len(nodes) - 1
        neighs = {index_of_node[n]: {index_of_node[s] for s in nbrdict}
                  for n, nbrdict in graph.adjacency()}
        neighs_table = [neighs.get(i, set()) for i in range(nb_nodes)]
        acquisitions_indexed = [{index_of_node[node]
                                 for node in nodes_per_acquisition[acq]}
                                for acq in acquisitions]
        acquisition_of_node = [acq_to_index[acquisition_request_id[nodes[i]]]
                               for i in range(nb_nodes)]
        model = dp.Model(maximize=True)
        node_object = model.add_object_type(number=nb_nodes)
        acq_object = model.add_object_type(number=nb_acquisitions)
        min_neighs_table = model.add_element_table([min(x, default=nb_nodes) for x in neighs_table])
        max_neighs_table = model.add_element_table([max(x, default=nb_nodes) for x in neighs_table])
        neighs_table = model.add_set_table(neighs_table, object_type=node_object)
        nodes_per_acquisition = model.add_set_table(acquisitions_indexed,
                                                    object_type=node_object)
        values_acq = model.add_int_table(values)
        acquisition_object = model.add_object_type(number=nb_acquisitions)
        acquisitions_done = model.add_set_var(object_type=acquisition_object,
                                              target=set())
        # acquisition_of_node_tb = model.add_int_table(acquisition_of_node)
        acquisition_of_node_tb = model.add_set_table(
            [{x} for x in acquisition_of_node],
            object_type=acquisition_object)
        acquisition_of_node_table = model.add_element_table(acquisition_of_node)
        closed = model.add_set_var(object_type=node_object,
                                   target=set())
        cur_node = model.add_element_var(object_type=node_object,
                                         target=0)
        last_node = model.add_element_var(object_type=node_object,
                                          target=0)
        values_node = model.add_int_table([int(self.problem.acquisition_dict.get(acquisition_request_id[n], 0))
                                           for n in nodes])
        times_table = model.add_int_table(times_array)
        # cur_node = model.add_int_resource_var(target=0)
        model.add_base_case([cur_node == nb_nodes-1])
        self.transitions = {}
        if kwargs["allow_revisit"]:
            pick = dp.Transition(name=f"pick_",
                                 cost=acquisitions_done.contains(acquisition_of_node_table[cur_node])
                                 .if_then_else(0, values_node[cur_node])
                                      + dp.IntExpr.state_cost(),
                                 effects=[(cur_node, cur_node + 1),
                                          (acquisitions_done,
                                           acquisitions_done.add(acquisition_of_node_table[cur_node])),
                                          (last_node, cur_node)],
                                 preconditions=[cur_node <= nb_nodes - 1,
                                                (neighs_table[last_node].contains(cur_node)) |
                                                (times_table[cur_node] >= times_table[last_node] + 30),
                                                ~closed.contains(cur_node)])
        else:
            pick = dp.Transition(name=f"pick_",
                                 cost=values_node[cur_node]+dp.IntExpr.state_cost(),
                                 effects=[(cur_node, min_neighs_table[cur_node]),
                                          (acquisitions_done, acquisitions_done.union(acquisition_of_node_tb[cur_node])),
                                          # (closed, closed | nodes_per_acquisition[acquisition_of_node_tb[cur_node]]),
                                          (last_node, cur_node)],
                                 preconditions=[cur_node <= nb_nodes-1,
                                                cur_node <= max_neighs_table[last_node],
                                                neighs_table[last_node].contains(cur_node),
                                                # acquisitions_done.intersection(acquisition_of_node_tb[cur_node]).is_empty(),
                                                ~acquisitions_done.contains(acquisition_of_node_table[cur_node]),
                                                # ~closed.contains(cur_node)
                                                ])
        model.add_transition(pick)
        self.transitions["pick"] = pick
        if kwargs["forced_transition"]:
            pick_force = dp.Transition(name=f"pickforce_",
                                       cost=values_node[cur_node] + dp.IntExpr.state_cost(),
                                       effects=[(cur_node, cur_node + 1),
                                                (acquisitions_done, acquisitions_done.union(acquisition_of_node_tb[cur_node])),
                                                (last_node, cur_node)],
                                       preconditions=[cur_node <= nb_nodes - 1,
                                                      cur_node <= max_neighs_table[last_node],
                                                      values_node[cur_node] >= kwargs["forced_transition_threshold"],
                                                      neighs_table[last_node].contains(cur_node),
                                                      # acquisitions_done.intersection(acquisition_of_node_tb[cur_node]).is_empty(),
                                                      ~acquisitions_done.contains(acquisition_of_node_table[cur_node])])
            model.add_transition(pick_force,
                                 forced=True)
            self.transitions["pick_force"] = pick_force
        ignore = dp.Transition(name=f"ignore_",
                               cost=dp.IntExpr.state_cost(),
                               effects=[(cur_node, cur_node + 1),
                                        (closed, closed.add(cur_node))],
                               preconditions=[cur_node <= nb_nodes - 1])
        model.add_transition(ignore)
        self.transitions["ignore"] = ignore
        if "expected_values" in kwargs:
            values = [int(kwargs["expected_values"][n]) for n in nodes]
            max_v = [max(values[i:]) for i in range(len(nodes))]
            values_table = model.add_int_table(max_v)
            model.add_dual_bound(values_table[cur_node])

        model.add_dual_bound(values_acq[acquisitions_done.complement()])
        self.model = model
        self.nodes = nodes
        self.cur_node = cur_node

    def init_model_pick_ignore_improved(self, **kwargs: Any) -> None:
        # --- 1. Setup based on GOTO_WITH_DOMINATION ---
        # We need the transitive closure to compute reachable acquisitions
        #self.problem.compute_transitive_closure_graph()
        #node_to_acquisition_descendants = self.problem.node_to_acquisition_descendants

        graph = self.problem.graph
        nodes = sorted(list(graph.nodes))
        nb_nodes = len(nodes)
        index_of_node = {nodes[i]: i for i in range(nb_nodes)}
        acquisition_request_id = nx.get_node_attributes(graph, "acquisition_request_id")

        nodes_per_acquisition = {}
        for n in acquisition_request_id:
            if acquisition_request_id[n] == "first":
                nodes_per_acquisition[-1] = {n}
                acquisition_request_id[n] = -1
            elif acquisition_request_id[n] == "last":
                nodes_per_acquisition[-2] = {n}
                acquisition_request_id[n] = -2
            else:
                if acquisition_request_id[n] not in nodes_per_acquisition:
                    nodes_per_acquisition[acquisition_request_id[n]] = set()
                nodes_per_acquisition[acquisition_request_id[n]].add(n)

        acquisitions = list(nodes_per_acquisition.keys())
        values = [int(self.problem.acquisition_dict.get(acq, 0)) for acq in acquisitions]
        acq_to_index = {acquisitions[i]: i for i in range(len(acquisitions))}
        nb_acquisitions = len(acquisitions)

        # Compute reachable acquisitions (by index) for the dual bound
        #node_to_acquisition_descendants_recomputed = []
        #for i in range(len(nodes)):
        #    node = nodes[i]
        #    acqs = [n for n in node_to_acquisition_descendants.get(node, [])
        #            if n not in {"first", 'last'} and n != acquisition_request_id[node]]
        #    node_to_acquisition_descendants_recomputed.append(set([acq_to_index[x] for x in acqs]))

        # --- 2. Setup based on PICK_IGNORE_2 ---
        neighs = {index_of_node[n]: {index_of_node[s] for s in nbrdict}
                  for n, nbrdict in graph.adjacency()}
        neighs_table_sets = [neighs.get(i, set()) for i in range(nb_nodes)]
        acquisition_of_node = [acq_to_index[acquisition_request_id[nodes[i]]]
                               for i in range(nb_nodes)]

        model = dp.Model(maximize=True)
        node_object = model.add_object_type(number=nb_nodes)
        acq_object = model.add_object_type(number=nb_acquisitions)

        # Tables for transitions
        min_neighs_table = model.add_element_table([min(x, default=nb_nodes) for x in neighs_table_sets])
        max_neighs_table = model.add_element_table([max(x, default=nb_nodes) for x in neighs_table_sets])
        neighs_table = model.add_set_table(neighs_table_sets, object_type=node_object)
        nodes_per_acquisition_table = model.add_set_table(
            [{index_of_node[node] for node in nodes_per_acquisition[acq]} for acq in acquisitions],
            object_type=node_object
        )

        # Tables for cost & objective
        values_acq = model.add_int_table(values)
        values_node_list = [int(self.problem.acquisition_dict.get(acquisition_request_id[n], 0)) for n in nodes]
        values_node = model.add_int_table(values_node_list)

        # Tables for state updates
        acquisition_of_node_tb = model.add_set_table(
            [{x} for x in acquisition_of_node],
            object_type=acq_object
        )

        acquisition_of_node_table = model.add_element_table(acquisition_of_node)

        # Table for NEW dual bound
        #node_to_acq_desc_table = model.add_set_table(node_to_acquisition_descendants_recomputed,
        #                                             object_type=acq_object)

        # --- 3. State Variables ---
        acquisitions_done = model.add_set_var(object_type=acq_object,
                                              target=set())
        # NEW state variable to track ignored nodes
        ignored_nodes = model.add_set_var(object_type=node_object,
                                          target=set())
        cur_node = model.add_element_var(object_type=node_object,
                                         target=0)
        last_node = model.add_element_var(object_type=node_object,
                                          target=0)
        neighs_list = []
        max_nb_neigh = max([len(x) for x in neighs_table_sets])+1
        nb_neighs = []
        for i in range(nb_nodes):
            neighs = sorted(list(neighs_table_sets[i]))
            values_ = [values_node_list[n] for n in neighs]
            if "expected_values" in kwargs:
                values_ = [int(kwargs["expected_values"][nodes[n]])
                           for n in neighs]
            sorted_neighs = sorted(range(len(neighs)), key=lambda x: values_[x], reverse=True)
            if len(sorted_neighs) > 0 and False:
                max_ = values_[sorted_neighs[0]]
                indexes = [sorted_neighs[i] for i in range(len(sorted_neighs))
                           if values_[sorted_neighs[i]] >= 0.01*max_
                           #or random.random() <= 100*values_[sorted_neighs[i]]/max_
                           #or random.random() <= 0.1
                           ]#[:5]
                print(len(indexes), len(neighs))
                neighs = sorted([neighs[x] for x in indexes])
            nb_neighs.append(len(neighs))
            # indexes = sorted_neighs[:min(20, len(neighs))]
            neighs_list.append(neighs+[nb_nodes-1]*(max_nb_neigh-len(neighs)))
        index_neigh_obj = model.add_object_type(number=max_nb_neigh)
        index_neigh = model.add_element_var(target=0, object_type=index_neigh_obj)
        nb_neighs_table = model.add_element_table(nb_neighs)
        neighs_big_table = model.add_element_table(neighs_list)
        model.add_base_case([cur_node == nb_nodes - 1])
        self.transitions = {}
        # --- 4. Transitions (assuming allow_revisit=False) ---
        pick = dp.Transition(
            name=f"pick_",
            cost=values_node[cur_node] + dp.IntExpr.state_cost(),
            effects=[
                (cur_node, neighs_big_table[cur_node, 0]),  # Smart jump
                (acquisitions_done, acquisitions_done.add(acquisition_of_node_table[cur_node])),
                (last_node, cur_node),
                (index_neigh, 0),
                # No effect on ignored_nodes
            ],
            preconditions=[
                # neighs_table[last_node].contains(cur_node),  # Graph constraint
                ~(acquisitions_done.contains(acquisition_of_node_table[cur_node])),  # Objective logic
                # ~ignored_nodes.contains(cur_node)  # NEW: Don't pick if ignored
            ]
        )
        id_pick = model.add_transition(pick)
        self.transitions["pick"] = pick
        if kwargs["allow_revisit"]:
            pick = dp.Transition(name=f"pick_already",
                                 cost=dp.IntExpr.state_cost(),
                                 effects=[
                                     (cur_node, neighs_big_table[cur_node, 0]),  # Smart jump
                                     #(acquisitions_done, acquisitions_done.add(acquisition_of_node_table[cur_node])),
                                     (last_node, cur_node),
                                     (index_neigh, 0),
                                 ],
                                 preconditions=[acquisitions_done.contains(acquisition_of_node_table[cur_node])])
            model.add_transition(pick)
        if False:
            pick_last = dp.Transition(
                name=f"pick_",
                cost=values_node[cur_node] + dp.IntExpr.state_cost(),
                effects=[
                    (cur_node, neighs_big_table[cur_node, 0]),  # Smart jump
                    (acquisitions_done, acquisitions_done.add(acquisition_of_node_table[cur_node])),
                    (last_node, cur_node),
                    (index_neigh, 0),
                    # No effect on ignored_nodes
                ],
                preconditions=[
                    index_neigh == nb_neighs_table[last_node]-1,
                    # cur_node <= max_neighs_table[last_node],  # Optimization
                    # neighs_table[last_node].contains(cur_node),  # Graph constraint
                    ~acquisitions_done.contains(acquisition_of_node_table[cur_node]),  # Objective logic
                    # ~ignored_nodes.contains(cur_node)  # NEW: Don't pick if ignored
                ]
            )
            id_last = model.add_transition(pick, forced=True)
            self.transitions["pick_last"] = pick_last

        if kwargs.get("forced_transition", False):
            # This logic remains the same
            pick_force = dp.Transition(
                name=f"pickforce_",
                cost=values_node[cur_node] + dp.IntExpr.state_cost(),
                effects=[
                    (cur_node, neighs_big_table[cur_node, 0]),  # Smart jump
                    (acquisitions_done, acquisitions_done.add(acquisition_of_node_table[cur_node])),
                    (last_node, cur_node),
                    (index_neigh, 0),
                ],
                preconditions=[
                    values_node[cur_node] >= kwargs["forced_transition_threshold"],
                    ~acquisitions_done.contains(acquisition_of_node_table[cur_node]),
                ]
            )
            model.add_transition(pick_force, forced=True)
            self.transitions["pick_force"] = pick_force

        ignore = dp.Transition(
            name=f"ignore_",
            cost=dp.IntExpr.state_cost(),
            effects=[
                (cur_node, neighs_big_table[last_node, index_neigh+1]),  # Standard progression
                (index_neigh, index_neigh+1),
                # (ignored_nodes, ignored_nodes.add(cur_node))  # NEW: Track ignored node
            ],
            preconditions=[cur_node <= nb_nodes - 1,
                           index_neigh < nb_neighs_table[last_node]
                           # cur_node < max_neighs_table[last_node]
                           ]
        )
        self.transitions["ignore"] = ignore
        id_ignore = model.add_transition(ignore)
        ignore_force = dp.Transition(
            name=f"ignore_force",
            cost=dp.IntExpr.state_cost(),
            effects=[
                (cur_node, neighs_big_table[last_node, index_neigh + 1]),  # Standard progression
                (index_neigh, index_neigh + 1),
                # (ignored_nodes, ignored_nodes.add(cur_node))  # NEW: Track ignored node
            ],
            preconditions=[acquisitions_done.contains(acquisition_of_node_table[cur_node])]
        )
        #id_ignore_force = model.add_transition(ignore_force, forced=True)
        self.transitions["ignore_force"] = ignore_force
        model.add_transition_dominance(id_ignore, id_pick,
                            [1.5*values_node[cur_node] < values_node[neighs_big_table[last_node, index_neigh+1]],
                                       ~(acquisitions_done.contains(acquisition_of_node_table[neighs_big_table[last_node, index_neigh+1]]))])
        model.add_transition_dominance(id_pick, id_ignore,
                                       [~(acquisitions_done.contains(acquisition_of_node_table[cur_node])),
                                        values_node[cur_node]>=100*values_node.max(neighs_table[last_node])])
        if "expected_values" in kwargs:
            values = [int(kwargs["expected_values"][n])
                      for n in nodes]
            values_table = model.add_int_table(values)
            model.add_transition_dominance(id_ignore, id_pick,
                                           [
                                            1000000+values_table[cur_node] < values_table[neighs_big_table[last_node,
                                                                                                           index_neigh + 1]],
                                            ~(acquisitions_done
                                              .contains(acquisition_of_node_table[neighs_big_table[last_node,
                                                                                                   index_neigh + 1]]))
                                           ])




        self.transitions["ignore"] = ignore

        # --- 5. Dual Bounds ---
        # Original bound (still valid)
        # model.add_dual_bound(values_acq[acquisitions_done.complement()])

        # NEW, tighter path-dependent bound
        # model.add_dual_bound(
        #    values_acq[node_to_acq_desc_table[last_node].difference(acquisitions_done)]
        # )
        if "expected_values" in kwargs:
            values = [int(kwargs["expected_values"][n])
                      for n in nodes]
            values_table = model.add_int_table(values)
            model.add_dual_bound(values_table.max(neighs_table[last_node]))
        self.model = model
        self.nodes = nodes
        self.cur_node = cur_node
        self.last_node = last_node
        self.neighs_list  = neighs_list
        self.visit = acquisitions_done

    def init_model_with_stereo(self, **kwargs: Any) -> None:
        graph = self.problem.graph
        # graph = nx.transitive_closure_dag(graph, topo_order=sorted(list(self.problem.graph.nodes)))
        # print("Transitive closure done : ")
        nodes = sorted(list(graph.nodes))
        nb_nodes = len(nodes)
        index_of_node = {nodes[i]: i for i in range(nb_nodes)}
        acquisition_request_id = nx.get_node_attributes(graph, "acquisition_request_id")
        times = nx.get_node_attributes(graph, "start")
        timespan = {t: times[t].timestamp() for t in times}
        min_ts = min(timespan.values())
        timespan = {t: int(timespan[t]-min_ts) for t in timespan}
        times_array = [timespan[n] for n in nodes]
        nodes_per_acquisition = {}
        for n in acquisition_request_id:
            if acquisition_request_id[n] == "first":
                nodes_per_acquisition[-1] = {n}
                acquisition_request_id[n] = -1
            elif acquisition_request_id[n] == "last":
                nodes_per_acquisition[-2] = {n}
                acquisition_request_id[n] = -2
            else:
                if acquisition_request_id[n] not in nodes_per_acquisition:
                    nodes_per_acquisition[acquisition_request_id[n]] = set()
                nodes_per_acquisition[acquisition_request_id[n]].add(n)
        acquisitions = list(nodes_per_acquisition.keys())
        values = [int(self.problem.acquisition_dict.get(acq, 0)) for acq in acquisitions]

        neighs = {index_of_node[n]: {index_of_node[s] for s in nbrdict}
                  for n, nbrdict in graph.adjacency()}
        neighs_table = [neighs.get(i, set()) for i in range(nb_nodes)]
        random_stereo = []
        for j in range(100):
            rand_ = random.randint(0, nb_nodes-1)
            if len(neighs_table[rand_]) > 0:
                rand_j = random.choice(list(neighs_table[rand_]))
                random_stereo.append((rand_, rand_j, 100000))
        is_on_stereo = [0 for i in range(nb_nodes)]
        is_inbound_stereo = [0 for i in range(nb_nodes)]
        is_outbound_stereo = [0 for i in range(nb_nodes)]
        new_random_stereo = []
        for i, j, val in random_stereo:
            if (is_inbound_stereo[i] == 0
                    and is_outbound_stereo[i] == 0
                    and is_inbound_stereo[j] == 0
                    and is_outbound_stereo[j] == 0):
                is_on_stereo[i] = 1
                is_on_stereo[j] = 1
                is_outbound_stereo[j] = 1
                is_inbound_stereo[i] = 1
                new_random_stereo.append((i, j, val))
        random_stereo = new_random_stereo
        print("Nb Stereo constraint", len(random_stereo))
        to_do = [set() for i in range(nb_nodes)]
        for i, j, val in random_stereo:
            to_do[i].add(j)
        model = dp.Model(maximize=True)
        node_object = model.add_object_type(number=nb_nodes)
        nb_acquisitions = len(acquisitions)
        acq_object = model.add_object_type(number=nb_acquisitions)
        to_dos = model.add_set_table(to_do, object_type=node_object)
        is_inbound_stereo = model.add_int_table(is_inbound_stereo)
        is_outbound_stereo = model.add_int_table(is_outbound_stereo)
        is_inbound_stereo_set = model.create_set_const(object_type=node_object,
                                                       value={i for i, _, _ in random_stereo})
        is_outbound_stereo_set = model.create_set_const(object_type=node_object,
                                                        value={i for _,i,_ in random_stereo})
        needed_node_before = [set() for i in range(nb_nodes)]
        for i, j, val in random_stereo:
            needed_node_before[j].add(i)
        needed_nodes = model.add_set_table(needed_node_before, object_type=node_object)

        values_node = [int(self.problem.acquisition_dict.get(acquisition_request_id[n], 0))
                       for n in nodes]
        for i, j, val in random_stereo:
            values_node[i] = val//2
            values_node[j] = val//2
        acq_to_index = {acquisitions[i]: i for i in range(len(acquisitions))}
        target_ = len(nodes) - 1

        acquisitions_indexed = [{index_of_node[node]
                                 for node in nodes_per_acquisition[acq]}
                                for acq in acquisitions]
        acquisition_of_node = [acq_to_index[acquisition_request_id[nodes[i]]]
                               for i in range(nb_nodes)]
        neighs_table = model.add_set_table(neighs_table, object_type=node_object)
        nodes_per_acquisition = model.add_set_table(acquisitions_indexed,
                                                    object_type=node_object)
        values_acq = model.add_int_table(values)
        acquisition_object = model.add_object_type(number=nb_acquisitions)
        acquisitions_done = model.add_set_var(object_type=acquisition_object,
                                              target=set())
        # acquisition_of_node_tb = model.add_int_table(acquisition_of_node)
        acquisition_of_node_tb = model.add_set_table(
            [{x} for x in acquisition_of_node],
            object_type=acquisition_object)
        acquisitions_of_node = model.add_int_table(acquisition_of_node)
        nodes_done = model.add_set_var(object_type=node_object,
                                       target=set())
        closed = model.add_set_var(object_type=node_object,
                                   target=set())
        cur_node = model.add_element_var(object_type=node_object,
                                         target=0)
        last_node = model.add_element_var(object_type=node_object,
                                          target=0)
        must_hit_tracker = model.add_set_var(object_type=node_object, target=set())
        potential_value = model.add_int_var(target=0)
        values_node = model.add_int_table(values_node)
        times_table = model.add_int_table(times_array)
        # cur_node = model.add_int_resource_var(target=0)
        ended = model.add_int_var(target=0)
        model.add_base_case([cur_node == target_,
                             ended == 1])
        end = dp.Transition(name=f"end",
                            cost=-potential_value+dp.IntExpr.state_cost(),
                            effects=[(ended, 1)],
                            preconditions=[cur_node == target_, ended == 0])
        model.add_transition(end, forced=True)
        pick = dp.Transition(name=f"pick_",
                             cost=values_node[cur_node]+dp.IntExpr.state_cost(),
                             effects=[(cur_node, cur_node+1),
                                      (acquisitions_done,
                                       acquisitions_done.union(acquisition_of_node_tb[cur_node])),
                                      (nodes_done, nodes_done.add(cur_node)),
                                      (last_node, cur_node)],
                             preconditions=[cur_node <= nb_nodes-1,
                                            is_inbound_stereo[cur_node] == 0,
                                            is_outbound_stereo[cur_node] == 0,
                                            neighs_table[last_node].contains(cur_node),
                                            acquisitions_done.intersection(acquisition_of_node_tb[cur_node]).is_empty(),
                                            ~closed.contains(cur_node)])
        model.add_transition(pick)

        pick = dp.Transition(name=f"pick_inbound",
                             cost=values_node[cur_node] + dp.IntExpr.state_cost(),
                             # cost=dp.IntExpr.state_cost(),
                             effects=[(cur_node, cur_node + 1),
                                      (acquisitions_done, acquisitions_done.union(acquisition_of_node_tb[cur_node])),
                                      (nodes_done, nodes_done.add(cur_node)),
                                      (potential_value, potential_value+values_node[cur_node]),
                                      (must_hit_tracker, must_hit_tracker.union(to_dos[cur_node])),
                                      (last_node, cur_node)],
                             preconditions=[cur_node <= nb_nodes - 1,
                                            is_inbound_stereo[cur_node] == 1,
                                            neighs_table[last_node].contains(cur_node),
                                            acquisitions_done.intersection(acquisition_of_node_tb[cur_node]).is_empty(),
                                            ~closed.contains(cur_node)])
        model.add_transition(pick, forced=True)

        pick_stereo = dp.Transition(name=f"pick_final_stereo",
                                    cost=values_node[cur_node] + dp.IntExpr.state_cost(),
                                    effects=[(cur_node, cur_node + 1),
                                             (acquisitions_done, acquisitions_done.union(acquisition_of_node_tb[cur_node])),
                                             (nodes_done, nodes_done.add(cur_node)),
                                             (potential_value, potential_value-values_node[cur_node]),
                                             (must_hit_tracker, must_hit_tracker.remove(cur_node)),
                                             (last_node, cur_node)],
                                    preconditions=[cur_node <= nb_nodes - 1,
                                                   is_outbound_stereo[cur_node] == 1,
                                                   must_hit_tracker.contains(cur_node),
                                                   # needed_nodes[cur_node].issubset(nodes_done),
                                                   neighs_table[last_node].contains(cur_node),
                                                   # acquisitions_done.intersection(acquisition_of_node_tb[cur_node]).is_empty(),
                                                   ~closed.contains(cur_node)])
        model.add_transition(pick_stereo, forced=True)

        pick_stereo_fail = dp.Transition(name=f"pick_final_stereo_fail_",
                                         cost=dp.IntExpr.state_cost(),
                                         effects=[(cur_node, cur_node + 1),
                                                  (acquisitions_done,
                                                   acquisitions_done.union(acquisition_of_node_tb[cur_node])),
                                                  (nodes_done, nodes_done.add(cur_node)),
                                                  (must_hit_tracker, must_hit_tracker.remove(cur_node)),
                                                  (last_node, cur_node)],
                                         preconditions=[cur_node <= nb_nodes - 1,
                                                        is_outbound_stereo[cur_node] == 1,
                                                        ~must_hit_tracker.contains(cur_node),
                                                        # needed_nodes[cur_node].issubset(nodes_done),
                                                        neighs_table[last_node].contains(cur_node),
                                                        # acquisitions_done.intersection(acquisition_of_node_tb[cur_node]).is_empty(),
                                                        ~closed.contains(cur_node)])
        model.add_transition(pick_stereo_fail)

        ignore = dp.Transition(name=f"ignore_",
                               cost=dp.IntExpr.state_cost(),
                               effects=[(cur_node, cur_node + 1),
                                        (closed, closed.add(cur_node))],
                               preconditions=[cur_node <= nb_nodes - 1])

        model.add_transition(ignore)
        # acq_of_node = model.add_set_table([{acq_to_index[acquisition_request_id[nodes[i]]]}
        #                                    for i in range(nb_nodes)],
        #                                   object_type=acq_object)
        # model.add_dual_bound(values_acq[acquisitions_done.complement()]//2)
        # model.add_dual_bound(values_acq[acquisitions_done.complement()])

        self.model = model
        self.nodes = nodes
        self.cur_node = cur_node
        self.must_hit = must_hit_tracker
        self.potential_value = potential_value

    def retrieve_solution(self, sol: dp.Solution) -> Solution:
        if self.modeling in {DpModeling.GOTO, DpModeling.GOTO_WITH_DOM}:
            return self.retrieve_solution_goto(sol)
        if self.modeling == DpModeling.PICK_IGNORE or self.modeling is None:
            return self.retrieve_solution_pick_2(sol)

    def retrieve_solution_goto(self, sol: dp.Solution) -> SppSolution:
        def extract_ints(word):
            return tuple(int(num) for num in re.findall(r'\d+', word))
        path = []
        for t in sol.transitions:
            n = extract_ints(t.name)[0]
            path.append(self.nodes[n])
        return SppSolution(problem=self.problem,
                           solution_path=path[:-1])

    def retrieve_solution_pick(self, sol: dp.Solution) -> SppSolution:
        def extract_ints(word):
            return tuple(int(num) for num in re.findall(r'\d+', word))
        path = []
        state = self.model.target_state
        cur_node_val = state[self.cur_node]
        for t in sol.transitions:
            state = t.apply(state, self.model)
            if t.name == "pick_" or t.name == "pickforce_" or "pick" in t.name:
                path.append(self.nodes[cur_node_val])
                if hasattr(self, "must_hit"):
                    print("Musthit ", state[self.must_hit])
                    print("Potential value", state[self.potential_value])
            cur_node_val = state[self.cur_node]
        return SppSolution(problem=self.problem,
                           solution_path=path)

    def retrieve_solution_pick_2(self, sol: dp.Solution) -> SppSolution:
        def extract_ints(word):
            return tuple(int(num) for num in re.findall(r'\d+', word))
        path = []
        state = self.model.target_state
        # cur_node_val = state[self.cur_node]
        # print(state[self.last_node])
        for t in sol.transitions:
            state = t.apply(state, self.model)
            if t.name == "pick_" or t.name == "pickforce_" or "pick" in t.name:
                # path.append(self.nodes[cur_node_val])
                #print(state[self.last_node])
                path.append(self.nodes[state[self.last_node]])
                if hasattr(self, "must_hit"):
                    print("Musthit ", state[self.must_hit])
                    print("Potential value", state[self.potential_value])
            # cur_node_val = state[self.cur_node]
        return SppSolution(problem=self.problem,
                           solution_path=path)


    def set_warm_start(self, solution: SppSolution):
        if self.modeling == DpModeling.PICK_IGNORE:
            self.set_warm_start_pick(solution)
        if self.modeling == DpModeling.GOTO:
            self.set_warm_start_goto(solution)

    def set_warm_start_pick(self, solution: SppSolution) -> None:
        nodes_list = self.nodes
        index_of_nodes = {nodes_list[i]: i for i in range(len(nodes_list))}
        path = solution.solution_path
        current_index = 0
        transitions = []
        cur_i = 0
        while cur_i < len(path):
            node = path[cur_i]
            value = self.problem.graph.nodes[node]["value"]
            index_of_node = index_of_nodes[node]
            while current_index < index_of_node:
                transitions.append(self.transitions["ignore"])
                current_index += 1
            if value >= 10**5:
                transitions.append(self.transitions["pick"])
            else:
                transitions.append(self.transitions["pick"])
            current_index += 1
            cur_i += 1
        last_node = index_of_nodes[path[-1]]
        nb_nodes = self.problem.nb_nodes
        print(last_node, nb_nodes)
        while last_node < nb_nodes-2:
            transitions.append(self.transitions["ignore"])
            last_node += 1
        self.initial_solution = transitions

    def set_warm_start_2(self, solution: SppSolution) -> None:
        nodes_list = self.nodes
        index_of_nodes = {nodes_list[i]: i for i in range(len(nodes_list))}
        path = solution.solution_path
        current_index = 0
        neighs = self.neighs_list[current_index]
        transitions = []
        cur_i = 0
        state = self.model.target_state
        while cur_i < len(path):
            node = path[cur_i]
            index_of_node = index_of_nodes[node]
            index_ = neighs.index(index_of_node)
            new_transitions = [self.transitions["ignore"]]*index_
            for t in new_transitions:
                print(t.name)
                print("Is applicable : ", t.is_applicable(state, self.model))
                state = t.apply(state, self.model)
            transitions.extend(new_transitions)
            transitions.append(self.transitions["pick"])
            print(transitions[-1].name)
            print("Is applicable : ", transitions[-1].is_applicable(state, self.model))
            state = transitions[-1].apply(state, self.model)
            neighs = self.neighs_list[index_of_node]
            cur_i += 1

        last_node = index_of_nodes[path[-1]]
        neighs = self.neighs_list[last_node]
        index_ = neighs.index(self.problem.nb_nodes-1)
        transitions.extend([self.transitions["ignore"]] * index_)

        print(self.model.is_base(state))

        self.initial_solution = transitions

    def set_warm_start_goto(self, solution: SppSolution) -> None:
        nodes_list = self.nodes
        index_of_nodes = {nodes_list[i]: i for i in range(len(nodes_list))}
        path = solution.solution_path
        transitions = []
        for i in range(len(path)):
            transitions.append(self.transitions[f"goto_{index_of_nodes[path[i]]}"])
        transitions.append(self.transitions[f"goto_{len(nodes_list)-1}"])
        self.initial_solution = transitions
