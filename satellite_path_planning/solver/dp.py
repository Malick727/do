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
        if modeling == DpModeling.PICK_IGNORE:
            self.init_model_pick_ignore(**kwargs)
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
                                         target=0)
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
        model.add_base_case([cur_node == target_])
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
        if self.modeling == DpModeling.GOTO:
            return self.retrieve_solution_goto(sol)
        if self.modeling == DpModeling.PICK_IGNORE or self.modeling is None:
            return self.retrieve_solution_pick(sol)

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

    def set_warm_start_goto(self, solution: SppSolution) -> None:
        nodes_list = self.nodes
        index_of_nodes = {nodes_list[i]: i for i in range(len(nodes_list))}
        path = solution.solution_path
        transitions = []
        for i in range(len(path)):
            transitions.append(self.transitions[f"goto_{index_of_nodes[path[i]]}"])
        transitions.append(self.transitions[f"goto_{len(nodes_list)-1}"])
        self.initial_solution = transitions
