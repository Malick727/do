from functools import partial
import time
from typing import Optional, List

from discrete_optimization.generic_tools.callbacks.callback import Callback, CallbackList
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from satellite_path_planning.problem import SppProblem, SppSolution
from satellite_path_planning.solver.solver_bf import repeated_call_to_function, long_path_DAG
from satellite_path_planning.solver.spp_solver import SppSolver
import networkx as nx
from heapq import heappop, heappush
from itertools import count
import logging

def build_heuristic(esd_best,
                    esf_best,
                    graph: nx.Graph,
                    rev: nx.Graph, force_forward=True):
    # esf = cpm(rev, end, start)
    esd = esd_best[0]
    esf = esf_best[0]
    path_d = esd_best[1]
    path_f = esf_best[1]
    max_esd = max(esd.values())
    max_esf = max(esf.values())
    print(max_esd, max_esf)
    start = min(graph.nodes())
    end = max(graph.nodes())
    heuristic_sd = {n: max_esd - esf[n] for n in esd}
    esf_cor = {n: esf[n] - graph.nodes[n]["value"]
               for n in esf}
    graph_of_interest = graph
    st_ = start
    end_ = end
    if max_esf >= max_esd:
        heuristic_to_end_1 = {n: esf_cor[n] for n in esd}
        for n in graph.nodes:
            graph.nodes[n]["heuristic"] = -heuristic_to_end_1[n]
            if n == end:
                graph.nodes[n]["heuristic"] = 0
        reverse = False
    elif max_esd > max_esf:
        esd_cor = {n: esd[n] - graph.nodes[n]["value"]
                   for n in esd}
        heuristic_to_end_1 = {n: esd_cor[n] for n in esd}
        for n in rev.nodes:
            rev.nodes[n]["heuristic"] = -heuristic_to_end_1[n]
            if n == end:
                rev.nodes[n]["heuristic"] = 0
        graph_of_interest = rev
        st_ = end
        end_ = start
        print("consider rev graph")
        reverse = True
    # real_lb = -1.*max(max_esd, max_esf)
    real_lb = -float('inf')
    print("Real lb", real_lb)
    return graph_of_interest, st_, end_, reverse


def astar_longest_p(graph: nx.DiGraph,
                    start_node,
                    goal_node,
                    heuristic_attr,
                    weight_heuristic: float = 1.0,
                    real_lb: float = -float('inf'),
                    max_visit_per_node: int = 10,
                    early_cut: bool = True,
                    reverse: bool = False,
                    res: ResultStorage = None,
                    callback: Callback = None,
                    problem: SppProblem = None,
                    solver: "AStarBasedSppSolver" = None,
                    max_time_seconds: int = 10):
    node_to_request = {}
    request_to_node = {}
    for n in graph.nodes():
        node_to_request[n] = graph.nodes[n]['acquisition_request_id']
        if isinstance(node_to_request[n], str):
            node_to_request[n] = None
        if node_to_request[n] is not None:
            req = node_to_request[n]
            if req not in request_to_node:
                request_to_node[req] = set()
            request_to_node[req].add(n)
    sorted_req = sorted(request_to_node.keys())
    req_to_index = {sorted_req[i]: i for i in range(len(sorted_req))}
    infos_per_node = {}
    infos_per_request = {}
    counter_per_node = {}
    counter_per_request = {}
    hash_per_node = {}
    c = count()
    queue = [(0, next(c), [start_node], start_node, tuple([0 for i in range(len(req_to_index))]),
              0)]
    explored_node = {}
    best_obj = -float("inf")
    immediate_successors = {
        n: set(nx.neighbors(graph, n)) for n in graph.nodes()
    }
    immediate_predecessors = {
        n: set(graph.predecessors(n)) for n in graph.nodes()
    }
    successors_map = {}
    predecessors_map = {}
    esd = {n: None for n in graph.nodes()}
    queue = [(0, 0, next(c), [start_node], start_node, tuple([0 for i in range(len(req_to_index))]),
              0)]
    method = 0
    best_obj = float("inf")
    obj_per_nodes = {}
    obj_done = set()
    t = time.perf_counter()
    best_path = None
    step_ = 2
    while queue:
        # print(len(queue))
        obj, real_obj, counter, path, cur_node, set_req, hash_code = heappop(queue)
        # print(obj, real_obj)
        # if (obj, real_obj) in obj_done:
        #     continue
        # obj_done.add((obj, real_obj))
        if time.perf_counter() > t+max_time_seconds:
            break
        if early_cut:
            if best_obj != float("inf"):
                if -(obj-best_obj)/best_obj >= 0.1:
                    break
        if cur_node in counter_per_node and counter_per_node[cur_node] > max_visit_per_node:
            # print(real_obj, obj)
            continue
        if cur_node == goal_node:
            if real_obj < best_obj:
                best_obj = real_obj
                best_path = path
                print("Improved !", " Best obj ", best_obj)
                if callback is not None:
                    if reverse:
                        sol = SppSolution(problem=problem, solution_path=path[::-1])
                    else:
                        sol = SppSolution(problem=problem, solution_path=path)
                    fit = solver.aggreg_from_sol(sol)
                    res.append((sol, fit))
                    callback.on_step_end(step_, res, solver)
                    step_ += 1
            else:
                print("ni", obj)
        if esd[cur_node] is None:
            esd[cur_node] = obj
        else:
            esd[cur_node] = min(obj, esd[cur_node])
        if cur_node not in counter_per_node:
            counter_per_node[cur_node] = 0
        if cur_node not in hash_per_node:
            hash_per_node[cur_node] = set()
        counter_per_node[cur_node] += 1
        req = node_to_request[cur_node]
        if req not in counter_per_request:
            counter_per_request[req] = 0
        counter_per_request[req] += 1
        # print(counter_per_request[req])
        # if hash_code in hash_per_node[cur_node]:
        #     continue
        hash_per_node[cur_node].add(hash_code)
        if counter_per_node[cur_node] >= 50:
            # print("cpn, ", counter_per_node[cur_node])
            continue
        # if counter_per_request[req] >= 1000:
             # print("cpr, ", counter_per_request[req], obj)
        #     continue
        if cur_node not in infos_per_node:
            infos_per_node[cur_node] = {"paths": [],
                                        "values": []}
        infos_per_node[cur_node]["paths"].append(path)
        # if len(infos_per_node[cur_node]["values"]) > 0:
        #     print(max(infos_per_node[cur_node]["values"]), obj)
        infos_per_node[cur_node]["values"].append(obj)
        for m in graph.neighbors(cur_node):
            req = node_to_request[m]
            val = 0
            nr = list(set_req)
            if req is not None:
                ind = req_to_index[req]
                if set_req[ind] == 0:
                    # val = graph.nodes[m]["operational_score"]
                    val = -graph.nodes[m]["value"]

                    nr[ind] = 1
            nr = tuple(nr)
            if m not in hash_per_node:
                hash_per_node[m] = set()
                obj_per_nodes[m] = set()
            if len(obj_per_nodes[m]) >= 1:
                if real_obj+val < min(obj_per_nodes[m]):
                    pass
                elif min([abs((real_obj+val-x)/(1 if x == 0 else x)) for x in obj_per_nodes[m]]) <= 1e-4:
                    continue
            hash_ = hash(nr)
            if hash_ in hash_per_node[m]:
                continue
            else:
                hash_per_node[m].add(hash_)
                obj_per_nodes[m].add(real_obj + val)
            if m == goal_node:
                if real_obj+val > best_obj:
                    continue
            heappush(queue,
                     (max(real_lb,
                          real_obj+val+weight_heuristic*graph.nodes[m][heuristic_attr]),
                      real_obj+val,
                      next(c),
                      path + [m],
                      m, nr,
                      hash_))
    return best_obj, best_path


class AStarBasedSppSolver(SppSolver):
    problem: SppProblem

    def solve(self,
              callbacks: list[Callback] = None,
              time_limit: int = 20,
              **kwargs) -> ResultStorage:
        callback = CallbackList(callbacks)
        callback.on_solve_start(self)
        graph = self.problem.graph
        rev = graph.reverse(copy=False)
        real_lb = -float('inf')
        st = min(graph.nodes())
        end = max(graph.nodes())
        res = self.create_result_storage([])
        func = partial(long_path_DAG, random_pick=0.1,
                       list_nodes=list(self.problem.graph.nodes()))
        best_sol_forward, esd_best = repeated_call_to_function(func,
                                                               graph=self.problem.graph,
                                                               forward=True,
                                                               nb_iteration=10)
        sol = SppSolution(problem=self.problem,
                          solution_path=esd_best[1][len(self.problem.graph.nodes())])
        fit = self.aggreg_from_sol(sol)
        res.append((sol, fit))
        callback.on_step_end(0, res, self)
        best_val_forward = esd_best[0][end]
        best_path_forward = esd_best[1]
        func = partial(long_path_DAG, random_pick=0.1,
                       list_nodes=list(self.problem.graph.nodes())[::-1])
        best_sol_backward, esf_best = repeated_call_to_function(func,
                                                                graph=self.problem.graph.reverse(),
                                                                forward=False,
                                                                nb_iteration=10)
        sol = SppSolution(problem=self.problem,
                          solution_path=esf_best[1][1])
        fit = self.aggreg_from_sol(sol)
        res.append((sol, fit))
        callback.on_step_end(1, res, self)
        best_val_backward = esf_best[0][st]
        best_path_backward = esf_best[1]
        value_request = {graph.nodes[n]["acquisition_request_id"]: graph.nodes[n]["value"]
                         for n in graph.nodes()}
        graph_of_interest, st_, end_, reverse = build_heuristic(esd_best=esd_best,
                                                                esf_best=esf_best,
                                                                graph=graph,
                                                                rev=rev)
        try:
            bo, path = astar_longest_p(graph_of_interest,
                                       start_node=st_,
                                       goal_node=end_,
                                       weight_heuristic=1.,
                                       heuristic_attr="heuristic",
                                       real_lb=real_lb,
                                       max_visit_per_node=20,
                                       early_cut=True,
                                       reverse=reverse,
                                       callback=callback,
                                       res=res,
                                       solver=self,
                                       problem=self.problem,
                                       max_time_seconds=time_limit)
            print("Best : ", bo)
            print("path", path)
            bo_astar = -bo
            sol = SppSolution(problem=self.problem, solution_path=path)
            fit = self.aggreg_from_sol(sol)
            res.append((sol, fit))
            print(path, fit)
            callback.on_step_end(2, res, self)
        except:
            bo_astar = 0
            path = []
        return res