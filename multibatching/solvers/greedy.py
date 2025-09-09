# Some greedy stuff
from collections import defaultdict
from copy import deepcopy
from typing import Optional, Any

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage

from multibatching.problem import (MultibatchingProblem, MultibatchingSolution, PackingTransport,
                                   TransportLink, TransportType, Product)
from discrete_optimization.generic_tools.do_solver import SolverDO, ParamsObjectiveFunction
from discrete_optimization.binpack.problem import BinPackProblem, BinPackSolution, ItemBinPack
from discrete_optimization.binpack.solvers.cpsat import CpSatBinPackSolver, ModelingBinPack
from discrete_optimization.generic_tools.cp_tools import ParametersCp
class GreedyPackingForMultibatching(SolverDO):

    problem: MultibatchingProblem

    def __init__(self, problem: MultibatchingProblem,
                 params_objective_function: Optional[ParamsObjectiveFunction] = None,
                 **kwargs: Any):
        super().__init__(problem, params_objective_function, **kwargs)

    def solve(self, solution: MultibatchingSolution,
              callbacks: Optional[list[Callback]] = None,
              **kwargs: Any) -> ResultStorage:
        transport_link_flows: dict[TransportLink, list[dict[Product, int]]] = defaultdict(lambda: [])
        aggregated_link_flow: dict[TransportLink, dict[Product, int]] = defaultdict(lambda: dict())
        for flow in solution.list_flows:
            tl = flow.transport_link
            nb_packing = flow.nb_packing
            transport_link_flows[tl].append({p: nb_packing*flow.product_packing[p]
                                             for p in flow.product_packing})
            if tl not in aggregated_link_flow:
                aggregated_link_flow[tl] = {}
            for p in flow.product_packing:
                if flow.product_packing[p] > 0:
                    if p not in aggregated_link_flow[tl]:
                        aggregated_link_flow[tl][p] = 0
                    aggregated_link_flow[tl][p] += nb_packing*flow.product_packing[p]
        packings = []
        for tl in aggregated_link_flow:
            packings_for_tl: list[PackingTransport] = []
            capacity = tl.transport_type.capacity
            current_packing = PackingTransport(transport_link=tl,
                                               product_packing=defaultdict(lambda: 0),
                                               nb_packing=1)
            current_packing.current_size = 0
            packings_for_tl.append(current_packing)
            for p in aggregated_link_flow[tl]:
                for _ in range(aggregated_link_flow[tl][p]):
                    success = False
                    for j in range(len(packings_for_tl)):
                        if packings_for_tl[j].current_size + p.size <= capacity:
                            packings_for_tl[j].product_packing[p] += 1
                            packings_for_tl[j].current_size += p.size
                            success = True
                            break
                    if not success:
                        current_packing = PackingTransport(transport_link=tl,
                                                           product_packing=defaultdict(lambda: 0),
                                                           nb_packing=1)
                        current_packing.product_packing[p] += 1
                        current_packing.current_size = p.size
                        packings_for_tl.append(current_packing)
            packings.extend(packings_for_tl)
        sol = MultibatchingSolution(problem=self.problem,
                                    list_flows=packings)
        fit = self.aggreg_from_sol(sol)
        return self.create_result_storage([(sol, fit)])


class PackingViaBinPacking(SolverDO):
    def solve(self, solution: MultibatchingSolution,
              time_limit_per_link = 2,
              callbacks: Optional[list[Callback]] = None,
              **kwargs: Any) -> ResultStorage:
        transport_link_flows: dict[TransportLink, list[dict[Product, int]]] = defaultdict(lambda: [])
        aggregated_link_flow: dict[TransportLink, dict[Product, int]] = defaultdict(lambda: dict())
        for flow in solution.list_flows:
            tl = flow.transport_link
            nb_packing = flow.nb_packing
            transport_link_flows[tl].append({p: nb_packing*flow.product_packing[p]
                                             for p in flow.product_packing})
            if tl not in aggregated_link_flow:
                aggregated_link_flow[tl] = {}
            for p in flow.product_packing:
                if flow.product_packing[p] > 0:
                    if p not in aggregated_link_flow[tl]:
                        aggregated_link_flow[tl][p] = 0
                    aggregated_link_flow[tl][p] += int(nb_packing*flow.product_packing[p])
        all_packings = []
        for tl in aggregated_link_flow:
            capacity = tl.transport_type.capacity
            list_items = []
            index_to_product = []
            cur_index = 0
            for p in aggregated_link_flow[tl]:
                list_items.extend([ItemBinPack(cur_index+j,
                                               int(p.size))
                                   for j in range(aggregated_link_flow[tl][p])])
                cur_index += aggregated_link_flow[tl][p]
                index_to_product.extend([p for _ in range(aggregated_link_flow[tl][p])])
            problem_bin_pack = BinPackProblem(list_items=list_items,
                                              capacity_bin=int(capacity))
            solver = CpSatBinPackSolver(problem=problem_bin_pack)
            solver.init_model(modeling=ModelingBinPack.SCHEDULING)
            res = solver.solve(parameters_cp=ParametersCp.default_cpsat(),
                               time_limit=time_limit_per_link)
            sol: BinPackSolution = res[-1][0]
            nb_bins = problem_bin_pack.evaluate(sol)["nb_bins"]
            #print("nb bins ", nb_bins)
            packings = [PackingTransport(transport_link=tl, product_packing=defaultdict(lambda: 0),
                                         nb_packing=1) for i in range(nb_bins)]
            for i in range(len(sol.allocation)):
                bin_i = sol.allocation[i]
                packings[bin_i].product_packing[index_to_product[i]] += 1

            all_packings.extend(packings)

        sol = MultibatchingSolution(problem=self.problem,
                                    list_flows=all_packings)
        fit = self.aggreg_from_sol(sol)
        return self.create_result_storage([(sol, fit)])



