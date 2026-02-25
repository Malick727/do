import os
import sys

this_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(this_folder, "../../../"))

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.sequential_metasolver import SequentialMetasolver
from discrete_optimization.generic_tools.study import SolverConfig
from multibatching.solvers.two_steps import TwoStepMultibatchingSolver
from multibatching.solvers.packing_subproblem import (PackingSubproblemSolver, GreedyPackingForMultibatching,
                                                      PackingViaBinPacking, CpsatPackingSubproblem)
from discrete_optimization.binpack.solvers.asp import AspBinPackingSolver
from discrete_optimization.binpack.solvers.greedy import GreedyBinPackSolver
from discrete_optimization.binpack.solvers.cpsat import CpSatBinPackSolver, ModelingBinPack
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from multibatching.solvers.netx import NetxMultibatchingSolver
from multibatching.solvers.cpsat import CpsatMultibatchingSolver, ModelingMultiBatch
from multibatching.solvers.milp_flow import GurobiMultibatchingSolverUnitFlow, GurobiMultibatchingSolver
from multibatching.solvers.asp import ClingconMultibatchingSolver
from multibatching.solvers.kh.solver_wrapper import MultibatchingMasterWrapper
time_limit = 100


p = ParametersCp.default_cpsat()
p.nb_process = 8

def subbricks_packing(timeout):
    greedy = SubBrick(GreedyPackingForMultibatching, {})
    cpsat = SubBrick(CpsatPackingSubproblem, {"time_limit": int(0.25*timeout), "parameters_cp": p})
    cpsat_binpack = SubBrick(cls=PackingViaBinPacking, kwargs={"bin_packing_solver":
                                                               SubBrick(cls=CpSatBinPackSolver,
                                                                        kwargs=dict(modeling=ModelingBinPack.SCHEDULING,
                                                                                    parameters_cp=p)),
                                                               "time_limit_per_link": 1})
    asp_binpack = SubBrick(cls=PackingViaBinPacking,
                           kwargs={"time_limit_per_link": 1,
                                   "bin_packing_solver": SubBrick(cls=AspBinPackingSolver,
                                                                  kwargs={})})
    return {"greedy": greedy, "cp": cpsat, "cp-binpack": cpsat_binpack, "asp-binpack": asp_binpack}

def subbricks_flow(timeout):
    c = dict(networkx=SubBrick(cls=NetxMultibatchingSolver, kwargs={}),
             cpsat=SubBrick(cls=CpsatMultibatchingSolver, kwargs={"modeling": ModelingMultiBatch.FLOW,
                                                                  "time_limit": int(0.75 * timeout),
                                                                  "parameters_cp": p}),
             milp=SubBrick(cls=GurobiMultibatchingSolver, kwargs={"time_limit": int(0.75 * timeout)}),
             cpsat_shortest=SubBrick(cls=CpsatMultibatchingSolver, kwargs={"modeling": ModelingMultiBatch.FLOW,
                                                                           "time_limit": int(0.75*timeout),
                                                                           "restrict_to_shortest_paths": True,
                                                                           "shortest_path_tolerance": 1.2,
                                                                           "parameters_cp": p}),
             milp_shortest=SubBrick(cls=GurobiMultibatchingSolver, kwargs={"time_limit": int(0.75*timeout),
                                                                          "restrict_to_shortest_paths": True,
                                                                          "shortest_path_tolerance": 1.2}),
             asp=SubBrick(cls=ClingconMultibatchingSolver, kwargs={"time_limit": int(0.75*timeout)}))
    return c

def subbricks_third_step(timeout):
    return dict(cpsat_no_delta = SubBrick(CpsatMultibatchingSolver, dict(modeling=ModelingMultiBatch.UNIT_FLOW,
                                                             time_limit=int(timeout),
                                                             ortools_cpsat_solver_kwargs=dict(log_search_progress=False),
                                                             parameters_cp=p)),
                cpsat_delta = SubBrick(CpsatMultibatchingSolver,
                                       dict(modeling=ModelingMultiBatch.UNIT_FLOW,
                                            time_limit=int(timeout),
                                            ortools_cpsat_solver_kwargs=dict(log_search_progress=False),
                                            parameters_cp=p,
                                            delta_to_solution=2),
                                       kwargs_from_solution={"solution": lambda sol: sol}),
                milp_no_delta = SubBrick(GurobiMultibatchingSolverUnitFlow,
                                         dict(time_limit=int(timeout))),
                milp_delta = SubBrick(GurobiMultibatchingSolverUnitFlow,
                         dict(time_limit=int(timeout),
                              delta_to_solution=2),
                         kwargs_from_solution={"solution": lambda sol: sol}
                         )
    )



def configs_direct_solve(timeout):
    c = dict()
    c["flow-milp"] = SolverConfig(GurobiMultibatchingSolverUnitFlow, {"time_limit": timeout})
    c["flow-cpsat"] = SolverConfig(CpsatMultibatchingSolver, {"modeling": ModelingMultiBatch.UNIT_FLOW,
                                                              "time_limit": timeout,
                                                              "parameters_cp": p})
    return c

def config_2step(timeout):
    c = {}
    flows = subbricks_flow(timeout)
    packings = subbricks_packing(timeout)

    for flow_method in flows:
        for packing_method in packings:
            c[(flow_method, packing_method)] = SolverConfig(cls=TwoStepMultibatchingSolver,
                                                            kwargs={"flow_solver": flows[flow_method],
                                                                    "packing_solver": packings[packing_method]})
    return c


def config_3steps(timeout):
    c = config_2step(timeout//2)
    c_3steps = subbricks_third_step(timeout//2)

    configs = {}

    for flow in ["cpsat", "milp", "cpsat_shortest", "milp_shortest"]:
        for third_step in c_3steps:
            configs[flow+"-greedy-"+third_step] = SolverConfig(cls=SequentialMetasolver,
                                                               kwargs=dict(list_subbricks=c[(flow, "greedy")]))
    return configs



def all_configs(timeout):
    c = dict()
    c.update(configs_direct_solve(timeout))
    c.update(config_2step(timeout))
    c.update(config_3steps(timeout))
    return c