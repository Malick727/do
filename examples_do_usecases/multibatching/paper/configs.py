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



def create_configs():
    solver_configs = {}
    # CpsatPackingSubproblem, GreedyPackingForMultibatching, PackingViaBinPacking
    for j in range(1, 10):
        for i in [4, 8, 16]:
            params_cp = ParametersCp.default_cpsat()
            params_cp.nb_process = i
            solver_configs[f"2steps_cpsat{i}_pack_binpackcp{j}"] = SolverConfig(cls=TwoStepMultibatchingSolver,
                                                                                kwargs=
                                                                                {"best_n_flow_solution": j,
                                                                                 "flow_solver":
                                                                                     SubBrick(cls=CpsatMultibatchingSolver,
                                                                                              kwargs={"modeling": ModelingMultiBatch.FLOW,
                                                                                                      "time_limit": 100,
                                                                                                      "ortools_cpsat_solver_kwargs":
                                                                                                          {"log_search_progress": True},
                                                                                                      "parameters_cp": params_cp}),
                                                                                 "packing_solver":
                                                                                     SubBrick(cls=PackingViaBinPacking,
                                                                                              kwargs={"bin_packing_solver":
                                                                                                          SubBrick(cls=CpSatBinPackSolver,
                                                                                                                   kwargs=dict(
                                                                                                                       modeling=ModelingBinPack.SCHEDULING,
                                                                                                                       parameters_cp=ParametersCp.default_cpsat())),
                                                                                                      "time_limit_per_link": 1}
                                                                                              )
                                                                                 })
            solver_configs[f"2steps_cpsat{i}_pack_greedy{j}"] = SolverConfig(cls=TwoStepMultibatchingSolver,
                                                                             kwargs=
                                                                             {"best_n_flow_solution": j,
                                                                              "flow_solver":
                                                                                  SubBrick(cls=CpsatMultibatchingSolver,
                                                                                           kwargs={
                                                                                               "modeling": ModelingMultiBatch.FLOW,
                                                                                               "time_limit": 100,
                                                                                               "ortools_cpsat_solver_kwargs":
                                                                                                   {"log_search_progress": True},
                                                                                               "parameters_cp": params_cp}),
                                                                              "packing_solver":
                                                                                  SubBrick(cls=GreedyPackingForMultibatching,
                                                                                           kwargs={}
                                                                                           )
                                                                              })
            solver_configs[f"2steps_cpsat{i}_pack_cpfull{j}"] = SolverConfig(cls=TwoStepMultibatchingSolver,
                                                                             kwargs=
                                                                             {"best_n_flow_solution": j,
                                                                              "flow_solver":
                                                                                  SubBrick(cls=CpsatMultibatchingSolver,
                                                                                           kwargs={
                                                                                               "modeling": ModelingMultiBatch.FLOW,
                                                                                               "time_limit": 100,
                                                                                               "ortools_cpsat_solver_kwargs":
                                                                                                   {"log_search_progress": True},
                                                                                               "parameters_cp": params_cp}),
                                                                              "packing_solver":
                                                                                  SubBrick(
                                                                                      cls=CpsatPackingSubproblem,
                                                                                      kwargs={"time_limit": 100,
                                                                                              "parameters_cp": params_cp}
                                                                                  )
                                                                              })
        solver_configs[f"2steps_guro_pack_binpackcp{j}"] = SolverConfig(cls=TwoStepMultibatchingSolver,
                                                                        kwargs=
                                                                        {"best_n_flow_solution": j,
                                                                         "flow_solver":
                                                                             SubBrick(cls=GurobiMultibatchingSolver,
                                                                                      kwargs={"time_limit": 100}),
                                                                         "packing_solver":
                                                                             SubBrick(cls=PackingViaBinPacking,
                                                                                      kwargs={"bin_packing_solver":
                                                                                                  SubBrick(cls=CpSatBinPackSolver,
                                                                                                           kwargs=dict(
                                                                                                               modeling=ModelingBinPack.SCHEDULING,
                                                                                                               parameters_cp=ParametersCp.default_cpsat())),
                                                                                              "time_limit_per_link": 1}
                                                                                      )
                                                                         })
        solver_configs[f"2steps_guro_pack_greedy{j}"] = SolverConfig(cls=TwoStepMultibatchingSolver,
                                                                     kwargs=
                                                                     {"best_n_flow_solution": j,
                                                                      "flow_solver":
                                                                          SubBrick(cls=GurobiMultibatchingSolver,
                                                                                   kwargs={"time_limit": 100}),
                                                                      "packing_solver":
                                                                          SubBrick(cls=GreedyPackingForMultibatching,
                                                                                   kwargs={}
                                                                                   )
                                                                      })
        solver_configs[f"2steps_guro_pack_cpfull{j}"] = SolverConfig(cls=TwoStepMultibatchingSolver,
                                                                     kwargs=
                                                                     {"best_n_flow_solution": j,
                                                                      "flow_solver":
                                                                          SubBrick(cls=GurobiMultibatchingSolver,
                                                                                   kwargs={"time_limit": 100}),
                                                                      "packing_solver":
                                                                          SubBrick(
                                                                              cls=CpsatPackingSubproblem,
                                                                              kwargs={"time_limit": 100,
                                                                                      "parameters_cp": params_cp}
                                                                          )
                                                                      })
    solver_configs["clingcon"] = SolverConfig(cls=ClingconMultibatchingSolver,
                                              kwargs={"time_limit": 200})
    return solver_configs


def configs_asp_binpack():
    params_cp = ParametersCp.default_cpsat()
    params_cp.nb_process = 8
    configs = {}
    configs[f"2steps_cpsat{8}_pack_binpackasp"] = SolverConfig(cls=TwoStepMultibatchingSolver,
                                                               kwargs=
                                                               {"best_n_flow_solution": 1,
                                                                "flow_solver":
                                                                    SubBrick(cls=CpsatMultibatchingSolver,
                                                                             kwargs={"modeling": ModelingMultiBatch.FLOW,
                                                                                     "time_limit": 100,
                                                                                     "ortools_cpsat_solver_kwargs":
                                                                                         {"log_search_progress": True},
                                                                                     "parameters_cp": params_cp}),
                                                                "packing_solver":
                                                                    SubBrick(cls=PackingViaBinPacking,
                                                                             kwargs={"bin_packing_solver":
                                                                                         SubBrick(cls=AspBinPackingSolver,
                                                                                                  kwargs={}),
                                                                                     "time_limit_per_link": 1})
                                                                })
    configs[f"2steps_guro_pack_binpackasp"] = SolverConfig(cls=TwoStepMultibatchingSolver,
                                                           kwargs=
                                                           {"best_n_flow_solution": 0,
                                                            "flow_solver":
                                                                SubBrick(cls=GurobiMultibatchingSolver,
                                                                         kwargs={"time_limit": 100}),
                                                            "packing_solver":
                                                                SubBrick(cls=PackingViaBinPacking,
                                                                         kwargs={"bin_packing_solver":
                                                                             SubBrick(
                                                                                 cls=AspBinPackingSolver,
                                                                                 kwargs={}),
                                                                             "time_limit_per_link": 1}
                                                                         )
                                                            })
    return configs

def config_sequential():
    params_cp = ParametersCp.default_cpsat()
    params_cp.nb_process = 10
    configs = {}
    configs["seq10"] = SolverConfig(cls=SequentialMetasolver,
                                  kwargs=dict(list_subbricks=[
                                      SubBrick(cls=TwoStepMultibatchingSolver,
                                               kwargs={"flow_solver":
                                                           SubBrick(cls=CpsatMultibatchingSolver,
                                                                    kwargs=
                                                                    {"modeling": ModelingMultiBatch.FLOW,
                                                                     "time_limit": 200,
                                                                     "ortools_cpsat_solver_kwargs":
                                                                     {"log_search_progress": True},
                                                                     "parameters_cp": params_cp}),
                                                       "packing_solver":
                                                           SubBrick(cls=GreedyPackingForMultibatching,
                                                                    kwargs={})}),
                                      SubBrick(CpsatMultibatchingSolver,
                                               dict(modeling=ModelingMultiBatch.UNIT_FLOW,
                                                    time_limit=200,
                                                    ortools_cpsat_solver_kwargs=dict(log_search_progress=True),
                                                    parameters_cp=params_cp,
                                                    delta_to_solution=2),
                                               kwargs_from_solution={"solution": lambda sol: sol}
                                               )]
                                  )
                                  )
    return configs

def config_sequential_milp():
    configs = {}
    configs["seqmilp"] = SolverConfig(cls=SequentialMetasolver,
                                      kwargs=dict(list_subbricks=[
                                      SubBrick(cls=TwoStepMultibatchingSolver,
                                               kwargs={"flow_solver":
                                                       SubBrick(cls=GurobiMultibatchingSolver,
                                                                kwargs={"time_limit": 100}),
                                                       "packing_solver":
                                                       SubBrick(cls=GreedyPackingForMultibatching,
                                                                kwargs={})}),
                                      SubBrick(GurobiMultibatchingSolverUnitFlow,
                                               dict(time_limit=100,
                                                    delta_to_solution=2),
                                               kwargs_from_solution={"solution": lambda sol: sol}
                                               )]
                                  )
                                  )
    return configs


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
    configs = {}
    configs[("cpsat-greedy-cpsat-nodelta")] = SolverConfig(cls=SequentialMetasolver,
                                            kwargs=dict(list_subbricks=[
                                                c[("cpsat", "greedy")],
                                                SubBrick(CpsatMultibatchingSolver,
                                                         dict(modeling=ModelingMultiBatch.UNIT_FLOW,
                                                              time_limit=int(timeout//2),
                                                              ortools_cpsat_solver_kwargs=dict(log_search_progress=False),
                                                              parameters_cp=p),
                                                         )])
                                            )
    configs[("cpsat-greedy-cpsat-2delta")] = SolverConfig(cls=SequentialMetasolver,
                 kwargs=dict(list_subbricks=[
                     c[("cpsat", "greedy")],
                     SubBrick(CpsatMultibatchingSolver,
                              dict(modeling=ModelingMultiBatch.UNIT_FLOW,
                                   time_limit=int(timeout//2),
                                   ortools_cpsat_solver_kwargs=dict(log_search_progress=False),
                                   parameters_cp=p,
                                   delta_to_solution=2),
                              kwargs_from_solution={"solution": lambda sol: sol}
                              )])
                 )

    configs[("cpsat-greedy-milp-nodelta")] = SolverConfig(cls=SequentialMetasolver,
                 kwargs=dict(list_subbricks=[
                     c[("cpsat", "greedy")],
                     SubBrick(GurobiMultibatchingSolverUnitFlow,
                              dict(time_limit=int(timeout // 2)),
                              )
                 ])
                 )


    configs[("cpsat-greedy-milp-2delta")] = SolverConfig(cls=SequentialMetasolver,
                 kwargs=dict(list_subbricks=[
                     c[("cpsat", "greedy")],
                     SubBrick(GurobiMultibatchingSolverUnitFlow,
                              dict(time_limit=int(timeout // 2),
                                   delta_to_solution=2),
                              kwargs_from_solution={"solution": lambda sol: sol}
                              )
                 ])
                 )

    configs[("milp-greedy-cpsat-nodelta")] = SolverConfig(cls=SequentialMetasolver,
                                                     kwargs=dict(list_subbricks=[
                                                         c[("milp", "greedy")],
                                                         SubBrick(CpsatMultibatchingSolver,
                                                                  dict(modeling=ModelingMultiBatch.UNIT_FLOW,
                                                                       time_limit=int(timeout // 2),
                                                                       ortools_cpsat_solver_kwargs=dict(
                                                                           log_search_progress=False),
                                                                       parameters_cp=p),
                                                                  kwargs_from_solution={"solution": lambda sol: sol}
                                                                  )])
                                                     )
    configs[("milp-greedy-cpsat-2delta")] = SolverConfig(cls=SequentialMetasolver,
                                                    kwargs=dict(list_subbricks=[
                                                        c[("cpsat", "greedy")],
                                                        SubBrick(CpsatMultibatchingSolver,
                                                                 dict(modeling=ModelingMultiBatch.UNIT_FLOW,
                                                                      time_limit=int(timeout // 2),
                                                                      ortools_cpsat_solver_kwargs=dict(
                                                                          log_search_progress=False),
                                                                      parameters_cp=p,
                                                                      delta_to_solution=2),
                                                                 kwargs_from_solution={"solution": lambda sol: sol}
                                                                 )])
                                                    )
    configs[("milp-greedy-milp-nodelta")] = SolverConfig(cls=SequentialMetasolver,
                                                          kwargs=dict(list_subbricks=[
                                                              c[("milp", "greedy")],
                                                              SubBrick(GurobiMultibatchingSolverUnitFlow,
                                                                       dict(time_limit=int(timeout // 2)))]
                                                          ))
    configs[("milp-greedy-milp-2delta")] = SolverConfig(cls=SequentialMetasolver,
                                                         kwargs=dict(list_subbricks=[
                                                             c[("cpsat", "greedy")],
                                                             SubBrick(GurobiMultibatchingSolverUnitFlow,
                                                                      dict(time_limit=int(timeout // 2),
                                                                           delta_to_solution=2),
                                                                      kwargs_from_solution={
                                                                          "solution": lambda sol: sol}
                                                                      )])
                                                         )
    return configs



def all_configs(timeout):
    c = dict()
    c.update(configs_direct_solve(timeout))
    c.update(config_2step(timeout))
    c.update(config_3steps(timeout))