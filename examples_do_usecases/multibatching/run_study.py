import logging
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
os.environ["GRB_CURLVERBOSE"] = "1"
os.environ["GRB_NO_REVOKE"] = "1"
import pandas as pd
from discrete_optimization.binpack.solvers.cpsat import ModelingBinPack, CpSatBinPackSolver
from discrete_optimization.generic_tools.callbacks.loggers import NbIterationTracker
from discrete_optimization.generic_tools.callbacks.stats_retrievers import (
    BasicStatsCallback,
    StatsWithBoundsCallback,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.generic_tools.study import (
    Experiment,
    Hdf5Database,
    SolverConfig,
)
from discrete_optimization.generic_tools.callbacks.early_stoppers import ObjectiveGapStopper
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import Hyperparameter, SubBrick
from multibatching.solvers.cpsat import (CpsatMultibatchingSolver,
                                         CpsatBendersMultibatchingSolver,
                                         CpsatPackingSubproblem,
                                         ModelingMultiBatch)
from multibatching.solvers.asp import ClingconMultibatchingSolver
from multibatching.solvers.milp_flow import (GurobiMultibatchingSolver,
                                             GurobiMultibatchingSolverUnitFlow,
                                             GurobiMultibatchingSolverWithLazyConstraint,
                                             GurobiMultibatchingSolverWithLC)
from multibatching.solvers.packing_subproblem import (PackingViaBinPacking,
                                                      GreedyPackingForMultibatching,
                                                      CpsatPackingSubproblem)
from multibatching.solvers.two_steps import TwoStepMultibatchingSolver
from discrete_optimization.generic_tools.sequential_metasolver import SequentialMetasolver
from discrete_optimization.binpack.solvers.asp import AspBinPackingSolver
from examples_do_usecases.multibatching.solve_industrial_use_case import parse_problem



logging.basicConfig(level=logging.INFO)

study_name = "multibatching-study-harbor"
problem = parse_problem(False, True)

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
                                               dict(time_limit=1000,
                                                    delta_to_solution=2),
                                               kwargs_from_solution={"solution": lambda sol: sol}
                                               )]
                                  )
                                  )
    return configs

def config_gurobi_lazy():
    configs = {}
    configs["gurobilazy"] = SolverConfig(cls=GurobiMultibatchingSolverWithLC,
                                         kwargs=dict(time_limit=5000))
    return configs


def config_gurobi_lazy_seq():
    configs = {}
    configs["seqmilplazy"] = SolverConfig(cls=SequentialMetasolver,
                                      kwargs=dict(list_subbricks=[
                                          SubBrick(cls=TwoStepMultibatchingSolver,
                                                   kwargs={"flow_solver":
                                                               SubBrick(cls=GurobiMultibatchingSolver,
                                                                        kwargs={"time_limit": 10}),
                                                           "packing_solver":
                                                               SubBrick(cls=GreedyPackingForMultibatching,
                                                                        kwargs={})}),
                                          SubBrick(GurobiMultibatchingSolverWithLC,
                                                   dict(time_limit=1000))]
                                      )
                                      )
    return configs

overwrite = False  # do we overwrite previous study with same name or not? if False, we possibly add duplicates
solver_configs = config_sequential_milp()
database_filepath = f"{study_name}.h5"
if overwrite:
    try:
        os.remove(database_filepath)
    except FileNotFoundError:
        pass

# loop over instances x configs
for k in range(1): # 5 tries
    for config_name, solver_config in solver_configs.items():
        logging.info(f"###### Running config {config_name} ######\n\n")
        solver = solver_config.cls(problem, **solver_config.kwargs)
        solver.init_model(**solver_config.kwargs)
        # solve
        stats_cb = BasicStatsCallback()
        result_store = solver.solve(
            callbacks=[stats_cb],
            **solver_config.kwargs)
        # get metrics and solver status
        sol = result_store[-1][0]
        print("satisfy", problem.satisfy(sol))
        status = solver.status_solver
        metrics = stats_cb.get_df_metrics()
        reason = ""
        # store corresponding experiment
        with Hdf5Database(
                database_filepath
        ) as database:  # ensure closing the database at the end of computation (even if error)
            xp_id = database.get_new_experiment_id()
            xp = Experiment.from_solver_config(
                xp_id=xp_id,
                instance="main_instance",
                config_name=config_name,
                solver_config=SolverConfig(solver_config.cls,
                                           config_name),
                metrics=metrics,
                status=status,
                reason=reason,
            )
            database.store(xp)
