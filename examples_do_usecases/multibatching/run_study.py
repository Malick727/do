import logging
import os
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
                                             GurobiMultibatchingSolverWithLazyConstraint)
from multibatching.solvers.packing_subproblem import (PackingViaBinPacking,
                                                      GreedyPackingForMultibatching,
                                                      CpsatPackingSubproblem)
from multibatching.solvers.two_steps import TwoStepMultibatchingSolver


from examples_do_usecases.multibatching.solve_industrial_use_case import parse_problem

logging.basicConfig(level=logging.INFO)

study_name = "multibatching-study"
overwrite = True  # do we overwrite previous study with same name or not? if False, we possibly add duplicates
problem = parse_problem(False)

def create_configs():
    solver_configs = {}
    params_cp_16 = ParametersCp.default_cpsat()
    params_cp_16.nb_process = 16
    params_cp_8 = ParametersCp.default_cpsat()
    params_cp_8.nb_process = 8
    # CpsatPackingSubproblem, GreedyPackingForMultibatching, PackingViaBinPacking
    for i in [1, 8, 16]:
        params_cp = ParametersCp.default_cpsat()
        params_cp.nb_process = i
        solver_configs[f"two_steps_cpsat{i}_pack_binpackcp"] = SolverConfig(cls=TwoStepMultibatchingSolver,
                                                             kwargs=
                                                             {"flow_solver":
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
        solver_configs[f"two_steps_cpsat{i}_pack_greedy"] = SolverConfig(cls=TwoStepMultibatchingSolver,
                                                                            kwargs=
                                                                            {"flow_solver":
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
        solver_configs[f"two_steps_cpsat{i}_pack_cpfull"] = SolverConfig(cls=TwoStepMultibatchingSolver,
                                                                         kwargs=
                                                                         {"flow_solver":
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
    solver_configs[f"two_steps_gurobi{i}_pack_binpackcp"] = SolverConfig(cls=TwoStepMultibatchingSolver,
                                                          kwargs=
                                                          {"flow_solver":
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
    solver_configs[f"two_steps_gurobi_pack_greedy"] = SolverConfig(cls=TwoStepMultibatchingSolver,
                                                                     kwargs=
                                                                     {"flow_solver":
                                                                      SubBrick(cls=GurobiMultibatchingSolver,
                                                                               kwargs={"time_limit": 100}),
                                                                      "packing_solver":
                                                                          SubBrick(cls=GreedyPackingForMultibatching,
                                                                                   kwargs={}
                                                                                   )
                                                                      })
    solver_configs[f"two_steps_gurobi_pack_cpfull"] = SolverConfig(cls=TwoStepMultibatchingSolver,
                                                                     kwargs=
                                                                     {"flow_solver":
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


solver_configs = create_configs()
database_filepath = f"{study_name}.h5"
if overwrite:
    try:
        os.remove(database_filepath)
    except FileNotFoundError:
        pass

# loop over instances x configs
for config_name, solver_config in solver_configs.items():
    logging.info(f"###### Running config {config_name} ######\n\n")
    solver = solver_config.cls(problem, **solver_config.kwargs)
    solver.init_model(**solver_config.kwargs)
            # solve
    stats_cb = BasicStatsCallback()
    result_store = solver.solve(
                callbacks=[stats_cb,
                           NbIterationTracker(step_verbosity_level=logging.INFO),
                           ObjectiveGapStopper(objective_gap_rel=0, objective_gap_abs=0)],
                **solver_config.kwargs)
    # get metrics and solver status
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
