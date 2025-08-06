import json
import logging

import pandas as pd

from discrete_optimization.generic_tools.callbacks.early_stoppers import ObjectiveGapStopper
from discrete_optimization.generic_tools.callbacks.loggers import NbIterationTracker
from discrete_optimization.generic_tools.callbacks.stats_retrievers import BasicStatsCallback
from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.generic_tools.study import SolverConfig, Hdf5Database, Experiment
from satellite_path_planning.problem import SppProblem, SppSolution
from satellite_path_planning.parser import parse_spp_problem
from satellite_path_planning.solver.solver_bf import (SppBackwardSolver, SppForwardSolver,
                                                      SppBackwardForwardSolver, SppSolverRandomPick)
from satellite_path_planning.solver.astar_based import AStarBasedSppSolver
import os
folder_data = os.environ["SPP_PATH"]
instance_medium = os.path.join(folder_data, "eosp_376_medium_cases/")


def get_instances_path(folder: str = None):
    if folder is None:
        folder = instance_medium
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if "json" in f]


study_name = "spp-study"
overwrite = False  # do we overwrite previous study with same name or not? if False, we possibly add duplicates
instances = get_instances_path()[::-1]


def create_configs():
    solver_configs = {}
    solver_configs["backward"] = SolverConfig(SppBackwardSolver,
                                              kwargs={})
    solver_configs["forward"] = SolverConfig(SppForwardSolver,
                                             kwargs={})
    solver_configs["backforw"] = SolverConfig(SppBackwardForwardSolver,
                                              kwargs={})
    for i in [2, 3, 4]:
        solver_configs[f"backward-mem-{i}"] = SolverConfig(SppBackwardSolver,
                                                           kwargs={"nb_way_memory": i})
        solver_configs[f"forward-mem-{i}"] = SolverConfig(SppForwardSolver,
                                                          kwargs={"nb_way_memory": i})
        solver_configs[f"backforw-mem-{i}"] = SolverConfig(SppBackwardForwardSolver,
                                                           kwargs={"nb_way_memory": i})
    for rand_pick in [0.05, 0.1]:
        solver_configs[f"backward-rand-{rand_pick}"] = SolverConfig(SppSolverRandomPick,
                                                                    kwargs={"random_pick": rand_pick,
                                                                            "forward": False,
                                                                            "nb_way_memory": 1,
                                                                            "nb_iterations": 50,
                                                                            "time_limit": 5
                                                                            })
        solver_configs[f"forward-rand-{rand_pick}"] = SolverConfig(SppSolverRandomPick,
                                                                   kwargs={"random_pick": rand_pick,
                                                                           "forward": True,
                                                                           "nb_way_memory": 1,
                                                                           "nb_iterations": 50,
                                                                           "time_limit": 5
                                                                           })
    solver_configs[f"astar_inspired"] = SolverConfig(AStarBasedSppSolver,
                                                     kwargs={"time_limit": 5})
    return solver_configs


solver_configs = create_configs()
database_filepath = f"{study_name}.h5"
if overwrite:
    try:
        os.remove(database_filepath)
    except FileNotFoundError:
        pass

# loop over instances x configs
basenames = [os.path.basename(p) for p in instances]
instance_to_index = {basenames[i]: i for i in range(len(instances))}
json.dump(instance_to_index, open('instance_to_index_spp_study.json', "w"),
          indent=4)
index_ = 0
for instance in instances:
    for config_name, solver_config in solver_configs.items():
        index_ += 1
        if index_ <= 1663:
            continue
        logging.info(f"###### Instance {instance}, config {config_name} ######\n\n")
        file = instance
        spp_problem = parse_spp_problem(file)
        try:
            # init problem
            file = instance
            spp_problem = parse_spp_problem(file)
            stats_cb = BasicStatsCallback()
            solver = solver_config.cls(spp_problem, **solver_config.kwargs)
            solver.init_model(**solver_config.kwargs)
            # solve
            result_store = solver.solve(
                callbacks=[
                    stats_cb,
                    NbIterationTracker(step_verbosity_level=logging.INFO),
                ],
                **solver_config.kwargs,
            )
        except Exception as e:
            # failed experiment
            metrics = pd.DataFrame([])
            status = StatusSolver.ERROR
            reason = f"{type(e)}: {str(e)}"
        else:
            # get metrics and solver status
            status = solver.status_solver
            metrics = stats_cb.get_df_metrics()
            if "ramp" in spp_problem.solution:
                metrics["rel_fit_ramp"] = metrics["fit"]/max(0.000001, spp_problem.solution["ramp"][0])
            if "optimal" in spp_problem.solution:
                metrics["rel_fit_optimal"] = metrics["fit"]/max(0.000001, spp_problem.solution["optimal"][0])
            if "greedy" in spp_problem.solution:
                metrics["rel_fit_greedy"] = metrics["fit"]/max(0.000001, spp_problem.solution["greedy"][0])
            reason = ""
        # store corresponding experiment
        with Hdf5Database(
            database_filepath
        ) as database:  # ensure closing the database at the end of computation (even if error)
            xp_id = database.get_new_experiment_id()
            print(solver_config.cls, solver_config.kwargs)
            print(reason)
            print(metrics, os.path.basename(instance))
            xp = Experiment.from_solver_config(
                xp_id=xp_id,
                instance=instance_to_index[os.path.basename(instance)],
                config_name=config_name,
                solver_config=solver_config,
                metrics=metrics,
                status=status,
                reason=reason,
            )
            database.store(xp)
