import config.env
from copy import copy
import logging
import os
import pandas as pd

from discrete_optimization.datasets import get_data_home
from discrete_optimization.generic_tools.callbacks.loggers import NbIterationTracker
from discrete_optimization.generic_tools.callbacks.stats_retrievers import (
    BasicStatsCallback,
    StatsWithBoundsCallback,
)
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.generic_tools.hub_solver.optal.generic_optal import OptalBasicCallback
from discrete_optimization.generic_tools.lp_tools import ParametersMilp
from discrete_optimization.generic_tools.study import (
    Experiment,
    Hdf5Database,
    SolverConfig,
)
from discrete_optimization.tsptw.parser import get_data_available, parse_tsptw_file
from discrete_optimization.tsptw.solvers.cpsat import CpSatTSPTWSolver
from discrete_optimization.tsptw.solvers.dp import DpTspTwSolver
from hexaly_solvers.tsptw.hexaly_tsptw_solver import TspTwHexalySolver
from cpopt_solvers.tsptw_cpopt_solver import CPOptSolverTSPTw
from discrete_optimization.tsptw.solvers.optal import OptalTspTwSolver
from discrete_optimization.tsptw.solvers.ortools_routing import OrtoolsTspTwSolver
from discrete_optimization.generic_tools.callbacks.early_stoppers import ObjectiveGapStopper
logging.basicConfig(level=logging.INFO)

study_name = "tsp-tw-study-ohlmann"
overwrite = True  # do we overwrite previous study with same name or not? if False, we possibly add duplicates
scaling_factor = 1


def create_configs(time_limit: int = 10, proc: int = 16):
    p = ParametersCp.default_cpsat()
    p.nb_process = proc
    solver_configs = dict()
    solver_configs[f"ortools-gpdp"] = SolverConfig(cls=OrtoolsTspTwSolver,
                                                   kwargs={"time_limit": time_limit})
    solver_configs["dp-lnbs"] = SolverConfig(DpTspTwSolver, {"time_limit": time_limit,
                                                             "threads": proc,
                                                             "add_dominated_transition": False,
                                                             "solver": "LNBS"})

    solver_configs["dp-cabs-True"] = SolverConfig(DpTspTwSolver, {"time_limit": time_limit,
                                                                  "threads": proc,
                                                                  "add_dominated_transition": True,
                                                                  "solver": "CABS"})
    solver_configs["dp-cabs-False"] = SolverConfig(DpTspTwSolver, {"time_limit": time_limit,
                                                                   "threads": proc,
                                                                   "add_dominated_transition": False,
                                                                   "solver": "CABS"})
    solver_configs[f"cpsat-proc-{proc}"] = SolverConfig(cls=CpSatTSPTWSolver,
                                                        kwargs={"time_limit": time_limit,
                                                                "scaling_factor": 1,
                                                                "ortools_cpsat_solver_kwargs":
                                                                    {"log_search_progress": True},
                                                                "parameters_cp": p})
    solver_configs["hexaly"] = SolverConfig(TspTwHexalySolver,
                                            {"time_limit": time_limit,
                                             "scaling": 1,
                                             "nb_threads": proc})

    solver_configs["cpopt"] = SolverConfig(CPOptSolverTSPTw,
                                           {"time_limit": time_limit,
                                            "scaling": 1,
                                            "parameters_cp": p})
    solver_configs["optal"] = SolverConfig(OptalTspTwSolver,
                                           {"time_limit": time_limit,
                                            "parameters_cp": p,
                                            "scaling": 1,
                                            "worker0-1.searchType": "fds",
                                            "worker0-1.noOverlapPropagationLevel": 4,
                                            "worker0-1.cumulPropagationLevel": 3})

    return solver_configs


solver_configs = create_configs(time_limit=20, proc=16)
database_filepath = f"{study_name}.h5"
if overwrite:
    try:
        os.remove(database_filepath)
    except FileNotFoundError:
        pass

# loop over instances x configs
data_available_ohlmann = get_data_available(
    data_folder=os.path.join(get_data_home(), "tsptw", "OhlmannThomas")
)
problems = [parse_tsptw_file(f) for f in data_available_ohlmann]
i = 0
for instance in problems:
    problem = instance
    problem.distance_matrix = 10 * problem.distance_matrix.astype(int)
    problem.time_windows = [(int(10 * x[0]), int(10 * x[1])) for x in problem.time_windows]
    for config_name, solver_config in solver_configs.items():
        logging.info(f"###### Instance {instance}, config {config_name} ######\n\n")
        try:
            # init problem
            from discrete_optimization.binpack.solvers.greedy import GreedyBinPackSolver
            stats_cb = StatsWithBoundsCallback()
            solver = solver_config.cls(problem=problem, **solver_config.kwargs)
            solver.init_model(**solver_config.kwargs)
            # solve
            transformed_kwargs = copy(solver_config.kwargs)
            if solver_config.cls == DpTspTwSolver or solver_config.cls == OrtoolsTspTwSolver:
                stats_cb = BasicStatsCallback()
                callbacks = [stats_cb, NbIterationTracker(step_verbosity_level=logging.INFO)]
            elif solver_config.cls == OptalTspTwSolver:
                callbacks = [NbIterationTracker()]
            elif solver_config.cls == TspTwHexalySolver:
                callbacks = [stats_cb]
            else:
                callbacks = [stats_cb,
                             NbIterationTracker(step_verbosity_level=logging.INFO),
                             ObjectiveGapStopper(objective_gap_rel=0, objective_gap_abs=0)]
            result_store = solver.solve(
                callbacks=callbacks,
                **transformed_kwargs,
            )
            if solver_config.cls == OptalTspTwSolver:
                solver: OptalTspTwSolver
                stats_cb = OptalBasicCallback(stats=solver.get_output_stats())
        except Exception as e:
            # failed experiment
            metrics = pd.DataFrame([])
            status = StatusSolver.ERROR
            reason = f"{type(e)}: {str(e)}"
            print("Failed ", reason)
        else:
            # get metrics and solver status
            status = solver.status_solver
            metrics = stats_cb.get_df_metrics()
            print(metrics)
            reason = ""
        # store corresponding experiment
        with Hdf5Database(
                database_filepath
        ) as database:  # ensure closing the database at the end of computation (even if error)
            xp_id = database.get_new_experiment_id()
            sconfig = SolverConfig(solver_config.cls,
                                   str(solver_config.kwargs))
            xp = Experiment.from_solver_config(
                xp_id=xp_id,
                instance=f"instance_{i}",
                config_name=config_name,
                solver_config=sconfig,
                metrics=metrics,
                status=status,
                reason=reason,
            )
            database.store(xp)
    i += 1
