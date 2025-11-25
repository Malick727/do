import logging
import os
import pandas as pd
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
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.rcpsp.solvers.cpsat import CpSatRcpspSolver
from discrete_optimization.rcpsp.solvers.greedy import GreedyRcpspSolver
from discrete_optimization.rcpsp.solvers.cpm import CpmRcpspSolver
from hexaly_solvers.rcpsp.hexaly_mrcpsp_solver import HexalyMultiModeRCPSP, ModelisationLSolver
from discrete_optimization.generic_tools.callbacks.early_stoppers import ObjectiveGapStopper

logging.basicConfig(level=logging.INFO)

study_name = "rcpsp-study-cpsat-vs-hexaly-vs-greedy"
overwrite = True  # do we overwrite previous study with same name or not? if False, we possibly add duplicates
folder_data = "/Users/poveda_g/discrete_optimization_data/rcpsp/RG300"
files = get_data_available(folder_data)
instances = [f for f in files[:50]]


def create_configs():
    solver_configs = {}
    for proc in [16]:
        p = ParametersCp.default_cpsat()
        p.nb_process = proc
        solver_configs[f"cpsat-proc-{proc}"] = SolverConfig(cls=CpSatRcpspSolver,
                                                            kwargs={"time_limit": 1,
                                                                    "ortools_cpsat_solver_kwargs":
                                                                    {"log_search_progress": True},
                                                                    "parameters_cp": p})
    solver_configs[f"criticalpath"] = SolverConfig(cls=CpmRcpspSolver,
                                                   kwargs={})
    solver_configs[f"greedyevolve"] = SolverConfig(cls=GreedyRcpspSolver,
                                                   kwargs={})
    #solver_configs["hexaly-interval"] = SolverConfig(HexalyMultiModeRCPSP,
    #                                        {"time_limit": 5,
    #                                         "modelisation_type": ModelisationLSolver.INTERVAL,
    #                                         "nb_threads": 16})
    #solver_configs["hexaly-opt-interval"] = SolverConfig(HexalyMultiModeRCPSP,
    #                                                 {"time_limit": 5,
    #                                                  "modelisation_type": ModelisationLSolver.OPT_INTERVAL,
    #                                                  "nb_threads": 16})
    return solver_configs


solver_configs = create_configs()
database_filepath = f"{study_name}.h5"
if overwrite:
    try:
        os.remove(database_filepath)
    except FileNotFoundError:
        pass

# loop over instances x configs
for instance in instances:
    for config_name, solver_config in solver_configs.items():
        logging.info(f"###### Instance {instance}, config {config_name} ######\n\n")
        try:
            # init problem
            file = instance
            color_problem = parse_file(file)
            if solver_config.cls in {CpSatRcpspSolver, HexalyMultiModeRCPSP}:
                stats_cb = StatsWithBoundsCallback()
                callbacks = [stats_cb,
                             NbIterationTracker(step_verbosity_level=logging.INFO),
                             ObjectiveGapStopper(objective_gap_rel=0, objective_gap_abs=0)]
            else:
                stats_cb = BasicStatsCallback()
                callbacks = [stats_cb, NbIterationTracker(step_verbosity_level=logging.INFO)]
            solver = solver_config.cls(color_problem, **solver_config.kwargs)
            solver.init_model(**solver_config.kwargs)
            # solve
            result_store = solver.solve(
                callbacks=callbacks,
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
            reason = ""
        # store corresponding experiment
        with Hdf5Database(
            database_filepath
        ) as database:  # ensure closing the database at the end of computation (even if error)
            xp_id = database.get_new_experiment_id()
            xp = Experiment.from_solver_config(
                xp_id=xp_id,
                instance=os.path.basename(instance),
                config_name=config_name,
                solver_config=solver_config,
                metrics=metrics,
                status=status,
                reason=reason,
            )
            database.store(xp)
