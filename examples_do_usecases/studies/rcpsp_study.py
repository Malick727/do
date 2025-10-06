import config.env
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
from discrete_optimization.rcpsp.parser import get_data_available, parse_file, get_data_home
from discrete_optimization.rcpsp.solvers.cpsat import CpSatRcpspSolver
from discrete_optimization.rcpsp.solvers.greedy import GreedyRcpspSolver
from discrete_optimization.rcpsp.solvers.cpm import CpmRcpspSolver
from hexaly_solvers.rcpsp.hexaly_mrcpsp_solver import (HexalyMultiModeRCPSP,
                                                       ModelisationLSolver)
from discrete_optimization.rcpsp.solvers.optal import OptalRcpspSolver
from discrete_optimization.generic_tools.hub_solver.optal.generic_optal import OptalBasicCallback
from discrete_optimization.generic_tools.callbacks.early_stoppers import ObjectiveGapStopper
from discrete_optimization.generic_tools.callbacks.loggers import ObjectiveLogger
import os
logging.basicConfig(level=logging.INFO)

study_name = "rcpsp-study-cpsat-vs-hexaly-vs-optal-greedy-sm"
overwrite = True  # do we overwrite previous study with same name or not? if False, we possibly add duplicates
folder_data = os.path.join(get_data_home(None),
                           "rcpsp/")
print(folder_data)
files = get_data_available(folder_data)
instances = [f for f in files]
time_limit = 10

def create_configs():
    solver_configs = {}
    for proc in [16]:
        p = ParametersCp.default_cpsat()
        p.nb_process = proc
        solver_configs[f"cpsat-proc-{proc}"] = SolverConfig(cls=CpSatRcpspSolver,
                                                            kwargs={"time_limit": time_limit,
                                                                    "ortools_cpsat_solver_kwargs":
                                                                    {"log_search_progress": False},
                                                                    "parameters_cp": p})
    p = ParametersCp.default_cpsat()
    p.nb_process = 16
    solver_configs["optal-cp"] = SolverConfig(OptalRcpspSolver, {"time_limit": time_limit,
                                                                 "parameters_cp": p})
    solver_configs["optal-cp-fdslb-w01"] = SolverConfig(OptalRcpspSolver, {"time_limit": time_limit,
                                                                           "parameters_cp": p,
                                                                           "worker0-1.searchType": "fdslb",
                                                                           "worker0-1.noOverlapPropagationLevel": 4,
                                                                           "worker0-1.cumulPropagationLevel": 3})

    solver_configs["optal-cp-fds-w01"] = SolverConfig(OptalRcpspSolver, {"time_limit": time_limit,
                                                                         "parameters_cp": p,
                                                                         "worker0-1.searchType": "fds",
                                                                         "worker0-1.noOverlapPropagationLevel": 4,
                                                                         "worker0-1.cumulPropagationLevel": 3})
    solver_configs[f"criticalpath"] = SolverConfig(cls=CpmRcpspSolver,
                                                   kwargs={})
    solver_configs[f"greedyevolve"] = SolverConfig(cls=GreedyRcpspSolver,
                                                   kwargs={})
    solver_configs["hexaly-interval"] = SolverConfig(HexalyMultiModeRCPSP,
                                            {"time_limit": time_limit,
                                             "modelisation_type": ModelisationLSolver.INTERVAL,
                                             "nb_threads": 16})
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
                             ObjectiveLogger(step_verbosity_level=logging.INFO,
                                             end_verbosity_level=logging.INFO),
                             ObjectiveGapStopper(objective_gap_rel=0, objective_gap_abs=0)]
            else:
                stats_cb = BasicStatsCallback()
                callbacks = [stats_cb, NbIterationTracker(step_verbosity_level=logging.INFO)]
            if solver_config.cls in {OptalRcpspSolver}:
                stats_cb = NbIterationTracker()
                callbacks = []
            solver = solver_config.cls(color_problem, **solver_config.kwargs)
            solver.init_model(**solver_config.kwargs)
            # solve
            result_store = solver.solve(
                callbacks=callbacks,
                **solver_config.kwargs,
            )
            if solver_config.cls == OptalRcpspSolver:
                solver: OptalRcpspSolver
                stats_cb = OptalBasicCallback(stats=solver.get_output_stats())
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
