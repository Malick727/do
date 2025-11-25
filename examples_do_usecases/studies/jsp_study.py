try:
    import config.env # Set env variables for Hexaly, Cpopt and Tempo
except:
    pass
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
from discrete_optimization.jsp.parser import get_data_available, parse_file
from discrete_optimization.jsp.solvers.cpsat import CpSatJspSolver
from hexaly_solvers.jsp.hexaly_jsp_solver import HexalyJspSolver
from optalcp_solvers.solvers.generic_optal import OptalBasicCallback
from optalcp_solvers.solvers.jsp_optal import OptalJspSolver
from cpopt_solvers.jsp_cpopt_solver import CPOJspSolver
from discrete_optimization.jsp.solvers.tempo import TempoJspScheduler
from discrete_optimization.generic_tools.hub_solver.tempo.tempo_tools import TempoLogsCallback
from discrete_optimization.generic_tools.callbacks.early_stoppers import ObjectiveGapStopper

logging.basicConfig(level=logging.INFO)

study_name = "jsp-study-cpsat-vs-hexaly-vs-optalcp-vs-tempo-vs-cpo-2"
overwrite = True  # do we overwrite previous study with same name or not? if False, we possibly add duplicates
instances = get_data_available()

time_limit = 30
def create_configs():
    solver_configs = {}
    p =  ParametersCp.default_cpsat()
    p.nb_process = 16
    solver_configs["optal-cp"] = SolverConfig(OptalJspSolver, {"time_limit": time_limit,
                                                               "parameters_cp": p})
    solver_configs["optal-cp-fdslb-w01"] = SolverConfig(OptalJspSolver, {"time_limit": time_limit,
                                                                         "parameters_cp": p,
                                                                         "worker0-1.searchType": "fdslb",
                                                                         "worker0-1.noOverlapPropagationLevel": 4,
                                                                         "worker0-1.cumulPropagationLevel": 3})

    solver_configs["optal-cp-fds-w01"] = SolverConfig(OptalJspSolver, {"time_limit": time_limit,
                                                                       "parameters_cp": p,
                                                                       "worker0-1.searchType": "fds",
                                                                       "worker0-1.noOverlapPropagationLevel": 4,
                                                                       "worker0-1.cumulPropagationLevel": 3})
    solver_configs[f"cpsat"] = SolverConfig(cls=CpSatJspSolver,
                                            kwargs={"time_limit": time_limit,
                                                    "ortools_cpsat_solver_kwargs":
                                                     {"log_search_progress": False},
                                                      "parameters_cp": p})
    solver_configs["hexaly"] = SolverConfig(HexalyJspSolver,
                                            {"time_limit": time_limit,
                                             "nb_threads": 16})
    solver_configs["tempo"] = SolverConfig(TempoJspScheduler, {"time_limit": time_limit,
                                                               "greedy_runs": 5,
                                                               "use_lns": True,
                                                               "search_limit": 1000000})
    solver_configs["cpo"] = SolverConfig(CPOJspSolver,
                                         {"time_limit": time_limit,
                                          "parameters_cp": p})
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
            from discrete_optimization.binpack.solvers.greedy import GreedyBinPackSolver
            stats_cb = StatsWithBoundsCallback()
            solver = solver_config.cls(color_problem, **solver_config.kwargs)
            solver.init_model(**solver_config.kwargs)
            if solver_config.cls == TempoJspScheduler:
                stats_cb = TempoLogsCallback()
            if solver_config.cls == OptalJspSolver:
                solver: OptalJspSolver
                stats_cb = NbIterationTracker()
            # solve
            result_store = solver.solve(
                callbacks=[
                    stats_cb,
                    NbIterationTracker(step_verbosity_level=logging.INFO),
                    ObjectiveGapStopper(objective_gap_rel=0, objective_gap_abs=0),
                ],
                **solver_config.kwargs,
            )
            if solver_config.cls == OptalJspSolver:
                solver: OptalJspSolver
                stats_cb = OptalBasicCallback(stats=solver.get_output_stats())

        except Exception as e:
            # failed experiment
            metrics = pd.DataFrame([])
            status = StatusSolver.ERROR
            reason = f"{type(e)}: {str(e)}"
            print('Error', e)
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
