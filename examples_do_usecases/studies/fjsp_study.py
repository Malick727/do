from optalcp_solvers.solvers.generic_optal import OptalBasicCallback

try:
    import config.env
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
from discrete_optimization.fjsp.parser import get_data_available, parse_file
from discrete_optimization.fjsp.solvers.cpsat import CpSatFjspSolver
from hexaly_solvers.fjsp.hexaly_fjsp_solver import HexalyFJspSolver, ModelingHexalyFJsp
from cpopt_solvers.fjsp_cpopt_solver import CPOFJspSolver
from optalcp_solvers.solvers.fjsp_optal import OptalFJspSolver
from discrete_optimization.generic_tools.callbacks.early_stoppers import ObjectiveGapStopper


logging.basicConfig(level=logging.INFO)

study_name = "fjsp-study-cpsat-vs-hexaly-vs-optal"
overwrite = True  # do we overwrite previous study with same name or not? if False, we possibly add duplicates
instances = get_data_available()

time_limit = 20
def create_configs():
    solver_configs = {}
    p = ParametersCp.default_cpsat()
    p.nb_process = 16
    solver_configs[f"cpsat"] = SolverConfig(cls=CpSatFjspSolver,
                                            kwargs={"time_limit": time_limit,
                                                    "ortools_cpsat_solver_kwargs": {"log_search_progress": True},
                                                    "parameters_cp": p})
    solver_configs["hexaly-partition"] = SolverConfig(HexalyFJspSolver,
                                                      {"time_limit": time_limit,
                                                       "modeling": ModelingHexalyFJsp.PARTITION,
                                                       "nb_threads": 16})
    solver_configs["hexaly-opt-interval"] = SolverConfig(HexalyFJspSolver,
                                                          {"time_limit": time_limit,
                                                           "modeling": ModelingHexalyFJsp.OPTIONAL_INTERVAL,
                                                           "nb_threads": 16})
    solver_configs["optal-cp"] = SolverConfig(OptalFJspSolver, {"time_limit": time_limit,
                                                                "parameters_cp": p})
    solver_configs["cpopt"] = SolverConfig(CPOFJspSolver, {"time_limit": time_limit,
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
            stats_cb = StatsWithBoundsCallback()
            solver = solver_config.cls(color_problem, **solver_config.kwargs)
            solver.init_model(**solver_config.kwargs)
            # solve
            if solver_config.cls == OptalFJspSolver:
                solver: OptalFJspSolver
                stats_cb = NbIterationTracker()
            result_store = solver.solve(
                callbacks=[
                    stats_cb,
                    NbIterationTracker(step_verbosity_level=logging.INFO),
                    ObjectiveGapStopper(objective_gap_rel=0, objective_gap_abs=0),
                ],
                **solver_config.kwargs,
            )
            if solver_config.cls == OptalFJspSolver:
                solver: OptalFJspSolver
                stats_cb = OptalBasicCallback(stats=solver.get_output_stats())
        except Exception as e:
            # failed experiment
            metrics = pd.DataFrame([])
            status = StatusSolver.ERROR
            reason = f"{type(e)}: {str(e)}"
            print("Reason : ", reason)
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
