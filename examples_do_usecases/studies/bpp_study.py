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
from discrete_optimization.binpack.parser import get_data_available_bppc, parse_bin_packing_constraint_file
from discrete_optimization.binpack.solvers.cpsat import CpSatBinPackSolver, ModelingBinPack
from hexaly_solvers.binpack.hexaly_bpp_solver import HexalyBinPackSolver
from discrete_optimization.generic_tools.callbacks.early_stoppers import ObjectiveGapStopper

logging.basicConfig(level=logging.INFO)

study_name = "bppc-study-cpsat-hexaly"
overwrite = True  # do we overwrite previous study with same name or not? if False, we possibly add duplicates
instances = [os.path.basename(p) for p in get_data_available_bppc() if "BPPC_2" in p]


def create_configs():
    solver_configs = {}
    for proc in [16]:
        p = ParametersCp.default_cpsat()
        p.nb_process = proc
        for modeling in [ModelingBinPack.SCHEDULING]:
            solver_configs[f"cpsat-proc-{proc}-{modeling.name}"] = SolverConfig(cls=CpSatBinPackSolver,
                                                                                kwargs={"time_limit": 60,
                                                                                        "modeling": modeling,
                                                                                        "ortools_cpsat_solver_kwargs":
                                                                                        {"log_search_progress": True},
                                                                                        "parameters_cp": p})
    solver_configs["hexaly"] = SolverConfig(HexalyBinPackSolver,
                                            {"time_limit": 60,
                                             "nb_threads": 16})

    # solver_configs = {"hexaly": solver_configs["hexaly"]}

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
            file = [f for f in get_data_available_bppc() if instance in f][0]
            color_problem = parse_bin_packing_constraint_file(file)
            from discrete_optimization.binpack.solvers.greedy import GreedyBinPackSolver
            s = GreedyBinPackSolver(color_problem)
            res_ = s.solve()
            sol = res_[-1][0]
            upper_bound = len(set(sol.allocation))
            stats_cb = StatsWithBoundsCallback()
            solver = solver_config.cls(color_problem, **solver_config.kwargs)
            solver.init_model(**solver_config.kwargs, upper_bound=upper_bound+20)
            # solve
            result_store = solver.solve(
                callbacks=[
                    stats_cb,
                    NbIterationTracker(step_verbosity_level=logging.INFO),
                    ObjectiveGapStopper(objective_gap_rel=0, objective_gap_abs=0),
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
            reason = ""
        # store corresponding experiment
        with Hdf5Database(
            database_filepath
        ) as database:  # ensure closing the database at the end of computation (even if error)
            xp_id = database.get_new_experiment_id()
            xp = Experiment.from_solver_config(
                xp_id=xp_id,
                instance=instance,
                config_name=config_name,
                solver_config=solver_config,
                metrics=metrics,
                status=status,
                reason=reason,
            )
            database.store(xp)
