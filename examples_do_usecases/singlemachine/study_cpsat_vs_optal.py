import config.env
from copy import copy
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
from discrete_optimization.generic_tools.hub_solver.optal.generic_optal import OptalBasicCallback
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.lp_tools import ParametersMilp
from discrete_optimization.generic_tools.sequential_metasolver import SequentialMetasolver
from discrete_optimization.generic_tools.study import (
    Experiment,
    Hdf5Database,
    SolverConfig,
)
from discrete_optimization.singlemachine.parser import get_data_available, parse_file
from discrete_optimization.singlemachine.solvers.cpsat import CpsatWTSolver
from discrete_optimization.singlemachine.solvers.dp import DpWTSolver
from discrete_optimization.singlemachine.solvers.greedy import GreedySingleMachineSolver
from hexaly_solvers.singlemachine.hexaly_singlemachine_solver import HexalySingleMachineSolver
from cpopt_solvers.singlemachine_cpopt_solver import CPOptSolverSingleMachine
from discrete_optimization.singlemachine.solvers.optal import OptalSingleMachineSolver
from discrete_optimization.singlemachine.solvers.lp import MathOptSingleMachineSolver, mathopt
from discrete_optimization.generic_tools.callbacks.early_stoppers import ObjectiveGapStopper
import didppy as dp
logging.basicConfig(level=logging.INFO)

study_name = "sm-study-with-seq"
overwrite = True  # do we overwrite previous study with same name or not? if False, we possibly add duplicates
time_limit = 30

def create_configs():
    solver_configs = dict()
    proc = 16
    p = ParametersCp.default_cpsat()
    p.nb_process = proc
    solver_configs["optal"] = SolverConfig(OptalSingleMachineSolver,
                                           {"time_limit": time_limit,
                                            "parameters_cp": p,
                                            "worker0-1.searchType": "fds",
                                            "worker0-1.noOverlapPropagationLevel": 4,
                                            "worker0-1.cumulPropagationLevel": 3})
    solver_configs["lp-highs"] = SolverConfig(MathOptSingleMachineSolver,
                                              {'time_limit': time_limit,
                                               "parameters_milp": ParametersMilp.default(),
                                               "mathopt_solver_type": mathopt.SolverType.HIGHS,
                                               })
    solver_configs["lp-cpsat"] = SolverConfig(MathOptSingleMachineSolver,
                                              {'time_limit': time_limit,
                                               "parameters_milp": ParametersMilp.default(),
                                               "mathopt_solver_type": mathopt.SolverType.CP_SAT,
                                               "mathopt_additional_solve_parameters":
                                                   mathopt.SolveParameters(threads=proc)
                                               })
    solver_configs["dp-lnbs"] = SolverConfig(DpWTSolver, {"time_limit": time_limit,
                                                          "threads": proc,
                                                          "add_dominated_transition": False,
                                                          "solver": "LNBS"})

    solver_configs["dp-cabs-True"] = SolverConfig(DpWTSolver, {"time_limit": time_limit,
                                                               "threads": proc,
                                                               "add_dominated_transition": True,
                                                               "solver": "CABS"})
    solver_configs["dp-cabs-False"] = SolverConfig(DpWTSolver, {"time_limit": time_limit,
                                                                "threads": proc,
                                                                "add_dominated_transition": False,
                                                                "solver": "CABS"})

    bricks = [SubBrick(GreedySingleMachineSolver, {})]
    subbrick_seq = [SubBrick(CpsatWTSolver,
                             dict(time_limit=time_limit/10,
                                  parameters_cp=p,
                                  ortools_cpsat_solver_kwargs={"log_search_progress": False})),
                    SubBrick(DpWTSolver,
                             dict(add_dominated_transition=False,
                                  time_limit=time_limit/10,
                                  threads=proc,
                                  solver="LNBS"))]
    for i in range(5):
        bricks.extend(subbrick_seq)
    solver_configs["seq"] = SolverConfig(SequentialMetasolver, {"list_subbricks": bricks})

    solver_configs[f"cpsat-proc-{proc}"] = SolverConfig(cls=CpsatWTSolver,
                                                        kwargs={"time_limit": time_limit,
                                                                "ortools_cpsat_solver_kwargs":
                                                                {"log_search_progress": False},
                                                                "parameters_cp": p})
    solver_configs["hexaly"] = SolverConfig(HexalySingleMachineSolver,
                                            {"time_limit": time_limit,
                                             "nb_threads": proc})
    solver_configs["cpopt"] = SolverConfig(CPOptSolverSingleMachine,
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

problems = parse_file(path=get_data_available()[1])
i = 0
for instance in problems:
    for config_name, solver_config in solver_configs.items():
        logging.info(f"###### Instance {instance}, config {config_name} ######\n\n")
        try:
            # init problem
            problem = instance
            from discrete_optimization.binpack.solvers.greedy import GreedyBinPackSolver
            stats_cb = StatsWithBoundsCallback()
            solver = solver_config.cls(problem=problem, **solver_config.kwargs)
            solver.init_model(**solver_config.kwargs)
            # solve
            transformed_kwargs = copy(solver_config.kwargs)
            if solver_config.cls in {DpWTSolver, SequentialMetasolver}:
                stats_cb = BasicStatsCallback()
                callbacks = [stats_cb, NbIterationTracker(step_verbosity_level=logging.INFO)]
            elif solver_config.cls == OptalSingleMachineSolver:
                callbacks = [NbIterationTracker()]
            else:
                callbacks = [stats_cb,
                             NbIterationTracker(step_verbosity_level=logging.INFO),
                             ObjectiveGapStopper(objective_gap_rel=0, objective_gap_abs=0)]
            result_store = solver.solve(
                callbacks=callbacks,
                **transformed_kwargs,
            )
            if solver_config.cls == OptalSingleMachineSolver:
                solver: OptalSingleMachineSolver
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
            s = str(solver_config.kwargs)
            if solver_config.cls == SequentialMetasolver:
                s = "sequential-solver"
            sconfig = SolverConfig(solver_config.cls,
                                   s)
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
