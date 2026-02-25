import os
import logging

from discrete_optimization.generic_tools.callbacks.stats_retrievers import BasicStatsCallback
from discrete_optimization.generic_tools.study import Hdf5Database, Experiment, SolverConfig
from examples_do_usecases.multibatching.paper.configs import all_configs
from examples_do_usecases.multibatching.solve_industrial_use_case import parse_problem


def main():
    study_name = "multibatching-airbus"
    problem = parse_problem(False, False,
                            scale_co2=1 / 1000000)
    overwrite = False  # do we overwrite previous study with same name or not? if False, we possibly add duplicates
    solver_configs = all_configs(timeout=500)
    database_filepath = f"{study_name}.h5"
    if overwrite:
        try:
            os.remove(database_filepath)
        except FileNotFoundError:
            pass
    import gurobipy as gp
    # loop over instances x configs
    for k in range(5): # 5 tries
        for config_name, solver_config in solver_configs.items():
            logging.info(f"###### Running config {config_name} ######\n\n")
            if "milp" in config_name or "gurobi" in config_name:
                with gp.Env() as env:
                    solver = solver_config.cls(problem, **solver_config.kwargs)
                    solver.init_model(**solver_config.kwargs)
                    # solve
                    stats_cb = BasicStatsCallback()
                    result_store = solver.solve(
                        callbacks=[stats_cb],
                        **solver_config.kwargs)
            else:
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
            print(problem.evaluate(sol))
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
