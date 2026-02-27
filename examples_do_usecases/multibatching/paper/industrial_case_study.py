import os
import logging
import sys

this_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(this_folder, "../../../"))
from examples_do_usecases.multibatching.paper.callback import StatsCallbackForExpe
from discrete_optimization.generic_tools.study import Hdf5Database, Experiment, SolverConfig
from examples_do_usecases.multibatching.paper.configs import all_configs
from examples_do_usecases.multibatching.solve_industrial_use_case import parse_problem
logging.basicConfig(level=logging.INFO)

def main():
    study_name = "multibatching-airbus"
    problem = parse_problem(False, False,
                            scale_co2=1 / 1000000)
    overwrite = True  # do we overwrite previous study with same name or not? if False, we possibly add duplicates
    solver_configs = all_configs(timeout=80,
                                 add_col_gen=False,
                                 verbose=True)
    sc = {}
    for key in solver_configs:
        if isinstance(key, tuple):
            sc[key[0]+'-'+key[1]] = solver_configs[key]
        else:
            sc[str(key)] = solver_configs[key]
    for key in sc:
        if len(key) > 32:
            print(key, len(key))
    solver_configs = sc
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
            if "milp" in config_name or "gurobi" in config_name or "gurobi" in str(solver_config.cls.__name__).lower():
                with gp.Env() as env:
                    solver = solver_config.cls(problem, **solver_config.kwargs)
                    solver.init_model(**solver_config.kwargs)
                    # solve
                    stats_cb = StatsCallbackForExpe()
                    result_store = solver.solve(
                        callbacks=[stats_cb],
                        **solver_config.kwargs)
            else:
                solver = solver_config.cls(problem, **solver_config.kwargs)
                solver.init_model(**solver_config.kwargs)
                # solve
                stats_cb = StatsCallbackForExpe()
                result_store = solver.solve(
                    callbacks=[stats_cb],
                    **solver_config.kwargs)
            # get metrics and solver status
            if len(result_store) > 0:
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
                    config_name=str(config_name),
                    solver_config=SolverConfig(solver_config.cls,
                                               config_name),
                    metrics=metrics,
                    status=status,
                    reason=reason,
                )
                database.store(xp)

if __name__ == "__main__":
    main()