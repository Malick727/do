import json
import os
import logging
import sys

from multibatching.parse_config_data import parse_json_to_problem

this_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(this_folder, "../../../"))
from examples_do_usecases.multibatching.paper.callback import StatsCallbackForExpe
from discrete_optimization.generic_tools.study import Hdf5Database, Experiment, SolverConfig
from examples_do_usecases.multibatching.paper.configs import all_configs
from examples_do_usecases.multibatching.solve_industrial_use_case import parse_problem
logging.basicConfig(level=logging.INFO)

def parse_anonymized_problem(fix_problem: bool = False,
                             harbor_constraint: bool = False,
                             scale_co2: float = 1):
    this_folder = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(this_folder,
                               "../datalocal/anonymized_parametric_fixed.json")
    d = json.load(open(config_file, "r"))
    problem = parse_json_to_problem(d, harbor_constraint=harbor_constraint,
                                    scale_co2=scale_co2)
    # Temporary fix = section2 transportable by any means..
    if fix_problem:
        problem.products[3].valid_transports = set(problem.transport_types)
    for product in problem.products:
        size = product.size
        product.valid_transports = set({t for t in product.valid_transports
                                        if t.capacity >= size})

    return problem

def main():
    study_name = "multibatching-anonymized-airbus-final"
    problem = parse_anonymized_problem(False, False,
                                        scale_co2=1 / 1000000)
    overwrite = True  # do we overwrite previous study with same name or not? if False, we possibly add duplicates
    solver_configs = all_configs(timeout=200,
                                 add_1step=False,
                                 add_3steps=False,
                                 add_asp=False,
                                 add_col_gen=False,
                                 verbose=False)
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