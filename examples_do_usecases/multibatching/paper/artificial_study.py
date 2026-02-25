
import os
import logging

from discrete_optimization.generic_tools.callbacks.stats_retrievers import BasicStatsCallback
from discrete_optimization.generic_tools.study import Hdf5Database, Experiment, SolverConfig
from examples_do_usecases.multibatching.paper.configs import all_configs
from examples_do_usecases.multibatching.solve_industrial_use_case import parse_problem
from multibatching.utils import generate_multibatching_problem


def get_artificial_problems():
    num_locations: int = 6,
    num_transport_types: int = 2,
    num_products: int = 2,
    min_dist: int = 1,
    max_dist: int = 10,
    min_capacity: int = 10,
    max_capacity: int = 20,
    min_product_size: int = 1,
    max_product_size: int = 5,
    min_product_value: int = 100,
    max_product_value: int = 1000,
    min_tt_cost: int = 10,
    max_tt_cost: int = 100,
    min_tt_speed: float = 0.1,
    max_tt_speed: float = 1.0,
    min_tt_emissions: int = 10,
    max_tt_emissions: int = 100,
    max_links_per_pair: int = 2,  # Max 2 links between any A->B pair (e.g., different transport types)
    demand_sparsity: float = 0.5,  # % of (location, product) pairs that will have non-zero demand
    max_demand_abs: int = 50,  # Max absolute supply/demand value for any product at a location
    seed: int = None  # For reproducibility

    simples = generate_multibatching_problem(num_locations=5, num_products=3,
                                             num_transport_types=3, seed=40)



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
            if "milp" in config_name or "gurobi" in config_name or "gurobi" in str(solver_config.cls.__name__).lower():
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

if __name__ == "__main__":
    main()