import os
import logging
import gurobipy as gp
from collections import defaultdict
from discrete_optimization.generic_tools.callbacks.stats_retrievers import BasicStatsCallback
from discrete_optimization.generic_tools.study import Hdf5Database, Experiment, SolverConfig

# Assuming these are available in your environment
from examples_do_usecases.multibatching.paper.configs import all_configs
from multibatching.utils import generate_multibatching_problem

# Set up logging to see progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_instance_definitions():
    """
    Defines parameter sets for different complexity levels.
    """
    return {
        "small": {
            "params": {
                "num_locations": 5,
                "num_transport_types": 2,
                "num_products": 3,
                "demand_sparsity": 0.4,
                "max_demand_abs": 20
            },
            "count": 10  # Number of instances to generate with different seeds
        },
        "medium": {
            "params": {
                "num_locations": 12,
                "num_transport_types": 3,
                "num_products": 6,
                "demand_sparsity": 0.6,
                "max_demand_abs": 50
            },
            "count": 10
        },
        "hard": {
            "params": {
                "num_locations": 25,
                "num_transport_types": 4,
                "num_products": 10,
                "demand_sparsity": 0.8,
                "max_demand_abs": 100
            },
            "count": 5
        }
    }


def main():
    study_name = "multibatching_artificial_benchmark"
    database_filepath = f"{study_name}.h5"
    overwrite = True

    if overwrite and os.path.exists(database_filepath):
        os.remove(database_filepath)
        logging.info(f"Deleted existing database: {database_filepath}")

    # Configuration for solvers (setting a 100s timeout for the benchmark)
    solver_configs = all_configs(timeout=100)
    instance_tiers = get_instance_definitions()

    for tier_name, tier_info in instance_tiers.items():
        logging.info(f"=== Starting Tier: {tier_name} ===")

        for i in range(tier_info["count"]):
            seed = 42 + i
            instance_id = f"{tier_name}_seed_{seed}"

            # 1. Generate the problem instance
            problem = generate_multibatching_problem(
                **tier_info["params"],
                seed=seed
            )
            logging.info(f"Generated instance: {instance_id}")

            # 2. Loop over solver configurations
            for config_name, solver_config in solver_configs.items():
                logging.info(f"Running {config_name} on {instance_id}...")

                try:
                    # Specific handling for Gurobi/MILP if needed
                    if "milp" in config_name.lower() or "gurobi" in str(solver_config.cls.__name__).lower():
                        with gp.Env(empty=True) as env:
                            env.setParam('OutputFlag', 0)  # Quiet mode
                            env.start()
                            solver = solver_config.cls(problem, **solver_config.kwargs)
                            solver.init_model(**solver_config.kwargs)
                            stats_cb = BasicStatsCallback()
                            result_store = solver.solve(callbacks=[stats_cb], **solver_config.kwargs)
                    else:
                        solver = solver_config.cls(problem, **solver_config.kwargs)
                        solver.init_model(**solver_config.kwargs)
                        stats_cb = BasicStatsCallback()
                        result_store = solver.solve(callbacks=[stats_cb], **solver_config.kwargs)

                    # 3. Collect Results
                    status = solver.status_solver
                    metrics = stats_cb.get_df_metrics()

                    # 4. Store in HDF5 Database
                    with Hdf5Database(database_filepath) as database:
                        xp_id = database.get_new_experiment_id()
                        xp = Experiment.from_solver_config(
                            xp_id=xp_id,
                            instance=instance_id,
                            config_name=config_name,
                            solver_config=SolverConfig(solver_config.cls, config_name),
                            metrics=metrics,
                            status=status,
                            reason=f"Tier: {tier_name}",
                        )
                        database.store(xp)

                except Exception as e:
                    logging.error(f"Failed to solve {instance_id} with {config_name}: {e}")

    logging.info(f"Study complete. Results saved to {database_filepath}")


if __name__ == "__main__":
    main()