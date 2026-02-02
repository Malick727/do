import numpy as np

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
from discrete_optimization.generic_tools.do_problem import get_default_objective_setup
from discrete_optimization.generic_tools.do_solver import StatusSolver
from discrete_optimization.generic_tools.ea.ga import Ga
from discrete_optimization.generic_tools.hub_solver.optal.generic_optal import OptalBasicCallback
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.lp_tools import ParametersMilp
from discrete_optimization.generic_tools.mutations.mixed_mutation import BasicPortfolioMutation
from discrete_optimization.generic_tools.mutations.mutation_catalog import get_available_mutations
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
from discrete_optimization.singlemachine.solvers.greedy import (GreedySingleMachineSolver,
                                                                GreedySingleMachineWSPT)
from discrete_optimization.singlemachine.problem import WeightedTardinessProblem
import didppy as dp
logging.basicConfig(level=logging.INFO)

study_name = "sm-study-with-ls-and-greedy"
overwrite = True  # do we overwrite previous study with same name or not? if False, we possibly add duplicates
time_limit = 30


def create_configs():
    solver_configs = dict()
    proc = 16
    p = ParametersCp.default_cpsat()
    p.nb_process = proc
    solver_configs["greedy1"] = SolverConfig(GreedySingleMachineSolver,
                                             {})
    solver_configs["greedy2"] = SolverConfig(GreedySingleMachineWSPT,
                                             {})
    solver_configs["ga"] = SolverConfig(Ga,
                                        {})
    return solver_configs


solver_configs = create_configs()
database_filepath = f"{study_name}.h5"
if overwrite:
    try:
        os.remove(database_filepath)
    except FileNotFoundError:
        pass

# loop over instances x configs

def solve_ga(problem: WeightedTardinessProblem, callback: BasicStatsCallback):
    params = get_default_objective_setup(problem)
    dummy = problem.get_dummy_solution()
    _, mutations = get_available_mutations(problem, dummy)
    list_mutation = [
        mutate[0].build(problem, dummy, **mutate[1])
        for mutate in mutations
    ]
    mixed_mutation = BasicPortfolioMutation(
        list_mutation, np.ones((len(list_mutation)))
    )
    ga_solver = Ga(
        problem,
        encoding="permutation",
        mutation=mixed_mutation,
        objective_handling=params.objective_handling,
        objectives=params.objectives,
        objective_weights=[-w for w in params.weights],
        pop_size=20,
        max_evals=100000,
        mut_rate=0.1,
        crossover_rate=0.9,
        tournament_size=0.2,  #
    )
    results = ga_solver.solve(callbacks=[callback])
    return ga_solver, results


problems = parse_file(path=get_data_available()[1])
i = 0
for instance in problems:
    for config_name, solver_config in solver_configs.items():
        logging.info(f"###### Instance {instance}, config {config_name} ######\n\n")
        try:
            # init problem
            problem = instance
            from discrete_optimization.binpack.solvers.greedy import GreedyBinPackSolver
            stats_cb = BasicStatsCallback()
            if solver_config.cls == Ga:
                solver, result_store = solve_ga(problem=problem, callback=stats_cb)
            else:
                solver = solver_config.cls(problem=problem)
                result_store = solver.solve(callbacks=[stats_cb])
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
