import random
from typing import Optional

from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage

from examples_do_usecases.oven.run_sa import SimulatedAnnealingSolver
from oven_scheduling_problem.problem import parse_dat_file
from oven_scheduling_problem.solvers.cpsat import (CpsatOvenSolver, CpsatOvenSolverEvolved,
                                                   OvenSchedulingSolution, OvenObjectivesEnum)
from oven_scheduling_problem.solvers.constructive_greedy import OvenGreedySolver
from discrete_optimization.generic_tools.ortools_cpsat_tools import ParametersCp
import os
import logging
logging.basicConfig(level=logging.INFO)

import random


def partial_shuffle(the_list: list, shuffle_fraction: float = 0.2):
    """
    Shuffles a random sub-section of a list in place.

    Args:
        the_list (list): The list to be partially shuffled.
        shuffle_fraction (float): The fraction of the list to shuffle (e.g., 0.2 for 20%).
                                  Must be between 0.0 and 1.0.
    """
    if not (0.0 <= shuffle_fraction <= 1.0):
        raise ValueError("shuffle_fraction must be between 0.0 and 1.0")

    list_len = len(the_list)
    # Determine the number of elements to shuffle
    k = int(list_len * shuffle_fraction)

    # To shuffle, we need at least 2 elements
    if k < 2:
        # If fraction is too small, just swap two random elements
        if list_len > 1:
            idx1, idx2 = random.sample(range(list_len), 2)
            the_list[idx1], the_list[idx2] = the_list[idx2], the_list[idx1]
        return

    # Choose a random starting point for the sublist shuffle
    start_index = random.randint(0, list_len - k)

    # Extract, shuffle, and re-insert the sublist
    sublist = the_list[start_index: start_index + k]
    random.shuffle(sublist)
    the_list[start_index: start_index + k] = sublist

def run():
    folder = "/Users/poveda_g/Documents/git_repos/osp-ls/instances/use-case-1/"
    dat_file = os.path.join(folder, '99RandomOvenSchedulingInstance-n250-k5-a5--2212-22.52.57.dat')
    dat_file = os.path.join(folder, "34RandomOvenSchedulingInstance-n25-k5-a2-WithInitialStates.dat")
    dat_file = os.path.join(folder, "06RandomOvenSchedulingInstance-n10-k2-a5-WithInitialStates.dat")
    dat_file = os.path.join(folder, "40RandomOvenSchedulingInstance-n25-k5-a5-WithInitialStates.dat")
    dat_file = os.path.join(folder, "120RandomOvenSchedulingInstance-n500-k5-a5--2312-11.10.45.dat")
    dat_file = os.path.join(folder, "60RandomOvenSchedulingInstance-n50-k5-a5-WithInitialStates.dat")
    folder = "/Users/poveda_g/Documents/git_repos/osp-ls/instances/use-case-3/"
    dat_file = os.path.join(folder, "50NewRandomOvenSchedulingInstance-n50-k2-a5--2904-15.05.46.dat")
    dat_file = os.path.join(folder, "01NewRandomOvenSchedulingInstance-n10-k2-a2--2904-10.37.39.dat")
    problem = parse_dat_file(dat_file)
    additional_data = problem.additional_data
    solver = CpsatOvenSolver(problem, known_ub=additional_data["ub"])
    solver.init_model(objectives=[OvenObjectivesEnum.PROCESSING_TIME,
                                  OvenObjectivesEnum.TARDINESS,
                                  OvenObjectivesEnum.SETUP_COST],
                      weights=[additional_data["weight_processing"],
                               additional_data["weight_tardiness"],
                               additional_data["weight_setup_cost"]
                               ])
    p = ParametersCp.default_cpsat()
    p.nb_process = 16

    res = solver.solve(parameters_cp=p,
                       time_limit=500,
                       ortools_cpsat_solver_kwargs=dict(log_search_progress=True))
    print(solver.solver.ObjectiveValue() / solver.known_ub, " final ratio")
    sol: OvenSchedulingSolution = res[-1][0]
    print(sol.batches_per_machine)
    print("Nb batches", sum([len(x) for x in sol.batches_per_machine]))
    print(problem.evaluate(sol))
    print(max([sol.timed_schedule[x]["completion_time"]
           for x in sol.timed_schedule]))
    print(problem.satisfy(sol))
    print(sol.timed_schedule)

def run_ws():
    folder = "/Users/poveda_g/Documents/git_repos/osp-ls/instances/use-case-3/"
    dat_file = os.path.join(folder, "100RandomOvenSchedulingInstance-n250-k5-a5--2212-22.59.37.dat")
    dat_file = os.path.join(folder, "102RandomOvenSchedulingInstance-n500-k2-a2--2312-08.39.52.dat")
    dat_file = os.path.join(folder, "103RandomOvenSchedulingInstance-n500-k2-a2--2312-08.42.31.dat")
    problem = parse_dat_file(dat_file)
    additional_data = problem.additional_data
    sa_solver = SimulatedAnnealingSolver(problem=problem,
                                         initial_temperature=20000,
                                         final_temperature=0.1,
                                         cooling_rate=0.995,
                                         iterations_per_temp=100,
                                         max_stagnation_iterations=200,
                                         time_limit_seconds=5,  # Stop after 30 seconds
                                         max_total_iterations=10000)  # Or 2000 iterations
    best_solution_found, best_fitness_found = sa_solver.solve()
    best_solution_found: OvenSchedulingSolution
    solver = CpsatOvenSolverEvolved(problem, known_ub=additional_data["ub"],
                                    max_nb_batch=max([len(m)+2
                                                      for m in best_solution_found.batches_per_machine]))
    solver.init_model(objectives=[OvenObjectivesEnum.PROCESSING_TIME,
                                  OvenObjectivesEnum.TARDINESS,
                                  OvenObjectivesEnum.SETUP_COST],
                      weights=[additional_data["weight_processing"],
                               additional_data["weight_tardiness"],
                               additional_data["weight_setup_cost"]])
    solver.set_warm_start(best_solution_found)
    p = ParametersCp.default_cpsat()
    p.nb_process = 16
    res = solver.solve(parameters_cp=p,
                       time_limit=500,
                       ortools_cpsat_solver_kwargs=dict(log_search_progress=True))
    print(solver.solver.ObjectiveValue() / solver.known_ub, " final ratio")
    sol: OvenSchedulingSolution = res[-1][0]
    print(sol.batches_per_machine)
    print("Nb batches", sum([len(x) for x in sol.batches_per_machine]))
    print(problem.evaluate(sol))
    print(max([sol.timed_schedule[x]["completion_time"]
           for x in sol.timed_schedule]))
    print(problem.satisfy(sol))
    print(sol.timed_schedule)


if __name__ == "__main__":
    run()
