#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import config.env
import logging
import plotly.io as pio
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.workforce.scheduling.parser import (
    get_data_available,
    parse_json_to_problem,
)
from cpopt_solvers.workforce_cpopt_solver import CPOptAllocScheduling
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.workforce.scheduling.solvers import ObjectivesEnum
from discrete_optimization.workforce.scheduling.utils import (
    compute_changes_between_solution,
    plotly_schedule_comparison,
)

pio.renderers.default = "browser"  # or "vscode", "notebook", "colab", etc.

logging.basicConfig(level=logging.INFO)


def run_cpopt():
    instance = get_data_available()[1]
    problem = parse_json_to_problem(instance)
    print(problem.number_tasks)
    solver = CPOptAllocScheduling(problem)
    solver.init_model(objectives=[ObjectivesEnum.NB_TEAMS,
                                  ObjectivesEnum.MIN_WORKLOAD])
    res = solver.solve(
        parameters_cp=ParametersCp.default_cpsat(),
        time_limit=10
    )
    sol = res[-1][0]
    print(problem.satisfy(sol), problem.evaluate(sol))
    plotly_schedule_comparison(sol, sol, problem, display=True)

def run_cpopt_lexico():
    instance = get_data_available()[1]
    problem = parse_json_to_problem(instance)
    print(problem.number_tasks)
    solver = CPOptAllocScheduling(problem)
    solver.init_model(objectives=[ObjectivesEnum.NB_TEAMS,
                                  ObjectivesEnum.MIN_WORKLOAD])

    from discrete_optimization.generic_tools.lexico_tools import LexicoSolver
    lexico_solver = LexicoSolver(subsolver=solver, problem=problem)
    res = lexico_solver.solve(objectives=[ObjectivesEnum.NB_TEAMS,
                                          ObjectivesEnum.DISPERSION],
                              parameters_cp=ParametersCp.default_cpsat(),
                              time_limit=10)
    sol = res[-1][0]
    print(problem.satisfy(sol), problem.evaluate(sol))
    plotly_schedule_comparison(sol, sol, problem, display=True)


if __name__ == "__main__":
    run_cpopt_lexico()