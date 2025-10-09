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
from hexaly_solvers.workforce.hexaly_workforce_scheduling import HexalyAllocSchedulingSolver, ModelingHexaly
from discrete_optimization.workforce.scheduling.utils import (
    compute_changes_between_solution,
    plotly_schedule_comparison,
)

pio.renderers.default = "browser"  # or "vscode", "notebook", "colab", etc.

logging.basicConfig(level=logging.INFO)


def run_hexaly():
    instance = get_data_available()[1]
    problem = parse_json_to_problem(instance)
    print(problem.number_tasks)
    solver = HexalyAllocSchedulingSolver(problem)
    solver.init_model(model_dispersion=True,
                      run_lexico=False,
                      run_multistep=False,
                      use_min_workload_to_guide_solver=True,
                      modeling=ModelingHexaly.OPT_INTERVAL)
    res = solver.solve(
        parameters_cp=ParametersCp.default_cpsat(),
        run_multistep=False,
        time_limit=15,
        nb_threads=10
    )
    sol = res[-1][0]
    print(problem.satisfy(sol), problem.evaluate(sol))
    plotly_schedule_comparison(sol, sol, problem, display=True)

if __name__ == "__main__":
    run_hexaly()