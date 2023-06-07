from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.do_solver import SolverDO

from path_planning.path_planning import PathPlanning


class PathPlanningSolver(SolverDO):
    def __init__(
        self,
        problem: PathPlanning,
        params_objective_function: ParamsObjectiveFunction = None,
    ):
        self.problem = problem
        (
            self.aggreg_sol,
            self.aggreg_from_dict_values,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            self.problem, params_objective_function=params_objective_function
        )
