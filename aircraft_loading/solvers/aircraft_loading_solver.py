from typing import Optional

from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction, \
    build_aggreg_function_and_params_objective
from discrete_optimization.generic_tools.do_solver import SolverDO
from aircraft_loading.aircraft_loading_problem import AircraftLoadingProblem


class BaseSolver(SolverDO):
    def __init__(
        self,
        problem: AircraftLoadingProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        **kwargs
    ):
        self.problem = problem
        self.problem = problem
        (
            self.aggreg_sol,
            self.aggreg_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.problem,
            params_objective_function=params_objective_function,
        )