from typing import Optional
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.do_problem import (ParamsObjectiveFunction,
                                                            build_aggreg_function_and_params_objective)
from maximum_independent_set.mis_problem import MisProblem


class MisSolver(SolverDO):
    problem: MisProblem

    def __init__(self, problem: MisProblem,
                 params_objective_function: Optional[ParamsObjectiveFunction] = None):
        self.problem = problem
        (
            self.aggreg_sol,
            self.aggreg_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.problem,
            params_objective_function=params_objective_function,
        )