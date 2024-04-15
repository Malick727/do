from typing import Optional, Dict, List

import localsolver

from discrete_optimization.vrp.vrp_model import VrpProblem, VrpProblem2D, VrpSolution
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction, \
    build_aggreg_function_and_params_objective


class VrpHexalySolver(SolverDO):
    def __init__(self, problem: VrpProblem,
                 params_objective_function: Optional[ParamsObjectiveFunction] = None):
        self.problem = problem
        (
            self.aggreg_sol,
            self.aggreg_dict,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            problem=self.tsp_model, params_objective_function=params_objective_function
        )
        self.ls: Optional[localsolver.LocalSolver] = None
        self.model: Optional[localsolver.LSModel] = None
        self.vars: Optional[Dict[str, List[localsolver.LSExpression]]] = {}

    def init_model(self):
        ls = localsolver.LocalSolver()
        model = ls.model
        nb_vehicle = self.problem.vehicle_count
        nb_nodes = self.problem.customer_count
