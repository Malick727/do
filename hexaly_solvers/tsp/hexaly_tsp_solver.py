from typing import Optional, Dict, Hashable, Any
import hexaly.optimizer
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from discrete_optimization.tsp.utils import build_matrice_distance
from discrete_optimization.tsp.problem import TspProblem, Point2DTspProblem, TspSolution
from discrete_optimization.tsp.solvers.tsp_solver import TspSolver
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction, \
    build_aggreg_function_and_params_objective
from hexaly_solvers.hexaly_solver import HexalySolver
import logging
logger = logging.getLogger(__name__)


class TspHexalySolver(HexalySolver, SolverDO):
    problem: TspProblem

    def __init__(self, problem: TspProblem,
                 params_objective_function: Optional[ParamsObjectiveFunction] = None, **kwargs: Any):
        super().__init__(problem, params_objective_function, **kwargs)
        self.ls: Optional[hexaly.optimizer.HexalyOptimizer] = None
        self.model: Optional[hexaly.optimizer.HxModel] = None
        self.vars: Optional[Dict[str, hexaly.optimizer.HxExpression]] = {}

    def init_model(self, **kwargs):
        ls = hexaly.optimizer.HexalyOptimizer()
        #
        # Declare the optimization model
        #
        model = ls.model
        nb_nodes = len(self.problem.list_points)
        matrix = build_matrice_distance(nodeCount=nb_nodes,
                                        method=self.problem.evaluate_function_indexes)
        matrix = model.array(matrix)
        routing = model.list(len(self.problem.list_points))
        model.constraint(model.count(routing) == len(self.problem.list_points))
        objective_lambda = model.lambda_function(lambda i: model.at(matrix, routing[i], routing[i+1]))
        model.constraint(routing[0] == self.problem.start_index)
        if self.problem.start_index == self.problem.end_index:
            objective = (model.sum(model.range(nb_nodes-1), objective_lambda) +
                         model.at(matrix, routing[nb_nodes-1], routing[0]))
        else:
            objective = model.sum(model.range(nb_nodes-1),
                                  objective_lambda)
            model.constraint(routing[nb_nodes-1] == self.problem.end_index)
        self.vars["routing"] = routing
        model.minimize(objective)
        self.ls = ls
        self.model = model
        model.close()

    def retrieve_solution(self) -> TspSolution:
        value = self.vars["routing"].value
        sol = TspSolution(problem=self.problem,
                          start_index=self.problem.start_index,
                          end_index=self.problem.end_index,
                          permutation=[v for v in value][1:])
        evaluation = self.aggreg_from_sol(sol)
        logger.info(f"eval = {evaluation}")
        logger.info(f"sat = {self.problem.satisfy(sol)}")
        return sol

