from typing import Optional, Dict, Hashable
import localsolver

from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from discrete_optimization.tsp.common_tools_tsp import build_matrice_distance
from discrete_optimization.tsp.tsp_model import TSPModel, TSPModelDistanceMatrix, SolutionTSP
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction, \
    build_aggreg_function_and_params_objective
import logging
logger = logging.getLogger(__name__)


class TspHexalySolver(SolverDO):
    def __init__(self, problem: TSPModelDistanceMatrix,
                 params_objective_function: Optional[ParamsObjectiveFunction] = None):
        self.problem = problem
        (
            self.aggreg_sol,
            self.aggreg_from_dict_values,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            self.problem, params_objective_function=params_objective_function
        )
        self.ls: Optional[localsolver.LocalSolver] = None
        self.model: Optional[localsolver.LSModel] = None
        self.vars: Optional[Dict[str, localsolver.LSExpression]] = {}

    def init_model(self, **kwargs):
        ls = localsolver.LocalSolver()
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
        model.constraint(routing[0]==self.problem.start_index)
        print(self.problem.start_index, self.problem.end_index)
        print("Nb nodes ", nb_nodes)
        if self.problem.start_index == self.problem.end_index:
            objective = (model.sum(model.range(nb_nodes-1), objective_lambda) +
                         model.at(matrix, routing[nb_nodes-1], routing[0]))
        else:
            model.constraint(routing[0] == self.problem.start_index)
            objective = model.sum(model.range(nb_nodes-1),
                                  objective_lambda)
            model.constraint(routing[nb_nodes-1] == self.problem.end_index)
        self.vars["routing"] = routing
        model.minimize(objective)
        self.ls = ls
        self.model = model
        model.close()

    def solve(self, **kwargs):
        if self.ls is None:
            self.init_model(**kwargs)
        time_limit = kwargs.get("timeout_seconds", 100)
        self.ls.param.time_limit = time_limit
        logger.info("Solve started")
        self.ls.solve()
        logger.info("Solve finished")
        logger.info(f"Stats of solver : {self.ls.statistics}")
        return self.retrieve_solution()

    def retrieve_solution(self) -> ResultStorage:
        value = self.vars["routing"].value
        sol = SolutionTSP(problem=self.problem,
                          permutation=[v for v in value][1:])
        eval = self.aggreg_sol(sol)
        logger.info(f"eval = {eval}")
        logger.info(f"sat = {self.problem.satisfy(sol)}")
        return ResultStorage([(sol, eval)],
                             mode_optim=self.params_objective_function.sense_function)

