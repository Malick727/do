from typing import Optional, Dict, Hashable, Any
import hexaly.optimizer
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from discrete_optimization.tsp.utils import build_matrice_distance
from discrete_optimization.tsp.problem import TspProblem, Point2DTspProblem, TspSolution
from discrete_optimization.tsp.solvers.tsp_solver import TspSolver
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction, \
    build_aggreg_function_and_params_objective, Solution
from hexaly_solvers.hexaly_solver import HexalySolver
import logging
import numpy as np
from discrete_optimization.tsptw.problem import TSPTWProblem, TSPTWSolution



class TspTwHexalySolver(HexalySolver):
    problem: TSPTWProblem
    def init_model(self, scaling: int=1, **kwargs: Any) -> None:
        ls = hexaly.optimizer.HexalyOptimizer()
        model = ls.model
        nb_customers = self.problem.nb_customers + 1
        # Sequence of customers visited by each truck
        customers_sequences = model.list(nb_customers)
        # All customers must be visited by exactly one truck
        model.constraint(model.count(customers_sequences) == nb_customers)
        # Create Hexaly arrays to be able to access them with an "at" operator
        earliest = model.array([int(scaling*self.problem.time_windows[i][0]) for i in range(nb_customers)])
        latest = model.array([int(scaling*self.problem.time_windows[i][1]) for i in range(nb_customers)])
        dist_matrix = model.array(np.asarray(scaling*self.problem.distance_matrix, dtype=int))
        dist_depot = model.array(np.asarray(scaling*self.problem.distance_matrix[:, self.problem.depot_node], dtype=int))
        model.constraint(customers_sequences[0] == self.problem.depot_node)
        sequence = customers_sequences
        c = model.count(sequence)
        # Distance traveled by each truck
        # dist_lambda = model.lambda_function(lambda i: model.at(dist_matrix,
        #                                                        sequence[i - 1],
        #                                                        sequence[i]))
        # dist_routes = model.sum(model.range(1, c), dist_lambda) + dist_depot[sequence[c - 1]]
        end_time_lambda = model.lambda_function(
                lambda i, prev:
                model.max(model.iif(i == 0,
                                    0,
                                    earliest[sequence[i]]),
                          model.iif(i == 0,
                                    0,
                                    prev + model.at(dist_matrix,
                                                    sequence[i - 1],
                                                    sequence[i]))))
        end_time = model.array(model.range(0, c), end_time_lambda, 0)

        final_time = end_time[nb_customers-1]+dist_depot[sequence[c-1]]

        late_lambda = model.lambda_function(lambda i: model.max(0, end_time[i] - latest[sequence[i]]))
        lateness = model.sum(model.range(0, c), late_lambda)
        total_lateness = model.sum(lateness)
        # Total distance traveled
        # total_distance = dist_routes
        # Objective: minimize the number of trucks used, then minimize the distance traveled
        model.minimize(total_lateness)
        model.minimize(final_time)
        # model.minimize(total_distance)

        model.close()
        self.model = model
        self.ls = ls
        self.variables = {"sequence": sequence}

    def retrieve_solution(self, **args) -> TSPTWSolution:
        perm = [v for v in self.variables["sequence"].value][1:]
        return TSPTWSolution(problem=self.problem,
                             permutation=perm)

