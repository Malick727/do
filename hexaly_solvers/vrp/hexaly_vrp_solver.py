import numpy as np
from typing import Optional, Dict, List, Any
import hexaly.optimizer
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from discrete_optimization.vrp.problem import VrpProblem, VrpSolution
from discrete_optimization.vrp.utils import compute_length_matrix
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.vrp.solvers.vrp_solver import VrpSolver
from hexaly_solvers.hexaly_solver import HexalySolver
import logging
logger = logging.getLogger(__name__)


class VrpHexalySolver(HexalySolver, VrpSolver):
    problem: VrpProblem

    def __init__(self, problem: VrpProblem, params_objective_function: Optional[ParamsObjectiveFunction] = None,
                 **kwargs: Any):
        super().__init__(problem, params_objective_function, **kwargs)
        self.ls: Optional[hexaly.optimizer.HexalyOptimizer] = None
        self.model: Optional[hexaly.optimizer.HxModel] = None
        self.vars: Optional[Dict[str, List[hexaly.optimizer.HxExpression]]] = {}

    def init_model(self, **kwargs):
        ls = hexaly.optimizer.HexalyOptimizer()
        model = ls.model
        nb_vehicle = self.problem.vehicle_count
        indexes_start = self.problem.start_indexes
        indexes_end = self.problem.end_indexes
        nb_nodes = self.problem.customer_count
        nodes_non_dummy = [j for j in range(nb_nodes) if j not in indexes_start and j not in indexes_end]
        map_index_index_nodes_to_original_index = {j: nodes_non_dummy[j] for j in range(len(nodes_non_dummy))}
        array_nodes = model.array(nodes_non_dummy)

        nb_non_dummy = len(nodes_non_dummy)
        _, matrix = compute_length_matrix(vrp_problem=self.problem)
        matrix_all_ = np.copy(matrix)
        matrix_all_ = model.array(matrix_all_)
        matrix = np.array([[matrix[i, j] for j in nodes_non_dummy]
                           for i in nodes_non_dummy])
        array_distance = model.array(matrix)
        paths = [model.list(nb_non_dummy) for _ in range(nb_vehicle)]
        model.constraint(model.partition(paths))
        vehicle_used = [(model.count(paths[k]) > 0) for k in range(nb_vehicle)]
        dist_path = [None] * nb_vehicle
        demands = [self.problem.customers[nodes_non_dummy[j]].demand for j in range(nb_non_dummy)]
        demands = model.array(demands)
        for k in range(nb_vehicle):
            sequence = paths[k]
            c = model.count(sequence)
            # The quantity needed in each route must not exceed the truck capacity
            demand_lambda = model.lambda_function(lambda j: demands[j])
            route_quantity = model.sum(sequence, demand_lambda)
            model.constraint(route_quantity <= self.problem.vehicle_capacities[k])
            # Distance traveled by each truck
            dist_lambda = model.lambda_function(lambda i:
                                                model.at(array_distance,
                                                         sequence[i - 1],
                                                         sequence[i]))
            dist_path[k] = (model.sum(model.range(1, c), dist_lambda)
                            + model.iif(c > 0,
                                        model.at(matrix_all_, self.problem.start_indexes[k],
                                                 array_nodes[sequence[0]])
                                        + model.at(matrix_all_, self.problem.end_indexes[k],
                                                   array_nodes[sequence[c-1]]),
                                        0))
        nb_trucks_used = model.sum(vehicle_used)
        # Total distance traveled
        total_distance = model.sum(dist_path)
        # Objective: minimize the number of trucks used, then minimize the distance traveled
        model.minimize(nb_trucks_used)
        model.minimize(total_distance)
        model.minimize(model.max(dist_path))
        self.model = model
        self.vars["paths"] = paths
        self.vars["map_index_index_nodes_to_original_index"] = map_index_index_nodes_to_original_index
        self.ls = ls
        self.model = model
        model.close()

    def retrieve_solution(self) -> ResultStorage:
        paths = [self.vars["paths"][i].value for i in range(len(self.vars["paths"]))]
        sol = VrpSolution(problem=self.problem,
                          list_end_index=self.problem.end_indexes,
                          list_start_index=self.problem.start_indexes,
                          list_paths=[[self.vars["map_index_index_nodes_to_original_index"][x]
                                       for x in paths[i]]
                                      for i in range(len(paths))])
        evaluation = self.aggreg_from_sol(sol)
        logger.info(f"evaluation = {evaluation}")
        logger.info(f"sat = {self.problem.satisfy(sol)}")
        return sol
