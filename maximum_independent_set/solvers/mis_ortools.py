from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from typing import Any, List, Optional
from maximum_independent_set.solvers.mis_solvers import MisSolver
from maximum_independent_set.mis_problem import MisProblem, MisSolution
from discrete_optimization.generic_tools.cp_tools import ParametersCP, StatusSolver
from ortools.sat.python.cp_model import VarArraySolutionPrinter
from ortools.sat.python import cp_model
import logging
import tqdm
logger = logging.getLogger(__file__)


class MisOrtoolsSolver(MisSolver):
    def init_model(self, **kwargs):
        model = cp_model.CpModel()
        in_set = [model.NewBoolVar(f'in_set_{i}') for i in range(self.problem.number_nodes)]
        # Add constraints to ensure that adjacent vertices are not both in the independent set.
        for edge in tqdm.tqdm(self.problem.edges):
            model.AddBoolOr([in_set[self.problem.nodes_to_index[edge[0]]].Not(),
                             in_set[self.problem.nodes_to_index[edge[1]]].Not()])
        # Maximize the sum of weights of the independent set.
        if self.problem.attribute_aggregate == "size":
            objective = model.NewIntVar(0, self.problem.number_nodes, 'objective')
            model.Add(objective == sum(in_set))
        else:
            objective = model.NewIntVar(0, int(sum(self.problem.attr_list)), 'objective')
            model.Add(objective == sum([in_set[i]*int(self.problem.attr_list[i])
                                       for i in range(len(in_set))]))
        model.Maximize(objective)
        self.model = model
        self.variables = {"in_set": in_set, "objective": objective}

    def solve(self, parameters_cp: Optional[ParametersCP] = None, **kwargs: Any) -> ResultStorage:
        if parameters_cp is None:
            parameters_cp = ParametersCP.default()
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = parameters_cp.time_limit
        solver.parameters.num_search_workers = parameters_cp.nb_process
        verbose_callback = kwargs.get("verbose", False)
        callback = VarArraySolutionPrinter([self.variables["objective"]])
        status = solver.Solve(self.model, callback)
        if status == cp_model.UNKNOWN:
            logger.info("Unknown")
            self.status_solver = StatusSolver.UNKNOWN
        if status == cp_model.INFEASIBLE:
            logger.info("Inf")
            self.status_solver = StatusSolver.UNSATISFIABLE
        if status == cp_model.OPTIMAL:
            logger.info("Optimal")
            self.status_solver = StatusSolver.OPTIMAL
        if status == cp_model.FEASIBLE:
            logger.info("Feasible")
            self.status_solver = StatusSolver.SATISFIED
        logger.info(
            f"Solver finished, status={solver.StatusName(status)}, objective = {solver.ObjectiveValue()},"
            f"best obj bound = {solver.BestObjectiveBound()}"
        )
        print(status, f"Solver finished, status={solver.StatusName(status)}, objective = {solver.ObjectiveValue()},"
            f"best obj bound = {solver.BestObjectiveBound()}")
        chosen: List[int] = [int(solver.Value(var)) for var in self.variables["in_set"]]
        sol = MisSolution(problem=self.problem, chosen=chosen)
        fit = self.aggreg_sol(sol)
        self.infos = {"status": self.status_solver.value,
                      "bound": solver.BestObjectiveBound()}
        return ResultStorage([(sol, fit)], mode_optim=self.params_objective_function.sense_function)

