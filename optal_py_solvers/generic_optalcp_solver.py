from abc import abstractmethod
from typing import Any, Optional
import optalcp as cp
from discrete_optimization.generic_tools.callbacks.callback import Callback, CallbackList
from discrete_optimization.generic_tools.cp_tools import CpSolver, ParametersCp, SignEnum, StatusSolver
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.exceptions import SolveEarlyStop
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from discrete_optimization.generic_tools.hub_solver.optal.generic_optal import OptalSolver
from discrete_optimization.generic_tools.do_solver import BoundsProviderMixin
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (CategoricalHyperparameter,
                                                                                IntegerHyperparameter)
import inspect
import logging

from optal_py_solvers.async_utils import run_async_synchronously

logger = logging.getLogger(__name__)

class OptalCpSolver(CpSolver, BoundsProviderMixin):
    hyperparameters = [CategoricalHyperparameter(name="searchType", choices=['Auto', 'LNS', 'FDS',
                                                                             'FDSDual', 'SetTimes', 'FDSLB'],
                                                 default="LNS")]
    cp_model: Optional[cp.Model] = None
    early_stopping_exception: Optional[Exception] = None
    current_bound: int|float|None=None
    current_obj: int|float|None=None
    status_solver: StatusSolver = None

    def get_current_best_internal_objective_bound(self) -> Optional[float]:
        return self.current_bound

    def get_current_best_internal_objective_value(self) -> Optional[float]:
        return self.current_obj

    def _solve_sync(self, callbacks: Optional[list[Callback]] = None,
              time_limit: Optional[int] = 60,
              parameters_cp: Optional[ParametersCp] = None,
              **args: Any) -> ResultStorage:
        args = self.complete_with_default_hyperparameters(args)
        kw = {"timeLimit": time_limit,
              "nbWorkers": parameters_cp.nb_process}
        kw.update(args)
        result = self.cp_model.solve(params=cp.Parameters(**kw))
        sol = self.retrieve_solution(result)
        fit = self.aggreg_from_sol(sol)
        if result.solution is not None:
            if result.proof:
                self.status_solver = StatusSolver.OPTIMAL
            else:
                self.status_solver = StatusSolver.SATISFIED
        else:
            if result.proof:
                self.status_solver = StatusSolver.UNSATISFIABLE
            else:
                self.status_solver = StatusSolver.UNKNOWN
        return self.create_result_storage([(sol, fit)])

    async def _solve_async(self, callbacks: Optional[list[Callback]] = None,
                           time_limit: Optional[int] = 60,
                           parameters_cp: Optional[ParametersCp] = None,
                           **args: Any):
        args = self.complete_with_default_hyperparameters(args)
        self.current_bound = None
        callback_do = CallbackList(callbacks)
        solver = cp.Solver()
        callback_optal = OptalSolutionCallback(do_solver=self, optal_solver=solver,
                                               callback=callback_do)
        solver.on_solution = callback_optal.on_solution
        solver.on_bound = callback_optal.on_bound
        kw = {"timeLimit": time_limit,
              "nbWorkers": parameters_cp.nb_process}
        kw.update(args)
        result = await solver.solve(model=self.cp_model, params=cp.Parameters(**kw))
        if result.solution is not None:
            if result.proof:
                self.status_solver = StatusSolver.OPTIMAL
            else:
                self.status_solver = StatusSolver.SATISFIED
        else:
            if result.proof:
                self.status_solver = StatusSolver.UNSATISFIABLE
            else:
                self.status_solver = StatusSolver.UNKNOWN
        return callback_optal.res


    def solve(
        self,
        callbacks: Optional[list[Callback]] = None,
        time_limit: Optional[int] = 60,
        parameters_cp: Optional[ParametersCp] = None,
        **args: Any,
    ) -> ResultStorage:
        return run_async_synchronously(
            self._solve_async(
                callbacks=callbacks,
                time_limit=time_limit,
                parameters_cp=parameters_cp,
                **args
            )
        )


    def minimize_variable(self, var: Any) -> None:
        self.cp_model.minimize(var)

    def add_bound_constraint(self, var: Any, sign: SignEnum, value: int) -> list[Any]:
        if sign == SignEnum.LEQ:
            expr = var <= value
        if sign == SignEnum.UEQ:
            expr = var >= value
        if sign == SignEnum.UP:
            expr = var > value
        if sign == SignEnum.LESS:
            expr = var < value
        if sign == SignEnum.EQUAL:
            expr = (var == value)
        self.cp_model += expr
        return  [expr]

    @abstractmethod
    def retrieve_solution(
            self, result: cp.SolveResult
    ) -> Solution:
        """Return a d-o solution from the variables computed by minizinc.

        Args:
            result: output of the cp.solve
        Returns:

        """
        ...



class OptalSolutionCallback:
    def __init__(self, do_solver: OptalCpSolver,
                 optal_solver: cp.Solver,
                 callback: Callback):
        super().__init__()
        self.do_solver = do_solver
        self.optal_solver = optal_solver
        self.callback = callback
        self.res = do_solver.create_result_storage([])
        self.nb_solutions = 0
        self.current_solve_result: Optional[cp.SolveResult] = None

    def on_solution(self, solution: cp.SolutionEvent) -> None:
        self.do_solver.current_obj = solution.solution.get_objective()
        sol = self.do_solver.retrieve_solution(solution)
        fit = self.do_solver.aggreg_from_sol(sol)
        self.res.append((sol, fit))
        self.nb_solutions += 1
        # end of step callback: stopping?
        logger.info(f"Solution #{self.nb_solutions} = {fit}")
        logger.info(f"Obj = {solution.solution.get_objective()}")
        try:
            stopping = self.callback.on_step_end(
                step=self.nb_solutions, res=self.res, solver=self.do_solver
            )
        except Exception as e:
            self.do_solver.early_stopping_exception = e
            stopping = True
        else:
            if stopping:
                self.do_solver.early_stopping_exception = SolveEarlyStop(
                    f"{self.do_solver.__class__.__name__}.solve() stopped by user callback."
                )
        if stopping:
            self.optal_solver.stop("stopped by usr callback")

    def on_bound(self, event: cp.ObjectiveBoundEntry) -> None:
        self.do_solver.current_bound = event.value
        try:
            stopping = self.callback.on_step_end(
                step=self.nb_solutions, res=self.res, solver=self.do_solver
            )
        except Exception as e:
            self.do_solver.early_stopping_exception = e
            stopping = True
        else:
            if stopping:
                self.do_solver.early_stopping_exception = SolveEarlyStop(
                    f"{self.do_solver.__class__.__name__}.solve() stopped by user callback."
                )
        if stopping:
            self.optal_solver.stop("stopped by usr callback")


