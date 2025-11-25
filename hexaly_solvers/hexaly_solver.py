from abc import abstractmethod
from typing import Optional, List, Any
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import SolverDO, BoundsProviderMixin
from discrete_optimization.generic_tools.callbacks.callback import Callback, CallbackList
import hexaly.optimizer

from discrete_optimization.generic_tools.exceptions import SolveEarlyStop
from discrete_optimization.generic_tools.cp_tools import StatusSolver
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
import logging

logger = logging.getLogger(__name__)


class HexalySolver(SolverDO, BoundsProviderMixin):
    status_solver: Optional[StatusSolver] = None
    ls: Optional[hexaly.optimizer.HexalyOptimizer] = None
    model: Optional[hexaly.optimizer.HxModel] = None

    @abstractmethod
    def init_model(self, **kwargs: Any) -> None:
        ...

    def solve(self,
              callbacks: Optional[List[Callback]] = None,
              time_limit: Optional[float] = 100.0,
              nb_threads: int = 10,
              **kwargs: Any) -> ResultStorage:
        if self.ls is None:
            self.init_model(**kwargs)
        self.ls.param.time_limit = time_limit
        self.ls.param.nb_threads = nb_threads
        logger.info("Solve started")
        callback = CallbackList(callbacks)
        ls_callback = LocalSearchIntermediaryCallback(do_solver=self,
                                                      callback=callback)
        callback.on_solve_start(solver=self)
        self.ls.add_callback(
            hexaly.optimizer.HxCallbackType.ITERATION_TICKED, ls_callback.my_callback
        )
        self.ls.solve()
        logger.info("Solve finished")
        logger.info(f"Stats of solver : {self.ls.statistics}")
        logger.info(f"Status of hexaly-solver : {self.ls.solution.get_status()}")
        self.status_solver = hexaly_status_to_dostatus(hexaly_status=self.ls.solution.get_status())
        logger.info(f"Status of solver : {self.status_solver}")
        sol = self.retrieve_solution()
        fit = self.aggreg_from_sol(sol)
        ls_callback.res.append((sol, fit))
        callback.on_solve_end(ls_callback.res, solver=self)
        return ls_callback.res

    @abstractmethod
    def retrieve_solution(self, **args) -> Solution:
        pass

    def get_current_best_internal_objective_bound(self) -> Optional[float]:
        return self.ls.solution.get_objective_bound(0)

    def get_current_best_internal_objective_value(self) -> Optional[float]:
        return self.ls.model.objectives[0].value


class LocalSearchIntermediaryCallback:

    # Adapted from
    # https://www.localsolver.com/docs/last/technicalfeatures/callbacks.html?highlight=callback
    def __init__(
        self,
        do_solver: HexalySolver,
        callback: Callback
    ):
        super().__init__()
        self.do_solver = do_solver
        self.callback = callback
        self.res = do_solver.create_result_storage([])
        self.nb_solutions = 0
        self.objs_value = []

    def my_callback(self, ls: hexaly.optimizer.HexalyOptimizer, cb_type):
        # stats = ls.statistics
        if cb_type == hexaly.optimizer.HxCallbackType.ITERATION_TICKED:
            objs = tuple([ls.model.objectives[i].value for i in range(ls.model.nb_objectives)])
            if ((len(self.objs_value) == 0 or objs != self.objs_value[-1])
                and ls.solution.status in {hexaly.optimizer.HxSolutionStatus.OPTIMAL,
                                           hexaly.optimizer.HxSolutionStatus.FEASIBLE}):
                logger.info(f"Objs = {objs}")
                logger.info(
                    f"Gaps = {tuple(ls.solution.get_objective_gap(i)*100 for i in range(ls.model.nb_objectives))}")
                sol = self.do_solver.retrieve_solution()
                fit = self.do_solver.aggreg_from_sol(sol)
                self.res.append((sol, fit))
                self.nb_solutions += 1
                self.objs_value.append(objs)
                logger.info(f"Solution #{self.nb_solutions} = {fit}")
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
                    ls.stop()


def hexaly_status_to_dostatus(hexaly_status: hexaly.optimizer.HxSolutionStatus) -> StatusSolver:
    if hexaly_status == hexaly.optimizer.HxSolutionStatus.OPTIMAL:
        return StatusSolver.OPTIMAL
    if hexaly_status == hexaly.optimizer.HxSolutionStatus.FEASIBLE:
        return StatusSolver.SATISFIED
    if hexaly_status == hexaly.optimizer.HxSolutionStatus.INFEASIBLE:
        return StatusSolver.UNKNOWN
    if hexaly_status == hexaly.optimizer.HxSolutionStatus.INCONSISTENT:
        return StatusSolver.UNSATISFIABLE
