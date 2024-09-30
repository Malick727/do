import docplex
from abc import abstractmethod
from docplex.cp.solution import CpoSolveResult
from docplex.cp.solution import (SOLVE_STATUS_UNKNOWN, SOLVE_STATUS_INFEASIBLE,
                                 SOLVE_STATUS_OPTIMAL, SOLVE_STATUS_FEASIBLE,
                                 SOLVE_STATUS_JOB_ABORTED, SOLVE_STATUS_JOB_FAILED)
from docplex.cp.solver.cpo_callback import EVENT_SOLUTION, CpoCallback
from typing import Optional, List, Any
from discrete_optimization.generic_tools.callbacks.callback import Callback, CallbackList
from discrete_optimization.generic_tools.cp_tools import CPSolver, ParametersCP, StatusSolver
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.exceptions import SolveEarlyStop
from docplex.cp.model import CpoModel
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
import logging
logger = logging.getLogger(__name__)


def cpostatus_to_dostatus(status_from_cpo: str)->StatusSolver:
    if status_from_cpo == SOLVE_STATUS_OPTIMAL:
        return StatusSolver.OPTIMAL
    if status_from_cpo == SOLVE_STATUS_FEASIBLE:
        return StatusSolver.SATISFIED
    if status_from_cpo in {SOLVE_STATUS_UNKNOWN, SOLVE_STATUS_JOB_ABORTED, SOLVE_STATUS_JOB_FAILED}:
        return StatusSolver.UNKNOWN
    if status_from_cpo == SOLVE_STATUS_INFEASIBLE:
        return StatusSolver.UNSATISFIABLE


class CPOSolver(CPSolver):
    """Generic ortools cpo solver."""

    cp_model: Optional[CpoModel] = None
    early_stopping_exception: Optional[Exception] = None

    @abstractmethod
    def retrieve_solution(self, sres: CpoSolveResult) -> Solution:
        """Construct a do solution from the cpo solver internal solution.
        Args:
            sres: CpoSolveResult object obtained when invoking a cpo callback

        Returns:
            the intermediate solution, at do format.
        """
        ...

    def solve(
        self,
        callbacks: Optional[List[Callback]] = None,
        parameters_cp: Optional[ParametersCP] = None,
        time_limit: Optional[float] = 100.0,
        **kwargs: Any,
    ) -> ResultStorage:
        """Solve the problem with a CPOpt solver

        Args:
            callbacks: list of callbacks used to hook into the various stage of the solve
            parameters_cp: parameters specific to cp solvers.
                We use here only `parameters_cp.time_limit` and `parameters_cp.nb_process`.
            time_limit: timeout (seconds)
            **kwargs: keyword arguments passed to `self.init_model()`

        Returns:

        A dedicated cpo callback is used to:
        - update a resultstorage each time a new solution is found by the cpo solver.
        - call the user (do) callbacks at each new solution, with the possibility of early stopping if the callback return True.

        This ortools callback use the method `self.retrieve_solution()` to reconstruct a do Solution from the cpsat solve internal state.

        """
        self.early_stopping_exception = None
        callbacks_list = CallbackList(callbacks=callbacks)
        callbacks_list.on_solve_start(solver=self)
        if self.cp_model is None:
            self.init_model(**kwargs)
        if parameters_cp is None:
            parameters_cp = ParametersCP.default_cpsat()
        callback = CpoSolutionCallback(do_solver=self,
                                       callback=callbacks_list)
        self.cp_model.add_solver_callback(callback)
        result: CpoSolveResult = self.cp_model.solve(
            TimeLimit=time_limit,
            execfile="/Applications/CPLEX_Studio2211/cpoptimizer/bin/arm64_osx/cpoptimizer",
            trace_log=True)
        if self.early_stopping_exception:
            if isinstance(self.early_stopping_exception, SolveEarlyStop):
                logger.info(self.early_stopping_exception)
            else:
                raise self.early_stopping_exception
        res = callback.res
        callbacks_list.on_solve_end(res=res, solver=self)
        self.status_solver = cpostatus_to_dostatus(status_from_cpo=result.solve_status)
        return callback.res

    def explain_infeasibility(self, parameters_cp: Optional[ParametersCP] = None,
                              explain_failure: bool = False,
                              refine_conflict: bool = True,
                              **args):
        if parameters_cp is None:
            parameters_cp = ParametersCP.default()
        if self.cp_model is None:
            self.init_model(**args)
        res_ = []
        if explain_failure:
            res_exp_failure = self.cp_model.explain_failure(
                    execfile="/Applications/CPLEX_Studio2211/cpoptimizer/bin/arm64_osx/cpoptimizer",
                    TimeLimit=100
            )
            res_.append(res_exp_failure)
        if refine_conflict:
            res_conflict = self.cp_model.refine_conflict(execfile=
                                                         "/Applications/CPLEX_Studio2211/cpoptimizer/"
                                                         "bin/arm64_osx/cpoptimizer",
                                                         ConflictRefinerTimeLimit=100)
            res_.append(res_conflict)
        return res_


class CpoSolutionCallback(CpoCallback):
    def __init__(self, do_solver: CPOSolver, callback: Callback):
        super().__init__()
        self.do_solver = do_solver
        self.callback = callback
        self.res = do_solver.create_result_storage([])
        self.nb_solutions = 0

    def invoke(self, solver: docplex.cp.model.CpoSolver, event, sres: CpoSolveResult):
        if event == EVENT_SOLUTION:
            sol = self.do_solver.retrieve_solution(sres)
            fit = self.do_solver.aggreg_from_sol(sol)
            self.res.append((sol, fit))
            self.nb_solutions += 1
            # end of step callback: stopping?
            logger.info(f"Solution #{self.nb_solutions} = {fit}")
            logger.info(f"Bounds = {sres.get_objective_bounds()}")
            logger.info(f"Gaps = {sres.get_objective_gaps()}")
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
                solver.abort_search()

