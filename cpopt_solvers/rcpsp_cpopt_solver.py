from docplex.cp.solution import CpoSolveResult, CpoIntervalVarSolution
from typing import Any, Optional
from discrete_optimization.generic_tools.do_problem import Solution, ParamsObjectiveFunction
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPSolution
from discrete_optimization.rcpsp.solver.rcpsp_solver import SolverRCPSP
from cpopt_solvers.cpopt_solver import CPOSolver
from docplex.cp.model import (CpoModel, interval_var, integer_var, alternative,
                              pulse, end_before_start, minimize, end_of)


class CPOptRCPSPSolver(CPOSolver, SolverRCPSP):
    def __init__(self, problem: RCPSPModel, params_objective_function: Optional[ParamsObjectiveFunction] = None,
                 **kwargs):
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = None

    def retrieve_solution(self, sres: CpoSolveResult) -> Solution:
        rcpsp_schedule = {}
        for t in self.variables["intervals"]:
            itv: CpoIntervalVarSolution = sres.get_var_solution(self.variables["intervals"][t])
            start = itv.get_start()
            end = itv.get_end()
            rcpsp_schedule[t] = {"start_time": start,
                                 "end_time": end}
        return RCPSPSolution(problem=self.problem,
                             rcpsp_schedule=rcpsp_schedule,
                             rcpsp_modes=[1 for _ in range(self.problem.n_jobs_non_dummy)])

    def init_model(self, **args: Any) -> None:
        self.cp_model = CpoModel(name="RCPSP")
        assert not self.problem.is_rcpsp_multimode()
        intervals = {t: interval_var(start=(0, self.problem.horizon),
                                     end=(0, self.problem.horizon),
                                     size=self.problem.mode_details[t][1]["duration"],
                                     name=f"interval_{t}")
                     for t in self.problem.tasks_list}
        for r in self.problem.resources_list:
            t_of_interest = [t for t in self.problem.tasks_list
                             if self.problem.mode_details[t][1].get(r, 0) > 0]
            # TODO : put the calendar variability.
            self.cp_model.add(sum(pulse(intervals[t],
                                        self.problem.mode_details[t][1][r])
                                  for t in t_of_interest) <= self.problem.get_max_resource_capacity(r))
        for s in self.problem.successors:
            for succ in self.problem.successors[s]:
                self.cp_model.add(end_before_start(intervals[s],
                                                   intervals[succ]))
        self.cp_model.add(minimize(end_of(intervals[self.problem.sink_task])))
        self.variables = dict()
        self.variables["intervals"] = intervals

