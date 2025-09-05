from docplex.cp.solution import CpoSolveResult
from typing import Any

from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.singlemachine.problem import WeightedTardinessProblem, WTSolution
from cpopt_solvers.cpopt_solver import CPOSolver, CpoModel
from docplex.cp.model import (CpoModel, interval_var, integer_var, alternative, sum_of,
                              max_of, pulse, end_before_start, minimize,
                              end_of, no_overlap)


class CPOptSolverSingleMachine(CPOSolver):
    problem: WeightedTardinessProblem
    vars: dict

    def init_model(self, **args: Any) -> None:
        self.cp_model = CpoModel(name="weighted_tardiness_single_machine")
        max_time = sum(self.problem.processing_times)
        intervals = [interval_var(start=(0, max_time-self.problem.processing_times[i]),
                                  end=(0, max_time),
                                  size=self.problem.processing_times[i],
                                  name=f"interval_{i}")
                     for i in range(self.problem.num_jobs)]
        self.cp_model.add_constraint(no_overlap(intervals))
        tardiness = sum_of([self.problem.weights[i]*max_of([end_of(intervals[i])-self.problem.due_dates[i],
                                                           0])
                            for i in range(self.problem.num_jobs)])
        self.cp_model.add(minimize(tardiness))
        self.vars = {"intervals": intervals}

    def retrieve_solution(self, sres: CpoSolveResult) -> WTSolution:
        schedule = []
        for i in range(self.problem.num_jobs):
            itv = sres.get_var_solution(self.vars["intervals"][i])
            start = itv.get_start()
            end = itv.get_end()
            schedule.append((start, end))
        return WTSolution(problem=self.problem,
                          schedule=schedule)