from docplex.cp.solution import CpoSolveResult, CpoIntervalVarSolution
from typing import Any, Optional
from discrete_optimization.generic_tools.do_problem import Solution, ParamsObjectiveFunction
from discrete_optimization.jsp.job_shop_problem import JobShopProblem, SolutionJobshop
from cpopt_solvers.cpopt_solver import CPOSolver
from docplex.cp.model import (CpoModel, interval_var,
                              no_overlap,
                              integer_var,
                              alternative, max_of,
                              pulse, end_before_start, minimize, end_of)


class CPOJspSolver(CPOSolver):
    problem: JobShopProblem

    def __init__(self, problem: JobShopProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)
        self.variables = {}

    def init_model(self, **args: Any) -> None:
        max_time = args.get("max_time", self.problem.horizon)
        self.cp_model = CpoModel()
        intervals = {i: {j: interval_var(start=(0, max_time),
                                         end=(0, max_time),
                                         size=self.problem.list_jobs[i][j].processing_time,
                                         name=f"interval_{i, j}")
                         for j in range(len(self.problem.list_jobs[i]))}
                     for i in range(self.problem.n_jobs)}
        for i in intervals:
            for j in range(1, len(self.problem.list_jobs[i])):
                self.cp_model.add(end_before_start(intervals[i][j-1], intervals[i][j]))
        for m in self.problem.job_per_machines:
            self.cp_model.add(no_overlap([intervals[x[0]][x[1]]
                                          for x in self.problem.job_per_machines[m]]))
        self.cp_model.add(minimize(
            max_of([end_of(intervals[i][len(self.problem.list_jobs[i])-1])
                    for i in range(self.problem.n_jobs)])))
        self.variables["intervals"] = intervals

    def retrieve_solution(self, sres: CpoSolveResult) -> Solution:
        print("Obj=", sres.get_objective_value(), "Obj bound", sres.get_objective_bound())
        schedule_dict = {}
        for i in self.variables["intervals"]:
            for j in self.variables["intervals"][i]:
                itv: CpoIntervalVarSolution = sres.get_var_solution(self.variables["intervals"][i][j])
                start = itv.get_start()
                end = itv.get_end()
                schedule_dict[(i, j)] = (start, end)
        return SolutionJobshop(problem=self.problem,
                               schedule=[[schedule_dict[(i,j)] for j in range(len(self.problem.list_jobs[i]))]
                                         for i in range(self.problem.n_jobs)])