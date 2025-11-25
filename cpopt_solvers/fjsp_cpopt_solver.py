from docplex.cp.solution import CpoSolveResult, CpoIntervalVarSolution
from typing import Any, Optional
from discrete_optimization.generic_tools.do_problem import Solution, ParamsObjectiveFunction
from discrete_optimization.fjsp.problem import FJobShopProblem, FJobShopSolution
from cpopt_solvers.cpopt_solver import CPOSolver
from docplex.cp.model import (CpoModel, interval_var,
                              no_overlap,
                              integer_var,
                              alternative, max_of,
                              pulse, end_before_start, minimize, end_of)


class CPOFJspSolver(CPOSolver):
    problem: FJobShopProblem

    def __init__(self, problem: FJobShopProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)
        self.variables = {}

    def init_model(self, **args: Any) -> None:
        max_time = args.get("max_time", self.problem.horizon)
        self.cp_model = CpoModel()
        intervals = {}
        opt_intervals = {}
        keys_per_machine = {m: set() for m in range(self.problem.n_machines)}
        for i in range(self.problem.n_jobs):
            for j in range(len(self.problem.list_jobs[i].sub_jobs)):
                opt_intervals[(i, j)] = []
                options = self.problem.list_jobs[i].sub_jobs[j]
                intervals[(i, j)] = interval_var(start=(0, max_time),
                                                 end=(0, max_time),
                                                 size=(min(opt.processing_time for opt in options),
                                                       max(opt.processing_time for opt in options)),
                                                 optional=False,
                                                 name=f"interval_{i, j}")
                if j >= 1:
                    self.cp_model.add(end_before_start(intervals[(i, j-1)], intervals[(i, j)]))
                for k in range(len(options)):
                    option = options[k]
                    opt_intervals[(i, j)].append(interval_var(start=(0, max_time),
                                                              end=(0, max_time),
                                                              size=option.processing_time,
                                                              optional=True,
                                                              name=f"optinterval_{i, j, k}"))
                    keys_per_machine[option.machine_id].add(((i, j), k))
                self.cp_model.add(alternative(intervals[(i, j)], opt_intervals[(i, j)]))
        for m in keys_per_machine:
            self.cp_model.add(no_overlap([opt_intervals[x[0]][x[1]]
                                          for x in keys_per_machine[m]]))
        self.cp_model.add(minimize(max_of([end_of(intervals[i, len(self.problem.list_jobs[i].sub_jobs)-1])
                                           for i in range(self.problem.n_jobs)])))
        self.variables["intervals"] = intervals
        self.variables["opt_intervals"] = opt_intervals

    def retrieve_solution(self, sres: CpoSolveResult) -> Solution:
        print("Obj=", sres.get_objective_value(), "Obj bound", sres.get_objective_bound())
        schedule_dict = {}
        for (i, j) in self.variables["intervals"]:
            itv: CpoIntervalVarSolution = sres.get_var_solution(self.variables["intervals"][(i, j)])
            start = itv.get_start()
            end = itv.get_end()
            machine = None
            for opt_i in range(len(self.variables["opt_intervals"][(i, j)])):
                itv = sres.get_var_solution(self.variables["opt_intervals"][(i, j)][opt_i])
                if itv.is_present():
                    machine = self.problem.list_jobs[i].sub_jobs[j][opt_i].machine_id
                    break
            schedule_dict[(i, j)] = (start, end, machine)
        return FJobShopSolution(problem=self.problem,
                                schedule=[[schedule_dict[(i, j)] for j in range(len(self.problem.list_jobs[i].sub_jobs))]
                                          for i in range(self.problem.n_jobs)])
