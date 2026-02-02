from typing import Any
from discrete_optimization.generic_tasks_tools.base import Task
from discrete_optimization.singlemachine.problem import WTSolution, WeightedTardinessProblem
from optal_py_solvers.generic_optalcp_task_solver import SchedulingOptalSolver
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction, Solution
import optalcp as cp

class OptalSingleMachineSolver(SchedulingOptalSolver):
    problem: WeightedTardinessProblem
    def __init__(self, problem: WTSolution, params_objective_function: ParamsObjectiveFunction = None, **args):
        super().__init__(problem, params_objective_function, **args)
        self.variables = {}

    def get_task_interval_variable(self, task: Task) -> cp.IntervalVar:
        return self.variables["intervals"][task]

    def retrieve_solution(self, result: cp.SolveResult) -> WTSolution:
        schedule = []
        for t in self.problem.tasks_list:
            st = result.solution.get_start(self.variables["intervals"][t])
            end = result.solution.get_end(self.variables["intervals"][t])
            schedule.append((st, end))
        return WTSolution(problem=self.problem,
                          schedule=schedule)

    def init_model(self, **args: Any) -> None:
        self.cp_model = cp.Model()
        intervals = {}
        for t in self.problem.tasks_list:
            intervals[t] = self.cp_model.interval_var(start=(self.problem.release_dates[t], None),
                                                      end=(self.problem.release_dates[t]
                                                           + self.problem.processing_times[t],
                                                           None),
                                                      length=self.problem.processing_times[t],
                                                      optional=False,
                                                      name=f"interval_{t}")
        self.cp_model.no_overlap([intervals[t] for t in self.problem.tasks_list])
        objective_expr = self.cp_model.sum([self.cp_model.max2(0,
                                                               self.problem.weights[t]*
                                                               (self.cp_model.end(intervals[t])-self.problem.due_dates[t]))
                                            for t in self.problem.tasks_list])
        self.variables["intervals"] = intervals
        self.cp_model.minimize(objective_expr)





