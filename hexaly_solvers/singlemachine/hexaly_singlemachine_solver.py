from typing import Any

from discrete_optimization.generic_tools.do_problem import Solution
from hexaly_solvers.hexaly_solver import HexalySolver
import hexaly.optimizer
from discrete_optimization.singlemachine.problem import WeightedTardinessProblem, WTSolution


class HexalySingleMachineSolver(HexalySolver):
    problem: WeightedTardinessProblem
    vars: dict

    def init_model(self, **kwargs: Any) -> None:
        ls = hexaly.optimizer.HexalyOptimizer()
        #
        # Declare the optimization model
        #
        max_time = sum(self.problem.processing_times)

        model = ls.model
        tasks = [model.interval(0, max_time)
                 for t in range(self.problem.num_jobs)]
        tasks_array = model.array(tasks)
        sequence = model.list(self.problem.num_jobs)
        for j in range(self.problem.num_jobs):
            model.constraint(model.length(tasks[j]) == self.problem.processing_times[j])

        model.constraint(model.eq(model.count(sequence), self.problem.num_jobs))
        sequence_lambda = model.lambda_function(
                lambda i: model.lt(model.at(tasks_array, sequence[i]),
                                   model.at(tasks_array, sequence[i + 1])))
        model.constraint(model.and_(model.range(0, self.problem.num_jobs-1), sequence_lambda))
        self.ls = ls
        self.model = model
        self.vars = {"intervals": tasks}
        tardiness = model.sum([self.problem.weights[i]*model.max(model.end(tasks[i])-self.problem.due_dates[i], 0)
                               for i in range(self.problem.num_jobs)])
        model.minimize(tardiness)
        model.close()

    def retrieve_solution(self, **args) -> WTSolution:
        sched = []
        for i in range(self.problem.num_jobs):
            start = int(self.vars["intervals"][i].value.start())
            end = int(self.vars["intervals"][i].value.end())
            sched.append((start, end))
        return WTSolution(problem=self.problem, schedule=sched)
