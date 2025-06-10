from typing import Any

from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.jsp.problem import JobShopProblem, JobShopSolution
import hexaly.optimizer
from hexaly_solvers.hexaly_solver import HexalySolver


class HexalyJspSolver(HexalySolver):
    problem: JobShopProblem

    def init_model(self, **kwargs: Any) -> None:
        horizon = kwargs.get("horizon",
                             sum(sum(subjob.processing_time for subjob in job) for job in self.problem.list_jobs))
        ls = hexaly.optimizer.HexalyOptimizer()
        #
        # Declare the optimization model
        #
        model = ls.model
        interval_tasks = [[model.interval(0, horizon) for _ in range(len(self.problem.list_jobs[i]))]
                          for i in range(self.problem.n_jobs)]
        for i in range(self.problem.n_jobs):
            for j in range(len(self.problem.list_jobs[i])):
                model.constraint(model.length(interval_tasks[i][j]) == self.problem.list_jobs[i][j].processing_time)
                # Precedence constraints between the tasks of a job
        for i in range(self.problem.n_jobs):
            for j in range(len(self.problem.list_jobs[i])-1):
                model.constraint(interval_tasks[i][j] < interval_tasks[i][j+1])
        # flatten_intervals = []
        # index_of_subjob = {}
        # l = 0
        # for i in range(self.problem.n_jobs):
        #     for j in range(self.problem.list_jobs[i]):
        #         flatten_intervals.append(interval_tasks[i][j])
        #         index_of_subjob[(i, j)] = l
        #         l += 1
        sequence_on_machine = [model.list(len(self.problem.job_per_machines[m]))
                               for m in range(self.problem.n_machines)]
        # job_machine = [model.array([index_of_subjob[subj] for subj in self.problem.job_per_machines[m]])
        #                for m in range(self.problem.n_machines)]
        interval_per_machine = [model.array([interval_tasks[x[0]][x[1]] for x in self.problem.job_per_machines[m]])
                                for m in range(self.problem.n_machines)]
        for m in range(self.problem.n_machines):
            sequence = sequence_on_machine[m]
            model.constraint(model.eq(model.count(sequence), len(self.problem.job_per_machines[m])))
            sequence_lambda = model.lambda_function(
                lambda i: model.lt(model.at(interval_per_machine[m],
                                            sequence[i]),
                                   model.at(interval_per_machine[m],
                                            sequence[i + 1])))
            model.constraint(model.and_(model.range(0, len(self.problem.job_per_machines[m]) - 1), sequence_lambda))
        makespan = model.max([model.end(interval_tasks[i][-1])
                              for i in range(self.problem.n_jobs)])
        model.minimize(makespan)
        model.close()
        self.ls = ls
        self.model = model
        self.variables = {}
        self.variables["intervals"] = interval_tasks

    def retrieve_solution(self, **args) -> JobShopSolution:
        schedule = {}
        objs = tuple([self.ls.model.objectives[i].value for i in range(self.ls.model.nb_objectives)])
        print("Objs =", objs)
        for i in range(len(self.variables["intervals"])):
            for j in range(len(self.variables["intervals"][i])):
                start = int(self.variables["intervals"][i][j].value.start())
                end = int(self.variables["intervals"][i][j].value.end())
                schedule[(i,j)] = (start, end)
        return JobShopSolution(problem=self.problem,
                               schedule=[[schedule[(i, j)]
                                         for j in range(len(self.problem.list_jobs[i]))]
                                         for i in range(len(self.problem.list_jobs))])
