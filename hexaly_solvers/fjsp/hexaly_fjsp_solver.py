from typing import Any
import numpy as np
from hexaly_solvers.hexaly_solver import HexalySolver
import hexaly.optimizer
from discrete_optimization.fjsp.problem import FJobShopProblem, FJobShopSolution


class HexalyFJspSolver(HexalySolver):
    problem: FJobShopProblem

    def init_model(self, **kwargs: Any) -> None:
        max_time = kwargs.get("max_time", self.problem.horizon)
        ls = hexaly.optimizer.HexalyOptimizer()
        model = ls.model
        nb_total_tasks = self.problem.n_all_jobs
        tasks = []
        duration_per_machines = -np.ones((self.problem.n_all_jobs, self.problem.n_machines))
        len_ = 0
        for i in range(self.problem.n_jobs):
            for j in range(len(self.problem.list_jobs[i].sub_jobs)):
                tasks.append((i, j))
                for option in self.problem.list_jobs[i].sub_jobs[j]:
                    duration_per_machines[len_, option.machine_id] = option.processing_time
                len_ += 1
        duration_per_machines_array = model.array(duration_per_machines)
        tasks_index = {tasks[i]: i for i in range(len(tasks))}
        tasks_interval = [model.interval(0, max_time) for i in range(len(tasks))]
        tasks_interval_array = model.array(tasks_interval)
        allocated_task_per_machine = [model.list(nb_total_tasks)
                                      for i in range(self.problem.n_machines)]
        allocated_task_per_machine_array = model.array(allocated_task_per_machine)
        model.constraint(model.partition(allocated_task_per_machine_array))
        machine_for_task = [model.find(allocated_task_per_machine_array, i)
                            for i in range(nb_total_tasks)]
        duration_for_task = [model.at(duration_per_machines_array, i, machine_for_task[i])
                             for i in range(nb_total_tasks)]
        # Compatibility
        for t_index in range(self.problem.n_all_jobs):
            task = tasks[t_index]
            options = self.problem.list_jobs[task[0]].sub_jobs[task[1]]
            machines = set([opt.machine_id for opt in options])
            for m in range(self.problem.n_machines):
                if m not in machines:
                    model.not_(model.contains(allocated_task_per_machine[m], t_index))

        # Length
        for t_index in range(self.problem.n_all_jobs):
            model.constraint(model.length(tasks_interval[t_index]) == duration_for_task[t_index])

        # Precedence
        for i in range(self.problem.n_jobs):
            for j in range(1, len(self.problem.list_jobs[i].sub_jobs)):
                index_1 = tasks_index[(i, j-1)]
                index_2 = tasks_index[(i, j)]
                model.constraint(tasks_interval[index_1] < tasks_interval[index_2])

        # Order on each machine
        for m in range(self.problem.n_machines):
            seq_m = allocated_task_per_machine[m]
            precedence_lambda = model.lambda_function(lambda i:
                                                      tasks_interval_array[seq_m[i]] <
                                                      tasks_interval_array[seq_m[i+1]])
            model.constraint(model.and_(model.range(0, model.count(seq_m) - 1), precedence_lambda))

        makespan = model.max([model.end(tasks_interval[i])
                              for i in range(self.problem.n_all_jobs)])
        model.minimize(makespan)
        model.close()
        self.ls = ls
        self.model = model
        self.variables = {}
        tasks_index = {tasks[i]: i for i in range(len(tasks))}
        self.variables["intervals"] = tasks_interval
        self.variables["machine_for_task"] = machine_for_task
        self.variables["tasks_mapping"] = {"tasks": tasks,
                                           "tasks_to_index": tasks_index}

    def retrieve_solution(self, **args) -> FJobShopSolution:
        objs = tuple([self.ls.model.objectives[i].value for i in range(self.ls.model.nb_objectives)])
        print("Objs =", objs)
        schedule_dict = {}
        for i in range(self.problem.n_jobs):
            for j in range(len(self.problem.list_jobs[i].sub_jobs)):
                t_index = self.variables["tasks_mapping"]["tasks_to_index"][(i, j)]
                start = self.variables["intervals"][t_index].value.start()
                end = self.variables["intervals"][t_index].value.end()
                machine = self.variables["machine_for_task"][t_index].value
                schedule_dict[(i, j)] = (start, end, machine)
        return FJobShopSolution(problem=self.problem,
                                schedule=[[schedule_dict[(i, j)]
                                           for j in range(len(self.problem.list_jobs[i].sub_jobs))]
                                          for i in range(self.problem.n_jobs)])