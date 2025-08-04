from collections import defaultdict
from typing import Any
import numpy as np
from discrete_optimization.generic_tools.do_problem import Solution

from hexaly_solvers.hexaly_solver import HexalySolver
import hexaly.optimizer
from discrete_optimization.fjsp.problem import FJobShopProblem, FJobShopSolution
from enum import Enum
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (CategoricalHyperparameter,
                                                                                EnumHyperparameter)


class ModelingHexalyFJsp(Enum):
    PARTITION = 0
    OPTIONAL_INTERVAL = 1


class HexalyFJspSolver(HexalySolver):
    problem: FJobShopProblem
    hyperparameters = [EnumHyperparameter(name="modeling", enum=ModelingHexalyFJsp,
                                          default=ModelingHexalyFJsp.PARTITION)]

    def __init__(self, problem: FJobShopProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)
        self.modeling: ModelingHexalyFJsp = None

    def init_model(self, **kwargs: Any) -> None:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        modeling = kwargs["modeling"]
        if modeling == ModelingHexalyFJsp.PARTITION:
            self.init_model_partition(**kwargs)
            self.modeling = modeling
        if modeling == ModelingHexalyFJsp.OPTIONAL_INTERVAL:
            self.init_model_optional_interval(**kwargs)
            self.modeling = modeling

    def init_model_optional_interval(self, **kwargs):
        max_time = kwargs.get("max_time", self.problem.horizon)
        ls = hexaly.optimizer.HexalyOptimizer()
        model = ls.model
        nb_total_tasks = self.problem.n_all_jobs
        tasks = []
        duration_per_machines = -np.ones((self.problem.n_all_jobs, self.problem.n_machines))
        len_ = 0
        all_tasks_vars = []
        all_tasks_vars_per_option = []
        all_presence_vars_per_option = []
        keys_for_machines = defaultdict(lambda: set())
        for i in range(self.problem.n_jobs):
            for j in range(len(self.problem.list_jobs[i].sub_jobs)):
                tasks.append((i, j))
                options_intervals = []
                options_presence = []
                k = 0
                for option in self.problem.list_jobs[i].sub_jobs[j]:
                    duration_per_machines[len_, option.machine_id] = option.processing_time
                    options_intervals.append(model.optional_interval(0, self.problem.horizon))
                    options_presence.append(model.presence(options_intervals[-1]))
                    keys_for_machines[option.machine_id].add((len_, k))
                    k += 1
                all_tasks_vars_per_option.append(options_intervals)
                all_tasks_vars.append(model.hull(options_intervals))
                all_presence_vars_per_option.append(options_presence)
                len_ += 1
        duration_per_machines_array = model.array(duration_per_machines)
        tasks_index = {tasks[i]: i for i in range(len(tasks))}

        for i in range(len(all_presence_vars_per_option)):
            job_index, subjob_index = tasks[i]
            model.constraint(model.eq(model.sum(all_presence_vars_per_option[i]), 1))  # One option per mode
            for j in range(len(all_presence_vars_per_option[i])):
                processing = self.problem.list_jobs[job_index].sub_jobs[subjob_index][j].processing_time
                model.constraint(model.iif(all_presence_vars_per_option[i][j],
                                           model.eq(model.length(all_tasks_vars[i]),
                                                    processing), 1))
                model.constraint(model.iif(all_presence_vars_per_option[i][j],
                                           model.eq(model.length(all_tasks_vars[i]),
                                                    processing), 1))
                model.constraint(model.iif(all_presence_vars_per_option[i][j],
                                           model.eq(model.length(all_tasks_vars_per_option[i][j]),
                                                    processing), 1))
        makespan = model.max([model.end(t) for t in all_tasks_vars])
        for m in keys_for_machines:
            def no_overlap(machine, time):
                total_weight = model.sum()
                for j, k_ in keys_for_machines[machine]:
                    total_weight.add_operand(
                        model.contains(all_tasks_vars_per_option[j][k_], time)
                    )
                return model.leq(total_weight, 1)
            no_overlap = model.lambda_function(lambda time: no_overlap(m, time))
            model.constraint(model.and_(model.range(makespan), no_overlap))

        for i in range(self.problem.n_jobs):
            for j in range(1, len(self.problem.list_jobs[i].sub_jobs)):
                index_1 = tasks_index[(i, j - 1)]
                index_2 = tasks_index[(i, j)]
                model.constraint(all_tasks_vars[index_1] < all_tasks_vars[index_2])
        self.variables = {}
        self.variables["intervals"] = all_tasks_vars
        self.variables["tasks_mapping"] = {"tasks": tasks,
                                           "tasks_to_index": tasks_index}
        self.variables["options_presence"] = all_presence_vars_per_option
        model.minimize(makespan)
        model.close()
        self.ls = ls
        self.model = model

    def init_model_partition(self, **kwargs):
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

    def retrieve_solution(self, **args) -> Solution:
        if self.modeling == ModelingHexalyFJsp.PARTITION:
            return self.retrieve_solution_partition(**args)
        if self.modeling == ModelingHexalyFJsp.OPTIONAL_INTERVAL:
            return self.retrieve_opt_intervals(**args)

    def retrieve_opt_intervals(self, **args) -> FJobShopSolution:
        objs = tuple([self.ls.model.objectives[i].value for i in range(self.ls.model.nb_objectives)])
        print("Objs =", objs)
        schedule_dict = {}
        for i in range(self.problem.n_jobs):
            for j in range(len(self.problem.list_jobs[i].sub_jobs)):
                t_index = self.variables["tasks_mapping"]["tasks_to_index"][(i, j)]
                start = self.variables["intervals"][t_index].value.start()
                end = self.variables["intervals"][t_index].value.end()
                options = self.problem.list_jobs[i].sub_jobs[j]
                for k in range(len(options)):
                    if self.variables["options_presence"][t_index][k].value == 1:
                        machine = options[k].machine_id
                        schedule_dict[(i, j)] = (start, end, machine)
                        break
        return FJobShopSolution(problem=self.problem,
                                schedule=[[schedule_dict[(i, j)]
                                           for j in range(len(self.problem.list_jobs[i].sub_jobs))]
                                          for i in range(self.problem.n_jobs)])

    def retrieve_solution_partition(self, **args) -> FJobShopSolution:
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