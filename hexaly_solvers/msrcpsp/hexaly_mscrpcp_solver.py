from collections import defaultdict

import numpy as np

from hexaly_solvers.hexaly_solver import HexalySolver
import logging
from typing import Dict, Optional, Hashable, Any
import hexaly.optimizer
import hexaly.modeler

from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction, Solution
from discrete_optimization.rcpsp_multiskill.problem import MultiskillRcpspProblem, MultiskillRcpspSolution


class HexalyMsrcpspSolver(HexalySolver):
    problem: MultiskillRcpspProblem

    def __init__(self, problem: MultiskillRcpspProblem,
                 params_objective_function: Optional[ParamsObjectiveFunction] = None, **args):
        super().__init__(problem, params_objective_function, **args)
        self.variables = {}

    def init_model(self,
                   **kwargs: Any) -> None:
        one_skill_used_per_worker = kwargs.get("one_skill_used_per_worker", False)
        ls = hexaly.optimizer.HexalyOptimizer()
        #
        # Declare the optimization model
        #
        model = ls.model
        horizon = self.problem.horizon
        tasks_list = self.problem.tasks_list
        nb_tasks = self.problem.n_jobs
        modes = [sorted(list(self.problem.mode_details[t].keys())) for t in tasks_list]
        nb_modes = [len(modes[i]) for i in range(len(modes))]
        tasks_in_mode = [[model.optional_interval(0, horizon) for _ in range(nb_modes[i])]
                         for i in range(nb_tasks)]
        present_mode = [[model.presence(tasks_in_mode[i][m]) for m in range(nb_modes[i])] for i in range(nb_tasks)]
        # Constraints 1: Exactly one active mode for each task
        for task in range(nb_tasks):
            model.constraint(model.eq(sum(present_mode[task]), 1))
        # Hull
        tasks = [model.hull(tasks_in_mode[task][mode] for mode in range(nb_modes[task])) for task in range(nb_tasks)]
        makespan = model.end(tasks[self.problem.index_task[self.problem.sink_task]])
        duration_array = [[self.problem.mode_details[tasks_list[i]][mode]["duration"]
                           for mode in modes[i]] for i in range(nb_tasks)]

        for task in self.problem.successors:
            index_ = self.problem.index_task[task]
            for succ in self.problem.successors[task]:
                index_succ = self.problem.index_task[succ]
                model.constraint(tasks[index_] < tasks[index_succ])

        # Constraints 2: Task duration
        for task in range(nb_tasks):
            for mode in range(nb_modes[task]):
                model.constraint(model.iif(
                    present_mode[task][mode],
                    model.eq(model.length(tasks_in_mode[task][mode]), duration_array[task][mode]),
                    1
                ))
        resource_needs = [
            [
                [self.problem.mode_details[tasks_list[i]][m].get(res, 0)
                 for res in self.problem.resources_list]
                for m in modes[i]
            ]

            for i in range(self.problem.n_jobs)]
        calendar = {res: self.problem.get_resource_availability_array(res)
                    for res in self.problem.resources_list}
        # Resource constraint
        for i_r in range(len(self.problem.resources_list)):
            res = self.problem.resources_list[i_r]
            if res in self.problem.non_renewable_resources:
                total_weight = model.sum()
                for task in range(nb_tasks):
                    for mode in range(nb_modes[task]):
                        total_weight.add_operand(resource_needs[task][mode][i_r] * present_mode[task][mode])
                model.constraint(model.leq(total_weight, self.problem.get_max_resource_capacity(res)))
            else:
                def capacity_respected(resource_index, time, calendar_array: np.array):
                    total_weight = model.sum()
                    for task in range(nb_tasks):
                        for mode in range(nb_modes[task]):
                            total_weight.add_operand(
                                resource_needs[task][mode][resource_index]
                                * model.contains(tasks_in_mode[task][mode], time)
                            )
                    return model.leq(total_weight, calendar_array[time])

                capacity_respected_lambda = model.lambda_function(lambda time:
                                                                  capacity_respected(i_r, time,
                                                                                     model.array(calendar[res])))
                model.constraint(model.and_(model.range(makespan), capacity_respected_lambda))

        # Skills stuff :
        skill_needs = [[[self.problem.mode_details[tasks_list[i]][m].get(s, 0)
                        for s in self.problem.skills_list]
                       for m in modes[i]]
                       for i in range(self.problem.n_jobs)]
        intervals_employee = {emp: [] for emp in self.problem.employees_list}
        skills_used_employee = {emp: {} for emp in self.problem.employees_list}
        for emp in intervals_employee:
            nz_skills = self.problem.employees[emp].get_non_zero_skills()
            index_nz_skills = set([self.problem.skills_list.index(s) for s in nz_skills])
            for i in range(self.problem.n_jobs):
                intersect_skills = [index_s for m in range(nb_modes[i])
                                    for index_s in index_nz_skills if skill_needs[i][m][index_s] > 0]
                if len(intersect_skills) > 0:
                    intervals_employee[emp].append(model.optional_interval(0, horizon))
                    skills_used_employee[emp][i] = {index_skill: model.bool() for index_skill in intersect_skills}
                    for i_s in skills_used_employee[emp][i]:
                        model.constraint(model.geq(model.presence(intervals_employee[emp][-1]),
                                                   skills_used_employee[emp][i][i_s]))
                    if one_skill_used_per_worker:
                        #model.constraint(model.leq(model.sum([skills_used_employee[emp][i][x]
                        #                                      for x in skills_used_employee[emp][i]]), 1))
                        model.constraint(model.iif(model.presence(intervals_employee[emp][-1]),
                                                   model.sum([skills_used_employee[emp][i][x]
                                                              for x in skills_used_employee[emp][i]])==1, 1))
                    model.constraint(model.iif(model.presence(intervals_employee[emp][-1]),
                                               tasks[i] == intervals_employee[emp][-1], 1))
                else:
                    intervals_employee[emp].append(None)
        # No overlap activities for all workers :

        for emp in intervals_employee:
            def no_overlap(employee, time):
                total_activity = model.sum()
                for interval in intervals_employee[employee]:
                    if interval is not None:
                        total_activity.add_operand(
                            model.contains(interval, time)
                        )
                return model.leq(total_activity, 1) #self.problem.employees[emp].calendar_employee[time])
            no_overlap_lmbda = model.lambda_function(lambda time: no_overlap(emp, time))
            model.constraint(model.and_(model.range(makespan), no_overlap_lmbda))

        # Constraint : skills respected.
        if True:
            for i in range(self.problem.n_jobs):
                nz_s = [index_s for m in range(nb_modes[i]) for index_s in range(len(self.problem.skills_list))
                        if skill_needs[i][m][index_s] > 0]
                for s in nz_s:
                    #skill_req = model.sum()
                    #for mode in range(nb_modes[i]):
                    #    skill_req.add_operand(model.presence(tasks_in_mode[i][mode])*skill_needs[i][mode][s])
                    skill_req = model.sum([model.presence(tasks_in_mode[i][mode])*skill_needs[i][mode][s] for mode in range(nb_modes[i])])
                    skill_allocated = model.sum()
                    for emp in skills_used_employee:
                        if i in skills_used_employee[emp]:
                            if s in skills_used_employee[emp][i]:
                                skill_allocated.add_operand(skills_used_employee[emp][i][s]*self.problem.employees[emp].get_skill_level(self.problem.skills_list[s]))
                    model.constraint(model.geq(skill_allocated, skill_req))

        self.variables = {'tasks_mode': tasks_in_mode,
                          "presence_of": present_mode,
                          "modes": modes,
                          "tasks_interval": tasks,
                          "intervals_employee": intervals_employee,
                          "intervals_employee_presence": {emp: [model.presence(intervals_employee[emp][i])
                                                                if intervals_employee[emp][i] is not None else None
                                                                for i in range(len(intervals_employee[emp]))]
                                                          for emp in intervals_employee},
                          "skills_used_employee": skills_used_employee}
        model.minimize(makespan)
        model.close()
        self.ls = ls
        self.model = model

    def retrieve_solution(self, **args) -> Solution:
        schedule = {}
        mode_dict = {}
        employee_usage = {}
        for i in range(self.problem.n_jobs):
            itv = self.variables["tasks_interval"][i]
            start = itv.value.start()
            end = itv.value.end()
            schedule[self.problem.tasks_list[i]] = {"start_time": start,
                                                    "end_time": end}
            employee_usage[i] = defaultdict(lambda: set())
            for i_mode in range(len(self.variables["presence_of"][i])):
                if self.variables["presence_of"][i][i_mode].value == 1:
                    mode_dict[i] = self.variables["modes"][i][i_mode]
            for emp in self.variables["skills_used_employee"]:
                if i in self.variables["skills_used_employee"][emp]:
                    for index_skill in self.variables["skills_used_employee"][emp][i]:
                        var = self.variables["skills_used_employee"][emp][i][index_skill]
                        if var.value == 1:
                            employee_usage[i][emp].add(self.problem.skills_list[index_skill])
        return MultiskillRcpspSolution(problem=self.problem,
                                       modes=mode_dict,
                                       schedule=schedule,
                                       employee_usage=employee_usage)






























