# Multi-mode rcpsp solver

import logging
from typing import Dict, Optional, Hashable
import hexaly.optimizer
import hexaly.modeler
import numpy as np
from discrete_optimization.generic_tools.do_problem import (
    Problem,
    ParamsObjectiveFunction,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.problem import RcpspProblem, RcpspSolution
from discrete_optimization.rcpsp.solvers.rcpsp_solver import RcpspSolver
from hexaly_solvers.hexaly_solver import HexalySolver
from enum import Enum
logger = logging.getLogger(__name__)


class ModelisationLSolver(Enum):
    # INTEGER = 0
    INTERVAL = 1
    OPT_INTERVAL = 2


class HexalyMultiModeRCPSP(HexalySolver, RcpspSolver):
    def __init__(self, problem: Problem, params_objective_function: Optional[ParamsObjectiveFunction] = None, **kwargs):
        super().__init__(problem, params_objective_function, **kwargs)
        self.ls: Optional[hexaly.optimizer.HexalyOptimizer] = None
        self.model: Optional[hexaly.optimizer.HxModel] = None
        self.vars: Optional[Dict[str, Dict[Hashable, hexaly.modeler.HxExpression]]] = {}
        self.modelisation: Optional[ModelisationLSolver] = None
        self.modes_correspondance = None

    def init_model(self, modelisation_type: ModelisationLSolver = ModelisationLSolver.INTERVAL,
                   **kwargs):
        if modelisation_type == ModelisationLSolver.INTERVAL:
            self.init_model_interval(**kwargs)
            self.modelisation = modelisation_type
        if modelisation_type == ModelisationLSolver.OPT_INTERVAL:
            self.init_model_opt_interval(**kwargs)
            self.modelisation = modelisation_type

    def init_model_interval(self, **kwargs):
        ls = hexaly.optimizer.HexalyOptimizer()
        #
        # Declare the optimization model
        #
        model = ls.model
        # Trivial upper bound for the start times of the tasks
        horizon = self.problem.horizon
        # Integer decisions: start time of each task
        tasks = self.problem.tasks_list
        nb_tasks = self.problem.n_jobs
        interval_tasks = [model.interval(0, horizon) for t in tasks]
        modes = [list(self.problem.mode_details[t]) for t in tasks]
        modes_var = [model.int(0, len(modes[i])-1)
                     for i in range(self.problem.n_jobs)]
        modes_array = model.array(modes_var)
        modes_details = [
            [
                [self.problem.mode_details[tasks[i]][m].get(res, 0)
                 for res in self.problem.resources_list]
                for m in modes[i]
            ]
            for i in range(self.problem.n_jobs)]
        modes_details_array = [model.array(modes_details[i])
                               for i in range(self.problem.n_jobs)]

        consumption_array = []
        nb_resources = len(self.problem.resources_list)
        for i in range(self.problem.n_jobs):
            c = []
            for i_res in range(nb_resources):
                min_cons = min([m[i_res] for m in modes_details[i]])
                max_cons = max([m[i_res] for m in modes_details[i]])
                c += [model.int(min_cons, max_cons)]
            consumption_array += [c]
        duration_of_modes = [[self.problem.mode_details[tasks[i]][m]["duration"]
                              for m in modes[i]]
                             for i in range(self.problem.n_jobs)]
        duration_of_modes = model.array(duration_of_modes)
        for i in range(nb_tasks):
            model.constraint(model.length(interval_tasks[i]) ==
                             model.at(duration_of_modes,
                                      i,
                                      model.at(modes_array, i)))
        # Set consumption resource
        for i in range(nb_tasks):
            for i_r in range(nb_resources):
                model.constraint(consumption_array[i][i_r] == model.at(modes_details_array[i],
                                                                       model.at(modes_array, i), i_r)
                                 )
        for t in tasks:
            index_t = self.problem.index_task[t]
            if t not in self.problem.successors:
                continue
            for s_t in self.problem.successors[t]:
                index_s_t = self.problem.index_task[s_t]
                model.constraint(interval_tasks[index_t] < interval_tasks[index_s_t])
        # Makespan: end of the last task
        makespan = model.end(interval_tasks[self.problem.index_task[self.problem.sink_task]])
        nb_resources = len(self.problem.resources_list)
        # Cumulative resource constraints
        for i_r in range(nb_resources):
            res = self.problem.resources_list[i_r]
            if res in self.problem.non_renewable_resources:
                model.constraint(model.sum(
                        consumption_array[index_task][i_r]
                        for index_task in range(self.problem.n_jobs)) <=
                         self.problem.get_max_resource_capacity(self.problem.resources_list[i_r]))
            else:
                capacity_respected = model.lambda_function(
                    lambda time: model.sum(
                        consumption_array[index_task][i_r] * model.contains(interval_tasks[index_task], time)
                        for index_task in range(self.problem.n_jobs)) <=
                                 self.problem.get_resource_available(self.problem.resources_list[i_r],
                                                                     time))
                model.constraint(model.and_(model.range(makespan), capacity_respected))
        # Minimize the makespan
        model.minimize(makespan)
        model.close()
        self.ls = ls
        self.model = model
        self.vars["intervals"] = interval_tasks
        self.vars["modes"] = modes_var
        self.modes_correspondance = [list(self.problem.mode_details[t]) for t in tasks]
        warm_start = kwargs.get("warm_start", False)
        warm_start_solution: Optional[RcpspSolution] = kwargs.get("warm_start_solution", None)
        if warm_start:
            if warm_start_solution is None:
                warm_start_solution = self.problem.get_dummy_solution()
            for st in self.vars["intervals"]:
                self.vars["intervals"][st].value = hexaly.optimizer.HxInterval(warm_start_solution.get_start_time(st),
                                                                               warm_start_solution.get_end_time(st))
                # self.vars["end"][st].value = warm_start_solution.get_end_time(st)

    def init_model_opt_interval(self, **kwargs):
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
        # Hull
        tasks = [model.hull(tasks_in_mode[task][mode] for mode in range(nb_modes[task])) for task in range(nb_tasks)]
        duration_array = [[self.problem.mode_details[tasks_list[i]][mode]["duration"]
                           for mode in modes[i]] for i in range(nb_tasks)]

        # Constraints: Task duration
        for task in range(nb_tasks):
            for mode in range(nb_modes[task]):
                model.constraint(model.iif(
                    present_mode[task][mode],
                    model.eq(model.length(tasks_in_mode[task][mode]), duration_array[task][mode]),
                    1
                ))

        for task in tasks_list:
            index_task = self.problem.index_task[task]
            for succ in self.problem.successors[task]:
                index_succ = self.problem.index_task[succ]
                model.constraint(model.lt(tasks[index_task], tasks[index_succ]))

        # Constraints: Exactly one active mode for each task
        for task in range(nb_tasks):
            model.constraint(model.eq(sum(present_mode[task]), 1))

        # Makespan: end of the last task
        makespan = model.end(tasks[self.problem.index_task[self.problem.sink_task]])

        # Constraints: Renewable resources
        weights = [
                        [
                            [self.problem.mode_details[tasks_list[i]][m].get(res, 0)
                             for res in self.problem.resources_list]
                            for m in modes[i]
                        ]

                  for i in range(self.problem.n_jobs)]

        def capacity_respected(resource_index, time, calendar_array: np.array):
            total_weight = model.sum()
            for task in range(nb_tasks):
                for mode in range(nb_modes[task]):
                    total_weight.add_operand(
                        weights[task][mode][resource_index]
                        * model.contains(tasks_in_mode[task][mode], time)
                    )
            return model.leq(total_weight, calendar_array[time])
        nb_resources = len(self.problem.resources_list)
        calendar = {res: self.problem.get_resource_availability_array(res)
                    for res in self.problem.resources_list}

        for i_r in range(nb_resources):
            res = self.problem.resources_list[i_r]
            if res not in self.problem.non_renewable_resources:
                capacity_respected_lambda = model.lambda_function(lambda time:
                                                                  capacity_respected(i_r, time,
                                                                                     model.array(calendar[res])))
                model.constraint(model.and_(model.range(makespan), capacity_respected_lambda))
            else:
                total_weight = model.sum()
                for task in range(nb_tasks):
                    for mode in range(nb_modes[task]):
                        total_weight.add_operand(weights[task][mode][i_r] * present_mode[task][mode])
                model.constraint(model.leq(total_weight, self.problem.get_max_resource_capacity(res)))
                pass
        # Objective: Minimize the makespan
        model.minimize(makespan)
        model.close()
        self.ls = ls
        self.model = model
        self.vars["intervals"] = tasks
        self.vars["present_mode"] = present_mode
        self.vars["modes_mapping"] = modes
        self.modelisation = ModelisationLSolver.OPT_INTERVAL

    def retrieve_solution(self) -> ResultStorage:
        if self.modelisation == ModelisationLSolver.INTERVAL:
            return self.retrieve_solution_interval()
        if self.modelisation == ModelisationLSolver.OPT_INTERVAL:
            return self.retrieve_solution_opt_interval()
        # if self.modelisation == ModelisationLSolver.INTEGER:
        #     return self.retrieve_solution_integer()

    def retrieve_solution_interval(self):
        schedule = {}
        modes = {}
        for i in range(self.problem.n_jobs):
            t = self.problem.tasks_list[i]
            start = int(self.vars["intervals"][i].value.start())
            end = int(self.vars["intervals"][i].value.end())
            mode = int(self.vars["modes"][i].value)
            real_mode = self.modes_correspondance[i][mode]
            modes[t] = real_mode
            schedule[t] = {"start_time": start, "end_time": end}
        sol = RcpspSolution(problem=self.problem,
                            rcpsp_schedule=schedule,
                            rcpsp_modes=[modes[k] for k in self.problem.tasks_list_non_dummy])
        fit = self.aggreg_from_sol(sol)
        logger.info(f"fitness = {fit}")
        return sol

    def retrieve_solution_opt_interval(self):
        schedule = {}
        modes = {}
        for i in range(self.problem.n_jobs):
            t = self.problem.tasks_list[i]
            start = int(self.vars["intervals"][i].value.start())
            end = int(self.vars["intervals"][i].value.end())
            for m in range(len(self.vars["modes_mapping"][i])):
                mode = self.vars["modes_mapping"][i][m]
                if self.vars["present_mode"][i][m].value == 1:
                    modes[t] = mode
            schedule[t] = {"start_time": start, "end_time": end}
        sol = RcpspSolution(problem=self.problem,
                            rcpsp_schedule=schedule,
                            rcpsp_modes=[modes[k] for k in self.problem.tasks_list_non_dummy])
        fit = self.aggreg_from_sol(sol)
        logger.info(f"fitness = {fit}")
        return sol

