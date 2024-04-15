# Monomode rcpsp solver

import logging
from typing import Any, Dict, List, Optional, Hashable
import localsolver
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.rcpsp_model import (
    RCPSPModel, RCPSPSolution
)
from discrete_optimization.generic_tools.do_solver import SolverDO
from enum import Enum
logger = logging.getLogger(__name__)


class ModelisationLSolver(Enum):
    #INTEGER = 0
    INTERVAL = 1


class LSolverMultiModeRCPSP(SolverDO):
    def __init__(self,  rcpsp_model: RCPSPModel,
                 params_objective_function: Optional[ParamsObjectiveFunction] = None):
        self.rcpsp_model = rcpsp_model
        (
            self.aggreg_sol,
            self.aggreg_from_dict_values,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            self.rcpsp_model, params_objective_function=params_objective_function
        )
        self.ls: Optional[localsolver.LocalSolver] = None
        self.model: Optional[localsolver.LSModel] = None
        self.vars: Optional[Dict[str, Dict[Hashable, localsolver.LSExpression]]] = {}
        self.modelisation: Optional[ModelisationLSolver] = None

    def init_model(self, modelisation_type: ModelisationLSolver=ModelisationLSolver.INTERVAL,
                   **kwargs):
        if modelisation_type == ModelisationLSolver.INTERVAL:
            self.init_model_interval(**kwargs)
            self.modelisation = modelisation_type
        #if modelisation_type == ModelisationLSolver.INTEGER:
        #    self.init_model_integer(**kwargs)
        #    self.modelisation = modelisation_type

    # def init_model_integer(self, **kwargs):
    #     ls = localsolver.LocalSolver()
    #     #
    #     # Declare the optimization model
    #     #
    #     model = ls.model
    #     # Trivial upper bound for the start times of the tasks
    #     horizon = self.rcpsp_model.horizon
    #     # Integer decisions: start time of each task
    #     tasks = self.rcpsp_model.tasks_list
    #     start = {t: model.int(0, horizon) for t in tasks}
    #     end = {t: start[t] + self.rcpsp_model.mode_details[t][1]["duration"]
    #            for t in tasks}
    #     for t in tasks:
    #         for s_t in self.rcpsp_model.successors[t]:
    #             model.constraint(end[t] <= start[s_t])
    #     # Makespan: end of the last task
    #     makespan = end[self.rcpsp_model.sink_task]
    #     nb_resources = len(self.rcpsp_model.resources_list)
    #     # Cumulative resource constraints
    #     # capacity = [self.rcpsp_model.get_max_resource_capacity(r) for r in self.rcpsp_model.resources_list]
    #     weight = [[self.rcpsp_model.mode_details[t][1][r] for r in self.rcpsp_model.resources_list] for t in tasks]
    #     for i_r in range(nb_resources):
    #         capacity_respected = model.lambda_function(
    #             lambda d: model.leq(
    #                 model.sum(
    #                     weight[t][i_r] * model.and_(model.leq(start[tasks[t]], d), model.gt(end[tasks[t]], d))
    #                     for t in range(self.rcpsp_model.n_jobs)),
    #                 self.rcpsp_model.get_resource_available(self.rcpsp_model.resources_list[i_r],
    #                                                         d)))
    #         model.constraint(model.and_(model.range(makespan), capacity_respected))
    #
    #     # Minimize the makespan
    #     model.minimize(makespan)
    #     model.close()
    #     self.ls = ls
    #     self.model = model
    #     self.vars["start"] = start
    #     self.vars["end"] = end
    #
    #     warm_start = kwargs.get("warm_start", True)
    #     warm_start_solution: Optional[RCPSPSolution] = kwargs.get("warm_start_solution", None)
    #     if warm_start:
    #         if warm_start_solution is None:
    #             warm_start_solution = self.rcpsp_model.get_dummy_solution()
    #         for st in self.vars["start"]:
    #             self.vars["start"][st].value = warm_start_solution.get_start_time(st)
    #             # self.vars["end"][st].value = warm_start_solution.get_end_time(st)

    def init_model_interval(self, **kwargs):
        ls = localsolver.LocalSolver()
        #
        # Declare the optimization model
        #
        model = ls.model
        # Trivial upper bound for the start times of the tasks
        horizon = self.rcpsp_model.horizon
        # Integer decisions: start time of each task
        tasks = self.rcpsp_model.tasks_list
        nb_tasks = self.rcpsp_model.n_jobs
        interval_tasks = [model.interval(0, horizon) for t in tasks]
        modes = [list(self.rcpsp_model.mode_details[t]) for t in tasks]
        modes_var = [model.int(0, len(modes[i])-1)
                     for i in range(self.rcpsp_model.n_jobs)]
        modes_array = model.array(modes_var)
        modes_details = [
            [
                [self.rcpsp_model.mode_details[tasks[i]][m].get(res, 0)
                 for res in self.rcpsp_model.resources_list]
                for m in modes[i]
            ]
            for i in range(self.rcpsp_model.n_jobs)]
        modes_details_array = [model.array(modes_details[i])
                               for i in range(self.rcpsp_model.n_jobs)]

        consumption_array = []
        nb_resources = len(self.rcpsp_model.resources_list)
        for i in range(self.rcpsp_model.n_jobs):
            c = []
            for i_res in range(nb_resources):
                min_cons = min([m[i_res] for m in modes_details[i]])
                max_cons = max([m[i_res] for m in modes_details[i]])
                c += [model.int(min_cons, max_cons)]
            consumption_array += [c]
        duration_of_modes = [[self.rcpsp_model.mode_details[tasks[i]][m]["duration"]
                              for m in modes[i]]
                             for i in range(self.rcpsp_model.n_jobs)]
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
            index_t = self.rcpsp_model.index_task[t]
            if t not in self.rcpsp_model.successors:
                continue
            for s_t in self.rcpsp_model.successors[t]:
                index_s_t = self.rcpsp_model.index_task[s_t]
                model.constraint(interval_tasks[index_t] < interval_tasks[index_s_t])
        # Makespan: end of the last task
        makespan = model.end(interval_tasks[self.rcpsp_model.index_task[self.rcpsp_model.sink_task]])
        nb_resources = len(self.rcpsp_model.resources_list)
        # Cumulative resource constraints
        for i_r in range(nb_resources):
            res = self.rcpsp_model.resources_list[i_r]
            if res in self.rcpsp_model.non_renewable_resources:
                model.constraint(model.sum(
                        consumption_array[index_task][i_r]
                        for index_task in range(self.rcpsp_model.n_jobs)) <=
                         self.rcpsp_model.get_max_resource_capacity(self.rcpsp_model.resources_list[i_r]))
            else:
                capacity_respected = model.lambda_function(
                    lambda time: model.sum(
                            consumption_array[index_task][i_r] * model.contains(interval_tasks[index_task], time)
                            for index_task in range(self.rcpsp_model.n_jobs)) <=
                                 self.rcpsp_model.get_resource_available(self.rcpsp_model.resources_list[i_r],
                                                                         time))
                model.constraint(model.and_(model.range(makespan), capacity_respected))
        # Minimize the makespan
        model.minimize(makespan)
        model.close()
        self.ls = ls
        self.model = model
        self.vars["intervals"] = interval_tasks
        self.vars["modes"] = modes_var
        self.modes_correspondance = [list(self.rcpsp_model.mode_details[t]) for t in tasks]
        warm_start = kwargs.get("warm_start", False)
        warm_start_solution: Optional[RCPSPSolution] = kwargs.get("warm_start_solution", None)
        if warm_start:
            if warm_start_solution is None:
                warm_start_solution = self.rcpsp_model.get_dummy_solution()
            for st in self.vars["intervals"]:
                self.vars["intervals"][st].value = localsolver.LSInterval(warm_start_solution.get_start_time(st),
                                                                          warm_start_solution.get_end_time(st))
                # self.vars["end"][st].value = warm_start_solution.get_end_time(st)

    def retrieve_solution(self) -> ResultStorage:
        if self.modelisation == ModelisationLSolver.INTERVAL:
            return self.retrieve_solution_interval()
        # if self.modelisation == ModelisationLSolver.INTEGER:
        #     return self.retrieve_solution_integer()

    def solve(self, **kwargs: Any) -> ResultStorage:
        if self.ls is None:
            self.init_model(**kwargs)
        time_limit = kwargs.get("timeout_seconds", 100)
        self.ls.param.time_limit = time_limit
        logger.info("Solve started")
        self.ls.solve()
        logger.info("Solve finished")
        logger.info(f"Stats of solver : {self.ls.statistics}")
        return self.retrieve_solution()

    # def retrieve_solution_integer(self) -> ResultStorage:
    #     schedule = {}
    #     for t in self.rcpsp_model.tasks_list:
    #         start = int(self.vars["start"][t].value)
    #         end = int(self.vars["end"][t].value)
    #         schedule[t] = {"start_time": start, "end_time": end}
    #     sol = RCPSPSolution(problem=self.rcpsp_model,
    #                         rcpsp_schedule=schedule)
    #     fit = self.aggreg_sol(sol)
    #     list_solution_fit = [(sol, fit)]
    #     return ResultStorage(list_solution_fits=list_solution_fit,
    #                          mode_optim=self.params_objective_function.sense_function)

    def retrieve_solution_interval(self):
        schedule = {}
        modes = {}
        for i in range(self.rcpsp_model.n_jobs):
            t = self.rcpsp_model.tasks_list[i]
            start = int(self.vars["intervals"][i].value.start())
            end = int(self.vars["intervals"][i].value.end())
            mode = int(self.vars["modes"][i].value)
            real_mode = self.modes_correspondance[i][mode]
            modes[t] = real_mode
            schedule[t] = {"start_time": start, "end_time": end}
        sol = RCPSPSolution(problem=self.rcpsp_model,
                            rcpsp_schedule=schedule,
                            rcpsp_modes=[modes[k] for k in self.rcpsp_model.tasks_list_non_dummy])
        fit = self.aggreg_sol(sol)
        list_solution_fit = [(sol, fit)]
        return ResultStorage(list_solution_fits=list_solution_fit,
                             mode_optim=self.params_objective_function.sense_function)



