import numpy as np
from typing import Optional, List, Any

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_problem import Problem, ParamsObjectiveFunction
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPSolution
from discrete_optimization.generic_tools.do_solver import SolverDO
import localsolver
import logging
logger = logging.getLogger(__name__)


def build_intensity_availability(problem: RCPSPModel):
    resource_calendar_dict = {
        r: np.array(problem.get_resource_availability_array(r))/problem.get_max_resource_capacity(r)
        for r in problem.resources_list
    }
    return resource_calendar_dict


class HexalyRCPSPBreakSolver(SolverDO):
    problem: RCPSPModel

    def __init__(self, problem: Problem,
                 params_objective_function: Optional[ParamsObjectiveFunction] = None,
                 **kwargs: Any):
        super().__init__(problem, params_objective_function, **kwargs)
        self.ls = None
        self.variables = None

    def init_model(self, **kwargs: Any) -> None:
        intensities = build_intensity_availability(self.problem)

        ls = localsolver.LocalSolver()
        #
        # Declare the optimization model
        #
        model = ls.model
        intensities_array = {r: model.array(intensities[r]) for r in intensities}
        # Trivial upper bound for the start times of the tasks
        horizon = self.problem.horizon
        # Integer decisions: start time of each task
        tasks = self.problem.tasks_list
        interval_tasks = {t: model.interval(0, horizon) for t in tasks}
        for t in tasks:
            if t not in self.problem.successors:
                continue
            for s_t in self.problem.successors[t]:
                model.constraint(interval_tasks[t] < interval_tasks[s_t])

        for t in tasks:
            resource_not_zeros = [
                r
                for r in self.problem.resources_list
                if self.problem.mode_details[t][1].get(r, 0) > 0
            ]
            duration = self.problem.mode_details[t][1]["duration"]
            if len(resource_not_zeros) > 0:
                for r in resource_not_zeros:
                    intensity_lambda = model.lambda_function(lambda t:
                                                             model.at(intensities_array[r], t))
                    model.constraint(
                        model.sum(interval_tasks[t], intensity_lambda)
                        >= duration)
                model.constraint(model.length(interval_tasks[t]) < duration*3)
        # Makespan: end of the last task
        makespan = model.end(interval_tasks[self.problem.sink_task])
        nb_resources = len(self.problem.resources_list)
        # Cumulative resource constraints
        # capacity = [self.rcpsp_model.get_max_resource_capacity(r) for r in self.rcpsp_model.resources_list]
        weight = [[self.problem.mode_details[t][1][r] for r in self.problem.resources_list] for t in tasks]
        for i_r in range(nb_resources):
            capacity_respected = model.lambda_function(
                lambda time: model.sum(
                    weight[index_task][i_r] * model.contains(interval_tasks[tasks[index_task]], time)
                    for index_task in range(self.problem.n_jobs))
                             <= self.problem.get_max_resource_capacity(self.problem.resources_list[i_r]))
            model.constraint(model.and_(model.range(makespan), capacity_respected))
        # Minimize the makespan

        if self.problem.includes_special_constraint():
            for t in self.problem.special_constraints.start_times_window:
                l,u = self.problem.special_constraints.start_times_window[t]
                if l is not None:
                    model.constraint(model.start(interval_tasks[t]) >= l)
                if u is not None:
                    model.constraint(model.start(interval_tasks[t]) <= u)
            for t in self.problem.special_constraints.end_times_window:
                l,u = self.problem.special_constraints.end_times_window[t]
                if l is not None:
                    model.constraint(model.end(interval_tasks[t]) >= l)
                if u is not None:
                    model.constraint(model.end(interval_tasks[t]) <= u)
        model.minimize(makespan)
        model.close()
        self.ls = ls
        self.model = model
        self.variables = {}
        self.variables["intervals"] = interval_tasks
        warm_start = kwargs.get("warm_start", True)
        warm_start_solution: Optional[RCPSPSolution] = kwargs.get("warm_start_solution", None)
        if warm_start:
            if warm_start_solution is None:
                warm_start_solution = self.problem.get_dummy_solution()
            if len(warm_start_solution.rcpsp_schedule) == self.problem.n_jobs:
                for st in self.variables["intervals"]:
                    self.variables["intervals"][st].value = localsolver.LSInterval(warm_start_solution.get_start_time(st),
                                                                                   warm_start_solution.get_end_time(st))
                    # self.vars["end"][st].value = warm_start_solution.get_end_time(st)
            else:
                # skipping
                pass

    def solve(self, callbacks: Optional[List[Callback]] = None, **kwargs: Any) -> ResultStorage:
        if self.ls is None:
            self.init_model(**kwargs)
        time_limit = kwargs.get("timeout_seconds", 100)
        self.ls.param.time_limit = time_limit
        logger.info("Solve started")
        self.ls.solve()
        logger.info("Solve finished")
        logger.info(f"Stats of solver : {self.ls.statistics}")
        return self.retrieve_solution()

    def retrieve_solution(self):
        schedule = {}
        for t in self.problem.tasks_list:
            start = int(self.variables["intervals"][t].value.start())
            end = int(self.variables["intervals"][t].value.end())
            schedule[t] = {"start_time": start, "end_time": end}
        sol = RCPSPSolution(problem=self.problem,
                            rcpsp_schedule=schedule)
        fit = self.aggreg_from_sol(sol)
        list_solution_fit = [(sol, fit)]
        return ResultStorage(list_solution_fits=list_solution_fit,
                             mode_optim=self.params_objective_function.sense_function)
