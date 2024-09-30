# Monomode rcpsp solver

import logging
from typing import Any, Dict, Optional, Hashable
import localsolver
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.rcpsp_model import (
    RCPSPModel, RCPSPSolution
)
from discrete_optimization.rcpsp.solver.rcpsp_solver import SolverRCPSP
from hexaly_solvers.hexaly_solver import HexalySolver
from enum import Enum
logger = logging.getLogger(__name__)


class ModelisationLSolver(Enum):
    INTEGER = 0
    INTERVAL = 1


class LSolverMonomodeRCPSP(HexalySolver, SolverRCPSP):
    def __init__(self, problem: RCPSPModel,
                 params_objective_function: Optional[ParamsObjectiveFunction] = None, **kwargs: Any):
        super().__init__(problem, params_objective_function, **kwargs)
        self.rcpsp_model = problem
        assert not problem.is_rcpsp_multimode()
        self.ls: Optional[localsolver.LocalSolver] = None
        self.model: Optional[localsolver.LSModel] = None
        self.vars: Optional[Dict[str, Dict[Hashable, localsolver.LSExpression]]] = {}
        self.modelisation: Optional[ModelisationLSolver] = None

    def init_model(self, modelisation_type: ModelisationLSolver = ModelisationLSolver.INTERVAL,
                   **kwargs):
        if modelisation_type == ModelisationLSolver.INTERVAL:
            self.init_model_interval(**kwargs)
            self.modelisation = modelisation_type
        if modelisation_type == ModelisationLSolver.INTEGER:
            self.init_model_integer(**kwargs)
            self.modelisation = modelisation_type

    def init_model_integer(self, **kwargs):
        ls = localsolver.LocalSolver()
        #
        # Declare the optimization model
        #
        model = ls.model
        # Trivial upper bound for the start times of the tasks
        horizon = self.rcpsp_model.horizon
        # Integer decisions: start time of each task
        tasks = self.rcpsp_model.tasks_list
        start = {t: model.int(0, horizon) for t in tasks}
        end = {t: start[t] + self.rcpsp_model.mode_details[t][1]["duration"]
               for t in tasks}
        for t in tasks:
            for s_t in self.rcpsp_model.successors[t]:
                model.constraint(end[t] <= start[s_t])
        # Makespan: end of the last task
        makespan = end[self.rcpsp_model.sink_task]
        nb_resources = len(self.rcpsp_model.resources_list)
        # Cumulative resource constraints
        # capacity = [self.rcpsp_model.get_max_resource_capacity(r) for r in self.rcpsp_model.resources_list]
        weight = [[self.rcpsp_model.mode_details[t][1][r] for r in self.rcpsp_model.resources_list] for t in tasks]
        for i_r in range(nb_resources):
            capacity_respected = model.lambda_function(
                lambda d: model.leq(
                    model.sum(
                        weight[t][i_r] * model.and_(model.leq(start[tasks[t]], d), model.gt(end[tasks[t]], d))
                        for t in range(self.rcpsp_model.n_jobs)),
                    self.rcpsp_model.get_resource_available(self.rcpsp_model.resources_list[i_r],
                                                            d)))
            model.constraint(model.and_(model.range(makespan), capacity_respected))

        # Minimize the makespan
        model.minimize(makespan)
        model.close()
        self.ls = ls
        self.model = model
        self.vars["start"] = start
        self.vars["end"] = end

        warm_start = kwargs.get("warm_start", True)
        warm_start_solution: Optional[RCPSPSolution] = kwargs.get("warm_start_solution", None)
        if warm_start:
            if warm_start_solution is None:
                warm_start_solution = self.rcpsp_model.get_dummy_solution()
            for st in self.vars["start"]:
                self.vars["start"][st].value = warm_start_solution.get_start_time(st)
                # self.vars["end"][st].value = warm_start_solution.get_end_time(st)

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
        interval_tasks = {t: model.interval(0, horizon) for t in tasks}
        for t in interval_tasks:
            model.constraint(model.length(interval_tasks[t]) ==
                             self.rcpsp_model.mode_details[t][1]["duration"])
        for t in tasks:
            if t not in self.rcpsp_model.successors:
                continue
            for s_t in self.rcpsp_model.successors[t]:
                model.constraint(interval_tasks[t] < interval_tasks[s_t])
        # Makespan: end of the last task
        makespan = model.end(interval_tasks[self.rcpsp_model.sink_task])
        nb_resources = len(self.rcpsp_model.resources_list)
        # Cumulative resource constraints
        # capacity = [self.rcpsp_model.get_max_resource_capacity(r) for r in self.rcpsp_model.resources_list]
        weight = [[self.rcpsp_model.mode_details[t][1][r] for r in self.rcpsp_model.resources_list] for t in tasks]
        for i_r in range(nb_resources):
            capacity_respected = model.lambda_function(
                lambda time: model.sum(
                        weight[index_task][i_r] * model.contains(interval_tasks[tasks[index_task]], time)
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
        warm_start = kwargs.get("warm_start", True)
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
        if self.modelisation == ModelisationLSolver.INTEGER:
            return self.retrieve_solution_integer()

    def retrieve_solution_integer(self) -> ResultStorage:
        schedule = {}
        for t in self.rcpsp_model.tasks_list:
            start = int(self.vars["start"][t].value)
            end = int(self.vars["end"][t].value)
            schedule[t] = {"start_time": start, "end_time": end}
        sol = RCPSPSolution(problem=self.rcpsp_model,
                            rcpsp_schedule=schedule)
        return sol

    def retrieve_solution_interval(self):
        schedule = {}
        for t in self.rcpsp_model.tasks_list:
            start = int(self.vars["intervals"][t].value.start())
            end = int(self.vars["intervals"][t].value.end())
            schedule[t] = {"start_time": start, "end_time": end}
        sol = RCPSPSolution(problem=self.rcpsp_model,
                            rcpsp_schedule=schedule)
        return sol
