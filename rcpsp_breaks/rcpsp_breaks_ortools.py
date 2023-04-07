from typing import List, Optional, Dict, Hashable, Tuple
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPSolution
from discrete_optimization.generic_tools.cp_tools import CPSolver, ParametersCP
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction, \
    build_aggreg_function_and_params_objective
from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import LinearExpr, Domain, VarArraySolutionPrinter
import numpy as np
import logging


logger = logging.getLogger(__name__)


def compute_duration_function_time(orig_duration: int,
                                   resource_calendar: np.ndarray,
                                   cumulative_resource_calendar: np.ndarray):
    duration = -np.ones((cumulative_resource_calendar.shape[0]))
    for i in range(cumulative_resource_calendar.shape[0]):
        if resource_calendar[i] == 0:
            continue
        x = cumulative_resource_calendar[i]
        if x == 0:
            continue
        index = next((j for j in range(i, cumulative_resource_calendar.shape[0])
                      if cumulative_resource_calendar[j] == x + orig_duration), None)
        if index is not None:
            duration[i] = index - i
        else:
            break
    return duration


def compute_duration_function_time_cluster(orig_duration: int,
                                           resource_calendar: np.ndarray,
                                           cumulative_resource_calendar: np.ndarray):
    duration = -np.ones((cumulative_resource_calendar.shape[0]))
    dict_of_interval_per_duration = {}
    current_interval = [0, 0]
    for i in range(cumulative_resource_calendar.shape[0]):
        if resource_calendar[i] == 0:
            if duration[i] == duration[i - 1]:
                current_interval[1] = i
            else:
                prev_d = duration[i - 1]
                if prev_d not in dict_of_interval_per_duration:
                    dict_of_interval_per_duration[prev_d] = []
                dict_of_interval_per_duration[prev_d] += [[current_interval[0], current_interval[1]]]
                current_interval = [i, i]
            continue
        x = cumulative_resource_calendar[i]
        if x == 0:
            continue
        index = next((j for j in range(i, cumulative_resource_calendar.shape[0])
                      if cumulative_resource_calendar[j] == x + orig_duration), None)
        if index is not None:
            duration[i] = index - i
            if i >= 1:
                if duration[i] == duration[i-1]:
                    current_interval[1] = i
                else:
                    prev_d = duration[i-1]
                    if prev_d not in dict_of_interval_per_duration:
                        dict_of_interval_per_duration[prev_d] = []
                    dict_of_interval_per_duration[prev_d] += [[current_interval[0], current_interval[1]]]
                    current_interval = [i, i]
        else:
            break
    return duration, dict_of_interval_per_duration


def compute_duration_tasks_function_time(rcpsp_model: RCPSPModel, method=None):
    if method is None:
        method = compute_duration_function_time
    resource_calendar_dict = {r: rcpsp_model.get_resource_availability_array(r)
                              == rcpsp_model.get_max_resource_capacity(r)  # "binarize" the calendar
                              for r in rcpsp_model.resources_list}
    cumulative_calendar_dict = {r: np.cumsum(resource_calendar_dict[r]) for r in resource_calendar_dict}
    durations = {t: None for t in rcpsp_model.tasks_list}
    for t in rcpsp_model.tasks_list:
        index_not_zeros = [r for r in rcpsp_model.resources_list
                           if rcpsp_model.mode_details[t][1].get(r, 0) > 0]
        if len(index_not_zeros) == 0:
            continue
        else:
            orig_duration = rcpsp_model.mode_details[t][1]["duration"]
            durations[t] = method(orig_duration=orig_duration,
                                  resource_calendar=resource_calendar_dict[index_not_zeros[0]],
                                  cumulative_resource_calendar=cumulative_calendar_dict[index_not_zeros[0]])
    return durations


class OrtoolsRCPSPBreakSolver(CPSolver):
    def __init__(self, problem: RCPSPModel,
                 params_objective_function: Optional[ParamsObjectiveFunction] = None):
        self.problem = problem
        (
            self.aggreg_sol,
            self.aggreg_from_dict_values,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            self.problem, params_objective_function=params_objective_function
        )
        self.model: cp_model.CpModel = None
        self.variables: Dict[str, List[LinearExpr]] = {}

    def init_model_fixed_duration_no_calendar(self, **kwargs):
        model = cp_model.CpModel()
        horizon = int(self.problem.horizon)
        nb_tasks = self.problem.n_jobs
        duration = [self.problem.mode_details[t][1]["duration"] for t in self.problem.tasks_list]
        starts = [
            model.NewIntVar(0, horizon - 1, f"start_task[{i}]")
            for i in range(nb_tasks)
        ]
        ends = [
            model.NewIntVar(0, horizon - 1, f"end_task[{i}]")
            for i in range(nb_tasks)
        ]
        durs = [model.NewConstant(int(duration[i])) for i in range(nb_tasks)]
        intervals = [
            model.NewIntervalVar(
                starts[i], durs[i], ends[i], f"interval_task[{i}]"
            )
            for i in range(len(durs))
        ]
        makespan = model.NewIntVar(0, horizon - 1, "makespan")
        for t in self.problem.successors:
            for s in self.problem.successors[t]:
                model.Add(ends[self.problem.index_task[t]] <= starts[self.problem.index_task[s]])
        for r in self.problem.resources_list:
            model.AddCumulative(
                [intervals[t] for t in range(nb_tasks)
                 if self.problem.mode_details[self.problem.tasks_list[t]][1].get(r, 0) > 0],
                [int(self.problem.mode_details[self.problem.tasks_list[t]][1].get(r, 0)) for t in range(nb_tasks)
                 if self.problem.mode_details[self.problem.tasks_list[t]][1].get(r, 0) > 0],
                self.problem.get_max_resource_capacity(r),
            )
        model.AddMaxEquality(makespan, ends)
        starts_hint = kwargs.get('starts_hint', None)
        if starts_hint is not None:
            for t in starts_hint:
                if t in self.problem.index_task:
                    index = self.problem.index_task[t]
                    model.AddHint(starts[index], int(starts_hint[t]))
        model.Minimize(makespan)
        self.variables["starts"] = starts
        self.variables["ends"] = ends
        self.model = model

    def init_model_durations_element_constraint(self, **kwargs):
        model = cp_model.CpModel()
        horizon = int(self.problem.horizon)
        nb_tasks = self.problem.n_jobs
        duration = [self.problem.mode_details[t][1]["duration"] for t in self.problem.tasks_list]
        starts = [
            model.NewIntVar(0, horizon - 1, f"start_task[{i}]")
            for i in range(nb_tasks)
        ]
        ends = [
            model.NewIntVar(0, horizon - 1, f"end_task[{i}]")
            for i in range(nb_tasks)
        ]
        # durs = [model.NewConstant(int(duration[i])) for i in range(nb_tasks)]
        duration_task = [model.NewIntVar(int(duration[i]), int(5 * duration[i]),
                                         f"duration_task[{i}]") for i in range(nb_tasks)]
        durs = compute_duration_tasks_function_time(self.problem)
        for i in range(nb_tasks):
            t = self.problem.tasks_list[i]
            if durs[t] is None:
                continue
            dd = [model.NewConstant(int(durs[t][j])) for j in range(len(durs[t]))]
            model.AddElement(index=starts[i],
                             variables=dd,
                             target=duration_task[i])
        intervals = [
            model.NewIntervalVar(
                starts[i], duration_task[i], ends[i], f"interval_task[{i}]"
            )
            for i in range(len(durs))
        ]
        makespan = model.NewIntVar(0, horizon - 1, "makespan")
        for t in self.problem.successors:
            for s in self.problem.successors[t]:
                model.Add(ends[self.problem.index_task[t]] <= starts[self.problem.index_task[s]])
        for r in self.problem.resources_list:
            model.AddCumulative(
                [intervals[t] for t in range(nb_tasks)
                 if self.problem.mode_details[self.problem.tasks_list[t]][1].get(r, 0) > 0],
                [int(self.problem.mode_details[self.problem.tasks_list[t]][1].get(r, 0)) for t in range(nb_tasks)
                 if self.problem.mode_details[self.problem.tasks_list[t]][1].get(r, 0) > 0],
                self.problem.get_max_resource_capacity(r),
            )
        model.AddMaxEquality(makespan, ends)
        starts_hint = kwargs.get('starts_hint', None)
        if starts_hint is not None:
            for t in starts_hint:
                if t in self.problem.index_task:
                    index = self.problem.index_task[t]
                    model.AddHint(starts[index], int(starts_hint[t]))
        model.Minimize(makespan)
        self.model = model
        self.variables["starts"] = starts
        self.variables["ends"] = ends

    def init_model_durations_indicator_method(self, **kwargs):
        # https://groups.google.com/g/or-tools-discuss/c/DA_4Pniwhn8?pli=1 Solution
        model = cp_model.CpModel()
        horizon = int(self.problem.horizon)
        nb_tasks = self.problem.n_jobs
        duration = [self.problem.mode_details[t][1]["duration"] for t in self.problem.tasks_list]
        starts = [
            model.NewIntVar(0, horizon - 1, f"start_task[{i}]")
            for i in range(nb_tasks)
        ]
        ends = [
            model.NewIntVar(0, horizon - 1, f"end_task[{i}]")
            for i in range(nb_tasks)
        ]
        # durs = [model.NewConstant(int(duration[i])) for i in range(nb_tasks)]
        duration_task = [model.NewIntVar(int(duration[i]), int(5 * duration[i]),
                                         f"duration_task[{i}]") for i in range(nb_tasks)]
        durs: Dict[Hashable, Tuple[List[int],
                                   Dict[int, List[Tuple[int, int]]]]] = \
            compute_duration_tasks_function_time(self.problem,
                                                 method=compute_duration_function_time_cluster)
        dictionnary_indicators = {}
        for i in range(nb_tasks):
            t = self.problem.tasks_list[i]
            if durs[t] is None:
                continue
            keys_ = set()
            for possible_duration in durs[t][1]:
                interval = Domain.FromIntervals(durs[t][1][possible_duration])
                # if possible_duration < 0:
                # Not sure if relevant.
                #    model.AddLinearExpressionInDomain(starts[i], interval.Complement())
                #    continue
                # Indicator of task t to be of size "possible_duration"
                dictionnary_indicators[(t, possible_duration)] = model.NewBoolVar(f"d_{t,possible_duration}")
                keys_.add((t, possible_duration))
                model.AddLinearExpressionInDomain(starts[i], interval).OnlyEnforceIf(
                    dictionnary_indicators[(t, possible_duration)])
                model.Add(duration_task[i] == int(possible_duration)).OnlyEnforceIf(
                    dictionnary_indicators[(t, possible_duration)])
            model.Add(sum([dictionnary_indicators[k]
                           for k in keys_]) == 1)  # Only one indicator activated.
        intervals = [
            model.NewIntervalVar(
                starts[i], duration_task[i], ends[i], f"interval_task[{i}]"
            )
            for i in range(len(durs))
        ]
        makespan = model.NewIntVar(0, horizon - 1, "makespan")
        for t in self.problem.successors:
            for s in self.problem.successors[t]:
                model.Add(ends[self.problem.index_task[t]] <= starts[self.problem.index_task[s]])
        for r in self.problem.resources_list:
            model.AddCumulative(
                [intervals[t] for t in range(nb_tasks)
                 if self.problem.mode_details[self.problem.tasks_list[t]][1].get(r, 0) > 0],
                [int(self.problem.mode_details[self.problem.tasks_list[t]][1].get(r, 0)) for t in range(nb_tasks)
                 if self.problem.mode_details[self.problem.tasks_list[t]][1].get(r, 0) > 0],
                self.problem.get_max_resource_capacity(r),
            )
        model.AddMaxEquality(makespan, ends)
        starts_hint = kwargs.get('starts_hint', None)
        if starts_hint is not None:
            for t in starts_hint:
                if t in self.problem.index_task:
                    index = self.problem.index_task[t]
                    model.AddHint(starts[index], int(starts_hint[t]))
        model.Minimize(makespan)
        self.model = model
        self.variables["starts"] = starts
        self.variables["ends"] = ends
        self.variables["indicators"] = dictionnary_indicators

    def solve(self, parameters_cp: Optional[ParametersCP] = None, **kwargs) -> ResultStorage:
        if parameters_cp is None:
            parameters_cp = ParametersCP.default()
        # Search for a feasible solution
        if self.model is None:
            self.init_model(**kwargs)
        add_solution_callback = kwargs.get("add_solution_callback", False)
        if add_solution_callback:
            solution_callback = VarArraySolutionPrinter(variables=
                                                        [self.variables["ends"]
                                                        [self.problem.index_task[self.problem.sink_task]]])
        else:
            solution_callback = None
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = parameters_cp.time_limit
        status = solver.Solve(self.model, solution_callback=solution_callback)
        st = ""
        if status == cp_model.OPTIMAL:
            st = "OPTIMAL"
        elif status == cp_model.INFEASIBLE:
            st = "INFEASIBLE"
        elif status == cp_model.FEASIBLE:
            st = "FEASIBLE"
        elif status == cp_model.UNKNOWN:
            st = "UNKNOWN"
        logger.info(f"Solver finished, status {st}")
        starts = [solver.Value(s) for s in self.variables["starts"]]
        ends = [solver.Value(s) for s in self.variables["ends"]]
        if "indicators" in self.variables:
            for key in self.variables["indicators"]:
                logger.debug(f"key : {key,solver.Value(self.variables['indicators'][key])}")
        schedule = {self.problem.tasks_list[t]: {"start_time": starts[t],
                                                 "end_time": ends[t]}
                    for t in range(len(starts))}
        sol = RCPSPSolution(problem=self.problem,
                            rcpsp_modes=[1] * self.problem.n_jobs_non_dummy,
                            rcpsp_schedule=schedule)
        fit = self.aggreg_sol(sol)
        res = ResultStorage(list_solution_fits=[(sol, fit)],
                            mode_optim=self.params_objective_function.sense_function)
        res.st = st
        return res
