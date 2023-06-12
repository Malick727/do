import logging
from enum import Enum
from typing import Any, Dict, Hashable, List, Optional, Tuple

import numpy as np
from discrete_optimization.generic_tools.cp_tools import CPSolver, ParametersCP
from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPSolution
try:
    from discrete_optimization.rcpsp.specialized_rcpsp.rcpsp_specialized_constraints import (
        RCPSPModelSpecialConstraints,
        SpecialConstraintsDescription,
    )
except:
    from discrete_optimization.rcpsp.specialized_rcpsp.rcpsp_specialized_constraints import \
        SpecialConstraintsDescription
from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import Domain, LinearExpr, VarArraySolutionPrinter

logger = logging.getLogger(__name__)


def compute_duration_function_time(
    orig_duration: int,
    resource_calendar: np.ndarray,
    cumulative_resource_calendar: np.ndarray,
):
    duration = -np.ones((cumulative_resource_calendar.shape[0]))
    for i in range(cumulative_resource_calendar.shape[0]):
        if resource_calendar[i] == 0:
            continue
        x = cumulative_resource_calendar[i]
        if x == 0:
            continue
        index = next(
            (
                j
                for j in range(i, cumulative_resource_calendar.shape[0])
                if cumulative_resource_calendar[j] == x + orig_duration - 1
            ),
            None,
        )
        if index is not None:
            duration[i] = index - i + 1
        else:
            break
    return duration


def compute_duration_function_time_cluster(
    orig_duration: int,
    resource_calendar: np.ndarray,
    cumulative_resource_calendar: np.ndarray,
):
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
                dict_of_interval_per_duration[prev_d] += [
                    [current_interval[0], current_interval[1]]
                ]
                current_interval = [i, i]
            continue
        x = cumulative_resource_calendar[i]
        if x == 0:
            continue
        index = next(
            (
                j
                for j in range(i, cumulative_resource_calendar.shape[0])
                if cumulative_resource_calendar[j] == x + orig_duration - 1
            ),
            None,
        )
        if index is not None:
            duration[i] = index - i + 1
            if i >= 1:
                if duration[i] == duration[i - 1]:
                    current_interval[1] = i
                else:
                    prev_d = duration[i - 1]
                    if prev_d not in dict_of_interval_per_duration:
                        dict_of_interval_per_duration[prev_d] = []
                    dict_of_interval_per_duration[prev_d] += [
                        [current_interval[0], current_interval[1]]
                    ]
                    current_interval = [i, i]
        else:
            break
    return duration, dict_of_interval_per_duration


def compute_duration_tasks_function_time(rcpsp_model: RCPSPModel, method=None):
    if method is None:
        method = compute_duration_function_time
    resource_calendar_dict = {
        r: rcpsp_model.get_resource_availability_array(r)
        == rcpsp_model.get_max_resource_capacity(r)  # "binarize" the calendar
        for r in rcpsp_model.resources_list
    }
    cumulative_calendar_dict = {
        r: np.cumsum(resource_calendar_dict[r]) for r in resource_calendar_dict
    }
    durations = {t: None for t in rcpsp_model.tasks_list}
    for t in rcpsp_model.tasks_list:
        index_not_zeros = [
            r
            for r in rcpsp_model.resources_list
            if rcpsp_model.mode_details[t][1].get(r, 0) > 0
        ]
        if len(index_not_zeros) == 0:
            continue
        elif len(index_not_zeros) == 1:
            orig_duration = rcpsp_model.mode_details[t][1]["duration"]
            durations[t] = method(
                orig_duration=orig_duration,
                resource_calendar=resource_calendar_dict[index_not_zeros[0]],
                cumulative_resource_calendar=cumulative_calendar_dict[
                    index_not_zeros[0]
                ],
            )
        else:
            orig_duration = rcpsp_model.mode_details[t][1]["duration"]
            tuple_res = tuple(
                [(r, rcpsp_model.mode_details[t][1][r]) for r in index_not_zeros]
            )
            if tuple_res not in resource_calendar_dict:
                b = np.array(resource_calendar_dict[index_not_zeros[0]])
                for r in tuple_res[1:]:
                    b *= resource_calendar_dict[r[0]]
                resource_calendar_dict[tuple_res] = b
                cumulative_calendar_dict[tuple_res] = np.cumsum(
                    resource_calendar_dict[tuple_res]
                )
            durations[t] = method(
                orig_duration=orig_duration,
                resource_calendar=resource_calendar_dict[tuple_res],
                cumulative_resource_calendar=cumulative_calendar_dict[tuple_res],
            )
    return durations


def compute_res_of_tasks(rcpsp_model: RCPSPModel):
    map_task_to_res = {}
    for t in rcpsp_model.tasks_list:
        index_not_zeros = [
            r
            for r in rcpsp_model.resources_list
            if rcpsp_model.mode_details[t][1].get(r, 0) > 0
        ]
        assert (
            len(index_not_zeros) <= 1
        )  # this modelisation only work with such problems
        if len(index_not_zeros) > 0:
            map_task_to_res[t] = index_not_zeros[0]
    return map_task_to_res


def compute_starts_per_resource(rcpsp_model: RCPSPModel):
    resource_calendar_dict = {
        r: rcpsp_model.get_resource_availability_array(r)
        == rcpsp_model.get_max_resource_capacity(r)  # "binarize" the calendar
        for r in rcpsp_model.resources_list
    }
    index_non_zeros = {
        r: np.nonzero(resource_calendar_dict[r])[0] for r in resource_calendar_dict
    }

    def get_real_date(res_name, date_from_resource_point_of_view):
        return index_non_zeros[res_name][date_from_resource_point_of_view]

    return index_non_zeros, get_real_date


class InitMethod(Enum):
    NO_CAL = 0
    ELEMENT_CNSTR = 1
    INDICATOR_CNSTR = 2
    PARALLEL_TIMELINE = 3


class OrtoolsRCPSPBreakSolver(CPSolver):
    def __init__(
        self,
        problem: RCPSPModel,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
    ):
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
        partial_solution = kwargs.get("partial_solution", None)
        if "special_constraints" in self.problem.__dict__.keys():
            if partial_solution is None:
                partial_solution = self.problem.special_constraints
        horizon = int(self.problem.horizon)
        tasks_list = self.problem.tasks_list
        min_start_time = {t: 0 for t in self.problem.tasks_list}
        max_start_time = {t: horizon - 1 for t in self.problem.tasks_list}
        min_end_time = {t: 0 for t in self.problem.tasks_list}
        max_end_time = {t: horizon - 1 for t in self.problem.tasks_list}
        if partial_solution is not None:
            for t in partial_solution.start_times_window:
                if partial_solution.start_times_window[t][0] is not None:
                    min_start_time[t] = partial_solution.start_times_window[t][0]
                if partial_solution.start_times_window[t][1] is not None:
                    max_start_time[t] = partial_solution.start_times_window[t][1]
            for t in partial_solution.end_times_window:
                if partial_solution.end_times_window[t][0] is not None:
                    min_end_time[t] = partial_solution.end_times_window[t][0]
                if partial_solution.end_times_window[t][1] is not None:
                    max_end_time[t] = partial_solution.end_times_window[t][1]
        model = cp_model.CpModel()
        horizon = int(self.problem.horizon)
        nb_tasks = self.problem.n_jobs
        duration = [
            self.problem.mode_details[t][1]["duration"] for t in self.problem.tasks_list
        ]
        starts = [
            model.NewIntVar(
                min_start_time[tasks_list[i]],
                max_start_time[tasks_list[i]],
                f"start_task[{i}]",
            )
            for i in range(nb_tasks)
        ]
        ends = [
            model.NewIntVar(
                min_end_time[tasks_list[i]],
                max_end_time[tasks_list[i]],
                f"end_task[{i}]",
            )
            for i in range(nb_tasks)
        ]
        if partial_solution is not None:
            self.add_generalized_constraints(
                starts_var=starts,
                ends_var=ends,
                special_constraints=partial_solution,
                model=model,
            )
        durs = [model.NewConstant(int(duration[i])) for i in range(nb_tasks)]
        intervals = [
            model.NewIntervalVar(starts[i], durs[i], ends[i], f"interval_task[{i}]")
            for i in range(len(durs))
        ]
        makespan = model.NewIntVar(0, horizon - 1, "makespan")
        for t in self.problem.successors:
            for s in self.problem.successors[t]:
                model.Add(
                    ends[self.problem.index_task[t]]
                    <= starts[self.problem.index_task[s]]
                )
        for r in self.problem.resources_list:
            model.AddCumulative(
                [
                    intervals[t]
                    for t in range(nb_tasks)
                    if self.problem.mode_details[self.problem.tasks_list[t]][1].get(
                        r, 0
                    )
                    > 0
                ],
                [
                    int(
                        self.problem.mode_details[self.problem.tasks_list[t]][1].get(
                            r, 0
                        )
                    )
                    for t in range(nb_tasks)
                    if self.problem.mode_details[self.problem.tasks_list[t]][1].get(
                        r, 0
                    )
                    > 0
                ],
                self.problem.get_max_resource_capacity(r),
            )
        model.AddMaxEquality(makespan, ends)
        starts_hint = kwargs.get("starts_hint", None)
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
        partial_solution = kwargs.get("partial_solution", None)
        if "special_constraints" in self.problem.__dict__.keys():
            if partial_solution is None:
                partial_solution = self.problem.special_constraints
        horizon = int(self.problem.horizon)
        min_start_time = {t: 0 for t in self.problem.tasks_list}
        max_start_time = {t: horizon - 1 for t in self.problem.tasks_list}
        min_end_time = {t: 0 for t in self.problem.tasks_list}
        max_end_time = {t: horizon - 1 for t in self.problem.tasks_list}
        model = cp_model.CpModel()
        if partial_solution is not None:
            for t in partial_solution.start_times_window:
                if partial_solution.start_times_window[t][0] is not None:
                    min_start_time[t] = partial_solution.start_times_window[t][0]
                if partial_solution.start_times_window[t][1] is not None:
                    max_start_time[t] = partial_solution.start_times_window[t][1]
            for t in partial_solution.end_times_window:
                if partial_solution.end_times_window[t][0] is not None:
                    min_end_time[t] = partial_solution.end_times_window[t][0]
                if partial_solution.end_times_window[t][1] is not None:
                    max_end_time[t] = partial_solution.end_times_window[t][1]
        tasks_list = self.problem.tasks_list
        nb_tasks = self.problem.n_jobs
        duration = [
            self.problem.mode_details[t][1]["duration"] for t in self.problem.tasks_list
        ]
        starts = [
            model.NewIntVar(
                lb=min_start_time[tasks_list[i]],
                ub=max_start_time[tasks_list[i]],
                name=f"start_task[{i}]",
            )
            for i in range(nb_tasks)
        ]
        ends = [
            model.NewIntVar(
                lb=min_end_time[tasks_list[i]],
                ub=max_end_time[tasks_list[i]],
                name=f"end_task[{i}]",
            )
            for i in range(nb_tasks)
        ]
        if partial_solution is not None:
            self.add_generalized_constraints(
                starts_var=starts,
                ends_var=ends,
                special_constraints=partial_solution,
                model=model,
            )
        duration_task = [
            model.NewIntVar(
                int(duration[i]), int(100 * duration[i]), f"duration_task[{i}]"
            )
            for i in range(nb_tasks)
        ]
        durs = compute_duration_tasks_function_time(self.problem)
        for i in range(nb_tasks):
            t = self.problem.tasks_list[i]
            if durs[t] is None:
                continue
            dd = [model.NewConstant(int(durs[t][j])) for j in range(len(durs[t]))]
            model.AddElement(index=starts[i], variables=dd, target=duration_task[i])
        intervals = [
            model.NewIntervalVar(
                starts[i], duration_task[i], ends[i], f"interval_task[{i}]"
            )
            for i in range(len(durs))
        ]
        makespan = model.NewIntVar(0, horizon - 1, "makespan")
        for t in self.problem.successors:
            for s in self.problem.successors[t]:
                model.Add(
                    ends[self.problem.index_task[t]]
                    <= starts[self.problem.index_task[s]]
                )
        for r in self.problem.resources_list:
            model.AddCumulative(
                [
                    intervals[t]
                    for t in range(nb_tasks)
                    if self.problem.mode_details[self.problem.tasks_list[t]][1].get(
                        r, 0
                    )
                    > 0
                ],
                [
                    int(
                        self.problem.mode_details[self.problem.tasks_list[t]][1].get(
                            r, 0
                        )
                    )
                    for t in range(nb_tasks)
                    if self.problem.mode_details[self.problem.tasks_list[t]][1].get(
                        r, 0
                    )
                    > 0
                ],
                int(self.problem.get_max_resource_capacity(r)),
            )
        model.AddMaxEquality(makespan, ends)
        starts_hint = kwargs.get("starts_hint", None)
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
        partial_solution = kwargs.get("partial_solution", None)
        if "special_constraints" in self.problem.__dict__.keys():
            if partial_solution is None:
                partial_solution = self.problem.special_constraints
        horizon = int(self.problem.horizon)
        min_start_time = {t: 0 for t in self.problem.tasks_list}
        max_start_time = {t: horizon - 1 for t in self.problem.tasks_list}
        min_end_time = {t: 0 for t in self.problem.tasks_list}
        max_end_time = {t: horizon - 1 for t in self.problem.tasks_list}
        if partial_solution is not None:
            for t in partial_solution.start_times_window:
                if partial_solution.start_times_window[t][0] is not None:
                    min_start_time[t] = partial_solution.start_times_window[t][0]
                if partial_solution.start_times_window[t][1] is not None:
                    max_start_time[t] = partial_solution.start_times_window[t][1]
            for t in partial_solution.end_times_window:
                if partial_solution.end_times_window[t][0] is not None:
                    min_end_time[t] = partial_solution.end_times_window[t][0]
                if partial_solution.end_times_window[t][1] is not None:
                    max_end_time[t] = partial_solution.end_times_window[t][1]
        model = cp_model.CpModel()
        tasks_list = self.problem.tasks_list
        nb_tasks = self.problem.n_jobs
        duration = [self.problem.mode_details[t][1]["duration"] for t in tasks_list]
        starts = [
            model.NewIntVar(
                lb=min_start_time[tasks_list[i]],
                ub=max_start_time[tasks_list[i]],
                name=f"start_task[{i}]",
            )
            for i in range(nb_tasks)
        ]
        ends = [
            model.NewIntVar(
                lb=min_end_time[tasks_list[i]],
                ub=max_end_time[tasks_list[i]],
                name=f"end_task[{i}]",
            )
            for i in range(nb_tasks)
        ]
        # durs = [model.NewConstant(int(duration[i])) for i in range(nb_tasks)]
        duration_task = [
            model.NewIntVar(
                int(duration[i]), int(100 * duration[i]), f"duration_task[{i}]"
            )
            for i in range(nb_tasks)
        ]
        if partial_solution is not None:
            self.add_generalized_constraints(
                starts_var=starts,
                ends_var=ends,
                special_constraints=partial_solution,
                model=model,
            )
        durs: Dict[
            Hashable, Tuple[List[int], Dict[int, List[Tuple[int, int]]]]
        ] = compute_duration_tasks_function_time(
            self.problem, method=compute_duration_function_time_cluster
        )
        dictionnary_indicators = {}
        for i in range(nb_tasks):
            t = self.problem.tasks_list[i]
            if durs[t] is None:
                continue
            if duration[i] == 0:
                continue
            keys_ = set()
            for possible_duration in durs[t][1]:
                interval = Domain.FromIntervals(durs[t][1][possible_duration])
                # if possible_duration < 0:
                # Not sure if relevant.
                #    model.AddLinearExpressionInDomain(starts[i], interval.Complement())
                #    continue
                # Indicator of task t to be of size "possible_duration"
                dictionnary_indicators[(t, possible_duration)] = model.NewBoolVar(
                    f"d_{t,possible_duration}"
                )
                keys_.add((t, possible_duration))
                model.AddLinearExpressionInDomain(starts[i], interval).OnlyEnforceIf(
                    dictionnary_indicators[(t, possible_duration)]
                )
                model.Add(duration_task[i] == int(possible_duration)).OnlyEnforceIf(
                    dictionnary_indicators[(t, possible_duration)]
                )
            model.Add(
                sum([dictionnary_indicators[k] for k in keys_]) == 1
            )  # Only one indicator activated.
        intervals = [
            model.NewIntervalVar(
                starts[i], duration_task[i], ends[i], f"interval_task[{i}]"
            )
            for i in range(len(durs))
        ]
        makespan = model.NewIntVar(0, horizon - 1, "makespan")
        for t in self.problem.successors:
            for s in self.problem.successors[t]:
                model.Add(
                    ends[self.problem.index_task[t]]
                    <= starts[self.problem.index_task[s]]
                )
        for r in self.problem.resources_list:
            model.AddCumulative(
                [
                    intervals[t]
                    for t in range(nb_tasks)
                    if self.problem.mode_details[self.problem.tasks_list[t]][1].get(
                        r, 0
                    )
                    > 0
                    and self.problem.mode_details[self.problem.tasks_list[t]][1][
                        "duration"
                    ]
                    > 0
                ],
                [
                    int(
                        self.problem.mode_details[self.problem.tasks_list[t]][1].get(
                            r, 0
                        )
                    )
                    for t in range(nb_tasks)
                    if self.problem.mode_details[self.problem.tasks_list[t]][1].get(
                        r, 0
                    )
                    > 0
                    and self.problem.mode_details[self.problem.tasks_list[t]][1][
                        "duration"
                    ]
                    > 0
                ],
                int(self.problem.get_max_resource_capacity(r)),
            )
        model.AddMaxEquality(makespan, ends)
        starts_hint = kwargs.get("starts_hint", None)
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

    def init_model_parallel_starts_vectors(self, **kwargs):
        model = cp_model.CpModel()
        horizon = int(self.problem.horizon)
        nb_tasks = self.problem.n_jobs
        duration = [
            self.problem.mode_details[t][1]["duration"] for t in self.problem.tasks_list
        ]
        starts_real = [
            model.NewIntVar(0, horizon - 1, f"start_task[{i}]") for i in range(nb_tasks)
        ]
        ends_real = [
            model.NewIntVar(0, horizon - 1, f"end_task[{i}]") for i in range(nb_tasks)
        ]
        for i in range(nb_tasks):
            model.Add(ends_real[i] >= starts_real[i] + duration[i])

        starts_resource_view = [
            model.NewIntVar(0, horizon - 1, f"start_task[{i}]") for i in range(nb_tasks)
        ]
        ends_resource_view = [
            model.NewIntVar(0, horizon - 1, f"end_task[{i}]") for i in range(nb_tasks)
        ]

        res_of_tasks: Dict[Hashable, str] = compute_res_of_tasks(self.problem)
        index_non_zeros, get_real_date = compute_starts_per_resource(self.problem)
        self.variables["b_starts"] = {}
        self.variables["b_ends"] = {}

        for i in range(nb_tasks):
            t = self.problem.tasks_list[i]
            if t not in res_of_tasks:
                model.Add(starts_real[i] == starts_resource_view[i])
                model.Add(ends_real[i] == ends_resource_view[i])
            else:
                b_starts = [
                    model.NewBoolVar(f"b_{i,time}")
                    for time in range(self.problem.horizon)
                ]
                b_ends = [
                    model.NewBoolVar(f"bends_{i, time}")
                    for time in range(self.problem.horizon)
                ]
                self.variables["b_starts"][i] = b_starts
                self.variables["b_ends"][i] = b_ends
                model.AddMapDomain(starts_resource_view[i], b_starts)
                model.AddMapDomain(ends_resource_view[i], b_ends)
                model.Add(
                    starts_real[i]
                    == LinearExpr.WeightedSum(
                        [
                            b_starts[j]
                            for j in range(len(index_non_zeros[res_of_tasks[t]]))
                        ],
                        index_non_zeros[res_of_tasks[t]],
                    )
                )
                model.Add(
                    ends_real[i]
                    == LinearExpr.WeightedSum(
                        [
                            b_ends[j]
                            for j in range(len(index_non_zeros[res_of_tasks[t]]))
                        ],
                        index_non_zeros[res_of_tasks[t]],
                    )
                )
                if True:
                    model.AddElement(
                        index=starts_resource_view[i],
                        variables=[
                            model.NewConstant(int(x))
                            for x in index_non_zeros[res_of_tasks[t]]
                        ],
                        target=starts_real[i],
                    )
                    model.AddElement(
                        index=ends_resource_view[i],
                        variables=[
                            model.NewConstant(int(x))
                            for x in index_non_zeros[res_of_tasks[t]]
                        ],
                        target=ends_real[i],
                    )
        durs = [model.NewConstant(int(duration[i])) for i in range(nb_tasks)]
        intervals_resource_view = [
            model.NewIntervalVar(
                starts_resource_view[i],
                durs[i],
                ends_resource_view[i],
                f"interval_task[{i}]",
            )
            for i in range(len(durs))
        ]
        makespan = model.NewIntVar(0, horizon - 1, "makespan")
        for t in self.problem.successors:
            for s in self.problem.successors[t]:
                model.Add(
                    ends_real[self.problem.index_task[t]]
                    <= starts_real[self.problem.index_task[s]]
                )
        for r in self.problem.resources_list:
            model.AddCumulative(
                [
                    intervals_resource_view[t]
                    for t in range(nb_tasks)
                    if self.problem.mode_details[self.problem.tasks_list[t]][1].get(
                        r, 0
                    )
                    > 0
                    and duration[t] > 0
                ],
                [
                    int(
                        self.problem.mode_details[self.problem.tasks_list[t]][1].get(
                            r, 0
                        )
                    )
                    for t in range(nb_tasks)
                    if self.problem.mode_details[self.problem.tasks_list[t]][1].get(
                        r, 0
                    )
                    > 0
                    and duration[t] > 0
                ],
                self.problem.get_max_resource_capacity(r),
            )
        model.AddMaxEquality(makespan, ends_real)
        starts_hint = kwargs.get("starts_hint", None)
        if starts_hint is not None:
            for t in starts_hint:
                if t in self.problem.index_task:
                    index = self.problem.index_task[t]
                    model.AddHint(starts_real[index], int(starts_hint[t]))
        model.Minimize(makespan)
        self.model = model
        self.variables["starts"] = starts_real
        self.variables["ends"] = ends_real

    def init_model(self, **args: Any) -> None:
        init_method = args.get("init_method", InitMethod.INDICATOR_CNSTR)
        if init_method == InitMethod.INDICATOR_CNSTR:
            self.init_model_durations_indicator_method(**args)
        if init_method == InitMethod.ELEMENT_CNSTR:
            self.init_model_durations_indicator_method(**args)
        if init_method == InitMethod.NO_CAL:
            self.init_model_fixed_duration_no_calendar(**args)
        if init_method == InitMethod.PARALLEL_TIMELINE:
            self.init_model_parallel_starts_vectors(**args)

    def add_generalized_constraints(
        self,
        starts_var: List[cp_model.IntVar],
        ends_var: List[cp_model.IntVar],
        special_constraints: Optional[SpecialConstraintsDescription],
        model: cp_model.CpModel,
    ):
        if special_constraints is None:
            return
        if special_constraints.start_times is not None:
            for task in special_constraints.start_times:
                s = special_constraints.start_times[task]
                if s is not None:
                    index_task = self.problem.index_task[task]
                    model.Add(starts_var[index_task] == s)
        if special_constraints.end_times is not None:
            for task in special_constraints.end_times:
                e = special_constraints.end_times[task]
                if e is not None:
                    index_task = self.problem.index_task[task]
                    model.Add(ends_var[index_task] == e)
        if (
            special_constraints.start_at_end is not None
            and len(special_constraints.start_at_end) >= 1
        ):
            for t1, t2 in special_constraints.start_at_end:
                index_task_1 = self.problem.index_task[t1]
                index_task_2 = self.problem.index_task[t2]
                model.Add(starts_var[index_task_2] == ends_var[index_task_1])
        if (
            special_constraints.start_together is not None
            and len(special_constraints.start_together) >= 1
        ):
            for t1, t2 in special_constraints.start_together:
                index_task_1 = self.problem.index_task[t1]
                index_task_2 = self.problem.index_task[t2]
                model.Add(starts_var[index_task_2] == starts_var[index_task_1])

        if (
            special_constraints.start_at_end_plus_offset is not None
            and len(special_constraints.start_at_end_plus_offset) >= 1
        ):
            for t1, t2, lag in special_constraints.start_together:
                index_task_1 = self.problem.index_task[t1]
                index_task_2 = self.problem.index_task[t2]
                model.Add(starts_var[index_task_2] >= ends_var[index_task_1] + lag)

    def retrieve_solutions(
        self, result: cp_model.CpSolver, parameters_cp: ParametersCP
    ) -> ResultStorage:
        starts = [result.Value(s) for s in self.variables["starts"]]
        ends = [result.Value(s) for s in self.variables["ends"]]
        if "b_starts" in self.variables:
            for i in self.variables["b_starts"]:
                for x in range(len(self.variables["b_starts"][i])):
                    if result.Value(self.variables["b_starts"][i][x]) >= 0.5:
                        logger.debug(f"ends {result.Value(self.variables['ends'][i])}")
                        logger.debug(f"bstarts{i, x}")
        if "indicators" in self.variables:
            for key in self.variables["indicators"]:
                logger.debug(
                    f"key : {key, result.Value(self.variables['indicators'][key])}"
                )
        schedule = {
            self.problem.tasks_list[t]: {"start_time": starts[t], "end_time": ends[t]}
            for t in range(len(starts))
        }
        sol = RCPSPSolution(
            problem=self.problem,
            rcpsp_modes=[1] * self.problem.n_jobs_non_dummy,
            rcpsp_schedule=schedule,
        )
        fit = self.aggreg_sol(sol)
        res = ResultStorage(
            list_solution_fits=[(sol, fit)],
            mode_optim=self.params_objective_function.sense_function,
        )
        return res

    def solve(
        self, parameters_cp: Optional[ParametersCP] = None, **kwargs
    ) -> ResultStorage:
        if parameters_cp is None:
            parameters_cp = ParametersCP.default()
        # Search for a feasible solution
        if self.model is None:
            self.init_model(**kwargs)
        add_solution_callback = kwargs.get("add_solution_callback", True)
        if add_solution_callback:
            solution_callback = VarArraySolutionPrinter(
                variables=[
                    self.variables["ends"][
                        self.problem.index_task[self.problem.sink_task]
                    ]
                ]
            )
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
        return self.retrieve_solutions(result=solver, parameters_cp=parameters_cp)
