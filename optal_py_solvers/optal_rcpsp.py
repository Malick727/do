from typing import Any, Optional

import optalcp as cp

from discrete_optimization.generic_tools.do_problem import Solution, ParamsObjectiveFunction
from discrete_optimization.generic_tools.ea.ga_tools import ParametersAltGa
from discrete_optimization.rcpsp.problem import RcpspProblem, RcpspSolution
from discrete_optimization.rcpsp.solvers import RcpspSolver
from discrete_optimization.rcpsp.utils import create_fake_tasks
from optal_py_solvers.generic_optalcp_solver import OptalCpSolver


class OptalRcpspSolver(OptalCpSolver, RcpspSolver):

    def __init__(self, problem: RcpspProblem, params_objective_function: Optional[ParamsObjectiveFunction] = None,
                 **kwargs) -> None:
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables_dict = {}

    def create_fake_tasks(self) -> list[dict[str, int]]:
        """
        Create tasks representing the variable resource availability.
        :return:
        """
        if self.problem.is_calendar:
            fake_task: list[dict[str, int]] = create_fake_tasks(
                rcpsp_problem=self.problem
            )
        else:
            fake_task = []
        return fake_task

    def init_model(self, **args: Any) -> None:
        self.cp_model = cp.Model()
        self.create_interval_vars()
        self.create_precedence_constraints()
        self.create_cumulative_constraints()
        self.cp_model.minimize(self.cp_model.end(self.variables_dict["intervals"][self.problem.sink_task]))

    def create_interval_vars(self):
        intervals = {}
        opt_intervals = {}
        for t in self.problem.tasks_list:
            modes = self.problem.mode_details[t]
            modes_keys = list(modes.keys())
            length = None
            if len(modes_keys) == 1:
                length = self.problem.mode_details[t][modes_keys[0]]["duration"]
            intervals[t] = self.cp_model.interval_var(start=(0, self.problem.horizon),
                                                      end=(0, self.problem.horizon),
                                                      length=length,
                                                      optional=False,
                                                      name=f"itv_{t}")
            opt_intervals[t] = {}
            if len(modes_keys) == 1:
                opt_intervals[t][modes_keys[0]] = intervals[t] # Useless
            else:
                for m in modes_keys:
                    length = self.problem.mode_details[t][m]["duration"]
                    opt_intervals[t][m] = self.cp_model.interval_var(start=(0, self.problem.horizon),
                                                                     end=(0, self.problem.horizon),
                                                                     length=length,
                                                                     optional=False,
                                                                     name=f"itv_{t}_{m}")
                self.cp_model.alternative(intervals[t], [opt_intervals[t][m] for m in opt_intervals[t]])
        self.variables_dict["intervals"] = intervals
        self.variables_dict["opt_intervals"] = opt_intervals

    def create_precedence_constraints(self):
        for t in self.problem.successors:
            for succ in self.problem.successors[t]:
                self.cp_model.end_before_start(self.variables_dict["intervals"][t],
                                               self.variables_dict["intervals"][succ])
    def create_cumulative_constraints(self):
        calendar_tasks = self.create_fake_tasks()
        for res in self.problem.resources_list:
            capa = self.problem.get_max_resource_capacity(res)
            # The calendar virtual intervals.
            list_pulse = [self.cp_model.pulse(interval=self.cp_model.interval_var(start=x["start"],
                                                                                  end=x["start"] + x["duration"],
                                                                                  optional=False),
                                              height=x[res])
                          for x in calendar_tasks if x.get(res, 0) > 0]
            for t in self.variables_dict["opt_intervals"]:
                for m in self.variables_dict["opt_intervals"][t]:
                    conso = self.problem.mode_details[t][m].get(res, 0)
                    if conso > 0:
                        self.cp_model.step_at()
                        list_pulse.append(self.cp_model.pulse(interval=self.variables_dict["opt_intervals"][t][m],
                                                              height=conso))
            self.cp_model.enforce(self.cp_model.sum(list_pulse) <= capa)


    def retrieve_solution(self, result: cp.SolveResult) -> Solution:
        schedule = {}
        modes_dict = {}
        if result.solution is not None:
            for t in self.variables_dict["intervals"]:
                start = result.solution.get_start(self.variables_dict["intervals"][t])
                end = result.solution.get_end(self.variables_dict["intervals"][t])
                schedule[t] = {"start_time": start, "end_time": end}
                for m in self.variables_dict["opt_intervals"][t]:
                    if result.solution.is_present(self.variables_dict["opt_intervals"][t][m]):
                        modes_dict[t] = m
            return RcpspSolution(problem=self.problem,
                                 rcpsp_schedule=schedule,
                                 rcpsp_modes=[modes_dict[t] for t in self.problem.tasks_list_non_dummy])
        return None
