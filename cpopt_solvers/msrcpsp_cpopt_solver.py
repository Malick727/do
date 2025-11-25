from typing import Any, Optional

from docplex.cp.model import (CpoModel, CpoSolveResult, alternative,
                              synchronize,
                              end_before_start,
                              end_of, length_of,
                              size_of,
                              interval_var,
                              presence_of, minimize, pulse)
from docplex.cp.solution import CpoIntervalVarSolution

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction, Solution)
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.rcpsp_multiskill.problem import (
    MultiskillRcpspProblem, MultiskillRcpspSolution)
from discrete_optimization.rcpsp_multiskill.problem import create_fake_tasks_multiskills
from cpopt_solvers.cpopt_solver import CPOSolver


class CPOptMultiSkillRCPSPSolver(CPOSolver, SolverDO):
    """
    CP Optimizer solver for the Multi-Skill Resource-Constrained Project Scheduling Problem (MSRCPSP).

    This solver uses docplex.cp.model to build a constraint programming model.
    It handles:
    - Multi-mode tasks
    - Renewable resource constraints
    - Non-renewable resource constraints
    - Employee skill requirements
    - Disjunctive constraints for employees (one task at a time)

    """
    problem: MultiskillRcpspProblem

    def __init__(self, problem: MultiskillRcpspProblem,
                 params_objective_function: Optional[ParamsObjectiveFunction] = None,
                 **kwargs):
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = None

    def retrieve_solution(self, sres: CpoSolveResult) -> Solution:
        schedule = {}
        modes = {}
        employee_usage = {t: {} for t in self.problem.tasks_list}

        # Retrieve schedule and modes
        for t in self.problem.tasks_list:
            # Get main task interval solution
            itv_sol: CpoIntervalVarSolution = sres.get_var_solution(self.variables["tasks_intervals"][t])
            schedule[t] = {"start_time": itv_sol.get_start(), "end_time": itv_sol.get_end()}

            # Find which mode was chosen
            for m in self.problem.mode_details[t]:
                mode_itv_sol: CpoIntervalVarSolution = sres.get_var_solution(
                    self.variables["task_modes_intervals"][t][m])
                if mode_itv_sol.is_present():
                    modes[t] = m
                    break

        # Retrieve employee assignments
        for emp in self.problem.employees:
            for t in self.problem.tasks_list:
                # Check if this employee can even work on this task
                if t not in self.variables["worker_task_intervals"][emp]:
                    continue

                worker_itv_sol: CpoIntervalVarSolution = sres.get_var_solution(
                    self.variables["worker_task_intervals"][emp][t])
                if worker_itv_sol.is_present():
                    # Infer used skills
                    task_mode = modes[t]
                    required_skills = {s for s, v in self.problem.mode_details[t][task_mode].items() if
                                       s in self.problem.skills_set and v > 0}
                    employee_skills = set(self.problem.employees[emp].get_non_zero_skills())
                    used_skills = required_skills.intersection(employee_skills)
                    if used_skills:
                        employee_usage[t][emp] = used_skills

        return MultiskillRcpspSolution(
            problem=self.problem,
            modes=modes,
            schedule=schedule,
            employee_usage=employee_usage
        )

    def init_model(self, **args: Any) -> None:
        self.cp_model = CpoModel(name="MultiSkillRCPSP")
        # --- Create main task interval variables using 'alternative' ---
        # This ensures exactly one mode is chosen for each task.
        tasks_intervals = {
            t: interval_var(start=(0, self.problem.horizon),
                            end=(0, self.problem.horizon),  # TODO Put better bounds
                            name=f"interval_{t}",
                            optional=False)
            for t in self.problem.tasks_list
        }
        task_modes_intervals = {}
        for t in self.problem.tasks_list:
            task_modes_intervals[t] = {}
            modes = list(self.problem.mode_details[t].keys())
            nb_mode = len(modes)
            if nb_mode == 1:
                task_modes_intervals[t][modes[0]] = tasks_intervals[t]
                duration = self.problem.mode_details[t][modes[0]]["duration"]
                self.cp_model.add(length_of(tasks_intervals[t])==duration)
            else:
                for m in modes:
                    task_modes_intervals[t][m] = interval_var(start=(0, self.problem.horizon),
                                                              end=(0, self.problem.horizon),
                                                              size=self.problem.mode_details[t][m]["duration"],
                                                              name=f"interval_{t}_{m}",
                                                              optional=True)
                self.cp_model.add(alternative(interval=tasks_intervals[t],
                                              array=[tasks_intervals[t][m] for m in tasks_intervals[t]],
                                              cardinality=1))

        # --- Precedence constraints ---
        for s in self.problem.successors:
            for succ in self.problem.successors[s]:
                self.cp_model.add(end_before_start(tasks_intervals[s], tasks_intervals[succ]))
            self.cp_model.add(end_before_start(tasks_intervals[s], tasks_intervals[self.problem.sink_task]))

        # --- Renewable resource constraints ---
        for r in self.problem.resources_list:
            if r in self.problem.non_renewable_resources:
                continue
            pulses = []
            for t in self.problem.tasks_list:
                for m in self.problem.mode_details[t]:
                    if self.problem.mode_details[t][m].get(r, 0) > 0:
                        pulses.append(
                            pulse(task_modes_intervals[t][m], self.problem.mode_details[t][m][r])
                        )
            self.cp_model.add(sum(pulses) <= self.problem.get_max_resource_capacity(r))

        # --- Non-renewable resource constraints ---
        for r in self.problem.non_renewable_resources:
            consumption = sum(
                presence_of(task_modes_intervals[t][m]) * self.problem.mode_details[t][m].get(r, 0)
                for t in self.problem.tasks_list
                for m in self.problem.mode_details[t]
            )
            self.cp_model.add(consumption <= self.problem.get_max_resource_capacity(r))

        # --- Employee and Skill Constraints ---
        # Create optional interval variables for each potential (employee, task) assignment
        worker_task_intervals = {emp: {} for emp in self.problem.employees}
        task_to_consider = set()
        for emp in self.problem.employees:
            for t in self.problem.tasks_list:
                # Optimization: only create variable if employee has a skill needed by the task
                emp_skills = set(self.problem.employees[emp].get_non_zero_skills())
                task_skills = {s for m in self.problem.mode_details[t] for s in self.problem.mode_details[t][m] if
                               s in self.problem.skills_set}
                if emp_skills.intersection(task_skills):
                    itv = interval_var(name=f"worker_{emp}_task_{t}", optional=True)
                    worker_task_intervals[emp][t] = itv
                    task_to_consider.add(t)
        for t in task_to_consider:
            self.cp_model.add(synchronize(interval=tasks_intervals[t],
                                          array=[worker_task_intervals[emp][t]
                                                 for emp in worker_task_intervals
                                                 if t in worker_task_intervals[emp]]))

        # --- Each employee can only work on one task at a time ---
        for emp in self.problem.employees:
            self.cp_model.add(
                sum(pulse(itv, 1) for itv in worker_task_intervals[emp].values()) <= 1
            )

        # --- Skill requirements must be met for each task ---
        for t in self.problem.tasks_list:
            skill_of_interest = set([s for s in self.problem.skills_list
                                     if any(m for m in self.problem.mode_details[t]
                                            if self.problem.mode_details[t][m].get(s, 0) > 0)])
            for s in skill_of_interest:
                # Skill provided by all employees assigned to the task
                skill_provided = sum(
                    presence_of(worker_task_intervals[emp][t]) * self.problem.employees[emp].get_skill_level(s)
                    for emp in self.problem.employees
                    if t in worker_task_intervals[emp] and self.problem.employees[emp].get_skill_level(s) > 0
                )
                # Skill required by the task's active mode
                skill_required = sum(
                    presence_of(task_modes_intervals[t][m]) * self.problem.mode_details[t][m].get(s, 0)
                    for m in self.problem.mode_details[t]
                )
                self.cp_model.add(skill_provided >= skill_required)

        # --- Objective function: Minimize makespan ---
        self.cp_model.add(minimize(end_of(tasks_intervals[self.problem.sink_task])))

        # --- Store variables for solution retrieval ---
        self.variables = {
            "tasks_intervals": tasks_intervals,
            "task_modes_intervals": task_modes_intervals,
            "worker_task_intervals": worker_task_intervals
        }