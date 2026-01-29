from collections import defaultdict
import numpy as np
from docplex.cp.expression import CpoIntervalVar, CpoIntVar
from docplex.cp.modeler import if_then, min_of
from functools import reduce
from docplex.cp.solution import CpoSolveResult, CpoIntervalVarSolution
from typing import Any, Optional, Iterable
from discrete_optimization.generic_tools.do_problem import Solution, ParamsObjectiveFunction
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from discrete_optimization.workforce.scheduling.problem import AllocSchedulingSolution, AllocSchedulingProblem
from cpopt_solvers.cpopt_solver import CPOSolver, CpoModel
from docplex.cp.model import (CpoModel, interval_var, integer_var, alternative, sum_of,
                              max_of, pulse, end_before_start, minimize,
                              end_of, no_overlap, presence_of)
from discrete_optimization.workforce.scheduling.solvers import ObjectivesEnum
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
import logging
logger = logging.getLogger(__name__)


class CPOptAllocScheduling(CPOSolver, WarmstartMixin):
    problem: AllocSchedulingProblem

    def __init__(self, problem: AllocSchedulingProblem,
                 params_objective_function: Optional[ParamsObjectiveFunction] = None, **kwargs):
        super().__init__(problem=problem, params_objective_function=params_objective_function, **kwargs)
        self.vars = {}

    def init_model(self, objectives: list[ObjectivesEnum], **args: Any) -> None:
        model = CpoModel(name="workforce-scheduling")
        self.cp_model = model
        self.vars = {}
        st_lb = [
            (
                int(self.problem.get_lb_start_window(t)),
                int(self.problem.get_ub_start_window(t)),
                int(self.problem.get_lb_end_window(t)),
                int(self.problem.get_ub_end_window(t)),
            )
            for t in self.problem.tasks_list
        ]
        dur = [
            self.problem.tasks_data[t].duration_task for t in self.problem.tasks_list
        ]
        tasks = [interval_var(start=(st_lb[i][0], st_lb[i][1]),
                              end=(st_lb[i][2], st_lb[i][3]),
                              size=dur[i],
                              optional=False)
                 for i in range(self.problem.number_tasks)]
        comp = self.problem.compatible_teams_index_all_activity()
        tasks_optional = [{} for i in range(self.problem.number_tasks)]
        key_per_team = defaultdict(lambda: set())
        for i in range(self.problem.number_tasks):
            for t in comp[i]:
                tasks_optional[i][t] = interval_var(start=(st_lb[i][0], st_lb[i][1]),
                                                    end=(st_lb[i][2], st_lb[i][3]),
                                                    size=dur[i], optional=True)
                key_per_team[t].add((i, t))
            model.add(alternative(tasks[i],
                                  tasks_optional[i].values()))
        for t in self.problem.precedence_constraints:
            i_t = self.problem.tasks_to_index[t]
            for t_suc in self.problem.precedence_constraints[t]:
                i_t_suc = self.problem.tasks_to_index[t_suc]
                model.add(end_before_start(tasks[i_t], tasks[i_t_suc]))

        # Same allocation constraints
        for l_t in self.problem.same_allocation:
            indexes = [self.problem.tasks_to_index[tt] for tt in l_t]
            common_teams = reduce(
                lambda x, y: x.intersection(set(tasks_optional[y].keys())),
                indexes,
                set(self.problem.index_to_team),
            )
            model.add(sum([presence_of(tasks_optional[indexes[0]][c]) for c in common_teams]) == 1)
            for c in common_teams:
                for i in range(len(indexes)-1):
                    model.add(presence_of(tasks_optional[indexes[i]][c]) ==
                              presence_of(tasks_optional[indexes[i+1]][c]))

        # Calendar constraints
        for index_team in range(self.problem.number_teams):
            unavailable = self.problem.compute_unavailability_calendar(
                self.problem.index_to_team[index_team]
            )
            fake_tasks_unavailable = [interval_var(start=x[0], end=x[1])
                                      for x in unavailable]
            tasks_team = [tasks_optional[i][index_team] for i in range(self.problem.number_tasks)
                          if index_team in tasks_optional[i]]
            if len(fake_tasks_unavailable) + len(tasks_team) > 0:
                model.add(no_overlap(tasks_team + fake_tasks_unavailable))
        used = None
        obj_dict = {}
        if ObjectivesEnum.MAKESPAN in objectives:
            makespan = self.create_makespan_obj(intervals=tasks,
                                                st_lb=st_lb)
            obj_dict[ObjectivesEnum.MAKESPAN] = makespan
            self.cp_model.add_kpi(makespan, ObjectivesEnum.MAKESPAN.name)

        if (ObjectivesEnum.NB_TEAMS in objectives
                or ObjectivesEnum.DISPERSION in objectives or ObjectivesEnum.MIN_WORKLOAD):
            used = self.create_used_variable(intervals=tasks_optional,
                                             key_per_team=key_per_team)
            self.vars["used"] = used
            obj_dict[ObjectivesEnum.NB_TEAMS] = sum_of([used[x] for x in used])
            self.cp_model.add_kpi(obj_dict[ObjectivesEnum.NB_TEAMS], ObjectivesEnum.NB_TEAMS.name)
        if ObjectivesEnum.DISPERSION in objectives or ObjectivesEnum.MIN_WORKLOAD in objectives:
            max_workload, min_workload, workload, workload_nz = (
                self.create_workload_variables(dur=dur, opt_interval_var=tasks_optional,
                                               key_per_team=key_per_team,
                                               used=used))
            self.vars["workload"] = workload
            self.vars["workload_nz"] = workload_nz
            obj_dict[ObjectivesEnum.MIN_WORKLOAD] = min_workload
            obj_dict[ObjectivesEnum.DISPERSION] = max_workload-min_workload
            self.cp_model.add_kpi(obj_dict[ObjectivesEnum.MIN_WORKLOAD], ObjectivesEnum.MIN_WORKLOAD.name)
            self.cp_model.add_kpi(obj_dict[ObjectivesEnum.DISPERSION], ObjectivesEnum.DISPERSION.name)




        objs = []
        weights = []
        for obj in objectives:
            if obj == ObjectivesEnum.NB_TEAMS:
                sum_used = sum_of([used[x] for x in used])
                objs.append(sum_used)
                weights.append(10000)
                obj_dict[obj] = sum_used
            if obj == ObjectivesEnum.MIN_WORKLOAD:
                objs.append(min_workload)
                obj_dict[obj] = min_workload
                weights.append(1.)
            if obj == ObjectivesEnum.DISPERSION:
                objs.append(max_workload - min_workload)
                obj_dict[obj] = max_workload-min_workload
                weights.append(1.)
            if obj == ObjectivesEnum.MAKESPAN:
                self.cp_model.add_kpi(makespan, "makespan")
                objs.append(makespan)
                obj_dict[obj] = makespan
                weights.append(1.)
        self.cp_model.add(minimize(sum(weights[i]*objs[i] for i in range(len(objs)))))
        self.vars["opt_intervals_var"] = tasks_optional
        self.vars["intervals_var"] = tasks
        self.vars["objs"] = objs
        self.vars["obj_dict"] = obj_dict

    def create_makespan_obj(self, intervals: dict[int, Any],
                            st_lb: list[tuple[int, int, int, int]] = None):
        if st_lb is None:
            st_lb = [(int(self.problem.get_lb_start_window(t)),
                      int(self.problem.get_ub_start_window(t)),
                      int(self.problem.get_lb_end_window(t)),
                      int(self.problem.get_ub_end_window(t)))
                     for t in self.problem.tasks_list]
        lb_makespan = max([x[2] for x in st_lb])
        ub_makespan = max([x[3] for x in st_lb])
        makespan = integer_var(min=lb_makespan, max=ub_makespan, name="makespan")
        self.cp_model.add(makespan == max_of([end_of(intervals[i]) for i in intervals]))
        return makespan

    def create_used_variable(self, intervals: dict[int, dict[int, Any]],
                             key_per_team: dict[int, list[tuple[int, int]]]):
        used = {index_team: integer_var(min=0, max=1, name=f"used_{index_team}")
                for index_team in key_per_team}
        for index_team in used:
            for x in key_per_team[index_team]:
                self.cp_model.add(used[index_team] >= presence_of(intervals[x[0]][x[1]]))
            self.cp_model.add(used[index_team] ==
                              max_of([presence_of(intervals[x[0]][x[1]])
                                      for x in key_per_team[index_team]]))
        return used

    def create_workload_variables(self, dur: list[int],
                                  opt_interval_var: dict[int, dict[int, CpoIntervalVar]],
                                  key_per_team: dict[int, list[tuple[int, int]]],
                                  used: dict[int, CpoIntVar]):
        upper_bound_workload = int(sum(dur))
        workload = {index_team: integer_var(min=0, max=upper_bound_workload,
                                            name=f"workload_{index_team}")
                    for index_team in key_per_team}
        workload_non_zero = {index_team: integer_var(
            min=0,
            max=upper_bound_workload,
            name=f"nzworkload_{index_team}",
        ) for index_team in key_per_team}
        for index_team in workload:
            self.cp_model.add(sum_of([dur[x[0]] * presence_of(opt_interval_var[x[0]][x[1]])
                                      for x in key_per_team[index_team]]) == workload[index_team])
            self.cp_model.add(if_then(used[index_team] == 1,
                                      workload_non_zero[index_team] == workload[index_team]))
            self.cp_model.add(if_then(used[index_team] == 0,
                                      workload_non_zero[index_team] == upper_bound_workload))
        max_workload = integer_var(
            min=0, max=upper_bound_workload, name=f"max_workload"
        )
        self.cp_model.add(max_workload == max_of([workload[x] for x in workload]))
        min_workload = integer_var(min=0, max=upper_bound_workload, name=f"min_workload")
        self.cp_model.add(min_workload == min_of([workload_non_zero[x] for x in workload_non_zero]))
        return max_workload, min_workload, workload, workload_non_zero

    def set_warm_start(self, solution: Solution) -> None:
        solution = self.cp_model.create_empty_solution()
        for i in self.vars["opt_intervals_var"]:
            solution.add_interval_var_solution(self.vars["intervals_var"][i],
                                               start=int(solution.schedule[i, 0]),
                                               end=int(solution.schedule[i, 1]))
            for j in self.vars["opt_intervals_var"][i]:
                solution.add_interval_var_solution(self.vars["opt_intervals_var"][i][j],
                                                   start=int(solution.schedule[i, 0]),
                                                   end=int(solution.schedule[i, 1]),
                                                   presence=1 if solution.allocation[i] == j else 0)
        teams_used = set()
        dur_per_team = {}
        upper_bound_workload = 0
        for i in range(self.problem.number_tasks):
            team = solution.allocation[i]
            teams_used.add(team)
            if team not in dur_per_team:
                dur_per_team[team] = 0
            dur_per_team[team] += self.problem.tasks_data[self.problem.index_to_task[i]].duration_task
            upper_bound_workload += self.problem.tasks_data[self.problem.index_to_task[i]].duration_task
        for i in self.vars["used"]:
            solution.add_integer_var_solution(self.vars["used"][i],
                                              value=1 if i in teams_used else 0)
            solution.add_integer_var_solution(self.vars["workload"][i],
                                              value=dur_per_team[i] if i in teams_used else 0)
            solution.add_integer_var_solution(self.vars["workload_nz"][i],
                                              value=dur_per_team[i] if i in teams_used else upper_bound_workload)
        return solution

    def retrieve_solution(self, sres: CpoSolveResult) -> Solution:
        schedule = np.zeros((self.problem.number_tasks, 2), dtype=int)
        allocation = np.zeros(self.problem.number_tasks, dtype=int)
        for i in range(self.problem.number_tasks):
            for j in self.vars["opt_intervals_var"][i]:
                res: CpoIntervalVarSolution = sres.get_var_solution(self.vars["opt_intervals_var"][i][j])
                if res.is_present():
                    allocation[i] = j
                    schedule[i, 0] = res.get_start()
                    schedule[i, 1] = res.get_end()
        logger.info(f"Kpis = {sres.get_kpis()}")
        sol = AllocSchedulingSolution(problem=self.problem,
                                      schedule=schedule, allocation=allocation)
        sol._intern_obj = {}
        for obj in self.vars["obj_dict"]:
            sol._intern_obj[obj] = sres.get_kpis()[obj.name]
        return sol

    def set_lexico_objective(self, obj: ObjectivesEnum) -> None:
        self.cp_model.remove(self.cp_model.get_objective_expression())
        self.cp_model.add(minimize(self.vars["obj_dict"][obj]))

    def implements_lexico_api(self) -> bool:
        return True

    def get_lexico_objectives_available(self) -> list[str]:
        return list(self.vars["obj_dict"].keys())

    def get_lexico_objective_value(self, obj: ObjectivesEnum, res: ResultStorage) -> float:
        sol = res[-1][0]
        return sol._intern_obj[obj]

    def add_lexico_constraint(self, obj: str, value: float) -> Iterable[Any]:
        c = self.cp_model.add(self.vars["obj_dict"][obj] <= value)
        return [c]



