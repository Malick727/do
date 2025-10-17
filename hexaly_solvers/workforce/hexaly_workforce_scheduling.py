from functools import reduce

import numpy as np
from typing import Optional, Any, List, Dict, Hashable

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import (CategoricalHyperparameter,
EnumHyperparameter)

from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from discrete_optimization.workforce.scheduling.problem import (AllocSchedulingProblem,
                                                                build_calendar_array_from_availability_slot,
                                                                AllocSchedulingSolution)
import hexaly.optimizer
import hexaly.modeler
import hexaly

from discrete_optimization.workforce.scheduling.utils import compute_equivalent_teams_scheduling_problem
from hexaly_solvers.hexaly_solver import HexalySolver
from enum import Enum

class ModelingHexaly(Enum):
    DISJ_LIST = 0
    OPT_INTERVAL = 1


class HexalyAllocSchedulingSolver(HexalySolver):
    problem: AllocSchedulingProblem
    hyperparameters = [EnumHyperparameter(name="modeling", enum=ModelingHexaly,
                                          default=ModelingHexaly.DISJ_LIST),
                       CategoricalHyperparameter(name="symmbreak_on_used",
                                                 choices=[True, False],
                                                 default=False),
                       CategoricalHyperparameter(name="use_custom_phase",
                                                 choices=[True, False],
                                                 default=False),
                       CategoricalHyperparameter(name="run_multistep",
                                                 choices=[True, False],
                                                 default=False),
                       CategoricalHyperparameter(name="use_min_workload_to_guide_solver",
                                                 choices=[True, False],
                                                 default=False)]

    def __init__(self, problem: AllocSchedulingProblem,
                 params_objective_function: Optional[ParamsObjectiveFunction] = None, **kwargs):
        super().__init__(problem, params_objective_function, **kwargs)
        self.ls: Optional[hexaly.optimizer.HexalyOptimizer] = None
        self.model: Optional[hexaly.optimizer.HxModel] = None
        self.vars: Optional[dict[str, hexaly.optimizer.HxExpression]] = {}
        self.modeling: ModelingHexaly = None

    def init_model(self, **kwargs):
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        if kwargs["modeling"] == ModelingHexaly.DISJ_LIST:
            self.init_model_list(**kwargs)
        if kwargs["modeling"] == ModelingHexaly.OPT_INTERVAL:
            self.init_model_opt_intervals(**kwargs)
        self.modeling = kwargs["modeling"]

    def init_model_list(self, **kwargs):
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        ls = hexaly.optimizer.HexalyOptimizer()
        model = ls.model
        nb_tasks = self.problem.number_tasks
        nb_teams = self.problem.number_teams
        allocated_task_per_team = [model.list(nb_tasks) for _ in range(nb_teams)]
        allocated_task_per_team_array = model.array(allocated_task_per_team)
        model.constraint(model.partition(allocated_task_per_team_array))
        allocated_team_per_task = [model.find(allocated_task_per_team_array, j)
                                   for j in range(nb_tasks)]
        allocated_team_per_task_array = model.array(allocated_team_per_task)
        st_lb = [(int(self.problem.get_lb_start_window(t)),
                  int(self.problem.get_ub_start_window(t)),
                  int(self.problem.get_lb_end_window(t)),
                  int(self.problem.get_ub_end_window(t)))
                 for t in self.problem.tasks_list]
        tasks = [model.interval(st_lb[i][0], st_lb[i][3]) for i in range(nb_tasks)]
        tasks_array = model.array(tasks)

        durations = [int(self.problem.tasks_data[self.problem.tasks_list[i]].duration_task)
                     for i in range(nb_tasks)]
        durations_array = model.array(durations)
        for i in range(nb_tasks):
            model.constraint(model.length(tasks[i]) == durations_array[i])
        for t in self.problem.start_window:
            i = self.problem.tasks_to_index[t]
            if self.problem.start_window[t][0] is not None:
                model.constraint(model.start(tasks[i]) >= int(self.problem.start_window[t][0]))
            if self.problem.start_window[t][1] is not None:
                model.constraint(model.start(tasks[i]) <= int(self.problem.start_window[t][1]))
            if self.problem.end_window[t][0] is not None:
                model.constraint(model.end(tasks[i]) >= int(self.problem.end_window[t][0]))
            if self.problem.end_window[t][1] is not None:
                model.constraint(model.end(tasks[i]) <= int(self.problem.end_window[t][1]))
        unallowed_team_per_tasks = self.problem.get_unavailable_teams_per_activity()
        for task in unallowed_team_per_tasks:
            i = self.problem.tasks_to_index[task]
            for team in unallowed_team_per_tasks[task]:
                index_team = self.problem.teams_to_index[team]
                model.constraint(model.not_(model.contains(allocated_task_per_team[index_team], i)))
                model.constraint(allocated_team_per_task[i] != index_team)

        team_availability = {res: model.array(self.problem.rcpsp.get_resource_availability_array(res))
                             for res in self.problem.calendar_team}
        for i_team in range(nb_teams):
            team = self.problem.team_names[i_team]
            lambda_availability = model.lambda_function(lambda t: model.at(team_availability[team], t))
            sequence = allocated_task_per_team[i_team]
            model.constraint(model.and_(model.range(0, model.count(sequence)-1),
                                        model.lambda_function(lambda j:
                                                              model.min(tasks_array[sequence[j]],
                                                                        lambda_availability) == 1)))
            # durations_array[sequence[j]])))
        # Order of activities for each team.
        for i_team in range(nb_teams):
            sequence = allocated_task_per_team[i_team]
            sequence_lambda = model.lambda_function(
                lambda i: tasks_array[sequence[i]] < tasks_array[sequence[i + 1]])
            model.constraint(model.and_(model.range(0, model.count(sequence) - 1), sequence_lambda))

        # Precedence constraints:
        for t in self.problem.precedence_constraints:
            i_t = self.problem.tasks_to_index[t]
            for t_suc in self.problem.precedence_constraints[t]:
                i_t_suc = self.problem.tasks_to_index[t_suc]
                model.constraint(tasks[i_t_suc] > tasks[i_t])

        for l_t in self.problem.same_allocation:
            list_index = [self.problem.tasks_to_index[t] for t in l_t]
            # list_index_array = model.array(list_index)
            for j in range(len(list_index)-1):
                model.constraint(allocated_team_per_task[list_index[j+1]] == allocated_team_per_task[list_index[j]])
            # model.constraint(model.and_(model.range(len(list_index)-1),
            #                             model.lambda_function(
            #                                 lambda j:
            #                                allocated_team_per_task_array[list_index_array[j+1]] ==
            #                                allocated_team_per_task_array[list_index_array[j]])))

        used = [model.count(allocated_task_per_team[j]) >= 1 for j in range(nb_teams)]
        used_array = model.array(used)
        if kwargs["symmbreak_on_used"]:
            equivalent_ = compute_equivalent_teams_scheduling_problem(self.problem)
            for group in equivalent_:
                gr = model.array(group)
                model.constraint(model.and_(model.range(len(group)-1),
                                            model.lambda_function(lambda i: model.at(used_array, model.at(gr, i+1)) <=
                                                                  model.at(used_array, model.at(gr, i)))))
        minimize_dispersion = kwargs.get("minimize_dispersion", True)
        objs = [model.sum(used)]
        if minimize_dispersion:
            workload_ = [model.sum(allocated_task_per_team[j],
                                   model.lambda_function(lambda k: durations_array[k]))
                         for j in range(nb_teams)]
            workload_array = model.array(workload_)
            max_workload = model.max(workload_array)
            min_workload = model.min([used[j]*workload_[j]+(1-used[j])*10e6 for j in range(nb_teams)])
            # wl = model.array([used[j]*workload_[j]+(1-used[j])*10e6 for j in range(nb_teams)])
            # objs.append(max_workload-min_workload)
            # nb_ac = [model.count(allocated_task_per_team[j]) for j in range(nb_teams)]
            # min_nb_ac = model.min([used[j]*nb_ac[j]+(1-used[j])*self.problem.number_tasks for j in range(nb_teams)])
            objs.append(max_workload-min_workload)
            # objs.append(min_workload)
            # model.minimize(min_workload)
            # model.maximize(wls[1])
            # model.minimize(max_workload-min_workload)
        if kwargs["symmbreak_on_used"]:
            equivalent_ = compute_equivalent_teams_scheduling_problem(self.problem)
            print("Equivalent", equivalent_)
            for group in equivalent_:
                gr = model.array(group)
                model.constraint(model.and_(model.range(len(group) - 1),
                                            model.lambda_function(lambda i: model.at(used_array, model.at(gr, i + 1)) <=
                                                                            model.at(used_array, model.at(gr, i)))))
                # if minimize_dispersion:
                #    model.constraint(model.and_(model.range(len(group) - 1),
                #                                model.lambda_function(
                #                                    lambda i: model.at(workload_array, model.at(gr, i + 1)) <=
                #                                              model.at(workload_array, model.at(gr, i)))))

        # model.maximize(model.max([model.count(allocated_task_per_team[j]) for j in range(nb_teams)]))
        makespan = model.max([model.end(tasks[i]) for i in range(nb_tasks)])
        objs.append(makespan)
        model.minimize(objs[0])
        if kwargs["use_min_workload_to_guide_solver"]:
            model.minimize(min_workload)
        if minimize_dispersion:
            model.minimize(objs[1])
        self.model = model
        if "solution" in kwargs:
            sol: AllocSchedulingSolution = kwargs["solution"]
            for i in range(nb_tasks):
                model.constraint(allocated_team_per_task[i] == int(sol.allocation[i]))
        if minimize_dispersion:
            self.vars["other_obj"] = [max_workload-min_workload]
        model.close()
        self.vars["tasks"] = tasks
        self.vars["allocated_team_per_task"] = allocated_team_per_task
        self.vars["allocated_task_per_team"] = allocated_task_per_team
        self.vars["objs"] = objs
        self.ls = ls

    def init_model_opt_intervals(self, **kwargs):
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        ls = hexaly.optimizer.HexalyOptimizer()
        model = ls.model
        nb_tasks = self.problem.number_tasks
        nb_teams = self.problem.number_teams
        st_lb = [(int(self.problem.get_lb_start_window(t)),
                  int(self.problem.get_ub_start_window(t)),
                  int(self.problem.get_lb_end_window(t)),
                  int(self.problem.get_ub_end_window(t)))
                 for t in self.problem.tasks_list]
        durations = [int(self.problem.tasks_data[self.problem.tasks_list[i]].duration_task)
                     for i in range(nb_tasks)]
        durations_array = model.array(durations)
        tasks_optional = [{} for i in range(nb_tasks)]
        tasks_presence = [{} for i in range(nb_tasks)]
        compatible_teams = self.problem.compatible_teams_index_all_activity()
        unallowed_team_per_tasks: Dict[Hashable, set[Hashable]] = self.problem.get_unavailable_teams_per_activity()

        for i in range(nb_tasks):
            for team_index in compatible_teams[i]:
                tasks_optional[i][team_index] = model.optional_interval(0, self.problem.horizon)
                tasks_presence[i][team_index] = model.presence(tasks_optional[i][team_index])
                model.constraint(model.iif(tasks_presence[i][team_index],
                                           model.eq(model.length(tasks_optional[i][team_index]),
                                                    durations[i]), 1))
            model.constraint(model.sum([tasks_presence[i][t] for t in tasks_presence[i]]) == 1)

        tasks = [model.hull([tasks_optional[i][t] for t in tasks_optional[i]])
                 for i in range(nb_tasks)]
        tasks_array = model.array(tasks)
        for i in range(nb_tasks):
            model.constraint(model.length(tasks[i]) == durations_array[i])
        for t in self.problem.start_window:
            i = self.problem.tasks_to_index[t]
            if self.problem.start_window[t][0] is not None:
                model.constraint(model.start(tasks[i]) >= int(self.problem.start_window[t][0]))
            if self.problem.start_window[t][1] is not None:
                model.constraint(model.start(tasks[i]) <= int(self.problem.start_window[t][1]))
            if self.problem.end_window[t][0] is not None:
                model.constraint(model.end(tasks[i]) >= int(self.problem.end_window[t][0]))
            if self.problem.end_window[t][1] is not None:
                model.constraint(model.end(tasks[i]) <= int(self.problem.end_window[t][1]))

        makespan = model.max([model.end(tasks[i]) for i in range(nb_tasks)])
        for team_index in range(self.problem.number_teams):
            team = self.problem.team_names[team_index]
            calendar = build_calendar_array_from_availability_slot(availability_slots=
                                                                   self.problem.calendar_team[team],
                                                                   horizon=self.problem.horizon,
                                                                   value=1)
            cal_array = model.array(calendar)
            def no_overlap(team_index, time):
                total_activity = model.sum()
                for i in range(len(tasks_optional)):
                    if team_index in tasks_optional[i]:
                        total_activity.add_operand(
                            model.contains(tasks_optional[i][team_index], time)
                        )
                return model.leq(total_activity, model.at(cal_array, time))
            no_overlap_lmbda = model.lambda_function(lambda time: no_overlap(team_index, time))
            model.constraint(model.and_(model.range(makespan), no_overlap_lmbda))

        # Precedence constraints:
        for t in self.problem.precedence_constraints:
            i_t = self.problem.tasks_to_index[t]
            for t_suc in self.problem.precedence_constraints[t]:
                i_t_suc = self.problem.tasks_to_index[t_suc]
                model.constraint(tasks[i_t_suc] > tasks[i_t])

        for l_t in self.problem.same_allocation:
            list_index = [self.problem.tasks_to_index[t] for t in l_t]
            common_teams = reduce(lambda x, y: x.intersection(y),
                                  [compatible_teams[l] for l in list_index],
                                  set(self.problem.index_to_team.keys()))
            # list_index_array = model.array(list_index)
            model.constraint(model.sum([tasks_presence[list_index[0]][c]
                                        for c in common_teams])==1)
            for j in range(len(list_index)-1):
                for c in common_teams:
                    model.constraint(tasks_presence[list_index[j+1]][c]==
                                     tasks_presence[list_index[j]][c])
        used = [model.bool() for _ in range(nb_teams)]
        for i in range(nb_tasks):
            for t in tasks_presence[i]:
                model.constraint(used[t] >= tasks_presence[i][t])
        used_array = model.array(used)
        if kwargs["symmbreak_on_used"]:
            equivalent_ = compute_equivalent_teams_scheduling_problem(self.problem)
            for group in equivalent_:
                gr = model.array(group)
                model.constraint(model.and_(model.range(len(group)-1),
                                            model.lambda_function(lambda i: model.at(used_array, model.at(gr, i+1)) <=
                                                                  model.at(used_array, model.at(gr, i)))))
        minimize_dispersion = kwargs.get("minimize_dispersion", True)
        objs = [model.sum(used)]
        if minimize_dispersion:

            workload_ = [model.sum([tasks_presence[i][j]*durations[i]
                                    for i in range(nb_tasks)
                                    if j in tasks_presence[i]])
                         for j in range(nb_teams)]
            workload_array = model.array(workload_)
            max_workload = model.max(workload_array)
            min_workload = model.min([used[j]*workload_[j]+(1-used[j])*10e6 for j in range(nb_teams)])
            # wl = model.array([used[j]*workload_[j]+(1-used[j])*10e6 for j in range(nb_teams)])
            # objs.append(max_workload-min_workload)
            # nb_ac = [model.count(allocated_task_per_team[j]) for j in range(nb_teams)]
            # min_nb_ac = model.min([used[j]*nb_ac[j]+(1-used[j])*self.problem.number_tasks for j in range(nb_teams)])
            objs.append(max_workload-min_workload)
            # objs.append(min_workload)
            # model.minimize(min_workload)
            # model.maximize(wls[1])
            # model.minimize(max_workload-min_workload)
        if kwargs["symmbreak_on_used"]:
            equivalent_ = compute_equivalent_teams_scheduling_problem(self.problem)
            print("Equivalent", equivalent_)
            for group in equivalent_:
                gr = model.array(group)
                model.constraint(model.and_(model.range(len(group) - 1),
                                            model.lambda_function(lambda i: model.at(used_array, model.at(gr, i + 1)) <=
                                                                            model.at(used_array, model.at(gr, i)))))
                # if minimize_dispersion:
                #    model.constraint(model.and_(model.range(len(group) - 1),
                #                                model.lambda_function(
                #                                    lambda i: model.at(workload_array, model.at(gr, i + 1)) <=
                #                                              model.at(workload_array, model.at(gr, i)))))

        # model.maximize(model.max([model.count(allocated_task_per_team[j]) for j in range(nb_teams)]))
        makespan = model.max([model.end(tasks[i]) for i in range(nb_tasks)])
        objs.append(makespan)
        # model.minimize(makespan)
        # model.minimize(makespan)
        model.minimize(objs[0])
        if minimize_dispersion:
            model.minimize(objs[1])
        if kwargs["use_min_workload_to_guide_solver"]:
            model.minimize(min_workload)
        self.model = model
        if minimize_dispersion:
            self.vars["other_obj"] = [max_workload-min_workload]
        model.close()
        self.vars["tasks"] = tasks
        self.vars["tasks_optional"] = tasks_optional
        self.vars["tasks_presence"] = tasks_presence
        self.vars["objs"] = objs
        self.ls = ls

    def retrieve_solution(self, **args) -> AllocSchedulingSolution:
        if self.modeling == ModelingHexaly.DISJ_LIST:
            schedule = np.zeros((self.problem.number_tasks, 2), dtype=int)
            allocation = np.zeros(self.problem.number_tasks, dtype=int)
            for t in range(self.problem.number_tasks):
                start = int(self.vars["tasks"][t].value.start())
                end = int(self.vars["tasks"][t].value.end())
                schedule[t, 0] = start
                schedule[t, 1] = end
                allocation[t] = self.vars["allocated_team_per_task"][t].value
            if np.sum(allocation == -1) > 0:
                return None
            return AllocSchedulingSolution(problem=self.problem, schedule=schedule, allocation=allocation)
        if self.modeling == ModelingHexaly.OPT_INTERVAL:
            schedule = np.zeros((self.problem.number_tasks, 2), dtype=int)
            allocation = np.zeros(self.problem.number_tasks, dtype=int)
            for t in range(self.problem.number_tasks):
                start = int(self.vars["tasks"][t].value.start())
                end = int(self.vars["tasks"][t].value.end())
                schedule[t, 0] = start
                schedule[t, 1] = end
                for team in self.vars["tasks_presence"][t]:
                    if self.vars["tasks_presence"][t][team].value == 1:
                        allocation[t] = team
            if np.sum(allocation == -1) > 0:
                return None
            return AllocSchedulingSolution(problem=self.problem, schedule=schedule,
                                           allocation=allocation)

    def solve(self, callbacks: Optional[List[Callback]] = None,
              time_limit: Optional[float] = 100.0,
              **kwargs: Any) -> ResultStorage:
        kwargs = self.complete_with_default_hyperparameters(kwargs)
        if self.ls is None:
            self.init_model(**kwargs)
        if kwargs["run_multistep"]:
            return self.solve_multistep(callbacks=callbacks,
                                        time_limit=time_limit,
                                        **kwargs)
        else:
            return super().solve(callbacks=callbacks, time_limit=time_limit,
                                 **kwargs)

    def solve_multistep(self, callbacks: Optional[List[Callback]] = None,
                        time_limit: float = 100.0,
                        **kwargs):
        res = super().solve(callbacks=callbacks,
                            time_limit=time_limit,
                            **kwargs)
        # Solve the problem with specific multi-objective objectives that help to reduce #Number of teams :D #
        self.model.open()
        for i in range(self.model.nb_objectives):
            self.model.remove_objective(0)  # unnatural thing but anyway.
        self.model.minimize(self.vars["objs"][0])
        # Reintroducing the fairness objective !
        self.model.minimize(self.vars["other_obj"][0])
        self.model.minimize(self.vars["objs"][-1])
        self.model.close()
        res_2 = super().solve(callbacks=callbacks,
                              time_limit=time_limit,
                              **kwargs)
        return ResultStorage(list_solution_fits=res.list_solution_fits+res_2.list_solution_fits,
                             mode_optim=self.params_objective_function.sense_function)
