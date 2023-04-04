from docplex.cp.expression import CpoIntVar
from docplex.cp.modeler import max_of, alternative, presence_of, scal_prod, if_then
from docplex.cp.solution import CpoVarSolution
from typing import List, Dict, Hashable, Set
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPSolution
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.cp_tools import CPSolver, CPSolverName, ParametersCP, SignEnum
from discrete_optimization.generic_tools.do_problem import Problem, ParamsObjectiveFunction, \
    build_aggreg_function_and_params_objective
from docplex.cp.model import interval_var, minimize, start_before_end, end_before_start, CpoModel, end_of, \
    pulse, start_of, CpoExpr, CpoStepFunction, INTERVAL_MIN, INTERVAL_MAX, CpoSolveResult, integer_var, sum_of, no_overlap
import docplex.cp.utils_visu as visu
import os


class CPOptSolver(CPSolver):
    def __init__(self, rcpsp_model: RCPSPModel,
                 cp_solver_name: CPSolverName = CPSolverName.CHUFFED,
                 params_objective_function: ParamsObjectiveFunction = None, **kwargs):
        self.rcpsp_model = rcpsp_model
        self.cp_solver_name = cp_solver_name
        self.key_decision_variable = ["s"]  # For now, I've put the var name of the CP model (not the rcpsp_model)
        self.aggreg_sol, self.aggreg_from_dict_values, self.params_objective_function = \
            build_aggreg_function_and_params_objective(self.rcpsp_model,
                                                       params_objective_function=params_objective_function)

    def init_model(self, **args):
        resource_needed: Dict[Hashable, Dict[str, Set[int]]] = args.get("resource_needed", {})
        resource_forbidden: Dict[Hashable, Dict[str, Set[int]]] = args.get("resource_forbidden", {})
        NB_TASKS, NB_RESOURCES = self.rcpsp_model.n_jobs, len(self.rcpsp_model.resources_list)
        CAPACITIES = [self.rcpsp_model.resources[r] for r in self.rcpsp_model.resources_list]
        tasks_list = self.rcpsp_model.tasks_list
        self.index_task = {tasks_list[i]: i for i in range(len(tasks_list))}
        # -----------------------------------------------------------------------------
        # Prepare the data for modeling
        # -----------------------------------------------------------------------------
        # Extract duration of each task
        DURATIONS = [self.rcpsp_model.mode_details[t][1]["duration"] for t in tasks_list]
        # Extract demand of each task
        DEMANDS = [[self.rcpsp_model.mode_details[t][1][r] for r in self.rcpsp_model.resources_list]
                   for t in tasks_list]
        # Extract successors of each task
        SUCCESSORS = [[self.index_task[tt] for tt in self.rcpsp_model.successors[t]]
                      for t in tasks_list]
        # -----------------------------------------------------------------------------
        # Build the model
        # -----------------------------------------------------------------------------

        # Create model
        mdl = CpoModel()
        # Create task interval variables
        resource_needed: Dict[Hashable, Dict[str, Set[int]]] = args.get("resource_needed", {})
        resource_forbidden: Dict[Hashable, Dict[str, Set[int]]] = args.get("resource_forbidden", {})
        dict_resource_allocation = {}
        for r in self.rcpsp_model.resources_list:
            dict_resource_allocation[r] = [[None for task in range(self.rcpsp_model.n_jobs)]
                                           for res_index in range(self.rcpsp_model.get_max_resource_capacity(r))]
            for t in tasks_list:
                index_task = self.index_task[t]
                if self.rcpsp_model.mode_details[t][1].get(r, 0) == 0:
                    for ri in range(len(dict_resource_allocation[r])):
                        dict_resource_allocation[r][ri][index_task] = 0
                    continue
                needed = resource_needed[t][r]
                forbidden = resource_forbidden[t][r]
                if len(needed) > 0:
                    for index in needed:
                        dict_resource_allocation[r][index][index_task] = 1
                    if needed == self.rcpsp_model.mode_details[t][1][r]:
                        for kk in range(self.rcpsp_model.get_max_resource_capacity(r)):
                            if kk not in needed:
                                dict_resource_allocation[r][kk][index_task] = 0
                    else:
                        for kk in range(self.rcpsp_model.get_max_resource_capacity(r)):
                            if kk not in needed and kk not in forbidden:
                                dict_resource_allocation[r][kk][index_task] = integer_var(min=0, max=1)
                if len(forbidden) > 0:
                    for index in forbidden:
                        dict_resource_allocation[r][index][index_task] = 0
                for ri in range(len(dict_resource_allocation[r])):
                    if dict_resource_allocation[r][ri][index_task] is None:
                        dict_resource_allocation[r][ri][index_task] = integer_var(min=0, max=1)

        once_needed = {r: set() for r in self.rcpsp_model.resources_list}
        for t in resource_needed:
            for r in resource_needed[t]:
                once_needed[r].update(resource_needed[t][r])
        not_needed = {r: [j for j in range(self.rcpsp_model.get_max_resource_capacity(r))
                          if j not in once_needed[r]] for r in self.rcpsp_model.resources_list}

        used = {r: [integer_var(min=0, max=1) for k in range(self.rcpsp_model.get_max_resource_capacity(r))]
                for r in self.rcpsp_model.resources_list}
        for r in used:
            for k in range(len(used[r])):
                mdl.add(used[r][k] == max_of(dict_resource_allocation[r][k]))
        for r in not_needed:
            for l in range(len(not_needed[r])-1):
                mdl.add(used[r][not_needed[r][l]] >= used[r][not_needed[r][l+1]])  # fill ressource in a given order.

        tasks = [interval_var(name='T{}'.format(i + 1), size=DURATIONS[i]) for i in range(NB_TASKS)]
        for r in dict_resource_allocation:
            for k in range(self.rcpsp_model.get_max_resource_capacity(r)):
                mdl.add(sum(dict_resource_allocation[r][k][t]*pulse(tasks[t], 1) for t in range(NB_TASKS)) <= 1)
                # Cumulative for resource index.

        # Add precedence constraints
        mdl.add(end_before_start(tasks[t], tasks[s]) for t in range(NB_TASKS) for s in SUCCESSORS[t])
        # Constrain capacity of resources
        mdl.add(sum(pulse(tasks[t], DEMANDS[t][r]) for t in range(NB_TASKS) if DEMANDS[t][r] > 0) <= CAPACITIES[r]
                for r in range(NB_RESOURCES))
        # Minimize end of all tasks
        self.objective_expression = minimize(sum_of([used[r][k]
                                                     for r in dict_resource_allocation
                                                     for k in range(len(dict_resource_allocation[r]))]))
        mdl.add(self.objective_expression)
        self.model = mdl
        self.variables = {"tasks": tasks, "resource_allocation": dict_resource_allocation}

    def init_model_with_alternative(self, **args):
        resource_needed: Dict[Hashable, Dict[str, Set[int]]] = args.get("resource_needed", {})
        resource_forbidden: Dict[Hashable, Dict[str, Set[int]]] = args.get("resource_forbidden", {})
        NB_TASKS, NB_RESOURCES = self.rcpsp_model.n_jobs, len(self.rcpsp_model.resources_list)
        CAPACITIES = [self.rcpsp_model.resources[r] for r in self.rcpsp_model.resources_list]
        tasks_list = self.rcpsp_model.tasks_list
        self.index_task = {tasks_list[i]: i for i in range(len(tasks_list))}
        # -----------------------------------------------------------------------------
        # Prepare the data for modeling
        # -----------------------------------------------------------------------------
        # Extract duration of each task
        DURATIONS = [self.rcpsp_model.mode_details[t][1]["duration"] for t in tasks_list]
        # Extract demand of each task
        DEMANDS = [[self.rcpsp_model.mode_details[t][1][r] for r in self.rcpsp_model.resources_list]
                   for t in tasks_list]
        # Extract successors of each task
        SUCCESSORS = [[self.index_task[tt] for tt in self.rcpsp_model.successors[t]]
                      for t in tasks_list]
        # -----------------------------------------------------------------------------
        # Build the model
        # -----------------------------------------------------------------------------

        # Create model
        mdl = CpoModel()
        # Create task interval variables
        resource_needed: Dict[Hashable, Dict[str, Set[int]]] = args.get("resource_needed", {})
        resource_forbidden: Dict[Hashable, Dict[str, Set[int]]] = args.get("resource_forbidden", {})
        dict_resource_allocation = {}
        for r in self.rcpsp_model.resources_list:
            dict_resource_allocation[r] = [[None for task in range(self.rcpsp_model.n_jobs)]
                                           for res_index in range(self.rcpsp_model.get_max_resource_capacity(r))]
            for t in tasks_list:
                index_task = self.index_task[t]
                if self.rcpsp_model.mode_details[t][1].get(r, 0) == 0:
                    for ri in range(len(dict_resource_allocation[r])):
                        dict_resource_allocation[r][ri][index_task] = 0
                    continue
                needed = resource_needed[t][r]
                forbidden = resource_forbidden[t][r]
                if len(needed) > 0:
                    for index in needed:
                        dict_resource_allocation[r][index][index_task] = 1
                    if needed == self.rcpsp_model.mode_details[t][1][r]:
                        for kk in range(self.rcpsp_model.get_max_resource_capacity(r)):
                            if kk not in needed:
                                dict_resource_allocation[r][kk][index_task] = 0
                    else:
                        for kk in range(self.rcpsp_model.get_max_resource_capacity(r)):
                            if kk not in needed and kk not in forbidden:
                                dict_resource_allocation[r][kk][index_task] = integer_var(min=0, max=1)
                if len(forbidden) > 0:
                    for index in forbidden:
                        dict_resource_allocation[r][index][index_task] = 0
                for ri in range(len(dict_resource_allocation[r])):
                    if dict_resource_allocation[r][ri][index_task] is None:
                        dict_resource_allocation[r][ri][index_task] = integer_var(min=0, max=1)

        def create_var(val, duration):
            try:
                if val == 1:
                    return interval_var(size=duration, optional=False)
                if val == 0:
                    return None
            except:
                return interval_var(size=duration, optional=True)

        alternative_tasks = {r: [[create_var(val=dict_resource_allocation[r][rindex][i],
                                             duration=DURATIONS[i])
                                  for i in range(NB_TASKS)]
                                 for rindex in range(self.rcpsp_model.get_max_resource_capacity(r))]
                             for r in dict_resource_allocation}
        tasks = [interval_var(name='T{}'.format(i + 1), size=DURATIONS[i], end=(0, self.rcpsp_model.horizon))
                 for i in range(NB_TASKS)]
        for task in tasks_list:
            for r in alternative_tasks:
                number_needed = self.rcpsp_model.mode_details[task][1].get(r, 0)
                if number_needed > 0:
                    mdl.add(alternative(tasks[self.index_task[task]],
                                       [alternative_tasks[r][ri][self.index_task[task]]
                                        for ri in range(len(alternative_tasks[r]))
                                        if alternative_tasks[r][ri][self.index_task[task]] is not None],
                                        cardinality=number_needed))
        for r in alternative_tasks:
            for ri in range(len(alternative_tasks[r])):
                for t in range(len(alternative_tasks[r][ri])):
                    if dict_resource_allocation[r][ri][t] not in {0, 1}:
                        mdl.add(presence_of(alternative_tasks[r][ri][t]) == dict_resource_allocation[r][ri][t])

        once_needed = {r: set() for r in self.rcpsp_model.resources_list}
        for t in resource_needed:
            for r in resource_needed[t]:
                once_needed[r].update(resource_needed[t][r])
        not_needed = {r: [j for j in range(self.rcpsp_model.get_max_resource_capacity(r))
                          if j not in once_needed[r]] for r in self.rcpsp_model.resources_list}

        used = {r: [integer_var(min=0, max=1) for k in range(self.rcpsp_model.get_max_resource_capacity(r))]
                for r in self.rcpsp_model.resources_list}
        for r in used:
            for k in range(len(used[r])):
                mdl.add(used[r][k] == max_of(dict_resource_allocation[r][k]))

        resource_allocation_break = args.get("resource_allocation_break", True)
        if resource_allocation_break:
            for r in not_needed:
                for l in range(len(not_needed[r])-1):
                    mdl.add(used[r][not_needed[r][l]] >= used[r][not_needed[r][l+1]])  # fill ressource in a given order.
                    mdl.add(if_then(used[r][not_needed[r][l+1]] == 1, used[r][not_needed[r][l]] == 1))  # fill ressource in a given order.

        include_instantaneous_resource_consumption = args.get("include_instantaneous_resource_consumption", True)
        if include_instantaneous_resource_consumption:
            max_res = {r: max([self.rcpsp_model.mode_details[t][1].get(r, 0)
                               for t in self.rcpsp_model.mode_details]) for r in self.rcpsp_model.resources_list}
            max_res_consumption = {r: integer_var(min=max_res[r], max=self.rcpsp_model.get_max_resource_capacity(r))
                                   for r in max_res}
            for r in max_res_consumption:
                mdl.add(max_res_consumption[r] <= sum_of(used[r]))
            mdl.add(sum(pulse(tasks[t], DEMANDS[t][r]) for t in range(NB_TASKS)
                        if DEMANDS[t][r] > 0) <= max_res_consumption[self.rcpsp_model.resources_list[r]]
                    for r in range(NB_RESOURCES))
        # for r in dict_resource_allocation:
        #    for k in range(self.rcpsp_model.get_max_resource_capacity(r)):
        #        mdl.add(sum(pulse(tasks[t], dict_resource_allocation[r][k][t]) for t in range(NB_TASKS)) <= 1)
        # Cumulative for resource index. -> doesnt work...

        # Add precedence constraints
        mdl.add(end_before_start(tasks[t], tasks[s]) for t in range(NB_TASKS) for s in SUCCESSORS[t])
        # Constrain capacity of resources
        #mdl.add(sum(pulse(tasks[t], DEMANDS[t][r]) for t in range(NB_TASKS) if DEMANDS[t][r] > 0) <= CAPACITIES[r]
        #        for r in range(NB_RESOURCES))

        for r in alternative_tasks:
            for ri in range(len(alternative_tasks[r])):
                mdl.add(sum(pulse(alternative_tasks[r][ri][t], 1)
                            for t in range(len(alternative_tasks[r][ri]))
                            if alternative_tasks[r][ri][t] is not None) <= 1)
        # Minimize end of all tasks
        self.objective_expression = minimize(sum_of([used[r][k]
                                                     for r in dict_resource_allocation
                                                     for k in range(len(dict_resource_allocation[r]))]))
        mdl.add(self.objective_expression)

        self.model = mdl
        self.variables = {"tasks": tasks, "resource_allocation": dict_resource_allocation}

    def init_resource(self, **args):
        NB_TASKS, NB_RESOURCES = self.rcpsp_model.n_jobs, len(self.rcpsp_model.resources_list)
        MAX_CAPACITIES = [self.rcpsp_model.resources[r] for r in self.rcpsp_model.resources_list]

        tasks_list = self.rcpsp_model.tasks_list
        self.index_task = {tasks_list[i]: i for i in range(len(tasks_list))}
        # -----------------------------------------------------------------------------
        # Prepare the data for modeling
        # -----------------------------------------------------------------------------
        # Extract duration of each task
        DURATIONS = [self.rcpsp_model.mode_details[t][1]["duration"] for t in tasks_list]

        # Extract demand of each task
        DEMANDS = [[self.rcpsp_model.mode_details[t][1][r] for r in self.rcpsp_model.resources_list]
                   for t in tasks_list]

        # Extract successors of each task
        SUCCESSORS = [[self.index_task[tt] for tt in self.rcpsp_model.successors[t]]
                      for t in tasks_list]

        # -----------------------------------------------------------------------------
        # Build the model
        # -----------------------------------------------------------------------------

        # Create model
        mdl = CpoModel()
        # Create task interval variables
        CAPACITIES = [integer_var(min=0, max=MAX_CAPACITIES[i]) for i in range(len(MAX_CAPACITIES))]
        for i in range(len(self.rcpsp_model.resources_list)):
            r = self.rcpsp_model.resources_list[i]
            max_ = max([self.rcpsp_model.mode_details[t][1].get(r, 0)
                        for t in self.rcpsp_model.tasks_list])
            mdl.add(CAPACITIES[i] >= max_)
        tasks = [interval_var(name='T{}'.format(i + 1), size=DURATIONS[i]) for i in range(NB_TASKS)]
        # Add precedence constraints
        mdl.add(end_before_start(tasks[t], tasks[s]) for t in range(NB_TASKS) for s in SUCCESSORS[t])
        # Constrain capacity of resources
        mdl.add(sum(pulse(tasks[t], DEMANDS[t][r]) for t in range(NB_TASKS) if DEMANDS[t][r] > 0) <= CAPACITIES[r]
                for r in range(NB_RESOURCES))
        # Minimize end of all tasks
        self.objective_expression = minimize(sum_of(CAPACITIES))
        mdl.add(self.objective_expression)
        self.model = mdl
        self.variables = {"tasks": tasks}

    def retrieve_solutions(self, result, parameters_cp: ParametersCP) -> ResultStorage:
        list_solution_fit = []
        schedule = {}
        for i in range(len(self.variables["tasks"])):
            itv = result.get_var_solution(self.variables["tasks"][i])

            schedule[self.rcpsp_model.tasks_list[i]] = {"start_time": itv.get_start(),
                                                        "end_time": itv.get_end()}
        dict_resource_allocation = {}
        for r in self.variables["resource_allocation"]:
            dict_resource_allocation[r] = {}
            for ri in range(len(self.variables["resource_allocation"][r])):
                dict_resource_allocation[r][ri] = []
                for ti in range(len(self.variables["resource_allocation"][r][ri])):
                    val = self.variables["resource_allocation"][r][ri][ti]
                    if isinstance(val, int):
                        if val == 1:
                            dict_resource_allocation[r][ri] += [self.rcpsp_model.tasks_list[ti]]
                    else:
                        x: CpoIntVar = val
                        itv: CpoVarSolution = result.get_var_solution(x)
                        if itv.get_value() == 1:
                            dict_resource_allocation[r][ri] += [self.rcpsp_model.tasks_list[ti]]
        solution = RCPSPSolution(problem=self.rcpsp_model,
                                 rcpsp_schedule=schedule,
                                 rcpsp_modes=[1]*self.rcpsp_model.n_jobs_non_dummy)
        solution.alloc = dict_resource_allocation
        fit = self.aggreg_sol(solution)
        return ResultStorage(list_solution_fits=[(solution, fit)],
                             mode_optim=self.params_objective_function.sense_function)

    def solve(self, parameters_cp: ParametersCP, **args) -> ResultStorage:
        result: CpoSolveResult = self.model.solve(TimeLimit=parameters_cp.TimeLimit,
                                                  execfile="/Applications/CPLEX_Studio201/cpoptimizer/bin/x86-64_osx/cpoptimizer") # put yours instead.
        self.results = result
        self.solver_infos = result.solver_infos
        return self.retrieve_solutions(result, parameters_cp)

    def get_stats(self):
        return self.solver_infos


    # Function to run LNS
    def constraint_objective_makespan(self):
        self.model.add(minimize(max(end_of(t) for t in self.variables["tasks"])))

    def constraint_objective_equal_makespan(self, task_sink):
        #TODO
        pass

    def constraint_objective_max_time_set_of_jobs(self, set_of_jobs):
        expression = minimize(max(end_of(self.variables["tasks"][self.index_task[t]])
                              for t in set_of_jobs))
        self.model.add(expression)
        return [expression]

    def constraint_start_time_string(self, task, start_time, sign: SignEnum = SignEnum.EQUAL) -> List[CpoExpr]:
        expression = None
        if sign == SignEnum.EQUAL:
            expression = start_of(self.variables["tasks"][self.index_task[task]]) == start_time
        elif sign == SignEnum.LEQ:
            expression = start_of(self.variables["tasks"][self.index_task[task]]) <= start_time
        elif sign == SignEnum.LESS:
            expression = start_of(self.variables["tasks"][self.index_task[task]]) < start_time
        elif sign == SignEnum.UP:
            expression = start_of(self.variables["tasks"][self.index_task[task]]) > start_time
        elif sign == SignEnum.UEQ:
            expression = start_of(self.variables["tasks"][self.index_task[task]]) >= start_time
        self.model.add(expression)
        return [expression]

    def constraint_end_time_string(self, task, start_time, sign: SignEnum = SignEnum.EQUAL) -> List[CpoExpr]:
        expression = None
        if sign == SignEnum.EQUAL:
            expression = start_of(self.variables["tasks"][self.index_task[task]]) == start_time
        elif sign == SignEnum.LEQ:
            expression = start_of(self.variables["tasks"][self.index_task[task]]) <= start_time
        elif sign == SignEnum.LESS:
            expression = start_of(self.variables["tasks"][self.index_task[task]]) < start_time
        elif sign == SignEnum.UP:
            expression = start_of(self.variables["tasks"][self.index_task[task]]) > start_time
        elif sign == SignEnum.UEQ:
            expression = start_of(self.variables["tasks"][self.index_task[task]]) >= start_time
        self.model.add(expression)
        return [expression]


def vizu(result_storage: ResultStorage):
    if visu.is_visu_enabled():
        solution: RCPSPSolution = result_storage.get_best_solution_fit()[0]
        problem: RCPSPModel = solution.problem
        load = {r: CpoStepFunction() for r in problem.resources_list}
        visu.timeline('Solution for RCPSP ')
        visu.panel('Tasks')
        for i in range(problem.n_jobs):
            visu.interval(solution.rcpsp_schedule[problem.tasks_list[i]]["start_time"],
                          solution.rcpsp_schedule[problem.tasks_list[i]]["end_time"],
                          i, str(problem.tasks_list[i]))
            for r in problem.resources_list:
                if problem.mode_details[problem.tasks_list[i]][1][r]>0:
                    load[r].add_value(solution.rcpsp_schedule[problem.tasks_list[i]]["start_time"],
                                      solution.rcpsp_schedule[problem.tasks_list[i]]["end_time"],
                                      problem.mode_details[problem.tasks_list[i]][1][r])

        j = 0
        for r in load:
            visu.panel('R-'+str(r))
            visu.function(segments=[(INTERVAL_MIN, INTERVAL_MAX, problem.resources[r])], style='area', color='lightgrey')
            visu.function(segments=load[r], style='area', color=j)
            j += 1
        visu.show()





