from typing import List, Optional, Dict
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPSolution
from discrete_optimization.generic_tools.cp_tools import CPSolver, ParametersCP, SignEnum
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction, \
    build_aggreg_function_and_params_objective
from docplex.cp.model import interval_var, minimize, end_before_start, CpoModel, end_of, \
    pulse, start_of, CpoExpr, CpoStepFunction, CpoSolveResult, CpoIntervalVar
from discrete_optimization.rcpsp.rcpsp_model_utils import create_fake_tasks


def build_intensity_step_functions(rcpsp_model: RCPSPModel):
    max_horizon = rcpsp_model.horizon
    resources = rcpsp_model.resources
    max_cap = {r: rcpsp_model.get_max_resource_capacity(r)
               for r in resources}
    # Dictionary of resource intensity step function
    step_dict = {r: CpoStepFunction() for r in resources}
    for r in step_dict:
        step_dict[r].set_value(0, max_horizon, 100)
    # by default the value is 100% of availability
    data = create_fake_tasks(rcpsp_problem=rcpsp_model)
    for record in data:
        d = record
        duration = d["duration"]
        start = d["start"]
        res_key = [r for r in d if r not in {"duration", "start"}][0]
        intensity = int((max_cap[res_key] - d[res_key]) / max_cap[res_key])
        step_dict[res_key].set_value(start, start + duration, intensity)
    return step_dict


class CPOptRCPSPBreakSolver(CPSolver):
    def __init__(self, rcpsp_model: RCPSPModel,
                 params_objective_function: ParamsObjectiveFunction = None, **kwargs):
        self.rcpsp_model = rcpsp_model
        self.aggreg_sol, self.aggreg_from_dict_values, self.params_objective_function = \
            build_aggreg_function_and_params_objective(self.rcpsp_model,
                                                       params_objective_function=params_objective_function)
        self.index_task = self.rcpsp_model.index_task
        self.model: Optional[CpoModel] = None
        self.variables: Dict[str, List[CpoIntervalVar]] = {}
        self.results = None
        self.solver_infos = None

    def init_model(self, **args):
        nb_tasks, nb_resources = self.rcpsp_model.n_jobs, len(self.rcpsp_model.resources_list)
        max_capacities = [self.rcpsp_model.get_max_resource_capacity(r) for r in self.rcpsp_model.resources_list]
        resources_list = self.rcpsp_model.resources_list
        tasks_list = self.rcpsp_model.tasks_list
        # -----------------------------------------------------------------------------
        # Prepare the data for modeling
        # -----------------------------------------------------------------------------

        # Extract duration of each task
        durations = [self.rcpsp_model.mode_details[t][1]["duration"] for t in tasks_list]
        # Extract demand of each task
        demands = [[self.rcpsp_model.mode_details[t][1][r] for r in self.rcpsp_model.resources_list]
                   for t in tasks_list]

        # Extract successors of each task
        successors = [[self.rcpsp_model.index_task[tt] for tt in self.rcpsp_model.successors[t]]
                      for t in tasks_list]
        # -----------------------------------------------------------------------------
        # Build the model
        # -----------------------------------------------------------------------------

        # Create model
        mdl = CpoModel("RCPSPBreaks")
        step_functions: Dict[str, CpoStepFunction] = build_intensity_step_functions(self.rcpsp_model)

        # Compute intensity per tasks.
        intensity_per_tasks = []
        for i in range(nb_tasks):
            index_not_zeros = [j for j in range(len(demands[i])) if demands[i][j] > 0]
            # find the resource needed by the resource. We will admit that there is at most 1 resource
            # (simplification !!!)
            if len(index_not_zeros) > 0:
                intensity_per_tasks += [step_functions[resources_list[index_not_zeros[0]]]]
            else:
                intensity_per_tasks += [None]

        # Create task interval variables
        tasks = [interval_var(name='Task_{}'.format(i + 1),
                              intensity=intensity_per_tasks[i],
                              length=(durations[i], 10 * durations[i]),
                              # hardcoded for now.
                              size=durations[i]) for i in range(nb_tasks)]
        # Add precedence constraints
        mdl.add(end_before_start(tasks[t], tasks[s]) for t in range(nb_tasks) for s in successors[t])
        # Constrain capacity of resources
        mdl.add(sum(pulse(tasks[t], demands[t][r]) for t in range(nb_tasks)
                    if demands[t][r] > 0) <= max_capacities[r]
                for r in range(nb_resources))
        # Minimize end of all tasks
        # self.objective_expression = minimize(max([end_of(t) for t in tasks]))
        mdl.add(minimize(end_of(tasks[self.rcpsp_model.index_task[self.rcpsp_model.sink_task]])))
        self.model = mdl
        self.variables = {"tasks": tasks}

    def retrieve_solutions(self, result: CpoSolveResult, parameters_cp: ParametersCP) -> ResultStorage:
        schedule = {}
        for i in range(len(self.variables["tasks"])):
            itv = result.get_var_solution(self.variables["tasks"][i])
            schedule[self.rcpsp_model.tasks_list[i]] = {"start_time": itv.get_start(),
                                                        "end_time": itv.get_end()}
        solution = RCPSPSolution(problem=self.rcpsp_model,
                                 rcpsp_schedule=schedule,
                                 rcpsp_modes=[1] * self.rcpsp_model.n_jobs_non_dummy)
        fit = self.aggreg_sol(solution)
        return ResultStorage(list_solution_fits=[(solution, fit)],
                             mode_optim=self.params_objective_function.sense_function)

    def solve(self, parameters_cp: ParametersCP = None, **args) -> ResultStorage:
        result: CpoSolveResult = self.model.solve(TimeLimit=parameters_cp.time_limit,
                                                  execfile="/Applications/CPLEX_Studio201/cpoptimizer/"
                                                           "bin/x86-64_osx/cpoptimizer")
        return self.retrieve_solutions(result, parameters_cp)

    def get_stats(self):
        return self.solver_infos

    # Function to run LNS
    def constraint_objective_makespan(self):
        self.model.add(minimize(max(end_of(t) for t in self.variables["tasks"])))

    def constraint_objective_equal_makespan(self, task_sink):
        # TODO
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
