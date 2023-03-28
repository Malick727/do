from typing import List
import os

from docplex.cp.modeler import step_at

os.environ["DO_SKIP_MZN_CHECK"] = "1"
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPSolution
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.cp_tools import CPSolver, CPSolverName, ParametersCP, SignEnum
from discrete_optimization.generic_tools.do_problem import Problem, ParamsObjectiveFunction, \
    build_aggreg_function_and_params_objective
from docplex.cp.model import interval_var, minimize, start_before_end, end_before_start, CpoModel, end_of, \
    pulse, start_of, CpoExpr, CpoStepFunction, INTERVAL_MIN, INTERVAL_MAX, CpoSolveResult, \
    integer_var, sum_of, no_overlap, max_of, min_of, height_at_start
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
        NB_TASKS, NB_RESOURCES = self.rcpsp_model.n_jobs, len(self.rcpsp_model.resources_list)
        CAPACITIES = [self.rcpsp_model.resources[r] for r in self.rcpsp_model.resources]
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
        tasks = [interval_var(name='T{}'.format(i + 1), size=DURATIONS[i]) for i in range(NB_TASKS)]
        # Add precedence constraints
        mdl.add(end_before_start(tasks[t], tasks[s]) for t in range(NB_TASKS) for s in SUCCESSORS[t])
        # Constrain capacity of resources
        mdl.add(sum(pulse(tasks[t], DEMANDS[t][r]) for t in range(NB_TASKS) if DEMANDS[t][r] > 0) <= CAPACITIES[r]
                for r in range(NB_RESOURCES))
        # Minimize end of all tasks
        print(tasks)
        # self.objective_expression = minimize(max([end_of(t) for t in tasks]))
        mdl.add(minimize(max_of([end_of(t) for t in tasks])))
        self.model = mdl
        self.variables = {"tasks": tasks}

    def flexible_ressource(self, **args):
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
        max_demand = [max(DEMANDS[i]) for i in range(len(DEMANDS))]
        index_demand = [DEMANDS[i].index(max_demand[i]) for i in range(len(max_demand))]

        # Extract successors of each task
        SUCCESSORS = [[self.index_task[tt] for tt in self.rcpsp_model.successors[t]]
                      for t in tasks_list]

        # -----------------------------------------------------------------------------
        # Build the model
        # -----------------------------------------------------------------------------

        # Create model
        mdl = CpoModel()
        # Create task interval variables
        tasks = [interval_var(name='T{}'.format(i + 1),
                              size=DURATIONS[i],
                              optional=False) for i in range(NB_TASKS)]

        res_used_per_task = {r: {} for r in range(len(self.rcpsp_model.resources_list))}
        cumulative_task = {t: 0 for t in range(self.rcpsp_model.n_jobs)}
        cumulative_res = {r: 0 for r in range(len(self.rcpsp_model.resources_list))}
        for j in range(self.rcpsp_model.n_jobs):
            if max_demand[j] == 0:
                continue
            else:
                ind_ = index_demand[j]
                for k in range(ind_, len(self.rcpsp_model.resources_list)):
                    res_used_per_task[k][j] = integer_var(0, max_demand[j], name=f"Res_{k}_Task_{j}")
                    pulse_res = pulse(tasks[j], height=(0, max_demand[j]))
                    mdl.add(height_at_start(tasks[j], pulse_res) == res_used_per_task[k][j])
                    cumulative_task[j] += pulse_res
                    cumulative_res[k] += pulse_res
                mdl.add(sum_of([res_used_per_task[k][j] for k in range(ind_,
                                                                       len(self.rcpsp_model.resources_list))]) ==
                        max_demand[j])
                #mdl.add(cumulative_task[j]-pulse(tasks[j], max_demand[j]) >= 0)
                mdl.add(height_at_start(tasks[j], cumulative_task[j]) == max_demand[j])
        for k in cumulative_res:
            mdl.add(cumulative_res[k] <= CAPACITIES[k])

        # Add precedence constraints
        mdl.add(end_before_start(tasks[t], tasks[s]) for t in range(NB_TASKS) for s in SUCCESSORS[t])
        # Constrain capacity of resources
        # mdl.add(sum(pulse(tasks[t], DEMANDS[t][r]) for t in range(NB_TASKS) if DEMANDS[t][r] > 0) <= CAPACITIES[r]
        #         for r in range(NB_RESOURCES))
        # Minimize end of all tasks
        print(tasks)
        # self.objective_expression = minimize(max([end_of(t) for t in tasks]))
        mdl.add(minimize(max_of([end_of(t) for t in tasks])))
        # for k in cumulative_task:
        #    print(cumulative_task[k])
        #    mdl.add_kpi(cumulative_task[k], name=f"res_alloc_to_{k}")
        self.model = mdl
        self.variables = {"tasks": tasks,
                          "res_per_resource": res_used_per_task,
                          "cumulative_per_task": cumulative_task}

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
        mdl.add(sum(pulse(tasks[t],
                          DEMANDS[t][r]) for t in range(NB_TASKS) if DEMANDS[t][r] > 0) <= CAPACITIES[r]
                for r in range(NB_RESOURCES))
        # Minimize end of all tasks
        #self.objective_expression = minimize(sum_of(CAPACITIES))
        #mdl.add(self.objective_expression)
        self.model = mdl
        self.variables = {"tasks": tasks}

    def retrieve_solutions(self, result: CpoSolveResult, parameters_cp: ParametersCP) -> ResultStorage:
        list_solution_fit = []
        schedule = {}
        if "res_per_resource" in self.variables:
            for j in range(self.rcpsp_model.n_jobs):
                print("TASK ", j)
                kk = [k for k in self.variables["res_per_resource"] if
                      j in self.variables["res_per_resource"][k]]
                for k in kk:
                    print(f"Res_{k}_task_{j},{result.get_var_solution(self.variables['res_per_resource'][k][j])}")
                print("----\n")
        for i in range(len(self.variables["tasks"])):
            itv = result.get_var_solution(self.variables["tasks"][i])
            schedule[self.rcpsp_model.tasks_list[i]] = {"start_time": itv.get_start(),
                                                        "end_time": itv.get_end()}
        solution = RCPSPSolution(problem=self.rcpsp_model, rcpsp_schedule=schedule,
                                 rcpsp_modes=[1]*self.rcpsp_model.n_jobs_non_dummy)
        fit = self.aggreg_sol(solution)
        return ResultStorage(list_solution_fits=[(solution, fit)],
                             mode_optim=self.params_objective_function.sense_function)

    def solve(self, parameters_cp: ParametersCP = None, **args) -> ResultStorage:
        result: CpoSolveResult = self.model.solve(TimeLimit=parameters_cp.time_limit,
                                                  execfile="/Applications/CPLEX_Studio201/cpoptimizer/"
                                                           "bin/x86-64_osx/cpoptimizer")
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


def main():
    from discrete_optimization.rcpsp.rcpsp_parser import parse_file, get_data_available
    import random
    file = random.choice([f for f in get_data_available() if "j12048_4.sm" in f])
    print(file)
    model = parse_file(file)
    print(model.resources_list)
    solver = CPOptSolver(rcpsp_model=model)
    solver.flexible_ressource()
    # solver.init_model()
    parameters = ParametersCP.default()
    solver.solve(parameters_cp=parameters)

if __name__ == "__main__":
    main()


