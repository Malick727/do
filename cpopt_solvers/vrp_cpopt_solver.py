from typing import Optional, List, Any

from docplex.cp.model import CpoModel, CpoSolver, integer_var, interval_var, sequence_var
from docplex.cp.modeler import sequence, first, last
from docplex.cp.model import *

from discrete_optimization.generic_tools.callbacks.callback import Callback
from discrete_optimization.generic_tools.cp_tools import ParametersCP
from discrete_optimization.generic_tools.do_problem import Problem, ParamsObjectiveFunction
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from discrete_optimization.vrp.vrp_toolbox import compute_length_matrix
from discrete_optimization.vrp.vrp_model import VrpProblem, VrpSolution
from discrete_optimization.vrp.solver.vrp_solver import SolverVrp


class CPOptSolverTSP(SolverVrp):
    def __init__(self, problem: Problem, params_objective_function: Optional[ParamsObjectiveFunction] = None,
                 **kwargs: Any):
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}
        self.cpo_model = None
        _, self.distance_matrix = compute_length_matrix(
            self.problem,
        )
        for k in range(self.problem.vehicle_count):
            self.distance_matrix[self.problem.end_indexes[k],
                                 self.problem.start_indexes[k]] = 0
            self.distance_matrix[self.problem.start_indexes[k],
                                 self.problem.end_indexes[k]] = 0
        self.distance_matrix = (10*self.distance_matrix).astype(int)
        print(self.distance_matrix)

    def init_model(self, **kwargs: Any) -> None:
        nb_nodes = self.problem.customer_count
        indexes_start = self.problem.start_indexes
        indexes_end = self.problem.end_indexes
        nodes_non_dummy = [j for j in range(nb_nodes) if j not in indexes_start and j not in indexes_end]
        # Create model
        mdl = CpoModel(name="CVRP")
        intervals_var_dict = {}
        sequence_var_dict = {}
        for vehicle in range(self.problem.vehicle_count):
            intervals_var_dict[vehicle] = [interval_var(size=0,
                                                        # start=(0, 5000),
                                                        # end=(0, 5000),
                                                        optional=True)
                                           for i in range(nb_nodes+1)]
            sequence_var_dict[vehicle] = sequence_var(intervals_var_dict[vehicle],
                                                      types=[i for i in range(nb_nodes)]
                                                      + [self.problem.end_indexes[vehicle]])
            mdl.add(presence_of(intervals_var_dict[vehicle][self.problem.start_indexes[vehicle]]))
            mdl.add(start_of(intervals_var_dict[vehicle][self.problem.start_indexes[vehicle]]) == 0)
            mdl.add(presence_of(intervals_var_dict[vehicle][-1]))
            mdl.add(first(sequence_var_dict[vehicle],
                          intervals_var_dict[vehicle][self.problem.start_indexes[vehicle]]))
            mdl.add(last(sequence_var_dict[vehicle], intervals_var_dict[vehicle][-1]))
            if self.problem.end_indexes[vehicle] != self.problem.start_indexes[vehicle]:
                mdl.add(previous(sequence,
                                 intervals_var_dict[vehicle][self.problem.end_indexes[vehicle]],
                                 intervals_var_dict[vehicle][-1]))
            mdl.add(no_overlap(sequence_var_dict[vehicle], distance_matrix=self.distance_matrix))
            # capacity
            mdl.add(sum_of([step_at_start(intervals_var_dict[vehicle][i],
                                          self.problem.customers[i].demand)
                            for i in range(nb_nodes)]) <= self.problem.vehicle_capacities[vehicle])
            # mdl.add(sum([presence_of(intervals_var_dict[vehicle][i])*self.problem.customers[i].demand
            #              for i in range(nb_nodes)]) <= self.problem.vehicle_capacities[vehicle])
        intervals_non_dummy = [interval_var(size=0, optional=False)
                               for j in range(len(nodes_non_dummy))]
        for j, node in enumerate(nodes_non_dummy):
            mdl.add(sum_of([presence_of(intervals_var_dict[vehicle][node])
                                        for vehicle in intervals_var_dict]) == 1)
            mdl.add(alternative(intervals_non_dummy[j],
                                [intervals_var_dict[vehicle][node]
                                for vehicle in intervals_var_dict]))
        number_of_vehicles = integer_var_list(self.problem.vehicle_count,0, 1)
        for k in range(self.problem.vehicle_count):
            mdl.add(number_of_vehicles[k]
                    == any(presence_of(intervals_var_dict[k][node])
                    for node in range(nb_nodes) if node != self.problem.start_indexes[k]))
        last_time = sum_of([start_of(intervals_var_dict[vehicle][-1])
                            for vehicle in intervals_var_dict])
        # mdl.add(minimize(sum(number_of_vehicles)))
        mdl.add(minimize_static_lex([last_time,
                                     max([end_of(s) for s in intervals_non_dummy])]))
        self.cpo_model = mdl
        self.variables["intervals"] = intervals_var_dict

    def solve(self, callbacks: Optional[List[Callback]] = None,
              parameters_cp: ParametersCP=None,
              **kwargs: Any) -> ResultStorage:
        if parameters_cp is None:
            parameters_cp = ParametersCP.default()
        result: CpoSolveResult = self.cpo_model.solve(
            TimeLimit=parameters_cp.time_limit,
            execfile="/Applications/CPLEX_Studio2211/cpoptimizer/bin/arm64_osx/cpoptimizer",
            trace_log=True,
            #SearchType="IterativeDiving"
        )  # put yours instead.
        sol = VrpSolution(problem=self.problem,
                          list_start_index=self.problem.start_indexes,
                          list_end_index=self.problem.end_indexes,
                          list_paths=[])
        for vehicle in self.variables["intervals"]:
            sched = {}
            for i in range(len(self.variables["intervals"][vehicle])):
                itv: CpoIntervalVarSolution = result.get_var_solution(self.variables["intervals"][vehicle][i])
                is_present = itv.is_present()
                if is_present:
                    start = itv.get_start()
                    sched[i] = start
            p = sorted(sched, key=lambda x: sched[x])
            sol.list_paths.append(p[1:-1])
        fit = self.aggreg_from_sol(sol)
        return ResultStorage([(sol, fit)], mode_optim=self.params_objective_function.sense_function)

