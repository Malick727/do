from typing import Optional, List, Any

from docplex.cp.model import CpoModel, CpoSolver, integer_var, interval_var, sequence_var
from docplex.cp.modeler import sequence, first, last
from docplex.cp.model import *

from cpopt_solvers.cpopt_solver import CPOSolver
from discrete_optimization.generic_tools.do_problem import Problem, ParamsObjectiveFunction, Solution
from discrete_optimization.tsptw.problem import TSPTWProblem, TSPTWSolution


class CPOptSolverTSPTw(CPOSolver):
    problem: TSPTWProblem

    def __init__(self, problem: Problem, params_objective_function: Optional[ParamsObjectiveFunction] = None,
                 **kwargs: Any):
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}
        self.distance_matrix = self.problem.distance_matrix
        self.distance_matrix = self.distance_matrix.astype(int)

    def init_model(self, scaling: float = 1, **kwargs: Any) -> None:
        time_window = self.problem.time_windows
        if scaling != 1:
            self.distance_matrix = (scaling*self.distance_matrix).astype(int)
            time_window = [(int(scaling*x[0]), int(scaling*x[1]))
                           for x in time_window]
        num_nodes = self.problem.nb_nodes
        # Create model
        mdl = CpoModel(name="TSP-TW")
        intervals_var = [interval_var(start=time_window[i],
                                      size=0, name=f"visit_{i}")
                         for i in range(num_nodes)]
        tour_end = interval_var(start=time_window[self.problem.depot_node],
                                size=0, name="tour_end")
        # Last transition is go back to depot
        seq = sequence_var(intervals_var,
                           # types=[i for i in range(num_nodes)
                           #        if i != self.problem.depot_node],
                           name="seq")
        mdl.add(no_overlap(seq, distance_matrix=self.distance_matrix))
        mdl.add(first(seq, intervals_var[self.problem.depot_node]))
        mdl.add(start_of(intervals_var[self.problem.depot_node]) == 0)
        for i in range(num_nodes):
            mdl.add(end_before_start(intervals_var[i], tour_end,
                                     self.distance_matrix[i][self.problem.depot_node]))
        last_time = start_of(tour_end)
        mdl.add(minimize(last_time))
        self.cp_model = mdl
        self.variables["intervals"] = intervals_var

    def retrieve_solution(self, sres: CpoSolveResult) -> Solution:
        schedule = {}
        for i in range(len(self.variables["intervals"])):
            itv = sres.get_var_solution(self.variables["intervals"][i])
            start = itv.get_start()
            schedule[i] = start
        sort_ = sorted(schedule, key=lambda x: schedule[x])
        return TSPTWSolution(problem=self.problem,
                             permutation=[s for s in sort_ if s != self.problem.depot_node])