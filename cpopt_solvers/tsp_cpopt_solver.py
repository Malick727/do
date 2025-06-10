from typing import Optional, List, Any

from docplex.cp.model import CpoModel, CpoSolver, integer_var, interval_var, sequence_var
from docplex.cp.modeler import sequence, first, last
from docplex.cp.model import *

from cpopt_solvers.cpopt_solver import CPOSolver
from discrete_optimization.generic_tools.do_problem import Problem, ParamsObjectiveFunction, Solution
from discrete_optimization.tsp.utils import build_matrice_distance
from discrete_optimization.tsp.problem import TspProblem, TspSolution
from discrete_optimization.tsp.solvers.tsp_solver import TspSolver


class CPOptSolverTSP(CPOSolver, TspSolver):
    def __init__(self, problem: Problem, params_objective_function: Optional[ParamsObjectiveFunction] = None,
                 **kwargs: Any):
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}
        self.distance_matrix = build_matrice_distance(
            self.problem.node_count,
            method=self.problem.evaluate_function_indexes,
        )
        self.distance_matrix[self.problem.end_index,
                             self.problem.start_index] = 0
        self.distance_matrix = self.distance_matrix.astype(int)

    def init_model(self, **kwargs: Any) -> None:
        num_nodes = self.problem.node_count
        # Create model
        mdl = CpoModel(name="TSP")
        intervals_var = [interval_var(size=0)
                         for i in range(num_nodes+1)]
        # Last transition is go back to depot
        seq = sequence_var(intervals_var,
                           types=[i for i in range(num_nodes)]+[self.problem.end_index],
                           name="seq")
        mdl.add(no_overlap(seq, distance_matrix=self.distance_matrix))
        mdl.add(first(seq, intervals_var[self.problem.start_index]))
        mdl.add(last(seq, intervals_var[-1]))
        last_time = start_of(intervals_var[-1])
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
        sol = TspSolution(problem=self.problem,
                          start_index=self.problem.start_index,
                          end_index=self.problem.end_index,
                          permutation=sort_[1:-1])
        return sol