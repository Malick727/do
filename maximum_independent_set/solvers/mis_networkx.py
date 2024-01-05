from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from typing import Any

from networkx.algorithms.approximation import maximum_independent_set
from maximum_independent_set.solvers.mis_solvers import MisSolver, MisProblem
from maximum_independent_set.mis_problem import MisProblem, MisSolution


class MisNXSolver(MisSolver):
    def solve(self, **kwargs: Any) -> ResultStorage:
        sol = maximum_independent_set(self.problem.graph.graph_nx)
        print(sol)
        chosen = [1 if self.problem.nodes[i] in sol
                  else 0 for i in range(self.problem.number_nodes)]

        solution = MisSolution(problem=self.problem,
                               chosen=chosen)
        fit = self.aggreg_sol(solution)
        return ResultStorage([(solution, fit)],
                             mode_optim=self.params_objective_function.sense_function)