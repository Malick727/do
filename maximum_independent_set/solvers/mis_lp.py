from typing import Any

from discrete_optimization.generic_tools.lp_tools import ParametersMilp, CplexMilpSolver
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage

from maximum_independent_set.solvers.mis_solvers import MisSolver
from maximum_independent_set.mis_problem import MisProblem, MisSolution
import docplex.mp.model as cplex


class MISLpSolver(CplexMilpSolver, MisSolver):

    def init_model(self, **kwargs: Any) -> None:
        self.model = cplex.Model("mis")
        x = self.model.var_list(keys=range(self.problem.number_nodes),
                                vartype=self.model.binary_vartype)
        self.model.add_constraints(
            (x[self.problem.nodes_to_index[e[0]]] + x[self.problem.nodes_to_index[e[1]]] <= 1
             for e in self.problem.edges))
        self.variables = {}
        self.variables["x"] = x
        obj = self.model.integer_var(lb=0, ub=10**9, name="obj")
        if self.problem.attribute_aggregate == "size":
            self.model.add(obj == sum(x))
        else:
            self.model.add(obj == sum(x[i]*self.problem.attr_list[i]
                                      for i in range(self.problem.number_nodes)))
        self.model.maximize(obj)
        self.variables["obj"] = obj

    def retrieve_solutions(self, parameters_milp: ParametersMilp) -> ResultStorage:
        list_solution_fit = []
        for i in range(self.nb_solutions):
            chosen = [self.get_var_value_for_ith_solution(xi, i)
                      for xi in self.variables["x"]]
            sol = MisSolution(problem=self.problem,
                              chosen=chosen)
            fit = self.aggreg_sol(sol)
            list_solution_fit += [(sol, fit)]
        return ResultStorage(list_solution_fit, mode_optim=self.params_objective_function.sense_function)

