from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
import tqdm
from typing import Any

import pytoulbar2
from maximum_independent_set.solvers.mis_solvers import MisSolver
from maximum_independent_set.mis_problem import MisProblem, MisSolution
import logging
import os
this_folder = os.path.abspath(os.path.dirname(__file__))
logger = logging.getLogger(__name__)


class MisToulbarSolver(MisSolver):
    def init_model(self, **kwargs):
        Problem = pytoulbar2.CFN(0)
        vars = [Problem.AddVariable(name=f"x_{i}",
                                    values=[0, 1])
                for i in range(self.problem.number_nodes)]
        #for i in range(len(vars)):
        #    Problem.AddFunction([vars[i]], [0, -int(self.problem.attr_list[i])])
        nb_neighbors = {n: len(list(self.problem.graph.neighbors(n)))
                        for n in self.problem.graph}
        for e in tqdm.tqdm(self.problem.edges):
            i0 = self.problem.nodes_to_index[e[0]]
            i1 = self.problem.nodes_to_index[e[1]]
            # Problem.AddSumConstraint([vars[i0], vars[i1]], '<=', 1)

            Problem.AddFunction([vars[i0], vars[i1]],
                                [10**12 if x == y == 1 else  (x*-int(100*self.problem.attr_list[i0]/nb_neighbors[e[0]])
                                                              +y*-int(100*self.problem.attr_list[i1]/nb_neighbors[e[1]]))
                                for x in [0, 1] for y in [0, 1]])
        self.model = Problem
        print(os.path.join(this_folder, "model_.wcsp"))
        # self.model.Dump(os.path.join(this_folder, "model_.wcsp"))

    def solve(self, **kwargs: Any) -> ResultStorage:
        time_limit = kwargs.get("time_limit", 20)
        self.model.CFN.timer(time_limit)
        solution = self.model.Solve(showSolutions=1)
        logger.info(f"=== Solution === \n {solution}")
        sol = MisSolution(
            problem=self.problem,
            chosen=solution[0][:],
        )
        fit = self.aggreg_sol(sol)
        return ResultStorage([(sol, fit)],
                             mode_optim=self.params_objective_function.sense_function)