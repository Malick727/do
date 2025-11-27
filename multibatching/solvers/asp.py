from typing import Any

import clingo

from discrete_optimization.generic_tools.asp_tools import AspClingoSolver
from discrete_optimization.generic_tools.do_problem import Solution
from multibatching.problem import MultibatchingProblem, MultibatchingSolution

class AspMultibatchingFlowSolver(AspClingoSolver):
    problem: MultibatchingProblem

    def init_model(self, **kwargs: Any) -> None:
        # See the bin packing example to setup the program/facts etc.
        pass
    def retrieve_solution(self, model: clingo.Model) -> Solution:
        # Rebuitl a solutin from the model
        return MultibatchingSolution(problem=self.problem, list_flows=[])


