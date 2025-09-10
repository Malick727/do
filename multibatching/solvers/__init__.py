from discrete_optimization.generic_tools.do_solver import SolverDO
from multibatching.problem import (MultibatchingProblem,
                                   MultibatchingSolution)


class MultibatchingSolver(SolverDO):
    problem: MultibatchingProblem
