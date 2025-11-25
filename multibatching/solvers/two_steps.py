from typing import Any

from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrickHyperparameter, SubBrick
from discrete_optimization.knapsack.solvers.greedy import GreedyBestKnapsackSolver
from multibatching.problem import MultibatchingProblem
from multibatching.solvers.packing_subproblem import CpsatPackingSubproblem, GreedyPackingForMultibatching, \
    PackingViaBinPacking, PackingSubproblemSolver
from multibatching.solvers.cpsat import CpsatMultibatchingSolver, ModelingMultiBatch
from multibatching.solvers.oe.solver_2 import OpenEvolveMultibatchingSolver
from multibatching.solvers.netx import NetxMultibatchingSolver
from multibatching.solvers.column_generation import ColumnGenerationSolver, MultibatchingSolver


class TwoStepMultibatchingSolver(SolverDO):
    hyperparameters = [SubBrickHyperparameter(name="flow_solver",
                                              choices=[CpsatMultibatchingSolver,
                                                       ColumnGenerationSolver,
                                                       OpenEvolveMultibatchingSolver,
                                                       NetxMultibatchingSolver],
                                              default=SubBrick(cls=CpsatMultibatchingSolver,
                                                               kwargs={"modeling": ModelingMultiBatch.FLOW})),
                       SubBrickHyperparameter(name="packing_solver",
                                              choices=[CpsatPackingSubproblem,
                                                       GreedyPackingForMultibatching,
                                                       PackingViaBinPacking],
                                              default=SubBrick(cls=GreedyBestKnapsackSolver, kwargs=None))]
    problem: MultibatchingProblem

    def __init__(self, problem: MultibatchingProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)

    def solve(self, **args):
        args = self.complete_with_default_hyperparameters(args)
        sub_brick_flow: SubBrick = args["flow_solver"]
        sub_brick_pack: SubBrick = args["packing_solver"]

        solver_flow: MultibatchingSolver = sub_brick_flow.cls(problem=self.problem)
        solver_flow.init_model(**sub_brick_flow.kwargs)
        res = solver_flow.solve(**sub_brick_flow.kwargs)
        solver_pack: PackingSubproblemSolver = sub_brick_pack.cls(problem=self.problem)
        solver_pack.init_from_solution(res[-1][0])
        res_ = solver_pack.solve(**sub_brick_pack.kwargs)
        return res_


