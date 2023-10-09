from typing import Any
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.ls.local_search import RestartHandlerLimit, ModeMutation
from discrete_optimization.generic_tools.ls.simulated_annealing import TemperatureSchedulingFactor, SimulatedAnnealing
from discrete_optimization.generic_tools.mutations.mixed_mutation import BasicPortfolioMutation
from discrete_optimization.generic_tools.mutations.mutation_catalog import get_available_mutations
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
import numpy as np
from aircraft_loading.aircraft_loading_problem import AircraftLoadingSolution, AircraftLoadingProblem
from aircraft_loading.solvers.aircraft_loading_solver import BaseSolver


class LocalSolverAircraftLoading(BaseSolver):
    problem: AircraftLoadingProblem

    def solve(self, **kwargs: Any) -> ResultStorage:
        solution = kwargs.get("dummy_solution", self.problem.get_dummy_solution())
        nb_iteration_max = kwargs.get("nb_iteration_max", 10000)
        _, list_mutation = get_available_mutations(self.problem, solution)
        res = RestartHandlerLimit(3000, solution, self.problem.evaluate(solution))
        list_mutation = [
            mutate[0].build(self.problem, solution, **mutate[1])
            for mutate in list_mutation
        ]
        weight = np.ones(len(list_mutation))
        mutate_portfolio = BasicPortfolioMutation(list_mutation, weight)
        sa = SimulatedAnnealing(
            evaluator=self.problem,
            mutator=mutate_portfolio,
            restart_handler=res,
            temperature_handler=TemperatureSchedulingFactor(
                temperature=10, restart_handler=res, coefficient=1
            ),
            mode_mutation=ModeMutation.MUTATE_AND_EVALUATE,
            params_objective_function=self.params_objective_function,
        )
        sol = sa.solve(
            initial_variable=solution, nb_iteration_max=nb_iteration_max
        )
        return sol