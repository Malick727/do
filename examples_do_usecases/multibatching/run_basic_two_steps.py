import os
import sys

this_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(this_folder, "../../"))

from examples_do_usecases.multibatching.instance2 import init_instance
from examples_do_usecases.multibatching.solve_industrial_use_case import parse_problem
from multibatching.solvers.cpsat import CpsatMultibatchingSolver
from multibatching.solvers.column_generation import ColumnGenerationSolver
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from multibatching.solvers.two_steps import (TwoStepMultibatchingSolver, NetxMultibatchingSolver,
                                             PackingViaBinPacking, GreedyPackingForMultibatching)
from multibatching.problem import MultibatchingProblem, MultibatchingSolution

def run_two_step():
    """
    Just run a solver that can compute either continuous or approximated flows,
     and then we simply postprocess it to get valid discrete transport with another "Packing" solver.
    :return:
    """
    problem = parse_problem()
    solver = TwoStepMultibatchingSolver(problem=problem)
    res = solver.solve(flow_solver=SubBrick(cls=NetxMultibatchingSolver,
                                            kwargs={}),
                       packing_solver=SubBrick(cls=PackingViaBinPacking,
                                               kwargs={"time_limit_per_link": 1}))
    
    sol1 = res.get_best_solution()
    

    res = solver.solve(flow_solver=SubBrick(cls=ColumnGenerationSolver,
                                            kwargs={}),
                       packing_solver=SubBrick(cls=PackingViaBinPacking,
                                               kwargs={"time_limit_per_link": 1}))
    
    sol2 = res.get_best_solution()
    print(problem.evaluate(sol1), problem.satisfy(sol1))
    print(problem.evaluate(sol2), problem.satisfy(sol2))

def run_two_step_column_gen():
    """
    Just run a solver that can compute either continuous or approximated flows,
     and then we simply postprocess it to get valid discrete transport with another "Packing" solver.
    :return:
    """
    problem = parse_problem()
    solver = TwoStepMultibatchingSolver(problem=problem)
    res = solver.solve(flow_solver=SubBrick(cls=ColumnGenerationSolver,
                                            kwargs={}),
                       packing_solver=SubBrick(cls=PackingViaBinPacking,
                                               kwargs={"time_limit_per_link": 1}))
    sol = res.get_best_solution()
    print(problem.evaluate(sol), problem.satisfy(sol))


def run_debug():
    import pickle
    sol = pickle.load(open("sol_debug.pk", "rb"))
    problem: MultibatchingProblem = sol.problem
    print(problem.satisfy(sol), problem.evaluate(sol))
    solver = PackingViaBinPacking(problem)
    solver.init_from_solution(sol)
    res = solver.solve(time_limit_per_link=1)
    sol = res.get_best_solution()
    print(problem.evaluate(sol), problem.satisfy(sol))


if __name__ == "__main__":
    run_two_step_column_gen()

