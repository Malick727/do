import os
import sys

from discrete_optimization.generic_tools.cp_tools import ParametersCp
from multibatching.solvers.packing_subproblem import CpsatPackingSubproblem

this_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(this_folder, "../../"))

from examples_do_usecases.multibatching.instance2 import init_instance
from examples_do_usecases.multibatching.solve_industrial_use_case import parse_problem
from multibatching.solvers.cpsat import CpsatMultibatchingSolver, ModelingMultiBatch
from multibatching.solvers.column_generation import ColumnGenerationSolver
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from multibatching.solvers.two_steps import (TwoStepMultibatchingSolver, NetxMultibatchingSolver,
                                             PackingViaBinPacking, GreedyPackingForMultibatching)
from multibatching.problem import MultibatchingProblem, MultibatchingSolution
import logging
logging.basicConfig(level=logging.INFO)
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
    print(sum(problem.evaluate(sol).values()))


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


def run_cp_after_milp():
    import pickle
    # This was the final solution of column gen.
    sol = pickle.load(open("sol_debug.pk", "rb"))
    problem: MultibatchingProblem = sol.problem
    print(problem.satisfy(sol), problem.evaluate(sol))
    # Rebuild discrete flow, possibly breaking flow and demand constraint slightly.
    solver = PackingViaBinPacking(problem)
    solver.init_from_solution(sol)
    res = solver.solve(time_limit_per_link=1)

    # CP solver reasoning on total flows.
    solver = CpsatMultibatchingSolver(problem=problem)
    solver.init_model(modeling=ModelingMultiBatch.FLOW)
    # Warm start with prev solution.
    solver.set_warm_start_flow(solution=res[-1][0])
    p = ParametersCp.default_cpsat()
    p.nb_process = 32
    res = solver.solve(parameters_cp=p,
                       time_limit=100,
                       ortools_cpsat_solver_kwargs={"log_search_progress": True})
    print(problem.evaluate(res[-1][0]), problem.satisfy(res[-1][0]))

    # Post pro the cp-flow solution.
    solver = CpsatPackingSubproblem(problem=problem, scaling_factor=1)
    solver.init_from_solution(res[-1][0])
    res = solver.solve(time_limit=30)
    sol = res[-1][0]
    print(problem.evaluate(sol), problem.satisfy(sol))
    print(sum(problem.evaluate(sol).values()))


def run_two_step_asp_binpack():
    """
    Just run a solver that can compute either continuous or approximated flows,
     and then we simply postprocess it to get valid discrete transport with another "Packing" solver.
    :return:
    """
    from discrete_optimization.binpack.solvers.asp import AspBinPackingSolver
    from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
    problem = parse_problem()
    solver = TwoStepMultibatchingSolver(problem=problem)
    params_cp = ParametersCp.default_cpsat()
    params_cp.nb_process = 16
    res = solver.solve(flow_solver=SubBrick(cls=CpsatMultibatchingSolver,
                                            kwargs=dict(modeling=ModelingMultiBatch.FLOW,
                                                        parameters_cp=params_cp,
                                                        ortools_cpsat_solver_kwargs={"log_search_progress": True},
                                                        time_limit=20)),
                       packing_solver=SubBrick(cls=PackingViaBinPacking,
                                               kwargs={"time_limit_per_link": 1,
                                                       "bin_packing_solver": SubBrick(cls=AspBinPackingSolver,
                                                                                      kwargs={})}))

    sol1 = res.get_best_solution()
    print(problem.evaluate(sol1), problem.satisfy(sol1))



if __name__ == "__main__":
    run_two_step_asp_binpack()

