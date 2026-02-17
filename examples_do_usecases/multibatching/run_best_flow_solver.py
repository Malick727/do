from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.sequential_metasolver import SequentialMetasolver
from discrete_optimization.generic_tools.study import SolverConfig
from multibatching.solvers.milp_flow import GurobiMultibatchingSolver, GurobiMultibatchingSolverUnitFlow
from multibatching.solvers.packing_subproblem import GreedyPackingForMultibatching
from multibatching.solvers.two_steps import TwoStepMultibatchingSolver
from examples_do_usecases.multibatching.solve_industrial_use_case import parse_problem

def run():
    solver_config = SolverConfig(cls=SequentialMetasolver,
                                 kwargs=dict(list_subbricks=[
                                     SubBrick(cls=TwoStepMultibatchingSolver,
                                              kwargs={"flow_solver":
                                                          SubBrick(cls=GurobiMultibatchingSolver,
                                                                   kwargs={"time_limit": 30}),
                                                      "packing_solver":
                                                          SubBrick(cls=GreedyPackingForMultibatching,
                                                                   kwargs={})}),
                                     SubBrick(GurobiMultibatchingSolverUnitFlow,
                                              dict(time_limit=30,
                                                   delta_to_solution=2),
                                              kwargs_from_solution={"solution": lambda sol: sol}
                                              )]
                                 )
                                 )
    problem = parse_problem(False, False)
    solver = solver_config.cls(problem, **solver_config.kwargs)
    solver.init_model(**solver_config.kwargs)
    res = solver.solve(**solver_config.kwargs)
    sol = res[-1][0]
    print(problem.evaluate(sol), problem.satisfy(sol))

if __name__ == "__main__":
    run()