import logging

from discrete_optimization.binpack.solvers.cpsat import CpSatBinPackSolver, ModelingBinPack
from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.sequential_metasolver import SequentialMetasolver
from discrete_optimization.generic_tools.study import SolverConfig
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from multibatching.solvers.milp_flow import GurobiMultibatchingSolver, GurobiMultibatchingSolverUnitFlow
from multibatching.solvers.packing_subproblem import (GreedyPackingForMultibatching,
                                                      PackingViaBinPacking)
from multibatching.solvers.two_steps import TwoStepMultibatchingSolver
from examples_do_usecases.multibatching.solve_industrial_use_case import parse_problem
import logging
logging.basicConfig(level=logging.INFO)
def run_best():
    solver_config = SolverConfig(cls=SequentialMetasolver,
                                 kwargs=dict(list_subbricks=[
                                     SubBrick(cls=TwoStepMultibatchingSolver,
                                              kwargs={"flow_solver":
                                                          SubBrick(cls=GurobiMultibatchingSolver,
                                                                   kwargs={"time_limit": 200,
                                                                           "add_lb_constraint_nb_trips": True}),
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

def run_cpsat():
    from multibatching.solvers.cpsat import CpsatMultibatchingSolver, ModelingMultiBatch
    from discrete_optimization.generic_tools.cp_tools import ParametersCp
    p = ParametersCp.default_cpsat()
    p.nb_process = 16
    solver_config = SolverConfig(cls=SequentialMetasolver,
                                 kwargs=dict(list_subbricks=[
                                     SubBrick(cls=TwoStepMultibatchingSolver,
                                              kwargs={"flow_solver":
                                                          SubBrick(cls=CpsatMultibatchingSolver,
                                                                   kwargs={"time_limit": 2000,
                                                                           "modeling": ModelingMultiBatch.FLOW,
                                                                           "parameters_cp": p,
                                                                           "restrict_to_shortest_paths": True,
                                                                           "prevent_incoming_at_source": True,
                                                                           "shortest_path_tolerance": 1.2,
                                                                           "add_lb_constraint_nb_trips": True,
                                                                           "detailed_capacity_constraint": False,
                                                                           "ortools_cpsat_solver_kwargs":
                                                                           {"log_search_progress": True}}),
                                                      "packing_solver":
                                                          SubBrick(cls=GreedyPackingForMultibatching,
                                                                   kwargs={})}),
                                     #SubBrick(GurobiMultibatchingSolverUnitFlow,
                                     #         dict(time_limit=30,
                                     #              delta_to_solution=2),
                                     #         kwargs_from_solution={"solution": lambda sol: sol}
                                     #         )
                                 ]
                                 )
                                 )
    problem = parse_problem(False, False, scale_co2=1/1000000)
    solver = solver_config.cls(problem, **solver_config.kwargs)
    solver.init_model(**solver_config.kwargs)
    res = solver.solve(**solver_config.kwargs)
    sol = res[-1][0]
    print(problem.evaluate(sol), problem.satisfy(sol))

def run_cpsat_lexico():
    from multibatching.solvers.cpsat import CpsatMultibatchingSolver, ModelingMultiBatch
    from discrete_optimization.generic_tools.lexico_tools import LexicoSolver
    from discrete_optimization.generic_tools.cp_tools import ParametersCp
    from discrete_optimization.generic_tools.callbacks.loggers import ProblemEvaluateLogger
    from discrete_optimization.generic_tools.callbacks.warm_start_callback import WarmStartCallbackLastRun
    problem = parse_problem(False,
                            True,
                            scale_co2=1/1000)
    p = ParametersCp.default_cpsat()
    p.nb_process = 8
    kwargs_subsolver = {"time_limit": 200,
                        "modeling": ModelingMultiBatch.FLOW,
                        "parameters_cp": p,
                        "restrict_to_shortest_paths": True,
                        "prevent_incoming_at_source": True,
                        "shortest_path_tolerance": 1.2,
                        "scaling_factor": 1,
                        "add_lb_constraint_nb_trips": True,
                        "detailed_capacity_constraint": False,
                        "ortools_cpsat_solver_kwargs":
                        {"log_search_progress": True}}
    subsolver = CpsatMultibatchingSolver(problem=problem, **kwargs_subsolver)
    lexico_subbrick = SubBrick(cls=LexicoSolver,
                               kwargs=dict(subsolver=subsolver,
                                           objectives=["emission",
                                                       "transport"],
                                           subsolver_callbacks=
                                           [ProblemEvaluateLogger(step_verbosity_level=logging.INFO,
                                                                  end_verbosity_level=logging.INFO)],
                                           callbacks=[WarmStartCallbackLastRun()],
                                           **kwargs_subsolver))
    solver_config = SolverConfig(cls=SequentialMetasolver,
                                 kwargs=dict(list_subbricks=[
                                     SubBrick(cls=TwoStepMultibatchingSolver,
                                              kwargs={"flow_solver":
                                                      lexico_subbrick,
                                                      "packing_solver":
                                                          SubBrick(cls=PackingViaBinPacking,
                                                                   kwargs={"bin_packing_solver":
                                                                               SubBrick(cls=CpSatBinPackSolver,
                                                                                        kwargs=dict(
                                                                                            modeling=ModelingBinPack.SCHEDULING,
                                                                                            parameters_cp=ParametersCp.default_cpsat()))
                                                                           })}


                                              )
                                 ]
                                 )
                                 )
    solver = solver_config.cls(problem, **solver_config.kwargs)
    solver.init_model(**solver_config.kwargs)
    res = solver.solve(**solver_config.kwargs)
    sol = res[-1][0]
    print(problem.evaluate(sol),
          problem.satisfy(sol))


def run_gurobi():
    from multibatching.solvers.cpsat import CpsatMultibatchingSolver, ModelingMultiBatch
    from discrete_optimization.generic_tools.cp_tools import ParametersCp
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    solver_config = SolverConfig(cls=SequentialMetasolver,
                                 kwargs=dict(list_subbricks=[
                                     SubBrick(cls=TwoStepMultibatchingSolver,
                                              kwargs={"flow_solver":
                                                          SubBrick(cls=GurobiMultibatchingSolver,
                                                                   kwargs={"time_limit": 2000,
                                                                           "scaling_factor": 1,
                                                                           "restrict_to_shortest_paths": True,
                                                                           "prevent_incoming_at_source": True,
                                                                           "shortest_path_tolerance": 1.1,
                                                                           "add_lb_constraint_nb_trips": True}),
                                                      "packing_solver":
                                                          SubBrick(cls=PackingViaBinPacking,
                                                                   kwargs={"bin_packing_solver":
                                                                               SubBrick(cls=CpSatBinPackSolver,
                                                               kwargs=dict(modeling=ModelingBinPack.SCHEDULING,
                                                                           parameters_cp=ParametersCp.default_cpsat()))
                                                                           })}
                                              )
                                 ]
                                 )
                                 )
    problem = parse_problem(False, False)
    solver = solver_config.cls(problem, **solver_config.kwargs)
    solver.init_model(**solver_config.kwargs)
    res = solver.solve(**solver_config.kwargs)
    sol = res[-1][0]
    print(problem.evaluate(sol), problem.satisfy(sol))

def run_gurobi_unit():
    from multibatching.solvers.cpsat import CpsatMultibatchingSolver, ModelingMultiBatch
    from discrete_optimization.generic_tools.cp_tools import ParametersCp
    p = ParametersCp.default_cpsat()
    p.nb_process = 10
    solver_config = SolverConfig(cls=SequentialMetasolver,
                                 kwargs=dict(list_subbricks=[SubBrick(cls=GurobiMultibatchingSolverUnitFlow,
                                                                   kwargs={"time_limit": 2000,
                                                                           "restrict_to_shortest_paths": True,
                                                                           "prevent_incoming_at_source": True,
                                                                           "noRelHeurTime": 30,
                                                                           "shortest_path_tolerance": 1.0})
                                     #SubBrick(GurobiMultibatchingSolverUnitFlow,
                                     #         dict(time_limit=30,
                                     #              delta_to_solution=2),
                                     #         kwargs_from_solution={"solution": lambda sol: sol}
                                     #         )
                                 ]
                                 )
                                 )
    problem = parse_problem(False, False)
    solver = solver_config.cls(problem, **solver_config.kwargs)
    solver.init_model(**solver_config.kwargs)
    res = solver.solve(**solver_config.kwargs)
    sol = res[-1][0]
    print(problem.evaluate(sol), problem.satisfy(sol))


if __name__ == "__main__":
    run_cpsat()