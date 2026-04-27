import logging

import numpy as np
from discrete_optimization.binpack.solvers.cpsat import CpSatBinPackSolver, ModelingBinPack
import os, sys
this_folder = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(this_folder, "../../")) 

from discrete_optimization.generic_tools.hyperparameters.hyperparameter import SubBrick
from discrete_optimization.generic_tools.sequential_metasolver import SequentialMetasolver
from discrete_optimization.generic_tools.study import SolverConfig
from multibatching.solvers.packing_subproblem import (GreedyPackingForMultibatching,
                                                      PackingViaBinPacking)
from multibatching.solvers.two_steps import TwoStepMultibatchingSolver
from examples_do_usecases.multibatching.solve_industrial_use_case import parse_problem
import logging
logging.basicConfig(level=logging.DEBUG)

def run_netx():
    from multibatching.solvers.netx import NetxMultibatchingSolver
    from discrete_optimization.generic_tools.cp_tools import ParametersCp
    problem = parse_problem(False, True, scale_co2=1/1000000)

    p = ParametersCp.default_cpsat()
    p.nb_process = 16
    config = []
    for weight_on_transport in [0, 1, 10, 100]:
        for weight_on_emission in [0, 1, 10, 100]:
            if weight_on_emission+weight_on_transport == 0:
                continue
            for restrict in [True, False]:
                if restrict:
                    for sp in np.arange(0, 0.1, 0.1):
                        kw = {"restrict_to_shortest_paths": restrict, "shortest_path_tolerance": sp,
                              "weight_on_transport": weight_on_transport, "weight_on_emission": weight_on_emission}
                        config.append(kw)
                else:
                    config.append({"restrict_to_shortest_paths": restrict,
                                   "weight_on_transport": weight_on_transport,
                                   "weight_on_emission": weight_on_emission})
    solver_configs = [SolverConfig(cls=SequentialMetasolver,
                                                     kwargs=dict(list_subbricks=[
                                                         SubBrick(cls=TwoStepMultibatchingSolver,
                                                                  kwargs={"flow_solver":
                                                                              SubBrick(cls=NetxMultibatchingSolver,
                                                                                       kwargs=kw),
                                                                          "packing_solver":
                                                                              SubBrick(cls=GreedyPackingForMultibatching,
                                                                                       kwargs={})})
                                                     ]
                                                     )
                                                     )
                      for kw in config]
    fits = []
    best_fit = float("inf")
    best_algo = None
    for solver_config in solver_configs:
        solver = solver_config.cls(problem, **solver_config.kwargs)
        solver.init_model(**solver_config.kwargs)
        res = solver.solve(**solver_config.kwargs)
        sol = res[-1][0]
        fit = res[-1][1]
        if fit < best_fit:
            best_algo = solver_config.kwargs["list_subbricks"][0].kwargs["flow_solver"].kwargs
            best_fit = fit
            print("Best fit ", best_fit, best_algo)
        fits.append(fit)
   # {'restrict_to_shortest_paths': True, 'shortest_path_tolerance': np.float64(0.0), 'weight_on_transport': 0,
   #  'weight_on_emission': 10}
   # 2489903.098801944

    print(best_algo, best_fit)

if __name__ == "__main__":
    run_netx()