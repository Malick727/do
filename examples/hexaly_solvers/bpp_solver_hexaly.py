#  Copyright (c) 2025 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from discrete_optimization.binpack.parser import get_data_available_bppc, parse_bin_packing_constraint_file
from hexaly_solvers.binpack.hexaly_bpp_solver import HexalyBinPackSolver
from discrete_optimization.generic_tools.callbacks.stats_retrievers import (BasicStatsCallback,
                                                                            StatsWithBoundsCallback)
import logging
logging.basicConfig(level=logging.INFO)


def run_hexaly():
    f = [ff for ff in get_data_available_bppc()
         if "BPPC_4_1_9.txt" in ff][0]
    problem = parse_bin_packing_constraint_file(f)
    solver = HexalyBinPackSolver(problem=problem)
    solver.init_model()
    clb = StatsWithBoundsCallback()
    res = solver.solve(time_limit=5, callbacks=[clb])
    sol = res[-1][0]
    print(clb.stats)
    print(problem.evaluate(sol))
    print(problem.satisfy(sol))


if __name__ == "__main__":
    run_hexaly()