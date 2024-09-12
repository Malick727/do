from discrete_optimization.generic_tools.do_solver import SolverDO
from satellite_path_planning.spp_model import SppProblem


class SppSolver(SolverDO):
    problem: SppProblem
