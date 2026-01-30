from optal_py_solvers.optal_rcpsp import OptalRcpspSolver
from discrete_optimization.rcpsp.problem import RcpspProblem, RcpspSolution
from discrete_optimization.rcpsp.parser import get_data_available, parse_file
from discrete_optimization.generic_tools.cp_tools import ParametersCp
import optalcp as cp


def main():
    file = [f for f in get_data_available() if "j301_1.sm" in f][0]
    problem = parse_file(file)
    solver = OptalRcpspSolver(problem)
    solver.init_model()
    p = ParametersCp.default_cpsat()
    fds_worker = cp.WorkerParameters(
        searchType="FDS",
        noOverlapPropagationLevel=4,
        cumulPropagationLevel=3,
        reservoirPropagationLevel=2
    )
    fdslb_worker = cp.WorkerParameters(
        searchType="FDSLB",
        noOverlapPropagationLevel=4,
        cumulPropagationLevel=3,
        reservoirPropagationLevel=2
    )
    res = solver.solve(parameters_cp=p,
                       time_limit=10,
                       workers=[fds_worker, fdslb_worker])
    sol, fit = res[-1]
    print(problem.evaluate(sol), problem.satisfy(sol))
    print(solver.status_solver)


if __name__ == "__main__":
    main()