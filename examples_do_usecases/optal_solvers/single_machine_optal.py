from optalcp import Parameters, WorkerParameters
from discrete_optimization.generic_tools.cp_tools import ParametersCp
from discrete_optimization.singlemachine.parser import get_data_available, parse_file
from optal_py_solvers.optal_singlemachine import OptalSingleMachineSolver


def run_optal():
    print(get_data_available())
    problems = parse_file(get_data_available()[1])
    print(len(problems))
    for p in problems:
        problem = p
        print(problem)
        solver = OptalSingleMachineSolver(problem)
        params_cp = ParametersCp.default_cpsat()
        fds_worker = WorkerParameters(
            searchType="FDS",
            noOverlapPropagationLevel=4,
            cumulPropagationLevel=3,
            reservoirPropagationLevel=2
        )
        fdslb_worker = WorkerParameters(
            searchType="FDSDual",
            noOverlapPropagationLevel=4,
            cumulPropagationLevel=3,
            reservoirPropagationLevel=2
        )
        res = solver.solve(parameters_cp=params_cp,
                           time_limit=10,
                           workers=[fds_worker, fdslb_worker])
        sol = res.get_best_solution()
        print(problem.evaluate(sol), problem.satisfy(sol))

if __name__ == "__main__":
    run_optal()