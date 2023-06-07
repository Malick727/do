from typing import ClassVar, Optional

from discrete_optimization.generic_tools.do_problem import (
    ParamsObjectiveFunction,
    build_aggreg_function_and_params_objective,
)
from discrete_optimization.generic_tools.do_solver import SolverDO
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPSolution
from discrete_optimization.rcpsp.rcpsp_solvers import (
    CP_MRCPSP_MZN,
    CP_RCPSP_MZN,
    LNS_CP_RCPSP_SOLVER,
    LNS_LP_RCPSP_SOLVER,
    LP_MRCPSP,
    LP_RCPSP,
    LargeNeighborhoodSearchScheduling,
    LS_RCPSP_Solver,
    PileSolverRCPSP,
    solve,
    solvers_map,
)
from wrapper_utils import jobshop_to_rcpsp

from jobshop.jobshop_model import JobShopProblem, JobShopSolution, TaskDetails


class RCPSPSolverForJobshop(SolverDO):
    def __init__(
        self,
        problem: JobShopProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        class_solver=None,
    ):
        self.problem = problem
        self.class_solver = class_solver
        if self.class_solver is None:
            self.class_solver = LS_RCPSP_Solver
        (
            self.aggreg_sol,
            self.aggreg_from_dict_values,
            self.params_objective_function,
        ) = build_aggreg_function_and_params_objective(
            self.problem, params_objective_function=params_objective_function
        )

    def solve(self, **kwargs) -> ResultStorage:
        dict_params = kwargs
        default_params = solvers_map[self.class_solver][1]
        for k in default_params:
            if k not in dict_params:
                dict_params[k] = default_params[k]
        rcpsp_model = jobshop_to_rcpsp(jobshop_problem=self.problem)
        results: ResultStorage = solve(
            method=self.class_solver, rcpsp_model=rcpsp_model, **dict_params
        )
        list_solution_fits = []
        for k, f in results.list_solution_fits:
            jobshop_schedule = [[] for i in range(self.problem.n_machines)]
            k: RCPSPSolution = k
            for t in k.rcpsp_schedule:
                if t not in {"source", "sink"}:
                    tt = self.problem.jobs[t[0]][t[1]]
                    machine = tt["machine_id"]
                    jobshop_schedule[machine] += [
                        TaskDetails(
                            job_id=t[0],
                            task_id=t[1],
                            start_time=k.get_start_time(t),
                            end_time=k.get_end_time(t),
                        )
                    ]
            for m in jobshop_schedule:
                m.sort(key=lambda x: x.start_time)
            jobshop_solution = JobShopSolution(
                problem=self.problem, jobshop_schedule=jobshop_schedule
            )
            fit = self.aggreg_sol(jobshop_solution)
            list_solution_fits += [(jobshop_solution, fit)]
        return ResultStorage(
            list_solution_fits=list_solution_fits,
            mode_optim=self.params_objective_function.sense_function,
        )


if __name__ == "__main__":
    jobs_problem1 = [  # task = (machine_id, processing_time).
        [
            {"machine_id": 0, "processing_time": 3},
            {"machine_id": 1, "processing_time": 2},
            {"machine_id": 2, "processing_time": 2},
        ],  # Job0
        [
            {"machine_id": 0, "processing_time": 2},
            {"machine_id": 2, "processing_time": 1},
            {"machine_id": 1, "processing_time": 4},
        ],  # Job1
        [
            {"machine_id": 1, "processing_time": 4},
            {"machine_id": 2, "processing_time": 3},
        ],  # Job2
    ]
    prob = JobShopProblem(jobs_problem1)
    import logging

    logging.basicConfig(level=logging.DEBUG)
    solver = RCPSPSolverForJobshop(problem=prob, class_solver=PileSolverRCPSP)
    result = solver.solve()
    best_sol = result.get_best_solution()
    print(best_sol)
