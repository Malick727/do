from discrete_optimization.rcpsp.rcpsp_model import RCPSPModel, RCPSPSolution

from jobshop.jobshop_model import JobShopProblem, JobShopSolution


def jobshop_to_rcpsp(jobshop_problem: JobShopProblem):
    tasks_names = [
        (i, j)
        for i in range(len(jobshop_problem.jobs))
        for j in range(len(jobshop_problem.jobs[i]))
    ] + ["source", "sink"]
    machine_ids = set(
        [subjob["machine_id"] for job in jobshop_problem.jobs for subjob in job]
    )
    resource_list = list(machine_ids)

    successors = {t: [] for t in tasks_names}
    for i in range(len(jobshop_problem.jobs)):
        successors["source"] += [(i, 0)]
        successors[(i, len(jobshop_problem.jobs[i]) - 1)] += ["sink"]
        for j in range(len(jobshop_problem.jobs[i]) - 1):
            successors[(i, j)] += [(i, j + 1)]
    mode_details = {t: {1: {m: 0 for m in resource_list}} for t in tasks_names}
    for i in range(len(jobshop_problem.jobs)):
        for j in range(len(jobshop_problem.jobs[i])):
            mode_details[(i, j)][1]["duration"] = jobshop_problem.jobs[i][j][
                "processing_time"
            ]
            mode_details[(i, j)][1][jobshop_problem.jobs[i][j]["machine_id"]] = 1
    mode_details["source"][1]["duration"] = 0
    mode_details["sink"][1]["duration"] = 0
    return RCPSPModel(
        resources={r: 1 for r in resource_list},
        non_renewable_resources=[],
        mode_details=mode_details,
        successors=successors,
        horizon=jobshop_problem.horizon,
        horizon_multiplier=1,
        tasks_list=tasks_names,
        source_task="source",
        sink_task="sink",
    )


def run():
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
    model = jobshop_to_rcpsp(prob)

    dummy_sol = model.get_dummy_solution()
    from discrete_optimization.rcpsp.rcpsp_solvers import LS_SOLVER, LS_RCPSP_Solver

    print(dummy_sol)
    import logging

    logging.basicConfig(level=logging.DEBUG)
    solver = LS_RCPSP_Solver(model=model, ls_solver=LS_SOLVER.SA)
    results = solver.solve(nb_iteration_max=1000)
    from discrete_optimization.rcpsp.rcpsp_utils import (
        plot_ressource_view,
        plot_task_gantt,
        plt,
    )

    print(model)
    plot_ressource_view(rcpsp_model=model, rcpsp_sol=results.get_best_solution())
    plot_task_gantt(rcpsp_model=model, rcpsp_sol=results.get_best_solution())
    plt.show()


if __name__ == "__main__":
    run()
