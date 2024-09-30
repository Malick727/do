from matplotlib import pyplot as plt
from typing import Any

from discrete_optimization.vrp.vrp_parser import get_data_available, parse_file
from discrete_optimization.vrp.vrp_model import VrpProblem2D
from hexaly_solvers.vrp.hexaly_vrp_solver import VrpHexalySolver, VrpSolution


def plot_vrp_solution(
    vrp_model: VrpProblem2D, solution: VrpSolution, ax: Any = None
) -> None:
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(10, 5))
    for vehicle in range(vrp_model.vehicle_count):
        ax.plot(
            [
                vrp_model.customers[n].x
                for n in [vrp_model.start_indexes[vehicle]]
                + solution.list_paths[vehicle]
                + [vrp_model.end_indexes[vehicle]]
            ],
            [
                vrp_model.customers[n].y
                for n in [vrp_model.start_indexes[vehicle]]
                + solution.list_paths[vehicle]
                + [vrp_model.end_indexes[vehicle]]
            ]
        )


def solve_with_hexaly():
    file = [f for f in get_data_available() if "vrp_151_12_2" in f][0]
    problem = parse_file(file_path=file)
    print(problem)
    solver = VrpHexalySolver(problem=problem)
    solver.init_model()
    res = solver.solve(timeout_seconds=10)
    sol, fit = res.get_best_solution_fit()
    sol: VrpSolution
    print(sol.lengths)
    print(fit)
    print(problem.satisfy(sol), problem.evaluate(sol))
    plot_vrp_solution(vrp_model=problem, solution=sol)
    plt.show()


if __name__ == "__main__":
    solve_with_hexaly()