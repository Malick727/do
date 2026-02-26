from typing import Optional

from time import perf_counter

from discrete_optimization.generic_tools.callbacks.stats_retrievers import BasicStatsCallback
from discrete_optimization.generic_tools.do_solver import SolverDO, BoundsProviderMixin
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage


class StatsCallbackForExpe(BasicStatsCallback):
    metric_columns = BasicStatsCallback.metric_columns + [#"obj",
                                                          #"bound",
                                                          "abstime"]
    def __init__(self):
        super().__init__()
        self.first_iter_starting_time = perf_counter()

    def on_solve_start(self, solver: SolverDO):
        super().on_solve_start(solver)
        self.starting_time = self.first_iter_starting_time

    def on_step_end(
                self, step: int, res: ResultStorage, solver: SolverDO
        ) -> Optional[bool]:
            #if not isinstance(solver, BoundsProviderMixin):
            #    raise ValueError(
            #        "The StatsWithBoundsCallback can be applied only to a solver deriving from BoundsProviderMixin."
            #    )
            super().on_step_end(step=step, res=res, solver=solver)
            t = perf_counter()

            self.stats[-1].update(
                    {
                        #"obj": solver.get_current_best_internal_objective_value(),
                        #"bound": solver.get_current_best_internal_objective_bound(),
                        "abstime": t,
                    }
            )
            if len(res) > 0:
                self.stats[-1].update(solver.problem.evaluate(res[-1][0]))
            #self.stats[-1].update(res[-1][0]._intern_obj)
            #if len(res) > 0:
            #    sol = res[-1][0]
            #    ev = solver.problem.evaluate(sol)
            #    for k in ev:
            #        self.stats[-1][k] = ev[k]

