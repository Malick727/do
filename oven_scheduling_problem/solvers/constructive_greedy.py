# constructive_greedy.py

from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from oven_scheduling_problem.problem import OvenSchedulingProblem, OvenSchedulingSolution
from discrete_optimization.generic_tools.do_solver import SolverDO


class OvenGreedySolver(SolverDO):
    problem: OvenSchedulingProblem
    """
    A constructive heuristic that builds a solution iteratively using different
    greedy strategies. The main `solve()` method can run a portfolio of strategies
    and will return the best solution found.
    """

    def __init__(self, problem: OvenSchedulingProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)
        self.machine_states = []
        # Store weights for the composite strategy
        self.weights = self.problem.additional_data if "additional_data" in self.problem.__dict__ else {}
        self.w_p = self.weights.get("weight_processing", 1)
        self.w_t = self.weights.get("weight_tardiness", 1)
        self.w_sc = self.weights.get("weight_setup_cost", 1)

    def solve(self, list_of_jobs: Optional[List[int]] = None, strategies: Optional[List[str]] = None,
              **kwargs) -> ResultStorage:
        """
        Runs multiple greedy strategies and returns the best solution found.
        """
        if strategies is None:
            # Add "composite" to the default list of strategies to run
            strategies = ["baseline", "setup_cost", "tardiness", "processing_time", "composite"]

        strategy_map: Dict[str, Callable] = {
            "baseline": self.solve_baseline,
            "setup_cost": self.solve_setup_cost,
            "tardiness": self.solve_tardiness,
            "processing_time": self.solve_processing_time,
            "composite": self.solve_composite  # New strategy
        }
        # ... (rest of the method is the same as before)
        best_solution: Optional[OvenSchedulingSolution] = None
        best_fit: float = float("inf")
        for strategy_name in strategies:
            if strategy_name not in strategy_map:
                #print(f"Warning: Strategy '{strategy_name}' not found. Skipping.")
                continue
            #print(f"--- Running Strategy: {strategy_name} ---")
            strategy_func = strategy_map[strategy_name]
            result_storage = strategy_func(list_of_jobs=list_of_jobs, **kwargs)
            sol, fit = result_storage.get_best_solution_fit()
            if fit < best_fit:
                #print(f"  -> New best fit found: {fit:,.2f}")
                best_fit = fit
                best_solution = sol
        if best_solution is None:
            raise RuntimeError("Greedy solver could not find any solution.")
        return self.create_result_storage([(best_solution, best_fit)])

    # --- Individual Strategy Implementations ---

    def solve_baseline(self, list_of_jobs: Optional[List[int]] = None, **kwargs) -> ResultStorage:
        placement_finder = self._find_placement_workload_balancing
        job_order = self._get_job_order(list_of_jobs, "est")
        return self._construct_solution(job_order, placement_finder)

    def solve_processing_time(self, list_of_jobs: Optional[List[int]] = None, **kwargs) -> ResultStorage:
        return self.solve_baseline(list_of_jobs=list_of_jobs, **kwargs)

    def solve_setup_cost(self, list_of_jobs: Optional[List[int]] = None, **kwargs) -> ResultStorage:
        placement_finder = self._find_placement_minimize_setup_cost
        job_order = self._get_job_order(list_of_jobs, "est")
        return self._construct_solution(job_order, placement_finder)

    def solve_tardiness(self, list_of_jobs: Optional[List[int]] = None, **kwargs) -> ResultStorage:
        placement_finder = self._find_placement_minimize_completion_time
        job_order = self._get_job_order(list_of_jobs, "tardiness")
        return self._construct_solution(job_order, placement_finder)

    def solve_composite(self, list_of_jobs: Optional[List[int]] = None, **kwargs) -> ResultStorage:
        """
        New strategy that uses a weighted-cost score to balance all objectives.
        """
        placement_finder = self._find_placement_composite
        job_order = self._get_job_order(list_of_jobs, "est")
        return self._construct_solution(job_order, placement_finder)

    # --- Core Construction and Placement Logic ---

    def _construct_solution(self, job_list: List[int], placement_finder: Callable) -> ResultStorage:
        batches_per_machine: List[List[List[int]]] = [[] for _ in range(self.problem.n_machines)]
        self._init_machine_states()
        for job_id in job_list:
            best_placement = placement_finder(job_id, batches_per_machine)
            if best_placement is None:
                print(f"Warning: Could not find valid placement for job {job_id}. Skipping.")
                continue
            machine_idx, batch_idx = best_placement["machine_idx"], best_placement["batch_idx"]
            if batch_idx == -1:
                batches_per_machine[machine_idx].append([job_id])
            else:
                batches_per_machine[machine_idx][batch_idx].append(job_id)
            self._update_machine_state(machine_idx, batches_per_machine[machine_idx])
        solution = OvenSchedulingSolution(self.problem, batches_per_machine)
        solution.timed_schedule = self.problem.compute_and_get_details(solution, use_cache=False)
        fit = self.aggreg_from_sol(solution)
        return self.create_result_storage([(solution, fit)])

    def _find_placement_composite(self, job_id, current_batches) -> Optional[Dict]:
        """
        Finds the best placement using a balanced, multi-criteria approach.

        This final version uses workload-balancing as the primary criterion to
        ensure a good schedule structure, and estimated completion time as a
        secondary criterion to reduce tardiness.
        """
        # --- Phase 1: Evaluate all options to ADD to an EXISTING batch ---
        append_options = []
        for m_idx in range(self.problem.n_machines):
            if m_idx not in self.problem.eligible_machines[job_id]:
                continue

            for b_idx, batch in enumerate(current_batches[m_idx]):
                if self._is_append_feasible(job_id, batch, m_idx):
                    # Calculate both primary (workload) and secondary (tardiness) scores
                    workload = self.machine_states[m_idx]["workload"]

                    old_proc_time = max(self.problem.min_times[j] for j in batch)
                    new_proc_time = max(self.problem.min_times[j] for j in batch + [job_id])
                    delta_p = new_proc_time - old_proc_time

                    # The tardiness proxy is the batch's new estimated completion time
                    tardiness_score = self.machine_states[m_idx]["last_completion"] + delta_p

                    append_options.append({
                        "machine_idx": m_idx,
                        "batch_idx": b_idx,
                        "workload": workload,
                        "tardiness_score": tardiness_score
                    })

        if append_options:
            # Choose the best append move by sorting first by workload, then by tardiness score.
            # This finds the least busy machine, then finishes the job as early as possible.
            return min(append_options, key=lambda x: (x["workload"], x["tardiness_score"]))

        # --- Phase 2: Only if no append was possible, evaluate creating a NEW batch ---
        create_options = []
        job_attr = self.problem.job_attributes[job_id]
        for m_idx in self.problem.eligible_machines[job_id]:
            current_workload = self.machine_states[m_idx]["workload"]
            last_attr = self.machine_states[m_idx]["last_attribute"]

            delta_p = self.problem.min_times[job_id]
            delta_sc = self.problem.setup_costs[last_attr][job_attr]
            est_completion = self._estimate_completion_time(job_id, m_idx)

            # The composite score is used as a tie-breaker after workload
            composite_score = self.w_p * delta_p + self.w_sc * delta_sc + self.w_t * est_completion

            create_options.append({
                "machine_idx": m_idx,
                "batch_idx": -1,
                "workload": current_workload,
                "score": composite_score
            })

        if create_options:
            # Sort first by workload, then by the overall composite score.
            return min(create_options, key=lambda x: (x["workload"], x["score"]))

        return None
    def _get_job_order(self, list_of_jobs: Optional[List[int]], default_sort: str) -> List[int]:
        if list_of_jobs is not None:
            return list(list_of_jobs)

        if default_sort == "est":
            return sorted(range(self.problem.n_jobs), key=lambda j: self.problem.earliest_starts[j])
        elif default_sort == "tardiness":
            return sorted(range(self.problem.n_jobs), key=lambda j: self.problem.latest_ends[j])
        else:
            return list(range(self.problem.n_jobs))

    def _find_placement_workload_balancing(self, job_id, current_batches) -> Optional[Dict]:
        append_options = []
        for m_idx in range(self.problem.n_machines):
            for b_idx, batch in enumerate(current_batches[m_idx]):
                if self._is_append_feasible(job_id, batch, m_idx):
                    append_options.append({
                        "machine_idx": m_idx, "batch_idx": b_idx,
                        "score": self.machine_states[m_idx]["workload"]
                    })
        if append_options:
            return min(append_options, key=lambda x: x["score"])
        create_options = []
        for m_idx in self.problem.eligible_machines[job_id]:
            create_options.append({
                "machine_idx": m_idx, "batch_idx": -1,
                "score": self.machine_states[m_idx]["workload"]
            })
        if create_options:
            return min(create_options, key=lambda x: x["score"])
        return None

    def _find_placement_minimize_setup_cost(self, job_id, current_batches) -> Optional[Dict]:
        options = []
        for m_idx in range(self.problem.n_machines):
            for b_idx, batch in enumerate(current_batches[m_idx]):
                if self._is_append_feasible(job_id, batch, m_idx):
                    options.append({"machine_idx": m_idx, "batch_idx": b_idx, "score": 0})
        job_attr = self.problem.job_attributes[job_id]
        for m_idx in self.problem.eligible_machines[job_id]:
            last_attr = self.machine_states[m_idx]["last_attribute"]
            setup_cost = self.problem.setup_costs[last_attr][job_attr]
            options.append({"machine_idx": m_idx, "batch_idx": -1, "score": setup_cost})
        if options:
            return min(options, key=lambda x: x["score"])
        return None

    def _find_placement_minimize_completion_time(self, job_id, current_batches) -> Optional[Dict]:
        options = []
        for m_idx in self.problem.eligible_machines[job_id]:
            est_completion = self._estimate_completion_time(job_id, m_idx)
            options.append({"machine_idx": m_idx, "batch_idx": -1, "score": est_completion})
        if options:
            return min(options, key=lambda x: x["score"])
        return None

    def _init_machine_states(self):
        self.machine_states = [
            {"last_completion": 0, "last_attribute": self.problem.initial_states[m], "workload": 0}
            for m in range(self.problem.n_machines)
        ]

    def _update_machine_state(self, m_idx, machine_batches):
        last_completion, last_attr, workload = 0, self.problem.initial_states[m_idx], 0
        for batch in machine_batches:
            batch_attr = self.problem.job_attributes[batch[0]]
            proc_time = max(self.problem.min_times[j] for j in batch)
            setup_time = self.problem.setup_times[last_attr][batch_attr]
            release = max(self.problem.earliest_starts[j] for j in batch)
            start_time = max(last_completion + setup_time, release)
            last_completion = start_time + proc_time
            last_attr = batch_attr
            workload += proc_time
        self.machine_states[m_idx] = {
            "last_completion": last_completion,
            "last_attribute": last_attr,
            "workload": workload
        }

    def _estimate_completion_time(self, job_id, m_idx) -> float:
        last_completion = self.machine_states[m_idx]["last_completion"]
        last_attr = self.machine_states[m_idx]["last_attribute"]
        job_attr = self.problem.job_attributes[job_id]
        proc_time = self.problem.min_times[job_id]
        setup_time = self.problem.setup_times[last_attr][job_attr]
        release = self.problem.earliest_starts[job_id]
        start_time = max(last_completion + setup_time, release)
        return start_time + proc_time

    def _is_append_feasible(self, job_id: int, batch: List[int], m_idx: int) -> bool:
        if m_idx not in self.problem.eligible_machines[job_id]: return False
        if self.problem.job_attributes[job_id] != self.problem.job_attributes[batch[0]]: return False
        if self.problem.job_sizes[job_id] + sum(self.problem.job_sizes[j] for j in batch) > \
                self.problem.machine_capacities[m_idx]: return False
        new_batch = batch + [job_id]
        if max(self.problem.min_times[j] for j in new_batch) > min(
            self.problem.max_times[j] for j in new_batch): return False
        return True