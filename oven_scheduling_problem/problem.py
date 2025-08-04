# osp_model.py

import re
from typing import Dict, List, Set, Tuple

# Imports from discrete-optimization library
from discrete_optimization.generic_tools.do_problem import (
    EncodingRegister,
    ModeOptim,
    ObjectiveDoc,
    ObjectiveHandling,
    ObjectiveRegister,
    Problem,
    Solution,
    TypeAttribute,
    TypeObjective,
)


class OvenSchedulingSolution(Solution):
    """
    Represents a solution to the Oven Scheduling Problem.

    The core of the solution is the `batches_per_machine` attribute, which defines
    the composition of batches and their processing sequence on each machine.

    The `timed_schedule` attribute is a cache to store the results of expensive
    timing calculations, avoiding re-computation on subsequent evaluations.
    """

    def __init__(self, problem: "OvenSchedulingProblem",
                 batches_per_machine: List[List[List[int]]]):
        """
        Initializes a solution.

        Args:
            problem (OvenSchedulingProblem): The problem instance.
            batches_per_machine (List[List[List[int]]]): Defines the batching and sequencing.
                - The **outer list** is indexed by machine_id.
                - The **middle list** is the ordered sequence of batches on that machine.
                - The **inner list** contains the job_ids for a single batch.
        """
        self.problem = problem
        self.batches_per_machine = batches_per_machine
        self.timed_schedule: Dict[Tuple[int, int], Dict] = {}  # Cache for computed timings

    def copy(self) -> "OvenSchedulingSolution":
        """Creates a deep copy of the solution."""
        new_batches = [[list(batch) for batch in machine] for machine in self.batches_per_machine]
        return OvenSchedulingSolution(self.problem, new_batches)

    def change_problem(self, new_problem: "Problem") -> None:
        """Updates the problem instance and clears the timing cache."""
        self.problem = new_problem
        self.timed_schedule = {}

    def get_job_completion_times(self):
        if not self.timed_schedule: self.problem.compute_and_get_details(self, use_cache=True)
        job_times = {}
        for (m, b), details in self.timed_schedule.items():
            if m < len(self.batches_per_machine) and b < len(self.batches_per_machine[m]):
                for job_id in self.batches_per_machine[m][b]:
                    job_times[job_id] = details
        return job_times


class OvenSchedulingProblem(Problem):
    """Defines an instance of the Oven Scheduling Problem (OSP) and its evaluation logic."""

    def __init__(self, n_jobs: int, n_machines: int,
                 n_attributes: int, job_sizes: list[int],
                 job_attributes: list[int],
                 min_times: list[int], max_times: list[int],
                 earliest_starts: list[int],
                 latest_ends: list[int],
                 eligible_machines: list[set[int]],
                 machine_capacities: list[int],
                 initial_states: list[int],
                 availability_interval: list[list[tuple[int, int]]],
                 setup_costs: list[list[int]],
                 setup_times: list[list[int]]):
        """Initializes the problem with all its parameters."""
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.n_attributes = n_attributes
        self.job_sizes = job_sizes
        self.job_attributes = job_attributes
        self.min_times = min_times
        self.max_times = max_times
        self.earliest_starts = earliest_starts
        self.latest_ends = latest_ends
        self.eligible_machines = eligible_machines
        self.machine_capacities = machine_capacities
        self.initial_states = initial_states
        self.availability_intervals = availability_interval
        self.setup_costs = setup_costs
        self.setup_times = setup_times

    def get_solution_type(self) -> type[Solution]:
        """Returns the solution class type."""
        return OvenSchedulingSolution

    def get_objective_register(self) -> ObjectiveRegister:
        """Defines the problem's objectives."""
        if "additional_data" in self.__dict__.keys():
            add = self.additional_data
            return ObjectiveRegister(
                objective_sense=ModeOptim.MINIMIZATION,
                objective_handling=ObjectiveHandling.AGGREGATE,
                dict_objective_to_doc={
                    "processing_time": ObjectiveDoc(type=TypeObjective.OBJECTIVE,
                                                    default_weight=add["weight_processing"]),
                    "nb_late_jobs": ObjectiveDoc(type=TypeObjective.OBJECTIVE,
                                                 default_weight=add["weight_tardiness"]),
                    "setup_cost": ObjectiveDoc(type=TypeObjective.OBJECTIVE,
                                               default_weight=add["weight_setup_cost"])
                }
            )
        else:
            return ObjectiveRegister(
                objective_sense=ModeOptim.MINIMIZATION,
                objective_handling=ObjectiveHandling.AGGREGATE,
                dict_objective_to_doc={
                    "processing_time": ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=1),
                    "nb_late_jobs": ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=1),
                    "setup_cost": ObjectiveDoc(type=TypeObjective.OBJECTIVE, default_weight=1)
                }
            )

    def get_attribute_register(self) -> EncodingRegister:
        """Defines the solution's encoding."""
        return EncodingRegister(dict_attribute_to_type={
            "batches_per_machine": {"type": [TypeAttribute.VRP_PATHS]}
        })

    def satisfy(self, variable: OvenSchedulingSolution) -> bool:
        """
        Checks if a solution is feasible by verifying static and temporal constraints.

        If a timed_schedule is already cached in the solution, this function will
        validate the cached timings. Otherwise, it will attempt to compute them.
        """
        # 1. Perform all static checks first (job uniqueness, attributes, capacity, etc.)
        all_jobs = set(range(self.n_jobs))
        scheduled_jobs = set()
        for m_idx, machine_batches in enumerate(variable.batches_per_machine):
            for batch in machine_batches:
                if not batch:
                    continue
                if not set(batch).isdisjoint(scheduled_jobs):
                    #print("scheduled several time")
                    return False
                scheduled_jobs.update(batch)

                first_attr = self.job_attributes[batch[0]]
                if sum(self.job_sizes[j] for j in batch) > self.machine_capacities[m_idx]:
                    #print("Capacities broken")
                    return False
                for job_idx in batch:
                    if self.job_attributes[job_idx] != first_attr:
                        #print("Several attr")
                        return False
                    if m_idx not in self.eligible_machines[job_idx]:
                        #print("incompatible machine")
                        return False
                if max(self.min_times[j] for j in batch) > min(self.max_times[j] for j in batch):
                    #print("Duration incompatible")
                    return False
        if scheduled_jobs != all_jobs:
            #print("Not all jobs scheduled")
            return False

        # 2. Check temporal constraints
        if variable.timed_schedule:
            # If timings are already cached, validate them directly.
            return self._validate_timed_schedule(variable)
        else:
            # Otherwise, try to compute the timings.
            try:
                self.compute_and_get_details(variable, use_cache=False)
                return True
            except RuntimeError:
                return False

    def _validate_timed_schedule(self, variable: OvenSchedulingSolution) -> bool:
        """
        Helper function to validate a pre-existing timed_schedule against all
        temporal constraints like availability, setups, and overlaps.
        """
        for m_idx, machine_batches in enumerate(variable.batches_per_machine):
            last_completion_time = 0
            last_attribute = self.initial_states[m_idx]

            for b_idx, batch in enumerate(machine_batches):
                if not batch:
                    continue

                details = variable.timed_schedule.get((m_idx, b_idx))
                if not details: return False  # Malformed cache

                start_time = details['start_time']
                completion_time = details['completion_time']

                # Check 1: Batch must start after its jobs are released.
                release_date = max(self.earliest_starts[j] for j in batch)
                if start_time < release_date:
                    # print("too early")
                    return False

                # Check 2: Batch must respect setup time from the previous batch.
                batch_attr = self.job_attributes[batch[0]]
                expected_setup_time = self.setup_times[last_attribute][batch_attr]
                if start_time < last_completion_time + expected_setup_time:
                    print("Setup machine ", m_idx, "batch", b_idx)
                    print("setup time not respected")
                    return False

                # Check 3: The entire batch processing must fit within a single availability window.
                is_in_valid_window = any(
                    avail_start <= start_time and completion_time <= avail_end
                    for avail_start, avail_end in self.availability_intervals[m_idx]
                )
                if not is_in_valid_window:
                    print("calendar not respected")
                    return False

                # Update state for the next batch in the sequence
                last_completion_time = completion_time
                last_attribute = batch_attr

        return True

    def evaluate(self, variable: OvenSchedulingSolution) -> Dict[str, float]:
        """Evaluates a solution's performance."""
        try:
            timed_details = self.compute_and_get_details(variable, use_cache=True)

            processing_time = sum(details['processing_time'] for details in timed_details.values())
            sc = sum(details['setup_cost'] for details in timed_details.values())

            tardy_jobs = 0
            for (m_idx, b_idx), details in timed_details.items():
                for job_idx in variable.batches_per_machine[m_idx][b_idx]:
                    if details['completion_time'] > self.latest_ends[job_idx]:
                        tardy_jobs += 1

            return {"processing_time": processing_time, "nb_late_jobs": float(tardy_jobs),
                    "setup_cost": sc}

        except RuntimeError:
            return {"processing_time": float("inf"), "nb_late_jobs": float("inf"),
                    "setup_cost": float("inf")}

    def compute_and_get_details(self, variable: OvenSchedulingSolution, use_cache: bool) -> Dict:
        """
        Internal helper to compute or retrieve schedule timings.
        """
        if use_cache and variable.timed_schedule:
            return variable.timed_schedule
        computed_details = {}
        for m_idx, machine_batches in enumerate(variable.batches_per_machine):
            last_completion_time, last_attribute = 0, self.initial_states[m_idx]

            for b_idx, batch in enumerate(machine_batches):
                if not batch:
                    continue

                batch_attr = self.job_attributes[batch[0]]
                processing_time = max(self.min_times[j] for j in batch)
                setup_time = self.setup_times[last_attribute][batch_attr]

                release_date = max(self.earliest_starts[j] for j in batch)
                earliest_start = max(last_completion_time + setup_time, release_date)

                start_time = -1
                for start_avail, end_avail in self.availability_intervals[m_idx]:
                    potential_start = max(earliest_start, start_avail)
                    if potential_start + processing_time <= end_avail:
                        start_time = potential_start
                        break

                if start_time == -1:
                    raise RuntimeError(f"No valid time slot for batch {batch} on machine {m_idx}.")

                computed_details[(m_idx, b_idx)] = {
                    'start_time': start_time,
                    'completion_time': start_time + processing_time,
                    'processing_time': processing_time,
                    'setup_cost': self.setup_costs[last_attribute][batch_attr]
                }
                last_completion_time = computed_details[(m_idx, b_idx)]['completion_time']
                last_attribute = batch_attr

        variable.timed_schedule = computed_details
        return computed_details


def parse_dat_file(file_path: str) -> OvenSchedulingProblem:
    """
    Parses a .dat file and returns an OvenSchedulingProblem instance.
    This version includes robust error handling.
    """
    with open(file_path, 'r') as f:
        content = f.read()

    def _robust_search_and_eval(name: str, is_int: bool = False):
        """Generic helper to find a value with a regex and raise a clear error on failure."""
        # The pattern uses \s* for optional whitespace and a non-greedy capture (.+?)
        # to correctly handle nested lists and multiline values.
        full_pattern = rf"{name}\s*=\s*(.+?);"
        match = re.search(full_pattern, content, re.DOTALL)

        if match is None:
            raise ValueError(f"Parser error: Could not find field '{name}' in '{file_path}'.")

        # The captured string is evaluated as Python code.
        value = eval(match.group(1))
        return int(value) if is_int else value

    try:
        n_jobs = _robust_search_and_eval("nJobs", is_int=True)
        n_machines = _robust_search_and_eval("nMachines", is_int=True)
        n_attributes = _robust_search_and_eval("nAttributes", is_int=True)

        upper_bound_integer_objective = _robust_search_and_eval("upper_bound_integer_objective",
                                                                is_int=True)
        mult_factor_total_runtime = _robust_search_and_eval("mult_factor_total_runtime",
                                                                is_int=True)
        mult_factor_finished_toolate = _robust_search_and_eval("mult_factor_finished_toolate",
                                                                is_int=True)
        mult_factor_total_setuptimes = _robust_search_and_eval("mult_factor_total_setuptimes",
                                                                is_int=True)
        mult_factor_total_setupcosts = _robust_search_and_eval("mult_factor_total_setupcosts",
                                                                is_int=True)
        running_time_bound = _robust_search_and_eval("upper_bound_integer_objective",
                                                     is_int=True)

        setup_costs_raw = _robust_search_and_eval("SetupCosts")
        setup_times_raw = _robust_search_and_eval("SetupTimes")
        setup_costs = setup_costs_raw[1:]
        setup_times = setup_times_raw[1:]

        shift_starts = _robust_search_and_eval("ShiftStartTimes")
        shift_ends = _robust_search_and_eval("ShiftEndTimes")
        avail_intervals = [[(s, e) for s, e in zip(ss, ee) if e > s] for ss, ee in zip(shift_starts, shift_ends)]

        eligible_machines_raw = _robust_search_and_eval("EligibleMachines")

        args = (
            n_jobs, n_machines, n_attributes,
            _robust_search_and_eval("JobSize"),
            [x-1 for x in _robust_search_and_eval("Attribute")],
            _robust_search_and_eval("MinTime"),
            _robust_search_and_eval("MaxTime"),
            _robust_search_and_eval("EarliestStart"),
            _robust_search_and_eval("LatestEnd"),
            [set([x-1 for x in s]) for s in eligible_machines_raw],
            _robust_search_and_eval("MaxCap"),
            [x-1 for x in _robust_search_and_eval("initState")],
            avail_intervals, setup_costs, setup_times
        )
        problem = OvenSchedulingProblem(*args)
        additional_data = {"ub": upper_bound_integer_objective,
                           "weight_tardiness": mult_factor_finished_toolate,
                           "weight_processing": mult_factor_total_runtime,
                           "weight_setup_cost": mult_factor_total_setupcosts,
                           "running_time_ub": running_time_bound}
        problem.additional_data = additional_data
        return problem

    except (ValueError, SyntaxError, NameError) as e:
        print(f"--- PARSER FAILED ---")
        print(f"An error occurred while parsing '{file_path}'. Please check the file format.")
        print(f"Details: {e}")
        raise


if __name__ == '__main__':
    import os
    folder = "/Users/poveda_g/Documents/git_repos/osp-ls/instances/use-case-1/"
    dat_file = os.path.join(folder, '99RandomOvenSchedulingInstance-n250-k5-a5--2212-22.52.57.dat')
    try:
        problem = parse_dat_file(dat_file)
        print(f"✅ Problem parsed successfully from {dat_file}.")

        # Create a naive solution for demonstration
        batches_per_machine = [[] for _ in range(problem.n_machines)]
        for j in range(problem.n_jobs):
            m_idx = list(problem.eligible_machines[j])[0]
            batches_per_machine[m_idx].append([j])

        solution = problem.get_solution_type()(problem, batches_per_machine)

        print("\n--- Evaluating Naive Solution ---")
        problem.compute_and_get_details(variable=solution, use_cache=True)

        if problem.satisfy(solution):
            print("Feasibility: ✅ Yes")
            results = problem.evaluate(solution)
            print(f"Objectives: processing_time={results['processing_time']:.0f}, "
                  f"nb late task={results['nb_late_jobs']:.0f}, setup cost={results['setup_cost']:.0f}")
        else:
            print("Feasibility: ❌ No")
    except Exception as e:
        print(f"An error occurred: {e}")