import math
from collections import defaultdict
import random
from typing import Any, Iterable

from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from ortools.sat.python.cp_model import Domain, IntVar, BoolVarT, CpSolverSolutionCallback, LinearExpr, Constraint
from discrete_optimization.generic_tools.ortools_cpsat_tools import CpModel, OrtoolsCpSatSolver
from oven_scheduling_problem.problem import OvenSchedulingSolution, OvenSchedulingProblem
from enum import Enum


class OvenObjectivesEnum(Enum):
    NB_BATCH = 0
    PROCESSING_TIME = 1
    SETUP_COST = 2
    TARDINESS = 3
    MAKESPAN = 4
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from discrete_optimization.generic_tools.lns_cp import OrtoolsCpSatConstraintHandler
class CpsatOvenSolver(OrtoolsCpSatSolver, WarmstartMixin):
    problem: OvenSchedulingProblem
    hyperparameters = []

    def __init__(self, problem: OvenSchedulingProblem,
                 known_ub: int,
                 max_nb_batch: int=None,
                 **kwargs: Any):
        super().__init__(problem, **kwargs)
        self.jobs_index_for_machine_i = {i: {j for j in range(self.problem.n_jobs)
                                             if i in self.problem.eligible_machines[j]}
                                         for i in range(self.problem.n_machines)}
        self.variables = {}
        self.max_nb_batch = max_nb_batch
        if self.max_nb_batch is None:
            self.max_nb_batch = int(4 * self.problem.n_jobs / self.problem.n_machines)
        self.known_ub = known_ub

    def implements_lexico_api(self) -> bool:
        return True

    def add_lexico_constraint(self, obj: str, value: float) -> Iterable[Any]:
        print(obj, value)
        return [self.cp_model.Add(self.variables["objectives"][obj]<=int(value))]

    def set_lexico_objective(self, obj: str) -> None:
        self.cp_model.minimize(self.variables["objectives"][obj])

    def get_lexico_objectives_available(self) -> list[str]:
        return list(self.variables["objectives"].keys())

    def get_lexico_objective_value(self, obj: str, res: ResultStorage) -> float:
        sol = res[-1][0]
        eval_ = self.problem.evaluate(sol)
        if obj == OvenObjectivesEnum.TARDINESS:
            return eval_["nb_late_jobs"]
        if obj == OvenObjectivesEnum.SETUP_COST:
            return eval_["setup_cost"]
        if obj == OvenObjectivesEnum.PROCESSING_TIME:
            return eval_["processing_time"]

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> Solution:
        sol = OvenSchedulingSolution(problem=self.problem, batches_per_machine=[])
        starts_per_machines = self.variables["starts_per_machines"]
        ends_per_machines = self.variables["ends_per_machines"]
        durations_per_machines = self.variables["durations_per_machines"]
        present_per_machines = self.variables["present_per_machines"]
        present_job_per_machine = self.variables["present_job_per_machine"]
        present_job_per_machine_and_batch = self.variables["present_job_per_machine_and_batch"]
        batches_per_machines = {m: defaultdict(lambda: []) for m in range(self.problem.n_machines)}
        type_per_machines = {m: {} for m in range(self.problem.n_machines)}
        schedule = {}
        # ChecK...
        alloc_job_to_machine = defaultdict(lambda: set())
        alloc_job_and_batch = defaultdict(lambda: set())
        for i in present_job_per_machine:
            for machine in present_job_per_machine[i]:
                if cpsolvercb.value(present_job_per_machine[i][machine]):
                    alloc_job_to_machine[i].add(machine)
                    if len(alloc_job_to_machine[i]) > 1:
                        print("problem with task ", i, alloc_job_to_machine[i])
                for b in range(len(present_job_per_machine_and_batch[i][machine])):
                    if cpsolvercb.value(present_job_per_machine_and_batch[i][machine][b]):
                        alloc_job_and_batch[i].add((machine, b))
                        if len(alloc_job_and_batch[i])>1:
                            print("Problem 2 with task,", i, alloc_job_and_batch[i])
        for m in present_per_machines:
            for b in range(len(present_per_machines[m])):
                if cpsolvercb.value(present_per_machines[m][b]) == 1:
                    schedule[(m, b)] = {'start_time': cpsolvercb.value(starts_per_machines[m][b]),
                                        'completion_time': cpsolvercb.value(ends_per_machines[m][b])}
                    schedule[(m, b)]["processing_time"] = (schedule[(m, b)]["completion_time"] -
                                                           schedule[(m, b)]["start_time"])

        for job in present_job_per_machine_and_batch:
            for m in present_job_per_machine_and_batch[job]:
                for b in range(len(present_job_per_machine_and_batch[job][m])):
                    if cpsolvercb.value(present_job_per_machine_and_batch[job][m][b]):
                        batches_per_machines[m][b].append(job)
                        type_per_machines[m][b] = self.problem.job_attributes[job]
        rebuild_batch_per_machine = []
        for m in range(self.problem.n_machines):
            if m not in batches_per_machines:
                rebuild_batch_per_machine.append([])
            else:
                rebuild_batch_per_machine.append([batches_per_machines[m][b]
                                                  for b in sorted(batches_per_machines[m])])
        sol.batches_per_machine = rebuild_batch_per_machine
        for m in range(self.problem.n_machines):
            prev = self.problem.initial_states[m]
            # print(rebuild_batch_per_machine[m])
            # print(type_per_machines[m])
            # print(len(rebuild_batch_per_machine[m]))
            # print(len(type_per_machines[m]))
            # print(sorted(batches_per_machines[m]))
            for i in range(len(rebuild_batch_per_machine[m])):
                new_type = type_per_machines[m][i]
                schedule[(m, i)]["setup_cost"] = self.problem.setup_costs[prev][new_type]
                prev = new_type
        sol.timed_schedule = schedule
        print(cpsolvercb.ObjectiveValue(), cpsolvercb.BestObjectiveBound())
        print([(str(obj), cpsolvercb.value(self.variables["objectives"][obj]))
               for obj in self.variables["objectives"]])
        print("Gap to known up ", self.aggreg_from_sol(sol)/self.known_ub)
        print(self.problem.satisfy(sol))
        return sol

    def init_model(self, objectives: list[OvenObjectivesEnum],
                   weights: list[float] = None, **args: Any) -> None:
        self.cp_model = CpModel()
        self.define_machines_intervals()
        self.define_jobs_intervals()
        self.constraint_convention_machines()
        self.constraint_convention_jobs()
        self.linking_jobs_and_machines()
        self.constraint_batch_size()
        self.constraint_setup_time()
        self.define_objectives(objectives=objectives,
                               weight=weights)

    def set_warm_start(self, sol: OvenSchedulingSolution):
        """
        Adds hints to the CP-SAT model based on a provided feasible solution.

        This final version sets hints for ALL decision variables, including
        batch presence, timings, job assignments, setup costs, tardiness, and makespan.
        """
        # 1. Ensure the solution's timing details are computed.
        if not sol.timed_schedule:
            self.problem.compute_and_get_details(sol, use_cache=True)
        self.cp_model.ClearHints()
        # 2. Build a reverse map for easy lookup of a job's assignment
        job_to_assignment = {
            job_id: (m, b)
            for m, machine_batches in enumerate(sol.batches_per_machine)
            for b, batch_content in enumerate(machine_batches)
            for job_id in batch_content
        }

        # 3. Hint all batch, job, and setup variables
        for m, machine_batches in enumerate(sol.batches_per_machine):
            last_attribute = self.problem.initial_states[m]
            for b, batch_content in enumerate(machine_batches):
                if not batch_content:
                    continue
                details = sol.timed_schedule[(m, b)]
                batch_attribute = self.problem.job_attributes[batch_content[0]]

                self.cp_model.AddHint(self.variables["present_per_machines"][m][b], 1)
                self.cp_model.AddHint(self.variables["starts_per_machines"][m][b], details['start_time'])
                self.cp_model.AddHint(self.variables["ends_per_machines"][m][b], details['completion_time'])
                self.cp_model.AddHint(self.variables["durations_per_machines"][m][b], details['processing_time'])
                self.cp_model.AddHint(self.variables["present_per_machines_per_attr"][m][batch_attribute][b], 1)
                for attr in self.variables["present_per_machines_per_attr"][m]:
                    if attr != batch_attribute:
                        self.cp_model.AddHint(self.variables["present_per_machines_per_attr"][m][attr][b], 0)

                if self.variables.get("setup_cost_per_machine"):
                    self.cp_model.AddHint(self.variables["setup_cost_per_machine"][m][b],
                                          self.problem.setup_costs[last_attribute][batch_attribute])
                if self.variables.get("setup_time_per_machine"):
                    self.cp_model.AddHint(self.variables["setup_time_per_machine"][m][b],
                                          self.problem.setup_times[last_attribute][batch_attribute])

                last_attribute = batch_attribute
            if len(machine_batches) >= 1:
                last_time = sol.timed_schedule[(m, len(machine_batches)-1)]["completion_time"]
            else:
                shift = next(s for s in self.problem.availability_intervals[m]
                             if s[1]>s[0])
                last_time = shift[0]
            for b in range(len(machine_batches), self.max_nb_batch):
                self.cp_model.AddHint(self.variables["present_per_machines"][m][b], 0)
                if self.variables.get("setup_cost_per_machine"):
                    self.cp_model.AddHint(self.variables["setup_cost_per_machine"][m][b], 0)
                if self.variables.get("setup_time_per_machine"):
                    self.cp_model.AddHint(self.variables["setup_time_per_machine"][m][b], 0)
                for attr in self.variables["present_per_machines_per_attr"][m]:
                    self.cp_model.AddHint(self.variables["present_per_machines_per_attr"][m][attr][b], 0)

                self.cp_model.AddHint(self.variables["starts_per_machines"][m][b], last_time)
                self.cp_model.AddHint(self.variables["ends_per_machines"][m][b], last_time)
                self.cp_model.AddHint(self.variables["durations_per_machines"][m][b], 0)


        # 4. Hint all job-specific and tardiness variables
        for job_id in range(self.problem.n_jobs):
            if job_id not in job_to_assignment: continue

            m, b = job_to_assignment[job_id]
            details = sol.timed_schedule[(m, b)]

            # Hint job timings and assignments
            if "starts_job" in self.variables:
                self.cp_model.AddHint(self.variables["starts_job"][job_id], details['start_time'])
                self.cp_model.AddHint(self.variables["ends_job"][job_id], details['completion_time'])
                self.cp_model.AddHint(self.variables["durations_job"][job_id], details['processing_time'])
            self.cp_model.AddHint(self.variables["present_job_per_machine"][job_id][m], 1)
            self.cp_model.AddHint(self.variables["present_job_per_machine_and_batch"][job_id][m][b], 1)
            #self.cp_model.AddHint(self.variables["starts_job_per_machine"][job_id][m], details['start_time'])
            #self.cp_model.AddHint(self.variables["ends_job_per_machine"][job_id][m], details['completion_time'])
            #self.cp_model.AddHint(self.variables["durations_job_per_machine"][job_id][m], details['processing_time'])

            for m_other in self.problem.eligible_machines[job_id]:
                if m_other != m:
                    self.cp_model.AddHint(self.variables["present_job_per_machine"][job_id][m_other], 0)
                # self.cp_model.AddHint(self.variables["starts_job_per_machine"][job_id][m_other], details['start_time'])
                # self.cp_model.AddHint(self.variables["ends_job_per_machine"][job_id][m_other], details['completion_time'])
                # self.cp_model.AddHint(self.variables["durations_job_per_machine"][job_id][m_other],
                #                       details['processing_time'])
                for b_ in range(len(self.variables["present_job_per_machine_and_batch"][job_id][m])):
                    if b_ != b or m_other != m:
                        self.cp_model.AddHint(self.variables["present_job_per_machine_and_batch"][job_id][m_other][b_],
                                              0)

            # Hint TARDINESS variable (NEW - requires your code change)
            if "late_jobs" in self.variables:
                is_late = 1 if details['completion_time'] > self.problem.latest_ends[job_id] else 0
                self.cp_model.AddHint(self.variables["late_jobs"][job_id], is_late)

        # 5. Hint MAKESPAN variable (NEW)
        if OvenObjectivesEnum.MAKESPAN in self.variables["objectives"]:
            makespan_variable = self.variables["objectives"][OvenObjectivesEnum.MAKESPAN]
            solution_makespan = 0
            if sol.timed_schedule:
                solution_makespan = max(d['completion_time'] for d in sol.timed_schedule.values())
            self.cp_model.AddHint(makespan_variable, solution_makespan)
        #print(len(self.cp_model.proto.variables),
        #      len(self.cp_model.proto.solution_hint.vars))
        #print( self.cp_model.proto.solution_hint.vars)
        #vars_in_hint = [self.cp_model.proto.variables[v].name for v in self.cp_model.proto.solution_hint.vars]
        #vars = [v.name for v in self.cp_model.proto.variables]
        #print(vars, "\n", vars_in_hint)
        #missing = [v for v in vars if v not in vars_in_hint]
        #print(self.cp_model.proto.variables[327])
        #print(missing)

    def define_objectives(self, objectives: list[OvenObjectivesEnum],
                          weight: list[float] = None):
        if weight is None:
            weight = [1]*len(objectives)
        self.variables["objectives"] = {}
        for obj in objectives:
            if obj == OvenObjectivesEnum.NB_BATCH:
                nb_batch = sum([x for m in self.variables["present_per_machines"]
                                for x in self.variables["present_per_machines"][m]])
                self.variables["objectives"][obj] = nb_batch
            if obj == OvenObjectivesEnum.PROCESSING_TIME:
                processing_time = sum([x for m in self.variables["durations_per_machines"]
                                       for x in self.variables["durations_per_machines"][m]])
                self.variables["objectives"][obj] = processing_time
            if obj == OvenObjectivesEnum.SETUP_COST:
                setup_cost = sum([x for m in self.variables["setup_cost_per_machine"]
                                  for x in self.variables["setup_cost_per_machine"][m]])
                self.variables["objectives"][obj] = setup_cost
            if obj == OvenObjectivesEnum.MAKESPAN:
                horizon = max([self.problem.availability_intervals[i][-1][1]
                               for i in range(self.problem.n_machines)])
                makespan = self.cp_model.NewIntVar(lb=max(self.problem.earliest_starts),
                                                   ub=horizon, name=f"makespan")
                self.cp_model.AddMaxEquality(makespan,
                                             [self.variables["ends_job"][i] for i in self.variables["ends_job"]])
                self.variables["objectives"][obj] = makespan
            if obj == OvenObjectivesEnum.TARDINESS:
                late = {i: self.cp_model.NewBoolVar(name=f"late_{i}")
                        for i in range(self.problem.n_jobs)}
                self.variables["late_jobs"] = late
                for i in late:
                    (self.cp_model.Add(self.variables["ends_job"][i] > self.problem.latest_ends[i])
                     .OnlyEnforceIf(late[i]))
                    (self.cp_model.Add(self.variables["ends_job"][i] <= self.problem.latest_ends[i])
                     .OnlyEnforceIf(late[i].Not()))
                self.variables["objectives"][OvenObjectivesEnum.TARDINESS] = sum([late[i] for i in late])

        self.cp_model.Minimize(sum([self.variables["objectives"][objectives[i]]*weight[i]
                                    for i in range(len(objectives))]))

    def define_machines_intervals(self):
        starts_per_machines = {i: [] for i in range(self.problem.n_machines)}
        ends_per_machines = {i: [] for i in range(self.problem.n_machines)}
        durations_per_machines = {i: [] for i in range(self.problem.n_machines)}
        present_per_machines = {i: [] for i in range(self.problem.n_machines)}
        intervals_per_machines = {i: [] for i in range(self.problem.n_machines)}

        starts_per_machines_per_attr = {i: {} for i in range(self.problem.n_machines)}  # useless for now ?
        ends_per_machines_per_attr = {i: {} for i in range(self.problem.n_machines)}  # useless for now ?
        durations_per_machines_per_attr = {i: {} for i in range(self.problem.n_machines)}  # useless for now ?
        present_per_machines_per_attr = {i: {} for i in range(self.problem.n_machines)}
        intervals_per_machines_per_attr = {i: {} for i in range(self.problem.n_machines)}  # useless for now ?
        max_nb_batch = self.max_nb_batch
        for i in range(len(starts_per_machines)):
            jobs = self.jobs_index_for_machine_i[i]
            types = set([self.problem.job_attributes[x] for x in jobs])
            possible_durations = ([self.problem.min_times[x] for x in jobs] +
                                  [self.problem.max_times[x] for x in jobs])
            possible_durations = sorted(set(possible_durations))
            possible_durations = [0]+possible_durations
            calendar = self.problem.availability_intervals[i]

            starts_per_machines[i] = [self.cp_model.NewIntVarFromDomain(domain=Domain.FromIntervals(calendar),
                                                                        name=f"start_machine_{i}_batch_{b}")
                                      for b in range(max_nb_batch)]
            ends_per_machines[i] = [self.cp_model.NewIntVarFromDomain(domain=Domain.FromIntervals(calendar),
                                                                      name=f"end_machine_{i}_batch_{b}")
                                    for b in range(max_nb_batch)]
            durations_per_machines[i] = [self.cp_model.NewIntVarFromDomain(domain=Domain.FromValues(possible_durations),
                                                                           name=f"duration_machine_{i}_batch_{b}")
                                         for b in range(max_nb_batch)]
            present_per_machines[i] = [self.cp_model.NewBoolVar(name=f"present_machine_{i}_batch_{b}")
                                       for b in range(max_nb_batch)]
            intervals_per_machines[i] = [self.cp_model.NewOptionalIntervalVar(start=starts_per_machines[i][b],
                                                                              size=durations_per_machines[i][b],
                                                                              end=ends_per_machines[i][b],
                                                                              is_present=present_per_machines[i][b],
                                                                              name=f"interval_machine_{i}_{b}")
                                         for b in range(max_nb_batch)]
            # create binary presence for each type
            for t in types:
                present_per_machines_per_attr[i][t] = [self.cp_model.NewBoolVar(f"present_machine_{i}_{b}_type_{t}")
                                                       for b in range(max_nb_batch)]

        self.variables["starts_per_machines"] = starts_per_machines
        self.variables["ends_per_machines"] = ends_per_machines
        self.variables["durations_per_machines"] = durations_per_machines
        self.variables["present_per_machines"] = present_per_machines
        self.variables["intervals_per_machines"] = intervals_per_machines
        self.variables["present_per_machines_per_attr"] = present_per_machines_per_attr

    def constraint_convention_machines(self):
        present_per_machines: dict[int, list[BoolVarT]] = self.variables["present_per_machines"]
        present_per_machines_per_attr: dict[int, dict[int, list[BoolVarT]]] = self.variables["present_per_machines_per_attr"]
        starts_per_machines: dict[int, list[IntVar]] = self.variables["starts_per_machines"]
        ends_per_machines: dict[int, list[IntVar]] = self.variables["ends_per_machines"]
        for machine_index in present_per_machines:
            for i in range(self.max_nb_batch-1):
                # self.cp_model.AddImplication(present_per_machines[machine_index][i].Not(),
                #                              present_per_machines[machine_index][i+1].Not())
                # When no batch at some point, the next ones are also absent.
                # Redundant, in case.
                self.cp_model.add(present_per_machines[machine_index][i] >= present_per_machines[machine_index][i+1])

                # Order of interval (increasing time!)
                (self.cp_model.add(starts_per_machines[machine_index][i+1] >= ends_per_machines[machine_index][i])
                 .only_enforce_if(present_per_machines[machine_index][i+1]))

            for i in range(self.max_nb_batch):
                # At most one type for each batch.
                self.cp_model.add_at_most_one([present_per_machines_per_attr[machine_index][t][i]
                                               for t in present_per_machines_per_attr[machine_index]])
                # Exactly one, when the batch is present !!
                (self.cp_model.add(sum([present_per_machines_per_attr[machine_index][t][i]
                                        for t in present_per_machines_per_attr[machine_index]]) == 1)
                 .only_enforce_if(present_per_machines[machine_index][i]))

    def define_jobs_intervals(self):
        horizon = max([self.problem.availability_intervals[i][-1][1]
                       for i in range(self.problem.n_machines)])
        starts_job = {i: None for i in range(self.problem.n_jobs)}
        ends_job = {i: None for i in range(self.problem.n_jobs)}
        durations_job = {i: None for i in range(self.problem.n_jobs)}

        starts_job_per_machine = {i: {} for i in range(self.problem.n_jobs)}
        ends_job_per_machine = {i: {} for i in range(self.problem.n_jobs)}
        durations_job_per_machine = {i: {} for i in range(self.problem.n_jobs)}
        present_job_per_machine = {i: {} for i in range(self.problem.n_jobs)}
        interval_job_per_machine = {i: {} for i in range(self.problem.n_jobs)}

        present_job_per_machine_and_batch = {i: {} for i in range(self.problem.n_jobs)}

        for i in range(self.problem.n_jobs):
            starts_job[i] = self.cp_model.NewIntVar(lb=self.problem.earliest_starts[i],
                                                    ub=horizon,
                                                    name=f"start_job_{i}")
            ends_job[i] = self.cp_model.NewIntVar(lb=self.problem.earliest_starts[i]+self.problem.min_times[i],
                                                  ub=horizon,
                                                  name=f"end_job_{i}")
            durations_job[i] = self.cp_model.NewIntVar(lb=self.problem.min_times[i],
                                                       ub=self.problem.max_times[i],
                                                       name=f"duration_job_{i}")
            for machine_index in self.problem.eligible_machines[i]:
                if False:
                    starts_job_per_machine[i][machine_index] = self.cp_model.NewIntVar(lb=self.problem.earliest_starts[i],
                                                                                       ub=horizon,
                                                                                       name=f"start_job_{i}_machine_"
                                                                                            f"{machine_index}")
                    ends_job_per_machine[i][machine_index] = self.cp_model.NewIntVar(lb=self.problem.earliest_starts[i],
                                                                                     ub=horizon,
                                                                                     name=f"end_job_{i}_machine_"
                                                                                          f"{machine_index}")
                    durations_job_per_machine[i][machine_index] = self.cp_model.NewIntVar(lb=self.problem.min_times[i],
                                                                                          ub=self.problem.max_times[i],
                                                                                          name=f"duration_job_{i}_machine_"
                                                                                               f"{machine_index}")
                if True:
                    starts_job_per_machine[i][machine_index] = starts_job[i]
                    ends_job_per_machine[i][machine_index] = ends_job[i]
                    durations_job_per_machine[i][machine_index] = durations_job[i]
                present_job_per_machine[i][machine_index] = self.cp_model.NewBoolVar(f"present_job_{i}"
                                                                                     f"_machine_{machine_index}")
                interval_job_per_machine[i][machine_index] = self.cp_model.NewOptionalIntervalVar(
                    start=starts_job_per_machine[i][machine_index],
                    end=ends_job_per_machine[i][machine_index],
                    size=durations_job_per_machine[i][machine_index],
                    is_present=present_job_per_machine[i][machine_index],
                    name=f"interval_{i}_machine_{machine_index}")
                present_job_per_machine_and_batch[i][machine_index] = [self.cp_model.NewBoolVar(f"present_job_{i}_"
                                                                                                f"machine_"
                                                                                                f"{machine_index}"
                                                                                                f"_batch_{b}")
                                                                       for b in range(self.max_nb_batch)]

        self.variables["starts_job"] = starts_job
        self.variables["ends_job"] = ends_job
        self.variables["durations_job"] = durations_job
        self.variables["starts_job_per_machine"] = starts_job_per_machine
        self.variables["ends_job_per_machine"] = ends_job_per_machine
        self.variables["durations_job_per_machine"] = durations_job_per_machine
        self.variables["present_job_per_machine"] = present_job_per_machine
        self.variables["interval_job_per_machine"] = interval_job_per_machine
        self.variables["present_job_per_machine_and_batch"] = present_job_per_machine_and_batch

    def constraint_convention_jobs(self):
        # starts_job = self.variables["starts_job"]
        # ends_job = self.variables["ends_job"]
        # self.variables["durations_job"] = durations_job
        # starts_job_per_machine = self.variables["starts_job_per_machine"]
        # ends_job_per_machine = self.variables["ends_job_per_machine"]
        # durations_job_per_machine = self.variables["durations_job_per_machine"]
        present_job_per_machine: dict[int, dict[int, BoolVarT]] = self.variables["present_job_per_machine"]
        present_job_per_machine_and_batch: dict[int, dict[int, list[BoolVarT]]] = self.variables["present_job_per_machine_and_batch"]

        for i in present_job_per_machine:
            self.cp_model.add_exactly_one([present_job_per_machine[i][m]
                                           for m in present_job_per_machine[i]])

            for m in present_job_per_machine_and_batch[i]:
                self.cp_model.add(sum(present_job_per_machine_and_batch[i][m]) <= 1)
                (self.cp_model.add(sum(present_job_per_machine_and_batch[i][m]) == 1)
                 .OnlyEnforceIf(present_job_per_machine[i][m]))
                (self.cp_model.add(sum(present_job_per_machine_and_batch[i][m]) == 0)
                 .OnlyEnforceIf(present_job_per_machine[i][m].Not()))

    def linking_jobs_and_machines(self):
        starts_job = self.variables["starts_job"]
        ends_job = self.variables["ends_job"]
        durations_job = self.variables["durations_job"]
        starts_job_per_machine = self.variables["starts_job_per_machine"]
        ends_job_per_machine = self.variables["ends_job_per_machine"]
        durations_job_per_machine = self.variables["durations_job_per_machine"]
        present_job_per_machine: dict[int, dict[int, BoolVarT]] = self.variables["present_job_per_machine"]
        present_job_per_machine_and_batch: dict[int, dict[int, list[BoolVarT]]] = (
            self.variables["present_job_per_machine_and_batch"])

        present_per_machines: dict[int, list[BoolVarT]] = self.variables["present_per_machines"]
        present_per_machines_per_attr: dict[int, dict[int, list[BoolVarT]]] = self.variables[
            "present_per_machines_per_attr"]
        starts_per_machines: dict[int, list[IntVar]] = self.variables["starts_per_machines"]
        ends_per_machines: dict[int, list[IntVar]] = self.variables["ends_per_machines"]
        durations_per_machines: dict[int, list[IntVar]] = self.variables["durations_per_machines"]

        for i in range(self.problem.n_jobs):
            type_ = self.problem.job_attributes[i]
            for m in present_job_per_machine[i]:
                for b in range(self.max_nb_batch):
                    present = present_job_per_machine_and_batch[i][m][b]
                    # -> Batch b is present
                    self.cp_model.add(present_per_machines[m][b] == 1).only_enforce_if(present)
                    # -> Job i is done on machine i
                    self.cp_model.add(present_job_per_machine[i][m] == 1).only_enforce_if(present)
                    # -> Type is activated for this batch.
                    self.cp_model.add(present_per_machines_per_attr[m][type_][b] == 1).only_enforce_if(present)

                    # Synchronisation of times/durations.
                    self.cp_model.add(starts_job[i] == starts_per_machines[m][b]).only_enforce_if(present)
                    self.cp_model.add(ends_job[i] == ends_per_machines[m][b]).only_enforce_if(present)
                    self.cp_model.add(durations_job[i] == durations_per_machines[m][b]).only_enforce_if(present)

                    #self.cp_model.add(starts_job[i] == starts_job_per_machine[i][m]).only_enforce_if(present)
                    #self.cp_model.add(ends_job[i] == ends_job_per_machine[i][m]).only_enforce_if(present)
                    #self.cp_model.add(durations_job[i] == durations_job_per_machine[i][m]).only_enforce_if(present)
        # if b-th batch of a machine is activated, some activities are put on it.
        for b in range(self.max_nb_batch):
            for m in range(self.problem.n_machines):
                (self.cp_model.add(sum([present_job_per_machine_and_batch[i][m][b]
                                        for i in present_job_per_machine if m in present_job_per_machine[i]]) >= 1)
                 .only_enforce_if(present_per_machines[m][b]))
                (self.cp_model.add(sum([present_job_per_machine_and_batch[i][m][b]
                                        for i in present_job_per_machine if m in present_job_per_machine[i]]) == 0)
                 .only_enforce_if(present_per_machines[m][b].Not()))

    def constraint_batch_size(self):
        interval_job_per_machine = self.variables["interval_job_per_machine"]
        present_job_per_machine_and_batch: dict[int, dict[int, list[BoolVarT]]] = (
            self.variables["present_job_per_machine_and_batch"])
        for m in range(self.problem.n_machines):
            intervals_and_size = [(interval_job_per_machine[i][m], self.problem.job_sizes[i])
                                  for i in interval_job_per_machine if m in interval_job_per_machine[i]]
            #self.cp_model.AddCumulative([x[0] for x in intervals_and_size],
            #                            [x[1] for x in intervals_and_size],
            #                            capacity=self.problem.machine_capacities[m])
            for b in range(self.max_nb_batch):
                literal_and_coef = [(present_job_per_machine_and_batch[i][m][b], self.problem.job_sizes[i])
                                    for i in present_job_per_machine_and_batch
                                    if m in present_job_per_machine_and_batch[i]]
                self.cp_model.add(LinearExpr.weighted_sum([x[0] for x in literal_and_coef],
                                                          [x[1] for x in literal_and_coef])
                                  <= self.problem.machine_capacities[m])

    def constraint_setup_time(self):
        starts_per_machines = self.variables["starts_per_machines"]
        ends_per_machines = self.variables["ends_per_machines"]
        present_per_machines = self.variables["present_per_machines"]
        present_per_machines_per_attr = self.variables["present_per_machines_per_attr"]
        possible_setup_times = sorted(set([0]+[x for i in range(len(self.problem.setup_times))
                                               for x in self.problem.setup_times[i]]))
        possible_setup_cost = sorted(set([0]+[x for i in range(len(self.problem.setup_costs))
                                              for x in self.problem.setup_costs[i]]))
        setup_time = {m: [self.cp_model.NewIntVarFromDomain(Domain.FromValues(possible_setup_times),
                                                            name=f"setup_machine_{m}_batch_{b}")
                          for b in range(self.max_nb_batch)]
                      for m in range(self.problem.n_machines)}
        setup_cost = {m: [self.cp_model.NewIntVarFromDomain(Domain.FromValues(possible_setup_cost),
                                                            name=f"setup_machine_{m}_batch_{b}")
                          for b in range(self.max_nb_batch)]
                      for m in range(self.problem.n_machines)}

        for machine in starts_per_machines:
            for t1 in present_per_machines_per_attr[machine]:
                self.cp_model.add(starts_per_machines[machine][0] >=
                                  self.problem.setup_times[self.problem.initial_states[machine]][t1]).only_enforce_if(
                    present_per_machines_per_attr[machine][t1][0],
                    present_per_machines[machine][0])
                self.cp_model.add(setup_time[machine][0] ==
                                  self.problem.setup_times[self.problem.initial_states[machine]][t1]).only_enforce_if(
                    present_per_machines_per_attr[machine][t1][0],
                    present_per_machines[machine][0])
                self.cp_model.add(setup_cost[machine][0] ==
                                  self.problem.setup_costs[self.problem.initial_states[machine]][t1]).only_enforce_if(
                    present_per_machines_per_attr[machine][t1][0],
                    present_per_machines[machine][0])
            for i in range(1, self.max_nb_batch):
                for t1 in present_per_machines_per_attr[machine]:
                    for t2 in present_per_machines_per_attr[machine]:
                        self.cp_model.add(starts_per_machines[machine][i] >=
                                          ends_per_machines[machine][i-1] +
                                          self.problem.setup_times[t1][t2]).only_enforce_if(
                            present_per_machines_per_attr[machine][t1][i-1],
                            present_per_machines_per_attr[machine][t2][i],
                            present_per_machines[machine][i-1],
                            present_per_machines[machine][i])

                        self.cp_model.add(setup_time[machine][i] ==
                                          self.problem.setup_times[t1][t2]).only_enforce_if(
                            present_per_machines_per_attr[machine][t1][i-1],
                            present_per_machines_per_attr[machine][t2][i],
                            present_per_machines[machine][i])
                        self.cp_model.add(setup_cost[machine][i] ==
                                          self.problem.setup_costs[t1][t2]).only_enforce_if(
                            present_per_machines_per_attr[machine][t1][i - 1],
                            present_per_machines_per_attr[machine][t2][i],
                            present_per_machines[machine][i])
            for i in range(self.max_nb_batch):
                self.cp_model.add(setup_cost[machine][i] == 0).only_enforce_if(
                    present_per_machines[machine][i].Not())

        self.variables["setup_time_per_machine"] = setup_time
        self.variables["setup_cost_per_machine"] = setup_cost

    def get_assignments_from_solution(self, sol: OvenSchedulingSolution) -> dict:
        """
        Translates a solution object into a comprehensive dictionary mapping all
        relevant variable names to their assigned values.
        """
        # 1. Ensure the solution's timing details are computed.
        if not sol.timed_schedule:
            self.problem.compute_and_get_details(sol, use_cache=True)

        # 2. Initialize the assignments dictionary to hold all variable values.
        assignments = defaultdict(lambda: defaultdict(dict))

        # 3. Build a reverse map for easy lookup of a job's assignment.
        job_to_assignment = {
            job_id: (m, b)
            for m, machine_batches in enumerate(sol.batches_per_machine)
            for b, batch_content in enumerate(machine_batches)
            for job_id in batch_content
        }

        # 4. Determine batch-related assignments for each machine.
        for m in range(self.problem.n_machines):
            last_attribute = self.problem.initial_states[m]
            num_batches_in_sol = len(sol.batches_per_machine[m])

            # Assign values for batches that ARE PRESENT in the solution
            for b in range(num_batches_in_sol):
                batch_content = sol.batches_per_machine[m][b]
                details = sol.timed_schedule[(m, b)]
                batch_attribute = self.problem.job_attributes[batch_content[0]]

                assignments["present_per_machines"][m][b] = 1
                assignments["starts_per_machines"][m][b] = details['start_time']
                assignments["ends_per_machines"][m][b] = details['completion_time']
                assignments["durations_per_machines"][m][b] = details['processing_time']

                # Assign attribute presence
                for attr in self.variables["present_per_machines_per_attr"][m]:
                    assignments["present_per_machines_per_attr"][m][attr, b] = 1 if attr == batch_attribute else 0

                # Assign setup costs and times
                assignments["setup_cost_per_machine"][m][b] = self.problem.setup_costs[last_attribute][batch_attribute]
                assignments["setup_time_per_machine"][m][b] = self.problem.setup_times[last_attribute][batch_attribute]

                last_attribute = batch_attribute

            # Assign default values for batches that ARE NOT PRESENT
            last_time = sol.timed_schedule[(m, num_batches_in_sol - 1)][
                "completion_time"] if num_batches_in_sol > 0 else 0
            for b in range(num_batches_in_sol, self.max_nb_batch):
                assignments["present_per_machines"][m][b] = 0
                assignments["starts_per_machines"][m][b] = last_time
                assignments["ends_per_machines"][m][b] = last_time
                assignments["durations_per_machines"][m][b] = 0
                for attr in self.variables["present_per_machines_per_attr"][m]:
                    assignments["present_per_machines_per_attr"][m][attr, b] = 0
                assignments["setup_cost_per_machine"][m][b] = 0
                assignments["setup_time_per_machine"][m][b] = 0

        # 5. Determine all job-specific assignments.
        for job_id in range(self.problem.n_jobs):
            if job_id in job_to_assignment:
                m, b = job_to_assignment[job_id]
                details = sol.timed_schedule[(m, b)]

                # Job timings
                assignments["starts_job"][job_id] = details['start_time']
                assignments["ends_job"][job_id] = details['completion_time']
                assignments["durations_job"][job_id] = details['processing_time']

                # Job assignments
                for m_other in self.problem.eligible_machines[job_id]:
                    assignments["present_job_per_machine"][job_id][m_other] = 1 if m_other == m else 0
                    for b_other in range(self.max_nb_batch):
                        is_assigned = 1 if m_other == m and b_other == b else 0
                        assignments["present_job_per_machine_and_batch"][job_id][m_other, b_other] = is_assigned

                # Job tardiness
                assignments["late_jobs"][job_id] = 1 if details['completion_time'] > self.problem.latest_ends[
                    job_id] else 0
            else:  # Should not happen in a valid solution
                # Assign default values for unassigned jobs
                assignments["late_jobs"][job_id] = 1

        # 6. Determine makespan.
        solution_makespan = 0
        if sol.timed_schedule:
            solution_makespan = max(d['completion_time'] for d in sol.timed_schedule.values() if d)
        assignments["makespan"] = solution_makespan

        return assignments


class CpsatOvenSolverEvolved(CpsatOvenSolver):
    problem: OvenSchedulingProblem
    hyperparameters = []

    def init_model(self, objectives: list[OvenObjectivesEnum],
                   weights: list[float] = None, **args: Any) -> None:
        self.cp_model = CpModel()
        self.define_machines_intervals()
        self.define_jobs_intervals()
        self.constraint_convention_machines()
        self.constraint_convention_jobs()
        self.linking_jobs_and_machines()
        self.constraint_batch_size()
        self.constraint_setup_time()
        self.define_objectives(objectives=objectives,
                               weight=weights)

    def define_machines_intervals(self):
        starts_per_machines = {i: [] for i in range(self.problem.n_machines)}
        ends_per_machines = {i: [] for i in range(self.problem.n_machines)}
        durations_per_machines = {i: [] for i in range(self.problem.n_machines)}
        present_per_machines = {i: [] for i in range(self.problem.n_machines)}
        intervals_per_machines = {i: [] for i in range(self.problem.n_machines)}

        present_per_machines_per_attr = {i: {} for i in range(self.problem.n_machines)}
        max_nb_batch = self.max_nb_batch
        for i in range(len(starts_per_machines)):
            jobs = self.jobs_index_for_machine_i[i]
            types = set([self.problem.job_attributes[x] for x in jobs])
            possible_durations = ([self.problem.min_times[x] for x in jobs] +
                                  [self.problem.max_times[x] for x in jobs])
            possible_durations = sorted(set(possible_durations))
            possible_durations = [0] + possible_durations
            calendar = self.problem.availability_intervals[i]

            starts_per_machines[i] = [self.cp_model.NewIntVarFromDomain(domain=Domain.FromIntervals(calendar),
                                                                        name=f"start_machine_{i}_batch_{b}")
                                      for b in range(max_nb_batch)]
            ends_per_machines[i] = [self.cp_model.NewIntVarFromDomain(domain=Domain.FromIntervals(calendar),
                                                                      name=f"end_machine_{i}_batch_{b}")
                                    for b in range(max_nb_batch)]
            durations_per_machines[i] = [self.cp_model.NewIntVarFromDomain(domain=Domain.FromValues(possible_durations),
                                                                           name=f"duration_machine_{i}_batch_{b}")
                                         for b in range(max_nb_batch)]
            present_per_machines[i] = [self.cp_model.NewBoolVar(name=f"present_machine_{i}_batch_{b}")
                                       for b in range(max_nb_batch)]
            intervals_per_machines[i] = [self.cp_model.NewOptionalIntervalVar(start=starts_per_machines[i][b],
                                                                              size=durations_per_machines[i][b],
                                                                              end=ends_per_machines[i][b],
                                                                              is_present=present_per_machines[i][b],
                                                                              name=f"interval_machine_{i}_{b}")
                                         for b in range(max_nb_batch)]
            # create binary presence for each type
            for t in types:
                present_per_machines_per_attr[i][t] = [self.cp_model.NewBoolVar(f"present_machine_{i}_{b}_type_{t}")
                                                       for b in range(max_nb_batch)]

        self.variables["starts_per_machines"] = starts_per_machines
        self.variables["ends_per_machines"] = ends_per_machines
        self.variables["durations_per_machines"] = durations_per_machines
        self.variables["present_per_machines"] = present_per_machines
        self.variables["intervals_per_machines"] = intervals_per_machines
        self.variables["present_per_machines_per_attr"] = present_per_machines_per_attr

    def constraint_convention_machines(self):
        present_per_machines: dict[int, list[BoolVarT]] = self.variables["present_per_machines"]
        present_per_machines_per_attr: dict[int, dict[int, list[BoolVarT]]] = self.variables[
            "present_per_machines_per_attr"]
        starts_per_machines: dict[int, list[IntVar]] = self.variables["starts_per_machines"]
        ends_per_machines: dict[int, list[IntVar]] = self.variables["ends_per_machines"]
        for machine_index in present_per_machines:
            for i in range(self.max_nb_batch - 1):
                # self.cp_model.AddImplication(present_per_machines[machine_index][i].Not(),
                #                              present_per_machines[machine_index][i+1].Not())
                # When no batch at some point, the next ones are also absent.
                # Redundant, in case.
                self.cp_model.add(present_per_machines[machine_index][i] >= present_per_machines[machine_index][i + 1])

                # Order of interval (increasing time!)
                (self.cp_model.add(starts_per_machines[machine_index][i + 1] >= ends_per_machines[machine_index][i])
                 .only_enforce_if(present_per_machines[machine_index][i + 1]))

            for i in range(self.max_nb_batch):
                # At most one type for each batch.
                self.cp_model.add_at_most_one([present_per_machines_per_attr[machine_index][t][i]
                                               for t in present_per_machines_per_attr[machine_index]])
                # Exactly one, when the batch is present !!
                (self.cp_model.add(sum([present_per_machines_per_attr[machine_index][t][i]
                                        for t in present_per_machines_per_attr[machine_index]]) == 1)
                 .only_enforce_if(present_per_machines[machine_index][i]))

    def define_jobs_intervals(self):
        horizon = max([self.problem.availability_intervals[i][-1][1]
                       for i in range(self.problem.n_machines)])
        present_job_per_machine = {i: {} for i in range(self.problem.n_jobs)}
        present_job_per_machine_and_batch = {i: {} for i in range(self.problem.n_jobs)}
        for i in range(self.problem.n_jobs):
            for machine_index in self.problem.eligible_machines[i]:
                present_job_per_machine[i][machine_index] = self.cp_model.NewBoolVar(f"present_job_{i}"
                                                                                     f"_machine_{machine_index}")
                present_job_per_machine_and_batch[i][machine_index] = [self.cp_model.NewBoolVar(f"present_job_{i}_"
                                                                                                f"machine_"
                                                                                                f"{machine_index}"
                                                                                                f"_batch_{b}")
                                                                       for b in range(self.max_nb_batch)]
        self.variables["present_job_per_machine"] = present_job_per_machine
        self.variables["present_job_per_machine_and_batch"] = present_job_per_machine_and_batch

    def define_objectives(self, objectives: list[OvenObjectivesEnum],
                          weight: list[float] = None):
        if weight is None:
            weight = [1]*len(objectives)
        self.variables["objectives"] = {}
        for obj in objectives:
            if obj == OvenObjectivesEnum.NB_BATCH:
                nb_batch = sum([x for m in self.variables["present_per_machines"]
                                for x in self.variables["present_per_machines"][m]])
                self.variables["objectives"][obj] = nb_batch
            if obj == OvenObjectivesEnum.PROCESSING_TIME:
                processing_time = sum([x for m in self.variables["durations_per_machines"]
                                       for x in self.variables["durations_per_machines"][m]])
                self.variables["objectives"][obj] = processing_time
            if obj == OvenObjectivesEnum.SETUP_COST:
                setup_cost = sum([x for m in self.variables["setup_cost_per_machine"]
                                  for x in self.variables["setup_cost_per_machine"][m]])
                self.variables["objectives"][obj] = setup_cost
            if obj == OvenObjectivesEnum.MAKESPAN:
                horizon = max([self.problem.availability_intervals[i][-1][1]
                               for i in range(self.problem.n_machines)])
                makespan = self.cp_model.NewIntVar(lb=max(self.problem.earliest_starts),
                                                   ub=horizon, name=f"makespan")
                self.cp_model.AddMaxEquality(makespan,
                                             [self.variables["ends_job"][i] for i in self.variables["ends_job"]])
                self.variables["objectives"][obj] = makespan
            if obj == OvenObjectivesEnum.TARDINESS:
                late = {i: self.cp_model.NewBoolVar(name=f"late_{i}")
                        for i in range(self.problem.n_jobs)}
                self.variables["late_jobs"] = late
                horizon = max([self.problem.availability_intervals[i][-1][1]
                               for i in range(self.problem.n_machines)])
                for i in late:
                    end_job = self.cp_model.NewIntVar(self.problem.earliest_starts[i], horizon,
                                                      name=f"end_{i}")
                    for m in self.variables["present_job_per_machine_and_batch"][i]:
                        for b in range(len(self.variables["present_job_per_machine_and_batch"][i][m])):
                            (self.cp_model.Add(end_job == self.variables["ends_per_machines"][m][b])
                             .only_enforce_if(self.variables["present_job_per_machine_and_batch"][i][m][b]))
                    (self.cp_model.Add(end_job > self.problem.latest_ends[i])
                     .OnlyEnforceIf(late[i]))
                    (self.cp_model.Add(end_job <= self.problem.latest_ends[i])
                     .OnlyEnforceIf(late[i].Not()))
                self.variables["objectives"][OvenObjectivesEnum.TARDINESS] = sum([late[i] for i in late])

        self.cp_model.Minimize(sum([self.variables["objectives"][objectives[i]]*weight[i]
                                    for i in range(len(objectives))]))

    def constraint_convention_jobs(self):
        # starts_job = self.variables["starts_job"]
        # ends_job = self.variables["ends_job"]
        # self.variables["durations_job"] = durations_job
        # starts_job_per_machine = self.variables["starts_job_per_machine"]
        # ends_job_per_machine = self.variables["ends_job_per_machine"]
        # durations_job_per_machine = self.variables["durations_job_per_machine"]
        present_job_per_machine: dict[int, dict[int, BoolVarT]] = self.variables["present_job_per_machine"]
        present_job_per_machine_and_batch: dict[int, dict[int, list[BoolVarT]]] = self.variables[
            "present_job_per_machine_and_batch"]

        for i in present_job_per_machine:
            self.cp_model.add_exactly_one([present_job_per_machine[i][m]
                                           for m in present_job_per_machine[i]])
            for m in present_job_per_machine_and_batch[i]:
                self.cp_model.add(sum(present_job_per_machine_and_batch[i][m]) <= 1)
                (self.cp_model.add(sum(present_job_per_machine_and_batch[i][m]) == 1)
                 .OnlyEnforceIf(present_job_per_machine[i][m]))
                (self.cp_model.add(sum(present_job_per_machine_and_batch[i][m]) == 0)
                 .OnlyEnforceIf(present_job_per_machine[i][m].Not()))

    def linking_jobs_and_machines(self):
        present_job_per_machine: dict[int, dict[int, BoolVarT]] = self.variables["present_job_per_machine"]
        present_job_per_machine_and_batch: dict[int, dict[int, list[BoolVarT]]] = (
            self.variables["present_job_per_machine_and_batch"])
        present_per_machines: dict[int, list[BoolVarT]] = self.variables["present_per_machines"]
        present_per_machines_per_attr: dict[int, dict[int, list[BoolVarT]]] = self.variables[
            "present_per_machines_per_attr"]
        starts_per_machines: dict[int, list[IntVar]] = self.variables["starts_per_machines"]
        ends_per_machines: dict[int, list[IntVar]] = self.variables["ends_per_machines"]
        durations_per_machines: dict[int, list[IntVar]] = self.variables["durations_per_machines"]

        for i in range(self.problem.n_jobs):
            type_ = self.problem.job_attributes[i]
            for m in present_job_per_machine[i]:
                for b in range(self.max_nb_batch):
                    present = present_job_per_machine_and_batch[i][m][b]
                    # -> Batch b is present
                    self.cp_model.add(present_per_machines[m][b] == 1).only_enforce_if(present)
                    self.cp_model.add(durations_per_machines[m][b] >= self.problem.min_times[i]).only_enforce_if(present)
                    self.cp_model.add(durations_per_machines[m][b] <= self.problem.max_times[i]).only_enforce_if(present)
                    # -> Job i is done on machine i
                    self.cp_model.add(present_job_per_machine[i][m] == 1).only_enforce_if(present)
                    self.cp_model.add(starts_per_machines[m][b] >= self.problem.earliest_starts[i]).only_enforce_if(present)
                    # -> Type is activated for this batch.
                    self.cp_model.add(present_per_machines_per_attr[m][type_][b] == 1).only_enforce_if(present)
        # if b-th batch of a machine is activated, some activities are put on it.
        for b in range(self.max_nb_batch):
            for m in range(self.problem.n_machines):
                (self.cp_model.add(sum([present_job_per_machine_and_batch[i][m][b]
                                        for i in present_job_per_machine if m in present_job_per_machine[i]]) >= 1)
                 .only_enforce_if(present_per_machines[m][b]))
                (self.cp_model.add(sum([present_job_per_machine_and_batch[i][m][b]
                                        for i in present_job_per_machine if m in present_job_per_machine[i]]) == 0)
                 .only_enforce_if(present_per_machines[m][b].Not()))

    def constraint_batch_size(self):
        present_job_per_machine_and_batch: dict[int, dict[int, list[BoolVarT]]] = (
            self.variables["present_job_per_machine_and_batch"])
        for m in range(self.problem.n_machines):
            for b in range(self.max_nb_batch):
                literal_and_coef = [(present_job_per_machine_and_batch[i][m][b], self.problem.job_sizes[i])
                                    for i in present_job_per_machine_and_batch
                                    if m in present_job_per_machine_and_batch[i]]
                self.cp_model.add(LinearExpr.weighted_sum([x[0] for x in literal_and_coef],
                                                          [x[1] for x in literal_and_coef])
                                  <= self.problem.machine_capacities[m])

    def constraint_setup_time(self):
        starts_per_machines = self.variables["starts_per_machines"]
        ends_per_machines = self.variables["ends_per_machines"]
        present_per_machines = self.variables["present_per_machines"]
        present_per_machines_per_attr = self.variables["present_per_machines_per_attr"]
        possible_setup_times = sorted(set([0] + [x for i in range(len(self.problem.setup_times))
                                                 for x in self.problem.setup_times[i]]))
        possible_setup_cost = sorted(set([0] + [x for i in range(len(self.problem.setup_costs))
                                                for x in self.problem.setup_costs[i]]))
        setup_time = {m: [self.cp_model.NewIntVarFromDomain(Domain.FromValues(possible_setup_times),
                                                            name=f"setup_machine_{m}_batch_{b}")
                          for b in range(self.max_nb_batch)]
                      for m in range(self.problem.n_machines)}
        setup_cost = {m: [self.cp_model.NewIntVarFromDomain(Domain.FromValues(possible_setup_cost),
                                                            name=f"setup_machine_{m}_batch_{b}")
                          for b in range(self.max_nb_batch)]
                      for m in range(self.problem.n_machines)}

        for machine in starts_per_machines:
            for t1 in present_per_machines_per_attr[machine]:
                self.cp_model.add(starts_per_machines[machine][0] >=
                                  self.problem.setup_times[self.problem.initial_states[machine]][t1]).only_enforce_if(
                    present_per_machines_per_attr[machine][t1][0],
                    present_per_machines[machine][0])
                self.cp_model.add(setup_time[machine][0] ==
                                  self.problem.setup_times[self.problem.initial_states[machine]][t1]).only_enforce_if(
                    present_per_machines_per_attr[machine][t1][0],
                    present_per_machines[machine][0])
                self.cp_model.add(setup_cost[machine][0] ==
                                  self.problem.setup_costs[self.problem.initial_states[machine]][t1]).only_enforce_if(
                    present_per_machines_per_attr[machine][t1][0],
                    present_per_machines[machine][0])
            for i in range(1, self.max_nb_batch):
                for t1 in present_per_machines_per_attr[machine]:
                    for t2 in present_per_machines_per_attr[machine]:
                        self.cp_model.add(starts_per_machines[machine][i] >=
                                          ends_per_machines[machine][i - 1] +
                                          self.problem.setup_times[t1][t2]).only_enforce_if(
                            present_per_machines_per_attr[machine][t1][i - 1],
                            present_per_machines_per_attr[machine][t2][i],
                            present_per_machines[machine][i - 1],
                            present_per_machines[machine][i])

                        self.cp_model.add(setup_time[machine][i] ==
                                          self.problem.setup_times[t1][t2]).only_enforce_if(
                            present_per_machines_per_attr[machine][t1][i - 1],
                            present_per_machines_per_attr[machine][t2][i],
                            present_per_machines[machine][i])
                        self.cp_model.add(setup_cost[machine][i] ==
                                          self.problem.setup_costs[t1][t2]).only_enforce_if(
                            present_per_machines_per_attr[machine][t1][i - 1],
                            present_per_machines_per_attr[machine][t2][i],
                            present_per_machines[machine][i])
            for i in range(self.max_nb_batch):
                self.cp_model.add(setup_cost[machine][i] == 0).only_enforce_if(
                    present_per_machines[machine][i].Not())
        self.variables["setup_time_per_machine"] = setup_time
        self.variables["setup_cost_per_machine"] = setup_cost

    def get_assignments_from_solution(self, sol: OvenSchedulingSolution) -> dict:
        """
        Translates a solution object into a comprehensive dictionary mapping all
        relevant variable names to their assigned values.
        """
        # 1. Ensure the solution's timing details are computed.
        if not sol.timed_schedule:
            self.problem.compute_and_get_details(sol, use_cache=True)

        # 2. Initialize the assignments dictionary to hold all variable values.
        assignments = defaultdict(lambda: defaultdict(dict))

        # 3. Build a reverse map for easy lookup of a job's assignment.
        job_to_assignment = {
            job_id: (m, b)
            for m, machine_batches in enumerate(sol.batches_per_machine)
            for b, batch_content in enumerate(machine_batches)
            for job_id in batch_content
        }

        # 4. Determine batch-related assignments for each machine.
        for m in range(self.problem.n_machines):
            last_attribute = self.problem.initial_states[m]
            num_batches_in_sol = len(sol.batches_per_machine[m])

            # Assign values for batches that ARE PRESENT in the solution
            for b in range(num_batches_in_sol):
                batch_content = sol.batches_per_machine[m][b]
                details = sol.timed_schedule[(m, b)]
                batch_attribute = self.problem.job_attributes[batch_content[0]]

                assignments["present_per_machines"][m][b] = 1
                assignments["starts_per_machines"][m][b] = details['start_time']
                assignments["ends_per_machines"][m][b] = details['completion_time']
                assignments["durations_per_machines"][m][b] = details['processing_time']

                # Assign attribute presence
                for attr in self.variables["present_per_machines_per_attr"][m]:
                    assignments["present_per_machines_per_attr"][m][attr, b] = 1 if attr == batch_attribute else 0

                # Assign setup costs and times
                assignments["setup_cost_per_machine"][m][b] = self.problem.setup_costs[last_attribute][batch_attribute]
                assignments["setup_time_per_machine"][m][b] = self.problem.setup_times[last_attribute][batch_attribute]

                last_attribute = batch_attribute

            # Assign default values for batches that ARE NOT PRESENT
            last_time = sol.timed_schedule[(m, num_batches_in_sol - 1)][
                "completion_time"] if num_batches_in_sol > 0 else 0
            for b in range(num_batches_in_sol, self.max_nb_batch):
                assignments["present_per_machines"][m][b] = 0
                assignments["starts_per_machines"][m][b] = last_time
                assignments["ends_per_machines"][m][b] = last_time
                assignments["durations_per_machines"][m][b] = 0
                for attr in self.variables["present_per_machines_per_attr"][m]:
                    assignments["present_per_machines_per_attr"][m][attr, b] = 0
                assignments["setup_cost_per_machine"][m][b] = 0
                assignments["setup_time_per_machine"][m][b] = 0

        # 5. Determine all job-specific assignments.
        for job_id in range(self.problem.n_jobs):
            if job_id in job_to_assignment:
                m, b = job_to_assignment[job_id]
                details = sol.timed_schedule[(m, b)]

                # Job timings
                assignments["starts_job"][job_id] = details['start_time']
                assignments["ends_job"][job_id] = details['completion_time']
                assignments["durations_job"][job_id] = details['processing_time']

                # Job assignments
                for m_other in self.problem.eligible_machines[job_id]:
                    assignments["present_job_per_machine"][job_id][m_other] = 1 if m_other == m else 0
                    for b_other in range(self.max_nb_batch):
                        is_assigned = 1 if m_other == m and b_other == b else 0
                        assignments["present_job_per_machine_and_batch"][job_id][m_other, b_other] = is_assigned

                # Job tardiness
                assignments["late_jobs"][job_id] = 1 if details['completion_time'] > self.problem.latest_ends[
                    job_id] else 0
            else:  # Should not happen in a valid solution
                # Assign default values for unassigned jobs
                assignments["late_jobs"][job_id] = 1

        # 6. Determine makespan.
        solution_makespan = 0
        if sol.timed_schedule:
            solution_makespan = max(d['completion_time'] for d in sol.timed_schedule.values() if d)
        assignments["makespan"] = solution_makespan

        return assignments


class ConstraintHandlerOvenCpsat(OrtoolsCpSatConstraintHandler):
    """
    A constraint handler for the Oven Scheduling problem that implements
    Large Neighborhood Search (LNS) strategies.

    Instead of freezing all variables, this handler freezes the solution
    except for a small, targeted "neighborhood" of variables that the

    solver is allowed to re-optimize.
    """

    def adding_constraint_from_results_store(
            self, solver: "CpsatOvenSolver", result_storage: ResultStorage, **kwargs: Any
    ) -> Iterable[Constraint]:
        """
        Main entry point for the LNS handler. It gets a solution and adds
        constraints to the model to define a neighborhood around it.

        Args:
            solver: The CpsatOvenSolver instance.
            result_storage: The storage containing the solution to build around.
            **kwargs:
                strategy (str): The neighborhood strategy to use.
                    "random_jobs": Frees a percentage of random jobs. (Default)
                    "late_jobs": Frees all jobs that are currently tardy.
                percentage_of_jobs_to_free (int): The percentage of jobs
                    to free for the "random_jobs" strategy.
        """
        solution = result_storage.get_best_solution()
        self.problem = solution.problem
        if solution is None:
            return []

        # Determine which neighborhood strategy to apply
        strategy = kwargs.get("strategy", "random_jobs")
        strategy = random.choice(["random_jobs"])

        # --- Get the set of jobs to 'free' based on the strategy ---
        if strategy == "late_jobs":
            jobs_to_free = self._get_late_jobs(solution)
            print(f"LNS Strategy: Freeing up {len(jobs_to_free)} late jobs.")
        else:  # Default to random_jobs
            percentage = kwargs.get("percentage_of_jobs_to_free", 10)
            percentage = 10
            jobs_to_free = self._get_random_jobs(percentage)
            print(f"LNS Strategy: Freeing up {len(jobs_to_free)} random jobs ({percentage}%).")

        if not jobs_to_free:
            # If no jobs are freed (e.g., no late jobs), don't add constraints.
            return []

        # --- Determine which machines are affected by the freed jobs ---
        # We need to free up batches on any machine a freed job could move to.
        machines_to_free = {m for job in jobs_to_free for m in solver.problem.eligible_machines[job]}

        # --- Get variable assignments from the provided solution ---
        assignments = solver.get_assignments_from_solution(solution)

        # --- Build the list of fixing constraints ---
        constraints = []
        # Freeze job assignments for all jobs that are NOT being freed
        constraints.extend(self._freeze_job_assignments(solver, assignments,
                                                        except_jobs=jobs_to_free))
        # Freeze batches on all machines that are NOT affected by the freed jobs
        #constraints.extend(self._freeze_batches(solver, assignments,
        #                                        except_machines=machines_to_free))

        return constraints

    def _get_late_jobs(self, solution: OvenSchedulingSolution) -> set[int]:
        """Identifies all jobs that are tardy in the given solution."""
        return {j for j, details in solution.get_job_completion_times().items()
                if details['completion_time'] > solution.problem.latest_ends[j]}

    def _get_random_jobs(self, percentage: int) -> set[int]:
        """Selects a random sample of jobs to free up."""
        problem = self.problem
        num_jobs_to_free = (problem.n_jobs * percentage) // 100
        return set(random.sample(range(problem.n_jobs), num_jobs_to_free))

    def _freeze_batches(self, solver: "CpsatOvenSolver", assignments: dict, except_machines: set) -> Iterable[
        Constraint]:
        """Returns constraints to fix the presence of batches on all but the excepted machines."""
        l = []
        for m in range(solver.problem.n_machines):
            if m in except_machines:
                continue
            for b in range(solver.max_nb_batch):
                val = assignments["present_per_machines"][m][b]
                l.append(solver.cp_model.Add(solver.variables["present_per_machines"][m][b] == val))
        return l

    def _freeze_job_assignments(self, solver: "CpsatOvenSolver", assignments: dict, except_jobs: set) -> Iterable[
        Constraint]:
        """Returns constraints to fix the machine assignment for all but the excepted jobs."""
        l = []
        for j in range(solver.problem.n_jobs):
            if j in except_jobs:
                continue
            for m in solver.problem.eligible_machines[j]:
                val = assignments["present_job_per_machine"][j][m]
                l.append(solver.cp_model.Add(solver.variables["present_job_per_machine"][j][m] == val))
        return l


class ConstraintHandlerOvenCpsat2(OrtoolsCpSatConstraintHandler):
    """
    A more advanced constraint handler for LNS that uses a portfolio of
    "smarter" neighborhood strategies.
    """

    def adding_constraint_from_results_store(
            self, solver: "CpsatOvenSolver", result_storage: ResultStorage, **kwargs: Any
    ) -> Iterable[Constraint]:
        """
        Randomly selects a neighborhood strategy and adds the corresponding
        constraints to the model.
        """
        solution = result_storage.get_best_solution()
        if solution is None:
            return []

        assignments = solver.get_assignments_from_solution(solution)

        # Randomly choose a strategy
        strategy = random.choice(["free_random_jobs", "free_late_jobs",
                                  "time_window", "reshuffle_machine"])
        print(f"LNS Strategy selected: {strategy}")

        if strategy == "free_late_jobs":
            jobs_to_free = {j for j, val in assignments["late_jobs"].items() if val == 1}
            return self._get_constraints_for_job_lns(solver, assignments, jobs_to_free)

        elif strategy == "time_window":
            time_delta = kwargs.get("time_delta", 500)
            return self._get_constraints_for_time_window_lns(solver, assignments, time_delta)

        elif strategy == "reshuffle_machine":
            num_machines = kwargs.get("num_machines_to_reshuffle", 1)
            return self._get_constraints_for_reshuffle_machine_lns(solver, assignments, num_machines)

        else:  # Default to "free_random_jobs"
            percentage = kwargs.get("percentage_of_jobs_to_free", 10)
            num_jobs_to_free = (solver.problem.n_jobs * percentage) // 100
            jobs_to_free = set(random.sample(range(solver.problem.n_jobs), num_jobs_to_free))
            return self._get_constraints_for_job_lns(solver, assignments, jobs_to_free)

    def _get_constraints_for_job_lns(self, solver: "CpsatOvenSolver", assignments: dict, jobs_to_free: set[int]) -> \
    Iterable[Constraint]:
        """
        Frees up a specific set of jobs (e.g., random or late) and the machines
        they are eligible for, while fixing everything else.
        """
        if not jobs_to_free:
            return []

        machines_to_free = {m for job in jobs_to_free for m in solver.problem.eligible_machines[job]}

        # Freeze job assignments for all jobs that are NOT being freed
        for j in range(solver.problem.n_jobs):
            if j in jobs_to_free:
                continue
            for m in solver.problem.eligible_machines[j]:
                val = assignments["present_job_per_machine"][j][m]
                yield solver.cp_model.Add(solver.variables["present_job_per_machine"][j][m] == val)

        # Freeze batches on all machines that are NOT affected by the freed jobs
        for m in range(solver.problem.n_machines):
            if m in machines_to_free:
                continue
            for b in range(solver.max_nb_batch):
                val = assignments["present_per_machines"][m][b]
                yield solver.cp_model.Add(solver.variables["present_per_machines"][m][b] == val)

    def _get_constraints_for_time_window_lns(self, solver: "CpsatOvenSolver", assignments: dict, time_delta: int) -> \
    Iterable[Constraint]:
        """
        Fixes all job/batch assignments but allows batch start times to move
        within a small window around their original values.
        """
        # 1. Fix ALL assignment variables completely
        for j in range(solver.problem.n_jobs):
            for m in solver.problem.eligible_machines[j]:
                for b in range(solver.max_nb_batch):
                    val = assignments["present_job_per_machine_and_batch"][j][m, b]
                    yield solver.cp_model.Add(solver.variables["present_job_per_machine_and_batch"][j][m][b] == val)

        # 2. Add time window constraints for the start time of each active batch
        for m in range(solver.problem.n_machines):
            for b in range(solver.max_nb_batch):
                if assignments["present_per_machines"][m][b] == 1:
                    original_start = assignments["starts_per_machines"][m][b]
                    start_var = solver.variables["starts_per_machines"][m][b]

                    # Constrain the start time to be in a window around the original
                    yield solver.cp_model.Add(start_var >= original_start - time_delta)
                    yield solver.cp_model.Add(start_var <= original_start + time_delta)

    def _get_constraints_for_reshuffle_machine_lns(self, solver: "CpsatOvenSolver", assignments: dict,
                                                   num_machines: int) -> Iterable[Constraint]:
        """
        Fixes which jobs are on which machine, but allows the solver to
        completely re-batch the jobs on a few selected machines.
        """
        machines_to_reshuffle = set(random.sample(range(solver.problem.n_machines), num_machines))

        # 1. Fix job-to-machine assignments for ALL jobs
        for j in range(solver.problem.n_jobs):
            for m in solver.problem.eligible_machines[j]:
                val = assignments["present_job_per_machine"][j][m]
                yield solver.cp_model.Add(solver.variables["present_job_per_machine"][j][m] == val)

        # 2. For machines we are NOT reshuffling, fix the detailed job-to-batch assignments
        for m in range(solver.problem.n_machines):
            if m in machines_to_reshuffle:
                continue  # On this machine, the solver is free to re-batch

            for b in range(solver.max_nb_batch):
                for j in solver.jobs_index_for_machine_i[m]:
                    val = assignments["present_job_per_machine_and_batch"][j][m, b]
                    yield solver.cp_model.Add(solver.variables["present_job_per_machine_and_batch"][j][m][b] == val)


class ConstraintHandlerOvenCpsat3(OrtoolsCpSatConstraintHandler):
    """
    A more advanced constraint handler for LNS that uses a portfolio of
    "smarter" neighborhood strategies.
    """

    def adding_constraint_from_results_store(
            self, solver: "CpsatOvenSolver", result_storage: ResultStorage, **kwargs: Any
    ) -> Iterable[Constraint]:
        """
        Randomly selects a neighborhood strategy and adds the corresponding
        constraints to the model.
        """
        solution = result_storage.get_best_solution()
        if solution is None:
            return []

        assignments = solver.get_assignments_from_solution(solution)
        solution: OvenSchedulingSolution
        problem: OvenSchedulingProblem = solution.problem
        # Randomly choose a strategy
        strategy = random.choice(["free_random_jobs", "free_late_jobs",
                                  "time_window", "reshuffle_machine"])
        print(f"LNS Strategy selected: {strategy}")
        constrs = []
        some_machine = random.randint(0, problem.n_machines-1)
        machines = random.sample(range(problem.n_machines),
                                 min(problem.n_machines, min(2, int(math.ceil(problem.n_machines/3)))))
        jobs_free = random.sample(range(problem.n_jobs),
                                  int(0.2*problem.n_jobs))
        sched = solution.get_job_completion_times()
        for m in range(problem.n_machines):
            if m not in machines:
                for job in range(problem.n_jobs):
                    if job in jobs_free:
                        continue
                    if m in problem.eligible_machines[job]:
                        if False:
                            for b in range(len(solver.variables["present_job_per_machine_and_batch"][job][m])):
                                val = assignments["present_job_per_machine_and_batch"][job][m, b]
                                constrs.append(
                                    solver.cp_model.Add(solver.variables["present_job_per_machine_and_batch"][job][m][b]
                                                        == val))
                        val = assignments["present_job_per_machine"][job][m]
                        constrs.append(solver.cp_model.Add(solver.variables["present_job_per_machine"][job][m] == val))
                last_time = problem.availability_intervals[m][-1][1]
                for b in range(solver.max_nb_batch):
                    if assignments["present_per_machines"][m][b] == 1:
                        original_start = assignments["starts_per_machines"][m][b]
                        original_end = assignments["ends_per_machines"][m][b]
                        last_time = assignments["ends_per_machines"][m][b]
                        start_var = solver.variables["starts_per_machines"][m][b]
                        constrs.append(solver.cp_model.add(start_var == original_start))
                        constrs.append(solver.cp_model.add(solver.variables["ends_per_machines"][m][b] == original_end))
                        constrs.append(solver.cp_model.add(solver.variables["durations_per_machines"][m][b] ==
                                       original_end-original_start))
                        constrs.append(solver.cp_model.add(solver.variables["present_per_machines"][m][b] == 1))
                    else:
                        constrs.append(solver.cp_model.add(solver.variables["starts_per_machines"][m][b] == last_time))
                        constrs.append(solver.cp_model.add(solver.variables["ends_per_machines"][m][b] == last_time))
                        constrs.append(solver.cp_model.add(solver.variables["durations_per_machines"][m][b] == 0))
                        constrs.append(solver.cp_model.add(solver.variables["present_per_machines"][m][b] == 0))
                    constrs.append(solver.cp_model.add(solver.variables["setup_cost_per_machine"][m][b] ==
                                                       assignments["setup_cost_per_machine"][m][b]))

        return constrs


class ConstraintHandlerOvenCpsat4(OrtoolsCpSatConstraintHandler):
    """
    A more advanced constraint handler for LNS that uses a portfolio of
    "smarter" neighborhood strategies.
    """

    def adding_constraint_from_results_store(
            self, solver: "CpsatOvenSolver", result_storage: ResultStorage, **kwargs: Any
    ) -> Iterable[Constraint]:
        """
        Randomly selects a neighborhood strategy and adds the corresponding
        constraints to the model.
        """
        solution = result_storage.get_best_solution()
        if solution is None:
            return []

        assignments = solver.get_assignments_from_solution(solution)
        solution: OvenSchedulingSolution
        problem: OvenSchedulingProblem = solution.problem
        # Randomly choose a strategy
        strategy = random.choice(["free_random_jobs", "free_late_jobs",
                                  "time_window", "reshuffle_machine"])
        print(f"LNS Strategy selected: {strategy}")
        constrs = []
        some_machine = random.randint(0, problem.n_machines-1)
        machines = random.sample(range(problem.n_machines),
                                 min(problem.n_machines, min(2, int(math.ceil(problem.n_machines/3)))))
        jobs_free = random.sample(range(problem.n_jobs),
                                  int(0.2*problem.n_jobs))
        sched = solution.get_job_completion_times()
        swaps = []
        change_machine = []
        reduce_batch = []
        for m in range(problem.n_machines):
            if m == some_machine:
                continue
            for job in range(problem.n_jobs):
                #if job in jobs_free:
                #    continue
                if m in problem.eligible_machines[job]:
                    for b in range(len(solver.variables["present_job_per_machine_and_batch"][job][m])):
                        val = assignments["present_job_per_machine_and_batch"][job][m, b]
                        if val == 1:
                            swaps.append(1-solver.variables["present_job_per_machine_and_batch"][job][m][b])
                        else:
                            swaps.append(solver.variables["present_job_per_machine_and_batch"][job][m][b])
                    val = assignments["present_job_per_machine"][job][m]
                    if val == 1:
                        change_machine.append(1-solver.variables["present_job_per_machine"][job][m])
                    else:
                        change_machine.append(solver.variables["present_job_per_machine"][job][m])
            last_time = problem.availability_intervals[m][-1][1]
            for b in range(solver.max_nb_batch):
                if assignments["present_per_machines"][m][b] == 1:
                    original_start = assignments["starts_per_machines"][m][b]
                    original_end = assignments["ends_per_machines"][m][b]
                    last_time = assignments["ends_per_machines"][m][b]
                    start_var = solver.variables["starts_per_machines"][m][b]
                    constrs.append(solver.cp_model.add(start_var <= original_start))
                    constrs.append(solver.cp_model.add(solver.variables["ends_per_machines"][m][b] <= original_end))
                    #constrs.append(solver.cp_model.add(solver.variables["durations_per_machines"][m][b] ==
                    #                                   original_end-original_start))
                    #constrs.append(solver.cp_model.add(solver.variables["present_per_machines"][m][b] == 1))
                else:
                    constrs.append(solver.cp_model.add(solver.variables["starts_per_machines"][m][b] <= last_time))
                    constrs.append(solver.cp_model.add(solver.variables["ends_per_machines"][m][b] <= last_time))
                    constrs.append(solver.cp_model.add(solver.variables["durations_per_machines"][m][b] == 0))
                    constrs.append(solver.cp_model.add(solver.variables["present_per_machines"][m][b] == 0))
                #constrs.append(solver.cp_model.add(solver.variables["setup_cost_per_machine"][m][b] ==
                #                                   assignments["setup_cost_per_machine"][m][b]))
        constrs.append(solver.cp_model.add(sum(change_machine)<=3))
        constrs.append(solver.cp_model.add(sum(swaps)<=10))
        return constrs
