from typing import Any

import numpy as np
from discrete_optimization.generic_tools.do_problem import Solution
from discrete_optimization.generic_tools.do_solver import WarmstartMixin
from hexaly_solvers.hexaly_solver import HexalySolver
import hexaly.optimizer
from oven_scheduling_problem.problem import OvenSchedulingProblem, OvenSchedulingSolution


class HexalyOvenScheduler(HexalySolver, WarmstartMixin):
    def set_warm_start(self, sol: OvenSchedulingSolution):
        """
        Sets the initial values of the Hexaly model's decision variables
        based on a provided feasible solution.

        This function assumes that 'type_of_job' and 'lateness' variables
        have been stored in self.variables during model initialization.

        Args:
            sol (OvenSchedulingSolution): A complete and feasible solution.
        """
        print("Setting warm start...")
        # 1. Ensure the solution's timing details are computed.
        if not sol.timed_schedule:
            self.problem.compute_and_get_details(sol, use_cache=True)

        # 2. Get references to the model's decision variables
        oven_intervals = self.variables["oven_intervals"]
        job_per_batch = self.variables["job_per_batch"]
        type_of_job = self.variables["type_of_job"]
        lateness = self.variables["lateness"]
        setup_costs_machines = self.variables["setup_costs_machines"]
        # = self.variables["jobs_ends"]

        # 3. Set values for batches based on the solution
        for m in range(self.problem.n_machines):
            num_batches_in_sol = len(sol.batches_per_machine[m])
            last_attribute = self.problem.initial_states[m]

            # Set values for the batches that are USED in the solution
            for b in range(num_batches_in_sol):
                batch_content = sol.batches_per_machine[m][b]
                details = sol.timed_schedule[(m, b)]
                # Set interval start and end times
                oven_intervals[m][b].value = hexaly.optimizer.HxInterval(details['start_time'],
                                                                         details['completion_time'])
                # Set the content of the batch
                job_per_batch[m][b].value.clear()
                for job_id in batch_content:
                    job_per_batch[m][b].value.add(job_id)

                # Set the attribute of the batch
                batch_attribute = self.problem.job_attributes[batch_content[0]]
                type_of_job[m][b].value = batch_attribute
                # Set setup cost for this batch
                setup_c = self.problem.setup_costs[last_attribute][batch_attribute]
                setup_costs_machines[m][b].value = setup_c

                # Update last_attribute for the next batch
                last_attribute = batch_attribute

            # Set values for the batches that are UNUSED
            last_value = self.problem.availability_intervals[m][-1][0]
            if num_batches_in_sol >= 1:
                last_value = sol.timed_schedule[(m, num_batches_in_sol-1)]["completion_time"]
            for b in range(num_batches_in_sol, self.max_nb_batch):
                oven_intervals[m][b].value = hexaly.optimizer.HxInterval(last_value,
                                                                         last_value)  # Non-present interval
                job_per_batch[m][b].value.clear()  # Empty batch
                type_of_job[m][b].value = self.problem.n_attributes  # Dummy value
                setup_costs_machines[m][b].value = 0

        # 4. Set values for the lateness of each job
        job_completion_times = sol.get_job_completion_times()
        for j in range(self.problem.n_jobs):
            completion_time = job_completion_times.get(j, {}).get('completion_time', 0)
            is_late = 1 if completion_time > self.problem.latest_ends[j] else 0
            lateness[j].value = is_late
            # job_ends[j].value = completion_time

        print("Warm start values set.")

    problem: OvenSchedulingProblem

    def __init__(self, problem: OvenSchedulingProblem,
                 known_ub: int,
                 max_nb_batch: int = None,
                 **kwargs: Any):

        super().__init__(problem, **kwargs)
        self.max_nb_batch = max_nb_batch
        if self.max_nb_batch is None:
            self.max_nb_batch = int(4 * self.problem.n_jobs / self.problem.n_machines)
        self.known_ub = known_ub

    def init_model(self, **kwargs: Any) -> None:
        horizon = max([self.problem.availability_intervals[m][-1][1]
                       for m in range(self.problem.n_machines)])
        max_nb_batch = self.max_nb_batch
        ls = hexaly.optimizer.HexalyOptimizer()
        model = ls.model
        self.create_oven_intervals(model=model,
                                   max_nb_batch=max_nb_batch,
                                   horizon=horizon)
        model.close()
        self.ls = ls
        self.model = model

    def create_oven_intervals(self, model: hexaly.optimizer.HxModel,
                              max_nb_batch: int, horizon: int):
        self.model = model
        oven_intervals = [[model.interval(0, horizon) for _ in range(max_nb_batch)]
                          for o in range(self.problem.n_machines)]
        oven_intervals_array = [model.array(itvs) for itvs in oven_intervals]
        # presence_oven_interval = [[model.presence(oven_intervals[o][b])
        #                           for b in range(len(oven_intervals[o]))]
        #                          for o in range(self.problem.n_machines)]
        job_per_batch = [[model.set(self.problem.n_jobs) for _ in range(max_nb_batch)]
                         for o in range(self.problem.n_machines)]
        job_per_batch_array = [model.array(jpb) for jpb in job_per_batch]
        model.constraint(model.partition([s for o in range(self.problem.n_machines)
                                          for s in job_per_batch[o]]))

        if True:
            incompatible_pairs = []
            for i in range(self.problem.n_jobs):
                for j in range(i + 1, self.problem.n_jobs):
                    incompatible = False

                    # Condition 1: Jobs have different attributes.
                    if self.problem.job_attributes[i] != self.problem.job_attributes[j]:
                        incompatible = True

                    # Condition 2: Jobs have non-overlapping processing time windows.
                    # This check is only needed if attributes are the same.
                    if not incompatible:
                        max_of_min_durations = max(self.problem.min_times[i], self.problem.min_times[j])
                        min_of_max_durations = min(self.problem.max_times[i], self.problem.max_times[j])
                        if max_of_min_durations > min_of_max_durations:
                            incompatible = True

                    # Condition 3: Jobs have no eligible machines in common.
                    #if not incompatible:
                    #    if self.problem.eligible_machines[i].isdisjoint(self.problem.eligible_machines[j]):
                    #        incompatible = True

                    if incompatible:
                        incompatible_pairs.append((i, j))
            print(f"Adding {len(incompatible_pairs)} incompatibility constraints.")
            if False:
                for i, j in incompatible_pairs:
                    for o in range(self.problem.n_machines):
                        for b in range(max_nb_batch):
                            # For each batch, it cannot contain both job i and job j.
                            model.constraint(
                                model.not_(model.and_(model.contains(job_per_batch[o][b], i),
                                                      model.contains(job_per_batch[o][b], j)))
                            )




        type_of_job = [[model.int(0, self.problem.n_attributes) for _ in range(max_nb_batch)]
                       for o in range(self.problem.n_machines)]
        type_of_jobs_array = [model.array(arr)
                              for arr in type_of_job]
        job_attributes_array = self.model.array(self.problem.job_attributes)
        size_array = self.model.array(self.problem.job_sizes)
        min_times_array = model.array(self.problem.min_times)
        max_times_array = model.array(self.problem.max_times)
        earliest_starts_array = model.array(self.problem.earliest_starts)
        get_type = model.lambda_function(lambda i: job_attributes_array[i])
        get_size = model.lambda_function(lambda i: size_array[i])
        get_min_time = model.lambda_function(lambda j: min_times_array[j])
        get_max_time = model.lambda_function(lambda j: max_times_array[j])
        get_earliest_start = model.lambda_function(lambda j: earliest_starts_array[j])
        max_setup_cost = np.max(np.array(self.problem.setup_costs))
        setup_times_with_dummy = []
        setup_costs_with_dummy = []
        for i in range(len(self.problem.setup_times)):
            setup_times_with_dummy.append(self.problem.setup_times[i]+[0])
            setup_costs_with_dummy.append(self.problem.setup_costs[i]+[0])
        setup_times_with_dummy.append([0]*len(setup_times_with_dummy[-1]))
        setup_costs_with_dummy.append([0]*len(setup_costs_with_dummy[-1]))
        setup_time_array = model.array(setup_times_with_dummy)
        setup_cost_array = model.array(setup_costs_with_dummy)
        stmachines = []
        stmachines_array = []
        lateness = [model.int(0, 1) for _ in range(self.problem.n_jobs)]
        #lateness_value = [model.int(-self.problem.latest_ends[j], horizon) for j in range(self.problem.n_jobs)]
        #lateness_value_array = model.array(lateness_value)
        lateness_array = model.array(lateness)

        for o in range(self.problem.n_machines):
            setup_costs_machines = [model.int(0, max_setup_cost)
                                    for _ in range(max_nb_batch)]
            setup_costs_machines_array = model.array(setup_costs_machines)
            not_allowed_jobs = {j for j in range(self.problem.n_jobs)
                                if o not in self.problem.eligible_machines[j]}
            model.constraint(model.and_(model.range(max_nb_batch),
                                        model.lambda_function(lambda b_:
                                        model.count(model.distinct(job_per_batch_array[o][b_],
                                                                   get_type)) <= 1)))
            model.constraint(model.and_(model.range(1, max_nb_batch),
                                        model.lambda_function(lambda ind:
                                                              model.start(oven_intervals_array[o][ind]) >=
                                                              model.end(oven_intervals_array[o][ind-1]))))
            for b in range(max_nb_batch):
                model.constraint(model.sum(job_per_batch[o][b], get_size) <=
                                 self.problem.machine_capacities[o])
                # One type.
                for job in not_allowed_jobs:
                    model.constraint(model.not_(model.contains(job_per_batch[o][b], job)))

                # model.constraint(model.length(oven_intervals[o][b]) >= model.max(job_per_batch[o][b],
                #                                                                  get_min_time))
                # model.constraint(model.length(oven_intervals[o][b]) <= model.min(job_per_batch[o][b],
                #                                                                  get_max_time))
                model.constraint(model.and_(job_per_batch[o][b],
                                            model.lambda_function(lambda j:
                                                                  model.start(oven_intervals[o][b])>=
                                                                  earliest_starts_array[j])))
                model.constraint(model.and_(job_per_batch[o][b],
                                            model.lambda_function(lambda j:
                                                                  model.length(oven_intervals[o][b]) >=
                                                                  min_times_array[j])))
                model.constraint(model.and_(job_per_batch[o][b],
                                            model.lambda_function(lambda j:
                                                                  model.length(oven_intervals[o][b]) <=
                                                                  max_times_array[j])))
                #model.constraint(model.and_(job_per_batch[o][b],
                #                            model.lambda_function(lambda j:
                #                                                  model.eq(lateness_value_array[j],
                #                                                  model.end(oven_intervals_array[o][b])-
                #                                                  max_times_array[j]))))
                #model.constraint(model.eq(model.length(oven_intervals[o][b]),
                #                          model.max(job_per_batch[o][b],
                #                                    get_min_time)))
                # model.constraint(model.iif(model.count(job_per_batch_array[o][b]) >= 1,
                #                            type_of_jobs_array[o][b] == get_type(job_per_batch[o][b][0]),
                #                            1))

                if True:
                    for job in range(self.problem.n_jobs):
                        model.constraint(model.iif(model.and_(model.contains(job_per_batch[o][b], job),
                                                              model.end(oven_intervals[o][b]) > self.problem.latest_ends[job]),
                                                              model.at(lateness_array, job) == 1,
                                                               1))
                        model.constraint(model.iif(model.contains(job_per_batch[o][b], job),
                                                   type_of_job[o][b] == self.problem.job_attributes[job], 1))

                        #model.constraint(model.iif(model.contains(job_per_batch[o][b], job),
                        #                           model.eq(lateness_value_array[job],
                        #                                    model.end(oven_intervals_array[o][b]) -
                        #                                    max_times_array[job])
                        #                           , 1))
                        # model.constraint(model.iif(model.contains(job_per_batch[o][b], job),
                        #                            model.geq(model.length(oven_intervals[o][b]),
                        #                                      self.problem.min_times[job]),
                        #                       1))
                        # model.constraint(model.iif(model.contains(job_per_batch[o][b], job),
                        #                            model.leq(model.length(oven_intervals[o][b]),
                        #                                      self.problem.max_times[job]),
                        #                            1))

                model.constraint(model.iif(model.count(job_per_batch[o][b]) == 0,
                                           model.length(oven_intervals[o][b]) == 0, 1))
                slot_constraints = []
                for slot in self.problem.availability_intervals[o]:
                    is_in_slot = model.and_(
                            model.start(oven_intervals_array[o][b]) >= slot[0],
                            model.end(oven_intervals_array[o][b]) <= slot[1]
                        )
                    slot_constraints.append(is_in_slot)
                model.constraint(model.or_(slot_constraints))
                model.constraint(model.iif(model.count(job_per_batch[o][b]) == 0,
                                           model.count(job_per_batch[o][b+1]) == 0, 1))

            initial_state = self.problem.initial_states[o]
            # Setup times
            model.constraint(model.start(oven_intervals[o][0]) >= self.model.at(setup_time_array,
                                                                          initial_state,
                                                                                type_of_job[o][0]))
            if True:
                lambda_setup_time = model.lambda_function(lambda index:
                                                          model.start(oven_intervals_array[o][index]) >=
                                                          model.end(oven_intervals_array[o][index-1]) +
                                                          self.model.at(setup_time_array,
                                                                  self.model.at(type_of_jobs_array[o], index-1),
                                                                        self.model.at(type_of_jobs_array[o], index)))
                model.constraint(model.and_(model.range(1, max_nb_batch), lambda_setup_time))
                model.constraint(model.at(setup_costs_machines_array, 0) == self.model.at(setup_cost_array,
                                                                                          initial_state,
                                                                                                type_of_job[o][0]))
                lambda_setup_cost = model.lambda_function(lambda index:
                                                          model.at(setup_costs_machines_array, index) ==
                                                          self.model.at(setup_cost_array,
                                                                        self.model.at(type_of_jobs_array[o],
                                                                                      index-1),
                                                                        self.model.at(type_of_jobs_array[o],
                                                                                      index)))
                model.constraint(model.and_(model.range(1, max_nb_batch), lambda_setup_cost))
                stmachines.append(setup_costs_machines)
                stmachines_array.append(setup_costs_machines_array)

        setup_cost = model.sum([model.sum(stm)
                                for stm in stmachines_array])
        processing_cost = model.sum([model.length(oven_intervals[o][b])
                                     for o in range(len(oven_intervals))
                                     for b in range(len(oven_intervals[o]))])
        late_cost = model.sum(lateness_array)
        additional_data = self.problem.additional_data
        w_pro = additional_data["weight_processing"]
        w_late = additional_data["weight_tardiness"]
        w_setup = additional_data["weight_setup_cost"]
        # self.model.minimize(0)
        #self.model.minimize(model.sum(lateness_value_array))
        #self.model.minimize(processing_cost)
        #self.model.minimize(setup_cost)
        self.model.minimize(w_pro*processing_cost+w_late*late_cost+w_setup*setup_cost)
        self.variables = {}
        self.variables["oven_intervals"] = oven_intervals
        self.variables["job_per_batch"] = job_per_batch
        self.variables["lateness"] = lateness
        self.variables["type_of_job"] = type_of_job
        self.variables["setup_costs_machines"] = stmachines
        # self.variables["jobs_ends"] = job_ends

    def retrieve_solution(self, **args) -> Solution:
        schedule = {}
        or_batches_per_machines = [[list(self.variables["job_per_batch"][o][b].value)
                                 for b in range(len(self.variables["job_per_batch"][o]))]
                                for o in range(self.problem.n_machines)]
        print(or_batches_per_machines)
        batches_per_machines = [[list(x) for x in or_batches_per_machines[o]
                                 if len(x) >= 1]
                                for o in range(self.problem.n_machines)]
        type_per_machines = {m: {} for m in range(self.problem.n_machines)}
        print(batches_per_machines)
        for o in range(self.problem.n_machines):
            nb_ = 0
            for b in range(len(self.variables["oven_intervals"][o])):
                #presence = self.variables["presence_oven_intervals"][o][b]
                #if presence.value == 1:
                if len(or_batches_per_machines[o][b]) > 0:
                    schedule[(o, nb_)] = {"start_time": self.variables["oven_intervals"][o][b].value.start(),
                                          "completion_time": self.variables["oven_intervals"][o][b].value.end()}
                    schedule[(o, nb_)]["processing_time"] = schedule[(o, nb_)]["completion_time"]-schedule[(o, nb_)]["start_time"]
                    type_per_machines[o][nb_] = self.problem.job_attributes[batches_per_machines[o][nb_][0]]
                    nb_ += 1
        for o in range(self.problem.n_machines):
            prev = self.problem.initial_states[o]
            for b in range(len(batches_per_machines[o])):
                new_type = type_per_machines[o][b]
                schedule[(o, b)]["setup_cost"] = self.problem.setup_costs[prev][new_type]
                prev = new_type
        sol = OvenSchedulingSolution(problem=self.problem, batches_per_machine=batches_per_machines)
        sol.timed_schedule = schedule
        objs = tuple([self.ls.model.objectives[i].value for i in range(self.ls.model.nb_objectives)])
        print("Objs =", objs)
        print("Gap to known up ", self.aggreg_from_sol(sol) / self.known_ub)
        print("Satisfy", self.problem.satisfy(sol))
        return sol
