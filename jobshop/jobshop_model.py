#  Copyright (c) 2022 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import os
from copy import deepcopy
from typing import Dict, List, NamedTuple
from discrete_optimization.generic_tools.do_problem import (
    EncodingRegister,
    MethodAggregating,
    ModeOptim,
    ObjectiveHandling,
    ObjectiveRegister,
    Problem,
    RobustProblem,
    Solution,
    TupleFitness,
    TypeAttribute,
    TypeObjective,
)


class TaskDetails:
    job_id: int
    task_id: int
    start_time: int
    end_time: int

    def __init__(self, job_id, task_id, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time
        self.task_id = task_id
        self.job_id = job_id

    def __str__(self):
        return (
                'job_%i_task_%i' % (self.job_id, self.task_id)
                + " [start_time : %i, end_time : %i" % (self.start_time, self.end_time)
        )

    def taskDetail_sort(self):
        return self.start_time


def sort_task_details(list_td: List[TaskDetails]):
    return sorted(list_td, key=lambda x: x.start_time)


class JobShopSolution(Solution):
    jobshop_schedule: List[List[TaskDetails]]
    # List of ordered jobs (matrix indices) per machines containing TaskDetails
    value: int  # Value of the end of last task

    def __init__(self, problem, jobshop_schedule, value=None):
        self.problem = problem
        self.jobshop_schedule = jobshop_schedule
        self.value = value

    def __str__(self):
        return str([str(tt) for t in self.jobshop_schedule for tt in t])+" makespan="+str(self.value)

    def copy(self) -> "Solution":
        return JobShopSolution(
            problem=self.problem,
            jobshop_schedule=deepcopy(self.jobshop_schedule),
            value=self.value
        )

    def change_problem(self, new_problem: "Problem"):
        self.__init__(
            problem=new_problem,
            jobshop_schedule=self.jobshop_schedule,
            value=self.value
        )

    def sort(self):
        for machine in self.jobshop_schedule:
            machine.sort(key=TaskDetails.taskDetail_sort)

    def print_solution(self):
        output = ''
        for machine_id, machine in enumerate(self.jobshop_schedule):
            # Sort by starting time
            sol_line_tasks = 'Machine ' + str(machine_id) + ': '
            sol_line = '           '

            for task_number, task_details in enumerate(machine):
                name = 'job_%i_task_%i' % (task_details.job_id,
                                           task_details.task_id)
                # Add spaces to output to align columns.
                sol_line_tasks += '%-15s' % name

                start = task_details.start_time
                end = task_details.end_time
                sol_tmp = '[%i,%i]' % (start, end)
                # Add spaces to output to align columns.
                sol_line += '%-15s' % sol_tmp

            sol_line += '\n'
            sol_line_tasks += '\n'
            output += sol_line_tasks
            output += sol_line

        # Finally print the solution found.
        print(f'Optimal Schedule Length: {self.value}')
        print(output)


class JobShopProblem(Problem):
    n_machines: int  # numbers of machines
    jobs: List[List[Dict[str, int]]]  # {"machine_id":__, "processing_time":__}
    # List of jobs containing ordered tasks (machine_id, processing_time)
    # that has to be processed
    horizon: int  # Maximum time needed for the tasks processing (sum of all task duration)

    def __init__(
            self,
            jobs: List[List[Dict[str, int]]]
    ):
        self.jobs = jobs
        self.n_machines = 1 + max(task["machine_id"] for job in jobs for task in job)
        self.horizon = sum(task["processing_time"] for job in jobs for task in job)

    def print_problem(self):
        final_output = 'Job_number       '
        output = ''
        max_task = 0
        for job_id, job in enumerate(self.jobs):
            # Sort by starting time
            pb_line_task = '    ' + str(job_id) + ':      '

            for task_number, task_details in enumerate(job):
                name = 'mach_%i_time_%i  ' % (task_details['machine_id'],
                                                   task_details['processing_time'])
                # Add spaces to output to align columns.
                pb_line_task += '%-20s' % name
                max_task = max(max_task, task_number)

            pb_line_task += '\n'
            output += pb_line_task
        for task in range(max_task+1):
            name = 'Task ' + str(task)
            final_output += '%-20s' % name
        final_output += '\n' + output
        # Finally print the solution found.
        print(final_output)

    def evaluate(self, jobshop_solution: JobShopSolution) -> Dict[str, float]:
        if jobshop_solution.value is None:
            val = self.evaluate_value(jobshop_solution)
            jobshop_solution.value = val
        else:
            val = jobshop_solution.value
        return {"value": val}

    def evaluate_value(self, jobshop_solution: JobShopSolution):
        value = 0
        for machine_schedule in jobshop_solution.jobshop_schedule:
            for task in machine_schedule:
                value = max(value, task.end_time)
        return value

    def check_precedence(self, jobshop_solution: JobShopSolution):
        machines_schedule = jobshop_solution.jobshop_schedule
        for job_id in range(len(self.jobs)):  # Retrieving the job_id
            for task_id in range(len(self.jobs[job_id]) - 1):
                # Retrieving the task_id. -1 because we want to check the precedence
                # comparing actual task (task_id) and next task (task_id+1)
                actual_task_machine_id = self.jobs[job_id][task_id]["machine_id"]
                next_task_machine_id = self.jobs[job_id][task_id + 1]["machine_id"]

                # Now we find the actual task and next task of a job in the machines schedule
                for actual_task in machines_schedule[actual_task_machine_id]:
                    if (job_id, task_id) == (actual_task.job_id, actual_task.task_id):
                        for next_task in machines_schedule[next_task_machine_id]:
                            if (job_id, task_id + 1) == (next_task.job_id, next_task.task_id):
                                if actual_task.end_time > next_task.start_time:
                                    return False
        return True

    def check_duration(self, jobshop_solution: JobShopSolution):
        machines_schedule = jobshop_solution.jobshop_schedule
        for job_id in range(len(self.jobs)):  # Retrieving the job_id
            for task_id in range(len(self.jobs[job_id])):
                machine_id = self.jobs[job_id][task_id]["machine_id"]
                processing_time = self.jobs[job_id][task_id]["processing_time"]
                for task in machines_schedule[machine_id]:
                    if (job_id, task_id) == (task.job_id, task.task_id):
                        if processing_time != (task.end_time - task.start_time):
                            return False
        return True

    def check_schedule_order(self, jobshop_solution: JobShopSolution):
        machines_schedule = jobshop_solution.jobshop_schedule
        for machine_schedule in machines_schedule:
            end_time = 0
            for task in machine_schedule:
                if task.start_time < end_time:
                    return False
                end_time = task.end_time
        return True

    def check_no_overlap(self, jobshop_solution: JobShopSolution):
        machines_schedule = jobshop_solution.jobshop_schedule
        for machine_schedule in machines_schedule:
            for task in range(len(machine_schedule) - 1):
                if machine_schedule[task].end_time > machine_schedule[task + 1].start_time:
                    return False
        return True

    def satisfy(self, jobshop_solution: JobShopSolution) -> bool:
        if jobshop_solution.value is None:
            return (self.check_no_overlap(jobshop_solution) and
                    self.check_precedence(jobshop_solution) and
                    self.horizon >= self.evaluate_value(jobshop_solution) and
                    self.check_duration(jobshop_solution) and
                    self.check_schedule_order(jobshop_solution)
                    )
        return (self.check_no_overlap(jobshop_solution) and
                self.check_precedence(jobshop_solution) and
                self.horizon >= jobshop_solution.value and
                self.check_duration(jobshop_solution) and
                self.check_schedule_order(jobshop_solution)
                )

    def get_attribute_register(self) -> EncodingRegister:
        dict_register = {
        }
        return EncodingRegister(dict_register)

    def get_solution_type(self):
        return JobShopSolution

    def get_objective_register(self) -> ObjectiveRegister:
        dict_objective = {
            "value": {"type": TypeObjective.OBJECTIVE, "default_weight": 1}
        }
        return ObjectiveRegister(
            objective_sense=ModeOptim.MINIMIZATION,
            objective_handling=ObjectiveHandling.SINGLE,
            dict_objective_to_doc=dict_objective
        )



class JSPLIB_instance():

    def __init__(self, instance):
        self.name = instance["name"]
        self.n_jobs = instance["jobs"]
        self.n_machines = instance["machines"]
        self.optimum = instance["optimum"]
        if self.optimum is None:
            self.lower_bound = instance['bounds']['lower']
            self.upper_bound = instance['bounds']['upper']
        else:
            self.lower_bound = None
            self.upper_bound = None
        self.path = instance['path']

    def jsplib_to_jobshop(self):
        file = open(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../JSPLIB/%s" % self.path)), 'r')
        split_lines = []
        lines = file.readlines()
        processed_line = 0
        problem = []
        for line in lines:
            if not (line.startswith("#")):
                split_line = line.split()
                job = []
                if processed_line == 0:
                    if not (int(split_line[0]) == self.n_jobs
                            and int(split_line[1]) == self.n_machines):
                        exit(1)
                else:
                    for num, n in enumerate(split_line):
                        if num % 2 == 0:
                            machine = int(n)
                        else:
                            job.append({"machine_id": machine, "processing_time": int(n)})
                    problem.append(job)
                processed_line += 1

        return JobShopProblem(problem)
