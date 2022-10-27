import collections
from copy import deepcopy

from ortools.sat.python import cp_model
from discrete_optimization.jobshop.jobshop_model import JobShopSolution
from discrete_optimization.jobshop.jobshop_model import JobShopProblem
from discrete_optimization.jobshop.jobshop_model import TaskDetails


class ORtools_CP_solver():

    def __init__(self):
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()

    def solve(self, problem: JobShopProblem):
        # problem : List[List[Dict[str, int]]]  # {"machine_id":__, "processing_time":__}
        # Create the model.

        # Named tuple to store information about created variables.
        task_type = collections.namedtuple('task_type', 'start end interval')
        # Named tuple to manipulate solution information.
        assigned_task_type = collections.namedtuple('assigned_task_type',
                                                    'start job index duration')

        # Creates job intervals and add to the corresponding machine lists.
        all_tasks = {}
        machine_to_intervals = collections.defaultdict(list)

        for job_id, job in enumerate(problem.jobs):
            for task_id, task in enumerate(job):
                machine = task["machine_id"]
                duration = task["processing_time"]
                suffix = '_%i_%i' % (job_id, task_id)
                start_var = self.model.NewIntVar(0, problem.horizon, 'start' + suffix)
                end_var = self.model.NewIntVar(0, problem.horizon, 'end' + suffix)
                interval_var = self.model.NewIntervalVar(start_var, duration, end_var,
                                                         'interval' + suffix)
                all_tasks[job_id, task_id] = task_type(start=start_var,
                                                       end=end_var,
                                                       interval=interval_var)
                machine_to_intervals[machine].append(interval_var)

        # Create and add disjunctive constraints.
        for machine in range(problem.n_machines):
            self.model.AddNoOverlap(machine_to_intervals[machine])

        # Precedences inside a job.
        for job_id, job in enumerate(problem.jobs):
            for task_id in range(len(job) - 1):
                self.model.Add(all_tasks[job_id, task_id +
                                         1].start >= all_tasks[job_id, task_id].end)

        # Value objective.
        obj_var = self.model.NewIntVar(0, problem.horizon, 'value')
        self.model.AddMaxEquality(obj_var, [
            all_tasks[job_id, len(job) - 1].end
            for job_id, job in enumerate(problem.jobs)
        ])
        self.model.Minimize(obj_var)

        # Creates the solver and solve.

        status = self.solver.Solve(self.model)
        jobshop_solution = JobShopSolution(problem, [])
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # Create one list of assigned tasks per machine.
            for machine in range(problem.n_machines):
                jobshop_solution.jobshop_schedule.append([])
            for job_id, job in enumerate(problem.jobs):
                for task_id, task in enumerate(job):
                    machine = task["machine_id"]
                    jobshop_solution.jobshop_schedule[machine].append(
                        TaskDetails(job_id,
                                    task_id,
                                    self.solver.Value(all_tasks[job_id, task_id].start),
                                    self.solver.Value(all_tasks[job_id, task_id].end)))
        for machine in jobshop_solution.jobshop_schedule:
            machine.sort(key=TaskDetails.taskDetail_sort)
        return jobshop_solution

    def print_statistics(self):  # Statistics.
        print('\nStatistics')
        print('  - conflicts: %i' % self.solver.NumConflicts())
        print('  - branches : %i' % self.solver.NumBranches())
        print('  - wall time: %f s' % self.solver.WallTime())
