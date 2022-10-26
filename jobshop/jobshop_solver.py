import collections
from ortools.sat.python import cp_model
from discrete_optimization.jobshop.jobshop_model import JobShopSolution
from discrete_optimization.jobshop.jobshop_model import JobShopProblem
from discrete_optimization.jobshop.jobshop_model import TaskDetails


def sort_taskDetail(td: TaskDetails):
    return td.start_time


def main(problem: JobShopProblem):
    # problem : List[List[Dict[str, int]]]  # {"machine_id":__, "processing_time":__}
    # Create the model.
    model = cp_model.CpModel()

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
            start_var = model.NewIntVar(0, problem.horizon, 'start' + suffix)
            end_var = model.NewIntVar(0, problem.horizon, 'end' + suffix)
            interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                'interval' + suffix)
            all_tasks[job_id, task_id] = task_type(start=start_var,
                                                   end=end_var,
                                                   interval=interval_var)
            machine_to_intervals[machine].append(interval_var)

    # Create and add disjunctive constraints.
    for machine in range(problem.n_machines):
        model.AddNoOverlap(machine_to_intervals[machine])

    # Precedences inside a job.
    for job_id, job in enumerate(problem.jobs):
        for task_id in range(len(job) - 1):
            model.Add(all_tasks[job_id, task_id +
                                1].start >= all_tasks[job_id, task_id].end)

    # Value objective.
    obj_var = model.NewIntVar(0, problem.horizon, 'value')
    model.AddMaxEquality(obj_var, [
        all_tasks[job_id, len(job) - 1].end
        for job_id, job in enumerate(problem.jobs)
    ])
    model.Minimize(obj_var)

    # Creates the solver and solve.
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print('Solution:')
        # Create one list of assigned tasks per machine.
        jobshop_solution = JobShopSolution(problem, [])
        for machine in range(problem.n_machines):
            jobshop_solution.jobshop_schedule.append([])
        for job_id, job in enumerate(problem.jobs):
            for task_id, task in enumerate(job):
                machine = task["machine_id"]
                print(all_tasks[job_id, task_id].start)
                jobshop_solution.jobshop_schedule[machine].append(
                    TaskDetails(job_id,
                                task_id,
                                solver.Value(all_tasks[job_id, task_id].start),
                                solver.Value(all_tasks[job_id, task_id].end)))

        # Create per machine output lines.
        output = ''
        for machine in range(problem.n_machines):
            # Sort by starting time.
            jobshop_solution.jobshop_schedule[machine].sort(key=TaskDetails.TaskDetail_sort)
            sol_line_tasks = 'Machine ' + str(machine) + ': '
            sol_line = '           '

            for assigned_task in jobshop_solution.jobshop_schedule[machine]:
                name = 'job_%i_task_%i' % (assigned_task.job_id,
                                           assigned_task.task_id)
                # Add spaces to output to align columns.
                sol_line_tasks += '%-15s' % name

                start = assigned_task.start_time
                duration = assigned_task.end_time - assigned_task.start_time
                sol_tmp = '[%i,%i]' % (start, start + duration)
                # Add spaces to output to align columns.
                sol_line += '%-15s' % sol_tmp

            sol_line += '\n'
            sol_line_tasks += '\n'
            output += sol_line_tasks
            output += sol_line

        # Finally print the solution found.
        print(f'Optimal Schedule Length: {solver.ObjectiveValue()}')
        print(output)
    else:
        print('No solution found.')

    # Statistics.
    print('\nStatistics')
    print('  - conflicts: %i' % solver.NumConflicts())
    print('  - branches : %i' % solver.NumBranches())
    print('  - wall time: %f s' % solver.WallTime())


if __name__ == '__main__':
    jobs_problem1 = [  # task = (machine_id, processing_time).
        [{"machine_id": 0, "processing_time": 3},
         {"machine_id": 1, "processing_time": 2},
         {"machine_id": 2, "processing_time": 2}],  # Job0
        [{"machine_id": 0, "processing_time": 2},
         {"machine_id": 2, "processing_time": 1},
         {"machine_id": 1, "processing_time": 4}],  # Job1
        [{"machine_id": 1, "processing_time": 4},
         {"machine_id": 2, "processing_time": 3}]  # Job2
    ]
    problem1 = JobShopProblem(jobs_problem1)
    main(problem1)
