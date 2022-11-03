import unittest
import os
import sys
import json

from discrete_optimization.rcpsp.solver import CP_RCPSP_MZN
from discrete_optimization.rcpsp.solver.ls_solver import LS_RCPSP_Solver
from discrete_optimization.rcpsp.solver.rcpsp_cp_lns_solver import LNS_CP_RCPSP_SOLVER

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(this_dir, "../../"))
import jobshop.jobshop_model
from jobshop.jobshop_model import JobShopSolution
from jobshop.jobshop_model import JobShopProblem
from jobshop.jobshop_model import TaskDetails
from jobshop.jobshop_solver import ORtools_CP_solver
from jobshop.jobshop_model import JSPLIBInstance
from jobshop.wrapper_solvers import RCPSPSolverForJobshop


class DatasTestModel:
    problem: JobShopProblem
    solution: JobShopSolution

    def __init__(self, problem, solution):
        self.problem = problem
        self.solution = solution


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

jobs_problem2 = [  # task = (machine_id, processing_time).
    [{"machine_id": 0, "processing_time": 3},
     {"machine_id": 1, "processing_time": 2},
     {"machine_id": 2, "processing_time": 2},
     {"machine_id": 0, "processing_time": 7},
     {"machine_id": 2, "processing_time": 3}],  # Job0
    [{"machine_id": 0, "processing_time": 2},
     {"machine_id": 2, "processing_time": 1},
     {"machine_id": 1, "processing_time": 4}],  # Job1
    [{"machine_id": 1, "processing_time": 4},
     {"machine_id": 2, "processing_time": 3},
     {"machine_id": 1, "processing_time": 5},
     {"machine_id": 2, "processing_time": 4},
     {"machine_id": 0, "processing_time": 3}],  # Job2
    [{"machine_id": 0, "processing_time": 6},
     {"machine_id": 2, "processing_time": 1},
     {"machine_id": 1, "processing_time": 2}]  # Job3
]

jobs_solution_ok_pb1 = [  # That solution is OK for jobs_problem1
    # machine0
    [TaskDetails(0, 0, 0, 3),
     TaskDetails(1, 0, 3, 5)],
    # machine1
    [TaskDetails(2, 0, 0, 4),
     TaskDetails(0, 1, 4, 6),
     TaskDetails(1, 2, 7, 11)],
    # machine2
    [TaskDetails(1, 1, 5, 6),
     TaskDetails(0, 2, 6, 8),
     TaskDetails(2, 1, 8, 11)]
]

jobs_solution_ko_overlap_pb1 = [  # That solution is KO for jobs_problem1 (overlap -> machine1&2)
    # machine0
    [TaskDetails(0, 0, 0, 3),
     TaskDetails(1, 0, 3, 5)],
    # machine1
    [TaskDetails(0, 1, 3, 5),
     TaskDetails(2, 0, 4, 8),
     TaskDetails(1, 2, 6, 10)],
    # machine2
    [TaskDetails(1, 1, 5, 6),
     TaskDetails(0, 2, 5, 7),
     TaskDetails(2, 1, 8, 11)]
]

jobs_solution_ko_precedence_pb1 = [  # That solution is KO for jobs_problem1 (non respect of precedence)
    # machine0
    [TaskDetails(0, 0, 0, 3),
     TaskDetails(1, 0, 3, 5)],
    # machine1
    [TaskDetails(2, 0, 0, 4),
     TaskDetails(0, 1, 3, 5),
     TaskDetails(1, 2, 7, 11)],
    # machine2
    [TaskDetails(1, 1, 2, 3),
     TaskDetails(0, 2, 4, 6),
     TaskDetails(2, 1, 8, 11)]
]

jobs_solution_ko_horizon_pb1 = [  # That solution is KO for jobs_problem1 (non respect of horizon)
    # machine0
    [TaskDetails(0, 0, 0, 3),
     TaskDetails(1, 0, 3, 5)],
    # machine1
    [TaskDetails(2, 0, 0, 4),
     TaskDetails(0, 1, 4, 6),
     TaskDetails(1, 2, 20, 24)],
    # machine2
    [TaskDetails(1, 1, 5, 6),
     TaskDetails(0, 2, 6, 8),
     TaskDetails(2, 1, 8, 11)]
]
jobs_solution_ko_schedule_order_pb1 = [  # That solution is KO for jobs_problem1 (non respect of horizon)
    # machine0
    [TaskDetails(0, 0, 0, 3),
     TaskDetails(1, 0, 3, 5)],
    # machine1
    [TaskDetails(1, 2, 7, 11),
     TaskDetails(2, 0, 0, 4),
     TaskDetails(0, 1, 4, 6)],
    # machine2
    [TaskDetails(1, 1, 5, 6),
     TaskDetails(0, 2, 6, 8),
     TaskDetails(2, 1, 8, 11)]
]

jobs_solution_ko_taskDuration = [  # That solution is OK for jobs_problem1
    # machine0
    [TaskDetails(0, 0, 0, 3),
     TaskDetails(1, 0, 3, 8)],
    # machine1
    [TaskDetails(2, 0, 0, 4),
     TaskDetails(0, 1, 4, 6),
     TaskDetails(1, 2, 7, 11)],
    # machine2
    [TaskDetails(1, 1, 5, 6),
     TaskDetails(0, 2, 6, 8),
     TaskDetails(2, 1, 8, 12)]
]

data_pb1_solOK = DatasTestModel(JobShopProblem(jobs_problem1),
                                JobShopSolution(jobs_problem1, jobs_solution_ok_pb1))
data_pb1_solOK_with_value = DatasTestModel(JobShopProblem(jobs_problem1),
                                           JobShopSolution(jobs_problem1, jobs_solution_ok_pb1, value=11))
data_pb1_solKO_overlap = DatasTestModel(JobShopProblem(jobs_problem1),
                                        JobShopSolution(jobs_problem1, jobs_solution_ko_overlap_pb1))
data_pb1_solKO_precedence = DatasTestModel(JobShopProblem(jobs_problem1),
                                           JobShopSolution(jobs_problem1, jobs_solution_ko_precedence_pb1))
data_pb1_solKO_horizon = DatasTestModel(JobShopProblem(jobs_problem1),
                                        JobShopSolution(jobs_problem1, jobs_solution_ko_horizon_pb1))
data_pb1_solKO_schedule_order = DatasTestModel(JobShopProblem(jobs_problem1),
                                               JobShopSolution(jobs_problem1, jobs_solution_ko_schedule_order_pb1))
data_pb1_solKO_taskDuration = DatasTestModel(JobShopProblem(jobs_problem1),
                                             JobShopSolution(jobs_problem1, jobs_solution_ko_taskDuration))


class MyTestCase(unittest.TestCase):
    def test_evaluate_value(self):
        print("\nTesting evaluate_value...")
        self.assertEqual(11, data_pb1_solOK.problem.evaluate_value(data_pb1_solOK.solution))
        print("Done")

    def test_evaluate(self):
        print("\nTesting evaluate...")
        self.assertEqual({'value': 11}, data_pb1_solOK.problem.evaluate(data_pb1_solOK.solution))
        self.assertEqual({'value': 11}, data_pb1_solOK_with_value.problem.evaluate(data_pb1_solOK_with_value.solution))
        print("Done")

    def test_satisfy(self):
        print("\nTesting satisfy...")
        self.assertEqual(True,
                         data_pb1_solOK.problem.satisfy(data_pb1_solOK.solution))
        self.assertEqual(False,
                         data_pb1_solKO_overlap.problem.satisfy(data_pb1_solKO_overlap.solution))
        self.assertEqual(False,
                         data_pb1_solKO_precedence.problem.satisfy(data_pb1_solKO_precedence.solution))
        self.assertEqual(False,
                         data_pb1_solKO_horizon.problem.satisfy(data_pb1_solKO_horizon.solution))
        self.assertEqual(False,
                         data_pb1_solKO_schedule_order.problem.satisfy(data_pb1_solKO_schedule_order.solution))
        self.assertEqual(False,
                         data_pb1_solKO_taskDuration.problem.satisfy(data_pb1_solKO_taskDuration.solution))
        print("Done")

    def test_duration(self):
        print("\nTesting check_duration...")
        self.assertEqual(True,
                         data_pb1_solOK.problem.satisfy(data_pb1_solOK.solution))
        self.assertEqual(False,
                         data_pb1_solKO_taskDuration.problem.check_duration(data_pb1_solKO_taskDuration.solution))
        print("Done")

    def test_schedule_order(self):
        print("\nTesting check_schedule_order...")
        self.assertEqual(True,
                         data_pb1_solOK.problem.satisfy(data_pb1_solOK.solution))
        self.assertEqual(False,
                         data_pb1_solKO_schedule_order.problem.check_schedule_order(
                             data_pb1_solKO_schedule_order.solution))
        print("Done")

    def test_precedence(self):
        print("\nTesting check_precedence...")
        self.assertEqual(True,
                         data_pb1_solOK.problem.satisfy(data_pb1_solOK.solution))
        self.assertEqual(False, data_pb1_solKO_precedence.problem.check_precedence(
            data_pb1_solKO_precedence.solution))
        print("Done")

    def test_overlap(self):
        print("\nTesting check_no_overlap...")
        self.assertEqual(True,
                         data_pb1_solOK.problem.satisfy(data_pb1_solOK.solution))
        self.assertEqual(False, data_pb1_solKO_overlap.problem.check_no_overlap(
            data_pb1_solKO_overlap.solution))
        print("Done")

    def test_ORtools_solver(self):
        print("\nTesting ORtools_solver...")
        solver = ORtools_CP_solver()
        solution = solver.solve(JobShopProblem(jobs_problem2))
        solution.print_solution()
        self.assertNotEqual(11, solution.value)
        self.assertEqual(True,
                         solution.problem.satisfy(solution))
        print("Done")

    def test_JSPLIB(self):
        print("\nTesting JSPLIB...")
        target = "ta20"
        jsp_lib_instance = jobshop.jobshop_model.create_jsplib_instance(target)
        problem = jsp_lib_instance.jsplib_to_jobshop()
        print("\nJSPLIB %s in JobshopProblem :\n" % target)
        problem.print_problem()

        solver = ORtools_CP_solver(time_out=600)
        solution = solver.solve(problem)

        if jsp_lib_instance.optimum is None:
            if jsp_lib_instance.lower_bound is None:
                print("optimum : nan")
            else:
                print("optimum : %s-%s" % (jsp_lib_instance.lower_bound, jsp_lib_instance.upper_bound))
        else:
            print("optimum : %s" % jsp_lib_instance.optimum)

        solution.print_solution()
        self.assertEqual(True,
                         solution.problem.satisfy(solution))
        print("Done")

    def test_jsplib_with_rcpsp_solver(self):
        print("\nTesting JSPLIB with rcpsp solver...")
        target = "abz5"
        jsp_lib_instance = jobshop.jobshop_model.create_jsplib_instance(target)
        problem = jsp_lib_instance.jsplib_to_jobshop()
        print("\nJSPLIB %s in JobshopProblem :\n" % target)
        problem.print_problem()
        solver = RCPSPSolverForJobshop(problem, class_solver=LNS_CP_RCPSP_SOLVER)
        result = solver.solve()
        best_sol = result.get_best_solution()
        print(best_sol)
        # self.assertEqual(True,???)


if __name__ == '__main__':
    unittest.main()
