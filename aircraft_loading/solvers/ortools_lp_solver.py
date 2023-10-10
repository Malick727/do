from typing import Optional, Dict, Hashable, Any
from aircraft_loading.solvers.aircraft_loading_solver import BaseSolver, AircraftLoadingProblem, ParamsObjectiveFunction
from aircraft_loading.aircraft_loading_problem import AircraftLoadingSolution, AircraftLoadingProblem
from ortools.linear_solver import _pywraplp
from ortools.linear_solver import pywraplp

from discrete_optimization.generic_tools.lp_tools import ParametersMilp
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
import logging
logger = logging.getLogger(__name__)


class OrtoolsLpTeamAllocationSolver(BaseSolver):

    CLP_LINEAR_PROGRAMMING = _pywraplp.Solver_CLP_LINEAR_PROGRAMMING
    GLPK_LINEAR_PROGRAMMING = _pywraplp.Solver_GLPK_LINEAR_PROGRAMMING
    GLOP_LINEAR_PROGRAMMING = _pywraplp.Solver_GLOP_LINEAR_PROGRAMMING
    PDLP_LINEAR_PROGRAMMING = _pywraplp.Solver_PDLP_LINEAR_PROGRAMMING
    SCIP_MIXED_INTEGER_PROGRAMMING = _pywraplp.Solver_SCIP_MIXED_INTEGER_PROGRAMMING
    GLPK_MIXED_INTEGER_PROGRAMMING = _pywraplp.Solver_GLPK_MIXED_INTEGER_PROGRAMMING
    CBC_MIXED_INTEGER_PROGRAMMING = _pywraplp.Solver_CBC_MIXED_INTEGER_PROGRAMMING
    GUROBI_LINEAR_PROGRAMMING = _pywraplp.Solver_GUROBI_LINEAR_PROGRAMMING
    GUROBI_MIXED_INTEGER_PROGRAMMING = _pywraplp.Solver_GUROBI_MIXED_INTEGER_PROGRAMMING
    CPLEX_LINEAR_PROGRAMMING = _pywraplp.Solver_CPLEX_LINEAR_PROGRAMMING
    CPLEX_MIXED_INTEGER_PROGRAMMING = _pywraplp.Solver_CPLEX_MIXED_INTEGER_PROGRAMMING
    XPRESS_LINEAR_PROGRAMMING = _pywraplp.Solver_XPRESS_LINEAR_PROGRAMMING
    XPRESS_MIXED_INTEGER_PROGRAMMING = _pywraplp.Solver_XPRESS_MIXED_INTEGER_PROGRAMMING
    BOP_INTEGER_PROGRAMMING = _pywraplp.Solver_BOP_INTEGER_PROGRAMMING
    SAT_INTEGER_PROGRAMMING = _pywraplp.Solver_SAT_INTEGER_PROGRAMMING

    def __init__(
        self,
        problem: AircraftLoadingProblem,
        params_objective_function: Optional[ParamsObjectiveFunction] = None,
        solver_name=pywraplp.Solver.SCIP_MIXED_INTEGER_PROGRAMMING,
    ):
        super().__init__(problem, params_objective_function)
        self.model: pywraplp.Solver = None
        self.variables: Dict[
            str, Dict[Hashable, Dict[Hashable, pywraplp.Variable]]
        ] = {}
        self.solver_name = solver_name

    def init_model(self, **kwargs):
        self.model: pywraplp.Solver = pywraplp.Solver.CreateSolver(self.solver_name)
        allocation_binary = {
            i: {
                j: self.model.BoolVar(name=f"allocation_{i}_{j}")
                for j in range(self.problem.number_of_slots)
            }
            for i in range(self.problem.number_of_containers)
        }
        for i in allocation_binary:
            self.model.Add(
                self.model.Sum([allocation_binary[i][j] for j in allocation_binary[i]])
                <= 1
            )
        loaded = {
            i: self.model.BoolVar(name=f"loaded_{i}")
            for i in range(self.problem.number_of_containers)
        }
        for i in allocation_binary:
            self.model.Add(loaded[i] == self.model.Sum(allocation_binary[i][j] for j in allocation_binary[i]))
        # Max total weight constraint :
        self.model.Add(self.model.Sum([loaded[j]*int(self.problem.weights[j]) for j in loaded])
                       <= self.problem.max_loading_weight)

        # Center of gravity
        total_weight = self.model.Sum([loaded[j]*int(self.problem.weights[j]) for j in loaded])
        cog = self.model.Sum([self.problem.weights[i]*self.problem.slot_distances[j]*allocation_binary[i][j]
                              for i in allocation_binary
                              for j in allocation_binary[i]])
        if kwargs.get("cog", True):
            self.model.Add(cog <= self.problem.cog_bounds[1]*total_weight)
            self.model.Add(cog >= self.problem.cog_bounds[0]*total_weight)

        # Cumulative/Overlaps
        if kwargs.get("overlap", True):
            for j in range(self.problem.number_of_slots):
                current_pos = self.problem.slot_distances[j]
                self.model.Add(
                    self.model.Sum(allocation_binary[i][k] * self.problem.heigth_container[i]
                                   for i in allocation_binary
                                   for k in allocation_binary[i]
                                   if
                                   self.problem.slot_distances[k] <= current_pos
                                   < self.problem.slot_distances[k] +
                                   self.problem.length_container[i])
                    <= 1
                )
        # Shear :
        if kwargs.get("shear", True):
            for j in range(self.problem.number_of_slots):
                current_pos = self.problem.slot_distances[j]
                indexes = set()
                if current_pos < 0:
                    indexes = set(range(j))
                if current_pos > 0:
                    indexes = set(range(j+1, self.problem.number_of_slots))
                weight_in_j = self.model.Sum([allocation_binary[i][j]*self.problem.weights[i]
                                              for i in allocation_binary])
                weights_further = self.model.Sum([allocation_binary[i][k]*self.problem.weights[i]
                                                  for i in allocation_binary
                                                  for k in indexes])
                self.model.Add(0.5*weight_in_j+weights_further <= self.problem.shear_profile(current_pos))
        self.model.Maximize(self.model.Sum([loaded[j]*int(self.problem.weights[j]) for j in loaded]))
        self.variables["allocation_binary"] = allocation_binary
        self.variables["loaded"] = loaded

    def retrieve_solutions(self, parameters_milp: ParametersMilp) -> ResultStorage:
        allocation_binary = self.variables["allocation_binary"]
        allocation = [-1 for i in range(self.problem.number_of_containers)]
        for i in allocation_binary:
            for j in allocation_binary[i]:
                val = allocation_binary[i][j].solution_value()
                if val >= 0.5:
                    allocation[i] = j
                    break
        sol = AircraftLoadingSolution(problem=self.problem, allocation=allocation)
        fit = self.aggreg_sol(sol)
        return ResultStorage(
            list_solution_fits=[(sol, fit)],
            mode_optim=self.params_objective_function.sense_function,
        )

    def solve(
        self, parameters_milp: Optional[ParametersMilp] = None, **kwargs: Any
    ) -> ResultStorage:
        if self.model is None:
            self.init_model(**kwargs)
        if parameters_milp is None:
            parameters_milp = ParametersMilp.default()
        self.model.SetTimeLimit(
            parameters_milp.time_limit * 1000
        )  # time limit is in ms in ortools/lp
        self.model.EnableOutput()
        res = self.model.Solve()
        resdict = {
            self.model.OPTIMAL: "OPTIMAL",
            self.model.FEASIBLE: "FEASIBLE",
            self.model.INFEASIBLE: "INFEASIBLE",
            self.model.UNBOUNDED: "UNBOUNDED",
            self.model.ABNORMAL: "ABNORMAL",
            self.model.MODEL_INVALID: "MODEL_INVALID",
            self.model.NOT_SOLVED: "NOT_SOLVED",
        }
        objective = self.model.Objective().Value()
        logger.info(f"Solver finished Result: {resdict[res]}")
        logger.info(f"Found objective value : {objective}")
        return self.retrieve_solutions(parameters_milp=parameters_milp)
