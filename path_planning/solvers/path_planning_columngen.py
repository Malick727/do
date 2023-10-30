import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type

import gurobipy
from discrete_optimization.generic_tools.do_problem import (
    ModeOptim,
    ParamsObjectiveFunction,
)
from discrete_optimization.generic_tools.graph_api import Graph
from discrete_optimization.generic_tools.lp_tools import ParametersMilp
from discrete_optimization.generic_tools.result_storage.result_storage import (
    ResultStorage,
)
from gurobipy import GRB, Column, Constr, LinExpr, Model, Var, quicksum

from path_planning.path_planning import Comparison, PathPlanning, PathSolution
from path_planning.solvers.path_planning_milp import LPPathPlanning_Gurobi
from path_planning.solvers.path_planning_solvers import PathPlanningSolver

logger = logging.getLogger(__name__)

map_to_grb = {
    Comparison.UEQ: GRB.GREATER_EQUAL,
    Comparison.UP: GRB.GREATER_EQUAL,
    Comparison.LE: GRB.LESS_EQUAL,
    Comparison.LEQ: GRB.LESS_EQUAL,
    Comparison.EQ: GRB.EQUAL,
}


def set_parameters_milp(model: Model, parameters_milp: ParametersMilp):
    model.setParam(gurobipy.GRB.Param.TimeLimit, parameters_milp.time_limit)
    model.setParam(gurobipy.GRB.Param.MIPGapAbs, parameters_milp.mip_gap_abs)
    model.setParam(gurobipy.GRB.Param.MIPGap, parameters_milp.mip_gap)
    model.setParam(gurobipy.GRB.Param.PoolSolutions, parameters_milp.pool_solutions)
    model.setParam("PoolSearchMode", parameters_milp.pool_search_mode)


class CostShift:
    """
    Object to represent the cost shifts required for the subproblems. Notice that
    these shifts are done in 2 levels: the single arc level (i.e., changing the
    cost of each arc) and the overall level (i.e., adding a constant to the solution
    found). The latter does not change the optimal policy, thus it is added in the
    end. This overall shift is extremely important and failing to do so will remove
    the soundness and correctness of column generation
    """

    def __init__(
        self,
        constant: float,
        obj_fnc: float,
        constraints: dict,
        constraints_description: Dict[str, str],
    ):
        self._constant = constant
        self._obj_fnc = obj_fnc
        for cname in constraints:
            assert isinstance(constraints[cname], float) or isinstance(
                constraints[cname], int
            )
        self._constraints = constraints
        self.constraints_description = constraints_description

    def __str__(self):
        s = "CostShift(%f + %f * obj_fnc" % (self._constant, self._obj_fnc)
        for cname in sorted(self._constraints):
            s += " + %f * %s" % (self._constraints[cname], cname)
        s += ")"
        return s

    def get_constraint_shifts(self):
        """Returns the coefficients for the constraint shifts as a mapping from
        constraint to coefficient.
        """
        return self._constraints

    def get_obj_fct_coefficient(self):
        """Returns the coefficient for the shift of the objective function."""
        return self._obj_fnc

    def shift_arc_cost(self, main_cost: float, constraints_cost: dict) -> float:
        """Given the main cost of an edge and its costs for each constraint, returns
        the new shifted costs. The costs for the constraints are given as a dict of
        constraint name to float"""
        assert self._constraints.keys() == constraints_cost.keys()
        shifted_cost = self._obj_fnc * main_cost
        for cname in self._constraints:
            shifted_cost += self._constraints[cname] * constraints_cost[cname]
        return shifted_cost

    def shift_solution_cost(self, solution_cost: float) -> float:
        """Given the cost of a solution using the shiftArcCost, this method performs
        the last adjustment to obtain the overall cost"""
        return solution_cost + self._constant


class PathPlanningColumnGen(PathPlanningSolver):
    def __init__(
        self,
        problem: PathPlanning,
        params_objective_function: ParamsObjectiveFunction = None,
    ):
        super().__init__(problem, params_objective_function)
        self.model = None
        self.sub_solver_type: Type[PathPlanningSolver] = None
        self.sub_solver: PathPlanningSolver = None
        self.params_subsolver: Dict[str, Any] = None
        self.paths = []
        # list of gurobi vars representing each paths
        self.paths_vars = []
        self.reduce_master_model: Optional[Model] = None
        self.constraints: Optional[Dict[str, Constr]] = {}
        self.constraints_name_to_attr: Optional[Dict[str, str]] = {}
        self.included_solutions: List[PathSolution] = []
        self.included_solution_dict: Dict[str, PathSolution] = {}
        # Key is the same key as the variable corresponding to this solution, in the self.variables["solution"] dict
        self.variables: Dict[str, Dict[Any, Var]] = {}
        self.solutions_count = 0
        self.rmp_is_feasible = False

    def init_subsolver(self, subsolver: Type[PathPlanningSolver], **params_subsolver):
        self.sub_solver_type = subsolver
        self.sub_solver = subsolver(
            problem=self.problem,
            params_objective_function=self.params_objective_function,
        )
        self.params_subsolver = params_subsolver

    def initialize_master_model(self):
        reduce_master_model = Model("RMP")
        reduce_master_model.setParam("InfUnbdInfo", 1)
        self.variables["solution"] = {}
        initial_solution: ResultStorage = self.sub_solver.solve()
        best_solution, f = initial_solution.get_best_solution_fit()
        evaluation = self.problem.evaluate(best_solution)
        logger.info(f"Initial solution found : fitness={f}, evaluation={evaluation}")
        # building the left hand side of the constraints
        linear_expr_constraint_path = {}
        if not self.problem.path_constraints.is_empty():
            for attr in self.problem.path_constraints.attributes:
                linear_expr_constraint_path[attr] = LinExpr()
                pass
        self.included_solutions = []
        for solution, fit in initial_solution.list_solution_fits:
            self.included_solutions.append(solution)
            name_var = f"sol_{self.solutions_count}"
            self.included_solution_dict[name_var] = solution.copy()
            self.variables["solution"][name_var] = reduce_master_model.addVar(
                name=name_var, lb=0, ub=1, vtype=GRB.CONTINUOUS, obj=fit, column=None
            )
            for attr in linear_expr_constraint_path:
                value = self.problem.compute_cumulated_value_path(solution, attr)
                linear_expr_constraint_path[attr].add(
                    self.variables["solution"][name_var], value
                )
            self.solutions_count += 1

        reduce_master_model.update()
        reduce_master_model.addConstr(
            quicksum(list(self.variables["solution"].values())) == 1,
            name="sum_solution",
        )
        # building the constraints
        for j in range(self.problem.path_constraints.nb_constraints):
            c: Tuple[
                str, Comparison, int
            ] = self.problem.path_constraints.list_constraint[j]
            bound = c[2]
            if c[1] == Comparison.UP:
                bound = c[2] + 0.00001
            if c[1] == Comparison.LE:
                bound = c[2] - 0.00001
            name_constraint = f"path_constraint_{j}"
            self.constraints[name_constraint] = reduce_master_model.addLConstr(
                linear_expr_constraint_path[c[0]],
                sense=map_to_grb[c[1]],
                rhs=bound,
                name=name_constraint,
            )
            self.constraints_name_to_attr[name_constraint] = c[0]

        reduce_master_model.update()
        reduce_master_model.setParam("OutputFlag", False)
        reduce_master_model.modelSense = (
            GRB.MINIMIZE
            if self.params_objective_function.sense_function == ModeOptim.MINIMIZATION
            else GRB.MAXIMIZE
        )
        self.reduce_master_model = reduce_master_model

    def compute_reduced_cost(self) -> CostShift:
        shifts = {}
        for constr in self.constraints:
            shifts[constr] = -self.reduce_master_model.getConstrByName(constr).getAttr(
                "Pi"
            )
        return CostShift(
            -self.reduce_master_model.getConstrByName("sum_solution").getAttr("Pi"),
            1,
            shifts,
            self.constraints_name_to_attr,
        )

    def farkas_cost_shifts(self) -> CostShift:
        # should be called only in the
        assert self.reduce_master_model.getAttr("status") == GRB.INFEASIBLE
        shifts = {}
        for constr in self.constraints:
            shifts[constr] = self.reduce_master_model.getConstrByName(constr).getAttr(
                "FarkasDual"
            )
        return CostShift(
            self.reduce_master_model.getConstrByName("sum_solution").getAttr(
                "FarkasDual"
            ),
            0,
            shifts,
            self.constraints_name_to_attr,
        )

    def add_solution_to_rmp(self, solution: PathSolution):
        """Add a column, i.e. a (usually) new policy."""
        self.included_solutions.append(solution)

        # creating var lambda_p as a column and then adding it to the model
        column = Column()

        # add lambda_p to convexity constraint
        column.addTerms(1.0, self.reduce_master_model.getConstrByName("sum_solution"))

        for constr in self.constraints:
            attr = self.constraints_name_to_attr[constr]
            value = self.problem.compute_cumulated_value_path(
                attribute=attr, variable=solution
            )
            column.addTerms(value, self.constraints[constr])

        # Now adding the column as a new variable
        self.solutions_count += 1
        name_var = f"sol_{self.solutions_count}"
        self.included_solution_dict[name_var] = solution.copy()
        true_cost = self.problem.compute_cumulated_value_path(
            solution, self.problem.length_attribute
        )
        self.variables["solution"][name_var] = self.reduce_master_model.addVar(
            name=name_var,
            lb=0,
            ub=1,
            obj=true_cost,
            column=column,
            vtype=GRB.CONTINUOUS,
        )
        self.reduce_master_model.update()

    def solve(
        self, parameters_milp: Optional[ParametersMilp] = None, **kwargs
    ) -> ResultStorage:
        if parameters_milp is None:
            parameters_milp = ParametersMilp.default()
        if self.sub_solver is None:
            self.init_subsolver(subsolver=LPPathPlanning_Gurobi)
        if self.reduce_master_model is None:
            self.initialize_master_model()
        iteration = 0
        largest_lb = float("-inf")
        # smallest_ub = float("inf")
        while True:
            set_parameters_milp(
                model=self.reduce_master_model, parameters_milp=parameters_milp
            )
            self.reduce_master_model.optimize()
            rmp_status = self.reduce_master_model.getAttr("status")
            if rmp_status == GRB.OPTIMAL:
                cur_cost = self.reduce_master_model.getAttr("ObjVal")
                logger.info("Found solution of RMP with cost = {}".format(cur_cost))
                self.rmp_is_feasible = True
                # Since this is a minimization problem, column generation finds
                # solutions with monotonically non-increasing cost. In other
                # words, at each iteration the solution cost cannot increase
                # because the previous solutions are still available for the
                # solver
                # assert cur_cost <= smallest_ub + 1e-6
                smallest_ub = cur_cost
                cost_shifts = self.compute_reduced_cost()
                logger.debug(
                    "Solving reduced cost problem with cost shifts {}".format(
                        cost_shifts
                    )
                )
                new_problem = create_new_problem(
                    original_problem=self.problem, cost_shifts=cost_shifts
                )
                params_objective_function = ParamsObjectiveFunction(
                    objective_handling=self.params_objective_function.objective_handling,
                    objectives=["length"],
                    weights=[1],
                    sense_function=self.params_objective_function.sense_function,
                )
                self.sub_solver = self.sub_solver_type(
                    problem=new_problem,
                    params_objective_function=params_objective_function,
                )
                result = self.sub_solver.solve(**self.params_subsolver)
                best_solution, fit = result.get_best_solution_fit()
                print("Fit ", fit, new_problem.evaluate(best_solution), new_problem.satisfy(best_solution))
                min_rc = new_problem.compute_cumulated_value_path(
                    attribute=new_problem.length_attribute, variable=best_solution
                )
                min_rc = cost_shifts.shift_solution_cost(min_rc)
                logger.info(f"Solution with costs shift : {best_solution}, fit={fit}")
                cur_lb = cur_cost + min_rc
                if largest_lb < cur_lb:
                    # the lower bounds are not guaranteed to be monotonically
                    # non-increasing, i.e., it might decrease from iteration to
                    # iteration. However, all the lower bounds observed are
                    # correct, thus we keep track of the largest lower bound
                    # seen so far
                    largest_lb = cur_lb
                logger.info("Opt. Sol. bound: [%f, %f]" % (largest_lb, smallest_ub))
                # Once the reduced costs are positive we have a solution
                if min_rc >= 0:
                    print(min_rc, " quit ")
                    logger.info("Min rc >0, quit the algo")
                    break
                # Consider only reduced costs below 0+ epsilon to deal with
                # numeric instability
                epsilon = 1e-7
                if min_rc + epsilon >= 0:
                    print(min_rc, " quit 2")
                    logger.info("Reduced costs are epsilon-close to 0., leaving")
                    break
                for sol, fit in result.list_solution_fits:
                    self.add_solution_to_rmp(solution=sol)
            else:
                # Using Farkas cost
                assert rmp_status in [GRB.INFEASIBLE, GRB.INF_OR_UNBD]
                # Once a single feasible solution for the RMP is found, the RMP
                # is never infeasible again
                assert not self.rmp_is_feasible
                cost_shifts = self.farkas_cost_shifts()
                logger.info(
                    "Solving reduced cost problem with cost shifts {}".format(
                        cost_shifts
                    )
                )
                new_problem = create_new_problem(
                    original_problem=self.problem, cost_shifts=cost_shifts
                )
                params_objective_function = ParamsObjectiveFunction(
                    objective_handling=self.params_objective_function.objective_handling,
                    objectives=["length"],
                    weights=[1],
                    sense_function=self.params_objective_function.sense_function,
                )
                self.sub_solver = self.sub_solver_type(
                    problem=new_problem,
                    params_objective_function=params_objective_function,
                )
                result = self.sub_solver.solve(**self.params_subsolver)
                best_solution, fit = result.get_best_solution_fit()
                print("Path", best_solution.path)
                print("evaluation : ", self.problem.evaluate(best_solution))
                min_fc = new_problem.compute_cumulated_value_path(
                    attribute=new_problem.length_attribute, variable=best_solution
                )
                logger.info(f"Obtained Farkas cost before shift :  {min_fc}")
                min_fc = cost_shifts.shift_solution_cost(min_fc)
                logger.info(
                    f"Solving Farkas cost problem with cost shifts {cost_shifts}"
                )
                logger.info(f"Obtained Farkas cost:  {min_fc}")
                if min_fc >= 0:
                    logger.info(
                        "\n\nFarkas Cost is >= 0, Thus the original problem "
                        "is INFEASIBLE.\n"
                    )
                    return

                # epsilon = 1e-7
                # if min_fc + epsilon >= 0:
                #    logger.info("\n\nFarkas Cost + epsilon is >= 0, Thus the original "
                #                "problem is very likely to be INFEASIBLE.\n")
                #    return
                for sol, fit in result.list_solution_fits:
                    self.add_solution_to_rmp(solution=sol)
            iteration += 1
        logger.info(f"Final value {cur_cost}")
        taken_sol = set()
        for sol in self.variables["solution"]:
            var: Var = self.variables["solution"][sol]
            if var.getAttr("X") > 0.1:
                logger.info(f"{sol} has been taken.")
                logger.info(f"{var.getAttr('X')} has been taken")
                taken_sol.add(sol)
        logger.info(f"Taken sols {taken_sol}")
        # I want one unique solution..
        assert len(taken_sol) == 1
        id_sol = taken_sol.pop()
        solution = self.included_solution_dict[id_sol]
        solution.change_problem(self.problem)
        fit = self.aggreg_sol(solution)
        return ResultStorage(
            list_solution_fits=[(solution, fit)],
            mode_optim=self.params_objective_function.sense_function,
        )


# Brutal way of building a new problem with new cost function


def create_new_problem(original_problem: PathPlanning, cost_shifts: CostShift):
    new_edges = []
    for e in original_problem.graph.edges:
        attrs = cost_shifts.constraints_description.values()
        costs = {attr: e[2][attr] for attr in attrs}
        shifted_cost = cost_shifts.shift_arc_cost(
            main_cost=e[2][original_problem.length_attribute],
            constraints_cost={
                constraint_name: costs[
                    cost_shifts.constraints_description[constraint_name]
                ]
                for constraint_name in cost_shifts.constraints_description
            },
        )
        ww = deepcopy(e[2])
        ww["shifted_cost"] = shifted_cost
        new_edges += [(e[0], e[1], ww)]
    new_graph = Graph(
        nodes=original_problem.graph.nodes,
        edges=new_edges,
        undirected=original_problem.graph.undirected,
        compute_predecessors=False,
    )
    return PathPlanning(
        graph=new_graph,
        source=original_problem.source,
        sink=original_problem.sink,
        length_attribute="shifted_cost",
        node_constraints=original_problem.node_constraints,
        path_constraints=original_problem.path_constraints,
        reindex_node_constraints=False
    )
