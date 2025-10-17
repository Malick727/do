#!/usr/bin/env python3.11
import numpy as np
import gurobipy as gp
import copy
import math
from gurobipy import GRB
from discrete_optimization.generic_tools.lp_tools import GurobiMilpSolver
from multibatching.problem import MultibatchingProblem, MultibatchingSolution, PackingTransport, TransportLink
from multibatching.solvers import MultibatchingSolver
from multibatching.solvers.column_generation_cp import ColumnGenerationSolver
from dataclasses import dataclass

@dataclass(frozen=True)
class BranchingRule:
    """ Store the rule : var_name sign value, e.g. : x_2 "lower" 2 """
    var_name : str
    sign : str
    value : int


class Node(ColumnGenerationSolver):
    def __init__(self, problem, listBranching, params_objective_function=None, **kwargs):
        super().__init__(problem, params_objective_function, **kwargs)
        self.init_model()
        self.branching_rules = listBranching
        for rule in self.branching_rules:
            self._apply_rule_to_model(rule)

    def _apply_rule_to_model(self, branching_rule : BranchingRule):
        """This helper method ONLY modifies the Gurobi model."""
        try:
            var = self.rp_model.getVarByName(branching_rule.var_name)
            if branching_rule.sign == "<=":
                self.rp_model.addConstr(var <= branching_rule.value)
                self.rp_model.update()
            elif branching_rule.sign == ">=":
                self.rp_model.addConstr(var >= branching_rule.value)
                self.rp_model.update()
            else:
                raise ValueError('The sign must be "<=" or ">="')
        except gp.GurobiError:
            print(f"Warning: Variable '{branching_rule.var_name}' not found in the model.")


    def is_sol_integer(self):
        integer = True
        all_vars = self.rp_model.getVars()
        i = 0
        fractional_vars = {}
        for var in all_vars:
            if var.VarName.startswith("x"):
                if not(math.isclose(var.X, math.ceil(var.X), rel_tol=1e-6) or math.isclose(var.X, math.floor(var.X), rel_tol=1e-6)):
                    integer = False
                    fractional_vars[var.VarName] = var.X
        return integer, fractional_vars
    

class BranchAndPriceSolver(MultibatchingSolver):

    problem : MultibatchingProblem

    def __init__(self, problem : MultibatchingProblem, params_objective_function = None, **kwargs):
        super().__init__(problem, params_objective_function, **kwargs)
        self.problem = problem
        self.best_value = math.inf #Minimization problem
        self.best_sol = None
        self.model = self._create_root_model()

    def _create_root_model(self):
        solver = ColumnGenerationSolver(self.problem)
        solver.init_model()
        return solver
    
    def solve(self, **kwargs):

        print("Solve the root problem")
        for var in self.model.rp_model.getVars():
            print(var.VarName)
        res = self.model.solve()
        for var in self.model.rp_model.getVars():
            print(var.VarName, var.X)
        self._explore(res)

        return self.best_sol, self.best_value
    
    def is_sol_integer(self):
        integer = True
        all_vars = self.model.rp_model.getVars()
        i = 0
        fractional_vars = {}
        for var in all_vars:
            if var.VarName.startswith("x"):
                if not(math.isclose(var.X, math.ceil(var.X), rel_tol=1e-6) or math.isclose(var.X, math.floor(var.X), rel_tol=1e-6)):
                    integer = False
                    fractional_vars[var.VarName] = var.X
        return integer, fractional_vars


    def select_next_branching_variable(self, fractional_vars):
        var_name =  list(fractional_vars.keys())[0]
        ceil = math.ceil(fractional_vars[var_name])
        floor = math.floor(fractional_vars[var_name])
        # Implement a better branching strategy
        return var_name, ceil, floor

    
    def _explore(self, res):
        
        if self.model.rp_model.Status == GRB.INFEASIBLE:
            return
        
        if self.model.rp_model.ObjVal >= self.best_value:
            return
        
        integer_sol, fractional_vars = self.is_sol_integer()

        if integer_sol:
            self.best_value = self.model.rp_model.ObjVal
            self.best_sol = res[-1][0]
            print(f"Found new integer solution: {self.best_value}")
            return
        else:
            var_name, ceil, floor = self.select_next_branching_variable(fractional_vars)
            print(f"Branch on {var_name}")
            print(f"Explore node {var_name} <= {floor} ")
            var = self.model.rp_model.getVarByName(var_name)
            c_inf = self.model.rp_model.addConstr(var <= floor)
            res = self.model.solve()
            # for var in self.model.rp_model.getVars():
            #     print(var.VarName, var.X)
            self._explore(res)

            # Backtrack
            self.model.rp_model.remove(c_inf)
            print(f"Backtracking, removed {var_name} <= {floor} ")


            print(f"Explore node {var_name} >= {ceil} ")
            c_sup = self.model.rp_model.addConstr(var>=ceil)
            res = self.model.solve()
            self._explore(res)

            # Backtrack
            self.model.rp_model.remove(c_sup)
            print(f"Backtracking, removed {var_name} >= {ceil} ")

         

    def get_final_solution(self):
        packing_solution = []
        for tl in self.problem.transport_links:
            # type_link = tl.transport_type
            for b in range(len(self.batches[tl])):
                nb_packing = self.rp_model.getVarByName(f"x_{tl.id}_b{b}").X
                if nb_packing>0:
                    packing_solution.append(PackingTransport(transport_link=tl, product_packing=self.batches[tl][b], nb_packing=nb_packing))
        return MultibatchingSolution(problem=self.problem, list_flows=packing_solution)




    def retrieve_current_solution(self, get_var_value_for_current_solution, get_obj_value_for_current_solution):
        packing_solution = []
        for tl in self.problem.transport_links:
            # type_link = tl.transport_type
            for b in range(len(self.batches[tl])):
                nb_packing = self.rp_model.getVarByName(f"x_{tl.id}_b{b}").X
                if nb_packing>0:
                    packing_solution.append(PackingTransport(transport_link=tl, product_packing=self.batches[tl][b], nb_packing=nb_packing))
        return MultibatchingSolution(problem=self.problem, list_flows=packing_solution)
        # raise NotImplementedError



   