#!/usr/bin/env python3.11
import numpy as np
import gurobipy as gp
import copy
import math
from gurobipy import GRB
from discrete_optimization.generic_tools.lp_tools import GurobiMilpSolver
from multibatching.problem import MultibatchingProblem, MultibatchingSolution, PackingTransport, TransportLink
from multibatching.solvers import MultibatchingSolver
from multibatching.solvers.column_generation import ColumnGenerationSolver
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
        self.list_nodes : list[Node]= []
        self.best_value = math.inf #Minimization problem
        self.vars = []

    def init_model(self, **kwargs):
        # Init root node
        root_node = Node(self.problem, [])
        self.list_nodes.append(root_node)
        self.vars = [var.VarName for var in root_node.rp_model.getVars()]


    def select_next_branching_variable(self, fractional_vars):
        var_name =  list(fractional_vars.keys())[0]
        ceil = math.ceil(fractional_vars[var_name])
        floor = math.floor(fractional_vars[var_name])
        # Implement a better branching strategy
        return var_name, ceil, floor

    
    def solve(self, **kwargs):
        
        while len(self.list_nodes)>0:
            current_node = self.list_nodes.pop(0)
            print("Nb variables", len(current_node.rp_model.getVars()))
            print("nb constraints : ", len(current_node.rp_model.getConstrs()))
            res = current_node.solve()
            print("Nb variables", len(current_node.rp_model.getVars()))
            current_obj = current_node.rp_model.ObjVal

            if current_obj < self.best_value:
                # test if integer solution
                integer_sol, fractional_vars = current_node.is_sol_integer()
                if integer_sol:
                    self.best_value = current_obj
                    self.best_sol = res[-1][0]
                else:
                    # Select a variable
                    var_name, ceil, floor = self.select_next_branching_variable(fractional_vars)
                    print(f"Branch on {var_name}")
                    parent_rules = current_node.branching_rules

                    rules_inf = parent_rules + [BranchingRule(var_name, "<=", floor)]
                    node_inf = copy.deepcopy(current_node)
                    node_inf._apply_rule_to_model(BranchingRule(var_name, "<=", floor))
                    # node_inf = Node(self.problem, rules_inf)
                    print(f" -> Created child node ('inf') with {len(node_inf.branching_rules)} rules.")

                    rules_sup = parent_rules + [BranchingRule(var_name, ">=", ceil)]
                    node_sup = Node(self.problem, rules_sup)
                    print(f" -> Created child node ('sup') with {len(node_sup.branching_rules)} rules.")

                    self.list_nodes.append(node_inf)
                    self.list_nodes.append(node_sup)

    def _explore(self):
        # 1. Check current node's solution
        # (Assuming the model has just been solved)
        if self.rp_model.status == GRB.INFEASIBLE:
            return # Prune by infeasibility

            

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



   