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
# from multibatching.solvers.column_generation import ColumnGenerationSolver
from dataclasses import dataclass
from numpy.linalg import inv

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
        self.branching_constraints = {}
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


    

class BranchAndPriceSolver(MultibatchingSolver):

    problem : MultibatchingProblem

    def __init__(self, problem : MultibatchingProblem, params_objective_function = None, **kwargs):
        super().__init__(problem, params_objective_function, **kwargs)
        self.problem = problem
        self.best_value = math.inf #Minimization problem
        self.best_sol = None
        self.model = self._create_root_model()
        self.node_depth = 0
        self.max_cut = 1
        self.nb_current_cut = 0
        self.cut_ID = 0

    def _create_root_model(self):
        solver = ColumnGenerationSolver(self.problem)
        solver.init_model()
        return solver
    
    def solve(self, **kwargs):

        print("Solve the root problem")

        res = self.model.solve()
        # for t in self.problem.transport_links:
        #     print("Link ", t.id, self.model.batches[t])
        # for x in self.model.rp_model.getVars():
        #     print(x,x.VBasis)
        self._explore(res)
        self.node_depth += 1

        return self.best_sol, self.best_value
    
    def _is_integer(self, x):
        return (math.isclose(x, math.ceil(x), rel_tol=1e-6) or math.isclose(x, math.floor(x), rel_tol=1e-6))
    
    def _is_sol_integer(self):
        integer = True
        all_vars = self.model.rp_model.getVars()
        i = 0
        fractional_vars = {}

        max_fractionality = 0
        selected_var = None

        max_float = 0
        frac_value = 0
        greater_than_one = False
        var_selected_value = 0

        
        for var in all_vars:
            if var.VarName.startswith("x"):
                if not(self._is_integer(var.X)):
                    integer = False
                    val = var.X
                    frac = abs(val - round(val))
                    rest_part = val - math.floor(val)
                    fractionality = min(frac, 1 - frac)

                    # if rest_part>max_float:
                    #     selected_var = var.VarName
                    #     max_float = rest_part

                    if fractionality > 1e-6 and fractionality > max_fractionality:
                        max_fractionality = fractionality
                        selected_var = var.VarName
                        frac_value = rest_part
                        var_selected_value = var.X
                        
                    fractional_vars[var.VarName] = var.X
        if math.floor(var_selected_value)>=1:
            greater_than_one = True
                    


        return integer, fractional_vars, selected_var, frac_value, greater_than_one
    
    def _compute_rhs_float(self, fractional_vars, tl, p):
        rhs = 0
        for v_name, v_value in fractional_vars.items():
            _, tl_str, b = v_name.split("_")
            b = b.split("b")[1]
            if tl_str == tl:
                float_var = v_value - math.floor(v_value)
                rhs += float_var*self.model.batches[self.problem.transport_links_id_to_object[tl]][int(b)][p]
        return rhs


    def _select_next_branching_variable(self, fractional_vars, max_key, frac_value):

        var_name = max_key
        # var_name =  list(fractional_vars.keys())[0]
        ceil = math.ceil(fractional_vars[var_name])
        floor = math.floor(fractional_vars[var_name])
        # Implement a better branching strategy
        return var_name, ceil, floor, frac_value
    
    def _get_product_to_constrain(self, var_name, frac_value):
        _, tl, b = var_name.split("_")
        b = b.split("b")[1]
        batch = self.model.batches[self.problem.transport_links_id_to_object[tl]][int(b)]
        max_val = 0
        p_to_branch = 0
        for p,val in batch.items():
            if frac_value*val > max_val:
                p_to_branch = p
                max_val = int(frac_value*val)
        return p_to_branch, max_val
    
    def _get_fractionnal_part(self,x):
        if x>= 0 :
            return math.modf(x)[0]
        else:
            return abs(math.floor(x))+x
        
    

    def _compute_Gomory_cut(self, var_idx, idx_vbasis):
        A = self.model.rp_model.getA().toarray()
        B = np.array(A[:, idx_vbasis])

        print("size B", np.shape(B))
        print("size A", np.shape(A))

        # Get the aggregator vector B^Tv = e_k
        e_k = np.zeros(len(idx_vbasis))
        e_k[var_idx] = 1
        print("size e", np.shape(e_k))
        v = np.dot(inv(B.T),e_k.T)
        rhs = self.model.rp_model.getAttr("RHS",self.model.rp_model.getConstrs())
        coef = np.dot(v,A)

        cut_coef = np.array([self._get_fractionnal_part(x) for x in coef])
        cut_rhs = self._get_fractionnal_part(np.dot(v, rhs))

        return cut_coef, cut_rhs
    
    def _addCut(self, var_idx, idx_vbasis):
        cut_coef, cut_rhs = self._compute_Gomory_cut(var_idx, idx_vbasis)
        self.model.rp_model.addVar(lb=0, name = f"s_cut_{self.cut_ID}")
        self.model.rp_model.update()
        self.model.rp_model.addConstr(sum([cut_coef[i]* self.model.rp_model.getVars()[i] for i in range(len(self.model.rp_model.getVars())-1)]) + self.model.rp_model.getVarByName(f"s_cut_{self.cut_ID}")== cut_rhs)
        self.model.rp_model.add_cut_infos(f"s_cut_{self.cut_ID}", cut_coef)
        self.cut_ID += 1
        self.model.rp_model.update()


    def _cut(self):
        for v in self.model.rp_model.getVars():
            if not(self._is_integer(v.X)) and v.VarName.startswith("x"):
                var = v.VarName
                break
        if var != None:
            idx_vbasis = []
            i = 0
            nb_basis = 0
            for v in self.model.rp_model.getVars():
                if (v.VBasis)==0:
                    idx_vbasis.append(i)
                    if v.VarName == var:
                        var_idx = nb_basis
                    nb_basis += 1
                i+=1

            self._addCut(var_idx, idx_vbasis)
            self.model.solve()
            integer_sol, _, _ = self._is_sol_integer()
            while not(integer_sol) and (self.nb_current_cut < self.max_cut):
                self.nb_current_cut += 1
                self._cut()

    
    def _explore(self, res):
        
        if self.model.rp_model.Status == GRB.INFEASIBLE:
            return
        
        if self.model.rp_model.ObjVal >= self.best_value:
            print("Cut branch")
            return
        
        # if self.best_sol != None:
        #     return
        
        integer_sol, fractional_vars, next_var, frac_value, greater_than_one = self._is_sol_integer()

        if integer_sol:
            self.best_value = self.model.rp_model.ObjVal
            self.best_sol = res[-1][0]
            print(f"Found new integer solution: {self.best_value}")
            return
        else:

            # Generate cuts
            # print("Generate Gomory cuts")
            # self.nb_current_cut = 0
            # self._cut()
            
            # Branch
            # integer_sol, fractional_vars, next_var, frac_value, greater_than_one = self._is_sol_integer()
            var_name, ceil, floor, frac_value = self._select_next_branching_variable(fractional_vars, next_var, frac_value)
            
            print(f"Branch on {var_name}")
            var = self.model.rp_model.getVarByName(var_name)

            # if not(greater_than_one):
            ### Left branch
            print(f"Explore node {var_name} == {floor}")

            # Add forbidden assignment for the pricing problem
            # self.model.addForbiddenBatch(var_name)
            p_id, rhs = self._get_product_to_constrain(var_name, frac_value)
            _, tl, b = var_name.split("_")

            for p in self.model.products_on_links[self.problem.transport_links_id_to_object[tl]]:
                # Get floating part of each variable on the transport link
                rhs_float = self._compute_rhs_float(fractional_vars, tl, p)
                self.model.addBatchConstraint(self.problem.transport_links_id_to_object[tl], p, "<=", int(round(rhs_float)))

            # self.model.addBatchConstraint(self.problem.transport_links_id_to_object[tl], p_id, "<=", rhs)

            c_inf = self.model.rp_model.addConstr(var == floor, name=f"{var_name} == {floor}")
            self.model.rp_model.update()
            res = self.model.solve()
            # for t in self.problem.transport_links:
            #     print("Link ", t.id, self.model.batches[t])
            # for x in self.model.rp_model.getVars():
            #     print(x,x.VBasis)
            # for var in self.model.rp_model.getVars():
            #     print(var.VarName, var.X)
            self.node_depth += 1
            print("Node depth :", self.node_depth)
            self._explore(res)

            # Backtrack
            self.model.rp_model.remove(c_inf)
            self.node_depth -= 1
            
            # Remove forbidden assignment
            # self.model.removeForbiddenBatch(var_name)
            print(f"Backtracking, removed {var_name} == {floor} ")
            print("Node depth :", self.node_depth)

            # else:
            #     ### Right branch
            #     # print(f"Explore node {var_name} >= {ceil} ")
            #     print(f"Explore node {var_name} <= {floor} ")
            #     # c_sup = self.model.rp_model.addConstr(var>=ceil, name=f"{var_name} >= {ceil}")
            #     c_sup = self.model.rp_model.addConstr(var<=floor, name=f"{var_name} <= {floor}")
            #     res = self.model.solve()
            #     self.node_depth += 1
            #     print("Node depth :", self.node_depth)
            #     self._explore(res)

            #     # Backtrack
            #     self.model.rp_model.remove(c_sup)
            #     self.node_depth -= 1
            #     print(f"Backtracking, removed {var_name} >= {ceil} ")
            #     print("Node depth :", self.node_depth)


            

         

    def get_final_solution(self):
        return self.best_sol





    def retrieve_current_solution(self, get_var_value_for_current_solution, get_obj_value_for_current_solution):
        packing_solution = []
        for tl in self.problem.transport_links:
            # type_link = tl.transport_type
            for b in range(len(self.batches[tl])):
                nb_packing = self.rp_model.getVarByName(f"x_{tl.id}_b{b}").X
                if nb_packing>0:
                    packing_solution.append(PackingTransport(transport_link=tl, product_packing=self.batches[tl][b], nb_packing=nb_packing))
        return MultibatchingSolution(problem=self.problem, list_flows=packing_solution)



   