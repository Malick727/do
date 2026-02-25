#!/usr/bin/env python3.11
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from discrete_optimization.generic_tools.lp_tools import GurobiMilpSolver
from multibatching.problem import MultibatchingProblem, MultibatchingSolution, PackingTransport, TransportLink, Product
from multibatching.solvers import MultibatchingSolver
from ortools.sat.python import cp_model
from ortools.sat.python.cp_model import CpModel, CpSolver
from typing import List
import copy
# import multiprocessing
import dill
from pathos.multiprocessing import Pool
# from multiprocessing import Pool
# multiprocessing.set_start_method('fork')


# from concurrent.futures import ProcessPoolExecutor
# from functools import partial






class Constraint(object):
    sens = ""
    rhs = 0

    # The class "constructor" - It's actually an initializer 
    def __init__(self, sens, rhs):
        self.sens = sens
        self.rhs = rhs

def pricing_greedy(data):

    found_patterns = []
    transport_link, duals, branch_cons, forbidden_assignment, max_demand = data
    
    # # Use BasePricer helper
    # capacity, duration, fixed_cost, valid_products = self.prepare_subproblem_data(route)
    
    # if not valid_products:
    #     continue

    candidates = []
    for p, p_size in transport_link['products']:
        mu = duals[transport_link["id"]][p]

        # Net impact on Reduced Cost (negative is good)
        net_cost_per_unit = - mu
        
        if net_cost_per_unit < 0:
            candidates.append((p, net_cost_per_unit, p_size))

    # Sort by Efficiency: Most negative cost per unit of size
    candidates.sort(key=lambda x: x[1] / x[2])

    current_load = 0.0
    pattern = {}
    
    for prod, cost_impact, size in candidates:
        if current_load + size <= capacity:
            qty = int((capacity - current_load) // size)
            if qty > 0:
                pattern[prod] = float(qty)
                current_load += qty * size

    # if pattern:
    #     variable_part_rc = sum(
    #         qty * (CostCalculator.calculate_interest(p.value, duration, self.settings) 
    #                 - duals.get((route.transport_link, p), 0))
    #         for p, qty in pattern.items()
    #     )
        
    #     total_rc = fixed_cost + variable_part_rc
        
    #     # Check Result using settings.epsilon
    #     if total_rc < -self.settings.epsilon:
    #         found_patterns.append({
    #             'route': route,
    #             'pattern': pattern,
    #             'duration': duration,
    #             'fixed_cost': fixed_cost
    #         })

    return found_patterns

def pricing_worker(data):
    """
    data is a tuple: (link_info, dual_values, branching_constraints, forbidden)
    """
    transport_link, duals, branch_cons, forbidden_assignment, max_demand = data
    
    model = CpModel()
    y = {}
    # Use IDs or local dicts, don't pass Gurobi objects here
    for p, p_size in transport_link['products']:
        y[p] = model.NewIntVar(0, int(transport_link['capacity'] // p_size), f"y_{p}")
        # Apply branching constraints from the passed dict
        for c in branch_cons[p]:
            if c.sens == "<=":
                model.Add(y[p] <= c.rhs)
            else:
                model.Add(y[p] >= c.rhs)
        model.add(y[p]<=max_demand[p])

    model.add(sum(y[p] * p_size for p, p_size in transport_link['products']) <= int(transport_link['capacity']))

    if len(forbidden_assignment)>0:
        model.add_forbidden_assignments([y[p] for p,_ in transport_link['products']], forbidden_assignment)
    
    
    # Define Objective: (Cost + Emissions) * Dist - Sum(y * dual_diff)
    obj_terms = []
    for p, _ in transport_link['products']:
        dual_diff = duals[transport_link['l2']][p] - duals[transport_link['l1']][p]
        obj_terms.append(y[p] * int(dual_diff * 10000)) # Scale if using CP-SAT (ints only)

    model.Minimize(transport_link['base_cost'] * 10000 - sum(obj_terms))
    
    solver = CpSolver()
    status = solver.Solve(model)
    
    if status in [4, 2]: # Optimal or Feasible
        return {p: solver.Value(y[p]) for p, _ in transport_link['products']}, solver.ObjectiveValue() / 10000
    return None, 0


class ColumnGenerationSolver(MultibatchingSolver):

    problem : MultibatchingProblem

    def __init__(self, problem : MultibatchingProblem, params_objective_function = None, **kwargs):
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}
        self.batches = {}
        self.reduced_costs = {}
        self.dual_values = {}
        self.sizes = {}
        self.forbidden_assignment = {}
        self.dual_values_cuts = {}
        self.coef_cuts = {}
        self.demand_tot = {}
        self.branching_constraints: dict[TransportLink, dict[Product, list[Constraint]]] = {}
        self.transport_links_active = {}


    def init_model(self, init_solution = None, flows_on_demand=False, **kwargs):
        model = gp.Model()
        max_frequency = 10 #Maximum frequency for each transport link, identical for all at the moment

        ###### Set initialization
        transport_links_in = {}
        transport_links_out = {}
        products_on_locations = {}
        
        for l in self.problem.locations:
            self.dual_values[l] = {}
            products_on_locations[l]=[]
            for p in self.problem.products:
                self.dual_values[l][p] = 0        
            transport_links_out[l.id]=[]
            transport_links_in[l.id] = []


        for tl in self.problem.transport_links:
            self.forbidden_assignment[tl] = []
            l1 = tl.location_l1.id
            l2 = tl.location_l2.id
            transport_links_out[l1].append(tl)
            transport_links_in[l2].append(tl)

        for p in self.problem.products:
            self.coef_cuts[p] = {}
            self.demand_tot[p] = 0
            for l in self.problem.locations:
                if p in l.net_supply.keys():
                    if l.net_supply[p] < 0 :
                        self.demand_tot[p] += -l.net_supply[p]

        # Flow on links per product
        products_on_links = {}
        for tl in self.problem.transport_links:
            self.reduced_costs[tl] = 0
            products_on_links[tl]=[]
            for p in self.problem.products:
                if tl.transport_type in p.valid_transports:
                    products_on_links[tl].append(p)
                    l1 = tl.location_l1
                    l2 = tl.location_l2
                    if not(p in products_on_locations[l1]):
                        products_on_locations[l1].append(p)
                    if not(p in products_on_locations[l2]):
                        products_on_locations[l2].append(p)
        self.products_on_links = products_on_links
        self.products_on_locations = products_on_locations

        for tl in self.problem.transport_links:
            self.branching_constraints[tl] = {}
            for p in products_on_links[tl]:
                self.branching_constraints[tl][p] = []

        # Batch initialization
        
        batches = {}
        batches_tt = {}

        for tt in self.problem.transport_types:
            batches_tt[tt] = []
            for p1 in self.problem.products:
                if tt in p1.valid_transports:
                    batch_init = {}
                    # batch_init[p1] = tt.capacity//p1.size
                    batch_init[p1] = 0
                    for p2 in self.problem.products:
                        if tt in p2.valid_transports:
                            if p1 != p2:
                                batch_init[p2] = 0
                    batches_tt[tt].append(batch_init)
        for tl in self.problem.transport_links:
             batches[tl] = copy.copy(batches_tt[tl.transport_type])

        list_tl = []

        if init_solution != None:
            for packing_transport in init_solution.list_flows:
                tl_id = packing_transport.transport_link.id
                tl = self.problem.transport_links_id_to_object[tl_id]
                list_tl.append(tl)
                new_batch = {}
                for p, nb in packing_transport.product_packing.items():
                    new_batch[p] = nb
                for p in self.products_on_links[tl]:
                    if p in new_batch.keys():
                        continue
                    new_batch[p] = 0
                batches[tl].append(new_batch)

        self.batches = batches
        for tl in list_tl:
            print(len(self.batches[tl]))


        # for transport_link in self.problem.transport_links:
        #     batches[transport_link]=[]

        #     for p1 in self.products_on_links[transport_link]:
        #         batch_init = {}
        #         batch_init[p1] = transport_link.transport_type.capacity//p1.size
        #         for p2 in self.products_on_links[transport_link]:
        #             if p1 != p2:
        #                 batch_init[p2] = 0
        #         batches[transport_link].append(batch_init)
        #         self.forbidden_assignment[transport_link].append([batch_init[p] for p in self.products_on_links[transport_link]])
        # self.batches = batches
        
        
        ################################## Decision variable definition
        y  = {}
        flow = {}
        for tl in self.problem.transport_links:
            y[tl] = {}
            flow[tl] = {}
            for p in products_on_links[tl]:
                y[tl][p] = model.addVar(lb=0, name=f"y_{tl.id}_{p}")
                flow[tl][p] = model.addVar(lb=0, name=f"f_{tl.id}_{p}")


        # Batch selection per arc
        x={}
        for tl in self.problem.transport_links:
                x[tl] = {}
                # type_link = tl.transport_type
                for b in range(len(self.batches[tl])):
                    x[tl][b] = model.addVar(lb=0, name=f"x_{tl.id}_b{b}")


        self.variables["x"] = x

        # Slack variable
        s = {}
        for l in self.problem.locations:
            s[l] = {}
            for p in self.problem.products:
                s[l][p] = model.addVar(lb=0, name=f"s_{l}_{p}")
        self.variables["s"] = s


        mu = {}
        
        bigM = 1e9

        print('Variables created')

        ##################################### Constraints
        for l in self.problem.locations:
            for p in self.products_on_locations[l]:
                if p in l.net_supply.keys():
                    demand_p = -l.net_supply[p]
                    if demand_p > 0:
                        model.addConstr(gp.quicksum(gp.quicksum(x[tl][b]*self.batches[tl][b][p] for b in range(len(self.batches[tl]))) for tl in transport_links_in[l.id] if p in self.products_on_links[tl])-gp.quicksum(gp.quicksum(x[tl][b]*self.batches[tl][b][p] for b in range(len(self.batches[tl]))) for tl in transport_links_out[l.id] if p in self.products_on_links[tl]) + s[l][p] >=  demand_p, name=f"demand_{l.id}_p{p.id}")
                        # model.addConstr(gp.quicksum(gp.quicksum(x[tl][b]*self.batches[tl][b][p] for b in range(len(self.batches[tl]))) for tl in transport_links_in[l.id])-gp.quicksum(gp.quicksum(x[tl][b]*self.batches[tl][b][p] for b in range(len(self.batches[tl]))) for tl in transport_links_out[l.id]) + s[l][p] ==  demand_p, name=f"demand_{l.id}_p{p.id}")
                        # model.addConstr(s[l][p] <= gp.quicksum(x[tl][b]*self.batches[tl][b][p] for b in range(len(self.batches[tl]))) - demand_p)
                    else:
                        model.addConstr(gp.quicksum(gp.quicksum(x[tl][b]*self.batches[tl][b][p] for b in range(len(self.batches[tl]))) for tl in transport_links_in[l.id] if p in self.products_on_links[tl])-gp.quicksum(gp.quicksum(x[tl][b]*self.batches[tl][b][p] for b in range(len(self.batches[tl]))) for tl in transport_links_out[l.id] if p in self.products_on_links[tl]) - s[l][p] <=  demand_p, name=f"demand_{l.id}_p{p.id}")
                        # model.addConstr(s[l][p] <= gp.quicksum(x[tl][b]*self.batches[tl][b][p] for b in range(len(self.batches[tl]))) + demand_p)
                else:
                        model.addConstr(gp.quicksum(gp.quicksum(x[tl][b]*self.batches[tl][b][p] for b in range(len(self.batches[tl]))) for tl in transport_links_in[l.id] if p in self.products_on_links[tl])-gp.quicksum(gp.quicksum(x[tl][b]*self.batches[tl][b][p] for b in range(len(self.batches[tl]))) for tl in transport_links_out[l.id] if p in self.products_on_links[tl]) ==  0, name=f"demand_{l.id}_p{p.id}")
                    

        # for tl in self.problem.transport_links:
        #     for p in products_on_links[tl]:
        #         model.addConstr(y[tl][p] - flow[tl][p] >= 0)


        # for l in self.problem.locations:
        #     for p in self.problem.products:
        #         if p in l.net_supply.keys():
        #             demand_p = -l.net_supply[p]
        #             if demand_p > 0:
        #                 model.addConstr(gp.quicksum(gp.quicksum(x[tl][b]*self.batches[tl][b][p] for b in range(len(self.batches[tl]))) for tl in transport_links_in[l.id])-gp.quicksum(gp.quicksum(x[tl][b]*self.batches[tl][b][p] for b in range(len(self.batches[tl]))) for tl in transport_links_out[l.id]) + s[l][p] ==  demand_p, name=f"demand_{l.id}_p{p.id}")
        #             else:
        #                 model.addConstr(gp.quicksum(gp.quicksum(x[tl][b]*self.batches[tl][b][p] for b in range(len(self.batches[tl]))) for tl in transport_links_out[l.id])-gp.quicksum(gp.quicksum(x[tl][b]*self.batches[tl][b][p] for b in range(len(self.batches[tl]))) for tl in transport_links_in[l.id]) + s[l][p] ==  -demand_p, name=f"demand_{l.id}_p{p.id}")
        #         else:
        #             model.addConstr(gp.quicksum(gp.quicksum(x[tl][b]*self.batches[tl][b][p] for b in range(len(self.batches[tl]))) for tl in transport_links_in[l.id])-gp.quicksum(gp.quicksum(x[tl][b]*self.batches[tl][b][p] for b in range(len(self.batches[tl]))) for tl in transport_links_out[l.id]) + s[l][p] ==  0, name=f"demand_{l.id}_p{p.id}")


        # Set objective
        # obj_co2 = model.addVar(name="obj_co2")
        # model.addConstr(gp.quicksum(x[tl][b]*tl.transport_type.emissions * tl.distance for tl in self.problem.transport_links for b in range(len(self.batches[tl])))==obj_co2, "co2_cost_constraint")

        # obj_transport = model.addVar(name="obj_transport")
        # model.addConstr(gp.quicksum(x[tl][b]*tl.transport_type.cost*tl.distance for tl in self.problem.transport_links for b in range(len(self.batches[tl])))==obj_transport, "transport_cost_constraint")


        # obj_affectation = model.addVar(name="obj_affectation")
        # model.addConstr(bigM*gp.quicksum(s[l][p] for l in self.problem.locations for p in self.problem.products)==obj_affectation, "affectation_constraint")

        # obj_capital = model.addVar(name="obj_capital")
        # capital_cost = {}


        # model.setObjective(obj_co2 + obj_transport + obj_affectation)
        model.setObjective(gp.quicksum(x[tl][b]*tl.transport_type.emissions * tl.distance for tl in self.problem.transport_links for b in range(len(self.batches[tl])))+gp.quicksum(x[tl][b]*tl.transport_type.cost*tl.distance for tl in self.problem.transport_links for b in range(len(self.batches[tl])))+bigM*gp.quicksum(s[l][p] for l in self.problem.locations for p in self.problem.products))
        # model.setObjective(obj_co2 + obj_transport)
        model.Params.LogToConsole = 0
        model.update()
        self.rp_model = model
        
        print("Model created")

    
    def solveRP(self):
        self.rp_model.update()
        self.rp_model.optimize() 
        return self.rp_model.Status
    
    def add_cut_infos(self, name_cut, coeff ):
        self.coef_cuts[name_cut] = {}
        # for 

    def addForbiddenBatch(self, varName):
        _, transport_link_ID, batch_ID = varName.split("_")
        transport_link = self.problem.transport_links_id_to_object[transport_link_ID]
        batch_number = int(batch_ID.split("b")[1])
        batch = self.batches[transport_link][batch_number]
        self.forbidden_assignment[transport_link].append([batch[p] for p in self.products_on_links[transport_link]])
        # print("Forbidden assignment added :",self.forbidden_assignment[transport_link])

    
    def removeForbiddenBatch(self, varName):
        _, transport_link_ID, batch_ID = varName.split("_")
        transport_link = self.problem.transport_links_id_to_object[transport_link_ID]
        batch_number = int(batch_ID.split("b")[1])
        batch = self.batches[transport_link][batch_number]
        self.forbidden_assignment[transport_link].remove(batch)

    def addBatchConstraint(self, tl, p, sens, RHS):
        self.branching_constraints[tl][p].append(Constraint(sens,RHS))

    def removeBatchConstraint(self, tl, p, sens, RHS):
        for c in self.branching_constraints[tl][p]:
            if (c.rhs == RHS) and c.sens == (sens):
                break
        self.branching_constraints[tl][p].remove(c)



    def solvePricingProblem(self, transport_link : TransportLink):
        """ forbidden_assignment : columns already in basis """

        #### Solve the problem
        model = CpModel()
        y={}
        max_missing = {}
        for p in self.products_on_links[transport_link]:
            y[p] = model.new_int_var(0, int(transport_link.transport_type.capacity//p.size), name=f"y_{p.id}")
            max_missing[p] = sum(self.variables["s"][l][p].X for l in self.problem.locations_pos_demand[p])
            for c in self.branching_constraints[transport_link][p]:
                if c.sens == "<=":
                    model.add(y[p] <= c.rhs)
                else:
                    model.add(y[p]>= c.rhs)
        model.add(sum([y[p]*int(p.size) for p in self.products_on_links[transport_link]]) <= int(transport_link.transport_type.capacity))
        for p in self.products_on_links[transport_link]:
            model.add(y[p] <= self.demand_tot[p])

        if len(self.forbidden_assignment[transport_link]) > 0:
            # print("Forbidden assignment :", self.forbidden_assignment[transport_link])
            model.add_forbidden_assignments([y[p] for p in self.products_on_links[transport_link]], self.forbidden_assignment[transport_link])
            
        # for p in self.products_on_links[transport_link]:
        #     if int(max_missing[p]) >0:
        #         model.add(y[p]<=int(max_missing[p]))

            
        model.minimize(-sum(y[p]*(self.dual_values[transport_link.location_l2][p]-self.dual_values[transport_link.location_l1][p]) for p in self.products_on_links[transport_link]))
        
        solver = CpSolver()
        print("Solved before")
        status = solver.solve(model)
        print("Solved")

        new_column = {}
        for p in self.products_on_links[transport_link]:
            new_column[p] = solver.value(y[p])
        # reduced_cost = solver.objective_value + (transport_link.transport_type.cost+transport_link.transport_type.emissions)*transport_link.distance
        reduced_cost = solver.objective_value + (transport_link.transport_type.cost+transport_link.transport_type.emissions)*transport_link.distance

        return new_column, reduced_cost
    
    def solvePricingProblemMILP(self, transport_link : TransportLink):

        #### Solve the problem
        model = gp.Model()
        model.Params.LogToConsole = 0
        y={}
        for p in self.products_on_links[transport_link]:
            y[p] = model.addVar(lb=0,ub=1000, vtype=GRB.INTEGER, name=f"y_{p.id}")
        
        model.addConstr(gp.quicksum(y[p]*p.size for p in self.products_on_links[transport_link]) <= transport_link.transport_type.capacity)
        model.setObjective((transport_link.transport_type.cost+transport_link.transport_type.emissions)*transport_link.distance-gp.quicksum(y[p]*(self.dual_values[transport_link.location_l2][p]-self.dual_values[transport_link.location_l1][p]) for p in self.products_on_links[transport_link]))
        model.optimize()

        new_column = {}
        for p in self.products_on_links[transport_link]:
            new_column[p] = model.getVarByName(f"y_{p.id}").X
        reduced_cost = model.ObjVal
        return new_column, reduced_cost

    # def solveSubproblems(self, list_links, current_obj): 

    #     new_columns = {}
    #     nb_suproblems = len(list_links)
    #     i=0

    #     # for transport_link in list_links:

    #     #     # Solve pricing problem
    #     #     new_column, reduced_cost = self.solvePricingProblem(transport_link)
    #     #     percentage_reduced_cost = reduced_cost/current_obj
    #     #     # if reduced_cost < -1e-6: 
    #     #     if percentage_reduced_cost < -0.0001: 
    #     #         new_columns[transport_link] = new_column
    #     #         i+=1
    #     #     self.reduced_costs[transport_link] = reduced_cost

    #     # Parallel
        

    #     pool = Pool(10)
    #     for result in pool.map(self.solvePricingProblem, list_links):
    #         print(result)
            
    #     print("------------ Number of columns added", i)

    #     return new_columns
    

    def solveSubproblems(self, list_links : List[TransportLink], current_obj):
        new_columns = {}
        # 1. Prepare minimal data payloads
        payloads = []
        for tl in list_links:
            payloads.append((
                {
                    'id': tl.id,
                    'l1': tl.location_l1,
                    'l2': tl.location_l2,
                    'capacity': int(tl.transport_type.capacity),
                    'base_cost': int((tl.transport_type.cost + tl.transport_type.emissions) * tl.distance),
                    'products': [(p, int(p.size)) for p in self.products_on_links[tl]]
                },
                self.dual_values, # Ensure this is a dict of dicts (floats), not Gurobi vars
                self.branching_constraints[tl], # Helper to convert objects to dicts
                self.forbidden_assignment[tl],
                self.demand_tot
            ))

        # 2. Use the Pool
        with Pool(processes=min(10, len(self.problem.products))) as pool:
            results = pool.map(pricing_worker, payloads)
            # results = pool.map(pricing_greedy, payloads)

        # 3. Process results
        for tl, (col_data, red_cost) in zip(list_links, results):
            if col_data and (red_cost / current_obj < -0.0001):
                new_columns[tl] = col_data
                self.reduced_costs[tl] = red_cost
                
        return new_columns

    def addColumns(self, new_columns):

        for transport_link in new_columns.keys():
            if self.reduced_costs[transport_link]<-1e-6:
                nb_batches = len(self.batches[transport_link])
                self.batches[transport_link].append(new_columns[transport_link])
                c = gp.Column()
                l1 = transport_link.location_l1
                l2 = transport_link.location_l2
                for p in self.products_on_links[transport_link]:
                    constr = self.rp_model.getConstrByName(f"demand_{l1.id}_p{p.id}")
                    coeff = new_columns[transport_link][p]
                    c.addTerms(-coeff, constr)

                    constr = self.rp_model.getConstrByName(f"demand_{l2.id}_p{p.id}")
                    coeff = new_columns[transport_link][p]
                    c.addTerms(coeff, constr)
                self.forbidden_assignment[transport_link].append([new_columns[transport_link][p] for p in self.products_on_links[transport_link]])

                        
                cost = (transport_link.transport_type.emissions + transport_link.transport_type.cost) * transport_link.distance 
                self.rp_model.addVar(obj=cost, lb=0, name=f"x_{transport_link.id}_b{nb_batches}", column=c)
                self.rp_model.update()
                
    
    def solve(self, **kwargs):

        optimal = False
        list_links = self.problem.transport_links

        # Add branching strategy
        # print("nb variables",len(self.rp_model.getVars()))
        
        while not(optimal):

            # Solve RP
            status = self.solveRP()
            
            if status == GRB.OPTIMAL:
                print("RP solved with objective ---- >", self.rp_model.ObjVal)

            # Compute reduced costs and store dual variables
            for l in self.problem.locations:
                for p in self.products_on_locations[l]:
                    c = self.rp_model.getConstrByName(f"demand_{l.id}_p{p.id}")
                    self.dual_values[self.problem.location_id_to_location[l.id]][self.problem.product_dict[p.id]] = c.Pi
            
            # for cut_name in self.dual_values_cuts.keys():
            #     c = self.rp_model.getConstrByName(cut_name)
            #     self.dual_values_cuts[c] = c.Pi



            new_columns = self.solveSubproblems(list_links, self.rp_model.ObjVal)
            
            optimal = True
            min_reduced_cost = 0
            # list_links = []

            # for tl in self.problem.transport_links:
            #     if self.reduced_costs[tl]<-1e-5:
            #         if self.reduced_costs[tl] < min_reduced_cost:
            #             min_reduced_cost = self.reduced_costs[tl]
            #         optimal = False

            if len(new_columns.keys())>0:
                for tl in new_columns.keys():
                    if self.reduced_costs[tl] < min_reduced_cost:
                        min_reduced_cost = self.reduced_costs[tl]
                    # list_links.append(tl)
                optimal = False

            print("----------- Min reduced cost :", min_reduced_cost)
            
            if not(optimal):
                self.addColumns(new_columns)
                print("Number of new columns :", len(new_columns))

        sol = self.get_final_solution()
        # print("nb variables",len(self.rp_model.getVars()))
        fit = self.aggreg_from_sol(sol)
        return self.create_result_storage([(sol,fit)])
    

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

  
