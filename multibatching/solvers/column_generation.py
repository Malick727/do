#!/usr/bin/env python3.11
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from discrete_optimization.generic_tools.lp_tools import GurobiMilpSolver
from multibatching.problem import MultibatchingProblem, MultibatchingSolution, PackingTransport, TransportLink
from multibatching.solvers import MultibatchingSolver

class ColumnGenerationSolver(MultibatchingSolver):

    problem : MultibatchingProblem

    def __init__(self, problem : MultibatchingProblem, params_objective_function = None, **kwargs):
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}
        self.batches = {}
        self.reduced_costs = {}
        self.dual_values = {}
        self.sizes = {}

    def init_model(self, **kwargs):
        model = gp.Model()
        max_frequency = 10 #Maximum frequency for each transport link, identical for all at the moment

        ###### Set initialization
        transport_links_in = {}
        transport_links_out = {}

        id_tl = []


        for l in self.problem.locations:
            self.dual_values[l] = {}
            for p in self.problem.products:
                self.dual_values[l][p] = 0        
            transport_links_out[l.id]=[]
            transport_links_in[l.id] = []

        for tl in self.problem.transport_links:
            l1 = tl.location_l1.id
            l2 = tl.location_l2.id
            transport_links_out[l1].append(tl)
            transport_links_in[l2].append(tl)


        # Batch initialization

        batches={}
        for transport_link in self.problem.transport_links:
            batches[transport_link]=[]

            for p1 in self.problem.products:
                batch_init = {}
                if transport_link.transport_type in p1.valid_transports:
                    batch_init[p1] = transport_link.transport_type.capacity//p1.size
                    for p2 in self.problem.products:
                        if p1 != p2:
                            batch_init[p2] = 0
                    batches[transport_link].append(batch_init)
        self.batches = batches
        
        
        
        ################################## Decision variable definition

        # Flow on links per product
        products_on_links = {}
        for tl in self.problem.transport_links:
            self.reduced_costs[tl] = 0
            products_on_links[tl]=[]
            for p in self.problem.products:
                if tl.transport_type in p.valid_transports:
                    products_on_links[tl].append(p)

        # Batch selection per arc
        x={}
        for tl in self.problem.transport_links:
            x[tl] = {}
            # type_link = tl.transport_type
            for b in range(len(self.batches[tl])):
                x[tl][b] = model.addVar(lb=0,name=f"x_{tl.id}_b{b}")

        self.variables["x"] = x

        print('Variables created')

        ##################################### Constraints
        for l in self.problem.locations:
            for p in self.problem.products:
                if p in l.net_supply.keys():
                    demand_p = -l.net_supply[p]
                    model.addConstr(gp.quicksum(gp.quicksum(x[tl][b]*self.batches[tl][b][p] for b in range(len(self.batches[tl]))) for tl in transport_links_in[l.id])-gp.quicksum(gp.quicksum(x[tl][b]*self.batches[tl][b][p] for b in range(len(self.batches[tl]))) for tl in transport_links_out[l.id]) ==  demand_p, name=f"demand_{l.id}_p{p.id}")
                else:
                    model.addConstr(gp.quicksum(gp.quicksum(x[tl][b]*self.batches[tl][b][p] for b in range(len(self.batches[tl]))) for tl in transport_links_in[l.id])-gp.quicksum(gp.quicksum(x[tl][b]*self.batches[tl][b][p] for b in range(len(self.batches[tl]))) for tl in transport_links_out[l.id]) ==  0, name=f"demand_{l.id}_p{p.id}")



        # Set objective
        obj_co2 = model.addVar(name="obj_co2")
        model.addConstr(gp.quicksum(x[tl][b]*tl.transport_type.emissions * tl.distance for tl in self.problem.transport_links for b in range(len(self.batches[tl])))==obj_co2, "co2_cost_constraint")

        obj_transport = model.addVar(name="obj_transport")
        model.addConstr(gp.quicksum(x[tl][b]*tl.transport_type.cost*tl.distance for tl in self.problem.transport_links for b in range(len(self.batches[tl])))==obj_transport, "transport_cost_constraint")

        obj_capital = model.addVar(name="obj_capital")
        capital_cost = {}


        model.setObjective(obj_co2 + obj_transport)
        model.Params.LogToConsole = 0
        self.rp_model = model
        self.products_on_links = products_on_links

    
    def solveRP(self):
        self.rp_model.optimize() 
        return self.rp_model.Status


    def solvePricingProblem(self, transport_link : TransportLink):

        #### Solve the problem
        model = gp.Model()
        model.Params.LogToConsole = 0
        y={}
        for p in self.products_on_links[transport_link]:
            y[p] = model.addVar(lb=0,ub=100, vtype=GRB.INTEGER, name=f"y_{p.id}")
        
        model.addConstr(gp.quicksum(y[p]*p.size for p in self.products_on_links[transport_link]) <= transport_link.transport_type.capacity)
        model.setObjective((transport_link.transport_type.cost+transport_link.transport_type.emissions)*transport_link.distance-gp.quicksum(y[p]*(self.dual_values[transport_link.location_l2][p]-self.dual_values[transport_link.location_l1][p]) for p in self.products_on_links[transport_link]))
        model.optimize()

        new_column = {}
        for p in self.products_on_links[transport_link]:
            new_column[p] = model.getVarByName(f"y_{p.id}").X
        reduced_cost = model.ObjVal
        return new_column, reduced_cost

    def solveSubproblems(self, list_links): 
        new_columns = {}
        nb_suproblems = len(list_links)
        i=0

        for transport_link in list_links:
            # Solve pricing problem
            new_column, reduced_cost = self.solvePricingProblem(transport_link)
            new_columns[transport_link] = new_column
            self.reduced_costs[transport_link] = reduced_cost
            i+=1
        print("------------ Nb subproblems", nb_suproblems)

        return new_columns

    def addColumns(self, new_columns):
        for transport_link in new_columns.keys():
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

            cost = (transport_link.transport_type.emissions + transport_link.transport_type.cost) * transport_link.distance 
            self.rp_model.addVar(obj=cost, lb=0, name=f"x_{transport_link.id}_b{nb_batches}", column=c)

    
    def solve(self, **kwargs):

        optimal = False
        list_links = self.problem.transport_links
        
        while not(optimal):

            # Solve RP
            status = self.solveRP()

            if status == GRB.OPTIMAL:
                print("RP solved with objective ---- >", self.rp_model.ObjVal)

            # Compute reduced costs and store dual variables
            for l in self.problem.locations:
                for p in self.problem.products:
                    c = self.rp_model.getConstrByName(f"demand_{l.id}_p{p.id}")
                    self.dual_values[self.problem.location_id_to_location[l.id]][self.problem.product_dict[p.id]] = c.Pi

            new_columns = self.solveSubproblems(list_links)
            
            optimal = True
            min_reduced_cost = 0
            list_links = []
            for tl in self.problem.transport_links:
                if self.reduced_costs[tl]<10^-6:
                    if self.reduced_costs[tl] < min_reduced_cost:
                        min_reduced_cost = self.reduced_costs[tl]
                    optimal = False
                    list_links.append(tl)
            print("----------- Min reduced cost :", min_reduced_cost)
            
            if not(optimal):
                self.addColumns(new_columns)
        
        sol = self.get_final_solution()
        print("get final ----------------", sol)
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

  
