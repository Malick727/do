#!/usr/bin/env python3.11
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from discrete_optimization.generic_tools.lp_tools import GurobiMilpSolver
from multibatching.problem import MultibatchingProblem, MultibatchingSolution, PackingTransport

class MilpSolver(GurobiMilpSolver):

    problem : MultibatchingProblem

    def __init__(self, problem : MultibatchingProblem, params_objective_function = None, **kwargs):
        super().__init__(problem, params_objective_function, **kwargs)
        self.variables = {}
        self.batches = {}

    def init_model(self, **kwargs):
        model = gp.Model()
        max_frequency = 10 #Maximum frequency for each transport link, identical for all at the moment
        

        ###### Batch initialization
        batches={}
        for transport_type in self.problem.transport_types:
            batches[transport_type]=[]
            for p1 in self.problem.products:
                batch_init = {}
                if transport_type in p1.valid_transports:
                    batch_init[p1] = transport_type.capacity//p1.size
                    for p2 in self.problem.products:
                        if p1 != p2:
                            batch_init[p2] = 0
                    batches[transport_type].append(batch_init)
        self.batches = batches

        ###### Set initialization
        transport_links_in = {}
        transport_links_out = {}


        for l in self.problem.locations:
                transport_links_out[l.id]=[]
                transport_links_in[l.id] = []

        for tl in self.problem.transport_links:
            l1 = tl.location_l1.id
            l2 = tl.location_l2.id

            transport_links_out[l1].append(tl)
            transport_links_in[l2].append(tl)
            
        
        
        ################################## Decision variable definition

        # Flow on links per product
        y={}
        products_on_links = {}
        for tl in self.problem.transport_links:
            y[tl]={}
            products_on_links[tl]=[]
            for p in self.problem.products:
                y[tl][p] = model.addVar(lb=0,name=f"y_{tl.id}_{p.name}")
                if tl.transport_type in p.valid_transports:
                    products_on_links[tl].append(p)
                else:
                    model.addConstr(y[tl][p]==0)
                    
        self.variables["y"] = y

        # Batch selection per arc
        x={}
        for tl in self.problem.transport_links:
            x[tl] = {}
            type_link = tl.transport_type
            for b in range(len(batches[type_link])):
                x[tl][b] = model.addVar(vtype=GRB.INTEGER, name=f"x_{tl.id}_b{b}")

        self.variables["x"] = x

        print('Variables created')

        ##################################### Constraints

        # Meet demand
        for l in self.problem.locations:
            for p, offer_p in l.net_supply.items():
                model.addConstr(gp.quicksum(y[tl][p] for tl in transport_links_out[l.id]) - gp.quicksum(y[tl][p] for tl in transport_links_in[l.id]) == offer_p, name=f"demand_{l.id}_p{p.id}")


        # Max frequency


        # Batch selection
        for tl in self.problem.transport_links:
            for p in products_on_links[tl]:
                model.addConstr(gp.quicksum(x[tl][b]*batches[tl.transport_type][b][p]
                                            for b in range(len(batches[tl.transport_type]))) == y[tl][p])


        # Set objective
        obj_co2 = model.addVar(name="obj_co2")
        model.addConstr(gp.quicksum(x[tl][b]*tl.transport_type.emissions * tl.distance for b in range(len(self.batches[type_link])) for tl in self.problem.transport_links)==obj_co2, "co2_cost_constraint")

        obj_transport = model.addVar(name="obj_transport")
        model.addConstr(gp.quicksum(x[tl][b]*tl.transport_type.cost*tl.distance for b in range(len(self.batches[type_link])) for tl in self.problem.transport_links)==obj_transport, "transport_cost_constraint")

        obj_capital = model.addVar(name="obj_capital")
        capital_cost = {}


        model.setObjective(obj_co2 + obj_transport)
        self.model = model
        self.products_on_links = products_on_links


    def get_final_solution(self):
        packing_solution = []
        for tl in self.problem.transport_links:
            type_link = tl.transport_type
            for b in range(len(self.batches[type_link])):
                nb_packing = self.variables["x"][tl][b].X
                if nb_packing>0:
                    packing_solution.append(PackingTransport(transport_link=tl, product_packing=self.batches[tl.transport_type][b], nb_packing=nb_packing))
        return MultibatchingSolution(problem=self.problem, list_flows=packing_solution)




    def retrieve_current_solution(self, get_var_value_for_current_solution, get_obj_value_for_current_solution):
        packing_solution = []
        for tl in self.problem.transport_links:
            packing_link = []
            type_link = tl.transport_type
            for b in range(len(self.batches[type_link])):
                nb_packing = get_var_value_for_current_solution(self.variables["x"][tl][b])
                if nb_packing>0:
                    packing_solution.append(PackingTransport(transport_link=tl, product_packing=self.batches[tl.transport_type][b], nb_packing=nb_packing))
        return MultibatchingSolution(problem=self.problem, list_flows=packing_solution)
        # raise NotImplementedError

    def convert_to_variable_values(self, solution):
        raise NotImplementedError

