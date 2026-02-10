import logging
from collections import defaultdict
from typing import Any, Optional, List, Dict

import clingo
from clingo.ast import parse_string, ProgramBuilder
from clingcon import ClingconTheory
from discrete_optimization.generic_tools.callbacks.callback import Callback, CallbackList
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction
from discrete_optimization.generic_tools.do_solver import SolverDO, StatusSolver
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from multibatching.problem import (
    MultibatchingProblem,
    MultibatchingSolution,
    PackingTransport,
    Product,
    TransportLink
)

import pandas as pd
import os
import csv
import time 
logger = logging.getLogger(__name__)


cap_size_divide = 100000
# Logic Program Content
ASP_MODEL_CONTENT = """
harbor(naplesHarbor;marseilleHarbor;montoirHarbor;tunisHarbor;mobileHarbor;dunkerqueHarbor;gruenendeichHarbor;taichungHarbor;shanghaiHarbor;hamburgHarbor;gibraltarHarbor;portoHarbor;tianjinHarbor;casablancaHarbor;sacheonHarbor;stNazaire;houstonHarbor;savannaHarbor).
ship(kugelbake;feederContainerShip;truckGeneralCargo;oversizedTianjinRoRoShip;truckonRoRoShip;smallContainerShip;airbusMobileExpressRoRoShip;oversizedRiverTransport;seaLargeContainerShip;oversizedUnderDeckRoRoShip).

totalSupply(P, Max) :- Max = #sum {V : demandOffer(P,_,V), V > 0; -1 * V : demandOffer(P,_,V), V < 0}, product(P).

validNode(L) :- demandOffer(_, L, _).
validNode(L) :- harbor(L).
noShipBetweenHarbors(From, To, TR) :- harbor(From), harbor(To), not ship(TR), transportResource(TR).

possibleFlow(From, To, TR, P) :- route(_, From, To, TR, _,Cost), 
%validNode(From), validNode(To),  
not noShipBetweenHarbors(From, To, TR), 
productTR(P,TR), tRCapSmall(TR, Cap), productSizeSmall(P,S), S<Cap. %, not invalidProductTR(P,TR).

tRCapSmall(TR,Cap) :- transportCapacity(TR,Cap).
productSizeSmall(P,S) :- productSize(P,S).

%invalidProductTR(P, TR) :- tRCapSmall(TR, Cap), productSizeSmall(P,S), S>Cap, productTR(P,TR).
%#show invalidProductTR/2.   

&dom { 0..Max } = load(From, To, TR, P) :- possibleFlow(From, To, TR,P), totalSupply(P, Max).
&sum { load(From, L, TR, P): possibleFlow(From, L, TR, P)} = flow_in(P, L) :-  location(L), product(P).
&sum { load(L, To, TR, P): possibleFlow(L, To, TR, P)} = flow_out(P, L) :-  location(L), product(P).

&sum{flow_in(P,L); -1*flow_out(P,L)} = 0 :- not demandOffer(P,L,_),  location(L), product(P).
&sum { flow_in(P,L); -1 * flow_out(P,L)} = -Net  :- demandOffer(P, L, Net).

%:- &sum{flow_in(P,L)} > M, demandOffer(P,L,N), N<0, M=N * -1.
%:- &sum{flow_out(P,L)} > N, demandOffer(P,L,N), N>0.
%:- &sum{flow_in(P,L)} !=0, demandOffer(P,L,N), N>0.
%:- &sum{flow_out(P,L)} !=0, demandOffer(P,L,N), N<0.

:- &sum{flow_out(P,From)} > Max, totalSupply(P, Max), possibleFlow(From, _, _,P).
:- &sum{flow_in(P,To)} > Max, totalSupply(P, Max), possibleFlow(_, To, _,P).

activeFlow(From,To,TR) :-  &sum { load(From, To, TR, P):possibleFlow(From, To, TR, P)} > 0,possibleFlow(From, To, TR, _).

freq(1..20).
1{routeFreq(From, To, TR, Freq, C*Freq) : freq(Freq), tRCapSmall(TR,C) }1 :- activeFlow(From,To,TR).

% upper bound
&sum{S*load(F, T, TR, P) : possibleFlow(F, T, TR, P),productSizeSmall(P,S)} <= TotalCap :-
routeFreq(F, T, TR, Freq, TotalCap).
% lower bound
&sum{S*load(F, T, TR, P) :possibleFlow(F, T, TR, P),productSizeSmall(P,S)} >= Minus1 :-
routeFreq(F, T, TR, Freq, TotalCap), tRCapSmall(TR,C), Minus1=TotalCap-C, Freq>1.

#minimize{C*Freq: routeFreq(F, T, TR, Freq, TotalCap),route(_, From, To, TR, _, C)}.
"""

class ClingconMultibatchingSolver(SolverDO):
    problem: MultibatchingProblem

    all_dataframes = {}
    modelIndex = 0
    start_solving = 0

    modelAtomTemplates = [
        {'name':'totalSupply', 'filter':lambda s: s.name=='totalSupply', 'columns':['product' ,'amount']},
        {'name':'route', 'filter':lambda s: s.name=='route', 'columns':['idx','from' ,'to', 'transportresource', 'distance','cost']},
        {'name':'routeCostTotal', 'filter':lambda s: s.name=='routeCostTotal', 'columns':['from' ,'to', 'transportresource', 'cost']},
        {'name':'routeFreq','filter':lambda s: s.name=='routeFreq', 'columns':['from', 'to', 'transportresource','freq', 'totalcapacity']},
        {'name':'activeFlow','filter':lambda s: s.name=='activeFlow', 'columns':['from', 'to', 'transportresource']},
        {'name':'demandOffer','filter':lambda s: s.name=='demandOffer', 'columns':['product', 'location', 'amount']},
        {'name':'requiredNet','filter':lambda s: s.name=='requiredNet', 'columns':['product', 'location', 'amount']},
        {'name':'productTR','filter':lambda s: s.name=='productTR', 'columns':['product', 'transportresource']},
        {'name':'productSize','filter':lambda s: s.name=='productSizeSmall', 'columns':['product', 'size']},
        {'name':'possibleFlow','filter':lambda s: s.name=='possibleFlow', 'columns':['from', 'to', 'transportresource','product']},
        {'name':'transportCapacity','filter':lambda s: s.name=='tRCapSmall', 'columns':['transportresource', 'capacity']},
        {'name':'productOnFlow','filter':lambda s: s.name=='productOnFlow', 'columns':['from', 'to', 'transportresource','product','amount']}
    ]

    modelAggregateTemplates = [ 
                            {'name':'route_freq_cap','filter':lambda s: s.name=='route_freq_cap', 'columns':['from', 'to', 'transportresource','freq', 'total']},
                             {'name':'flow_out', 'filter':lambda s: s.name=='flow_out', 'columns':['product' ,'location', 'amount']},
                            {'name':'flow_in', 'filter':lambda s: s.name=='flow_in', 'columns':['product' ,'location', 'amount']},
                            {'name':'req', 'filter':lambda s: s.name=='req', 'columns':['product' ,'location', 'amount']},
                            {'name':'route_cost_total', 'filter':lambda s: s.name=='route_cost_total', 'columns':['from' ,'to', 'transportresource', 'cost']},
                            {'name':'load', 'filter':lambda s: s.name=='load', 'columns':['from' ,'to', 'transportresource', 'product', 'amount']},
                            {'name':'active_flow', 'filter':lambda s: s.name=='active_flow', 'columns':['from' ,'to', 'transportresource','val']}
                            ]

    def __init__(
            self,
            problem: MultibatchingProblem,
            params_objective_function: Optional[ParamsObjectiveFunction] = None,
            **kwargs: Any,
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.ctl: Optional[clingo.Control] = None
        self.theory: Optional[ClingconTheory] = None
        self.name_to_location: Dict[str, Any] = {}
        self.name_to_transport_type: Dict[str, Any] = {}
        self.name_to_product: Dict[str, Any] = {}
        self.route_key_to_link: Dict[tuple, TransportLink] = {}


    def processModelAtoms(self, model: clingo.Model):
        path_symbols = []
        for name in [next(iter(d.values())) for d in self.modelAtomTemplates]:
            path_symbols += list(filter(lambda s: s.name == name, model.symbols(atoms=True)))
        
        for modelAtomTemplate in self.modelAtomTemplates:
            dataName = modelAtomTemplate['name']
            filteredSymbols = list(filter(modelAtomTemplate['filter'], path_symbols))
            arguments = list(map(lambda s: s.arguments, filteredSymbols))
            df = pd.DataFrame(data=arguments, columns= modelAtomTemplate['columns'])
            self.all_dataframes[dataName] = df

            output_dir = f"output/model_{self.modelIndex}"
            os.makedirs(output_dir, exist_ok=True)
            df.to_csv(f"{output_dir}/{dataName}.csv", index=False)


    def init_model(self, **kwargs: Any) -> None:
        # Default options
        clingo_args = kwargs.get("clingo_args", [#"--models=0",
                                                 "--opt-mode=opt"])
        self.ctl = clingo.Control(clingo_args)
        self.theory = ClingconTheory()

        # 1. Register Clingcon theory
        self.theory.register(self.ctl)

        # 2. Add Logic Program using the rewriter
        # This fixes the "no definition found for theory atom" error
        with ProgramBuilder(self.ctl) as bld:
            # Parse main model
            parse_string(ASP_MODEL_CONTENT, lambda ast: self.theory.rewrite_ast(ast, bld.add))

            # Generate and parse facts
            facts_str = self._generate_facts()
            logger.debug("Generated ASP Facts:\n" + facts_str)
            parse_string(facts_str, lambda ast: self.theory.rewrite_ast(ast, bld.add))

        # 3. Ground

        start_time_grounding = time.process_time()
        logger.info("Grounding logic program...")
        self.ctl.ground([("base", [])])
        end_time_grounding = time.process_time()
        logger.info(f"Grounding done ({round((end_time_grounding - start_time_grounding) / 60, 2)} minutes).")
    
        # 4. Prepare theory (must happen after grounding)
        self.theory.prepare(self.ctl)

    def solve(
            self,
            callbacks: Optional[List[Callback]] = None,
            time_limit: Optional[float] = None,
            **kwargs: Any
    ) -> ResultStorage:

        if self.ctl is None:
            self.init_model(**kwargs)

        callbacks_list = CallbackList(callbacks=callbacks)
        callbacks_list.on_solve_start(solver=self)

        res = self.create_result_storage()

        def on_model_wrapper(model: clingo.Model):
            self.modelIndex += 1
            logger.info(f"Model [{self.modelIndex}] found after {round((time.process_time() - self.start_solving) / 60, 2)} minutes of solving.")

            self.processModelAtoms(model)
            sol = self.retrieve_solution(model)
            fit = self.aggreg_from_sol(sol)

            res.append((sol, fit))
            self.compute_costs()
            stopping = callbacks_list.on_step_end(step=len(res), res=res, solver=self)
            print('fit', fit)
            print(self.problem.satisfy(sol))
            print('satisfy', self.problem.satisfy(sol))
            print("Check flows ", self.problem.check_flows(sol))
            print("stopping", stopping)
            
            return not stopping
        

        logger.info("Starting Clingcon solve...")

        # Timeout handling via async handle
        self.start_solving = time.process_time()
        if time_limit is None:
            self.ctl.solve(on_model=on_model_wrapper)
        else:
            with self.ctl.solve(on_model=on_model_wrapper, async_=True) as handle:
                finished = handle.wait(time_limit)
                if not finished:
                    logger.info(f"Time limit of {time_limit}s reached.")
                    handle.cancel()
        if len(res) > 0:
            self.status_solver = StatusSolver.SATISFIED
        else:
            self.status_solver = StatusSolver.UNSATISFIABLE

        callbacks_list.on_solve_end(res=res, solver=self)

        return res

    def retrieve_solution(self, model: clingo.Model) -> MultibatchingSolution:
        logger.info(f"Optimality proven : {model.optimality_proven}")
        assignment = list(self.theory.assignment(model.thread_id))
        flows_aggregation = defaultdict(lambda: defaultdict(int))
        #print([f'{key}={val}' for key, val in assignment if key.name == "active_flow" and val != 0])
        for template in self.modelAggregateTemplates:
            df = pd.DataFrame(data={}, columns=template["columns"])
            self.all_dataframes[template["name"]] = df

        for name, value in assignment:
            if name.name == "load" and value > 0:
                try:
                    args = name.arguments
                    from_name, to_name, tr_name, p_name = map(str, args)
                    route_key = (from_name, to_name, tr_name)
                    link = self.route_key_to_link.get(route_key)
                    product = self.name_to_product.get(p_name)
                    if link and product:
                        flows_aggregation[link][product] += value
                        df = self.all_dataframes["load"]
                        df.loc[len(df)]  = {"from": from_name, "to":to_name , "transportresource": tr_name, "product":p_name, "amount":value}
                        
                    output_dir = f"output/model_{self.modelIndex}"
                    os.makedirs(output_dir, exist_ok=True)
                    df.to_csv(f"{output_dir}/load.csv", index=False)

                except Exception as e:
                    logger.error(f"Error parsing variable {name}: {e}")
            
            if name.name == "flow_in" and value > 0:
                df = self.all_dataframes["flow_in"]
                args = name.arguments
                p_name, to_name = map(str, args)
                product = self.name_to_product.get(p_name)
                df.loc[len(df)] = {"product": p_name, "location":to_name, "amount":value}
            
            if name.name == "flow_out" and value > 0:
                df = self.all_dataframes["flow_out"]
                args = name.arguments
                p_name, from_name = map(str, args)
                product = self.name_to_product.get(p_name)
                df.loc[len(df)] = {"product": p_name, "location":from_name, "amount":value}

            if name.name == "route_cost_total" and value > 0:
                df = self.all_dataframes["route_cost_total"]
                args = name.arguments 
                from_name, to_name, tr_name = map(str, args)
               
                df.loc[len(df)]= {"from": from_name, "to":to_name, "transportresource": tr_name, "cost":value}
                
            if name.name == "req" and value > 0:
                df = self.all_dataframes["req"]
                args = name.arguments
                p_name, loc_name  = map(str, args)
                df.loc[len(df)] = {"product": p_name, "location":loc_name, "amount":value}

        list_flows = []

        for link, product_quantities in flows_aggregation.items():
            #print("link", link)
            packing = {p: q for p, q in product_quantities.items() if q > 0}
            if packing:
                from_loc = str(link.location_l1.name).strip()
                to_loc = str(link.location_l2.name).strip()
                mode = str(link.transport_type.name).strip()
                routeFreq_df = self.all_dataframes["routeFreq"]
                mask = (routeFreq_df['from'].astype(str) == from_loc) & \
                    (routeFreq_df['to'].astype(str) == to_loc) & \
                    (routeFreq_df['transportresource'].astype(str) == mode)
                freq = routeFreq_df.loc[mask, 'freq'].values[0]
                list_flows.append(PackingTransport(link, packing, freq))

        sol = MultibatchingSolution(problem=self.problem, list_flows=list_flows)
        print("Sol", sol)

        #kpis = self.problem.evaluate(sol)
        #print("kpis", kpis)
        #logger.info(f"cost of current sol : kpis={kpis}")
        return sol #MultibatchingSolution(problem=self.problem, list_flows=list_flows)

    def compute_costs(self):
        
        active_flows_df = self.all_dataframes["activeFlow"]
        logger.info(f"Total active flows for model: {active_flows_df.shape[0]}")
        
        route_df = self.all_dataframes["route"]
        route_df['cost'] = route_df['cost'].values
        route_freq_df = self.all_dataframes["routeFreq"]
        route_freq_df['freq'] = route_freq_df['freq'].values
        
        merge_cols = ['from', 'to', 'transportresource']
    
        for df in [route_df, route_freq_df]:
            for col in merge_cols:
                # Convert to string and strip whitespace to ensure matching
                df[col] = df[col].astype(str).str.strip()

        merged_df = pd.merge(
        route_freq_df, 
        route_df, 
        on=['from', 'to', 'transportresource'], 
        how='inner'
        )
        def extract_clingo_number(val):
            # Case A: It's a Clingo Symbol (has .number attribute)
            if hasattr(val, 'number'): 
                return val.number
            # Case B: It's already a Python int or float
            if isinstance(val, (int, float)):
                return val
            # Case C: It's a string that looks like a number ("6")
            try:
                return float(val)
            except (ValueError, TypeError):
                return 0  # Default fallback if data is bad

        merged_df['freq'] = merged_df['freq'].apply(extract_clingo_number)
        merged_df['cost'] = merged_df['cost'].apply(extract_clingo_number)

        merged_df['total_cost'] = merged_df['freq'] * merged_df['cost']
        
        logger.info(f"Total costs for model: {merged_df['total_cost'].sum()}")
        
    def _generate_facts(self) -> str:
        # Same generation logic as before, ensure it returns a valid ASP string
        facts = []

        def sanitize(s: str) -> str:
            s = str(s).replace(".", "").replace("-", "").replace(" ", "")
            return s[0].lower() + s[1:]

        def sanitize_num(val) -> int:
            return max(1, int(val))
        


        # Locations
        self.name_to_location = {}
        for loc in self.problem.locations:
            s_name = sanitize(loc.name)
            self.name_to_location[s_name] = loc
            facts.append(f"location({s_name}).")

        # Transport Resources
        self.name_to_transport_type = {}
        for tt in self.problem.transport_types:
            s_name = sanitize(tt.name if tt.name else f"tr{tt.id}")
            self.name_to_transport_type[s_name] = tt
            facts.append(f"transportResource({s_name}).")
            facts.append(f"transportCO2({s_name},{sanitize_num(tt.emissions)}).")
            facts.append(f"transportCost({s_name},{sanitize_num(tt.cost)}).")
            scaled_cap = max(1, int(tt.capacity / cap_size_divide))
            facts.append(f"transportCapacity({s_name},{scaled_cap}).")
            facts.append(f"transportSpeed({s_name},{sanitize_num(tt.speed)}).")

        # Routes
        self.route_key_to_link = {}
        for idx, tl in enumerate(self.problem.transport_links):
            source = sanitize(tl.location_l1.name)
            dest = sanitize(tl.location_l2.name)
            tr_name = sanitize(tl.transport_type.name if tl.transport_type.name else f"tr{tl.transport_type.id}")
            self.route_key_to_link[(source, dest, tr_name)] = tl
            dist = int(tl.distance)
            cost = int(sanitize_num(tl.transport_type.cost) * dist)
            facts.append(f"route({idx},{source},{dest},{tr_name},{dist},{cost}).")

        # Products
        self.name_to_product = {}
        for p in self.problem.products:
            s_name = sanitize(p.name if p.name else f"p{p.id}")
            self.name_to_product[s_name] = p
            facts.append(f"product({s_name}).")
            scaled_size = int(p.size / cap_size_divide)
            facts.append(f"productSize({s_name},{scaled_size}).")
            for valid_tr in p.valid_transports:
                tr_s_name = sanitize(valid_tr.name if valid_tr.name else f"tr{valid_tr.id}")
                facts.append(f"productTR({s_name},{tr_s_name}).")

        # Demand/Offers
        for loc in self.problem.locations:
            loc_s_name = sanitize(loc.name)
            for p, amount in loc.net_supply.items():
                if amount != 0:
                    p_s_name = sanitize(p.name if p.name else f"p{p.id}")
                    facts.append(f"demandOffer({p_s_name},{loc_s_name},{int(amount)}).")
            
        return "\n".join(facts)