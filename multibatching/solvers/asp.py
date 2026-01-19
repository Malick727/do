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
logger = logging.getLogger(__name__)


cap_size_divide = 1
# Logic Program Content
ASP_MODEL_CONTENT = """
root(eARFullFuselage). 

totalSupply(P, S) :- S = #sum {V : demandOffer(P,_,V), V > 0;-1* V : demandOffer(P,_,V), V < 0 }, product(P).
#show totalSupply/2.

hasDemand(P, L) :- demandOffer(P, L, V), V < 0.
hasOffer(P, L)  :- demandOffer(P, L, V), V > 0.

demand(P, L, Val * -1) :- demandOffer(P, L, Val), Val < 0.
demand(P, L, 0)        :- product(P), location(L), not hasDemand(P, L).

offer(P, L, Val) :- demandOffer(P, L, Val), Val > 0.
offer(P, L, 0)   :- product(P), location(L), not hasOffer(P, L).

requiredNet(P, L, Net) :- demand(P,L,D), offer(P,L,O), Net = D - O.
requiredNet(P, L, 0) :- not demand(P,L,_), not offer(P,L,_), location(L), product(P).
&dom { N..N } = req(P,L) :- requiredNet(P, L, N).

%%%
possibleFlow(From, To, TR, P) :- route(_, From, To, TR, _,Cost),  productTR(P,TR), not root(P), transportCapacity(TR, Cap), Cap>1, Cost>0.

&dom { 0..Max } = load(From, To, TR, P) :- possibleFlow(From, To, TR,P), totalSupply(P, Max).
&sum { load(From, L, TR, P): possibleFlow(From, L, TR, P)} = flow_in(P, L) :-  location(L), product(P).
&sum { load(L, To, TR, P): possibleFlow(L, To, TR, P)} = flow_out(P, L) :-  location(L), product(P).

:- &sum{flow_out(P,From)}!=V, offer(P,From,V), V!=0.
:- &sum{flow_in(P,To)}!= V, demand(P,To,V), V!=0.
:- &sum{flow_in(P,From)} !=0, offer(P,From,V), V!=0.
:- &sum{flow_out(P,From)} !=0, demand(P,From,V), V!=0.
:- &sum{flow_in(P,L); -1*flow_out(P,L)} !=0, not demandOffer(P,L,_),  location(L), product(P).

:- &sum{flow_out(P,From)} > Total_flow_out, Total_flow_out=#sum{V: offer(P,_,V)}, possibleFlow(From, _, _,P).
:- &sum{flow_in(P,To)} > Total_flow_in, Total_flow_in=#sum{V: demand(P,_,V)}, possibleFlow(_,To, _,P).

% Balance Constraint
% when demandOffer is 0, then flow_in and flow_out need to be the same
&sum { flow_in(P,L); -1 * flow_out(P,L)} = 0 :- demandOffer(P,L,0).
% when demandOffer is <0 (i.e. there is demand), then in-out needs to be -1*V  
&sum { flow_in(P,L); -1 * flow_out(P,L)} = -1*V :- demandOffer(P,L,V), V<0.
% when demandOffer is > 0 (i.e. there is supply), then the out-in needs to be V
&sum { flow_out(P,L); -1 * flow_in(P,L)} = V :- demandOffer(P,L,V), V>0.

#const maxFreq = 30.
&dom {0..maxFreq} = estimated_freq(From,To,TR) :- possibleFlow(From, To, TR, _).

&sum {C * estimated_freq(From, To, TR);
   -1 * S * load(From, To, TR, P) : productSize(P, S), possibleFlow(From, To, TR, P)
} >= 0 :- possibleFlow(From, To, TR, _), transportCapacity(TR, C).

% it cannot be the case that there is a load > 0 and no estimated freq for this trajectory
:- &sum {load(From, To, TR, P) : possibleFlow(From, To, TR, P)}>0, estimated_freq(From, To, TR)=0, possibleFlow(From, To, TR, P).

&sum {load(From, To, TR, P) : possibleFlow(From, To, TR, P); -1* estimated_freq(From, To, TR)} >= 0 :- possibleFlow(From, To, TR, _).

#const limit = 1000000000.
&dom { 0..limit } = route_cost_total(From, To, TR) :- possibleFlow(From, To, TR,_).
&sum {  C * estimated_freq(From,To,TR)} = route_cost_total(From, To, TR) :- possibleFlow(From, To, TR,_),route(_, From, To, TR, _, C).

&minimize{ route_cost_total(From, To, TR):route(_, From, To, TR, _, C) }.
&minimize{ load(From, To, TR, P):possibleFlow(From, To, TR, P) }.


"""


class ClingconMultibatchingSolver(SolverDO):
    problem: MultibatchingProblem

    all_dataframes = {}

    modelAggregateTemplates = [ {'name':'flow_out', 'filter':lambda s: s.name=='flow_out', 'columns':['product' ,'location', 'amount']},
                            {'name':'flow_in', 'filter':lambda s: s.name=='flow_in', 'columns':['product' ,'location', 'amount']},
                            {'name':'req', 'filter':lambda s: s.name=='req', 'columns':['product' ,'location', 'amount']},
                            {'name':'route_cost_total', 'filter':lambda s: s.name=='route_cost_total', 'columns':['from' ,'to', 'transportresource', 'cost']},
                            {'name':'load', 'filter':lambda s: s.name=='load', 'columns':['from' ,'to', 'transportresource', 'product', 'amount']}
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
        logger.info("Grounding logic program...")
        self.ctl.ground([("base", [])])

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
            sol = self.retrieve_solution(model)
            fit = self.aggreg_from_sol(sol)
            res.append((sol, fit))
            stopping = callbacks_list.on_step_end(step=len(res), res=res, solver=self)
            print("stopping", stopping)
            return not stopping

        logger.info("Starting Clingcon solve...")

        # Timeout handling via async handle
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
        #print([f'{key}={val}' for key, val in assignment if val!=0])
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
                        df.loc[len(df)]  = {"from": from_name, "to":to_name , "transportresource": tr_name, "product":product, "amount":value}
                        
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

        route_cost_total_df = self.all_dataframes["route_cost_total"]
        logger.info(f"Total costs for model: {route_cost_total_df['cost'].sum()}")
>>>>>>> f66ca6e4842c6ec462a791ba3f82edb5348855fd

        list_flows = []
        for link, product_quantities in flows_aggregation.items():
            packing = {p: q for p, q in product_quantities.items() if q > 0}
            if packing:
                list_flows.append(PackingTransport(link, packing, 1))
        #sol = MultibatchingSolution(problem=self.problem, list_flows=list_flows)
        #print("aggreg", self.aggreg_from_sol(sol))
        return MultibatchingSolution(problem=self.problem, list_flows=list_flows)


<<<<<<< HEAD
=======
    def compute_costs(self):
        route_cost_total_df = self.all_dataframes["route_cost_total"]
        logger.info(f"Total costs for model: {route_cost_total_df['cost'].sum()}")
>>>>>>> f66ca6e4842c6ec462a791ba3f82edb5348855fd
    
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