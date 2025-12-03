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

logger = logging.getLogger(__name__)

# Logic Program Content
ASP_MODEL_CONTENT = """
transportResource(intrasite). 
transportCO2(intrasite,0). 
transportCost(intrasite,0).
transportCapacity(intrasite,2000000).
partTR(Part,intrasite) :- part(Part).

root(eARFullFuselage). 

% 2. ROBUST DEMAND/OFFER
has_demand(P, L) :- demandOffer(P, L, V), V < 0.
has_offer(P, L)  :- demandOffer(P, L, V), V > 0.

demand(P, L, Val * -1) :- demandOffer(P, L, Val), Val < 0.
demand(P, L, 0)        :- part(P), location(L), not has_demand(P, L).
offer(P, L, Val) :- demandOffer(P, L, Val), Val > 0.
offer(P, L, 0)   :- part(P), location(L), not has_offer(P, L).

% 3. CALCULATE NET REQUIREMENT
required_net(P, L, Net) :- demand(P,L,D), offer(P,L,O), Net = D - O.
&dom { N..N } = req(P,L) :- required_net(P, L, N).

%%%
possible_flow(From, To, TR, P) :- route(_, From, To, TR, _,_), part(P), not root(P), transportCapacity(TR, Cap), Cap>0.
{partOnFlow(From, To, TR, P)} :- possible_flow(From, To, TR, P).

active_flow(From, To, TR) :- partOnFlow(From, To, TR, _).

total_supply(P, S) :- S = #sum { V*2 : demandOffer(P,_,V), V > 0 }, part(P).

&dom { 0..Max } = load(From, To, TR, P) :- 
    possible_flow(From, To, TR, P), total_supply(P, Max).

% CONSTRAINT 1 (The Off Switch):
&sum { load(From, To, TR, P) } = 0 :- possible_flow(From, To, TR, P), 
                                      not partOnFlow(From, To, TR, P).

% CONSTRAINT 2 (The On Switch - Optional but helpful):
&sum { load(From, To, TR, P) } >= 1 :- partOnFlow(From, To, TR, P).

% =========================================================
% 5. CALCULATIONS & BALANCE
% =========================================================

&sum { load(From, L, TR, P): possible_flow(From, L, TR, P)} = flowIn(P, L)
    :- location(L), part(P).

&sum { load(L, To, TR, P): possible_flow(L, To, TR, P)} = flowOut(P, L)
    :-  location(L), part(P).

% Balance Constraint
&sum { flowIn(P,L); -1 * flowOut(P,L) } = req(P,L) :- part(P), location(L).

&dom { 0..100000000 } = total_weight(From, To, TR) :- active_flow(From, To, TR).

&sum {  S*load(From, To, TR, P) : partSize(P, S) } = total_weight(From, To, TR)
    :- active_flow(From, To, TR).

&dom { 0..10 } = estimatedFreq(From, To, TR) :- active_flow(From, To, TR).

&sum { Cap * estimatedFreq(From, To, TR) } >= total_weight(From, To, TR)
    :- active_flow(From, To, TR), transportCapacity(TR, Cap).

% 2. Tightly constrain the estimatedFreq domain
&dom { 0..10 } = estimatedFreq(From, To, TR) :- 
    active_flow(From, To, TR).

&dom { 0..4654102 } = routeCostTotal(From, To, TR) :- active_flow(From, To, TR).

&sum {  C * estimatedFreq(From, To, TR) :route(_, From, To, TR, _, C)} = routeCostTotal(From, To, TR) :- active_flow(From, To, TR).

&minimize{ routeCostTotal(From, To, TR) :possible_flow(From,To,TR,_) }.
"""


class ClingconMultibatchingSolver(SolverDO):
    problem: MultibatchingProblem
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
        clingo_args = kwargs.get("clingo_args", [# "--models=1",
                                                 "--opt-mode=optN"])

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
            return stopping

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
        logger.info(f"Optimality proven : {model.optimality_proven}, cost = {model.cost}")
        assignment = list(self.theory.assignment(model.thread_id))
        flows_aggregation = defaultdict(lambda: defaultdict(int))

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
                except Exception as e:
                    logger.error(f"Error parsing variable {name}: {e}")

        list_flows = []
        for link, product_quantities in flows_aggregation.items():
            packing = {p: q for p, q in product_quantities.items() if q > 0}
            if packing:
                list_flows.append(PackingTransport(link, packing, 1))

        return MultibatchingSolution(problem=self.problem, list_flows=list_flows)

    def _generate_facts(self) -> str:
        # Same generation logic as before, ensure it returns a valid ASP string
        facts = []

        def sanitize(s: str) -> str:
            s = str(s).replace(".", "").replace("-", "").replace(" ", "")
            return s[0].lower() + s[1:]

        def sanitize_num(val) -> int:
            return max(1, int(val))

        cap_size_divide = 10000

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
            facts.append(f"part({s_name}).")
            scaled_size = int(p.size / cap_size_divide)
            facts.append(f"partSize({s_name},{scaled_size}).")
            for valid_tr in p.valid_transports:
                tr_s_name = sanitize(valid_tr.name if valid_tr.name else f"tr{valid_tr.id}")
                facts.append(f"partTR({s_name},{tr_s_name}).")

        # Demand/Offers
        for loc in self.problem.locations:
            loc_s_name = sanitize(loc.name)
            for p, amount in loc.net_supply.items():
                if amount != 0:
                    p_s_name = sanitize(p.name if p.name else f"p{p.id}")
                    facts.append(f"demandOffer({p_s_name},{loc_s_name},{int(amount)}).")

        return "\n".join(facts)