transportResource(intrasite). 
transportCO2(intrasite,0). 
transportCost(intrasite,0).
transportCapacity(intrasite,2000000).
partTR(Part,intrasite) :- part(Part).


total_supply(P, S) :- S = #sum { V*2 : demandOffer(P,_,V), V > 0 }, part(P).

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
possible_flow(From, To, TR, P) :- route(From, To, TR, _, _), part(P), not root(P), transportCapacity(TR, Cap), Cap>0.

{partOnFlow(From, To, TR, P)} :- possible_flow(From, To, TR, P).


active_flow(From, To, TR) :- partOnFlow(From, To, TR, _).

%:- partOnFlow(From, To, TR, P), partSize(P,S), transportCapacity(TR, Cap), S > Cap.

&dom { 0..Max } = load(From, To, TR, P) :- 
    possible_flow(From, To, TR, P),total_supply(P, Max) .

% CONSTRAINT 1 (The Off Switch):
% If ASP says "No Flow", Load MUST be 0.
&sum { load(From, To, TR, P) } = 0 :- possible_flow(From, To, TR, P), 
                                      not partOnFlow(From, To, TR, P).

% CONSTRAINT 2 (The On Switch - Optional but helpful):
% If ASP says "Flow", Load must be at least 1.
%&sum { load(From, To, TR, P) } >= 1 :- partOnFlow(From, To, TR, P).

:- load(From, To, TR, P) >= 1, not partOnFlow(From, To, TR, P), possible_flow(From, To, TR, P).
% =========================================================
% 5. CALCULATIONS & BALANCE
% =========================================================
% Dynamic Domain for FlowIn/Out based on Total Supply


&sum { load(From, L, TR, P): possible_flow(From, L, TR, P)} = flowIn(P, L)
    :- location(L), part(P).

&sum { load(L, To, TR, P): possible_flow(L, To, TR, P)} = flowOut(P, L)
    :-  location(L), part(P).

% Balance Constraint
&sum { flowIn(P,L); -1 * flowOut(P,L) } = req(P,L) :- part(P), location(L).


active_flow(From, To, TR) :- partOnFlow(From, To, TR, _).
&dom { 0..100000000 } = total_weight(From, To, TR) :- active_flow(From, To, TR).
&sum {  S*load(From, To, TR, P) : partSize(P, S) } = total_weight(From, To, TR)
    :- active_flow(From, To, TR).


&dom { 0..10 } = estimatedFreq(From, To, TR) :- active_flow(From, To, TR).
&sum { Cap * estimatedFreq(From, To, TR) } >= total_weight(From, To, TR) :- active_flow(From, To, TR), transportCapacity(TR, Cap).


% 2. Tightly constrain the estimatedFreq domain
% Assuming Max=10 is safe, we now restrict the variable to [Min..10]
&dom { 0..10 } = estimatedFreq(From, To, TR) :- active_flow(From, To, TR).
    
&dom { 0..4654102 } = routeCostTotal(From, To, TR) :- active_flow(From, To, TR).

&sum { Cost * estimatedFreq(From, To, TR) :route(From, To, TR, _, Cost) } = routeCostTotal(From, To, TR) :- active_flow(From, To, TR).
  
%&dom { 0..4654102 } = total_transport_cost.

%&sum { routeCostTotal(From, To, TR) } = total_transport_cost 
%    :- active_flow(From, To, TR).

% 3. The correct way to minimize the total cost
&minimize {routeCostTotal(From, To, TR):possible_flow(From, To, TR,_)}.
