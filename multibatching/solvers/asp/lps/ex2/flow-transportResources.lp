transportResource(intrasite). 
transportCO2(intrasite,0). 
transportCost(intrasite,0).
transportCapacity(intrasite,2000000).
%partTR(Part,intrasite) :- part(Part).

num(0..20).

% =========================================================
% 1. DATA PREPARATION (Efficient)
% =========================================================

% 1. CALCULATE BOUNDS (For Domain Efficiency)
% We calculate the total supply per part to limit domains strictly.
% If total supply is 500, no edge needs a domain of 20,000.

numFlowPart(P,1..Max) :- Max= #sum { V*2 : demandOffer(P,_,V), V > 0 }, part(P).
total_supply(P, S) :- S = #sum { V*2 : demandOffer(P,_,V), V > 0 }, part(P).

% 2. ROBUST DEMAND/OFFER
has_demand(P, L) :- demandOffer(P, L, V), V < 0.
has_offer(P, L)  :- demandOffer(P, L, V), V > 0.

demand(P, L, Val * -1) :- demandOffer(P, L, Val), Val < 0.
demand(P, L, 0)        :- part(P), location(L), not has_demand(P, L).

offer(P, L, Val) :- demandOffer(P, L, Val), Val > 0.
offer(P, L, 0)   :- part(P), location(L), not has_offer(P, L).

% 3. CALCULATE NET REQUIREMENT
requiredNet(P, L, Net) :- demand(P,L,D), offer(P,L,O), Net = D - O.
&dom { N..N } = req(P,L) :- requiredNet(P, L, N).

% We removed 'A' from this definition. 
% We only care that a route EXISTS between From and To.
possible_flow(From, To, TR, P) :- route(From, To, TR, _, _), part(P).

% =========================================================
% 3. THE GENERATOR (Boolean Only!)
% =========================================================
{ partOnFlow(From, To, TR, P)} :- possible_flow(From, To, TR, P).

% =========================================================
% 4. CSP VARIABLE SWITCH
% =========================================================

% DYNAMIC DOMAIN: Limit load to the Total Supply of that part.
% (Prevents domain explosion).
&dom { 0..Max } = load(From, To, TR, P) :- 
    possible_flow(From, To, TR, P),total_supply(P, Max) .

% CONSTRAINT 1 (The Off Switch):
% If ASP says "No Flow", Load MUST be 0.
&sum { load(From, To, TR, P) } = 0 :- possible_flow(From, To, TR, P), 
                                      not partOnFlow(From, To, TR, P).

% CONSTRAINT 2 (The On Switch - Optional but helpful):
% If ASP says "Flow", Load must be at least 1.
&sum { load(From, To, TR, P) } >= 1 :- partOnFlow(From, To, TR, P).

% =========================================================
% 5. CALCULATIONS & BALANCE
% =========================================================
% Dynamic Domain for FlowIn/Out based on Total Supply

&dom { 0..Max } = flowIn(P, L)  :- part(P), location(L), total_supply(P, Max).
&dom { 0..Max } = flowOut(P, L) :- part(P), location(L), total_supply(P, Max).

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
&sum { Cap * estimatedFreq(From, To, TR) } >= total_weight(From, To, TR)
    :- active_flow(From, To, TR), transportCapacity(TR, Cap).

%&minimize {load(From, To, TR, P) : possible_flow(From, To, TR, P)}.

&dom { 0..10000000 } = routeCostTotal(From, To, TR) :- active_flow(From, To, TR).

&sum { Cost * estimatedFreq(From, To, TR) :route(From, To, TR, _, Cost) } = routeCostTotal(From, To, TR)
    :- active_flow(From, To, TR).
%&sum { routeCostTotal(From, To, TR) } = total_transport_cost :- active_flow(From, To, TR).

%&minimize {total_transport_cost}.
&minimize {routeCostTotal(From, To, TR):possible_flow(From, To, TR,_)}.

