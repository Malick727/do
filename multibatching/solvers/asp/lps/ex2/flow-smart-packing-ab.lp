transportResource(intrasite). 
transportCO2(intrasite,0). 
transportCost(intrasite,0).
transportCapacity(intrasite,2000000).
partTR(Part,intrasite) :- part(Part).

#const maxFreq = 20.
num(0..maxFreq).

numFlowPart(Part,1..Max) :- Max=#sum{V:offer(Part,_,V)}, part(Part).

offer(Part,Location,Val):- demandOffer(Part,Location,Val), Val>0.
demand(Part,Location,PosVal):- demandOffer(Part,Location,Val), Val<0, PosVal=Val*-1.

root(eARFullFuselage). 

% flow

1{flow(To,To,Part,D)}1:- offer(To,Part,D), root(Part).

1{flow(From,To,Part,N): numFlowPart(Part,N),N<=O, N<=D}1 :- offer(Part,From,O), demand(Part,To,D), not root(Part). 

:- Val=#sum{N,From:flow(From,Loc,Part,N);-1*N,To:flow(Loc,To,Part,N)}, demandOffer(Part,Location,Val).


%:- DS!=#sum{N,From:flow(From,Loc,Part,N)}, demand(Part,Loc,DS).
%:- DS!=#sum{N,To:flow(Loc,To,Part,N)}, offer(Part,Loc,DS), not root(Part).

% routes

direct(From,To,Part,TR) :- route(From,To,TR,_,_), demand(Part,To,_), offer(Part,From,_), partTR(Part,TR).
via1(From,(Via1,To),Part,(TR1,TR2)) :-  route(From,Via1,TR1,_,_), route(Via1,To,TR2,_,_), demand(Part,To,_), offer(Part,From,_),partTR(Part,TR1),partTR(Part,TR2).
via2(From,(Via1,Via2,To),Part,(TR1,TR2,TR3)) :-  route(From,Via1,TR1,_,_), route(Via1,Via2,TR2,_,_), harbor(Via1), harbor(Via2), ship(TR2), route(Via2,To,TR3,_,_),  
offer(Part,From,_), demand(Part,To,_),partTR(Part,TR1),partTR(Part,TR2),partTR(Part,TR3).


% smart packing
bin(1..18).  
binTR(B,TR) :- bin(B), transportResource(TR).

1{packing(B,TR,Part,V):numFlowPart(Part,V)} :- binTR(B,TR).

:- binTR(B,TR),  transportCapacity(TR,Cap), Cap<#sum{V*Size,Part:packing(B,TR,Part,V),partSize(Part,Size)}.
:- packing(B,TR,Part,V1), packing(B,TR,Part,V2), V1!=V2.
#show packing/4.

#maximize{V*Size,Part:packing(B,TR,Part,V),partSize(Part,Size)}.

1{routeflow(From,To,Part,TR,B):
% condition for direct
direct(From,To,Part,TR),packing(B,TR,Part,M), M<=N;
routeflow(From,(Via1,To),Part,(TR1,TR2),(B1,B2)): 
% condition for via1
via1(From,(Via1,To),Part,(TR1,TR2)), packing(B1,TR1,Part,M1),packing(B2,TR2,Part,M2), M1+M2<=N; 
routeflow(From,(Via1,Via2,To),Part,(TR1,TR2,TR3),(B1,B2,B3)):
packing(B1,TR1,Part,M1),packing(B2,TR2,Part,M2),packing(B3,TR3,Part,M3),packing(B3,TR3,Part,M3), M1+M2+M3<=N,
via2(From,(Via1,Via2,To),Part,(TR1,TR2,TR3)
% condition for via2
)
}1 :- flow(From,To,Part,N), From!=To.

:- D!=M1+M2+M3, M1=routeflow(From,To,Part,TR,B),packing(B,TR,Part,M),
M2=routeflow(From,(Via1,To),Part,(TR1,TR2),(B1,B2)),packing(B2,TR2,Part,M2),
M3=routeflow(From,(Via1,Via2,To),Part,(TR1,TR2,TR3),(B1,B2,B3)),packing(B3,TR3,Part,M3),
flow(From,To,Part,D).
