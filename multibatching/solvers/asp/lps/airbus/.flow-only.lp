%#const maxNrParts = 100.
%numFlow(1..maxNrParts).

transportResource(intrasite). 
transportCO2(intrasite,0). 
transportCost(intrasite,0).
transportCapacity(intrasite,2000000).
partTR(Part,intrasite) :- part(Part).

numFlowPart(Part,1..N) :- N=#sum{V:offer(Part,_,V)}, part(Part).

offer(Part,Location,Val):- demandOffer(Part,Location,Val), Val>0.
demand(Part,Location,PosVal):- demandOffer(Part,Location,Val), Val<0, PosVal=Val*-1.

root(eARFullFuselage). 

#show transportCapacity/2.

#const maxNrParts = 100.
numFlow(Part,1..Max) :- Max=#sum{V:offer(Part,_,V)}, part(Part).


% constraints

1{flow(From,To,Part,N): numFlow(N),N<=O, N<=D}1 :- offer(Part,From,O), demand(Part,To,D). 

 
:- DS!=#sum{N,From:flow(From,Loc,Part,N)}, demand(Part,Loc,DS).
:- DS!=#sum{N,To:flow(Loc,To,Part,N)}, offer(Part,Loc,DS), not root(Part).


direct(From,To,Part,TR) :- route(From,To,TR,_,_), demand(Part,To,_), offer(Part,From,_), partTR(Part,TR).
via1(From,(Via1,To),Part,(TR1,TR2)) :-  route(From,Via1,TR1,_,_), route(Via1,To,TR2,_,_), demand(Part,To,_), offer(Part,From,_),partTR(Part,TR1),partTR(Part,TR2).
via2(From,(Via1,Via2,To),Part,(TR1,TR2,TR3)) :-  route(From,Via1,TR1,_,_), route(Via1,Via2,TR2,_,_), route(Via2,To,TR3,_,_),  
offer(Part,From,_), demand(Part,To,_),partTR(Part,TR1),partTR(Part,TR2),partTR(Part,TR3).


1{routeflow(From,To,Part,N,TR):
direct(From,To,Part,TR);
routeflow(From,(Via1,To),Part,(N,N),(TR1,TR2)): via1(From,(Via1,To),Part,(TR1,TR2));
routeflow(From,(Via1,Via2,To),Part,(N,N,N),(TR1,TR2,TR3)):via2(From,(Via1,Via2,To),Part,(TR1,TR2,TR3))}1
:- flow(From,To,Part,N), From!=To.
