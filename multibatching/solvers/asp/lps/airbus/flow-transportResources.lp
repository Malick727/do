transportResource(intrasite). 
transportCO2(intrasite,0). 
transportCost(intrasite,0).
transportCapacity(intrasite,2000000).
partTR(Part,intrasite) :- part(Part).

#const maxFreq = 20.
num(0..maxFreq).
root(eARFullFuselage). 

numFlowPart(Part,1..Max) :- Max=#sum{V:offer(Part,_,V)}, part(Part).

offer(Part,Location,Val):- demandOffer(Part,Location,Val), Val>0.
demand(Part,Location,PosVal):- demandOffer(Part,Location,Val), Val<0, PosVal=Val*-1.

% flow
1{flow(To,To,Part,D)}1:- offer(To,Part,D), root(Part).
1{flow(From,To,Part,N): numFlowPart(Part,N),N<=O, N<=D}1 :- offer(Part,From,O), demand(Part,To,D), not root(Part). 

:- Val!=#sum{N*-1,From:flow(From,Loc,Part,N);M,To:flow(Loc,To,Part,M)}, demandOffer(Part,Loc,Val), Val<0.
:- Val!=#sum{N*-1,From:flow(From,Loc,Part,N);M,To:flow(Loc,To,Part,M)}, demandOffer(Part,Loc,Val), Val>0.

% routes
direct(From,To,Part,TR) :- route(From,To,TR,_,_), demand(Part,To,_), offer(Part,From,_), partTR(Part,TR).
via1(From,(Via1,To),Part,(TR1,TR2)) :-  route(From,Via1,TR1,_,_), route(Via1,To,TR2,_,_), demand(Part,To,_), offer(Part,From,_),partTR(Part,TR1),partTR(Part,TR2).
via2(From,(Via1,Via2,To),Part,(TR1,TR2,TR3)) :-  route(From,Via1,TR1,_,_), route(Via1,Via2,TR2,_,_), harbor(Via1), harbor(Via2), route(Via2,To,TR3,_,_),  
offer(Part,From,_), demand(Part,To,_),partTR(Part,TR1),partTR(Part,TR2),partTR(Part,TR3).

flowWithoutRoute(From,To,Part) :- notDirect(From,To,Part), notVia1(From,To,Part), notVia2(From,To,Part).

notDirect(From,To,Part) :- not direct(From,To,Part,_), location(From), location(To), part(Part), flow(From,To,Part,_).
notVia1(From,To,Part) :- not via1(From,(_,To),Part,_), location(From), location(To), part(Part), flow(From,To,Part,_).
notVia2(From,To,Part) :- not via2(From,(_,_,To),Part,_), location(From), location(To), part(Part), flow(From,To,Part,_).

:- flowWithoutRoute(From,To, Part).

#show flowWithoutRoute/3.

1{route(From,To,Part,TR):
% condition for direct
direct(From,To,Part,TR);
route(From,(Via1,To),Part,(TR1,TR2)): 
% condition for via1
via1(From,(Via1,To),Part,(TR1,TR2));
% condition for via2
route(From,(Via1,Via2,To),Part,(TR1,TR2,TR3)):
via2(From,(Via1,Via2,To),Part,(TR1,TR2,TR3))
}1 :- flow(From,To,Part,_), From!=To.

1{routeflow(From,To,Part,TR,N1):
% condition for direct
direct(From,To,Part,TR), numFlowPart(Part,N1);
routeflow(From,(Via1,To),Part,(TR1,TR2),(N1,N2)): 
% condition for via1
via1(From,(Via1,To),Part,(TR1,TR2)), numFlowPart(Part,N1), numFlowPart(Part,N2);
% condition for via2
routeflow(From,(Via1,Via2,To),Part,(TR1,TR2,TR3),(N,N,N)):
via2(From,(Via1,Via2,To),Part,(TR1,TR2,TR3))
}1 :- flow(From,To,Part,N), route(From,To,Part,TR).

:- routeflow(From,To,Part,_,N), flow(From,To,Part,Val), N>Val.
:- routeflow(From,To,Part,_,(N,_)), flow(From,To,Part,Val), N>Val.
:- routeflow(From,To,Part,_,(N,_)), flow(From,To,Part,Val), N>Val.
:- routeflow(From,To,Part,_,(N,_,_)), flow(From,To,Part,Val), N>Val.
:- routeflow(From,To,Part,_,(_,N,_)), flow(From,To,Part,Val), N>Val.
:- routeflow(From,To,Part,_,(_,_,N)), flow(From,To,Part,Val), N>Val.



%:- Val!=#sum{N1,TR1:routeflow(From,Loc,Part,TR1,N1);
%N2,TR2:routeflow(From,(_,Loc),Part,TR2,N2);
%N3,TR3:routeflow(From,(_,_,Loc),Part,TR3,N3)}, 
%flow(From,Loc,Part,Val).

