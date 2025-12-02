

demandOffer(P,L,0) :- not offer(P,L,_), not demand(P,L,_), part(P), location(L).
demandOffer(P,L,O) :- offer(P,L,O).
demandOffer(P,L,D) :- demand(P,L,M), D=M*-1.

#const maxFreq = 10.

num(0..maxFreq).
numFlowPart(Part,1..Max) :- Max=#sum{V:offer(Part,_,V)}, part(Part).

% constraints

% Specify possible flow

1{flow(From,To,Part,N): numFlowPart(Part,N),N<=O, N<=D}1 :- offer(Part,From,O), demand(Part,To,D). 

:- flow(From,To,Part,N1), flow(From,To,Part,N2), N1!=N2.
:- flow(From,To,Part,N), demand(Part,From,D).
:- flow(From,To,Part,N), offer(Part,To,D).

% make sure that flow constraints hold

 
:- DS!=#sum{N,From:flow(From,Loc,Part,N)}, demand(Part,Loc,DS).
:- DS!=#sum{N,To:flow(Loc,To,Part,N)}, offer(Part,Loc,DS), not root(Part).

direct(From,To,Part,TR) :- route(From,To,TR,_,_), demand(Part,To,_), offer(Part,From,_), partTR(Part,TR).
via1(From,(Via1,To),Part,(TR1,TR2)) :-  route(From,Via1,TR1,_,_), route(Via1,To,TR2,_,_), demand(Part,To,_), offer(Part,From,_),partTR(Part,TR1),partTR(Part,TR2).
via2(From,(Via1,Via2,To),Part,(TR1,TR2,TR3)) :-  route(From,Via1,TR1,_,_), route(Via1,Via2,TR2,_,_), route(Via2,To,TR3,_,_),  
offer(Part,From,_), demand(Part,To,_),partTR(Part,TR1),partTR(Part,TR2),partTR(Part,TR3).

%#maximize{V*Size,Part:packing(B,TR,Part,V),partSize(Part,Size)}.

1{routeflow(From,To,Part,N1,Freq):
% condition for direct
direct(From,To,Part,TR),numFlowPart(Part,N1),num(Freq);
% condition for via1
routeflow(From,(Via1,To),Part,(N1,N2),(Freq1,Freq2)): 
via1(From,(Via1,To),Part,(TR1,TR2)),numFlowPart(Part,N1), numFlowPart(Part,N2), num(Freq1), num(Freq2);
routeflow(From,(Via1,Via2,To),Part,(N1,N2,N3),(Freq1,Freq2,Freq3)):
% condition for via2
via2(From,(Via1,Via2,To),Part,(TR1,TR2,TR3)),numFlowPart(Part,N1), numFlowPart(Part,N2), numFlowPart(Part,N3), num(Freq1), num(Freq2), num(Freq3)}1
:- flow(From,To,Part,N), From!=To.

:- routeflow(From,To,Part,N1,Freq), flow(From,To,Part,N), location(To), N1*Freq>N.
:- routeflow(From,(_,To),Part,(N1,N2),(Freq1,Freq2)), flow(From,To,Part,N), location(To), N1*Freq1>N.
:- routeflow(From,(_,To),Part,(N1,N2),(Freq1,Freq2)), flow(From,To,Part,N), location(To), N2*Freq2>N.
:- routeflow(From,(_,_,To),Part,(N1,N2,N3),(Freq1,Freq2,Freq3)), flow(From,To,Part,N), location(To),  N1*Freq1>N.
:- routeflow(From,(_,_,To),Part,(N1,N2,N3),(Freq1,Freq2,Freq3)), flow(From,To,Part,N), location(To),  N2*Freq2>N.
:- routeflow(From,(_,_,To),Part,(N1,N2,N3),(Freq1,Freq2,Freq3)), flow(From,To,Part,N), location(To),  N3*Freq3>N.

#show routeflow/5.

% smart packing
bin(1..10).  
binTR(B,TR) :- bin(B), transportResource(TR).

1{packing(B,TR,Part,V):numFlowPart(Part,V),binTR(B,TR),V<=N} :- routeflow(From,_,Part,N).

:- binTR(B,TR),  transportCapacity(TR,Cap), Cap<#sum{V*Size,Part:packing(B,TR,Part,V),partSize(Part,Size)}.
:- packing(B,TR,Part,V1), packing(B,TR,Part,V2), V1!=V2.
%#show packing/4.

1{routepacking(From,ToTo,Part,B):
% condition for direct
location(ToTo),packing(B,TR,Part,_);
% condition for via1
routepacking(From,(Via1,To),Part,(B1,B2)): 
ToTo=(Via1,To), location(To), packing(B1,TR1,Part,_),packing(B2,TR2,Part,_);
routepacking(From,(Via1,Via2,To),Part,(B1,B2,B3)):
% condition for via2
ToTo=(Via1,Via2,To),
packing(B1,TR1,Part,_),packing(B2,TR2,Part,_),packing(B3,TR3,Part,_)}1
:- routeflow(From,ToTo,Part,_).

#show routepacking/4.

%:- routepacking(From,ToTo,Part,B), routeflow(From,ToTo,Part,N), 


%:- routeflow(From,ToTo,Part,N), routepacking(From,ToTo,Part,B), packing(B,TR,Part,A), A!=N.

%helperPartPackFrom(From,Part,T) :- 
%T=#sum{A,To:routepacking(From,To,Part,B),packing(B,_,Part,A);A,To:routepacking(From,(Via1,To),Part,(B1,_)),
%packing(B1,_,Part,A);A,To:routepacking(From,(Via1,Via2,To),Part,(B1,_,_)),
%packing(B1,_,Part,A)}, flow(From,_,Part).



#show helperPartPackFrom/3.
#show helperPartPackTo/3.
%:- DS!=#sum{N,To:flow(Loc,To,Part,N)}, offer(Part,Loc,DS), not root(Part).


%1{routepacking(From,ToTo,Part,B):
%% condition for direct
%location(ToTo),packing(B,TR,Part,M),M<=NN;
%% condition for via1
%routepacking(From,(Via1,To),Part,(B1,B2)): 
%ToTo=(Via1,To),
%NN=(N1,N2), location(To), packing(B1,TR1,Part,M1),packing(B2,TR2,Part,M2),M1<=N1,M2<=N2;
%routepacking(From,(Via1,Via2,To),Part,(B1,B2,B3)):
%% condition for via2
%ToTo=(Via1,Via2,To), NN=(N1,N2,N3),
%packing(B1,TR1,Part,M1),packing(B2,TR2,Part,M2),packing(B3,TR3,Part,M3),M1<=N1,M2<=N2,M3<=N3}1
%:- routeflow(From,ToTo,Part,NN).

%#show routepacking/4.
