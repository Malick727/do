% location: name
location(l1).
location(l2).
location(l3).
location(l4).
location(l5).
location(l6).

% so via2 rule is applicable
harbor(l1;l2;l3;l4;l5;l6).

% transportResource: name, capacity, co2emissions, cost, speed
transportResource(tr1).
transportResource(tr2).

transportCapacity(tr1,10).
transportCapacity(tr2,15).

transportCO2(tr1,60).
transportCO2(tr2,45).

transportCost(tr1,56).
transportCost(tr2,42).

transportSpeed(tr1,3).
transportSpeed(tr2,2).

% Part: name, validTR, size, value
part(p1).
part(p2).
partSize(p1,4).
partSize(p2,3).
partTR(p1,tr1).
partTR(p1,tr2).
partTR(p2,tr1).
partTR(p2,tr2).
partVal(p1,1000).
partVal(p2,600).



demandOffer(p1,l1,1).
demandOffer(p2,l1,4).
demandOffer(p1,l2,2).

demandOffer(p2,l2,-1).
demandOffer(p1,l5,-1).
demandOffer(p2,l5,-2).
demandOffer(p1,l6,-2).
demandOffer(p2,l6,-1).

%demandOffer(p1,l1,10).
%demandOffer(p2,l1,40).
%demandOffer(p1,l2,20).

%demandOffer(p2,l2,-10).
%demandOffer(p1,l5,-10).
%demandOffer(p2,l5,-20).
%demandOffer(p1,l6,-20).
%demandOffer(p2,l6,-10).


% 'real' simple example including production plan constraints
% productionPlan(fullFuselageShort,section34).
%productionPlan(fullFuselageShort,section1).
%productionPlan(fullFuselageShort,section34).
%productionPlan(fullFuselageShort,section67).
%productionPlan(section34,section3).
%productionPlan(section34,section4).
%productionPlan(section67,section7).
%productionPlan(section67,section6).

%offer(l1,fullFuselageShort,10).
%demand(l1,section1,10).
%demand(l1,section34,10).
%demand(l1,section67,10).

%offer(l2,section1,10).

%offer(l4,section34,10).
%demand(l4,section3,10).
%demand(l4,section4,10).

%offer(l6,section67,10).
%demand(l6,section6,10).
%demand(l6,section7,10).

% route: from, to, transport mean, distance, cost (per 1)
route(1,l1,l2,tr1,1,56).
route(2,l1,l2,tr2,2,84).
route(3,l1,l3,tr1,3,168).
route(4,l1,l3,tr2,4,168).
route(5,l1,l4,tr1,4,224).
route(6,l1,l4,tr2,8,336).
route(7,l2,l3,tr1,1,56).
route(8,l2,l3,tr2,2,84).
route(9,l2,l4,tr1,2,112).
route(10,l2,l4,tr2,6,252).
route(11,l2,l6,tr1,2,112).
route(12,l3,l5,tr1,1,56).
route(13,l3,l5,tr2,3,126).
route(14,l3,l6,tr1,1,56).
route(15,l3,l6,tr2,2,84).
route(16,l4,l5,tr1,2,112).
route(17,l4,l5,tr2,4,168).
route(18,l4,l6,tr1,3,168).
route(19,l4,l6,tr2,5,210).