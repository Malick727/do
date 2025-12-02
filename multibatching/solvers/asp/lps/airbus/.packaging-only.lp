
offer(Part,Location,Val):- demandOffer(Part,Location,Val), Val>0.
demand(Part,Location,PosVal):- demandOffer(Part,Location,Val), Val<0, PosVal=Val*-1.

numFlowPart(Part,0..Max) :- Max=#sum{V:offer(Part,_,V)}, part(Part).

#show numFlowPart/2.

bin(1..20).

binTR(B,TR) :- bin(B), transportResource(TR).

1{packing(B,TR,Part,Freq):numFlowPart(Part,Freq)} :- binTR(B,TR).

:- binTR(B,TR),  transportCapacity(TR,Cap), Cap<#sum{Freq*Size,Part:packing(B,TR,Part,Freq),partSize(Part,Size)}.

#show packing/4.

#maximize{Freq*Size,Part:packing(B,TR,Part,Freq),partSize(Part,Size)}.

