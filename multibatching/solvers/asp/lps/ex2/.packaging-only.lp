numFlowPart(Part,1..Max) :- Max=#sum{V:offer(Part,_,V)}, part(Part).

bin(1..20).

binTR(B,TR) :- transportResource(TR), bin(B).

1{smartPack(B,TR,Part,Freq):numFlowPart(Part,Freq)} :- binTR(B,TR).

:- binTR(TR,B),  transportCapacity(TR,Cap), Cap < #sum{Freq*Size,Part:smartPack(B,TR,Part,Freq),partSize(Part,Size)}.


#show smartPack/4.

#maximize{Freq*Size,Part:smartPack(B,TR,Part,Freq),partSize(Part,Size)}.