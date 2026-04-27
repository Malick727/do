
# Problem description : 

Object descriptions : 
- Set of assembly line, with a given takt pace, 
and a list of MSN that are scheduled on them (entry time/end time, engine installation date)
- Set of MSN object, with engine type attribute, entrytime etc
- Set of factories that prepare the engines before their installation. The factory is able to 
prepare engines of given types only. Factory has also a takt or a limited capacity per week something like that.
or just a duration of engine preparation actually.
- list of engines instance, having some type, and possibly some preallocation to MSN and preparation factory, 
and possibly (start, end) time of assembly, and a delivery time (unprepared)
- for each final assembly / preparation factory, some travel time and cost per engine

Constraint/Model description :
i want to allocate each engine instance to a MSN
i want to allocate each engine instance to a factory /preparation plant
i want to schedule the preparation of the engine in this factory plant
i want to know when the engine arrives at the fal where the msn allocated to it 

the constraint are : 
you can allocate the engine only to compatible factory, 
after its delivery time, the preparation time should be equal to the duration of the engine prep on the
allocated factory. etc,

Objective function : 
minimize the number of changes versus the common baselines (possible pre-allocation to msn and preparatory 
factories... start time of prep...) + sum of delays of arrival of the prepared engines for its MSN.


Please come up with a object oriented problem definition, that is highly understandable, 
use Problem/Solution approach of discrete-optimization library, try to implement the evaluate and satisfy accordingly.

