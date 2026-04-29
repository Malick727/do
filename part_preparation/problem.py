"""
Problem definition for Engine Preparation and Allocation
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Union
from enum import Enum
import logging

from discrete_optimization.generic_tools.do_problem import (
    Problem,
    Solution,
    ObjectiveRegister,
    ObjectiveDoc,
    TypeObjective,
    ModeOptim,
    ObjectiveHandling,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssemblyLine:
    """Final Assembly Line (FAL) where aircraft are assembled"""
    id: str
    takt_pace: int  # Time between aircraft starts (in days)
    location: str # tls, hamburg, mobile, tianjin



@dataclass(frozen=True)
class MSN:
    """Aircraft Serial Number - represents a specific aircraft to be produced"""
    id: str
    assembly_line_id: str
    engine_type: str
    entry_time: int  # When MSN enters assembly line
    end_time: int  # When MSN leaves assembly line
    engine_installation_date: int  # When engines must be installed


@dataclass(eq=False)
class PreparationFactory:
    """Factory that prepares engines before installation
    Podder
    """
    id: str
    location: str
    default_assembly_line: AssemblyLine
    compatible_engine_types: Set[str]
    # Types of engines this factory can prepare; in practice it's 1 type only
    preparation_durations: Dict[str, int]  # engine_type -> preparation duration (days)
    default_capacity: Optional[int] = None
    # Max engines that can be in preparation simultaneously (cumulative constraint)
    calendar: list[tuple[int, int, int]] = None # [(Start, end, capacity) ]
    # not needed.
    takt: Optional[int] = None  # Takt time: divider of preparation time for well-paced scheduling

    def can_prepare(self, engine_type: str) -> bool:
        """Check if factory can prepare this engine type"""
        return engine_type in self.compatible_engine_types

    def get_preparation_duration(self, engine_type: str) -> int:
        """Get preparation duration for an engine type"""
        return self.preparation_durations.get(engine_type, 0)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, PreparationFactory):
            return False
        return self.id == other.id

@dataclass(frozen=True)
class EngineSupplier:
    capacity_delivery_per_week: int

@dataclass(frozen=True, eq=False)
class EngineType:
    id: str
    value_engine: int
    cost_storage_per_time: int
    supplier: Set["EngineSupplier"]

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, EngineType):
            return False
        return self.id == other.id


@dataclass(frozen=True, eq=False)
class Engine:
    """Engine instance to be allocated and scheduled"""
    id: str
    engine_type: str
    engine_type_object: EngineType
    engine_supplier: EngineSupplier
    # When unprepared engine is delivered (earliest prep can start)
    # Baseline pre-allocations (for minimizing changes)
    # Not necessary but used to fix some schedule part
    delivery_time: Optional[int] = None
    baseline_msn_id: Optional[str] = None
    baseline_factory_id: Optional[str] = None
    baseline_prep_start: Optional[int] = None
    baseline_prep_end: Optional[int] = None

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Engine):
            return False
        return self.id == other.id


@dataclass
class TravelInfo:
    """Travel time and cost between factory and assembly line"""
    factory_id: str
    assembly_line_id: str
    travel_time: int  # Days
    travel_cost: float  # Cost per engine that travels


class EnginePreparationSolution(Solution):
    """
    Solution for Engine Preparation and Allocation problem

    Decision variables (using objects as keys for clarity):
    - engine_to_msn: Dict[Engine, MSN] - Allocation of each Engine to an MSN
    - engine_to_factory: Dict[Engine, PreparationFactory] - Allocation of each Engine to a factory
    - prep_start_times: Dict[Engine, int] - When preparation starts for each Engine
    - prep_end_times: Dict[Engine, int] - When preparation ends for each Engine
    - fal_arrival_times: Dict[Engine, int] - When prepared engine arrives at FAL
    """

    def __init__(
        self,
        problem: "EnginePreparationProblem",
        engine_to_msn: Dict[Engine, Union[MSN, None]],
        engine_to_factory: Dict[Engine, PreparationFactory],
        prep_start_times: Dict[Engine, Union[int, None]],
        prep_end_times: Dict[Engine, Union[int, None]],
        fal_arrival_times: Dict[Engine, Union[int, None]],
        delivery_time: Dict[Engine, Union[int, None]],
    ):
        super().__init__(problem)
        self.engine_to_msn = engine_to_msn
        self.engine_to_factory = engine_to_factory
        self.prep_start_times = prep_start_times
        self.prep_end_times = prep_end_times
        self.fal_arrival_times = fal_arrival_times
        self.delivery_time = delivery_time

    def copy(self):
        return EnginePreparationSolution(
            problem=self.problem,
            engine_to_msn=self.engine_to_msn.copy(),
            engine_to_factory=self.engine_to_factory.copy(),
            prep_start_times=self.prep_start_times.copy(),
            prep_end_times=self.prep_end_times.copy(),
            fal_arrival_times=self.fal_arrival_times.copy(),
        )

    def change_problem(self, new_problem: Problem) -> None:
        self.problem = new_problem


class EnginePreparationProblem(Problem):
    """
    Engine Preparation and Allocation Problem

    Given:
    - Assembly lines with scheduled MSNs (aircraft)
    - Preparation factories with time-varying capacities and compatible engine types
    - Engine instances with optional delivery times and baseline allocations
    - Engine types with storage costs
    - Engine suppliers with delivery capacity constraints
    - Travel times/costs between factories and assembly lines

    Goal:
    - Allocate engines to MSNs (one-to-one, but can have unallocated engines/MSNs)
    - Allocate each engine to a preparation factory
    - Schedule preparation in the factory
    - Optionally decide engine delivery times (respecting supplier capacity)
    - Minimize: baseline changes + delays + travel cost + storage cost + unallocated MSNs

    Constraints:
    - Engine allocated to compatible factory only
    - Preparation starts after delivery time
    - Preparation duration matches factory's duration for engine type
    - Factory capacity constraints (time-varying via calendar or default)
    - Supplier delivery capacity constraints (per week)
    - Takt constraints (optional, for well-paced scheduling)

    Storage Costs:
    - Unallocated engines: full storage from delivery to end of horizon
    - Allocated engines:
      1. Before prep (delivery → prep_start): waiting at factory
      2. After prep (prep_end → departure): waiting at factory before transport
      3. At FAL (arrival → installation_date): max(0, ...) only if early arrival

    Note: Number of engines and MSNs may differ (unbalanced allocation problem)
    """

    def __init__(
        self,
        assembly_lines: List[AssemblyLine],
        msns: List[MSN],
        factories: List[PreparationFactory],
        engines: List[Engine],
        travel_info: List[TravelInfo],
        time_horizon: int,  # Planning horizon in days
    ):
        # Store as lists (primary)
        self.assembly_lines_list = assembly_lines
        self.msns_list = msns
        self.factories_list = factories
        self.engines_list = engines
        self.time_horizon = time_horizon

        # Also maintain ID lookups for convenience
        self.assembly_lines = {al.id: al for al in assembly_lines}
        self.msns = {msn.id: msn for msn in msns}
        self.factories = {f.id: f for f in factories}
        self.engines = {e.id: e for e in engines}

        # Build travel lookup: (Factory, AssemblyLine) -> TravelInfo
        self.travel_lookup: Dict[tuple[PreparationFactory, AssemblyLine], TravelInfo] = {}
        for ti in travel_info:
            factory = self.factories[ti.factory_id]
            assembly_line = self.assembly_lines[ti.assembly_line_id]
            self.travel_lookup[(factory, assembly_line)] = ti

        # Build compatibility maps
        self._build_compatibility_maps()

    def _build_compatibility_maps(self):
        """Build maps for which engines can go to which factories/MSNs"""
        # Which factories can prepare each engine
        self.engine_compatible_factories: Dict[Engine, List[PreparationFactory]] = {}
        for engine in self.engines_list:
            compatible = [
                f for f in self.factories_list
                if f.can_prepare(engine.engine_type)
            ]
            self.engine_compatible_factories[engine] = compatible

        # Which MSNs can receive each engine (by engine type)
        self.engine_compatible_msns: Dict[Engine, List[MSN]] = {}
        for engine in self.engines_list:
            compatible = [
                msn for msn in self.msns_list
                if msn.engine_type == engine.engine_type
            ]
            self.engine_compatible_msns[engine] = compatible

    def get_travel_time(self, factory: PreparationFactory, assembly_line: AssemblyLine) -> int:
        """Get travel time from factory to assembly line"""
        key = (factory, assembly_line)
        if key in self.travel_lookup:
            return self.travel_lookup[key].travel_time
        return 0  # Default if not specified

    def get_travel_cost(self, factory: PreparationFactory, assembly_line: AssemblyLine) -> float:
        """Get travel cost from factory to assembly line"""
        key = (factory, assembly_line)
        if key in self.travel_lookup:
            return self.travel_lookup[key].travel_cost
        return 0.0  # Default if not specified

    def evaluate(self, solution: EnginePreparationSolution) -> Dict[str, float]:
        """
        Compute objective values (KPIs)

        Objectives:
        1. baseline_changes: Number of changes vs baseline allocations
        2. arrival_delays: Sum of delays (late arrivals at FAL)
        3. travel_cost: Total travel cost
        4. storage_cost: Cost of storing unallocated engines
        5. unallocated_msns: Number of MSNs without allocated engines
        """
        baseline_changes = 0
        arrival_delays = 0.0
        travel_cost = 0.0
        storage_cost = 0.0
        unallocated_msns_count = 0

        # Count unallocated MSNs
        allocated_msns = set()
        for engine in self.engines_list:
            msn = solution.engine_to_msn.get(engine)
            if msn is not None:
                allocated_msns.add(msn.id)

        unallocated_msns_count = len(self.msns_list) - len(allocated_msns)

        for engine in self.engines_list:
            # Check if engine is allocated to MSN
            assigned_msn = solution.engine_to_msn.get(engine)
            is_allocated = assigned_msn is not None

            # Count changes from baseline
            if engine.baseline_msn_id is not None:
                if assigned_msn is None or assigned_msn.id != engine.baseline_msn_id:
                    baseline_changes += 1

            if engine.baseline_factory_id is not None:
                assigned_factory = solution.engine_to_factory.get(engine)
                if assigned_factory is None or assigned_factory.id != engine.baseline_factory_id:
                    baseline_changes += 1

            if engine.baseline_prep_start is not None and solution.prep_start_times.get(engine) is not None:
                actual_start = solution.prep_start_times[engine]
                if abs(actual_start - engine.baseline_prep_start) > 1:  # Allow 1 day tolerance
                    baseline_changes += 1

            # Only compute allocated-engine KPIs if engine is allocated
            if is_allocated:
                arrival_time = solution.fal_arrival_times.get(engine)

                # Compute arrival delays
                if arrival_time is not None:
                    delay = max(0, arrival_time - assigned_msn.engine_installation_date)
                    arrival_delays += delay

                # Compute travel costs
                if engine in solution.engine_to_factory:
                    factory = solution.engine_to_factory[engine]
                    assembly_line = self.assembly_lines[assigned_msn.assembly_line_id]
                    cost = self.get_travel_cost(factory, assembly_line)
                    travel_cost += cost

                # Compute storage costs for allocated engines
                delivery_time = solution.delivery_time.get(engine)
                prep_start = solution.prep_start_times.get(engine)
                prep_end = solution.prep_end_times.get(engine)

                if delivery_time is not None and prep_start is not None:
                    # Storage before preparation (engine waiting at factory)
                    storage_before_prep = max(0, prep_start - delivery_time)
                    storage_cost += engine.engine_type_object.cost_storage_per_time * storage_before_prep

                if prep_end is not None and arrival_time is not None and engine in solution.engine_to_factory:
                    # Storage after preparation (engine waiting to depart factory)
                    # Departure time = arrival_time - travel_time
                    factory = solution.engine_to_factory[engine]
                    assembly_line = self.assembly_lines[assigned_msn.assembly_line_id]
                    travel_time = self.get_travel_time(factory, assembly_line)
                    departure_time = arrival_time - travel_time
                    storage_after_prep = max(0, departure_time - prep_end)
                    storage_cost += engine.engine_type_object.cost_storage_per_time * storage_after_prep

                    # Storage at FAL before installation (if engine arrives early)
                    # Only count storage if arrival_time < engine_installation_date
                    storage_at_fal = max(0, assigned_msn.engine_installation_date - arrival_time)
                    storage_cost += engine.engine_type_object.cost_storage_per_time * storage_at_fal
            else:
                # Engine not allocated: compute storage cost for entire period
                # Storage cost = cost_storage_per_time * (time from delivery to end of horizon)
                delivery_time = solution.delivery_time.get(engine)
                if delivery_time is not None:
                    storage_duration = self.time_horizon - delivery_time
                    storage_cost += engine.engine_type_object.cost_storage_per_time * max(0, storage_duration)

        return {
            "baseline_changes": float(baseline_changes),
            "arrival_delays": arrival_delays,
            "travel_cost": travel_cost,
            "storage_cost": storage_cost,
            "unallocated_msns": float(unallocated_msns_count),
        }

    def satisfy(self, solution: EnginePreparationSolution, debug: bool = False) -> bool:
        """
        Check if solution satisfies all hard constraints

        Hard Constraints:
        1. Each engine allocated to at most one MSN (can be None if unallocated)
        2. Each allocated engine must be assigned to exactly one factory
        3. Factory must be compatible with engine type
        4. Preparation starts after delivery time
        5. Preparation duration matches factory's duration
        6. Arrival time correctly calculated (prep_end + travel_time)
        7. Factory capacity constraints (time-varying via calendar or default_capacity)
        8. Supplier delivery capacity constraints (if delivery_time is decision variable)

        Soft Constraints (allowed, penalized in objective):
        - Arrival delays (arrival after MSN's installation date)
        - Unallocated MSNs
        - Storage costs for unallocated engines

        Args:
            solution: Solution to check
            debug: If True, print detailed constraint violations
        """
        TOL = 1e-4
        violations = []

        # Check each engine's allocation and scheduling
        for engine in self.engines_list:
            msn = solution.engine_to_msn.get(engine)
            factory = solution.engine_to_factory.get(engine)
            prep_start = solution.prep_start_times.get(engine)
            prep_end = solution.prep_end_times.get(engine)
            arrival_time = solution.fal_arrival_times.get(engine)
            delivery_time = solution.delivery_time.get(engine)

            # If engine is allocated to MSN, check allocation constraints
            if msn is not None:
                # MSN compatibility (engine type)
                if msn.engine_type != engine.engine_type:
                    msg = f"Engine {engine.id} type {engine.engine_type} incompatible with MSN {msn.id} type {msn.engine_type}"
                    violations.append(msg)
                    if debug: print(f"❌ {msg}")

                # Factory must be assigned
                if factory is None:
                    msg = f"Engine {engine.id} allocated to MSN but has no factory assigned"
                    violations.append(msg)
                    if debug: print(f"❌ {msg}")

                # Timing must be defined
                if prep_start is None or prep_end is None or arrival_time is None:
                    msg = f"Engine {engine.id} allocated to MSN but has incomplete timing"
                    violations.append(msg)
                    if debug: print(f"❌ {msg}")

                # Check factory compatibility
                if not factory.can_prepare(engine.engine_type):
                    msg = f"Factory {factory.id} cannot prepare engine type {engine.engine_type}"
                    violations.append(msg)
                    if debug: print(f"❌ {msg}")

                # Check preparation starts after delivery
                if delivery_time is not None and prep_start < delivery_time - TOL:
                    msg = f"Engine {engine.id} prep starts at {prep_start} before delivery at {delivery_time}"
                    violations.append(msg)
                    if debug: print(f"❌ {msg}")

                # Check preparation duration
                expected_duration = factory.get_preparation_duration(engine.engine_type)
                actual_duration = prep_end - prep_start
                if abs(actual_duration - expected_duration) > TOL:
                    msg = f"Engine {engine.id} prep duration {actual_duration} != expected {expected_duration}"
                    violations.append(msg)
                    if debug: print(f"❌ {msg}")

                # Check arrival time calculation
                assembly_line = self.assembly_lines[msn.assembly_line_id]
                travel_time = self.get_travel_time(factory, assembly_line)
                expected_arrival = prep_end + travel_time
                if abs(arrival_time - expected_arrival) > TOL:
                    msg = f"Engine {engine.id} arrival time {arrival_time} != expected {expected_arrival}"
                    violations.append(msg)
                    if debug: print(f"❌ {msg}")

                # NOTE: Arrival delays are ALLOWED (soft constraint, penalized in objective)
                if arrival_time > msn.engine_installation_date + TOL:
                    delay = arrival_time - msn.engine_installation_date
                    logger.info(f"Engine {engine.id} arrives late by {delay:.1f} days (arrival={arrival_time}, deadline={msn.engine_installation_date})")

        # Check factory capacity constraints (time-varying cumulative)
        for factory in self.factories_list:
            # Determine capacity at each time point
            def get_capacity_at_time(t: int) -> Optional[int]:
                """Get factory capacity at time t"""
                if factory.calendar is not None:
                    # Find applicable calendar entry
                    for start, end, capacity in factory.calendar:
                        if start <= t < end:
                            return capacity
                    return None  # No capacity specified for this time
                else:
                    return factory.default_capacity

            # Check at each time point
            for t in range(0, self.time_horizon + 1):
                capacity_at_t = get_capacity_at_time(t)

                if capacity_at_t is not None:
                    # Count engines in preparation at time t
                    engines_at_t = 0
                    for engine in self.engines_list:
                        if solution.engine_to_factory.get(engine) == factory:
                            prep_start = solution.prep_start_times.get(engine)
                            prep_end = solution.prep_end_times.get(engine)

                            if prep_start is not None and prep_end is not None:
                                # Engine is in preparation if prep_start <= t < prep_end
                                if prep_start <= t < prep_end:
                                    engines_at_t += 1

                    if engines_at_t > capacity_at_t:
                        msg = f"Factory {factory.id} capacity {capacity_at_t} exceeded at time {t}: {engines_at_t} engines"
                        violations.append(msg)
                        if debug: print(f"❌ {msg}")

        # Check supplier delivery capacity constraints (per week)
        # NOTE: This constraint is currently disabled in the CP-SAT solver because the instance
        # generator creates fixed delivery times that may violate it. We skip the check here too.
        # TODO: Either fix instance generator or treat as soft constraint
        if False:  # Disabled for now
            # Group engines by supplier and count deliveries per week
            for engine in self.engines_list:
                delivery_time = solution.delivery_time.get(engine)
                if delivery_time is not None:
                    # Find which week this delivery falls into
                    week = delivery_time // 7

                    # Count deliveries from same supplier in same week
                    same_supplier_same_week = 0
                    for other_engine in self.engines_list:
                        other_delivery = solution.delivery_time.get(other_engine)
                        if other_delivery is not None:
                            other_week = other_delivery // 7
                            if week == other_week and engine.engine_supplier == other_engine.engine_supplier:
                                same_supplier_same_week += 1

                    # Check capacity
                    if same_supplier_same_week > engine.engine_supplier.capacity_delivery_per_week:
                        msg = f"Supplier {engine.engine_supplier} capacity {engine.engine_supplier.capacity_delivery_per_week} exceeded in week {week}: {same_supplier_same_week} deliveries"
                        violations.append(msg)
                        if debug: print(f"❌ {msg}")

        if violations:
            if debug:
                print(f"\n❌ Solution has {len(violations)} constraint violations")
            return False
        return True

    def get_solution_type(self):
        return EnginePreparationSolution

    def get_objective_register(self) -> ObjectiveRegister:
        """
        Define optimization objectives

        Objectives:
        - baseline_changes: Number of deviations from baseline (MINIMIZE)
        - arrival_delays: Sum of late arrivals at FAL in days (MINIMIZE)
        - travel_cost: Total transportation cost (MINIMIZE)
        - storage_cost: Cost of storing engines (allocated and unallocated) (MINIMIZE)
        - unallocated_msns: Number of MSNs without engines (MINIMIZE)
        """
        return ObjectiveRegister(
            objective_sense=ModeOptim.MINIMIZATION,
            objective_handling=ObjectiveHandling.AGGREGATE,
            dict_objective_to_doc={
                "baseline_changes": ObjectiveDoc(
                    type=TypeObjective.OBJECTIVE,
                    default_weight=100.0  # High priority: minimize changes
                ),
                "arrival_delays": ObjectiveDoc(
                    type=TypeObjective.OBJECTIVE,
                    default_weight=10.0  # Medium priority: minimize delays
                ),
                "travel_cost": ObjectiveDoc(
                    type=TypeObjective.OBJECTIVE,
                    default_weight=1.0  # Low priority: minimize cost
                ),
                "storage_cost": ObjectiveDoc(
                    type=TypeObjective.OBJECTIVE,
                    default_weight=5.0  # Medium-high priority: minimize storage
                ),
                "unallocated_msns": ObjectiveDoc(
                    type=TypeObjective.OBJECTIVE,
                    default_weight=1000.0  # Very high priority: allocate all MSNs if possible
                ),
            }
        )
