"""
Problem definition for Engine Preparation and Allocation
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Set
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
    capacity: Optional[int] = None
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

@dataclass
class EngineType:
    id: str
    value_engine: int
    cost_storage_per_time: int
    supplier: Set["EngineSupplier"]

@dataclass
class EngineSupplier:
    capacity_delivery_per_week: int


@dataclass(frozen=True)
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
        engine_to_msn: Dict[Engine, MSN],
        engine_to_factory: Dict[Engine, PreparationFactory],
        prep_start_times: Dict[Engine, int],
        prep_end_times: Dict[Engine, int],
        fal_arrival_times: Dict[Engine, int],
        delivery_time: Dict[Engine, int],
    ):
        super().__init__(problem)
        self.engine_to_msn = engine_to_msn
        self.engine_to_factory = engine_to_factory
        self.prep_start_times = prep_start_times
        self.prep_end_times = prep_end_times
        self.fal_arrival_times = fal_arrival_times

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
    - Assembly lines with scheduled MSN (aircraft)
    - Preparation factories with capacities and compatible engine types
    - Engine instances with delivery times and optional baseline allocations
    - Travel times/costs between factories and assembly lines

    Goal:
    - Allocate each engine to an MSN
    - Allocate each engine to a preparation factory
    - Schedule preparation in the factory
    - Minimize changes from baseline + delays at FAL

    Constraints:
    - Engine allocated to compatible factory only
    - Preparation starts after delivery time
    - Preparation duration matches factory's duration for engine type
    - Engine arrives at FAL before MSN's engine installation date
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
        """
        baseline_changes = 0
        arrival_delays = 0.0
        travel_cost = 0.0

        for engine in self.engines_list:
            # Count changes from baseline
            if engine.baseline_msn_id is not None:
                assigned_msn = solution.engine_to_msn.get(engine)
                if assigned_msn is None or assigned_msn.id != engine.baseline_msn_id:
                    baseline_changes += 1

            if engine.baseline_factory_id is not None:
                assigned_factory = solution.engine_to_factory.get(engine)
                if assigned_factory is None or assigned_factory.id != engine.baseline_factory_id:
                    baseline_changes += 1

            if engine.baseline_prep_start is not None:
                actual_start = solution.prep_start_times.get(engine, 0)
                if abs(actual_start - engine.baseline_prep_start) > 1:  # Allow 1 day tolerance
                    baseline_changes += 1

            # Compute arrival delays
            if engine in solution.engine_to_msn and engine in solution.fal_arrival_times:
                msn = solution.engine_to_msn[engine]
                arrival_time = solution.fal_arrival_times[engine]

                # Delay = max(0, arrival_time - installation_date)
                delay = max(0, arrival_time - msn.engine_installation_date)
                arrival_delays += delay

            # Compute travel costs
            if engine in solution.engine_to_factory and engine in solution.engine_to_msn:
                factory = solution.engine_to_factory[engine]
                msn = solution.engine_to_msn[engine]
                assembly_line = self.assembly_lines[msn.assembly_line_id]

                cost = self.get_travel_cost(factory, assembly_line)
                travel_cost += cost

        return {
            "baseline_changes": float(baseline_changes),
            "arrival_delays": arrival_delays,
            "travel_cost": travel_cost,
        }

    def satisfy(self, solution: EnginePreparationSolution) -> bool:
        """
        Check if solution satisfies all hard constraints

        Hard Constraints:
        1. Each engine allocated to exactly one MSN
        2. Each engine allocated to exactly one factory
        3. Factory must be compatible with engine type
        4. Preparation starts after delivery time
        5. Preparation duration matches factory's duration
        6. Arrival time correctly calculated (prep_end + travel_time)
        7. Factory capacity constraints (if specified)

        Soft Constraints (allowed, penalized in objective):
        - Arrival delays (arrival after MSN's installation date)
        """
        TOL = 1e-4

        # Check all engines are allocated
        for engine in self.engines_list:
            if engine not in solution.engine_to_msn:
                logger.warning(f"Engine {engine.id} not allocated to MSN")
                return False

            if engine not in solution.engine_to_factory:
                logger.warning(f"Engine {engine.id} not allocated to factory")
                return False

            if engine not in solution.prep_start_times:
                logger.warning(f"Engine {engine.id} has no prep start time")
                return False

        # Check each engine's allocation and scheduling
        for engine in self.engines_list:
            msn = solution.engine_to_msn.get(engine)
            factory = solution.engine_to_factory.get(engine)
            prep_start = solution.prep_start_times.get(engine, 0)
            prep_end = solution.prep_end_times.get(engine, 0)
            arrival_time = solution.fal_arrival_times.get(engine, 0)

            # Check MSN compatibility (engine type)
            if msn is None:
                logger.warning(f"Engine {engine.id} has no MSN assigned")
                return False

            if msn.engine_type != engine.engine_type:
                logger.warning(f"Engine {engine.id} type {engine.engine_type} incompatible with MSN {msn.id} type {msn.engine_type}")
                return False

            # Check factory compatibility
            if factory is None:
                logger.warning(f"Engine {engine.id} has no factory assigned")
                return False

            if not factory.can_prepare(engine.engine_type):
                logger.warning(f"Factory {factory.id} cannot prepare engine type {engine.engine_type}")
                return False

            # Check preparation starts after delivery
            if prep_start < engine.delivery_time - TOL:
                logger.warning(f"Engine {engine.id} prep starts at {prep_start} before delivery at {engine.delivery_time}")
                return False

            # Check preparation duration
            expected_duration = factory.get_preparation_duration(engine.engine_type)
            actual_duration = prep_end - prep_start
            if abs(actual_duration - expected_duration) > TOL:
                logger.warning(f"Engine {engine.id} prep duration {actual_duration} != expected {expected_duration}")
                return False

            # Check arrival time calculation
            assembly_line = self.assembly_lines[msn.assembly_line_id]
            travel_time = self.get_travel_time(factory, assembly_line)
            expected_arrival = prep_end + travel_time
            if abs(arrival_time - expected_arrival) > TOL:
                logger.warning(f"Engine {engine.id} arrival time {arrival_time} != expected {expected_arrival}")
                return False

            # NOTE: Arrival delays are ALLOWED (soft constraint, penalized in objective)
            # We log them for information but don't fail the feasibility check
            if arrival_time > msn.engine_installation_date + TOL:
                delay = arrival_time - msn.engine_installation_date
                logger.info(f"Engine {engine.id} arrives late by {delay:.1f} days (arrival={arrival_time}, deadline={msn.engine_installation_date})")

        # Check factory capacity constraints (cumulative)
        for factory in self.factories_list:
            if factory.capacity is not None:
                # Check at each time point
                for t in range(0, self.time_horizon + 1):
                    # Count engines in preparation at time t
                    engines_at_t = 0
                    for engine in self.engines_list:
                        if solution.engine_to_factory.get(engine) == factory:
                            prep_start = solution.prep_start_times[engine]
                            prep_end = solution.prep_end_times[engine]

                            # Engine is in preparation if prep_start <= t < prep_end
                            if prep_start <= t < prep_end:
                                engines_at_t += 1

                    if engines_at_t > factory.capacity:
                        logger.warning(f"Factory {factory.id} capacity {factory.capacity} exceeded at time {t}: {engines_at_t} engines")
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
            }
        )
