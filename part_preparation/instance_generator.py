"""
Systematic instance generator for Engine Preparation and Allocation Problem

Creates realistic instances with controllable difficulty
"""
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
import random

from part_preparation.problem import (
    EnginePreparationProblem,
    AssemblyLine,
    MSN,
    PreparationFactory,
    Engine,
    TravelInfo,
)


@dataclass
class InstanceParams:
    """Parameters for instance generation"""
    # Problem size
    n_fals: int = 4
    n_msns_per_fal: int = 5
    n_engine_types: int = 2
    n_factories_per_type: int = 4

    # Timing parameters
    time_horizon: int = 100
    fal_takt_pace_range: Tuple[int, int] = (5, 8)  # days between MSNs
    msn_production_duration_range: Tuple[int, int] = (30, 50)  # MSN stays in FAL
    engine_installation_offset_range: Tuple[int, int] = (15, 25)  # installation before end

    # Factory parameters
    preparation_duration_range: Tuple[int, int] = (10, 15)  # days to prepare engine
    factory_capacity_utilization: float = 0.95  # Target 95% capacity usage (creates bottleneck)

    # Travel parameters
    local_travel_time_range: Tuple[int, int] = (1, 3)  # factory → nearby FAL
    local_travel_cost_range: Tuple[int, int] = (100, 500)
    distant_travel_time_range: Tuple[int, int] = (5, 8)  # factory → far FAL
    distant_travel_cost_range: Tuple[int, int] = (1500, 3500)

    # Baseline disturbance (makes problem interesting)
    baseline_wrong_factory_prob: float = 0.3
    baseline_wrong_msn_prob: float = 0.2
    baseline_timing_disturbance_days: int = 5

    # Random seed
    seed: int = 42


def generate_instance(params: InstanceParams) -> EnginePreparationProblem:
    """
    Generate a realistic instance with capacity bottlenecks

    Strategy:
    1. Create FALs with regular MSN schedules
    2. Determine engine type distribution across MSNs
    3. Calculate total demand per engine type
    4. Size factory capacities to be just sufficient (creates bottleneck)
    5. Assign factories geographically to FALs
    6. Generate engines with baseline allocations
    """
    random.seed(params.seed)

    # =====================================================
    # Step 1: Create Final Assembly Lines
    # =====================================================
    fal_locations = ["Hamburg", "Toulouse", "Mobile", "Tianjin", "Seattle", "Montreal", "Tokyo", "Beijing"][:params.n_fals]

    assembly_lines = []
    for i, location in enumerate(fal_locations):
        takt_pace = random.randint(*params.fal_takt_pace_range)
        assembly_lines.append(AssemblyLine(
            id=f"FAL_{location}",
            takt_pace=takt_pace,
            location=location
        ))

    # =====================================================
    # Step 2: Create MSN schedule for each FAL
    # =====================================================
    engine_types = [f"EngineType_{i+1}" for i in range(params.n_engine_types)]

    msns = []
    msn_id = 1
    current_time = 0

    for fal in assembly_lines:
        # MSNs enter FAL at regular takt intervals (no holes)
        # Each MSN i enters at time i * takt_pace
        for msn_idx in range(params.n_msns_per_fal):
            # Alternate or distribute engine types realistically
            engine_type = engine_types[msn_idx % params.n_engine_types]

            # MSN timing: entries are spaced by takt_pace
            entry_time = msn_idx * fal.takt_pace

            # Production duration is fixed for this FAL (determined by takt)
            # End time = when next MSN enters
            if msn_idx < params.n_msns_per_fal - 1:
                end_time = (msn_idx + 1) * fal.takt_pace
            else:
                # Last MSN: use average production duration
                production_duration = random.randint(*params.msn_production_duration_range)
                end_time = entry_time + production_duration

            # Engine installation happens before MSN ends
            installation_offset = random.randint(*params.engine_installation_offset_range)
            engine_installation_date = end_time - installation_offset

            msns.append(MSN(
                id=f"MSN{msn_id:04d}",
                assembly_line_id=fal.id,
                engine_type=engine_type,
                entry_time=entry_time,
                end_time=end_time,
                engine_installation_date=engine_installation_date
            ))

            msn_id += 1

    # =====================================================
    # Step 3: Calculate demand per engine type
    # =====================================================
    demand_per_type = {etype: 0 for etype in engine_types}
    for msn in msns:
        demand_per_type[msn.engine_type] += 1

    print(f"Demand per engine type: {demand_per_type}")

    # =====================================================
    # Step 4: Create factories with capacity bottleneck
    # =====================================================
    factory_base_names = ["Derby", "Toulouse", "Virginia", "Singapore", "Durham", "Bordeaux", "Cincinnati", "Shanghai"]

    factories = []
    factory_id = 0

    for etype in engine_types:
        # Calculate total preparation time needed for this engine type
        msns_of_type = [m for m in msns if m.engine_type == etype]
        avg_prep_duration = sum(params.preparation_duration_range) / 2

        # Find the critical time window (when most engines need preparation)
        # This is more realistic than spreading over entire time horizon
        installation_times = [m.engine_installation_date for m in msns_of_type]
        earliest = min(installation_times) if installation_times else 0
        latest = max(installation_times) if installation_times else params.time_horizon

        # Engines need to be prepared within this window
        critical_window = max(1, latest - earliest + avg_prep_duration)

        # Calculate required throughput (engines per day)
        # If we need N engines prepared in W days, throughput = N/W
        required_throughput = len(msns_of_type) / critical_window  # engines per day

        # With capacity C and prep_duration D:
        # - C engines can be in preparation simultaneously
        # - Each takes D days
        # - Throughput = C/D engines per day
        # So: C = required_throughput * D

        # Total capacity needed across all factories
        total_capacity_needed = required_throughput * avg_prep_duration

        # Add slack based on utilization target (e.g., 0.95 → divide by 0.95)
        total_capacity = total_capacity_needed / params.factory_capacity_utilization

        # Distribute capacity across factories
        # Scale up to get reasonable integer capacities (min 3-5 concurrent engines)
        min_capacity_per_factory = 3
        capacity_per_factory = max(min_capacity_per_factory, total_capacity / params.n_factories_per_type)

        print(f"Engine type {etype}: {len(msns_of_type)} engines, "
              f"critical window: {critical_window:.0f} days, "
              f"capacity per factory: {capacity_per_factory:.2f} concurrent engines")

        for i in range(params.n_factories_per_type):
            prep_duration = random.randint(*params.preparation_duration_range)

            # Assign location (geographically distributed)
            location = factory_base_names[factory_id % len(factory_base_names)]

            # Round capacity to integer
            # Add some variation: some factories slightly over/under capacity
            variation = random.choice([-1, 0, 0, 1])  # ±1 engine variation
            capacity = max(2, round(capacity_per_factory) + variation)

            # Calculate takt: divider of prep_duration for well-paced scheduling
            # Choose a divisor that creates reasonable staggering (e.g., prep_duration / 5)
            # This means engines start every takt days
            possible_takts = [d for d in [1, 2, 3, 5] if prep_duration % d == 0]
            if possible_takts:
                takt = max(possible_takts)  # Use largest divisor for smoother pacing
            else:
                takt = 1  # Fallback

            factories.append(PreparationFactory(
                id=f"Factory_{etype}_{location}",
                location=location,
                compatible_engine_types={etype},
                preparation_durations={etype: prep_duration},
                capacity=capacity,  # Maximum concurrent engines in preparation
                takt=takt  # Takt time for well-paced scheduling
            ))

            factory_id += 1

    # =====================================================
    # Step 5: Create travel info (geographic proximity)
    # =====================================================
    # Assign each factory to a "home" FAL (closest geographically)
    factory_to_home_fal = {}
    factories_per_etype = {etype: [f for f in factories if etype in f.compatible_engine_types]
                          for etype in engine_types}

    for etype, type_factories in factories_per_etype.items():
        # Distribute factories across FALs
        for i, factory in enumerate(type_factories):
            home_fal = assembly_lines[i % len(assembly_lines)]
            factory_to_home_fal[factory.id] = home_fal.id

    travel_info = []

    for factory in factories:
        home_fal_id = factory_to_home_fal[factory.id]

        for fal in assembly_lines:
            if fal.id == home_fal_id:
                # Local travel (close)
                travel_time = random.randint(*params.local_travel_time_range)
                travel_cost = random.randint(*params.local_travel_cost_range)
            else:
                # Distant travel (expensive)
                travel_time = random.randint(*params.distant_travel_time_range)
                travel_cost = random.randint(*params.distant_travel_cost_range)

            travel_info.append(TravelInfo(
                factory_id=factory.id,
                assembly_line_id=fal.id,
                travel_time=travel_time,
                travel_cost=travel_cost
            ))

    # =====================================================
    # Step 6: Generate engines with baseline allocations
    # =====================================================
    engines = []
    engine_id_counter = 0

    for etype in engine_types:
        msns_of_type = [m for m in msns if m.engine_type == etype]
        type_factories = [f for f in factories if etype in f.compatible_engine_types]

        for i, msn in enumerate(msns_of_type):
            # Choose baseline factory
            home_fal_id = msn.assembly_line_id

            # Find factories close to this FAL
            close_factories = [f for f in type_factories
                             if factory_to_home_fal[f.id] == home_fal_id]

            if random.random() < params.baseline_wrong_factory_prob or not close_factories:
                # Assign to random factory (may be far)
                baseline_factory = random.choice(type_factories)
            else:
                # Assign to close factory
                baseline_factory = random.choice(close_factories)

            # Baseline MSN (may be wrong)
            if random.random() < params.baseline_wrong_msn_prob and i < len(msns_of_type) - 1:
                baseline_msn = msns_of_type[i + 1]
            else:
                baseline_msn = msn

            # Calculate baseline timing
            prep_duration = baseline_factory.get_preparation_duration(etype)
            fal = next(f for f in assembly_lines if f.id == baseline_msn.assembly_line_id)
            travel_time = next(t.travel_time for t in travel_info
                             if t.factory_id == baseline_factory.id and t.assembly_line_id == fal.id)

            # Work backwards from installation date
            ideal_start = baseline_msn.engine_installation_date - travel_time - prep_duration

            # Add timing disturbance
            if random.random() < 0.4:
                baseline_start = ideal_start + random.randint(
                    -params.baseline_timing_disturbance_days,
                    params.baseline_timing_disturbance_days
                )
            else:
                baseline_start = ideal_start

            baseline_start = max(0, baseline_start)

            # Engine delivery time (a few days before prep start)
            delivery_time = max(0, baseline_start - random.randint(2, 5))

            engines.append(Engine(
                id=f"ENG_{etype}_{engine_id_counter:03d}",
                engine_type=etype,
                delivery_time=delivery_time,
                baseline_msn_id=baseline_msn.id,
                baseline_factory_id=baseline_factory.id,
                baseline_prep_start=baseline_start,
                baseline_prep_end=baseline_start + prep_duration
            ))

            engine_id_counter += 1

    # =====================================================
    # Step 7: Create problem instance
    # =====================================================
    problem = EnginePreparationProblem(
        assembly_lines=assembly_lines,
        msns=msns,
        factories=factories,
        engines=engines,
        travel_info=travel_info,
        time_horizon=params.time_horizon
    )

    # Print summary
    print("\n" + "=" * 80)
    print("INSTANCE GENERATED")
    print("=" * 80)
    print(f"FALs:               {len(assembly_lines)}")
    print(f"MSNs:               {len(msns)}")
    print(f"Engine types:       {len(engine_types)}")
    print(f"Factories:          {len(factories)}")
    print(f"Engines:            {len(engines)}")
    print(f"Time horizon:       {params.time_horizon} days")
    print(f"Capacity target:    {params.factory_capacity_utilization*100:.0f}% utilization")
    print("\nFactory-FAL proximity:")
    for factory in factories[:min(4, len(factories))]:
        home_fal = factory_to_home_fal[factory.id]
        print(f"  {factory.id:<30} → {home_fal}")
    print("=" * 80)

    return problem


# Predefined instance configurations
SMALL_INSTANCE = InstanceParams(
    n_fals=2,
    n_msns_per_fal=3,
    n_engine_types=2,
    n_factories_per_type=2,
    time_horizon=60,
    factory_capacity_utilization=0.9,
)

MEDIUM_INSTANCE = InstanceParams(
    n_fals=4,
    n_msns_per_fal=5,
    n_engine_types=2,
    n_factories_per_type=4,
    time_horizon=100,
    factory_capacity_utilization=0.95,
)

LARGE_INSTANCE = InstanceParams(
    n_fals=6,
    n_msns_per_fal=8,
    n_engine_types=3,
    n_factories_per_type=5,
    time_horizon=150,
    factory_capacity_utilization=0.98,
)

STRESS_TEST = InstanceParams(
    n_fals=8,
    n_msns_per_fal=10,
    n_engine_types=4,
    n_factories_per_type=6,
    time_horizon=200,
    factory_capacity_utilization=0.99,  # Very tight capacity
)
