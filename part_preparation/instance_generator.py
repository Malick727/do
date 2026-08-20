"""
Systematic instance generator for Engine Preparation and Allocation Problem

Creates realistic instances with controllable difficulty
"""
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
import logging
import random
import csv
from datetime import datetime
from part_preparation.problem import (
    EnginePreparationProblem,
    AssemblyLine,
    MSN,
    PreparationFactory,
    Engine,
    EngineType,
    EngineSupplier,
    TravelInfo,
    STORAGE_COST_RATE_PER_YEAR,
    daily_rate,
)


import numpy as np
logger = logging.getLogger(__name__)


@dataclass
class FactoryDefinition:
    """Manual factory definition for precise control"""
    engine_type: str
    home_fal_index: Optional[int]  # Which FAL is "close" (0-indexed), None = distant from all
    capacity: int
    preparation_duration: int
    location: str


@dataclass
class InstanceParams:
    """Parameters for instance generation"""
    # Problem size
    n_fals: int = 4
    n_msns_per_fal: int = 5
    n_engine_types: int = 2
    n_factories_per_type: int = 4

    # FAL production rates (can specify per FAL or use range)
    fal_takt_paces: Optional[List[int]] = None  # Specific takt for each FAL, or None to use range
    fal_takt_pace_range: Tuple[int, int] = (5, 8)  # days between MSNs if not specified

    # Engine/MSN ratio per type (can create imbalance)
    # engine_ratio > 1.0 = more engines than MSNs (storage problem)
    # engine_ratio < 1.0 = fewer engines than MSNs (shortage problem)
    engine_to_msn_ratio: Dict[str, float] = None  # e.g., {"EngineType_1": 1.2, "EngineType_2": 0.8}

    # Supplier effectiveness parameters
    supplier_regularity: Dict[str, float] = None  # 1.0 = perfect, 0.5 = noisy (variance in delivery times)
    supplier_production_rate: Dict[str, float] = None  # 1.0 = aligned with demand, 1.2 = 20% over-production

    # Timing parameters
    time_horizon: int = 100
    msn_production_duration_range: Tuple[int, int] = (30, 50)  # MSN stays in FAL
    engine_installation_offset_range: Tuple[int, int] = (15, 25)  # installation before end

    # Factory parameters
    preparation_duration_range: Tuple[int, int] = (10, 15)  # days to prepare engine
    factory_capacity_utilization: float = 0.95  # Target 95% capacity usage (creates bottleneck)

    # Manual factory definitions (if provided, overrides auto-generation)
    factory_definitions: Optional[List[FactoryDefinition]] = None

    # Travel parameters
    local_travel_time_range: Tuple[int, int] = (1, 3)  # factory → nearby FAL
    local_travel_cost_range: Tuple[int, int] = (100, 500)
    distant_travel_time_range: Tuple[int, int] = (5, 8)  # factory → far FAL
    distant_travel_cost_range: Tuple[int, int] = (1500, 3500)

    # Baseline disturbance (makes problem interesting)
    baseline_wrong_factory_prob: float = 0.3
    baseline_wrong_msn_prob: float = 0.2
    baseline_timing_disturbance_days: int = 5

    # Engine delivery flexibility
    flexible_delivery_prob: float = 0.2  # Probability that delivery time is decision variable

    # Random seed
    seed: int = 42


def generate_instance_from_csv(
    csv_file_path: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    allowed_engine_types: Optional[List[str]] = None
) -> EnginePreparationProblem:
    """
    Parses a Skywise PDP export CSV file to generate an EnginePreparationProblem instance.
    Includes capability to filter records by engine type and an installation date range.
    """
    raw_records = []
    
    # 1. Read, filter, and parse the CSV file
    with open(csv_file_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Focus on Engine Installation events
            if row.get('planning_event_name') == 'Engine_inst' or 'Engine Installation' in row.get('reference_point_name', ''):
                try:
                    inst_date = datetime.strptime(row['start_time'], '%Y-%m-%d %H:%M:%S')
                    engine_type = row['ac_engine'] # e.g., LEAP-1A33
                    
                    # ---- FILTER 1: Predefined Engine Types ----
                    if allowed_engine_types is not None and engine_type not in allowed_engine_types:
                        continue
                        
                    # ---- FILTER 2: Date Range Windows ----
                    if start_date is not None and inst_date < start_date:
                        continue
                    if end_date is not None and inst_date > end_date:
                        continue
                    
                    # Record passes all filters
                    raw_records.append({
                        'msn': row['msn'],
                        'engine_type': engine_type,
                        'fal_id': f"FAL_{row['fal']}" if row['fal'] else f"FAL_{row['responsible_site']}",
                        'fal_location': row['responsible_site'],
                        'calendar_date': inst_date
                    })
                except (ValueError, KeyError):
                    # Skip malformed rows or missing fields safely
                    continue

    if not raw_records:
        raise ValueError("No records matched the filter criteria or the CSV dataset is empty.")

    # 2. Establish baseline timeline offset (Relative day zero calculated from remaining records)
    min_date = min(rec['calendar_date'] for rec in raw_records)
    
    # 3. Identify unique infrastructure entities
    unique_fals = {}
    unique_engine_types = set()
    for rec in raw_records:
        unique_fals[rec['fal_id']] = rec['fal_location']
        unique_engine_types.add(rec['engine_type'])

    # 4. Create Domain Objects (FALs, Engine Types, and Mock Suppliers)
    assembly_lines = [
        AssemblyLine(id=fal_id, takt_pace=5, location=loc) 
        for fal_id, loc in unique_fals.items()
    ]

    engine_type_objects = {}
    suppliers = {}
    for etype in unique_engine_types:
        supplier = EngineSupplier(capacity_delivery_per_week=5)
        suppliers[etype] = supplier
        value_engine = 120000
        engine_type_objects[etype] = EngineType(
            id=etype,
            value_engine=value_engine,
            # Dérivé du prix du moteur au taux annuel de possession, au lieu d'une constante
            # arbitraire : des types de prix différents obtiennent automatiquement des coûts
            # de stockage cohérents entre eux.
            cost_storage_per_time=daily_rate(STORAGE_COST_RATE_PER_YEAR, value_engine),
            supplier={supplier},
        )

    # 5. Populate filtered MSNs and matching Engine allocations
    msns = []
    engines = []
    for idx, rec in enumerate(raw_records):
        relative_install_day = (rec['calendar_date'] - min_date).days
        
        msn_obj = MSN(
            id=f"MSN_{rec['msn']}",
            assembly_line_id=rec['fal_id'],
            engine_type=rec['engine_type'],
            entry_time=max(0, relative_install_day - 15),
            end_time=relative_install_day + 10,
            engine_installation_date=relative_install_day
        )
        msns.append(msn_obj)
    for idx, rec in enumerate(raw_records):
        
        relative_install_day = (rec['calendar_date'] - min_date).days
        engine_obj = Engine(
            id=f"ENG_{rec['engine_type']}_{idx:03d}",
            engine_type=rec['engine_type'],
            engine_type_object=engine_type_objects[rec['engine_type']],
            engine_supplier=suppliers[rec['engine_type']],
            delivery_time=max(0, relative_install_day - 20),
            baseline_msn_id=msn_obj.id,
            baseline_factory_id=f"Factory_{rec['engine_type']}_Default",
            baseline_prep_start=max(0, relative_install_day - 19),
            baseline_prep_end=max(0, relative_install_day - 5)
        )
        engines.append(engine_obj)

    # 6. Generate fallback co-located preparation hub structures 
    factories = []
    travel_info = []
    for etype in unique_engine_types:
        factory = PreparationFactory(
            id=f"Factory_{etype}_Default",
            location="Co-located Hub",
            default_assembly_line=assembly_lines[0],
            compatible_engine_types={etype},
            preparation_durations={etype: 14},
            default_capacity=10,
            calendar=None,
            takt=None
        )
        factories.append(factory)
        for fal in assembly_lines:
            travel_info.append(TravelInfo(
                factory_id=factory.id, assembly_line_id=fal.id, travel_time=2, travel_cost=500
            ))

    max_day = max(m.end_time for m in msns)

    return EnginePreparationProblem(
        assembly_lines=assembly_lines,
        msns=msns,
        factories=factories,
        engines=engines,
        travel_info=travel_info,
        time_horizon=max_day + 10
    )

def generate_instance(params: InstanceParams) -> EnginePreparationProblem:
    """
    Generate a realistic instance with capacity bottlenecks

    Strategy:
    1. Create FALs with regular MSN schedules
    2. Determine engine type distribution across MSNs
    3. Calculate total demand per engine type
    4. Create factories (manual or auto-generated)
    5. Generate engines with baseline allocations
    """
    random.seed(params.seed)

    # =====================================================
    # Step 1: Create Final Assembly Lines
    # =====================================================
    fal_locations = ["Hamburg", "Toulouse", "Mobile", "Tianjin", "Seattle", "Montreal", "Tokyo", "Beijing"][:params.n_fals]

    assembly_lines = []
    for i, location in enumerate(fal_locations):
        # Use specific takt if provided, otherwise random from range
        if params.fal_takt_paces is not None and i < len(params.fal_takt_paces):
            takt_pace = params.fal_takt_paces[i]
        else:
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

    # Calculate total production capacity (inverse of takt pace)
    # Faster FALs (low takt) get more MSNs
    total_production_capacity = sum(1.0 / fal.takt_pace for fal in assembly_lines)

    msns = []
    msn_id = 1

    for fal in assembly_lines:
        # Distribute MSNs proportionally to production rate
        fal_capacity_ratio = (1.0 / fal.takt_pace) / total_production_capacity
        n_msns_this_fal = int(params.n_msns_per_fal * len(assembly_lines) * fal_capacity_ratio)

        # Ensure at least some MSNs
        n_msns_this_fal = max(5, n_msns_this_fal)

        # MSNs enter FAL at regular takt intervals (no holes)
        for msn_idx in range(n_msns_this_fal):
            # Alternate or distribute engine types realistically
            engine_type = engine_types[msn_idx % params.n_engine_types]

            # MSN timing: entries are spaced by takt_pace
            # Add offset to avoid clustering at time 0
            planning_start_offset = 10  # Start MSNs at day 10
            entry_time = planning_start_offset + (msn_idx * fal.takt_pace)

            # Production duration is fixed for this FAL (determined by takt)
            # End time = when next MSN enters
            if msn_idx < n_msns_this_fal - 1:
                end_time = planning_start_offset + ((msn_idx + 1) * fal.takt_pace)
            else:
                # Last MSN: use average production duration
                production_duration = random.randint(*params.msn_production_duration_range)
                end_time = entry_time + production_duration

            # Engine installation happens before MSN ends
            installation_offset = random.randint(*params.engine_installation_offset_range)
            engine_installation_date = end_time - installation_offset
            # Ensure installation date is after entry_time and not negative
            engine_installation_date = max(entry_time, engine_installation_date)
            engine_installation_date = max(0, engine_installation_date)
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
    # Step 3.5: Setup ratios and supplier parameters
    # =====================================================
    # Default ratios if not specified
    if params.engine_to_msn_ratio is None:
        engine_to_msn_ratio = {etype: 1.0 for etype in engine_types}
    else:
        engine_to_msn_ratio = params.engine_to_msn_ratio

    # Default supplier parameters if not specified
    if params.supplier_regularity is None:
        supplier_regularity = {etype: 1.0 for etype in engine_types}  # Perfect regularity
    else:
        supplier_regularity = params.supplier_regularity

    if params.supplier_production_rate is None:
        supplier_production_rate = {etype: 1.0 for etype in engine_types}  # Aligned with demand
    else:
        supplier_production_rate = params.supplier_production_rate

    # =====================================================
    # Step 4: Create Engine Types and Suppliers
    # =====================================================
    engine_type_objects = {}
    suppliers = {}

    for etype in engine_types:
        # Calculate number of engines for this type based on ratio
        n_engines_for_type = int(demand_per_type[etype] * engine_to_msn_ratio[etype])

        # Supplier capacity based on production rate
        # Production rate > 1.0 means supplier delivers faster than demand
        weeks_in_horizon = params.time_horizon / 7
        engines_per_week = (n_engines_for_type / weeks_in_horizon) * supplier_production_rate[etype]
        supplier_capacity = max(1, int(engines_per_week * 1.5))  # Add buffer

        # Create supplier for this engine type
        supplier = EngineSupplier(
            capacity_delivery_per_week=supplier_capacity
        )
        suppliers[etype] = supplier

        # Create engine type
        value_engine = 100000  # Base value (arbitrary)
        engine_type_obj = EngineType(
            id=etype,
            value_engine=value_engine,
            cost_storage_per_time=daily_rate(STORAGE_COST_RATE_PER_YEAR, value_engine),
            supplier={supplier}
        )
        engine_type_objects[etype] = engine_type_obj

        print(f"Engine type {etype}: {n_engines_for_type} engines (ratio={engine_to_msn_ratio[etype]:.2f}), "
              f"supplier capacity={supplier_capacity}/week (rate={supplier_production_rate[etype]:.2f}), "
              f"regularity={supplier_regularity[etype]:.2f}")

    # =====================================================
    # Step 5: Create factories (manual or auto-generated)
    # =====================================================
    factories = []
    factory_to_home_fal = {}

    if params.factory_definitions is not None:
        # Manual factory creation
        print("\nUsing manual factory definitions:")
        for i, factory_def in enumerate(params.factory_definitions):
            # Determine home FAL
            if factory_def.home_fal_index is not None:
                default_fal = assembly_lines[factory_def.home_fal_index]
                factory_to_home_fal[f"Factory_{factory_def.engine_type}_{factory_def.location}"] = default_fal.id
            else:
                # No home FAL - pick first one as default (but travel will be expensive to all)
                default_fal = assembly_lines[0]
                factory_to_home_fal[f"Factory_{factory_def.engine_type}_{factory_def.location}"] = None

            factory = PreparationFactory(
                id=f"Factory_{factory_def.engine_type}_{factory_def.location}",
                location=factory_def.location,
                default_assembly_line=default_fal,
                compatible_engine_types={factory_def.engine_type},
                preparation_durations={factory_def.engine_type: factory_def.preparation_duration},
                default_capacity=factory_def.capacity,
                calendar=None,
                takt=None
            )

            factories.append(factory)

            home_fal_str = default_fal.id if factory_def.home_fal_index is not None else "None (distant)"
            print(f"  {factory.id}: home_FAL={home_fal_str}, "
                  f"capacity={factory_def.capacity}, prep_dur={factory_def.preparation_duration}")

    else:
        # Auto-generate factories (old behavior)
        factory_base_names = ["Derby", "Toulouse", "Virginia", "Singapore", "Durham", "Bordeaux", "Cincinnati", "Shanghai"]
        factory_id = 0

        for etype in engine_types:
            msns_of_type = [m for m in msns if m.engine_type == etype]

            # Create one factory per FAL for this engine type
            for i, default_fal in enumerate(assembly_lines):
                # Calculate demand from this FAL for this engine type
                msns_at_fal = [m for m in msns_of_type if m.assembly_line_id == default_fal.id]

                if len(msns_at_fal) == 0:
                    # No demand from this FAL, skip or create minimal factory
                    continue

                # Calculate production rate needed
                # Engines arrive at rate of FAL takt pace / n_engine_types
                engines_per_day = 1.0 / (default_fal.takt_pace * params.n_engine_types)

                # Preparation duration
                prep_duration = random.randint(*params.preparation_duration_range)

                # Required capacity: engines_per_day * prep_duration
                # This is the theoretical capacity to keep up with FAL
                theoretical_capacity = engines_per_day * prep_duration

                # Add intentional mismatch to create bottlenecks
                # Some factories slightly under-capacity (bottleneck)
                # Some factories slightly over-capacity (buffer)
                capacity_adjustment = random.choice([0.85, 0.9, 0.95, 1.0, 1.05, 1.1])
                capacity = theoretical_capacity * capacity_adjustment / params.factory_capacity_utilization

                # Round to integer, minimum 2
                capacity = max(2, round(capacity))

                # Assign location (geographically distributed)
                location = factory_base_names[factory_id % len(factory_base_names)]

                factory = PreparationFactory(
                    id=f"Factory_{etype}_{location}",
                    location=location,
                    default_assembly_line=default_fal,
                    compatible_engine_types={etype},
                    preparation_durations={etype: prep_duration},
                    default_capacity=capacity,
                    calendar=None,
                    takt=None
                )

                factories.append(factory)
                factory_to_home_fal[factory.id] = default_fal.id

                print(f"Factory {factory.id}: FAL={default_fal.id} (takt={default_fal.takt_pace}), "
                      f"{len(msns_at_fal)} MSNs, prep_dur={prep_duration}, "
                      f"theoretical_cap={theoretical_capacity:.2f}, capacity={capacity}")

                factory_id += 1

    # =====================================================
    # Step 6: Create travel info (geographic proximity)
    # =====================================================
    travel_info = []

    for factory in factories:
        home_fal_id = factory_to_home_fal.get(factory.id)

        for fal in assembly_lines:
            if home_fal_id is not None and fal.id == home_fal_id:
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
    # Step 7: Generate engines with baseline allocations
    # =====================================================
    engines = []
    engine_id_counter = 0

    for etype in engine_types:
        msns_of_type = [m for m in msns if m.engine_type == etype]
        # Sort MSNs by installation date to create a demand timeline
        msns_of_type_sorted = sorted(msns_of_type, key=lambda m: m.engine_installation_date)

        type_factories = [f for f in factories if etype in f.compatible_engine_types]

        # Calculate number of engines to create based on ratio
        n_engines_for_type = int(demand_per_type[etype] * engine_to_msn_ratio[etype])

        # Get supplier parameters
        regularity = supplier_regularity[etype]
        production_rate = supplier_production_rate[etype]

        # Create a demand timeline: when are engines actually needed?
        # This will be used to correlate deliveries with actual demand
        if msns_of_type_sorted:
            installation_dates = [m.engine_installation_date for m in msns_of_type_sorted]
            min_install = min(installation_dates)
            max_install = max(installation_dates)
            # Average prep duration for this type
            avg_prep_duration = sum(f.get_preparation_duration(etype) for f in type_factories) / len(type_factories)
        else:
            min_install = 0
            max_install = params.time_horizon
            avg_prep_duration = sum(params.preparation_duration_range) / 2

        for i in range(n_engines_for_type):
            # Assign to MSN if available (may have more engines than MSNs)
            if i < len(msns_of_type_sorted):
                msn = msns_of_type_sorted[i]
            else:
                # No MSN available for this engine (surplus)
                msn = None
            # Baseline factory selection
            if msn is not None:
                home_fal_id = msn.assembly_line_id

                # Find factories close to this FAL
                close_factories = [f for f in type_factories
                                 if factory_to_home_fal.get(f.id) == home_fal_id]

                if random.random() < params.baseline_wrong_factory_prob or not close_factories:
                    baseline_factory = random.choice(type_factories)
                else:
                    baseline_factory = random.choice(close_factories)

                # Baseline MSN (may be wrong)
                if random.random() < params.baseline_wrong_msn_prob and i < len(msns_of_type_sorted) - 1:
                    baseline_msn = msns_of_type_sorted[i + 1]
                else:
                    baseline_msn = msn
            else:
                # Unallocated engine: assign to random factory
                baseline_factory = random.choice(type_factories)
                baseline_msn = None

            # Calculate baseline timing
            prep_duration = baseline_factory.get_preparation_duration(etype)

            if baseline_msn is not None:
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
            else:
                # Unallocated engine: no specific timing
                baseline_start = None

            # ===================================================================
            # IMPROVED DELIVERY DATE GENERATION
            # Correlate delivery with actual engine need (installation date)
            # ===================================================================
            if msn is not None:
                # Engine is needed for this MSN
                # Ideal delivery: installation_date - avg_travel - avg_prep - buffer
                installation_date = msn.engine_installation_date
                avg_travel = 3  # Average travel time
                buffer_before_prep = 3  # Small buffer before prep starts

                # Target delivery: correlated with when engine is needed
                target_delivery = installation_date - avg_travel - avg_prep_duration - buffer_before_prep

                # Apply supplier production rate
                # production_rate > 1.0: supplier delivers ahead of schedule (engines arrive early)
                # production_rate < 1.0: supplier delivers behind schedule (engines arrive late)
                if production_rate != 1.0:
                    # Shift delivery proportional to how far we are in the timeline
                    time_to_need = max(1, installation_date)
                    if production_rate > 1.0:
                        # Over-production: deliver earlier
                        # Higher production rate = larger early shift
                        delivery_shift = int((production_rate - 1.0) * time_to_need * 0.3)
                        target_delivery = target_delivery - delivery_shift
                    else:
                        # Under-production: deliver later (may cause delays)
                        # Lower production rate = larger late shift
                        delivery_shift = int((1.0 - production_rate) * time_to_need * 0.5)
                        target_delivery = target_delivery + delivery_shift

                # Apply supplier regularity (variance in delivery)
                if regularity < 1.0:
                    # Lower regularity = more noise
                    noise_amplitude = int((1.0 - regularity) * 15)  # Up to ±15 days for regularity=0
                    delivery_noise = random.randint(-noise_amplitude, noise_amplitude)
                    target_delivery = target_delivery + delivery_noise

                # Clamp to valid range
                target_delivery = max(0, min(params.time_horizon - 10, target_delivery))

                # Sometimes leave as decision variable
                if random.random() < params.flexible_delivery_prob:
                    delivery_time = None
                else:
                    delivery_time = target_delivery

            else:
                # Unallocated engine (surplus)
                # Spread across timeline but correlated with demand pattern
                if msns_of_type_sorted:
                    # Follow the demand rhythm but shifted/spread
                    # Map engine index to a position in the demand timeline
                    surplus_index = i - len(msns_of_type_sorted)
                    total_surplus = n_engines_for_type - len(msns_of_type_sorted)

                    # Position in timeline (0 to 1)
                    if total_surplus > 0:
                        position = surplus_index / total_surplus
                    else:
                        position = 0.5

                    # Interpolate in the demand timeline
                    target_delivery = int(min_install + position * (max_install - min_install))

                    # Apply production rate shift
                    if production_rate > 1.0:
                        shift = int((production_rate - 1.0) * target_delivery * 0.3)
                        target_delivery = target_delivery - shift
                    elif production_rate < 1.0:
                        shift = int((1.0 - production_rate) * target_delivery * 0.5)
                        target_delivery = target_delivery + shift

                    # Apply regularity noise
                    if regularity < 1.0:
                        noise = int((1.0 - regularity) * 15)
                        target_delivery = target_delivery + random.randint(-noise, noise)
                else:
                    # No MSNs at all, spread evenly
                    target_delivery = int(params.time_horizon * (i / n_engines_for_type))

                target_delivery = max(0, min(params.time_horizon - 10, target_delivery))

                if random.random() < params.flexible_delivery_prob:
                    delivery_time = None
                else:
                    delivery_time = target_delivery

            engines.append(Engine(
                id=f"ENG_{etype}_{engine_id_counter:03d}",
                engine_type=etype,
                engine_type_object=engine_type_objects[etype],
                engine_supplier=suppliers[etype],
                delivery_time=delivery_time,
                baseline_msn_id=baseline_msn.id if baseline_msn else None,
                baseline_factory_id=baseline_factory.id,
                baseline_prep_start=baseline_start,
                baseline_prep_end=baseline_start + prep_duration if baseline_start is not None else None
            ))

            engine_id_counter += 1

    # =====================================================
    # Step 8: Create problem instance
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
    for fal in assembly_lines:
        msns_at_fal = [m for m in msns if m.assembly_line_id == fal.id]
        print(f"  {fal.id:<15} takt={fal.takt_pace} days, {len(msns_at_fal)} MSNs")
    print(f"\nMSNs:               {len(msns)}")
    print(f"Engine types:       {len(engine_types)}")
    for etype in engine_types:
        n_engines = len([e for e in engines if e.engine_type == etype])
        n_msns = len([m for m in msns if m.engine_type == etype])
        print(f"  {etype}: {n_engines} engines for {n_msns} MSNs (ratio={n_engines/n_msns:.2f})")
    print(f"\nFactories:          {len(factories)}")
    print(f"Engines:            {len(engines)}")
    print(f"Time horizon:       {params.time_horizon} days")
    if params.factory_definitions is None:
        print(f"Capacity target:    {params.factory_capacity_utilization*100:.0f}% utilization")
    print("\nFactory-FAL proximity:")
    for factory in factories[:min(8, len(factories))]:
        home_fal = factory_to_home_fal.get(factory.id)
        home_str = home_fal if home_fal else "None (distant from all)"
        print(f"  {factory.id:<30} → {home_str}")

    # Print delivery-installation correlation statistics
    print("\n📅 Delivery vs Installation Date Correlation:")
    for etype in engine_types:
        engines_of_type = [e for e in engines if e.engine_type == etype]
        msns_of_type = [m for m in msns if m.engine_type == etype]

        # For allocated engines, compute correlation metrics
        allocated_engines = [e for e in engines_of_type if e.baseline_msn_id is not None]
        if allocated_engines and msns_of_type:
            # Compute delivery-to-installation gaps
            gaps = []
            for engine in allocated_engines:
                if engine.delivery_time is not None:
                    # Find the MSN this engine is baseline-allocated to
                    matching_msn = next((m for m in msns_of_type if m.id == engine.baseline_msn_id), None)
                    if matching_msn:
                        gap = matching_msn.engine_installation_date - engine.delivery_time
                        gaps.append(gap)

            if gaps:
                avg_gap = sum(gaps) / len(gaps)
                min_gap = min(gaps)
                max_gap = max(gaps)
                std_gap = (sum((g - avg_gap)**2 for g in gaps) / len(gaps))**0.5

                # Count early vs late deliveries
                n_early = sum(1 for g in gaps if g > 20)  # Delivered more than 20 days early
                n_late = sum(1 for g in gaps if g < 10)   # Delivered less than 10 days before need
                n_optimal = len(gaps) - n_early - n_late

                print(f"\n  {etype}:")
                print(f"    Avg gap (installation - delivery): {avg_gap:.1f} days")
                print(f"    Range: [{min_gap}, {max_gap}] days")
                print(f"    Std dev: {std_gap:.1f} days")
                print(f"    Distribution: {n_early} early (>20d), {n_optimal} optimal (10-20d), {n_late} tight (<10d)")

    print("=" * 80)

    return problem


def generate_training_dataset(num_problems=50, solutions_per_problem=100):
    dataset = []

    for p in range(num_problems):
        # 1. Générer une nouvelle instance
        problem = generate_instance(MEDIUM_INSTANCE)

        # 2. Générer N solutions aléatoires ou heuristiques
        solutions = generate_random_solutions(problem, solutions_per_problem)

        # 3. Calculer la vraie Fitness (l'opération lourde qu'on veut remplacer plus tard)
        for sol in solutions:
            sol.true_cost = calculate_exact_cost(sol, problem)

        # 4. Créer des paires pour le réseau siamois
        for i in range(len(solutions)):
            # On compare chaque solution avec 5 autres (par exemple)
            for j in range(i + 1, min(i + 6, len(solutions))):
                sol_A = solutions[i]
                sol_B = solutions[j]

                # Extraction des variables (les tableaux de chiffres)
                features_A = extract_features(sol_A, problem)
                features_B = extract_features(sol_B, problem)

                # Label : 1.0 si A est meilleure (coût plus faible), sinon 0.0
                label = 1.0 if sol_A.true_cost < sol_B.true_cost else 0.0

                dataset.append({
                    "features_A": features_A,
                    "features_B": features_B,
                    "label": label
                })

    return dataset



# Predefined instance configurations
SMALL_INSTANCE = InstanceParams(
    n_fals=2,
    n_msns_per_fal=3,
    n_engine_types=2,
    n_factories_per_type=2,
    time_horizon=60,
    factory_capacity_utilization=0.9,
    engine_to_msn_ratio={"EngineType_1": 1.0, "EngineType_2": 1.0},
    supplier_regularity={"EngineType_1": 1.0, "EngineType_2": 1.0},
    supplier_production_rate={"EngineType_1": 1.0, "EngineType_2": 1.0},
)

MEDIUM_INSTANCE = InstanceParams(
    n_fals=4,
    n_msns_per_fal=5,
    n_engine_types=2,
    n_factories_per_type=4,
    time_horizon=100,
    factory_capacity_utilization=0.95,
    engine_to_msn_ratio={"EngineType_1": 1.0, "EngineType_2": 1.0},
    supplier_regularity={"EngineType_1": 1.0, "EngineType_2": 1.0},
    supplier_production_rate={"EngineType_1": 1.0, "EngineType_2": 1.0},
)

LARGE_INSTANCE = InstanceParams(
    # 4 FALs: 2 fast (takt=2 days), 2 slow (takt=7 days)
    # MSN distribution is proportional to production rate:
    #   Fast FALs (takt=2): ~194 MSNs each (0.5 MSN/day)
    #   Slow FALs (takt=7): ~55 MSNs each (0.14 MSN/day)
    n_fals=4,
    fal_takt_paces=[2, 2, 7, 7],  # Fast, Fast, Slow, Slow
    n_msns_per_fal=125,  # Base number (will be adjusted proportionally)
    n_engine_types=2,
    n_factories_per_type=4,  # 8 factories total (1 per engine type per FAL)
    time_horizon=365,  # 1 year
    factory_capacity_utilization=0.95,
    # Supplier 1: Very productive and regular (slight over-production)
    # Supplier 2: Subject to noise and shortage (under-production)
    engine_to_msn_ratio={
        "EngineType_1": 1.1,  # 10% surplus → storage problem
        "EngineType_2": 0.85  # 15% shortage → unallocated MSNs
    },
    supplier_regularity={
        "EngineType_1": 0.95,  # Very regular deliveries
        "EngineType_2": 0.6    # Noisy deliveries (high variance)
    },
    supplier_production_rate={
        "EngineType_1": 1.15,  # 15% over-production (engines arrive early/frequently)
        "EngineType_2": 0.9    # 10% under-production (engines arrive late/rarely)
    },
    flexible_delivery_prob=0.3,  # 30% of engines have flexible delivery
)

STRESS_TEST = InstanceParams(
    n_fals=8,
    n_msns_per_fal=10,
    n_engine_types=4,
    n_factories_per_type=6,
    time_horizon=200,
    factory_capacity_utilization=0.99,  # Very tight capacity
    engine_to_msn_ratio={f"EngineType_{i+1}": 1.0 for i in range(4)},
    supplier_regularity={f"EngineType_{i+1}": 0.8 for i in range(4)},
    supplier_production_rate={f"EngineType_{i+1}": 1.0 for i in range(4)},
)