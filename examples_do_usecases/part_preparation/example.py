"""
Example usage of Engine Preparation and Allocation Problem
"""
from part_preparation.problem import (
    EnginePreparationProblem,
    AssemblyLine,
    MSN,
    PreparationFactory,
    Engine,
    TravelInfo,
)
from part_preparation.solvers.greedy import GreedyEngineSolver


def create_simple_example():
    """Create a simple example problem instance"""

    # Create assembly lines
    assembly_lines = [
        AssemblyLine(id="FAL_Hamburg", takt_pace=7, location="Hamburg"),
        AssemblyLine(id="FAL_Toulouse", takt_pace=5, location="Toulouse"),
    ]

    # Create MSNs (aircraft) scheduled on assembly lines
    msns = [
        MSN(
            id="MSN001",
            assembly_line_id="FAL_Hamburg",
            engine_type="TrentXWB",
            entry_time=10,
            end_time=50,
            engine_installation_date=30,
        ),
        MSN(
            id="MSN002",
            assembly_line_id="FAL_Hamburg",
            engine_type="TrentXWB",
            entry_time=20,
            end_time=60,
            engine_installation_date=40,
        ),
        MSN(
            id="MSN003",
            assembly_line_id="FAL_Toulouse",
            engine_type="GE90",
            entry_time=15,
            end_time=55,
            engine_installation_date=35,
        ),
    ]

    # Create preparation factories
    factories = [
        PreparationFactory(
            id="Factory_UK",
            location="Derby",
            compatible_engine_types={"TrentXWB", "Trent1000"},
            preparation_durations={"TrentXWB": 10, "Trent1000": 8},
            capacity=5,
        ),
        PreparationFactory(
            id="Factory_France",
            location="Toulouse",
            compatible_engine_types={"GE90", "CFM56"},
            preparation_durations={"GE90": 12, "CFM56": 9},
            capacity=3,
        ),
    ]

    # Create engines with delivery times and baseline allocations
    engines = [
        Engine(
            id="ENG001",
            engine_type="TrentXWB",
            delivery_time=5,
            baseline_msn_id="MSN001",  # Pre-allocated to MSN001
            baseline_factory_id="Factory_UK",
            baseline_prep_start=5,
        ),
        Engine(
            id="ENG002",
            engine_type="TrentXWB",
            delivery_time=10,
            baseline_msn_id="MSN002",
            baseline_factory_id="Factory_UK",
            baseline_prep_start=10,
        ),
        Engine(
            id="ENG003",
            engine_type="GE90",
            delivery_time=8,
            baseline_msn_id="MSN003",
            baseline_factory_id="Factory_France",
            baseline_prep_start=8,
        ),
    ]

    # Define travel times and costs
    travel_info = [
        TravelInfo(
            factory_id="Factory_UK",
            assembly_line_id="FAL_Hamburg",
            travel_time=2,  # 2 days
            travel_cost=500.0,  # €500 per engine
        ),
        TravelInfo(
            factory_id="Factory_UK",
            assembly_line_id="FAL_Toulouse",
            travel_time=3,
            travel_cost=800.0,
        ),
        TravelInfo(
            factory_id="Factory_France",
            assembly_line_id="FAL_Hamburg",
            travel_time=3,
            travel_cost=750.0,
        ),
        TravelInfo(
            factory_id="Factory_France",
            assembly_line_id="FAL_Toulouse",
            travel_time=1,  # Factory is at same location as FAL
            travel_cost=100.0,
        ),
    ]

    # Create problem
    problem = EnginePreparationProblem(
        assembly_lines=assembly_lines,
        msns=msns,
        factories=factories,
        engines=engines,
        travel_info=travel_info,
        time_horizon=100,
    )

    return problem
