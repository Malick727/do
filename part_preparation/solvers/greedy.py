"""
Greedy heuristic solver for Engine Preparation and Allocation
"""
from typing import Dict
import logging

from discrete_optimization.generic_tools.do_solver import SolverDO, ResultStorage

from part_preparation.problem import (
    EnginePreparationProblem,
    EnginePreparationSolution,
    Engine,
    MSN,
    PreparationFactory,
)

logger = logging.getLogger(__name__)


class GreedyEngineSolver(SolverDO):
    """
    Greedy heuristic for Engine Preparation and Allocation

    Strategy:
    1. Sort engines by delivery time (earliest first)
    2. For each engine:
       - Try to assign to baseline MSN if available, else first compatible MSN
       - Try to assign to baseline factory if available, else closest compatible factory
       - Schedule preparation ASAP after delivery
       - Compute arrival time at FAL
    """

    def __init__(self, problem: EnginePreparationProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.problem = problem

    def init_model(self, **kwargs) -> None:
        """Initialize (no model to build for greedy)"""
        pass

    def solve(self, **kwargs) -> ResultStorage:
        """
        Execute greedy algorithm

        Returns:
            ResultStorage with the constructed solution
        """
        # Sort engines by delivery time (earliest first)
        sorted_engines = sorted(self.problem.engines_list, key=lambda e: e.delivery_time)

        # Track MSN assignments (to avoid duplicates)
        msn_assigned_engines: Dict[MSN, list] = {msn: [] for msn in self.problem.msns_list}

        # Decision variables
        engine_to_msn: Dict[Engine, MSN] = {}
        engine_to_factory: Dict[Engine, PreparationFactory] = {}
        prep_start_times: Dict[Engine, int] = {}
        prep_end_times: Dict[Engine, int] = {}
        fal_arrival_times: Dict[Engine, int] = {}

        for engine in sorted_engines:
            # Select MSN
            msn = self._select_msn_for_engine(engine, msn_assigned_engines)
            if msn is None:
                logger.warning(f"No compatible MSN found for engine {engine.id}")
                continue

            engine_to_msn[engine] = msn
            msn_assigned_engines[msn].append(engine)

            # Select factory
            factory = self._select_factory_for_engine(engine, msn)
            if factory is None:
                logger.warning(f"No compatible factory found for engine {engine.id}")
                continue

            engine_to_factory[engine] = factory

            # Schedule preparation ASAP (right after delivery)
            prep_start = engine.delivery_time
            prep_duration = factory.get_preparation_duration(engine.engine_type)
            prep_end = prep_start + prep_duration

            prep_start_times[engine] = prep_start
            prep_end_times[engine] = prep_end

            # Compute arrival at FAL
            assembly_line = self.problem.assembly_lines[msn.assembly_line_id]
            travel_time = self.problem.get_travel_time(factory, assembly_line)
            arrival_time = prep_end + travel_time

            fal_arrival_times[engine] = arrival_time

        # Build solution
        solution = EnginePreparationSolution(
            problem=self.problem,
            engine_to_msn=engine_to_msn,
            engine_to_factory=engine_to_factory,
            prep_start_times=prep_start_times,
            prep_end_times=prep_end_times,
            fal_arrival_times=fal_arrival_times,
        )

        # Evaluate solution
        fitness = self.aggreg_from_sol(solution)

        return ResultStorage(
            list_solution_fits=[(solution, fitness)],
            mode_optim=self.params_objective_function.sense_function,
        )

    def _select_msn_for_engine(
        self, engine: Engine, msn_assigned_engines: Dict[MSN, list]
    ) -> MSN:
        """
        Select MSN for an engine

        Priority:
        1. Baseline MSN if available and compatible
        2. First compatible MSN with no engines assigned yet
        3. Any compatible MSN
        """
        compatible_msns = self.problem.engine_compatible_msns.get(engine, [])

        # Try baseline first
        if engine.baseline_msn_id is not None:
            for msn in compatible_msns:
                if msn.id == engine.baseline_msn_id:
                    return msn

        # Try unassigned MSNs first
        for msn in compatible_msns:
            if len(msn_assigned_engines[msn]) == 0:
                return msn

        # Any compatible MSN (prioritize those needing engine soonest)
        if compatible_msns:
            return min(compatible_msns, key=lambda m: m.engine_installation_date)

        return None

    def _select_factory_for_engine(self, engine: Engine, msn: MSN) -> PreparationFactory:
        """
        Select factory for an engine

        Priority:
        1. Baseline factory if compatible
        2. Factory that minimizes travel cost to MSN's assembly line
        """
        compatible_factories = self.problem.engine_compatible_factories.get(engine, [])

        # Try baseline first
        if engine.baseline_factory_id is not None:
            for factory in compatible_factories:
                if factory.id == engine.baseline_factory_id:
                    return factory

        # Select factory with minimum travel cost
        if compatible_factories:
            assembly_line = self.problem.assembly_lines[msn.assembly_line_id]
            return min(
                compatible_factories,
                key=lambda f: self.problem.get_travel_cost(f, assembly_line),
            )

        return None