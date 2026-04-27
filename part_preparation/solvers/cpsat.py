"""
CP-SAT solver for Engine Preparation and Allocation using discrete-optimization API
"""
from typing import Any, Dict, List
import logging

from ortools.sat.python.cp_model import CpSolverSolutionCallback
from discrete_optimization.generic_tools.ortools_cpsat_tools import CpModel, OrtoolsCpSatSolver
from discrete_optimization.generic_tools.do_problem import Solution

from part_preparation.problem import (
    EnginePreparationProblem,
    EnginePreparationSolution,
    Engine,
    MSN,
    PreparationFactory,
    AssemblyLine,
)

logger = logging.getLogger(__name__)


class CPSatEngineSolver(OrtoolsCpSatSolver):
    """
    CP-SAT solver for Engine Preparation and Allocation

    Model:
    - Binary variables for engine-msn and engine-factory assignments
    - Interval variables for preparation tasks (for cumulative capacity)
    - Integer variables for arrival times
    - Constraints: compatibility, timing, capacity (cumulative)
    - Objective: weighted sum of baseline changes, delays, and travel costs
    """

    def __init__(self, problem: EnginePreparationProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)
        self.problem: EnginePreparationProblem = problem

        # Decision variables
        self.engine_to_msn_vars: Dict[tuple[Engine, MSN], Any] = {}
        self.engine_to_factory_vars: Dict[tuple[Engine, PreparationFactory], Any] = {}
        self.prep_start_vars: Dict[Engine, Any] = {}
        self.prep_end_vars: Dict[Engine, Any] = {}
        self.prep_intervals: Dict[tuple[Engine, PreparationFactory], Any] = {}  # Optional intervals
        self.arrival_vars: Dict[Engine, Any] = {}

        # Objective component variables
        self.baseline_changes_var = None
        self.arrival_delays_var = None
        self.travel_cost_var = None

    def init_model(self, **kwargs: Any) -> None:
        """
        Initialize the CP-SAT model

        Creates decision variables and constraints for engine allocation and scheduling.
        """
        self.cp_model = CpModel()

        # Build model in modular steps
        self._create_decision_variables()
        self._create_assignment_constraints()
        self._create_timing_constraints()
        self._create_capacity_constraints()
        self._create_takt_constraints()
        self._create_objective()

    def _create_decision_variables(self) -> None:
        """Create all decision variables"""

        # Binary assignment variables: engine -> MSN
        for engine in self.problem.engines_list:
            compatible_msns = self.problem.engine_compatible_msns.get(engine, [])
            for msn in compatible_msns:
                self.engine_to_msn_vars[(engine, msn)] = self.cp_model.new_bool_var(
                    f"assign_e{engine.id}_m{msn.id}"
                )

        # Binary assignment variables: engine -> factory
        for engine in self.problem.engines_list:
            compatible_factories = self.problem.engine_compatible_factories.get(engine, [])
            for factory in compatible_factories:
                self.engine_to_factory_vars[(engine, factory)] = self.cp_model.new_bool_var(
                    f"assign_e{engine.id}_f{factory.id}"
                )

        # Integer variables for timing
        for engine in self.problem.engines_list:
            # Preparation start time (after delivery)
            self.prep_start_vars[engine] = self.cp_model.new_int_var(
                engine.delivery_time,
                self.problem.time_horizon,
                f"prep_start_{engine.id}"
            )

            # Preparation end time
            self.prep_end_vars[engine] = self.cp_model.new_int_var(
                engine.delivery_time,
                self.problem.time_horizon,
                f"prep_end_{engine.id}"
            )

            # Arrival time at FAL
            self.arrival_vars[engine] = self.cp_model.new_int_var(
                engine.delivery_time,
                self.problem.time_horizon,
                f"arrival_{engine.id}"
            )

        # Optional interval variables for capacity constraints
        for engine in self.problem.engines_list:
            compatible_factories = self.problem.engine_compatible_factories.get(engine, [])
            for factory in compatible_factories:
                # Interval is present if engine assigned to this factory
                is_present = self.engine_to_factory_vars[(engine, factory)]

                # Duration depends on factory
                duration = factory.get_preparation_duration(engine.engine_type)

                # Create optional interval: [prep_start, prep_start + duration)
                interval = self.cp_model.new_optional_interval_var(
                    start=self.prep_start_vars[engine],
                    size=duration,
                    end=self.prep_end_vars[engine],
                    is_present=is_present,
                    name=f"interval_e{engine.id}_f{factory.id}"
                )

                self.prep_intervals[(engine, factory)] = interval

        # Optional takt interval variables for no-overlap constraints
        # These represent the first takt period of each engine's preparation
        self.takt_intervals: Dict[tuple[Engine, PreparationFactory], Any] = {}
        for engine in self.problem.engines_list:
            compatible_factories = self.problem.engine_compatible_factories.get(engine, [])
            for factory in compatible_factories:
                if factory.takt is not None:
                    # Interval is present if engine assigned to this factory
                    is_present = self.engine_to_factory_vars[(engine, factory)]

                    # Takt interval: [prep_start, prep_start + takt)
                    takt_interval = self.cp_model.new_optional_interval_var(
                        start=self.prep_start_vars[engine],
                        size=factory.takt,
                        end=self.prep_start_vars[engine] + factory.takt,
                        is_present=is_present,
                        name=f"takt_e{engine.id}_f{factory.id}"
                    )

                    self.takt_intervals[(engine, factory)] = takt_interval

    def _create_assignment_constraints(self) -> None:
        """Create assignment constraints: each engine to exactly one MSN and factory"""

        # Each engine assigned to exactly one MSN
        for engine in self.problem.engines_list:
            compatible_msns = self.problem.engine_compatible_msns.get(engine, [])
            assignment_vars = [
                self.engine_to_msn_vars[(engine, msn)]
                for msn in compatible_msns
            ]
            if assignment_vars:
                self.cp_model.add_exactly_one(assignment_vars)

        # Each engine assigned to exactly one factory
        for engine in self.problem.engines_list:
            compatible_factories = self.problem.engine_compatible_factories.get(engine, [])
            assignment_vars = [
                self.engine_to_factory_vars[(engine, factory)]
                for factory in compatible_factories
            ]
            if assignment_vars:
                self.cp_model.add_exactly_one(assignment_vars)

    def _create_timing_constraints(self) -> None:
        """Create timing constraints: prep duration and arrival time calculation"""

        for engine in self.problem.engines_list:
            compatible_factories = self.problem.engine_compatible_factories.get(engine, [])
            compatible_msns = self.problem.engine_compatible_msns.get(engine, [])

            # Preparation duration: prep_end = prep_start + duration (enforced by interval)
            # Already handled by the optional interval variables

            # Arrival time depends on factory and MSN
            for factory in compatible_factories:
                for msn in compatible_msns:
                    assembly_line = self.problem.assembly_lines[msn.assembly_line_id]
                    travel_time = self.problem.get_travel_time(factory, assembly_line)

                    is_factory_assigned = self.engine_to_factory_vars[(engine, factory)]
                    is_msn_assigned = self.engine_to_msn_vars[(engine, msn)]

                    # Both must be assigned
                    both_assigned = self.cp_model.new_bool_var(
                        f"both_{engine.id}_{factory.id}_{msn.id}"
                    )
                    self.cp_model.add_bool_and([is_factory_assigned, is_msn_assigned]).only_enforce_if(both_assigned)
                    self.cp_model.add_bool_or([is_factory_assigned.negated(), is_msn_assigned.negated()]).only_enforce_if(both_assigned.negated())

                    # If both assigned: arrival = prep_end + travel_time
                    self.cp_model.add(
                        self.arrival_vars[engine] == self.prep_end_vars[engine] + travel_time
                    ).only_enforce_if(both_assigned)

            # NOTE: We do NOT constrain arrival <= installation_date
            # Delays are ALLOWED but PENALIZED in the objective function
            # This makes it a soft constraint rather than a hard constraint

    def _create_capacity_constraints(self) -> None:
        """Create factory capacity constraints using cumulative constraints with intervals"""

        for factory in self.problem.factories_list:
            if factory.capacity is None:
                continue  # No capacity constraint for this factory

            # Collect all intervals for engines assigned to this factory
            intervals_for_factory: List[Any] = []
            demands_for_factory: List[int] = []

            for engine in self.problem.engines_list:
                if (engine, factory) in self.prep_intervals:
                    interval = self.prep_intervals[(engine, factory)]
                    intervals_for_factory.append(interval)
                    demands_for_factory.append(1)  # Each engine uses 1 unit of capacity

            if not intervals_for_factory:
                continue  # No engines can use this factory

            # Add cumulative constraint: at most 'capacity' engines in preparation simultaneously
            # This directly models the factory capacity as the maximum number of concurrent preparations
            self.cp_model.add_cumulative(
                intervals=intervals_for_factory,
                demands=demands_for_factory,
                capacity=factory.capacity
            )

    def _create_takt_constraints(self) -> None:
        """Create takt no-overlap constraints for well-paced scheduling"""

        for factory in self.problem.factories_list:
            if factory.takt is None:
                continue  # No takt constraint for this factory

            # Collect all takt intervals for engines assigned to this factory
            takt_intervals_for_factory: List[Any] = []

            for engine in self.problem.engines_list:
                if (engine, factory) in self.takt_intervals:
                    takt_interval = self.takt_intervals[(engine, factory)]
                    takt_intervals_for_factory.append(takt_interval)

            if not takt_intervals_for_factory:
                continue  # No engines can use this factory

            # Add no-overlap constraint: engines cannot overlap during their first takt period
            # This creates well-paced scheduling where engines start at staggered times
            self.cp_model.add_no_overlap(takt_intervals_for_factory)

    def _create_objective(self) -> None:
        """Create objective function: minimize baseline changes + delays + travel cost"""

        # Component 1: Baseline changes
        baseline_change_indicators = []
        for engine in self.problem.engines_list:
            # Change in MSN assignment
            if engine.baseline_msn_id is not None:
                baseline_msn = next((m for m in self.problem.msns_list if m.id == engine.baseline_msn_id), None)
                if baseline_msn and (engine, baseline_msn) in self.engine_to_msn_vars:
                    # Changed if NOT assigned to baseline MSN
                    changed = self.cp_model.new_bool_var(f"msn_changed_{engine.id}")
                    self.cp_model.add(self.engine_to_msn_vars[(engine, baseline_msn)] == 0).only_enforce_if(changed)
                    self.cp_model.add(self.engine_to_msn_vars[(engine, baseline_msn)] == 1).only_enforce_if(changed.negated())
                    baseline_change_indicators.append(changed)

            # Change in factory assignment
            if engine.baseline_factory_id is not None:
                baseline_factory = next((f for f in self.problem.factories_list if f.id == engine.baseline_factory_id), None)
                if baseline_factory and (engine, baseline_factory) in self.engine_to_factory_vars:
                    changed = self.cp_model.new_bool_var(f"factory_changed_{engine.id}")
                    self.cp_model.add(self.engine_to_factory_vars[(engine, baseline_factory)] == 0).only_enforce_if(changed)
                    self.cp_model.add(self.engine_to_factory_vars[(engine, baseline_factory)] == 1).only_enforce_if(changed.negated())
                    baseline_change_indicators.append(changed)

            # Change in prep start time (allow 1 day tolerance)
            if engine.baseline_prep_start is not None:
                changed = self.cp_model.new_bool_var(f"start_changed_{engine.id}")
                # Changed if |prep_start - baseline| > 1
                self.cp_model.add(
                    self.prep_start_vars[engine] < engine.baseline_prep_start - 1
                ).only_enforce_if(changed)
                self.cp_model.add(
                    self.prep_start_vars[engine] > engine.baseline_prep_start + 1
                ).only_enforce_if(changed)
                baseline_change_indicators.append(changed)

        self.baseline_changes_var = self.cp_model.new_int_var(
            0, 3 * len(self.problem.engines_list), "baseline_changes"
        )
        if baseline_change_indicators:
            self.cp_model.add(self.baseline_changes_var == sum(baseline_change_indicators))
        else:
            self.cp_model.add(self.baseline_changes_var == 0)

        # Component 2: Arrival delays
        # Delay = sum over all engines of max(0, arrival_time - installation_date)
        delay_vars = []
        for engine in self.problem.engines_list:
            # For each engine, compute its delay based on which MSN it's assigned to
            # delay = max(0, arrival_time - assigned_msn.installation_date)

            compatible_msns = self.problem.engine_compatible_msns.get(engine, [])

            # Create a delay variable for this engine
            engine_delay = self.cp_model.new_int_var(0, self.problem.time_horizon, f"engine_delay_{engine.id}")

            # For each possible MSN assignment
            for msn in compatible_msns:
                is_assigned = self.engine_to_msn_vars[(engine, msn)]

                # If assigned to this MSN: delay >= arrival - installation_date AND delay >= 0
                # This is: delay >= max(0, arrival - installation_date)
                lateness = self.arrival_vars[engine] - msn.engine_installation_date

                # If assigned: engine_delay >= lateness (bounded below by 0 from variable domain)
                self.cp_model.add(engine_delay >= lateness).only_enforce_if(is_assigned)

                # If assigned: also minimize it (objective will do this)
                # We don't need to force delay = max(0, lateness) exactly,
                # just delay >= max(0, lateness) is enough since we minimize

            # If not assigned to any compatible MSN: this shouldn't happen (exactly_one constraint)
            # but for safety: engine_delay can be 0

            delay_vars.append(engine_delay)

        self.arrival_delays_var = self.cp_model.new_int_var(
            0, len(self.problem.engines_list) * self.problem.time_horizon, "total_arrival_delays"
        )
        if delay_vars:
            self.cp_model.add(self.arrival_delays_var == sum(delay_vars))
        else:
            self.cp_model.add(self.arrival_delays_var == 0)

        # Component 3: Travel cost (scaled to integer)
        cost_scale = 1  # Scale costs by 1 (costs should be integer or we scale them)
        cost_vars = []
        for engine in self.problem.engines_list:
            compatible_factories = self.problem.engine_compatible_factories.get(engine, [])
            compatible_msns = self.problem.engine_compatible_msns.get(engine, [])

            for factory in compatible_factories:
                for msn in compatible_msns:
                    assembly_line = self.problem.assembly_lines[msn.assembly_line_id]
                    cost = int(self.problem.get_travel_cost(factory, assembly_line) * cost_scale)

                    if cost > 0:
                        # Cost applies if both are assigned
                        is_factory_assigned = self.engine_to_factory_vars[(engine, factory)]
                        is_msn_assigned = self.engine_to_msn_vars[(engine, msn)]

                        both_assigned = self.cp_model.new_bool_var(
                            f"cost_e{engine.id}_f{factory.id}_m{msn.id}"
                        )
                        self.cp_model.add_bool_and([is_factory_assigned, is_msn_assigned]).only_enforce_if(both_assigned)

                        cost_vars.append(cost * both_assigned)

        self.travel_cost_var = self.cp_model.new_int_var(0, len(self.problem.engines_list) * 10000 * cost_scale, "travel_cost")
        if cost_vars:
            self.cp_model.add(self.travel_cost_var == sum(cost_vars))
        else:
            self.cp_model.add(self.travel_cost_var == 0)

        # Weighted objective (use weights from params_objective_function)
        obj_dict = self.problem.get_objective_register().dict_objective_to_doc
        weight_baseline = obj_dict["baseline_changes"].default_weight
        weight_delays = obj_dict["arrival_delays"].default_weight
        weight_cost = obj_dict["travel_cost"].default_weight

        total_objective = (
            int(weight_baseline) * self.baseline_changes_var +
            int(weight_delays) * self.arrival_delays_var +
            int(weight_cost) * self.travel_cost_var
        )

        self.cp_model.minimize(total_objective)

    def retrieve_solution(self, cpsolvercb: CpSolverSolutionCallback) -> Solution:
        """
        Extract solution from CP-SAT solver callback

        Args:
            cpsolvercb: CP-SAT solver callback with solution values

        Returns:
            EnginePreparationSolution
        """
        engine_to_msn: Dict[Engine, MSN] = {}
        engine_to_factory: Dict[Engine, PreparationFactory] = {}
        prep_start_times: Dict[Engine, int] = {}
        prep_end_times: Dict[Engine, int] = {}
        fal_arrival_times: Dict[Engine, int] = {}

        # Extract assignments
        for (engine, msn), var in self.engine_to_msn_vars.items():
            if cpsolvercb.value(var) == 1:
                engine_to_msn[engine] = msn

        for (engine, factory), var in self.engine_to_factory_vars.items():
            if cpsolvercb.value(var) == 1:
                engine_to_factory[engine] = factory

        # Extract timing
        for engine in self.problem.engines_list:
            prep_start_times[engine] = cpsolvercb.value(self.prep_start_vars[engine])
            prep_end_times[engine] = cpsolvercb.value(self.prep_end_vars[engine])
            fal_arrival_times[engine] = cpsolvercb.value(self.arrival_vars[engine])

        solution = EnginePreparationSolution(
            problem=self.problem,
            engine_to_msn=engine_to_msn,
            engine_to_factory=engine_to_factory,
            prep_start_times=prep_start_times,
            prep_end_times=prep_end_times,
            fal_arrival_times=fal_arrival_times,
        )

        # Log objective values
        logger.info(f"Objective value: {cpsolvercb.objective_value}")
        logger.info(f"Baseline changes: {cpsolvercb.value(self.baseline_changes_var)}")
        logger.info(f"Arrival delays: {cpsolvercb.value(self.arrival_delays_var)}")
        logger.info(f"Travel cost: {cpsolvercb.value(self.travel_cost_var)}")

        return solution
