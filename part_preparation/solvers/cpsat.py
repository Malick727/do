"""
CP-SAT solver for Engine Preparation and Allocation using discrete-optimization API
"""
from typing import Any, Dict, List, Optional
import logging

from ortools.sat.python.cp_model import CpSolverSolutionCallback
from discrete_optimization.generic_tools.ortools_cpsat_tools import CpModel, OrtoolsCpSatSolver
from discrete_optimization.generic_tools.do_problem import Solution, ParamsObjectiveFunction

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
    - Binary variables for engine-msn (optional) and engine-factory assignments
    - Integer variables for delivery times (when not fixed)
    - Interval variables for preparation tasks (for cumulative capacity)
    - Integer variables for arrival times
    - Constraints: compatibility, timing, capacity (cumulative), supplier delivery capacity
    - Objective: weighted sum of baseline changes, delays, travel costs, storage costs, unallocated MSNs
    """

    def __init__(
        self,
        problem: EnginePreparationProblem,
        params_objective_function: ParamsObjectiveFunction = None,
        **kwargs: Any
    ):
        super().__init__(problem, params_objective_function, **kwargs)
        self.problem: EnginePreparationProblem = problem
        # Decision variables
        self.engine_allocated_vars: Dict[Engine, Any] = {}  # Binary: is engine allocated to an MSN?
        self.engine_to_msn_vars: Dict[tuple[Engine, MSN], Any] = {}
        self.engine_to_factory_vars: Dict[tuple[Engine, PreparationFactory], Any] = {}
        self.delivery_time_vars: Dict[Engine, Any] = {}  # For engines with flexible delivery
        self.prep_start_vars: Dict[Engine, Any] = {}
        self.prep_end_vars: Dict[Engine, Any] = {}
        self.prep_intervals: Dict[tuple[Engine, PreparationFactory], Any] = {}  # Optional intervals
        self.arrival_vars: Dict[Engine, Any] = {}

        # Objective component variables
        self.baseline_changes_var = None
        self.arrival_delays_var = None
        self.travel_cost_var = None
        self.storage_cost_var = None
        self.unallocated_msns_var = None

    def init_model(self, **kwargs: Any) -> None:
        """
        Initialize the CP-SAT model
        Creates decision variables and constraints for engine allocation and scheduling.
        """
        self.cp_model = CpModel()

        # Build model in modular steps
        self._create_decision_variables()
        self._create_assignment_constraints()
        self._create_delivery_constraints()
        self._create_timing_constraints()
        self._create_capacity_constraints()
        self._create_takt_constraints()
        self._create_objective()

    def _create_decision_variables(self) -> None:
        """Create all decision variables"""

        # Binary variables: is engine allocated to an MSN?
        for engine in self.problem.engines_list:
            self.engine_allocated_vars[engine] = (
                self.cp_model.new_bool_var(
                f"allocated_{engine.id}"
            ))

        # Binary assignment variables: engine -> MSN (only if allocated)
        for engine in self.problem.engines_list:
            compatible_msns = self.problem.engine_compatible_msns.get(engine, [])
            for msn in compatible_msns:
                self.engine_to_msn_vars[(engine, msn)] = self.cp_model.new_bool_var(
                    f"assign_e{engine.id}_m{msn.id}"
                )

        # Binary assignment variables: engine -> factory (only if allocated)
        for engine in self.problem.engines_list:
            compatible_factories = self.problem.engine_compatible_factories.get(engine, [])
            for factory in compatible_factories:
                self.engine_to_factory_vars[(engine, factory)] = self.cp_model.new_bool_var(
                    f"assign_e{engine.id}_f{factory.id}"
                )

        # Delivery time variables (for engines with flexible delivery)
        for engine in self.problem.engines_list:
            if engine.delivery_time is None:
                # Decision variable: when is the engine delivered?
                self.delivery_time_vars[engine] = self.cp_model.new_int_var(
                    0,
                    self.problem.time_horizon,
                    f"delivery_{engine.id}"
                )
            else:
                # Fixed delivery time
                self.delivery_time_vars[engine] = int(engine.delivery_time)

        # Integer variables for timing
        for engine in self.problem.engines_list:
            # Preparation start time (after delivery)
            self.prep_start_vars[engine] = self.cp_model.new_int_var(
                0,
                self.problem.time_horizon,
                f"prep_start_{engine.id}"
            )

            # Preparation end time
            self.prep_end_vars[engine] = self.cp_model.new_int_var(
                0,
                self.problem.time_horizon,
                f"prep_end_{engine.id}"
            )

            # Arrival time at FAL (only meaningful if allocated)
            self.arrival_vars[engine] = self.cp_model.new_int_var(
                0,
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

        # ====================================================================
        # both_assigned, indexé par (moteur, usine, FAL) et NON par (moteur, usine, MSN)
        #
        # Les trois endroits qui consomment ces booléens -- coût de transport, date d'arrivée,
        # stockage après préparation -- ne lisent du MSN que `msn.assembly_line_id`, puis
        # appellent get_travel_cost / get_travel_time sur le couple (usine, FAL). Aucun ne
        # dépend d'une propriété propre au MSN. Or il y a 998 MSN pour 4 FAL sur xlarge.
        #
        # Indexer par MSN créait donc 2 042 216 booléens portant l'information de 16 368, avec
        # deux contraintes de réification chacun : 11.8 M de contraintes, 134 s de construction
        # pour une limite de résolution de 20 s, et un modèle qui ne tenait pas en mémoire.
        #
        # Le passage par is_fal_assigned coûte 4 092 booléens supplémentaires (moteurs x FAL) et
        # une disjonction chacun -- négligeable devant ce qu'il évite.
        # ====================================================================
        self.is_fal_assigned_vars: Dict[tuple[Engine, AssemblyLine], Any] = {}
        for engine in self.problem.engines_list:
            compatible_msns = self.problem.engine_compatible_msns.get(engine, [])
            msns_by_fal: Dict[str, List[MSN]] = {}
            for msn in compatible_msns:
                msns_by_fal.setdefault(msn.assembly_line_id, []).append(msn)

            for fal_id, msns in msns_by_fal.items():
                assembly_line = self.problem.assembly_lines[fal_id]
                is_fal_assigned = self.cp_model.new_bool_var(f"fal_{engine.id}_{fal_id}")
                msn_literals = [self.engine_to_msn_vars[(engine, msn)] for msn in msns]

                # is_fal_assigned <=> au moins un MSN de cette FAL est affecté a ce moteur.
                # L'unicite (un moteur -> au plus un MSN) est garantie ailleurs, donc ce OU
                # designe bien la FAL du MSN retenu.
                self.cp_model.add_bool_or(msn_literals).only_enforce_if(is_fal_assigned)
                for literal in msn_literals:
                    self.cp_model.add_implication(literal, is_fal_assigned)
                self.is_fal_assigned_vars[(engine, assembly_line)] = is_fal_assigned

        self.both_assigned_vars: Dict[tuple[Engine, PreparationFactory, AssemblyLine], Any] = {}
        for engine in self.problem.engines_list:
            compatible_factories = self.problem.engine_compatible_factories.get(engine, [])

            for factory in compatible_factories:
                is_factory_assigned = self.engine_to_factory_vars[(engine, factory)]

                for assembly_line in self.problem.assembly_lines_list:
                    is_fal_assigned = self.is_fal_assigned_vars.get((engine, assembly_line))
                    if is_fal_assigned is None:
                        continue    # aucun MSN compatible sur cette FAL

                    both_assigned = self.cp_model.new_bool_var(
                        f"both_{engine.id}_{factory.id}_{assembly_line.id}")
                    self.cp_model.add_bool_and(
                        [is_factory_assigned, is_fal_assigned]).only_enforce_if(both_assigned)
                    self.cp_model.add_bool_or(
                        [is_factory_assigned.negated(), is_fal_assigned.negated()]
                    ).only_enforce_if(both_assigned.negated())
                    self.both_assigned_vars[(engine, factory, assembly_line)] = both_assigned

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
        """Create assignment constraints: engines can be allocated or not"""

        for engine in self.problem.engines_list:
            compatible_msns = self.problem.engine_compatible_msns.get(engine, [])
            compatible_factories = self.problem.engine_compatible_factories.get(engine, [])

            is_allocated = self.engine_allocated_vars[engine]

            # If allocated: assigned to exactly one MSN
            if compatible_msns:
                msn_assignment_vars = [
                    self.engine_to_msn_vars[(engine, msn)]
                    for msn in compatible_msns
                ]
                # Allocated ==> exactly one MSN assigned
                self.cp_model.add_at_most_one(msn_assignment_vars)
                self.cp_model.add(sum(msn_assignment_vars) == 1).only_enforce_if(is_allocated)
                # Not allocated ==> no MSN assigned
                self.cp_model.add(sum(msn_assignment_vars) == 0).only_enforce_if(is_allocated.negated())
            else:
                # No compatible MSNs => cannot be allocated
                self.cp_model.add(is_allocated == 0)

            # If allocated: assigned to exactly one factory
            if compatible_factories:
                factory_assignment_vars = [
                    self.engine_to_factory_vars[(engine, factory)]
                    for factory in compatible_factories
                ]
                # Allocated ==> exactly one factory assigned
                self.cp_model.add_at_most_one(factory_assignment_vars)
                self.cp_model.add(sum(factory_assignment_vars) == 1).only_enforce_if(is_allocated)
                # Not allocated ==> no factory assigned
                self.cp_model.add(sum(factory_assignment_vars) == 0).only_enforce_if(is_allocated.negated())
            else:
                # No compatible factories => cannot be allocated
                self.cp_model.add(is_allocated == 0)

        # Each MSN can receive at most one engine
        for msn in self.problem.msns_list:
            engine_vars = [
                self.engine_to_msn_vars[(engine, msn)]
                for engine in self.problem.engines_list
                if (engine, msn) in self.engine_to_msn_vars
            ]
            if engine_vars:
                self.cp_model.add_at_most_one(engine_vars)

    def _create_delivery_constraints(self) -> None:
        """Create supplier delivery capacity constraints per calendar week"""

        # Only add constraint if at least one engine has flexible delivery time
        has_flexible_delivery = any(e.delivery_time is None for e in self.problem.engines_list)
        if not has_flexible_delivery:
            return  # All deliveries are fixed, constraint would be redundant (and possibly infeasible)

        # NOTE: For now, we skip this constraint as it makes the problem infeasible
        # The instance generator creates fixed deliveries that may violate supplier capacity
        # TODO: Either relax supplier capacity or fix instance generator
        return

        # Group engines by supplier
        # Constraint: at most capacity_delivery_per_week engines delivered per CALENDAR week
        # Week 0 = [0, 7), Week 1 = [7, 14), etc.

        suppliers = set(e.engine_supplier for e in self.problem.engines_list)

        for supplier in suppliers:
            engines_for_supplier = [
                e for e in self.problem.engines_list
                if e.engine_supplier == supplier
            ]

            if not engines_for_supplier:
                continue

            # For each calendar week, count deliveries
            max_weeks = (self.problem.time_horizon // 7) + 1

            for week in range(max_weeks):
                week_start = week * 7
                week_end = (week + 1) * 7

                # Collect engines delivered in this calendar week
                engines_in_week = []
                for engine in engines_for_supplier:
                    delivery_var = self.delivery_time_vars[engine]

                    # Create boolean: is engine delivered in this calendar week?
                    in_week = self.cp_model.new_bool_var(f"delivery_e{engine.id}_week{week}")

                    if isinstance(delivery_var, int):
                        # Fixed delivery time
                        if week_start <= delivery_var < week_end:
                            self.cp_model.add(in_week == 1)
                        else:
                            self.cp_model.add(in_week == 0)
                    else:
                        # Variable delivery time: in_week <==> week_start <= delivery_var < week_end
                        # Channel the boolean: in_week == (delivery_var >= week_start AND delivery_var < week_end)

                        # Create intermediate booleans for the conditions
                        at_or_after_start = self.cp_model.new_bool_var(f"delivery_e{engine.id}_>=w{week}")
                        before_end = self.cp_model.new_bool_var(f"delivery_e{engine.id}_<w{week}end")

                        # Channel: at_or_after_start <==> (delivery_var >= week_start)
                        self.cp_model.add(delivery_var >= week_start).only_enforce_if(at_or_after_start)
                        self.cp_model.add(delivery_var < week_start).only_enforce_if(at_or_after_start.negated())

                        # Channel: before_end <==> (delivery_var < week_end)
                        self.cp_model.add(delivery_var < week_end).only_enforce_if(before_end)
                        self.cp_model.add(delivery_var >= week_end).only_enforce_if(before_end.negated())

                        # in_week <==> (at_or_after_start AND before_end)
                        self.cp_model.add_bool_and([at_or_after_start, before_end]).only_enforce_if(in_week)
                        self.cp_model.add_bool_or([at_or_after_start.negated(), before_end.negated()]).only_enforce_if(in_week.negated())

                    engines_in_week.append(in_week)

                # Capacity constraint: at most capacity_delivery_per_week deliveries in this week
                if engines_in_week:
                    self.cp_model.add(
                        sum(engines_in_week) <= supplier.capacity_delivery_per_week
                    )

    def _create_timing_constraints(self) -> None:
        """Create timing constraints: prep duration and arrival time calculation"""

        for engine in self.problem.engines_list:
            compatible_factories = self.problem.engine_compatible_factories.get(engine, [])
            compatible_msns = self.problem.engine_compatible_msns.get(engine, [])

            is_allocated = self.engine_allocated_vars[engine]
            delivery_var = self.delivery_time_vars[engine]

            # If allocated: prep_start >= delivery_time
            if isinstance(delivery_var, int):
                self.cp_model.add(
                    self.prep_start_vars[engine] >= delivery_var
                ).only_enforce_if(is_allocated)
            else:
                self.cp_model.add(
                    self.prep_start_vars[engine] >= delivery_var
                ).only_enforce_if(is_allocated)

            # Preparation duration: prep_end = prep_start + duration (enforced by interval)
            # Already handled by the optional interval variables

            # Arrival time depends on the factory and the MSN's FAL -- not on the MSN itself,
            # which is why both_assigned is indexed by FAL. One constraint per (factory, FAL)
            # instead of one per (factory, MSN).
            for factory in compatible_factories:
                for assembly_line in self.problem.assembly_lines_list:
                    both_assigned = self.both_assigned_vars.get((engine, factory, assembly_line))
                    if both_assigned is None:
                        continue
                    travel_time = self.problem.get_travel_time(factory, assembly_line)

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
            # Use default_capacity if calendar not specified
            if factory.calendar is None and factory.default_capacity is None:
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

            # For now, we use default_capacity (time-varying capacity would require more complex modeling)
            # TODO: Handle time-varying capacity from calendar
            if factory.default_capacity is not None:
                # Add cumulative constraint: at most 'default_capacity' engines in preparation simultaneously
                # This directly models the factory capacity as the maximum number of concurrent preparations
                self.cp_model.add_cumulative(
                    intervals=intervals_for_factory,
                    demands=demands_for_factory,
                    capacity=factory.default_capacity
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
        """
        Create objective function based on params_objective_function

        Only create objective components that are specified in params_objective_function.
        If params_objective_function is None, use all objectives with default weights.
        """

        # Determine which objectives to include and their weights
        if self.params_objective_function is not None:
            objectives_to_include = set(self.params_objective_function.objectives)
            objective_weights = dict(zip(
                self.params_objective_function.objectives,
                self.params_objective_function.weights
            ))
        else:
            # Use all objectives with default weights from objective register
            obj_register = self.problem.get_objective_register()
            objectives_to_include = set(obj_register.dict_objective_to_doc.keys())
            objective_weights = {
                name: doc.default_weight
                for name, doc in obj_register.dict_objective_to_doc.items()
            }

        # Un terme de poids nul ne peut déplacer aucun optimum : construire ses variables et ses
        # contraintes est du travail pur perdu, payé deux fois -- à la construction du modèle,
        # puis en présolve, qui doit les éliminer.
        #
        # Ce n'est pas une micro-optimisation. storage_cost pèse 0 dans l'objectif canonique
        # (cf. problem.AGGREGATED_WEIGHTS) et son bloc boucle sur les couples (usine, MSN) de
        # CHAQUE moteur : il représente à lui seul 43% des contraintes du modèle sur `medium`
        # (297k -> 169k), et de l'ordre de 5 M contraintes sur `xlarge`, où la seule
        # construction du modèle prend déjà plus de deux minutes.
        #
        # Conséquence assumée : la variable correspondante reste None, donc l'objectif ne figure
        # plus dans get_lexico_objectives_available(). C'est cohérent -- un terme de poids nul
        # n'est pas un critère -- et l'API lexico gère déjà le cas None. Pour optimiser un
        # objectif en lexico, il faut donc lui donner un poids non nul.
        weighted_out = {
            name for name in objectives_to_include if int(objective_weights[name]) == 0
        }
        for name in sorted(weighted_out):
            # Les poids sont utilisés via int() dans le terme d'objectif ci-dessous, donc un
            # poids de 0.5 est DÉJÀ ignoré silencieusement aujourd'hui. On le dit, au lieu de
            # laisser croire que le terme est pris en compte.
            if objective_weights[name] != 0:
                logger.warning(
                    "Objectif %r de poids %s tronqué à 0 par int() : il est ignoré. "
                    "Utiliser un poids entier non nul, ou remettre à l'échelle les autres.",
                    name, objective_weights[name],
                )
        if weighted_out:
            logger.info("Objectifs de poids nul, non construits : %s", sorted(weighted_out))
        objectives_to_include = objectives_to_include - weighted_out

        # Component 1: Baseline changes (if requested)
        if "baseline_changes" in objectives_to_include:
            baseline_change_indicators = []
            for engine in self.problem.engines_list:
                is_allocated = self.engine_allocated_vars[engine]

                # Change in MSN assignment
                if engine.baseline_msn_id is not None:
                    baseline_msn = next((m for m in self.problem.msns_list if m.id == engine.baseline_msn_id), None)
                    if baseline_msn and (engine, baseline_msn) in self.engine_to_msn_vars:
                        # Changed if allocated AND NOT assigned to baseline MSN
                        changed = self.cp_model.new_bool_var(f"msn_changed_{engine.id}")
                        not_baseline = self.cp_model.new_bool_var(f"not_baseline_msn_{engine.id}")
                        self.cp_model.add(self.engine_to_msn_vars[(engine, baseline_msn)] == 0).only_enforce_if(not_baseline)
                        self.cp_model.add(self.engine_to_msn_vars[(engine, baseline_msn)] == 1).only_enforce_if(not_baseline.negated())
                        self.cp_model.add_bool_and([is_allocated, not_baseline]).only_enforce_if(changed)
                        baseline_change_indicators.append(changed)

                # Change in factory assignment
                if engine.baseline_factory_id is not None:
                    baseline_factory = next((f for f in self.problem.factories_list if f.id == engine.baseline_factory_id), None)
                    if baseline_factory and (engine, baseline_factory) in self.engine_to_factory_vars:
                        changed = self.cp_model.new_bool_var(f"factory_changed_{engine.id}")
                        not_baseline = self.cp_model.new_bool_var(f"not_baseline_factory_{engine.id}")
                        self.cp_model.add(self.engine_to_factory_vars[(engine, baseline_factory)] == 0).only_enforce_if(not_baseline)
                        self.cp_model.add(self.engine_to_factory_vars[(engine, baseline_factory)] == 1).only_enforce_if(not_baseline.negated())
                        self.cp_model.add_bool_and([is_allocated, not_baseline]).only_enforce_if(changed)
                        baseline_change_indicators.append(changed)

                # Change in prep start time (allow 1 day tolerance)
                if engine.baseline_prep_start is not None:
                    changed = self.cp_model.new_bool_var(f"start_changed_{engine.id}")
                    early = self.cp_model.new_bool_var(f"start_early_{engine.id}")
                    late = self.cp_model.new_bool_var(f"start_late_{engine.id}")
                    self.cp_model.add(self.prep_start_vars[engine] < engine.baseline_prep_start - 1).only_enforce_if(early)
                    self.cp_model.add(self.prep_start_vars[engine] > engine.baseline_prep_start + 1).only_enforce_if(late)
                    self.cp_model.add_bool_or([early, late]).only_enforce_if(changed)
                    self.cp_model.add_bool_and([is_allocated, changed])
                    baseline_change_indicators.append(changed)

            self.baseline_changes_var = self.cp_model.new_int_var(
                0, 3 * len(self.problem.engines_list), "baseline_changes"
            )
            if baseline_change_indicators:
                self.cp_model.add(self.baseline_changes_var == sum(baseline_change_indicators))
            else:
                self.cp_model.add(self.baseline_changes_var == 0)
        else:
            self.baseline_changes_var = None

        # Component 2: Arrival delays (if requested)
        if "arrival_delays" in objectives_to_include:
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
        else:
            self.arrival_delays_var = None

        # Component 3: Travel cost (if requested)
        if "travel_cost" in objectives_to_include:
            cost_scale = 1  # Scale costs by 1 (costs should be integer or we scale them)
            cost_vars = []
            for engine in self.problem.engines_list:
                compatible_factories = self.problem.engine_compatible_factories.get(engine, [])
                compatible_msns = self.problem.engine_compatible_msns.get(engine, [])

                for factory in compatible_factories:
                    for assembly_line in self.problem.assembly_lines_list:
                        both_assigned = self.both_assigned_vars.get((engine, factory, assembly_line))
                        if both_assigned is None:
                            continue
                        cost = int(self.problem.get_travel_cost(factory, assembly_line) * cost_scale)
                        if cost > 0:
                            cost_vars.append(cost * both_assigned)

            self.travel_cost_var = self.cp_model.new_int_var(0, len(self.problem.engines_list) * 10000 * cost_scale, "travel_cost")
            if cost_vars:
                self.cp_model.add(self.travel_cost_var == sum(cost_vars))
            else:
                self.cp_model.add(self.travel_cost_var == 0)
        else:
            self.travel_cost_var = None

        # Component 4: Storage costs (if requested)
        if "storage_cost" in objectives_to_include:
            # Storage cost for allocated engines = cost_storage_per_time * (before_prep + after_prep + at_fal)
            # 1. Before prep: prep_start - delivery_time (waiting at factory before prep)
            # 2. After prep: departure_time - prep_end (waiting at factory before transport)
            # 3. At FAL: max(0, installation_date - arrival_time) (waiting at FAL, only if early)
            # Storage cost for unallocated engines = cost_storage_per_time * (time_horizon - delivery_time)
            storage_cost_vars = []
            for engine in self.problem.engines_list:
                is_allocated = self.engine_allocated_vars[engine]
                delivery_var = self.delivery_time_vars[engine]
                storage_rate = engine.engine_type_object.cost_storage_per_time

                # Storage duration variable (can be up to 3x horizon in theory)
                storage_duration = self.cp_model.new_int_var(
                    0, 3 * self.problem.time_horizon,
                    f"storage_duration_{engine.id}"
                )

                # If allocated: storage = storage_before_prep + storage_after_prep + storage_at_fal
                storage_before_prep = self.cp_model.new_int_var(
                    0, self.problem.time_horizon,
                    f"storage_before_prep_{engine.id}"
                )
                storage_after_prep = self.cp_model.new_int_var(
                    0, self.problem.time_horizon,
                    f"storage_after_prep_{engine.id}"
                )
                storage_at_fal = self.cp_model.new_int_var(
                    0, self.problem.time_horizon,
                    f"storage_at_fal_{engine.id}"
                )

                # 1. Storage before prep: prep_start - delivery
                if isinstance(delivery_var, int):
                    self.cp_model.add(
                        storage_before_prep == self.prep_start_vars[engine] - delivery_var
                    ).only_enforce_if(is_allocated)
                else:
                    self.cp_model.add(
                        storage_before_prep == self.prep_start_vars[engine] - delivery_var
                    ).only_enforce_if(is_allocated)

                # 2. Storage after prep: departure_time - prep_end
                # We need to compute departure_time based on factory/MSN assignment
                # departure_time = arrival_time - travel_time
                compatible_factories = self.problem.engine_compatible_factories.get(engine, [])
                compatible_msns = self.problem.engine_compatible_msns.get(engine, [])

                for factory in compatible_factories:
                    for assembly_line in self.problem.assembly_lines_list:
                        both_assigned = self.both_assigned_vars.get((engine, factory, assembly_line))
                        if both_assigned is None:
                            continue

                        # If both assigned: departure_time = arrival - travel_time
                        travel_time = self.problem.get_travel_time(factory, assembly_line)
                        departure_time = self.arrival_vars[engine] - travel_time

                        # storage_after_prep = max(0, departure_time - prep_end)
                        self.cp_model.add(
                            storage_after_prep >= departure_time - self.prep_end_vars[engine]
                        ).only_enforce_if(both_assigned)
                        self.cp_model.add(storage_after_prep >= 0).only_enforce_if(both_assigned)

                # 3. Storage at FAL: max(0, installation_date - arrival_time)
                for msn in compatible_msns:
                    is_msn_assigned = self.engine_to_msn_vars[(engine, msn)]

                    # If this MSN is assigned: storage_at_fal = max(0, installation_date - arrival)
                    waiting_time = msn.engine_installation_date - self.arrival_vars[engine]

                    # storage_at_fal >= max(0, waiting_time)
                    self.cp_model.add(storage_at_fal >= waiting_time).only_enforce_if(is_msn_assigned)
                    self.cp_model.add(storage_at_fal >= 0).only_enforce_if(is_msn_assigned)

                # If allocated: total storage = before_prep + after_prep + at_fal
                self.cp_model.add(
                    storage_duration == storage_before_prep + storage_after_prep + storage_at_fal
                ).only_enforce_if(is_allocated)

                # If NOT allocated: storage = time_horizon - delivery (full duration in storage)
                if isinstance(delivery_var, int):
                    self.cp_model.add(
                        storage_duration == self.problem.time_horizon - delivery_var
                    ).only_enforce_if(is_allocated.negated())
                else:
                    self.cp_model.add(
                        storage_duration == self.problem.time_horizon - delivery_var
                    ).only_enforce_if(is_allocated.negated())

                # Storage cost for this engine
                storage_cost_vars.append(storage_rate * storage_duration)

            self.storage_cost_var = self.cp_model.new_int_var(
                0, len(self.problem.engines_list) * 3 * self.problem.time_horizon * 1000,
                "storage_cost"
            )
            if storage_cost_vars:
                self.cp_model.add(self.storage_cost_var == sum(storage_cost_vars))
            else:
                self.cp_model.add(self.storage_cost_var == 0)
        else:
            self.storage_cost_var = None

        # Component 5: Unallocated MSNs (if requested)
        if "unallocated_msns" in objectives_to_include:
            # Count MSNs that receive no engine
            unallocated_msn_indicators = []
            for msn in self.problem.msns_list:
                # Is this MSN unallocated?
                engine_vars = [
                    self.engine_to_msn_vars[(engine, msn)]
                    for engine in self.problem.engines_list
                    if (engine, msn) in self.engine_to_msn_vars
                ]
                if engine_vars:
                    unallocated = self.cp_model.new_bool_var(f"msn_unallocated_{msn.id}")
                    # Unallocated if no engine assigned
                    self.cp_model.add(sum(engine_vars) == 0).only_enforce_if(unallocated)
                    self.cp_model.add(sum(engine_vars) >= 1).only_enforce_if(unallocated.negated())
                    unallocated_msn_indicators.append(unallocated)

            self.unallocated_msns_var = self.cp_model.new_int_var(
                0, len(self.problem.msns_list), "unallocated_msns"
            )
            if unallocated_msn_indicators:
                self.cp_model.add(self.unallocated_msns_var == sum(unallocated_msn_indicators))
            else:
                self.cp_model.add(self.unallocated_msns_var == 0)

            # Coupe REDONDANTE, mais indispensable à la borne inférieure. Elle est valide par
            # pigeonnier : un moteur de type T ne convient qu'aux MSN de type T, donc au moins
            # max(0, #MSN_T - #moteurs_T) MSN de chaque type restent découverts dans TOUTE
            # solution faisable (cf. EnginePreparationProblem.uncoverable_msns).
            #
            # Sans elle, CP-SAT ignore ce plancher alors que l'objectif retranche déjà
            # `uncoverable_msns` : le minimum théorique du terme de couverture descend donc à
            # -1e6 * uncoverable_msns, et la borne rendue est cette valeur négative absurde. Elle
            # reste correcte, mais si lâche que le gap d'optimalité qu'on en tire ne mesure plus
            # rien -- or c'est précisément ce que le modèle complet apporte face à un GA, qui ne
            # sait produire aucune borne. La recherche, elle, est inchangée : la coupe ne retire
            # aucune solution faisable.
            self.cp_model.add(self.unallocated_msns_var >= self.problem.uncoverable_msns)
        else:
            self.unallocated_msns_var = None

        # Build weighted objective from components that were created
        objective_terms = []

        for objective_name in objectives_to_include:
            weight = int(objective_weights[objective_name])

            if objective_name == "baseline_changes" and self.baseline_changes_var is not None:
                objective_terms.append(weight * self.baseline_changes_var)
            elif objective_name == "arrival_delays" and self.arrival_delays_var is not None:
                objective_terms.append(weight * self.arrival_delays_var)
            elif objective_name == "travel_cost" and self.travel_cost_var is not None:
                objective_terms.append(weight * self.travel_cost_var)
            elif objective_name == "storage_cost" and self.storage_cost_var is not None:
                # CRITICAL: Storage costs apply to BOTH allocated and unallocated engines!
                # Unallocated engines cost MORE (time_horizon - delivery_time)
                objective_terms.append(weight * self.storage_cost_var)
            elif objective_name == "unallocated_msns" and self.unallocated_msns_var is not None:
                # On retranche le plancher structurel, exactement comme le fait le KPI
                # "unallocated_msns_avoidable" côté problem.evaluate() : sans ça, l'objectif
                # CP-SAT et la fitness du GA différeraient de weight * uncoverable_msns.
                # C'est une constante, donc la recherche est inchangée -- seule la valeur
                # rapportée l'est, et elle vaut désormais 0 quand tout le couvrable est couvert.
                objective_terms.append(
                    weight * (self.unallocated_msns_var - self.problem.uncoverable_msns)
                )

        if objective_terms:
            total_objective = sum(objective_terms)
            self.cp_model.minimize(total_objective)
        else:
            # No objectives specified - just minimize a dummy variable
            logger.warning("No objectives specified in params_objective_function!")
            self.cp_model.minimize(0)

    # ========================================================================
    # Fixation de l'allocation -- niveau INFÉRIEUR de la décomposition bilevel
    # ========================================================================

    def fix_allocation(
        self, allocation: Dict[str, Dict[str, Optional[str]]]
    ) -> None:
        """
        Fige l'allocation décidée par le niveau SUPÉRIEUR : moteur -> MSN et moteur -> usine.

        C'est le point clé de l'architecture bilevel : le modèle du niveau inférieur n'est
        PAS écrit à la main, il est DÉRIVÉ du modèle complet en y ajoutant des égalités sur
        les binaires d'affectation. Un seul modèle CP-SAT à maintenir, donc aucun risque de
        voir le niveau bas et le modèle complet diverger sur une contrainte -- ce qui rendrait
        leurs scores incomparables.

        Ce qui reste libre après fixation : prep_start / prep_end / arrival (et delivery_time
        pour les moteurs à livraison flexible). Autrement dit l'ORDONNANCEMENT seul.

        Args:
            allocation: engine_id -> {"msn_id": str | None, "factory_id": str | None}.
                `msn_id=None` signifie moteur NON alloué (et non "à décider") : le moteur est
                explicitement désactivé via engine_allocated_vars. Un moteur absent du dict
                est en revanche laissé LIBRE, ce qui permet de ne figer qu'une partie de
                l'allocation.

        Raises:
            RuntimeError: si init_model() n'a pas été appelé avant.
            ValueError: si l'allocation demandée est incompatible (type de moteur / usine),
                car les binaires correspondants n'existent alors pas dans le modèle.

        Note:
            Les contraintes ajoutées ne sont PAS retirables (CP-SAT ne sait pas défaire une
            contrainte postée). Pour évaluer une autre allocation, il faut réinstancier le
            solveur et rappeler init_model().
        """
        if self.cp_model is None:
            raise RuntimeError("fix_allocation() doit être appelé après init_model().")

        for engine_id, decision in allocation.items():
            engine = self.problem.engines.get(engine_id)
            if engine is None:
                raise ValueError(f"Moteur inconnu dans l'allocation : {engine_id}")

            msn_id = decision.get("msn_id")
            factory_id = decision.get("factory_id")
            is_allocated_var = self.engine_allocated_vars[engine]

            # --- moteur non alloué : on désactive, le reste suit par les contraintes
            # d'assignation (allocated=0 ==> aucun MSN, aucune usine).
            if msn_id is None:
                self.cp_model.add(is_allocated_var == 0)
                continue

            self.cp_model.add(is_allocated_var == 1)

            # --- MSN : on fixe le binaire visé à 1. Les contraintes d'assignation déjà
            # postées (exactly-one par moteur, at-most-one par MSN) mettent les autres à 0,
            # inutile de les énumérer.
            msn = self.problem.msns.get(msn_id)
            if msn is None:
                raise ValueError(f"MSN inconnu dans l'allocation : {msn_id}")
            if (engine, msn) not in self.engine_to_msn_vars:
                raise ValueError(
                    f"Allocation incompatible : moteur {engine_id} (type {engine.engine_type}) "
                    f"-> MSN {msn_id} (type {msn.engine_type})"
                )
            self.cp_model.add(self.engine_to_msn_vars[(engine, msn)] == 1)

            # --- usine : idem. Un moteur alloué DOIT avoir une usine, sinon le modèle
            # devient infaisable de façon silencieuse -- on préfère l'erreur explicite.
            if factory_id is None:
                raise ValueError(
                    f"Moteur {engine_id} alloué au MSN {msn_id} mais sans usine : "
                    "le niveau supérieur doit décider les deux en décomposition 1."
                )
            factory = self.problem.factories.get(factory_id)
            if factory is None:
                raise ValueError(f"Usine inconnue dans l'allocation : {factory_id}")
            if (engine, factory) not in self.engine_to_factory_vars:
                raise ValueError(
                    f"Allocation incompatible : usine {factory_id} ne peut pas préparer "
                    f"un moteur de type {engine.engine_type} (moteur {engine_id})"
                )
            self.cp_model.add(self.engine_to_factory_vars[(engine, factory)] == 1)

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
        delivery_time: Dict[Engine, int] = {}

        # Extract assignments (only if engine is allocated)
        for (engine, msn), var in self.engine_to_msn_vars.items():
            if cpsolvercb.value(var) == 1:
                engine_to_msn[engine] = msn

        for (engine, factory), var in self.engine_to_factory_vars.items():
            if cpsolvercb.value(var) == 1:
                engine_to_factory[engine] = factory

        # Extract timing
        for engine in self.problem.engines_list:
            # Extract delivery time (fixed or decision variable)
            delivery_var = self.delivery_time_vars[engine]
            if isinstance(delivery_var, int):
                delivery_time[engine] = delivery_var
            else:
                delivery_time[engine] = cpsolvercb.value(delivery_var)

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
            delivery_time=delivery_time,
        )

        # Log CP-SAT objective component values
        logger.info("=" * 80)
        logger.info("CP-SAT OBJECTIVE COMPONENTS (from model variables)")
        logger.info("=" * 80)
        logger.info(f"Total objective value: {cpsolvercb.objective_value}")
        if self.baseline_changes_var is not None:
            logger.info(f"  Baseline changes:     {cpsolvercb.value(self.baseline_changes_var)}")
        if self.arrival_delays_var is not None:
            logger.info(f"  Arrival delays:       {cpsolvercb.value(self.arrival_delays_var)} days")
        if self.travel_cost_var is not None:
            logger.info(f"  Travel cost:          €{cpsolvercb.value(self.travel_cost_var):,.2f}")
        if self.storage_cost_var is not None:
            logger.info(f"  Storage cost:         €{cpsolvercb.value(self.storage_cost_var):,.2f}")
        if self.unallocated_msns_var is not None:
            logger.info(f"  Unallocated MSNs:     {cpsolvercb.value(self.unallocated_msns_var)}")

        # Evaluate solution with problem.evaluate() to verify correctness
        kpis = self.problem.evaluate(solution)
        logger.info("")
        logger.info("=" * 80)
        logger.info("PROBLEM EVALUATION (from solution.evaluate())")
        logger.info("=" * 80)
        for kpi_name, kpi_value in kpis.items():
            if isinstance(kpi_value, float):
                logger.info(f"  {kpi_name:20s}: {kpi_value:.2f}")
            else:
                logger.info(f"  {kpi_name:20s}: {kpi_value}")

        # Check feasibility
        feasible = self.problem.satisfy(solution)
        logger.info("")
        logger.info(f"Feasibility check: {'✅ FEASIBLE' if feasible else '❌ INFEASIBLE'}")
        logger.info("=" * 80)

        return solution

    # ========================================================================
    # Lexicographic Optimization API
    # ========================================================================

    @staticmethod
    def implements_lexico_api() -> bool:
        """Check if this solver implements lexicographic optimization."""
        return True

    def get_lexico_objectives_available(self) -> list[str]:
        """Get list of objectives available for lexicographic optimization."""
        available = []
        if self.baseline_changes_var is not None:
            available.append("baseline_changes")
        if self.arrival_delays_var is not None:
            available.append("arrival_delays")
        if self.travel_cost_var is not None:
            available.append("travel_cost")
        if self.storage_cost_var is not None:
            available.append("storage_cost")
        if self.unallocated_msns_var is not None:
            available.append("unallocated_msns")
        return available

    def _get_objective_var(self, obj_name: str):
        """Get the CP-SAT variable for a given objective."""
        if obj_name == "baseline_changes":
            return self.baseline_changes_var
        elif obj_name == "arrival_delays":
            return self.arrival_delays_var
        elif obj_name == "travel_cost":
            return self.travel_cost_var
        elif obj_name == "storage_cost":
            return self.storage_cost_var
        elif obj_name == "unallocated_msns":
            return self.unallocated_msns_var
        else:
            raise ValueError(f"Unknown objective: {obj_name}")

    def set_lexico_objective(self, obj: str) -> None:
        """Set the objective to optimize for lexicographic optimization.

        Args:
            obj: Objective name (e.g., "unallocated_msns", "arrival_delays", "storage_cost")
        """
        obj_var = self._get_objective_var(obj)
        if obj_var is None:
            raise ValueError(f"Objective {obj} not available (variable is None)")

        logger.info(f"Setting lexicographic objective: {obj}")
        self.cp_model.minimize(obj_var)

    def add_lexico_constraint(self, obj: str, value: float) -> list:
        """Fix objective at given value before optimizing next objective.

        Args:
            obj: Objective name
            value: Value to fix the objective at (typically the optimal value found)

        Returns:
            List of added constraints
        """
        obj_var = self._get_objective_var(obj)
        if obj_var is None:
            raise ValueError(f"Objective {obj} not available (variable is None)")

        # Fix objective to not exceed the optimal value found
        # For minimization objectives, we want obj_var <= value
        logger.info(f"Adding lexicographic constraint: {obj} <= {value}")
        constraint = self.cp_model.add(obj_var <= int(value))
        return [constraint]

    def get_lexico_objective_value(self, obj: str, result_storage) -> float:
        """Extract objective value from result storage.

        Args:
            obj: Objective name
            result_storage: ResultStorage from solve()

        Returns:
            Value of the objective in the best solution
        """
        if len(result_storage) == 0:
            raise ValueError("No solutions in result_storage")

        best_solution = result_storage.get_best_solution()
        kpis = self.problem.evaluate(best_solution)

        if obj not in kpis:
            raise ValueError(f"Objective {obj} not in evaluated KPIs: {list(kpis.keys())}")

        return kpis[obj]

    def get_range_objective(self, obj: str) -> tuple[float, float]:
        """Get the range (lower bound, upper bound) for an objective.

        Args:
            obj: Objective name

        Returns:
            Tuple (lower_bound, upper_bound) for the objective
        """
        obj_var = self._get_objective_var(obj)
        if obj_var is None:
            raise ValueError(f"Objective {obj} not available (variable is None)")

        # For CP-SAT integer variables, get domain bounds
        # Note: CP-SAT variables have Proto() method to access their definition
        proto = obj_var.proto
        if len(proto.domain) >= 2:
            lb = proto.domain[0]
            ub = proto.domain[-1]
            return (float(lb), float(ub))
        else:
            # Fallback: return (0, large_value)
            return (0.0, 1e9)
