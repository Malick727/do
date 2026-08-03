
import logging
import gurobipy as gp
from typing import Dict, Any, Tuple, List, Optional
import os
from part_preparation.problem import (
    EnginePreparationProblem,
    EnginePreparationSolution,
    Engine,
    MSN,
    PreparationFactory,
    AssemblyLine,
)

logger = logging.getLogger(__name__)





os.environ["GRB_LICENSE_FILE"] = "C:\\Users\\ESOWNN21\\gurobi.lic"

class GurobiEngineSolver:
    """
    Solveur MILP Gurobi à temps continu pour la préparation et l'allocation des moteurs.

    Avantages de la formulation continue :
    - Élimination de l'explosion du nombre de variables binaires indexées par le temps.
    - Empreinte mémoire (RAM) très faible (< 500 Mo au lieu de plusieurs Go).
    - Gestion de la capacité des usines via des postes/baies virtuels et contraintes disjonctives.
    """

    def __init__(
            self,
            problem: EnginePreparationProblem,
            params_objective_function: Any = None,
            **kwargs: Any
    ):
        self.problem: EnginePreparationProblem = problem
        self.params_objective_function = params_objective_function

        # Variables de décision d'affectation (Binaires)
        self.engine_allocated_vars: Dict[Engine, gp.Var] = {}
        self.engine_to_msn_vars: Dict[Tuple[Engine, MSN], gp.Var] = {}
        self.engine_to_factory_vars: Dict[Tuple[Engine, PreparationFactory], gp.Var] = {}
        self.engine_to_bay_vars: Dict[Tuple[Engine, PreparationFactory, int], gp.Var] = {}
        self.both_assigned_vars: Dict[Tuple[Engine, PreparationFactory, MSN], gp.Var] = {}

        # Variables temporelles (Continues)
        self.prep_start_vars: Dict[Engine, gp.Var] = {}
        self.prep_end_vars: Dict[Engine, gp.Var] = {}
        self.arrival_vars: Dict[Engine, gp.Var] = {}
        self.delivery_time_vars: Dict[Engine, Any] = {}

        # Variables de composants d'objectif
        self.baseline_changes_var: Optional[gp.Var] = None
        self.arrival_delays_var: Optional[gp.Var] = None
        self.travel_cost_var: Optional[gp.Var] = None
        self.storage_cost_var: Optional[gp.Var] = None
        self.unallocated_msns_var: Optional[gp.Var] = None

        self.model = gp.Model("continuous_gurobi_engine_solver")

    def init_model(self, **kwargs: Any) -> None:
        """Initialise le modèle Gurobi et construit l'ensemble des contraintes."""
        self._create_decision_variables()
        self._create_assignment_constraints()
        self._create_delivery_constraints()
        self._create_timing_constraints()
        self._create_capacity_constraints()
        self._create_takt_constraints()
        self._create_objective()

    def _create_decision_variables(self) -> None:
        """Crée toutes les variables de décision binaires et continues."""
        time_horizon = float(getattr(self.problem, 'time_horizon', 1000.0))

        for engine in self.problem.engines_list:
            # Indicateur d'allocation
            self.engine_allocated_vars[engine] = self.model.addVar(
                vtype=gp.GRB.BINARY, name=f"allocated_{engine.id}"
            )

            # Dates continues (Début, Fin, Arrivée à la FAL)
            self.prep_start_vars[engine] = self.model.addVar(
                vtype=gp.GRB.CONTINUOUS, lb=0.0, ub=time_horizon, name=f"start_{engine.id}"
            )
            self.prep_end_vars[engine] = self.model.addVar(
                vtype=gp.GRB.CONTINUOUS, lb=0.0, ub=time_horizon, name=f"end_{engine.id}"
            )
            self.arrival_vars[engine] = self.model.addVar(
                vtype=gp.GRB.CONTINUOUS, lb=0.0, ub=time_horizon, name=f"arrival_{engine.id}"
            )

            # Date de livraison flexible ou fixe
            if engine.delivery_time is None:
                self.delivery_time_vars[engine] = self.model.addVar(
                    vtype=gp.GRB.CONTINUOUS, lb=0.0, ub=time_horizon, name=f"deliv_{engine.id}"
                )
            else:
                self.delivery_time_vars[engine] = float(engine.delivery_time)

            # Affectation aux MSN compatibles
            compatible_msns = self.problem.engine_compatible_msns.get(engine, [])
            for msn in compatible_msns:
                self.engine_to_msn_vars[(engine, msn)] = self.model.addVar(
                    vtype=gp.GRB.BINARY, name=f"e{engine.id}_m{msn.id}"
                )

            # Affectation aux usines et postes de travail (Bays)
            compatible_factories = self.problem.engine_compatible_factories.get(engine, [])
            for factory in compatible_factories:
                self.engine_to_factory_vars[(engine, factory)] = self.model.addVar(
                    vtype=gp.GRB.BINARY, name=f"e{engine.id}_f{factory.id}"
                )

                capacity = factory.default_capacity if factory.default_capacity is not None else 1
                for bay in range(capacity):
                    self.engine_to_bay_vars[(engine, factory, bay)] = self.model.addVar(
                        vtype=gp.GRB.BINARY, name=f"e{engine.id}_f{factory.id}_b{bay}"
                    )

            # Pré-calcul des variables combinées (Usine ET MSN)
            for factory in compatible_factories:
                for msn in compatible_msns:
                    both_var = self.model.addVar(
                        vtype=gp.GRB.BINARY, name=f"both_{engine.id}_{factory.id}_{msn.id}"
                    )
                    x_factory = self.engine_to_factory_vars[(engine, factory)]
                    x_msn = self.engine_to_msn_vars[(engine, msn)]

                    # Linéarisation : both_var == (x_factory AND x_msn)
                    self.model.addConstr(both_var <= x_factory)
                    self.model.addConstr(both_var <= x_msn)
                    self.model.addConstr(both_var >= x_factory + x_msn - 1)
                    self.both_assigned_vars[(engine, factory, msn)] = both_var

    def _create_assignment_constraints(self) -> None:
        """Gère les contraintes d'affectation unique et de capacité des MSN."""
        for engine in self.problem.engines_list:
            compatible_msns = self.problem.engine_compatible_msns.get(engine, [])
            compatible_factories = self.problem.engine_compatible_factories.get(engine, [])
            is_allocated = self.engine_allocated_vars[engine]

            # 1. MSN : Exactement 1 MSN si le moteur est alloué
            if compatible_msns:
                self.model.addConstr(
                    gp.quicksum(self.engine_to_msn_vars[(engine, msn)] for msn in compatible_msns) == is_allocated,
                    name=f"msn_assign_{engine.id}"
                )
            else:
                self.model.addConstr(is_allocated == 0)

            # 2. Usine : Exactement 1 usine si le moteur est alloué
            if compatible_factories:
                self.model.addConstr(
                    gp.quicksum(self.engine_to_factory_vars[(engine, factory)] for factory in
                                compatible_factories) == is_allocated,
                    name=f"factory_assign_{engine.id}"
                )
            else:
                self.model.addConstr(is_allocated == 0)

            # 3. Postes de travail : Exactement 1 poste dans l'usine choisie
            for factory in compatible_factories:
                x_factory = self.engine_to_factory_vars[(engine, factory)]
                capacity = factory.default_capacity if factory.default_capacity is not None else 1
                bay_vars = [self.engine_to_bay_vars[(engine, factory, b)] for b in range(capacity)]
                self.model.addConstr(gp.quicksum(bay_vars) == x_factory)

        # 4. Chaque MSN reçoit au maximum 1 moteur
        for msn in self.problem.msns_list:
            engine_vars = [
                self.engine_to_msn_vars[(engine, msn)]
                for engine in self.problem.engines_list
                if (engine, msn) in self.engine_to_msn_vars
            ]
            if engine_vars:
                self.model.addConstr(gp.quicksum(engine_vars) <= 1, name=f"msn_cap_{msn.id}")

    def _create_delivery_constraints(self) -> None:
        """Limite les livraisons par fournisseur et par semaine calendaire."""
        has_flexible_delivery = any(e.delivery_time is None for e in self.problem.engines_list)
        if not has_flexible_delivery:
            return

        time_horizon = float(getattr(self.problem, 'time_horizon', 1000.0))
        suppliers = set(e.engine_supplier for e in self.problem.engines_list)
        max_weeks = int(time_horizon // 7) + 1

        for supplier in suppliers:
            supplier_engines = [e for e in self.problem.engines_list if e.engine_supplier == supplier]
            if not supplier_engines:
                continue

            for week in range(max_weeks):
                w_start, w_end = float(week * 7), float((week + 1) * 7)
                in_week_vars = []

                for engine in supplier_engines:
                    deliv_var = self.delivery_time_vars[engine]
                    in_week = self.model.addVar(vtype=gp.GRB.BINARY, name=f"deliv_e{engine.id}_w{week}")

                    if isinstance(deliv_var, (int, float)):
                        if w_start <= deliv_var < w_end:
                            self.model.addConstr(in_week == 1)
                        else:
                            self.model.addConstr(in_week == 0)
                    else:
                        M = time_horizon * 2.0
                        self.model.addConstr(deliv_var >= w_start - M * (1 - in_week))
                        self.model.addConstr(deliv_var <= (w_end - 0.001) + M * (1 - in_week))

                    in_week_vars.append(in_week)

                if in_week_vars:
                    self.model.addConstr(
                        gp.quicksum(in_week_vars) <= supplier.capacity_delivery_per_week,
                        name=f"supplier_cap_{supplier}_{week}"
                    )

    def _create_timing_constraints(self) -> None:
        """Lie le début de préparation, la durée, le transport et l'arrivée à la FAL."""
        time_horizon = float(getattr(self.problem, 'time_horizon', 1000.0))
        M = time_horizon * 2.0

        for engine in self.problem.engines_list:
            is_allocated = self.engine_allocated_vars[engine]
            deliv = self.delivery_time_vars[engine]
            start = self.prep_start_vars[engine]
            end = self.prep_end_vars[engine]

            # Début >= Livraison
            self.model.addConstr(start >= deliv - M * (1 - is_allocated))

            # Fin = Début + Durée de prépa selon l'usine
            compatible_factories = self.problem.engine_compatible_factories.get(engine, [])
            for factory in compatible_factories:
                x_factory = self.engine_to_factory_vars[(engine, factory)]
                duration = float(factory.get_preparation_duration(engine.engine_type))

                self.model.addConstr(end >= start + duration - M * (1 - x_factory))
                self.model.addConstr(end <= start + duration + M * (1 - x_factory))

            # Arrivée = Fin + Temps de transport
            compatible_msns = self.problem.engine_compatible_msns.get(engine, [])
            for factory in compatible_factories:
                for msn in compatible_msns:
                    both_assigned = self.both_assigned_vars[(engine, factory, msn)]
                    assembly_line = self.problem.assembly_lines[msn.assembly_line_id]
                    travel_time = float(self.problem.get_travel_time(factory, assembly_line))

                    self.model.addConstr(self.arrival_vars[engine] >= end + travel_time - M * (1 - both_assigned))
                    self.model.addConstr(self.arrival_vars[engine] <= end + travel_time + M * (1 - both_assigned))

    def _create_capacity_constraints(self) -> None:
        """Gère la capacité par non-chevauchement disjonctif sur chaque poste de travail."""
        time_horizon = float(getattr(self.problem, 'time_horizon', 1000.0))
        M = time_horizon * 2.0

        for factory in self.problem.factories_list:
            capacity = factory.default_capacity if factory.default_capacity is not None else 1
            engines_f = [e for e in self.problem.engines_list if
                         factory in self.problem.engine_compatible_factories.get(e, [])]

            for bay in range(capacity):
                for i in range(len(engines_f)):
                    for j in range(i + 1, len(engines_f)):
                        e1, e2 = engines_f[i], engines_f[j]
                        w1 = self.engine_to_bay_vars[(e1, factory, bay)]
                        w2 = self.engine_to_bay_vars[(e2, factory, bay)]

                        # Variable de séquence : delta = 1 si e1 est traité avant e2
                        delta = self.model.addVar(vtype=gp.GRB.BINARY, name=f"seq_{e1.id}_{e2.id}_f{factory.id}_b{bay}")

                        # Seule l'une des deux préparations peut se produire en premier sur un même poste
                        self.model.addConstr(
                            self.prep_start_vars[e2] >= self.prep_end_vars[e1] - M * (1 - delta) - M * (2 - w1 - w2)
                        )
                        self.model.addConstr(
                            self.prep_start_vars[e1] >= self.prep_end_vars[e2] - M * delta - M * (2 - w1 - w2)
                        )

    def _create_takt_constraints(self) -> None:
        """Assure un espacement cadencé (takt) entre les débuts de préparation dans la même usine."""
        time_horizon = float(getattr(self.problem, 'time_horizon', 1000.0))
        M = time_horizon * 2.0

        for factory in self.problem.factories_list:
            if factory.takt is None or factory.takt <= 0:
                continue

            takt = float(factory.takt)
            engines_f = [e for e in self.problem.engines_list if
                         factory in self.problem.engine_compatible_factories.get(e, [])]

            for i in range(len(engines_f)):
                for j in range(i + 1, len(engines_f)):
                    e1, e2 = engines_f[i], engines_f[j]
                    x1 = self.engine_to_factory_vars[(e1, factory)]
                    x2 = self.engine_to_factory_vars[(e2, factory)]

                    theta = self.model.addVar(vtype=gp.GRB.BINARY, name=f"takt_seq_{e1.id}_{e2.id}_f{factory.id}")

                    self.model.addConstr(
                        self.prep_start_vars[e2] >= self.prep_start_vars[e1] + takt - M * (1 - theta) - M * (
                                    2 - x1 - x2)
                    )
                    self.model.addConstr(
                        self.prep_start_vars[e1] >= self.prep_start_vars[e2] + takt - M * theta - M * (2 - x1 - x2)
                    )

    def _create_objective(self) -> None:
        """Formule la fonction objectif linéaire pondérée."""
        time_horizon = float(getattr(self.problem, 'time_horizon', 1000.0))
        M = time_horizon * 2.0

        if self.params_objective_function is not None:
            objectives_to_include = set(self.params_objective_function.objectives)
            objective_weights = dict(zip(
                self.params_objective_function.objectives,
                self.params_objective_function.weights
            ))
        else:
            obj_register = self.problem.get_objective_register()
            objectives_to_include = set(obj_register.dict_objective_to_doc.keys())
            objective_weights = {
                name: doc.default_weight
                for name, doc in obj_register.dict_objective_to_doc.items()
            }

        # 1. Baseline Changes
        if "baseline_changes" in objectives_to_include:
            baseline_change_indicators = []
            for engine in self.problem.engines_list:
                is_allocated = self.engine_allocated_vars[engine]

                if engine.baseline_msn_id is not None:
                    baseline_msn = next((m for m in self.problem.msns_list if m.id == engine.baseline_msn_id), None)
                    if baseline_msn and (engine, baseline_msn) in self.engine_to_msn_vars:
                        msn_changed = self.model.addVar(vtype=gp.GRB.BINARY, name=f"msn_changed_{engine.id}")
                        self.model.addConstr(
                            msn_changed == is_allocated - self.engine_to_msn_vars[(engine, baseline_msn)])
                        baseline_change_indicators.append(msn_changed)

                if engine.baseline_factory_id is not None:
                    baseline_factory = next(
                        (f for f in self.problem.factories_list if f.id == engine.baseline_factory_id), None)
                    if baseline_factory and (engine, baseline_factory) in self.engine_to_factory_vars:
                        factory_changed = self.model.addVar(vtype=gp.GRB.BINARY, name=f"factory_changed_{engine.id}")
                        self.model.addConstr(
                            factory_changed == is_allocated - self.engine_to_factory_vars[(engine, baseline_factory)])
                        baseline_change_indicators.append(factory_changed)

                if engine.baseline_prep_start is not None:
                    start_diff = self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name=f"start_diff_{engine.id}")
                    self.model.addConstr(start_diff >= self.prep_start_vars[engine] - engine.baseline_prep_start - M * (
                                1 - is_allocated))
                    self.model.addConstr(start_diff >= engine.baseline_prep_start - self.prep_start_vars[engine] - M * (
                                1 - is_allocated))
                    baseline_change_indicators.append(start_diff)

            self.baseline_changes_var = self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name="baseline_changes")
            if baseline_change_indicators:
                self.model.addConstr(self.baseline_changes_var == gp.quicksum(baseline_change_indicators))
            else:
                self.model.addConstr(self.baseline_changes_var == 0)

        # 2. Arrival Delays
        if "arrival_delays" in objectives_to_include:
            delay_vars = []
            for engine in self.problem.engines_list:
                compatible_msns = self.problem.engine_compatible_msns.get(engine, [])
                engine_delay = self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name=f"delay_{engine.id}")

                for msn in compatible_msns:
                    x_msn = self.engine_to_msn_vars[(engine, msn)]
                    lateness = self.arrival_vars[engine] - float(msn.engine_installation_date)
                    self.model.addConstr(engine_delay >= lateness - M * (1 - x_msn))

                delay_vars.append(engine_delay)

            self.arrival_delays_var = self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name="total_delays")
            if delay_vars:
                self.model.addConstr(self.arrival_delays_var == gp.quicksum(delay_vars))
            else:
                self.model.addConstr(self.arrival_delays_var == 0)

        # 3. Travel Cost
        if "travel_cost" in objectives_to_include:
            cost_vars = []
            for engine in self.problem.engines_list:
                compatible_factories = self.problem.engine_compatible_factories.get(engine, [])
                compatible_msns = self.problem.engine_compatible_msns.get(engine, [])

                for factory in compatible_factories:
                    for msn in compatible_msns:
                        assembly_line = self.problem.assembly_lines[msn.assembly_line_id]
                        cost = float(self.problem.get_travel_cost(factory, assembly_line))
                        if cost > 0:
                            both_assigned = self.both_assigned_vars[(engine, factory, msn)]
                            cost_vars.append(cost * both_assigned)

            self.travel_cost_var = self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name="travel_cost")
            if cost_vars:
                self.model.addConstr(self.travel_cost_var == gp.quicksum(cost_vars))
            else:
                self.model.addConstr(self.travel_cost_var == 0)

        # 4. Storage Cost
        if "storage_cost" in objectives_to_include:
            storage_cost_vars = []
            for engine in self.problem.engines_list:
                is_allocated = self.engine_allocated_vars[engine]
                deliv_var = self.delivery_time_vars[engine]
                storage_rate = float(engine.engine_type_object.cost_storage_per_time)

                storage_duration = self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name=f"storage_dur_{engine.id}")

                # Délais de stockage avant prépa, après prépa et sur la FAL
                storage_before = self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0)
                self.model.addConstr(
                    storage_before >= self.prep_start_vars[engine] - deliv_var - M * (1 - is_allocated))

                storage_after = self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0)
                compatible_factories = self.problem.engine_compatible_factories.get(engine, [])
                compatible_msns = self.problem.engine_compatible_msns.get(engine, [])

                for factory in compatible_factories:
                    for msn in compatible_msns:
                        both_assigned = self.both_assigned_vars[(engine, factory, msn)]
                        assembly_line = self.problem.assembly_lines[msn.assembly_line_id]
                        travel_time = float(self.problem.get_travel_time(factory, assembly_line))

                        departure_time = self.arrival_vars[engine] - travel_time
                        self.model.addConstr(
                            storage_after >= departure_time - self.prep_end_vars[engine] - M * (1 - both_assigned))

                storage_at_fal = self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0)
                for msn in compatible_msns:
                    x_msn = self.engine_to_msn_vars[(engine, msn)]
                    waiting_time = float(msn.engine_installation_date) - self.arrival_vars[engine]
                    self.model.addConstr(storage_at_fal >= waiting_time - M * (1 - x_msn))

                # Allocation vs Non-allocation
                self.model.addConstr(
                    storage_duration >= (storage_before + storage_after + storage_at_fal) - M * (1 - is_allocated))
                self.model.addConstr(storage_duration >= (time_horizon - deliv_var) - M * is_allocated)

                storage_cost_vars.append(storage_rate * storage_duration)

            self.storage_cost_var = self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name="storage_cost")
            if storage_cost_vars:
                self.model.addConstr(self.storage_cost_var == gp.quicksum(storage_cost_vars))
            else:
                self.model.addConstr(self.storage_cost_var == 0)

        # 5. Unallocated MSNs
        if "unallocated_msns" in objectives_to_include:
            unallocated_msn_indicators = []
            for msn in self.problem.msns_list:
                engine_vars = [
                    self.engine_to_msn_vars[(engine, msn)]
                    for engine in self.problem.engines_list
                    if (engine, msn) in self.engine_to_msn_vars
                ]
                if engine_vars:
                    unallocated = self.model.addVar(vtype=gp.GRB.BINARY, name=f"msn_unallocated_{msn.id}")
                    self.model.addConstr(unallocated == 1 - gp.quicksum(engine_vars))
                    unallocated_msn_indicators.append(unallocated)

            self.unallocated_msns_var = self.model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name="unallocated_msns")
            if unallocated_msn_indicators:
                self.model.addConstr(self.unallocated_msns_var == gp.quicksum(unallocated_msn_indicators))
            else:
                self.model.addConstr(self.unallocated_msns_var == 0)

        # Objectif Global Pondéré
        objective_terms = []
        for obj_name in objectives_to_include:
            w = float(objective_weights[obj_name])
            var = getattr(self, f"{obj_name}_var", None)
            if var is not None:
                objective_terms.append(w * var)

        if objective_terms:
            self.model.setObjective(gp.quicksum(objective_terms), gp.GRB.MINIMIZE)
        else:
            self.model.setObjective(0, gp.GRB.MINIMIZE)

    def solve(self, time_limit: float = 300.0, **kwargs: Any) -> EnginePreparationSolution:
        """Lance l'optimisation Gurobi et extrait la solution."""
        self.init_model()
        self.model.setParam("TimeLimit", time_limit)
        greedy_solution = build_greedy_warm_start(self.problem)
        print("Greedy solution:  lalala")
        self.set_warm_start(greedy_solution)

        # Optimisation mémoire
        self.model.setParam("MIPFocus", 1)  # Favorise la découverte rapide de solutions
        self.model.optimize()

        if self.model.status in [gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT] and self.model.SolCount > 0:
            return self.retrieve_solution()
        else:
            raise RuntimeError(f"Gurobi n'a pas trouvé de solution réalisable. Statut code: {self.model.status}")

    def retrieve_solution(self) -> EnginePreparationSolution:
        """Extrait les valeurs continues résolues et reconstruit la solution métier."""
        engine_to_msn: Dict[Engine, MSN] = {}
        engine_to_factory: Dict[Engine, PreparationFactory] = {}
        prep_start_times: Dict[Engine, int] = {}
        prep_end_times: Dict[Engine, int] = {}
        fal_arrival_times: Dict[Engine, int] = {}
        delivery_time: Dict[Engine, int] = {}

        # 1. Extraction des affectations
        for (engine, msn), var in self.engine_to_msn_vars.items():
            if var.X > 0.5:
                engine_to_msn[engine] = msn

        for (engine, factory), var in self.engine_to_factory_vars.items():
            if var.X > 0.5:
                engine_to_factory[engine] = factory

        # 2. Extraction des dates continues (arrondies aux entiers pour le format de solution)
        for engine in self.problem.engines_list:
            deliv_var = self.delivery_time_vars.get(engine)
            if isinstance(deliv_var, (int, float)):
                delivery_time[engine] = int(round(deliv_var))
            elif deliv_var is not None:
                delivery_time[engine] = int(round(deliv_var.X))
            else:
                delivery_time[engine] = int(engine.delivery_time) if engine.delivery_time is not None else 0

            prep_start_times[engine] = int(round(self.prep_start_vars[engine].X))
            prep_end_times[engine] = int(round(self.prep_end_vars[engine].X))
            fal_arrival_times[engine] = int(round(self.arrival_vars[engine].X))

        solution = EnginePreparationSolution(
            problem=self.problem,
            engine_to_msn=engine_to_msn,
            engine_to_factory=engine_to_factory,
            prep_start_times=prep_start_times,
            prep_end_times=prep_end_times,
            fal_arrival_times=fal_arrival_times,
            delivery_time=delivery_time,
        )

        logger.info("=" * 80)
        logger.info(f"RÉSOLU PAR GUROBI - Objectif: {self.model.ObjVal:.2f}")
        logger.info("=" * 80)
        return solution

    def set_warm_start(self, initial_solution: EnginePreparationSolution) -> None:
        """Injecte une solution initiale (MIP Start) dans les variables Gurobi."""

        for engine in self.problem.engines_list:
            is_allocated = 1.0 if engine in initial_solution.engine_to_msn else 0.0

            # 1. Variable d'allocation
            if engine in self.engine_allocated_vars:
                self.engine_allocated_vars[engine].Start = is_allocated

            # 2. Variable de début et fin de préparation
            if is_allocated > 0.5:
                self.prep_start_vars[engine].Start = float(initial_solution.prep_start_times[engine])
                self.prep_end_vars[engine].Start = float(initial_solution.prep_end_times[engine])
                self.arrival_vars[engine].Start = float(initial_solution.fal_arrival_times[engine])

                # 3. Affectation MSN
                assigned_msn = initial_solution.engine_to_msn[engine]
                for msn in self.problem.engine_compatible_msns.get(engine, []):
                    val = 1.0 if msn == assigned_msn else 0.0
                    if (engine, msn) in self.engine_to_msn_vars:
                        self.engine_to_msn_vars[(engine, msn)].Start = val

                # 4. Affectation Usine
                assigned_factory = initial_solution.engine_to_factory[engine]
                for factory in self.problem.engine_compatible_factories.get(engine, []):
                    val = 1.0 if factory == assigned_factory else 0.0
                    if (engine, factory) in self.engine_to_factory_vars:
                        self.engine_to_factory_vars[(engine, factory)].Start = val

    def solve(self, time_limit: float = 300.0, warm_start_solution: Optional[EnginePreparationSolution] = None,
              **kwargs: Any) -> EnginePreparationSolution:
        self.init_model()

        # Injection du warm start si fourni
        if warm_start_solution is not None:
            logger.info("Injection de la solution initiale (Warm Start) dans Gurobi...")
            self.set_warm_start(warm_start_solution)
        greedy_solution = build_greedy_warm_start(self.problem)
        self.set_warm_start(greedy_solution)
        print("greedy solution")
        self.model.setParam("TimeLimit", time_limit)
        self.model.setParam("MIPFocus", 1)
        self.model.optimize()

        return self.retrieve_solution()
    # =========================================================================
    # API d'optimisation lexicographique
    # =========================================================================

    @staticmethod
    def implements_lexico_api() -> bool:
        return True

    def get_lexico_objectives_available(self) -> List[str]:
        return [
            name for name in ["baseline_changes", "arrival_delays", "travel_cost", "storage_cost", "unallocated_msns"]
            if getattr(self, f"{name}_var", None) is not None
        ]

    def _get_objective_var(self, obj_name: str) -> gp.Var:
        var = getattr(self, f"{obj_name}_var", None)
        if var is None:
            raise ValueError(f"Objectif '{obj_name}' non défini.")
        return var

    def set_lexico_objective(self, obj: str) -> None:
        obj_var = self._get_objective_var(obj)
        self.model.setObjective(obj_var, gp.GRB.MINIMIZE)

    def add_lexico_constraint(self, obj: str, value: float) -> List[gp.Constr]:
        obj_var = self._get_objective_var(obj)
        constr = self.model.addConstr(obj_var <= value, name=f"lexico_{obj}")
        return [constr]

    def get_lexico_objective_value(self, obj: str, result_storage: Any) -> float:
        if len(result_storage) == 0:
            raise ValueError("Aucune solution trouvée.")
        best_solution = result_storage.get_best_solution()
        kpis = self.problem.evaluate(best_solution)
        return float(kpis[obj])


def build_greedy_warm_start(problem: EnginePreparationProblem) -> EnginePreparationSolution:
    """
    Génère une solution réalisable rapide via une stratégie gloutonne (Greedy).
    """
    engine_to_msn: Dict[Engine, MSN] = {}
    engine_to_factory: Dict[Engine, PreparationFactory] = {}
    prep_start_times: Dict[Engine, int] = {}
    prep_end_times: Dict[Engine, int] = {}
    fal_arrival_times: Dict[Engine, int] = {}
    delivery_time: Dict[Engine, int] = {}

    # Suivi des disponibilités (ex. date de fin du dernier moteur sur l'usine)
    factory_next_available: Dict[PreparationFactory, float] = {f: 0.0 for f in problem.factories_list}
    assigned_msns = set()

    # Trier les moteurs par date de livraison prévisionnelle ou ID
    sorted_engines = sorted(
        problem.engines_list,
        key=lambda e: e.delivery_time if e.delivery_time is not None else 0
    )

    for engine in sorted_engines:
        e_deliv = float(engine.delivery_time) if engine.delivery_time is not None else 0.0
        delivery_time[engine] = int(e_deliv)

        compatible_msns = problem.engine_compatible_msns.get(engine, [])
        compatible_factories = problem.engine_compatible_factories.get(engine, [])

        # Trouver un MSN libre
        available_msn = next((m for m in compatible_msns if m not in assigned_msns), None)

        if available_msn and compatible_factories:
            # Choisir l'usine disponible le plus tôt
            factory = min(compatible_factories, key=lambda f: factory_next_available[f])
            duration = float(factory.get_preparation_duration(engine.engine_type))

            # Calcul du début de préparation : max(livraison, disponibilité usine)
            start_t = max(e_deliv, factory_next_available[factory])
            end_t = start_t + duration

            # Calcul du transport vers la FAL
            assembly_line = problem.assembly_lines[available_msn.assembly_line_id]
            travel_time = float(problem.get_travel_time(factory, assembly_line))
            arrival_t = end_t + travel_time

            # Mettre à jour les affectations
            engine_to_msn[engine] = available_msn
            engine_to_factory[engine] = factory
            prep_start_times[engine] = int(round(start_t))
            prep_end_times[engine] = int(round(end_t))
            fal_arrival_times[engine] = int(round(arrival_t))

            assigned_msns.add(available_msn)

            # Mise à jour de la disponibilité de l'usine (prise en compte simplifiée du takt)
            takt = float(factory.takt) if factory.takt else 1.0
            factory_next_available[factory] = start_t + max(duration, takt)

    return EnginePreparationSolution(
        problem=problem,
        engine_to_msn=engine_to_msn,
        engine_to_factory=engine_to_factory,
        prep_start_times=prep_start_times,
        prep_end_times=prep_end_times,
        fal_arrival_times=fal_arrival_times,
        delivery_time=delivery_time
    )