"""
Niveau INFÉRIEUR de la décomposition 1 : l'allocation (moteur -> MSN, moteur -> usine) est
FIXÉE par le niveau supérieur, il ne reste qu'à ordonnancer la préparation.

Ce module ne contient que des fonctions : ce sont les trois façons de résoudre ce
sous-problème, pas une hiérarchie de classes. L'unique abstraction de l'architecture est le
`Ranker` (cf. bilevel.ranking), et c'est lui qui appelle ces fonctions.

Les trois méthodes, par coût croissant :

    simulate_greedy_from_allocation   heuristique de liste           ~0.3 ms
    schedule_allocation_with_cpsat    modèle réduit (capacité+retards) ~0.2 s
    solve_allocation_with_full_cpsat  modèle complet figé            20-600 s

Elles ne résolvent PAS le même modèle -- la version réduite ignore le takt et le stockage --
donc leurs scores ne sont pas interchangeables. C'est précisément ce que la comparaison
d'approches doit mesurer.
"""
from typing import Any, Dict, Optional, Tuple
import math

from part_preparation.problem import (
    EnginePreparationProblem,
    EnginePreparationSolution,
    aggregate_score,
    analytic_score,
)

# Score attribué à une allocation dont l'ordonnancement échoue (modèle infaisable ou
# time limit atteint sans aucune solution). Volontairement très au-dessus de tout score
# atteignable, pour que ces individus soient éliminés par la sélection sans faire planter
# le GA.
INFEASIBLE_SCORE = 9999999.0

# Plafond de temps par évaluation CP-SAT. La valeur historique (1s) rendait le mode trompeur :
# le solveur s'arrêtait le plus souvent sur une solution FEASIBLE, donc la fitness d'une
# allocation était une borne SUPÉRIEURE de la valeur optimale d'ordonnancement, et deux
# allocations pouvaient s'inverser dans le classement selon la chance du solveur.
CPSAT_EVAL_TIME_LIMIT = 10.0


# ==========================================================================================
# 1. GREEDY -- heuristique de liste
# ==========================================================================================
def simulate_greedy_from_allocation(
    allocation: Dict[str, Dict[str, Any]], problem: EnginePreparationProblem
) -> EnginePreparationSolution:
    """
    Ordonnance une allocation DÉJÀ décodée (dict complet avec engine_obj/msn_obj/factory_obj,
    cf. get_allocation_from_chromosome) par une heuristique de liste.

    Principe : chaque moteur a une date de démarrage IDÉALE (celle qui le fait arriver
    pile à la date d'installation de son MSN) ; on traite les moteurs par date idéale
    croissante et on les pose au plus tôt dans leur usine.

    La capacité des usines est respectée PAR CONSTRUCTION : `factory_queues` maintient une
    file par unité de capacité, donc au plus `default_capacity` moteurs se chevauchent. C'est
    ce qui permet au GA de ne pas appeler problem.satisfy() dans sa boucle.
    """
    engine_to_msn, engine_to_factory = {}, {}
    prep_start_times, prep_end_times, fal_arrival_times, delivery_times = {}, {}, {}, {}
    engine_jobs = []

    for engine_id, data in allocation.items():
        engine = data["engine_obj"]
        msn = data["msn_obj"]
        factory = data["factory_obj"]

        engine_to_factory[engine] = factory
        engine_to_msn[engine] = msn
        prep_duration = math.ceil(factory.get_preparation_duration(engine.engine_type))

        if msn is not None:
            assembly_line = problem.assembly_lines[msn.assembly_line_id]
            travel_time = math.ceil(problem.get_travel_time(factory, assembly_line))
            ideal_start_time = msn.engine_installation_date - travel_time - prep_duration
        else:
            # Moteur non alloué : aucune date cible, on le repousse en fin de liste pour
            # qu'il ne prenne pas la capacité d'un moteur qui, lui, a une échéance.
            travel_time = 0
            ideal_start_time = float("inf")

        engine_jobs.append({
            "engine": engine, "factory": factory, "msn": msn,
            "ideal_start": ideal_start_time, "duration": prep_duration, "travel": travel_time,
        })

    engine_jobs.sort(key=lambda job: job["ideal_start"])
    factory_queues = {f: [0] * (f.default_capacity or 1) for f in problem.factories_list}

    for job in engine_jobs:
        factory, engine, msn = job["factory"], job["engine"], job["msn"]
        ideal_start, prep_duration = job["ideal_start"], job["duration"]

        factory_queues[factory].sort()
        delivery_time = engine.delivery_time if engine.delivery_time is not None else 0
        delivery_times[engine] = engine.delivery_time

        earliest_start = math.ceil(max(factory_queues[factory][0], delivery_time))

        if msn is not None:
            # On ne démarre pas plus tôt que nécessaire : partir avant `ideal_start`
            # ferait arriver le moteur en avance, donc du stockage à la FAL pour rien.
            start_time = max(earliest_start, ideal_start)
            prep_start_times[engine] = start_time
            prep_end = start_time + prep_duration
            fal_arrival_times[engine] = prep_end + job["travel"]
        else:
            start_time = earliest_start
            prep_start_times[engine] = start_time
            prep_end = start_time + prep_duration
            fal_arrival_times[engine] = None

        prep_end_times[engine] = prep_end
        factory_queues[factory][0] = prep_end

    return EnginePreparationSolution(
        problem=problem, engine_to_msn=engine_to_msn, engine_to_factory=engine_to_factory,
        prep_start_times=prep_start_times, prep_end_times=prep_end_times,
        fal_arrival_times=fal_arrival_times, delivery_time=delivery_times,
    )


# ==========================================================================================
# 2. CP-SAT RÉDUIT -- capacité cumulative + minimisation des retards
# ==========================================================================================
def _scheduling_upper_bound(
    allocation: Dict[str, Dict[str, Any]], problem: EnginePreparationProblem
) -> int:
    """
    Borne supérieure des dates du modèle d'ordonnancement, GARANTIE atteignable.

    Ne surtout pas utiliser `time_horizon` ici. L'horizon n'est pas une échéance dure : un
    moteur qui arrive après la date d'installation est PÉNALISÉ (cf. problem.evaluate et
    daily_delay_cost), pas interdit -- et le simulateur greedy, lui, ordonnance librement
    au-delà. Borner les dates à l'horizon transformait donc un coût en contrainte, et rendait
    les deux ordonnanceurs incomparables sur le sens même de « faisable ».

    Le symptôme n'apparaissait qu'à grande échelle. Sur 1000 MSN, un chromosome aléatoire
    envoie ~130 moteurs vers une usine de capacité 1 et de durée 15 jours, soit 1950 jours de
    préparation pour un horizon de 1600 : le modèle était INFAISABLE pour presque tout
    chromosome, et le ranker CP-SAT ne rendait plus aucun signal de classement.

    La borne retenue est le pire cas strict : tout ordonnancer en séquence dans une seule
    usine, à partir de la dernière livraison. Très lâche, mais CP-SAT ne souffre pas d'un
    domaine large -- alors qu'une borne trop serrée rend le modèle infaisable en silence.
    """
    latest_delivery = 0
    total_duration = 0

    for data in allocation.values():
        engine = data["engine_obj"]
        factory = data["factory_obj"]
        if factory is None:
            continue
        if engine.delivery_time is not None:
            latest_delivery = max(latest_delivery, int(round(engine.delivery_time)))
        total_duration += int(round(factory.get_preparation_duration(engine.engine_type)))

    return int(max(problem.time_horizon, latest_delivery + total_duration) + 200)


def schedule_allocation_with_cpsat(
    allocation: Dict[str, Dict[str, Any]],
    problem: EnginePreparationProblem,
    time_limit: float = CPSAT_EVAL_TIME_LIMIT,
    stats: Optional[Dict[str, int]] = None,
    num_workers: Optional[int] = None,
) -> Optional[EnginePreparationSolution]:
    """
    Ordonnance une allocation DÉJÀ décodée avec un modèle CP-SAT RÉDUIT : capacité cumulative
    des usines + minimisation de la somme des retards. Ni takt, ni coût de stockage.

    Ce modèle est écrit à la main (et non dérivé de CPSatEngineSolver) parce qu'il doit tenir
    dans la boucle du GA : ~0.2 s contre 20 s et plus pour le modèle complet. Le prix à payer
    est qu'il n'optimise pas exactement le même critère que le modèle complet -- d'où
    l'existence de solve_allocation_with_full_cpsat comme référence.

    L'unicité des MSN (un MSN <-> au plus un moteur) est garantie en amont par le décodeur du
    chromosome : ici les moteurs déjà décodés sont planifiés sans conflit possible.

    `stats`, si fourni, est incrémenté avec le nombre d'appels et le nombre de résolutions
    réellement prouvées OPTIMAL : sans ça, on n'a aucun moyen de savoir si le plafond de temps
    est suffisant sur la taille d'instance courante -- et donc si le classement produit est
    un vrai classement ou du bruit.

    `num_workers` fixe le parallélisme INTERNE de CP-SAT. `None` laisse le défaut d'OR-Tools,
    qui résout en « auto » et lance un portfolio d'une vingtaine de sous-solveurs -- le bon
    réglage quand on appelle cette fonction depuis un process unique (la boucle du GA).

    Mettre 1 dès que l'appelant est LUI-MÊME parallélisé, ce qui est le cas de l'étiquetage de
    dataset (surrogate.dataset_builder, un process par réplicat). Sinon les deux niveaux de
    parallélisme se multiplient : 8 process x ~22 threads = 176 threads pour 24 cœurs, chaque
    sous-solveur tournant sur un septième de cœur. Le plafond étant en temps HORLOGE, la
    sur-souscription ne rallonge pas le run -- elle dégrade silencieusement chaque label, ce
    qui est pire, parce que rien dans la sortie ne le signale.

    Returns:
        La solution complète, ou None si CP-SAT n'a rien trouvé dans le temps imparti.
    """
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    max_time = _scheduling_upper_bound(allocation, problem)

    start_vars, end_vars, delay_vars = {}, {}, {}
    factory_intervals = {f.id: [] for f in problem.factories_list}
    factory_demands = {f.id: [] for f in problem.factories_list}

    for engine_id, data in allocation.items():
        engine = data["engine_obj"]
        msn = data["msn_obj"]
        factory = data["factory_obj"]

        if msn is None or factory is None:
            continue

        assembly_line = problem.assembly_lines[msn.assembly_line_id]

        min_start = int(round(engine.delivery_time if engine.delivery_time is not None else 0))
        S = model.NewIntVar(min_start, max_time, f"S_{engine_id}")
        E = model.NewIntVar(min_start, max_time, f"E_{engine_id}")

        prep_duration = int(round(factory.get_preparation_duration(engine.engine_type)))
        interval = model.NewIntervalVar(S, prep_duration, E, f"I_{engine_id}")

        start_vars[engine_id] = S
        end_vars[engine_id] = E
        factory_intervals[factory.id].append(interval)
        factory_demands[factory.id].append(1)

        travel_time = int(round(problem.get_travel_time(factory, assembly_line)))
        target_date = int(round(msn.engine_installation_date))

        D = model.NewIntVar(0, max_time, f"D_{engine_id}")
        model.AddMaxEquality(D, [0, E + travel_time - target_date])
        delay_vars[engine_id] = D

    for f in problem.factories_list:
        intervals = factory_intervals[f.id]
        if intervals and f.default_capacity is not None:
            model.AddCumulative(intervals, factory_demands[f.id], int(f.default_capacity))

    # Le transport et les MSN non couverts sont des CONSTANTES à allocation figée : les
    # inclure dans l'objectif CP-SAT ne changerait pas l'argmin, seulement la taille du modèle.
    model.Minimize(sum(delay_vars.values()))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    if num_workers is not None:
        solver.parameters.num_workers = num_workers
    status = solver.Solve(model)

    if stats is not None:
        stats["calls"] = stats.get("calls", 0) + 1
        stats["optimal"] = stats.get("optimal", 0) + int(status == cp_model.OPTIMAL)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    engine_to_msn, engine_to_factory = {}, {}
    prep_start_times, prep_end_times, fal_arrival_times, delivery_times = {}, {}, {}, {}

    for engine_id, data in allocation.items():
        engine = data["engine_obj"]
        engine_to_msn[engine] = data["msn_obj"]
        engine_to_factory[engine] = data["factory_obj"]
        delivery_times[engine] = engine.delivery_time

        if engine_id in start_vars:
            end_t = solver.Value(end_vars[engine_id])
            prep_start_times[engine] = solver.Value(start_vars[engine_id])
            prep_end_times[engine] = end_t

            assembly_line = problem.assembly_lines[data["msn_obj"].assembly_line_id]
            travel_time = problem.get_travel_time(data["factory_obj"], assembly_line)
            fal_arrival_times[engine] = end_t + travel_time
        else:
            prep_start_times[engine] = None
            prep_end_times[engine] = None
            fal_arrival_times[engine] = None

    return EnginePreparationSolution(
        problem=problem, engine_to_msn=engine_to_msn, engine_to_factory=engine_to_factory,
        prep_start_times=prep_start_times, prep_end_times=prep_end_times,
        fal_arrival_times=fal_arrival_times, delivery_time=delivery_times,
    )


# ==========================================================================================
# 3. CP-SAT COMPLET -- vérité terrain
# ==========================================================================================
def solve_allocation_with_full_cpsat(
    allocation: Dict[str, Dict[str, Any]],
    problem: EnginePreparationProblem,
    time_limit: float = 600.0,
) -> Tuple[Optional[EnginePreparationSolution], bool]:
    """
    Valeur de RÉFÉRENCE d'une allocation : optimum du modèle d'ordonnancement COMPLET
    (retards + stockage + takt + capacité), obtenu en figeant l'allocation dans
    CPSatEngineSolver via fix_allocation().

    C'est ici que se joue l'automatisation de la formulation : le modèle du niveau inférieur
    n'est pas réécrit, il est DÉRIVÉ du modèle complet par ajout d'égalités sur les binaires
    d'affectation. Un seul modèle à maintenir, donc niveau bas et modèle complet ne peuvent
    pas diverger sur une contrainte.

    À distinguer de schedule_allocation_with_cpsat, qui résout un modèle réduit en ~0.2 s.
    Celui-ci prend 20 s à 600 s par évaluation : il est inutilisable dans une boucle de GA, et
    sert à produire la vérité terrain que le surrogate doit approcher, ou à arbitrer les
    finalistes en post-traitement.

    L'objectif CP-SAT interne est restreint à `arrival_delays + storage_cost`, c'est-à-dire
    aux seuls termes qui VARIENT à allocation figée -- transport et couverture y sont des
    constantes additives. Attention à une conséquence : le stockage entre dans l'objectif
    interne alors que son poids est 0 dans le score agrégé rendu (cf. AGGREGATED_WEIGHTS). Il
    joue donc le rôle de DÉPARTAGEUR entre plannings de retard égal, en préférant celui qui
    immobilise le moins. C'est un choix, pas un effet de bord.

    Returns:
        (solution, is_optimal). solution vaut None si rien n'a été trouvé.
    """
    from discrete_optimization.generic_tools.cp_tools import ParametersCp
    from discrete_optimization.generic_tools.do_problem import (
        ParamsObjectiveFunction, ObjectiveHandling, ModeOptim,
    )
    from part_preparation.solvers.cpsat import CPSatEngineSolver

    flat_allocation = {
        engine_id: {"msn_id": data["msn_id"], "factory_id": data["factory_id"]}
        for engine_id, data in allocation.items()
    }

    solver = CPSatEngineSolver(
        problem=problem,
        params_objective_function=ParamsObjectiveFunction(
            objective_handling=ObjectiveHandling.AGGREGATE,
            objectives=["arrival_delays", "storage_cost"],
            weights=[1.0, 1.0],
            sense_function=ModeOptim.MINIMIZATION,
        ),
    )
    solver.init_model()
    solver.fix_allocation(flat_allocation)

    parameters_cp = ParametersCp.default_cpsat()
    parameters_cp.nb_process = 1
    result = solver.solve(parameters_cp=parameters_cp, time_limit=time_limit)

    if len(result) == 0:
        return None, False

    return result[-1][0], "OPTIMAL" in str(solver.status_solver)


# ==========================================================================================
# 4. TERMES CALCULABLES SANS ORDONNANCER
# ==========================================================================================
def exact_allocation_terms(
    allocation: Dict[str, Dict[str, Any]], problem: EnginePreparationProblem
) -> Tuple[float, int]:
    """
    Termes de l'objectif calculables EXACTEMENT depuis la seule allocation, sans ordonnancer :
    coût de transport et nombre de MSN non alloués. Ils représentent ~98% du score, et les
    faire prédire par un réseau serait du gaspillage de capacité -- un surrogate ne doit
    apprendre que le résidu (retards + stockage), qui lui dépend du planning.
    """
    travel_cost = 0.0
    allocated_msn_ids = set()

    for data in allocation.values():
        msn = data.get("msn_obj")
        factory = data.get("factory_obj")
        if msn is None or factory is None:
            continue
        allocated_msn_ids.add(msn.id)
        assembly_line = problem.assembly_lines[msn.assembly_line_id]
        travel_cost += problem.get_travel_cost(factory, assembly_line)

    return travel_cost, len(problem.msns_list) - len(allocated_msn_ids)


def analytic_score_from_allocation(
    allocation: Dict[str, Dict[str, Any]], problem: EnginePreparationProblem
) -> float:
    """
    Part EXACTE du score, calculée depuis la seule allocation, sans planning ni réseau
    (cf. problem.ANALYTIC_OBJECTIVES). C'est le complément de ce qu'un surrogate prédit.
    """
    travel_cost, unallocated = exact_allocation_terms(allocation, problem)
    return analytic_score({
        "travel_cost": travel_cost,
        # Même convention que problem.evaluate() : on ne facture que l'évitable.
        "unallocated_msns_avoidable": float(max(0, unallocated - problem.uncoverable_msns)),
    })


def greedy_score_and_kpis(
    allocation: Dict[str, Dict[str, Any]], problem: EnginePreparationProblem
) -> Tuple[float, Dict[str, float]]:
    """
    Ordonnance en greedy et renvoie (score, KPI bruts).

    Les KPI en plus du score, parce que la génération de dataset a besoin de la décomposition
    analytique/résiduelle (cf. problem.residual_score) et non du seul total : sans ce point
    d'entrée, elle devrait re-simuler le planning une deuxième fois pour les récupérer.
    """
    solution = simulate_greedy_from_allocation(allocation, problem)
    kpis = problem.evaluate(solution)
    return aggregate_score(kpis), kpis


def score_solution(solution: EnginePreparationSolution, problem: EnginePreparationProblem) -> float:
    """
    Score d'une solution complète -- point de vérité unique de toute l'architecture.

    Numériquement ÉGAL à l'objectif minimisé par CPSatEngineSolver sur la même solution : une
    fitness de GA et une valeur d'objectif CP-SAT sont donc directement comparables. Les
    termes et poids sont déclarés dans problem.AGGREGATED_OBJECTIVES / AGGREGATED_WEIGHTS,
    pour qu'un changement de pondération s'applique mécaniquement à tous les rankers.
    """
    return aggregate_score(problem.evaluate(solution))
