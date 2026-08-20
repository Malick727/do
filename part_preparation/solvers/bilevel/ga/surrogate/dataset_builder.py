"""
Étiquetage des allocations : produit les datasets que les surrogates de D1 apprennent.

Un échantillon = une allocation du niveau supérieur, plus le coût que le VRAI niveau bas lui
attribue. Les instances viennent de part_preparation.instances, et la CLI est en bas de ce
fichier :

    python -m part_preparation.solvers.bilevel.ga.surrogate.dataset_builder --instance medium

Aucune dépendance à PyTorch ni à DEAP ici -- on ne fait que tirer des allocations et appeler
bilevel.lower.
"""
from typing import Any, Dict, List, Optional
import json
import multiprocessing as mp
import os
import random
import time

from part_preparation import instances
from part_preparation.problem import (
    EnginePreparationProblem, aggregate_score, analytic_score, residual_score,
)
from part_preparation.solvers.bilevel.ga import encoding
from part_preparation.solvers.bilevel.lower import INFEASIBLE_SCORE, greedy_score_and_kpis


def collect_allocations_via_ga(
    problem: EnginePreparationProblem,
    pop_size: int = 40,
    ngen: int = 35,
    snapshot_every: int = 3,
    cxpb: float = 0.7,
    mutpb: float = 0.2,
    indpb: float = 0.1,
    tournsize: int = 3,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Fait tourner un GA JETABLE à évaluation greedy (coût réel, rapide -- pas de CP-SAT) et
    prend des instantanés de la population à intervalles réguliers.

    Le but n'est pas de résoudre le problème mais de produire des allocations VARIÉES à
    étiqueter. Comparer uniquement des allocations tirées au hasard donne des paires toujours
    faciles à départager (gros écarts de qualité) ; en collectant aussi les générations
    tardives -- où la population a convergé et devient homogène -- on obtient naturellement
    des paires « proches », celles dont le réseau a besoin pour apprendre à discriminer les
    cas fins.

    Ce GA garde sa propre boucle plutôt que d'utiliser BilevelGASolverD1 : ce dernier ne
    conserve que le CLASSEMENT d'une population, alors qu'ici on veut le score réel de CHAQUE
    individu, qui est précisément l'étiquette à stocker.

    Returns:
        Liste de {"alloc", "cost", "residual", "analytic"}. `alloc` est la forme PLATE
        (engine_id -> {msn_id, factory_id}), la seule sérialisable en JSON.
    """
    rng = random.Random(seed)
    if seed is not None:
        # encoding.* tire sur le RNG global du module random.
        random.seed(seed)

    def evaluate(chromosome):
        # Décodage UNIQUE : le coût stocké doit venir de CETTE allocation, pas d'un second
        # décodage. Le décodeur est déterministe, mais s'en remettre à cette propriété pour
        # apparier une étiquette à son allocation serait fragile.
        allocation = encoding.decode(chromosome, problem)
        cost, kpis = greedy_score_and_kpis(allocation, problem)
        return allocation, cost, kpis

    population = [encoding.create_chromosome(problem) for _ in range(pop_size)]
    evaluated = [evaluate(chromosome) for chromosome in population]
    pool: List[Dict[str, Any]] = []

    def snapshot(generation: int):
        for allocation, cost, kpis in evaluated:
            pool.append({
                "generation": generation,
                "alloc": {
                    engine_id: {"msn_id": data["msn_id"], "factory_id": data["factory_id"]}
                    for engine_id, data in allocation.items()
                },
                "cost": cost,
                # On stocke la DÉCOMPOSITION, pas seulement le total : le surrogate n'apprend
                # que `residual`, et `analytic` est recalculé exactement à l'inférence. Les
                # deux se resomment en `cost` (cf. problem.aggregate_score).
                "residual": residual_score(kpis),
                "analytic": analytic_score(kpis),
            })

    snapshot(0)

    for generation in range(1, ngen + 1):
        scores = [cost for _, cost, _ in evaluated]

        parents = []
        for _ in range(pop_size):
            aspirants = rng.sample(range(pop_size), min(tournsize, pop_size))
            parents.append(population[min(aspirants, key=lambda i: scores[i])])

        offspring = [list(parent) for parent in parents]
        for i in range(1, len(offspring), 2):
            if rng.random() < cxpb:
                encoding.crossover_blocks(offspring[i - 1], offspring[i], problem)
        for i in range(len(offspring)):
            if rng.random() < mutpb:
                encoding.mutate_blocks(offspring[i], indpb, problem)

        population = offspring
        evaluated = [evaluate(chromosome) for chromosome in population]

        if generation % snapshot_every == 0:
            snapshot(generation)

    return pool


def deduplicate(pool: List[Dict[str, Any]], instance_id: int,
                instance_name: str) -> List[Dict[str, Any]]:
    """
    Déduplique les allocations d'un pool et y attache son origine (nom d'instance + réplicat).

    `instance_name` est écrit dans CHAQUE échantillon, et pas seulement passé en argument du
    script. L'entraînement doit reconstruire le problem de chaque `instance_id` pour recalculer
    les features (cf. surrogate.features) ; sans le nom, il doit le lui redemander en ligne de
    commande, et se tromper de nom ne lève rien de visible -- si les tailles coïncident, les
    features sont simplement calculées sur des dates de MSN qui n'ont pas produit ces coûts, et
    le réseau apprend un bruit. Le dataset porte donc de quoi se relire seul.

    Le GA converge : la population tardive contient des individus strictement identiques. Sans
    déduplication, les optima locaux seraient repondérés artificiellement -- le réseau
    apprendrait surtout à reconnaître les allocations que le GA visite souvent.

    Les échantillons infaisables (cf. lower.INFEASIBLE_SCORE) sont écartés : leur coût est une
    sentinelle, pas une mesure, et l'apprendre n'aurait aucun sens.
    """
    seen = set()
    samples = []

    for entry in pool:
        if entry["cost"] >= INFEASIBLE_SCORE:
            continue
        key = json.dumps(entry["alloc"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        samples.append({
            "instance_id": instance_id,
            "instance": instance_name,
            # Première génération où l'allocation apparaît : la déduplication garde la
            # première occurrence, donc c'est bien la date d'APPARITION et non la dernière
            # fois qu'elle a survécu dans l'archive.
            "generation": entry["generation"],
            "alloc": entry["alloc"],
            "cost": entry["cost"],
            "residual": entry["residual"],
            "analytic": entry["analytic"],
        })

    return samples


# ==========================================================================================
# Collecte ALÉATOIRE -- pas de GA, une instance donne N allocations indépendantes
# ==========================================================================================
def collect_random_allocations(
    problem: EnginePreparationProblem,
    labeller: "Ranker",
    nb_allocations: int = 30,
    perturbation_steps: int = 0,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Tire `nb_allocations` allocations INDÉPENDANTES et les étiquette en un seul lot.

    Beaucoup moins cher qu'une collecte par GA -- aucune trajectoire à dérouler, donc le coût
    est exactement `nb_allocations` résolutions du niveau bas -- et trivialement
    parallélisable sur les instances.

    LIMITE À CONNAÎTRE : des allocations tirées uniformément sont toutes mauvaises, et
    mauvaises de façon SEMBLABLE. Un réseau entraîné dessus apprend à départager des paires à
    gros écart, alors qu'en fin de GA il devra départager une population convergée, donc
    homogène. La validation sera bonne sur le cas facile et muette sur le cas difficile.

    `perturbation_steps` atténue ça à coût quasi nul : au lieu de tirer chaque allocation
    indépendamment, une fraction est obtenue en MUTANT une allocation déjà tirée. On obtient
    des paires proches (individus apparentés) sans dérouler de GA. Mettre 0 pour du purement
    aléatoire.

    Returns:
        Liste de {"generation", "alloc", "cost", "residual", "analytic"}. `generation` vaut
        None : la notion n'a pas de sens hors d'une trajectoire de GA, et mettre 0 laisserait
        croire à une première génération.
    """
    rng = random.Random(seed)
    if seed is not None:
        random.seed(seed)  # encoding.* tire sur le RNG global

    chromosomes: List[List[int]] = []
    for index in range(nb_allocations):
        if perturbation_steps > 0 and chromosomes and index % 2 == 1:
            # Voisin d'une allocation déjà tirée : c'est ce qui fournit les paires « proches »
            # que le tirage uniforme ne produit jamais.
            chromosome = list(rng.choice(chromosomes))
            for _ in range(perturbation_steps):
                encoding.mutate_blocks(chromosome, 1.0 / max(1, len(problem.engines_list)), problem)
        else:
            chromosome = encoding.create_chromosome(problem)
        chromosomes.append(chromosome)

    ranking = labeller.rank(chromosomes, problem, encoding.decode)
    if not ranking.has_true_scores:
        raise ValueError(
            f"L'étiqueteur {labeller.name!r} ne produit pas de scores réels : "
            "il ne peut pas servir de cible d'apprentissage."
        )

    samples = []
    for index, chromosome in enumerate(chromosomes):
        solution = ranking.solutions[index] if ranking.solutions else None
        if solution is None:
            continue  # niveau bas infaisable ou time limit : pas d'étiquette exploitable
        samples.append(_sample_from_solution(chromosome, solution, problem, generation=None))
    return samples


# ==========================================================================================
# Collecte PILOTÉE PAR UN RANKER (GA + CP-SAT, GA + greedy, ...)
# ==========================================================================================
def collect_labelled_allocations(
    problem: EnginePreparationProblem,
    driver: "Ranker",
    labeller: Optional["Ranker"] = None,
    pop_size: int = 40,
    ngen: int = 35,
    snapshot_every: int = 3,
    time_limit: Optional[float] = None,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Collecte des allocations en faisant tourner le VRAI GA, avec pilote et étiqueteur séparés.

    Deux rôles distincts, qu'il ne faut pas confondre :

      - le PILOTE (`driver`) décide quelles allocations sont visitées, donc la DISTRIBUTION du
        dataset ;
      - l'ÉTIQUETEUR (`labeller`) décide des valeurs cibles.

    Les séparer est ce qui permet de corriger un décalage de distribution : un surrogate
    destiné à remplacer CP-SAT dans la boucle du GA sera interrogé sur les allocations qu'un
    GA piloté par CP-SAT visite, qui ne sont pas celles qu'un GA piloté par greedy visite. Un
    dataset piloté ET étiqueté par greedy apprend au réseau à imiter greedy sur la mauvaise
    région de l'espace.

    Args:
        driver: ranker qui pilote la recherche. GreedyRanker pour beaucoup de générations,
            CpSatRanker pour une trajectoire fidèle à celle qu'on veut imiter -- mais ~1000x
            plus lent par génération, donc à utiliser avec `time_limit`.
        labeller: ranker qui produit les cibles. `None` réutilise les scores du pilote, ce qui
            évite de tout réordonnancer -- possible seulement si le pilote produit de vrais
            scores (un ranker siamois n'en produit pas).
        time_limit: budget en secondes pour la boucle. Indispensable dès que le pilote est
            CP-SAT, sinon `ngen` générations peuvent prendre des heures.

    Returns:
        Liste de {"alloc", "cost", "residual", "analytic"}, dédupliquée sur l'allocation.
    """
    from part_preparation.solvers.bilevel.ga.solver import BilevelGASolverD1

    if labeller is None and not _produces_true_scores(driver, problem):
        raise ValueError(
            f"Le pilote {driver.name!r} ne produit pas de scores réels : il ne peut pas servir "
            "d'étiqueteur. Passer un `labeller` explicite."
        )

    snapshots: List[List[int]] = []
    reused: List[Dict[str, Any]] = []

    def on_generation(generation, pool, ranking):
        if generation % snapshot_every != 0:
            return
        if labeller is None:
            # Le pilote a déjà ordonnancé toute la population : réutiliser ses solutions
            # évite de payer une seconde fois le niveau bas (décisif quand c'est CP-SAT).
            for index, chromosome in enumerate(pool):
                solution = ranking.solutions[index] if ranking.solutions else None
                if solution is None:
                    continue
                reused.append(
                    _sample_from_solution(chromosome, solution, problem, generation)
                )
        else:
            snapshots.extend((generation, list(chromosome)) for chromosome in pool)

    BilevelGASolverD1(
        problem=problem,
        ranker=driver,
        pop_size=pop_size,
        ngen=ngen,
        seed=seed,
        time_limit=time_limit,
        referee=driver if labeller is None else labeller,
        verbose=False,
    ).solve(on_generation=on_generation)

    if labeller is None:
        return reused

    # Étiquetage en UN SEUL appel sur toute la collecte : un ranker peut alors amortir son
    # coût sur le lot (le siamois fait un forward groupé, CP-SAT réutilise son process).
    ranking = labeller.rank([chromosome for _, chromosome in snapshots], problem, encoding.decode)
    if not ranking.has_true_scores:
        raise ValueError(
            f"L'étiqueteur {labeller.name!r} ne produit pas de scores réels : "
            "il ne peut pas servir de cible d'apprentissage."
        )

    samples = []
    for index, (generation, chromosome) in enumerate(snapshots):
        solution = ranking.solutions[index] if ranking.solutions else None
        if solution is None:
            continue
        samples.append(_sample_from_solution(chromosome, solution, problem, generation))
    return samples


def _produces_true_scores(ranker, problem) -> bool:
    """Un ranker sait-il chiffrer ? Testé sur une population d'un seul chromosome."""
    probe = ranker.rank([encoding.create_chromosome(problem)], problem, encoding.decode)
    return probe.has_true_scores


def _sample_from_solution(chromosome, solution, problem, generation: int) -> Dict[str, Any]:
    """Un échantillon (allocation plate + coût décomposé) à partir d'une solution ordonnancée."""
    allocation = encoding.decode(chromosome, problem)
    kpis = problem.evaluate(solution)
    return {
        # Génération à laquelle ce chromosome a été observé. C'est un indicateur de MATURITÉ
        # de la recherche : les premières générations sont diverses et médiocres, les dernières
        # resserrées et bonnes. Le conserver permet de stratifier train/validation par
        # génération, donc de vérifier que le réseau discrimine encore quand la population a
        # convergé -- le régime difficile, et le seul qui compte en fin de GA.
        "generation": generation,
        "alloc": {
            engine_id: {"msn_id": data["msn_id"], "factory_id": data["factory_id"]}
            for engine_id, data in allocation.items()
        },
        "cost": aggregate_score(kpis),
        "residual": residual_score(kpis),
        "analytic": analytic_score(kpis),
    }


# ==========================================================================================
# Génération complète, parallélisée sur les réplicats
# ==========================================================================================
def build_ranker(name: str, cpsat_time_limit: float = 5.0):
    """
    Fabrique un ranker par son nom. Import local : rankers.py ne dépend pas de torch.

    `num_workers=1` sur CpSatRanker parce que CE module parallélise déjà, un process par
    réplicat. Sans ça les deux niveaux se multiplient : mesuré à 8 process x 22 threads pour
    24 cœurs, soit chaque sous-solveur CP-SAT sur un septième de cœur. Le plafond étant en
    temps horloge, le run ne dure pas plus longtemps -- il produit des labels plus lâches, et
    rien dans la sortie ne le dit.
    """
    from part_preparation.solvers.bilevel.ga.rankers import (
        CpSatExactRanker, CpSatRanker, GreedyRanker,
    )
    if name == "greedy":
        return GreedyRanker()
    if name == "cpsat":
        return CpSatRanker(time_limit=cpsat_time_limit, num_workers=1)
    if name == "cpsat_exact":
        return CpSatExactRanker(time_limit=600.0)
    raise ValueError(f"Ranker inconnu : {name!r}")


def _samples_for_replicate(task):
    """
    Produit les échantillons d'UN réplicat. Fonction au niveau module (donc picklable) pour
    être distribuée sur un pool de process : les réplicats sont indépendants et
    instances.build() réensemence le RNG, donc chaque worker rejoue exactement la même
    trajectoire qu'en séquentiel -- la parallélisation ne change pas le dataset, seulement le
    temps de calcul.

    Le nom de l'instance transite par l'ARGUMENT et non par une globale : une globale posée
    dans le parent n'atteint les workers que sous `fork` (Linux), pas sous `spawn`. On ne
    transmet que le nom, jamais le problem lui-même, qui serait picklé à chaque tâche --
    instances.build() étant déterministe, c'est strictement équivalent et bien moins coûteux.
    """
    (instance_name, replicate_id, mode, driver_name, labeller_name,
     cpsat_time_limit, budget, nb_allocations, perturbation_steps) = task
    problem = instances.build(instance_name, replicate_id=replicate_id)

    if mode == "random":
        pool = collect_random_allocations(
            problem,
            labeller=build_ranker(labeller_name or "greedy", cpsat_time_limit),
            nb_allocations=nb_allocations,
            perturbation_steps=perturbation_steps,
            seed=replicate_id,
        )
    else:
        pool = collect_labelled_allocations(
            problem,
            driver=build_ranker(driver_name, cpsat_time_limit),
            labeller=None if labeller_name is None else build_ranker(labeller_name, cpsat_time_limit),
            time_limit=budget,
            seed=replicate_id,
        )
    return replicate_id, len(pool), deduplicate(pool, replicate_id, instance_name)


def generate_full_dataset(output_file, instance_name=instances.DEFAULT_INSTANCE,
                          nb_replicates: int = 100, nb_workers: Optional[int] = None,
                          mode: str = "ga", driver: str = "greedy",
                          labeller: Optional[str] = None,
                          cpsat_time_limit: float = 5.0,
                          budget_seconds: Optional[float] = None,
                          nb_allocations: int = 30,
                          perturbation_steps: int = 0) -> int:
    """
    Écrit le dataset au format JSON Lines (un échantillon compact par ligne) plutôt qu'un
    unique tableau JSON indenté : on peut alors écrire au fil de l'eau et relire ligne par
    ligne, sans jamais matérialiser les ~450 Mo en mémoire d'un coup.

    UN SEUL NOM D'INSTANCE par dataset -- la diversité vient des RÉPLICATS. Ce n'est pas une
    limitation arbitraire : le vecteur de features est de taille fixe
    (NB_FEATURES_PER_ENGINE x nb_moteurs + NB_GLOBAL_FEATURES, cf. surrogate.features), donc
    mélanger `small` (39 moteurs) et `medium` (162) produirait des entrées de tailles
    différentes que le réseau ne peut pas consommer. Les réplicats d'un même nom ont, eux, un
    nombre de moteurs constant et des dates différentes -- exactement la variation utile.

    Args:
        mode: \"ga\" déroule une trajectoire de GA (populations réalistes, coûteux) ;
            \"random\" tire `nb_allocations` allocations indépendantes par instance
            (bien moins cher, mais toutes de qualité médiocre et semblable).
        driver: mode \"ga\" seulement -- ranker qui pilote. Détermine la DISTRIBUTION.
        labeller: ranker qui étiquette. En mode \"ga\", `None` réutilise les scores du pilote.
        budget_seconds: budget par réplicat en mode \"ga\". Indispensable si driver=\"cpsat\".
        nb_allocations: mode \"random\" seulement -- allocations tirées par instance.
        perturbation_steps: mode \"random\" -- si > 0, une allocation sur deux est un VOISIN
            d'une allocation déjà tirée, ce qui fournit les paires « proches » que le tirage
            uniforme ne produit jamais.

    Returns:
        Le nombre total d'échantillons écrits.
    """
    if nb_workers is None:
        nb_workers = max(1, min(nb_replicates, (os.cpu_count() or 2) - 2))

    started = time.time()
    detail = (f"{nb_allocations} allocations aleatoires/instance" if mode == "random"
              else f"pilote={driver}")
    print(f"Génération de {nb_replicates} réplicats de '{instance_name}' sur {nb_workers} "
          f"process | mode={mode} | {detail} | "
          f"étiqueteur={labeller or (driver + ' (réutilisé)')}", flush=True)

    total_samples, done = 0, 0
    with open(output_file, "w", encoding="utf-8") as f:
        with mp.Pool(processes=nb_workers) as process_pool:
            for replicate_id, pool_size, samples in process_pool.imap_unordered(
                _samples_for_replicate,
                [(instance_name, replicate_id, mode, driver, labeller, cpsat_time_limit,
                  budget_seconds, nb_allocations, perturbation_steps)
                 for replicate_id in range(nb_replicates)],
            ):
                for sample in samples:
                    f.write(json.dumps(sample, separators=(",", ":")) + "\n")
                total_samples += len(samples)
                done += 1
                elapsed = time.time() - started
                eta = (elapsed / done) * (nb_replicates - done)
                print(f"  [{done}/{nb_replicates}] réplicat {replicate_id}: "
                      f"{len(samples)} échantillons uniques / {pool_size} collectés "
                      f"(écoulé: {elapsed:.0f}s, ETA: {eta:.0f}s)", flush=True)

    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\nTerminé en {time.time() - started:.1f}s — {total_samples} échantillons, "
          f"{size_mb:.0f} Mo\n   -> {output_file}", flush=True)
    return total_samples


def _cli():
    """
    CLI du module : `python -m part_preparation.solvers.bilevel.ga.surrogate.dataset_builder`.

    Elle vit ici et non dans un script d'exemple : les définitions d'instances viennent de
    part_preparation.instances, il ne restait donc dans le script qu'un argparse qui n'avait
    aucune raison d'être ailleurs que dans le module qu'il pilote.
    """
    import argparse
    from pathlib import Path

    from part_preparation.solvers.bilevel.ga.surrogate.paths import (
        SAMPLES_DATASET, ensure_dirs,
    )

    parser = argparse.ArgumentParser(
        description="Génère le dataset allocation -> coût des surrogates de D1."
    )
    parser.add_argument("--instance", choices=sorted(instances.INSTANCES),
                        default=instances.DEFAULT_INSTANCE)
    parser.add_argument("--replicates", type=int, default=100,
                        help="Nombre de réplicats (mêmes paramètres, dates différentes)")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--mode", choices=["ga", "random"], default="ga",
                        help="ga = trajectoire de GA ; random = allocations indépendantes")
    parser.add_argument("--allocations", type=int, default=30,
                        help="Mode random : allocations tirées par instance")
    parser.add_argument("--perturbation-steps", type=int, default=0,
                        help="Mode random : > 0 produit des paires proches (voisins mutés)")
    parser.add_argument("--driver", choices=["greedy", "cpsat"], default="greedy",
                        help="Ranker qui pilote le GA (détermine la distribution)")
    parser.add_argument("--labeller", choices=["greedy", "cpsat", "cpsat_exact"], default=None,
                        help="Ranker qui étiquette (défaut : réutilise les scores du pilote)")
    parser.add_argument("--cpsat-time-limit", type=float, default=5.0)
    parser.add_argument("--budget", type=float, default=None,
                        help="Secondes par réplicat. Obligatoire si --driver cpsat.")
    parser.add_argument("--output", type=Path, default=SAMPLES_DATASET)
    args = parser.parse_args()

    if args.mode == "ga" and args.driver == "cpsat" and args.budget is None:
        parser.error("--driver cpsat exige --budget : sans plafond, une seule génération "
                     "peut prendre des minutes et le run ne se termine jamais.")

    ensure_dirs()
    generate_full_dataset(
        output_file=str(args.output),
        instance_name=args.instance,
        nb_replicates=args.replicates,
        nb_workers=args.workers,
        mode=args.mode,
        driver=args.driver,
        labeller=args.labeller,
        cpsat_time_limit=args.cpsat_time_limit,
        budget_seconds=args.budget,
        nb_allocations=args.allocations,
        perturbation_steps=args.perturbation_steps,
    )


if __name__ == "__main__":
    _cli()
