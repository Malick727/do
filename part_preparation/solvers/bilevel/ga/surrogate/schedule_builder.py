"""Dataset « génération de planning » : (problème + allocation) -> planning.

L'allocation est une ENTRÉE, le planning est la CIBLE. C'est le dataset d'un modèle génératif
de type DIFUSCO, appelé par le GA pour évaluer un chromosome : le modèle produit un planning,
et la fitness est le coût RÉEL de ce planning (problem.evaluate) -- pas une prédiction. Le GA
n'a donc jamais à faire confiance à une valeur estimée.

Étiquetage par le modèle CP-SAT RÉDUIT à 30s, deux choix mesurés :
  - réduit plutôt que complet : le modèle complet n'atteint jamais OPTIMAL et son incumbent
    reste MOINS bon que celui du réduit pour 6x le temps de calcul ;
  - 30s plutôt que 600s : mesuré sur 10 allocations, 600s n'apporte que +0.76% de coût et
    ne change AUCUN rang (accord par paires 100%). Vingt fois le temps pour rien.

Les allocations sont groupées par problème (ALLOCATIONS_PER_PROBLEM par instance) : les ~107s
de collecte GA sont ainsi amorties sur plusieurs allocations, et surtout on obtient des paires
d'allocations COMPARABLES entre elles -- deux allocations de problèmes différents ont des coûts
incomparables, ce qui interdirait toute évaluation par classement.
"""
import json
import multiprocessing as mp
import os
import random
import time

from part_preparation import instances
from part_preparation.problem import aggregate_score
from part_preparation.solvers.bilevel.lower import (
    INFEASIBLE_SCORE,
    schedule_allocation_with_cpsat,
)
from part_preparation.solvers.bilevel.ga.surrogate.dataset_builder import (
    collect_allocations_via_ga,
)

NB_PROBLEMS = 100
ALLOCATIONS_PER_PROBLEM = 10
TIME_LIMIT_S = 30  # plafond par résolution


def decode_flat_allocation(flat, problem):
    """Forme plate (ids) -> forme décodée (objets), attendue par les ordonnanceurs."""
    return {
        engine.id: {
            "msn_id": flat.get(engine.id, {}).get("msn_id"),
            "factory_id": flat.get(engine.id, {}).get("factory_id"),
            "engine_obj": engine,
            "msn_obj": problem.msns.get(flat.get(engine.id, {}).get("msn_id"))
            if flat.get(engine.id, {}).get("msn_id") else None,
            "factory_obj": problem.factories.get(flat.get(engine.id, {}).get("factory_id"))
            if flat.get(engine.id, {}).get("factory_id") else None,
        }
        for engine in problem.engines_list
    }


def pick_allocations(problem, rng, nb_allocations):
    """
    Tire plusieurs allocations dans la trajectoire du GA, réparties sur tout le spectre de
    qualité plutôt que concentrées sur les meilleures : le modèle devra planifier aussi bien
    les allocations médiocres que le GA lui soumettra en début de recherche que les bonnes.

    On échantillonne à intervalles réguliers dans le pool trié par coût, ce qui garantit une
    couverture du spectre là où un tirage uniforme laisserait des trous.
    """
    pool = [p for p in collect_allocations_via_ga(problem) if p["cost"] < INFEASIBLE_SCORE]
    if not pool:
        return []

    unique = {json.dumps(p["alloc"], sort_keys=True): p for p in pool}
    pool = sorted(unique.values(), key=lambda p: p["cost"])
    if len(pool) <= nb_allocations:
        chosen = pool
    else:
        step = len(pool) / nb_allocations
        chosen = [pool[min(int(i * step + rng.random() * step), len(pool) - 1)]
                  for i in range(nb_allocations)]

    return [decode_flat_allocation(p["alloc"], problem) for p in chosen]


def solve_one_problem(task):
    """
    Un problème -> ALLOCATIONS_PER_PROBLEM échantillons (allocation, planning).

    Groupé par problème plutôt que par allocation pour deux raisons : les ~107s de collecte GA
    sont amorties sur toutes les allocations du lot, et les échantillons d'un même problème sont
    COMPARABLES entre eux, ce qui est indispensable pour évaluer un classement.
    """
    instance_name, problem_id, nb_allocations, time_limit = task
    params = instances.get_params(instance_name)
    problem = instances.build(instance_name, replicate_id=problem_id)
    rng = random.Random(instances.instance_seed(problem_id, params))

    allocations = pick_allocations(problem, rng, nb_allocations)
    if not allocations:
        return problem_id, [], "aucune allocation valide"

    records = []
    for k, allocation in enumerate(allocations):
        start = time.perf_counter()
        solution = schedule_allocation_with_cpsat(
            allocation, problem, time_limit=time_limit,
            # 1 thread : CE module parallélise déjà, un process par problème. Laissé au défaut
            # d'OR-Tools, CP-SAT lancerait son portfolio d'une vingtaine de threads dans chaque
            # process, et les deux niveaux se multiplieraient. Le plafond étant en temps
            # horloge, la sur-souscription ne rallonge pas le run : elle dégrade chaque planning
            # sans que rien ne le signale. Cf. lower.schedule_allocation_with_cpsat.
            num_workers=1,
        )
        elapsed = time.perf_counter() - start
        if solution is None:
            continue

        kpis = problem.evaluate(solution)

        # Vecteurs alignés sur problem.engines_list : le MÊME ordre fixe que build_graph, pour
        # que l'entrée et la cible du réseau se correspondent index par index. None pour un
        # moteur non alloué (il n'a pas de préparation).
        prep_starts = [
            int(solution.prep_start_times[engine])
            if solution.prep_start_times.get(engine) is not None else None
            for engine in problem.engines_list
        ]

        records.append({
            "problem_id": problem_id,
            # Nom de l'instance dans CHAQUE enregistrement : l'entraînement doit reconstruire le
            # problem de chaque problem_id pour rebâtir le graphe, et sans ce champ il faut le
            # lui redonner en ligne de commande -- où se tromper de nom ne lève rien de visible.
            # Même correctif que dans dataset_builder.deduplicate.
            "instance": instance_name,
            "alloc_id": k,
            "alloc": {
                engine_id: {"msn_id": data["msn_id"], "factory_id": data["factory_id"]}
                for engine_id, data in allocation.items()
            },
            "prep_starts": prep_starts,
            # Étiquette de coût, avec EXACTEMENT la fonction utilisée par le GA -- laquelle est
            # désormais l'objectif minimisé par CP-SAT (cf. problem.AGGREGATED_OBJECTIVES).
            "score": aggregate_score(kpis),
            # Décomposition de `score` terme à terme, pour pouvoir n'apprendre que le résidu.
            # Sous l'objectif aligné, DEUX des trois termes non nuls sont calculables
            # analytiquement depuis la seule allocation, sans ordonnancer (cf.
            # bilevel.lower.exact_allocation_terms) : travel_cost et unallocated_msns. Seul
            # arrival_delays_engines dépend du planning, et c'est donc la seule cible qu'un
            # surrogate ait un intérêt à prédire.
            "arrival_delays_engines": kpis["arrival_delays_engines"],
            "travel_cost": kpis["travel_cost"],
            "unallocated_msns": kpis["unallocated_msns"],
            # Hors objectif (poids 0) mais conservés comme diagnostics.
            "storage_cost": kpis["storage_cost"],
            "delay_cost": kpis["delay_cost"],
            "solve_time_s": round(elapsed, 2),
        })

    return problem_id, records, None


def generate_schedule_dataset(output_file, instance_name=instances.DEFAULT_INSTANCE,
                              nb_problems=NB_PROBLEMS, nb_workers=None,
                              nb_allocations=ALLOCATIONS_PER_PROBLEM,
                              time_limit=TIME_LIMIT_S):
    """
    Le nom de l'instance transite par les ARGUMENTS des tâches, jamais par une globale : une
    globale posée dans le parent n'atteint les workers que sous `fork` (Linux), pas sous
    `spawn`. Même raison pour `nb_allocations` et `time_limit`, qui étaient des constantes de
    module -- donc non réglables sans éditer le fichier, alors que ce sont les deux curseurs qui
    fixent le coût du run et la densité des paires disponibles à l'entraînement (n allocations
    par problème donnent C(n, 2) paires).
    """
    if nb_workers is None:
        nb_workers = max(1, min(nb_problems, (os.cpu_count() or 2) - 2))

    start_time = time.time()
    total_expected = nb_problems * nb_allocations
    print(f"🚀 {nb_problems} problèmes x {nb_allocations} allocations "
          f"= {total_expected} échantillons | plafond {time_limit}s | "
          f"{nb_workers} workers...", flush=True)

    done = errors = 0
    scores = []

    with open(output_file, 'w', encoding='utf-8') as f:
        with mp.Pool(processes=nb_workers) as pool:
            for problem_id, records, error in pool.imap_unordered(
                solve_one_problem,
                [(instance_name, problem_id, nb_allocations, time_limit)
                 for problem_id in range(nb_problems)],
            ):
                done += 1
                if error is not None:
                    errors += 1
                    print(f"  [{done}/{nb_problems}] problème {problem_id}: ❌ {error}", flush=True)
                    continue

                for record in records:
                    f.write(json.dumps(record, separators=(',', ':')) + "\n")
                f.flush()
                scores.extend(r["score"] for r in records)

                elapsed = time.time() - start_time
                eta = (elapsed / done) * (nb_problems - done)
                span = (max(r["score"] for r in records) - min(r["score"] for r in records))
                print(f"  [{done}/{nb_problems}] problème {problem_id}: {len(records)} plannings | "
                      f"score {min(r['score'] for r in records):,.0f} .. "
                      f"{max(r['score'] for r in records):,.0f} € (étendue {span:,.0f}) | "
                      f"ETA {eta/60:.0f}min", flush=True)

    elapsed = time.time() - start_time
    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n🎉 Terminé en {elapsed/60:.1f} min — {len(scores)} échantillons, "
          f"{errors} problèmes en erreur, {size_mb:.0f} Mo", flush=True)
    if scores:
        print(f"   Score: min {min(scores):,.0f} € | moyenne {sum(scores)/len(scores):,.0f} € "
              f"| max {max(scores):,.0f} €", flush=True)
    print(f"   -> {output_file}", flush=True)


def _cli():
    """CLI : `python -m part_preparation.solvers.bilevel.ga.surrogate.schedule_builder`."""
    import argparse
    from pathlib import Path

    from part_preparation.solvers.bilevel.ga.surrogate.paths import (
        SCHEDULES_DATASET, ensure_dirs,
    )

    parser = argparse.ArgumentParser(
        description="Génère le dataset allocation -> planning du surrogate GNN."
    )
    parser.add_argument("--instance", choices=sorted(instances.INSTANCES),
                        default=instances.DEFAULT_INSTANCE)
    parser.add_argument("--problems", type=int, default=NB_PROBLEMS)
    parser.add_argument("--allocations", type=int, default=ALLOCATIONS_PER_PROBLEM,
                        help="Allocations par problème. n donne C(n,2) paires à l'entraînement "
                             "siamois : 10 -> 45, 30 -> 435")
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT_S,
                        help="Plafond CP-SAT par planning, en secondes")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output", type=Path, default=SCHEDULES_DATASET)
    args = parser.parse_args()

    ensure_dirs()
    generate_schedule_dataset(
        output_file=str(args.output),
        instance_name=args.instance,
        nb_problems=args.problems,
        nb_workers=args.workers,
        nb_allocations=args.allocations,
        time_limit=args.time_limit,
    )


if __name__ == "__main__":
    _cli()
