"""
Qualité de CLASSEMENT d'un ranker approché, mesurée contre un ranker de référence.

POURQUOI PAS L'ACCURACY PAR PAIRES. C'est la métrique que l'entraînement affiche, et elle
répond à une autre question. Elle est calculée sur des paires TIRÉES d'un dataset, dont les
deux membres viennent d'allocations échantillonnées sur tout le spectre de qualité : une bonne
part d'entre elles opposent une allocation excellente à une allocation médiocre, et les
départager ne demande presque rien. Ce que le GA demande, lui, c'est d'ordonner une POPULATION
-- soixante allocations issues des mêmes parents, donc voisines, et de plus en plus voisines à
mesure que la recherche converge. Un modèle à 90% d'accuracy par paires peut n'avoir aucune
corrélation de rangs sur une population de génération 50.

CE QUI EST MESURÉ. Trois quantités, qui ne disent pas la même chose :

    spearman     corrélation des rangs sur toute la population. La vue d'ensemble.
    kendall_tau  proportion de paires concordantes, remise à l'échelle [-1, 1]. Plus sévère que
                 Spearman sur les inversions locales, qui sont exactement celles qui comptent
                 quand la population a convergé.
    top1_regret  écart RELATIF, au sens du ranker de référence, entre l'individu que le
                 candidat place premier et le vrai meilleur. La seule métrique qui parle le
                 langage du problème : un ranker peut avoir un Spearman médiocre et un regret
                 nul (il brouille le classement des mauvais, mais reconnaît le bon), ou
                 l'inverse. La sélection du GA se soucie d'abord du haut du classement.

LE RANKER DE RÉFÉRENCE. `CpSatExactRanker` est la vérité terrain de la décomposition -- modèle
complet, allocation figée. Il coûte 20 à 600 s par individu, ce qui plafonne la population
mesurable à une dizaine d'individus. `CpSatRanker` est un compromis défendable pour explorer,
mais il n'est PAS une vérité terrain : son score est une borne supérieure, et sur les instances
où `pct_proven_optimal` chute, une part des inversions qu'il attribue au candidat sont les
siennes.

Usage:
    python -m part_preparation.solvers.bilevel.ga.rank_quality \\
        --instance medium --candidate greedy --reference cpsat_exact --pop-size 10
    python -m part_preparation.solvers.bilevel.ga.rank_quality \\
        --instance xlarge --candidate gnn_siamese \\
        --weights data/part_preparation/models/gnn_siamese_xlarge.pth --reference cpsat
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import random
import time

from part_preparation import instances
from part_preparation.problem import EnginePreparationProblem
from part_preparation.solvers.bilevel.ga.ranking import Ranker, Ranking
from part_preparation.solvers.bilevel.ga.encoding import create_chromosome, decode

from part_preparation.solvers.bilevel.lower import INFEASIBLE_SCORE


def comparable_scores(ranking: Ranking) -> List[float]:
    """
    Vecteur de scores aligné sur la population, quel que soit ce que le ranker sait produire.

    Un ranker qui chiffre remplit `scores` ; un surrogate ne remplit que `meta["predicted"]`.
    À défaut des deux, on retombe sur le RANG lui-même -- suffisant pour une corrélation de
    rangs, qui ne lit de toute façon que l'ordre, mais pas pour un regret, d'où le contrôle
    dans `top1_regret`.
    """
    if ranking.scores is not None:
        return list(ranking.scores)
    if "predicted" in ranking.meta:
        return list(ranking.meta["predicted"])

    positions = [0.0] * len(ranking.order)
    for rank, index in enumerate(ranking.order):
        positions[index] = float(rank)
    return positions


def _ranks(values: Sequence[float]) -> List[float]:
    """Rangs moyens (ex aequo partagés), convention nécessaire à un Spearman correct."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        shared = (start + stop) / 2.0
        for position in range(start, stop + 1):
            ranks[order[position]] = shared
        start = stop + 1
    return ranks


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    n = len(a)
    mean_a, mean_b = sum(a) / n, sum(b) / n
    cov = sum((x - mean_a) * (y - mean_b) for x, y in zip(a, b))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((y - mean_b) ** 2 for y in b)
    if var_a <= 0 or var_b <= 0:
        return float("nan")
    return cov / (var_a * var_b) ** 0.5


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    return _pearson(_ranks(a), _ranks(b))


def kendall_tau(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Tau-b : concordantes moins discordantes, normalisé en tenant compte des ex aequo.

    Implémentation en O(n²), délibérément. Les populations mesurées ici comptent au plus
    quelques dizaines d'individus, parce que le ranker de référence coûte des minutes par
    individu -- une version en O(n log n) optimiserait la seule partie gratuite du calcul.
    """
    concordant = discordant = ties_a = ties_b = 0
    n = len(a)
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 and db == 0:
                continue
            if da == 0:
                ties_a += 1
            elif db == 0:
                ties_b += 1
            elif (da > 0) == (db > 0):
                concordant += 1
            else:
                discordant += 1

    denominator = ((concordant + discordant + ties_a) *
                   (concordant + discordant + ties_b)) ** 0.5
    return (concordant - discordant) / denominator if denominator > 0 else float("nan")


@dataclass
class RankQuality:
    """
    Une mesure : un candidat, une référence, une population.

    `nb_infeasible` est rapporté à part et n'est pas une métrique de classement : si la
    référence juge une partie de la population infaisable, tous ces individus portent le même
    INFEASIBLE_SCORE et forment un bloc d'ex aequo qui gonfle artificiellement les corrélations
    -- le candidat n'a plus qu'à les mettre ensemble au fond, ce que n'importe quel modèle
    réussit. Une mesure avec beaucoup d'infaisables se lit avec méfiance.
    """
    candidate: str
    reference: str
    pop_size: int
    spearman: float
    kendall_tau: float
    top1_regret: Optional[float]
    candidate_time: float
    reference_time: float
    nb_infeasible: int
    nb_distinct: int = -1  # chromosomes DISTINCTS dans la population : cf. describe()
    nb_reference_ties: int = 0

    def describe(self) -> str:
        regret = "-" if self.top1_regret is None else f"{self.top1_regret:+.2%}"
        line = (f"{self.candidate:<14} vs {self.reference:<12} | n={self.pop_size:>3} "
                f"(distincts {self.nb_distinct:>3}) | "
                f"spearman {self.spearman:>6.3f} | tau {self.kendall_tau:>6.3f} | "
                f"regret@1 {regret:>8} | {self.candidate_time:>7.2f}s vs "
                f"{self.reference_time:>8.1f}s | infaisables {self.nb_infeasible}")

        # Une population effondrée rend la mesure VIDE, et rien d'autre ne le montrerait : les
        # clones reçoivent le même score, forment un bloc d'ex aequo que tout ranker « ordonne »
        # correctement, et la corrélation qui en sort ne mesure aucune compétence. Mesuré une
        # fois : 10 individus dont 2 distincts après 10 générations, d'où un Spearman de 0.522
        # qui ne disait rien. Le seuil est bas volontairement -- en dessous de la moitié
        # d'individus distincts, le tableau ne doit pas être lu comme un résultat.
        if 0 <= self.nb_distinct < max(2, self.pop_size // 2):
            line += f"\n  ⚠ population effondrée ({self.nb_distinct}/{self.pop_size} distincts) :"\
                    " corrélation non interprétable, augmenter --pop-size ou réduire"\
                    " --ga-generations"
        return line


def sample_population(problem: EnginePreparationProblem, pop_size: int, seed: int = 0,
                      source: str = "random", nb_generations: int = 10) -> List[List[int]]:
    """
    Args:
        source: `random` tire des chromosomes uniformes ; `ga` prend la population d'un GA
            après `nb_generations` générations.

            Les deux ne mesurent pas la même chose et il faut les deux. Une population
            aléatoire est FACILE à classer : ses individus diffèrent surtout par le nombre de
            MSN couverts, que la part analytique tranche exactement, sans que le surrogate ait
            à se prononcer. Une population de GA est le régime réel -- des individus voisins,
            de parts analytiques proches, où le classement se joue précisément sur le résidu
            que le surrogate prédit. Un modèle qui brille en `random` et s'effondre en `ga`
            n'apporte rien au GA, et seule la seconde mesure le montre.
    """
    random.seed(seed)
    if source == "random":
        return [create_chromosome(problem) for _ in range(pop_size)]

    if source != "ga":
        raise ValueError(f"source inconnue : {source!r}")

    from part_preparation.solvers.bilevel.ga.solver import BilevelGASolverD1
    from part_preparation.solvers.bilevel.ga.rankers import GreedyRanker

    # GreedyRanker et non le candidat : la population de test doit être la MÊME quel que soit le
    # ranker mesuré, sinon chacun serait jugé sur les individus que lui-même a produits et les
    # colonnes cesseraient d'être comparables.
    solver = BilevelGASolverD1(
        problem=problem, ranker=GreedyRanker(), pop_size=pop_size, ngen=nb_generations,
        seed=seed, verbose=False,
    )

    # `GAResult` ne porte que l'archive (les `archive_size` finalistes), pas la population : on
    # récupère celle-ci par le callback, qui reçoit pool + ranking à chaque génération. Le pool
    # inclut l'archive, d'où la troncature à pop_size -- et on prend la DERNIÈRE génération,
    # celle où les individus sont les plus proches les uns des autres, donc le régime où le
    # classement est le plus difficile.
    last_pool: List[List[int]] = []

    def capture(generation, pool, ranking):
        nonlocal last_pool
        last_pool = [list(individual) for individual in pool]

    solver.solve(on_generation=capture)
    if not last_pool:
        raise RuntimeError("Le GA n'a produit aucune génération.")
    return last_pool[:pop_size]


def measure(candidate: Ranker, reference: Ranker, problem: EnginePreparationProblem,
            chromosomes: Sequence[Sequence[int]],
            reference_ranking: Optional[Ranking] = None,
            reference_time: float = 0.0) -> RankQuality:
    """
    Classe la MÊME population avec les deux rankers et compare les ordres obtenus.

    Args:
        reference_ranking: classement de référence DÉJÀ calculé. À fournir dès qu'on mesure
            plusieurs candidats sur une même population, et pas seulement pour économiser les
            cinq minutes que coûte CP-SAT.

            La raison est de correctness : CP-SAT sous plafond de temps n'est pas
            déterministe, et deux appels sur la même population rendent des scores différents.
            Recalculer la référence par candidat, ce que faisait la version précédente, jugeait
            donc chaque candidat contre une vérité terrain DIFFÉRENTE -- deux colonnes du
            tableau cessaient d'être comparables, et l'écart entre elles mélangeait la qualité
            des candidats et la variance du solveur. C'est ce qui a produit deux `top1_regret`
            distincts (+0.01% et +0.06%) pour deux candidats dont les classements étaient en
            réalité IDENTIQUES.
    """
    started = time.perf_counter()
    candidate_ranking = candidate.rank(chromosomes, problem, decode)
    candidate_time = time.perf_counter() - started

    if reference_ranking is None:
        started = time.perf_counter()
        reference_ranking = reference.rank(chromosomes, problem, decode)
        reference_time = time.perf_counter() - started

    candidate_scores = comparable_scores(candidate_ranking)
    reference_scores = comparable_scores(reference_ranking)

    # Le regret n'a de sens que si la référence chiffre réellement : le calculer sur des rangs
    # rendrait un « pourcentage » qui n'est qu'une différence de positions.
    top1_regret = None
    if reference_ranking.scores is not None:
        best_reference = min(reference_scores)
        picked = candidate_ranking.order[0]
        if best_reference not in (0, INFEASIBLE_SCORE):
            top1_regret = (reference_scores[picked] - best_reference) / abs(best_reference)

    return RankQuality(
        candidate=candidate.name,
        reference=reference.name,
        pop_size=len(chromosomes),
        spearman=spearman(candidate_scores, reference_scores),
        kendall_tau=kendall_tau(candidate_scores, reference_scores),
        top1_regret=top1_regret,
        candidate_time=candidate_time,
        reference_time=reference_time,
        nb_infeasible=sum(1 for s in reference_scores if s >= INFEASIBLE_SCORE),
        nb_distinct=len({tuple(chromosome) for chromosome in chromosomes}),
        nb_reference_ties=len(reference_scores) - len(set(reference_scores)),
    )


def _build_ranker(name: str, args) -> Ranker:
    from part_preparation.solvers.bilevel.ga.rankers import (
        CpSatExactRanker, CpSatRanker, GnnSiameseRanker, GreedyRanker,
    )

    if name == "greedy":
        return GreedyRanker()
    if name == "cpsat":
        return CpSatRanker(time_limit=args.cpsat_time_limit, num_workers=args.workers)
    if name == "cpsat_exact":
        return CpSatExactRanker(time_limit=args.exact_time_limit)
    if name == "gnn_siamese":
        if args.weights is None:
            raise SystemExit("--weights est requis pour le candidat 'gnn_siamese'")
        return GnnSiameseRanker.from_files(
            args.weights, args.calibration, hidden=args.hidden, nb_layers=args.layers)
    raise SystemExit(f"ranker inconnu : {name}")


def _cli():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", choices=sorted(instances.INSTANCES), default="medium")
    parser.add_argument("--replicate", type=int, default=None,
                        help="Réplicat de l'instance. En mesurer un NON VU à l'entraînement "
                             "est le seul cas informatif pour un surrogate")
    parser.add_argument("--candidate", nargs="+", default=["greedy"],
                        choices=["greedy", "cpsat", "cpsat_exact", "gnn_siamese"])
    parser.add_argument("--reference", default="cpsat_exact",
                        choices=["greedy", "cpsat", "cpsat_exact"])
    parser.add_argument("--pop-size", type=int, default=10,
                        help="Petite par défaut : cpsat_exact coûte 20-600 s par individu")
    parser.add_argument("--population", choices=["random", "ga"], default="random")
    parser.add_argument("--ga-generations", type=int, default=10,
                        help="Générations du GA produisant la population de test (--population ga)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpsat-time-limit", type=float, default=10.0)
    parser.add_argument("--exact-time-limit", type=float, default=300.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--calibration", type=Path, default=None,
                        help="Défaut : <weights>.calib.json")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    args = parser.parse_args()

    problem = instances.build(args.instance, replicate_id=args.replicate)
    label = args.instance if args.replicate is None else f"{args.instance}#{args.replicate}"
    print(f"=== {label} : {len(problem.engines_list)} moteurs, {len(problem.msns_list)} MSN ===")

    chromosomes = sample_population(problem, args.pop_size, seed=args.seed,
                                    source=args.population,
                                    nb_generations=args.ga_generations)
    nb_distinct = len({tuple(chromosome) for chromosome in chromosomes})
    print(f"Population '{args.population}' de {len(chromosomes)} individus "
          f"({nb_distinct} distincts)\n")

    # La référence est calculée UNE FOIS pour tous les candidats : elle coûte des minutes, et
    # surtout CP-SAT sous plafond n'est pas déterministe -- la recalculer par candidat les
    # jugerait contre des vérités terrain différentes. Cf. docstring de `measure`.
    reference = _build_ranker(args.reference, args)
    print(f"Classement de référence par '{args.reference}'...", flush=True)
    started = time.perf_counter()
    reference_ranking = reference.rank(chromosomes, problem, decode)
    reference_time = time.perf_counter() - started
    print(f"   {reference_time:.1f}s\n", flush=True)

    for name in args.candidate:
        result = measure(_build_ranker(name, args), reference, problem, chromosomes,
                         reference_ranking=reference_ranking, reference_time=reference_time)
        print(result.describe(), flush=True)


if __name__ == "__main__":
    _cli()
