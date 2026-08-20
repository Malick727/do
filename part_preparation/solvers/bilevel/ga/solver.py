"""
GA du niveau SUPÉRIEUR pour la décomposition 1.

Le chromosome encode l'allocation (moteur -> MSN, moteur -> usine) ; le niveau inférieur
complète en ordonnançant. Le GA ne connaît du niveau bas qu'un objet `Ranker` : changer de
méthode de résolution du niveau bas, c'est changer cet objet, rien d'autre.

DEAP n'est plus utilisé. Deux raisons :

  1. `algorithms.eaSimple` évalue les individus UN PAR UN via toolbox.evaluate. Un ranker
     travaille sur la population entière -- c'est tout son intérêt -- donc la boucle
     générationnelle doit être écrite ici. Elle tient en une trentaine de lignes.
  2. Les opérateurs (cf. ga.encoding) sont de simples fonctions sur des listes d'entiers ;
     passer par `creator.create` et son état global n'apportait rien.

FITNESS = POSITION DANS LE CLASSEMENT, pour tous les rankers indifféremment. C'est ce qui met
un ranker qui chiffre (greedy, CP-SAT) et un ranker qui ne fait que comparer (siamois) sur
exactement le même pied, et donc ce qui rend la comparaison d'approches honnête : aucune
méthode ne bénéficie d'une information de fitness plus riche qu'une autre.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence
import random
import time

from part_preparation.problem import EnginePreparationProblem, EnginePreparationSolution
from part_preparation.solvers.bilevel.ga.ranking import Ranker
from part_preparation.solvers.bilevel.ga import encoding
from part_preparation.solvers.bilevel.ga.rankers import GreedyRanker


@dataclass
class GenerationRecord:
    """Trace d'une génération, pour tracer les courbes de convergence."""
    generation: int
    best_true_score: Optional[float]  # None si le ranker ne produit pas de score réel
    mean_true_score: Optional[float]
    rank_wall_time: float
    ranker_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GAResult:
    best_solution: Optional[EnginePreparationSolution]
    best_score: Optional[float]  # jugé par l'ARBITRE, pas par le ranker de la boucle
    best_chromosome: Optional[List[int]]
    archive: List[List[int]]
    history: List[GenerationRecord]
    total_wall_time: float

    @property
    def nb_generations(self) -> int:
        """Générations réellement effectuées -- peut être < ngen si le budget temps a coupé."""
        return len(self.history)


class BilevelGASolverD1:
    """
    GA sur l'allocation, niveau bas délégué à un `Ranker`.

    Args:
        ranker: résolution du niveau bas. C'est LE paramètre de comparaison d'approches.
        referee: ranker servant à départager l'archive finale. Il doit produire de vrais
            scores (`has_true_scores`), sinon on ne saurait pas sur quelle échelle comparer.
            Par défaut GreedyRanker -- à remplacer par CpSatExactRanker pour un arbitrage de
            référence, au prix du temps de calcul.
        archive_size: nombre de finalistes conservés d'une génération à l'autre.
        time_limit: budget en secondes pour la BOUCLE (l'arbitrage final n'y entre pas).
            La recherche s'arrête à la première génération qui dépasse ce budget, `ngen`
            servant alors de plafond de sécurité.

            C'est le paramètre à utiliser pour comparer des rankers entre eux. À `ngen` fixé
            on ne compare pas les méthodes mais leurs coûts unitaires : une génération de
            GreedyRanker coûte ~0.3 ms par individu, une de CpSatRanker ~0.2 s, soit trois
            ordres de grandeur. Le surrogate n'a d'intérêt que s'il convertit sa vitesse en
            générations supplémentaires, et cela ne se voit qu'à budget de TEMPS égal.
    """

    def __init__(
        self,
        problem: EnginePreparationProblem,
        ranker: Ranker,
        pop_size: int = 100,
        ngen: int = 100,
        cxpb: float = 0.7,
        mutpb: float = 0.2,
        indpb: float = 0.1,
        tournsize: int = 3,
        archive_size: int = 10,
        referee: Optional[Ranker] = None,
        seed: Optional[int] = None,
        time_limit: Optional[float] = None,
        verbose: bool = True,
    ):
        self.problem = problem
        self.ranker = ranker
        self.pop_size = pop_size
        self.ngen = ngen
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.indpb = indpb
        self.tournsize = tournsize
        self.archive_size = archive_size
        self.referee = referee if referee is not None else GreedyRanker()
        self.seed = seed
        self.time_limit = time_limit
        self.verbose = verbose

    # ----------------------------------------------------------------------------------
    # Opérateurs de variation
    # ----------------------------------------------------------------------------------
    def _vary(self, parents: Sequence[Sequence[int]]) -> List[List[int]]:
        """Croisement puis mutation, sur des COPIES (les parents restent dans l'archive)."""
        offspring = [list(parent) for parent in parents]

        for i in range(1, len(offspring), 2):
            if random.random() < self.cxpb:
                encoding.crossover_blocks(offspring[i - 1], offspring[i], self.problem)

        for i in range(len(offspring)):
            if random.random() < self.mutpb:
                encoding.mutate_blocks(offspring[i], self.indpb, self.problem)

        return offspring

    def _tournament(self, pool: Sequence[Sequence[int]], ranks: Sequence[int], k: int):
        """
        Sélection par tournoi sur le RANG (plus petit = meilleur).

        Le tournoi ne compare que des positions, jamais des valeurs : il fonctionne donc
        identiquement que le ranker sache chiffrer ou seulement comparer. C'est la propriété
        qui permet d'utiliser un surrogate purement comparatif sans changer la sélection.
        """
        chosen = []
        for _ in range(k):
            aspirants = random.sample(range(len(pool)), min(self.tournsize, len(pool)))
            winner = min(aspirants, key=lambda i: ranks[i])
            chosen.append(pool[winner])
        return chosen

    # ----------------------------------------------------------------------------------
    # Boucle principale
    # ----------------------------------------------------------------------------------
    def solve(self, on_generation: Optional[Callable[[int, List[List[int]], Any], None]] = None) -> GAResult:
        """
        Args:
            on_generation: appelé après chaque classement avec (generation, pool, ranking).
                `pool` est la population + l'archive, `ranking` le Ranking correspondant.
                Sert à observer la trajectoire sans la modifier -- c'est par là que la
                génération de dataset récupère des populations réellement visitées par le GA,
                plutôt que des allocations tirées au hasard.
        """
        if self.seed is not None:
            random.seed(self.seed)

        started = time.perf_counter()
        population = [encoding.create_chromosome(self.problem) for _ in range(self.pop_size)]
        archive: List[List[int]] = []
        history: List[GenerationRecord] = []

        for generation in range(self.ngen + 1):
            # L'archive est reclassée AVEC la population, en un seul appel. C'est
            # indispensable : la fitness étant une position, elle n'a de sens qu'à
            # l'intérieur d'un classement. Comparer le rang 0 de la génération 1 au rang 0
            # de la génération 40 ne veut rien dire, donc un HallOfFame classique serait
            # incorrect ici. En rejugeant tout ensemble, l'élitisme n'utilise que les
            # comparaisons du ranker lui-même -- aucun score absolu n'est requis, ce qui
            # laisse le siamois utilisable tel quel.
            pool = population + archive
            ranking = self.ranker.rank(pool, self.problem, encoding.decode)

            ranks = [0] * len(pool)
            for position, index in enumerate(ranking.order):
                ranks[index] = position

            archive = [list(pool[i]) for i in ranking.order[: self.archive_size]]

            history.append(GenerationRecord(
                generation=generation,
                best_true_score=(
                    min(ranking.scores) if ranking.has_true_scores else None
                ),
                mean_true_score=(
                    sum(ranking.scores) / len(ranking.scores) if ranking.has_true_scores else None
                ),
                rank_wall_time=ranking.meta.get("wall_time", 0.0),
                ranker_meta={k: v for k, v in ranking.meta.items() if k != "predicted"},
            ))

            if self.verbose:
                best = history[-1].best_true_score
                shown = f"{best:,.0f}" if best is not None else "n/a (comparateur seul)"
                print(f"  gen {generation:4d} | meilleur score réel : {shown} "
                      f"| classement : {history[-1].rank_wall_time:.2f}s")

            if on_generation is not None:
                on_generation(generation, pool, ranking)

            if generation == self.ngen:
                break

            if self.time_limit is not None and time.perf_counter() - started >= self.time_limit:
                if self.verbose:
                    print(f"  budget de {self.time_limit:.0f}s atteint "
                          f"apres {generation + 1} generations")
                break

            # Les parents sont tirés dans population + archive : l'élitisme est ainsi porté
            # par la sélection, sans recopie forcée des élites dans la descendance.
            parents = self._tournament(pool, ranks, self.pop_size)
            population = self._vary(parents)

        best_solution, best_score, best_chromosome = self._arbitrate(archive)

        return GAResult(
            best_solution=best_solution,
            best_score=best_score,
            best_chromosome=best_chromosome,
            archive=archive,
            history=history,
            total_wall_time=time.perf_counter() - started,
        )

    def _arbitrate(self, archive: List[List[int]]):
        """
        Départage les finalistes avec l'ARBITRE, et non avec le ranker de la boucle.

        Nécessaire dans les deux cas de figure, pour deux raisons différentes :
          - ranker purement comparatif (siamois) : il n'a produit aucune valeur, donc rien
            qui puisse être comparé aux résultats d'une autre approche ;
          - ranker approché (greedy) : sa valeur est une borne supérieure, et deux approches
            évaluées par des bornes différentes ne sont pas comparables entre elles.

        Faire passer toutes les approches par le MÊME arbitre est ce qui rend la comparaison
        valide -- c'est le seul point de l'architecture où les échelles se rejoignent.
        """
        if not archive:
            return None, None, None

        ranking = self.referee.rank(archive, self.problem, encoding.decode)
        if not ranking.has_true_scores:
            raise ValueError(
                f"L'arbitre {self.referee.name} ne produit pas de scores réels : "
                "il ne peut pas départager les finalistes sur une échelle commune."
            )

        best_index = ranking.order[0]
        best_solution = ranking.solutions[best_index] if ranking.solutions else None
        return best_solution, ranking.scores[best_index], archive[best_index]
