
"""
Contrat PARTAGÉ par toutes les décompositions : le `Ranker`.

C'est l'unique abstraction de l'architecture. Le GA lui passe une POPULATION de chromosomes,
il la rend classée du meilleur au pire. Cette interface plutôt qu'un « évalue-moi ce
chromosome », pour une raison de fond : toutes les méthodes ne savent pas produire un score
absolu. Un réseau siamois est entraîné à COMPARER deux allocations ; lui demander de chiffrer
une allocation isolée lui demande davantage que ce qu'il sait faire, alors que classer une
population est exactement sa sortie naturelle.

Ce module ne contient QUE le contrat. Les rankers concrets dépendent de ce que le niveau
supérieur a figé, donc de la décomposition, et vivent dans le sous-package correspondant
(bilevel.ga.rankers pour la décomposition 1). Ce qui ne change pas d'une décomposition à
l'autre -- la signature, la sémantique de `order`, la distinction score réel / prédiction --
est ici, et une seule fois.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from part_preparation.problem import EnginePreparationProblem, EnginePreparationSolution

# Un décodeur transforme un chromosome en allocation. Il appartient au GA, donc à l'ENCODAGE
# du niveau supérieur : le ranker le reçoit en paramètre et reste indépendant de la
# représentation. C'est ce qui permet au même ranker de servir plusieurs encodages.
Decoder = Callable[[Sequence[int], EnginePreparationProblem], Dict[str, Dict[str, Any]]]


@dataclass
class Ranking:
    """
    Sortie d'un ranker.

    `order` est la seule information que le GA utilise, et il l'utilise de la même façon quel
    que soit le ranker : la fitness d'un individu est sa POSITION dans le classement. Toutes
    les méthodes sont ainsi mises sur le même pied, y compris celles qui ne savent pas chiffrer.

    `scores` porte les VRAIES valeurs d'objectif quand le ranker en dispose, et vaut None
    sinon. Ce n'est pas pour le GA mais pour la comparaison d'approches : sans valeurs réelles,
    deux rankers ne sont pas sur la même échelle et leurs résultats doivent passer par un
    arbitre commun avant d'être comparés. `scores is None` est donc exactement le signal
    « cette approche a besoin d'un arbitre ».
    """
    order: List[int]  # indices dans la population, du MEILLEUR au PIRE
    scores: Optional[List[float]] = None  # aligné sur la POPULATION, pas sur order
    solutions: Optional[List[Optional[EnginePreparationSolution]]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_true_scores(self) -> bool:
        return self.scores is not None


class Ranker(ABC):
    """Classe une population de chromosomes du meilleur au pire."""

    name: str = "ranker"

    @abstractmethod
    def rank(
        self,
        chromosomes: Sequence[Sequence[int]],
        problem: EnginePreparationProblem,
        decode: Decoder,
    ) -> Ranking:
        """
        Args:
            chromosomes: la population à classer.
            problem: l'instance.
            decode: chromosome -> allocation, fourni par le GA.
        """
        ...


def order_by_score(scores: Sequence[float]) -> List[int]:
    """Indices triés par score croissant (le problème est une MINIMISATION)."""
    return sorted(range(len(scores)), key=lambda i: scores[i])
