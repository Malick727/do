"""
Rankers de la DÉCOMPOSITION 1 : l'allocation est figée en haut, le niveau bas ordonnance.

Chaque ranker est libre de sa méthode -- évaluer un par un puis trier (greedy, CP-SAT), ou
classer directement par passe groupée (siamois). Aucune classe intermédiaire ne l'y contraint :
le contrat commun se limite à `rank(chromosomes, problem, decode) -> Ranking`.

    GreedyRanker       heuristique de liste, ~0.3 ms/individu
    CpSatRanker        modèle CP-SAT réduit, ~0.2 s/individu
    CpSatExactRanker   modèle complet figé par fix_allocation, 20-600 s/individu
    SiameseRanker      forward groupé du MLP, aucun planning construit
    GnnSiameseRanker   idem, encodeur GNN + recalage logits -> euros

Comparer les approches de niveau bas revient à changer l'objet passé au GA, tout le reste
étant tenu constant.
"""
from typing import Dict
import time

from part_preparation.solvers.bilevel.ga.ranking import Ranker, Ranking, order_by_score
from part_preparation.solvers.bilevel.lower import (
    INFEASIBLE_SCORE,
    CPSAT_EVAL_TIME_LIMIT,
    analytic_score_from_allocation,
    schedule_allocation_with_cpsat,
    score_solution,
    simulate_greedy_from_allocation,
    solve_allocation_with_full_cpsat,
)


class GreedyRanker(Ranker):
    """
    Ordonnance chaque allocation par heuristique de liste, puis trie sur le score réel.

    Le moins cher des rankers exacts (~0.3 ms par individu) : c'est celui qui permet de faire
    tourner le GA sur des milliers d'évaluations. Son score est une BORNE SUPÉRIEURE de la
    valeur optimale d'ordonnancement -- un bon planning greedy ne prouve pas que l'allocation
    est bonne, seulement qu'elle admet un planning à ce coût.
    """

    name = "greedy"

    def rank(self, chromosomes, problem, decode) -> Ranking:
        started = time.perf_counter()
        scores, solutions = [], []

        for chromosome in chromosomes:
            allocation = decode(chromosome, problem)
            solution = simulate_greedy_from_allocation(allocation, problem)
            solutions.append(solution)
            scores.append(score_solution(solution, problem))

        return Ranking(
            order=order_by_score(scores),
            scores=scores,
            solutions=solutions,
            meta={"ranker": self.name, "wall_time": time.perf_counter() - started},
        )


class CpSatRanker(Ranker):
    """
    Ordonnance chaque allocation avec le modèle CP-SAT réduit (capacité + retards), puis trie.

    `meta["pct_proven_optimal"]` est la statistique à surveiller : si elle chute, le plafond
    de temps est trop court pour la taille d'instance, les scores ne sont plus que des bornes
    supérieures, et le classement devient partiellement du bruit -- deux allocations peuvent
    s'inverser selon la chance du solveur.
    """

    name = "cpsat"

    def __init__(self, time_limit: float = CPSAT_EVAL_TIME_LIMIT, num_workers=None):
        """
        Args:
            num_workers: parallélisme interne de CP-SAT. `None` = défaut d'OR-Tools (portfolio
                d'une vingtaine de threads), qui convient à la boucle du GA puisqu'elle tourne
                dans un process unique. Mettre 1 quand l'APPELANT est déjà parallélisé, sans
                quoi les deux niveaux se multiplient et chaque évaluation se retrouve à un
                fragment de cœur -- cf. lower.schedule_allocation_with_cpsat.
        """
        self.time_limit = time_limit
        self.num_workers = num_workers

    def rank(self, chromosomes, problem, decode) -> Ranking:
        started = time.perf_counter()
        stats: Dict[str, int] = {}
        scores, solutions = [], []

        for chromosome in chromosomes:
            allocation = decode(chromosome, problem)
            solution = schedule_allocation_with_cpsat(
                allocation, problem, time_limit=self.time_limit, stats=stats,
                num_workers=self.num_workers,
            )
            solutions.append(solution)
            # Le score passe par problem.evaluate() comme pour le greedy, et non par
            # l'objectif interne de CP-SAT : les deux rankers sont ainsi mesurés sur
            # exactement la même fonction, sinon leur comparaison n'aurait pas de sens.
            scores.append(
                INFEASIBLE_SCORE if solution is None else score_solution(solution, problem)
            )

        calls = stats.get("calls", 0)
        return Ranking(
            order=order_by_score(scores),
            scores=scores,
            solutions=solutions,
            meta={
                "ranker": self.name,
                "wall_time": time.perf_counter() - started,
                "cpsat_calls": calls,
                "pct_proven_optimal": stats.get("optimal", 0) / calls if calls else None,
            },
        )


class CpSatExactRanker(Ranker):
    """
    Classement de RÉFÉRENCE : modèle CP-SAT complet, allocation figée par fix_allocation.

    20 à 600 s par individu, donc inutilisable dans la boucle du GA. Deux usages : arbitrer
    les finalistes en post-traitement, et servir de vérité terrain pour mesurer la qualité de
    classement des rankers approchés (corrélation de rangs contre celui-ci).
    """

    name = "cpsat_exact"

    def __init__(self, time_limit: float = 600.0):
        self.time_limit = time_limit

    def rank(self, chromosomes, problem, decode) -> Ranking:
        started = time.perf_counter()
        scores, solutions, nb_optimal = [], [], 0

        for chromosome in chromosomes:
            allocation = decode(chromosome, problem)
            solution, is_optimal = solve_allocation_with_full_cpsat(
                allocation, problem, time_limit=self.time_limit,
            )
            nb_optimal += int(is_optimal)
            solutions.append(solution)
            scores.append(
                INFEASIBLE_SCORE if solution is None else score_solution(solution, problem)
            )

        return Ranking(
            order=order_by_score(scores),
            scores=scores,
            solutions=solutions,
            meta={
                "ranker": self.name,
                "wall_time": time.perf_counter() - started,
                "pct_proven_optimal": nb_optimal / len(chromosomes) if chromosomes else None,
            },
        )


class SiameseRanker(Ranker):
    """
    Classe la population par une passe GROUPÉE du réseau, sans jamais ordonnancer.

    Deux points de conception :

    1. Le réseau ne prédit que le RÉSIDU (la part du score qui dépend du planning). La part
       analytique -- transport et MSN non couverts, ~98% du score -- est recalculée
       EXACTEMENT ici. Classer sur le seul résidu serait faux : deux allocations peuvent avoir
       des résidus voisins et des parts analytiques très différentes.

    2. Le forward porte sur la population entière, en un seul batch. Ce n'est pas qu'une
       optimisation : `SiameseAllocationNet.scorer` contient un BatchNorm1d, et l'évaluer
       individu par individu le nourrit avec des batchs de taille 1. `model.eval()` est donc
       posé explicitement, pour que la normalisation utilise les statistiques courantes.

    `scores` reste None : les valeurs produites sont des PRÉDICTIONS, pas des valeurs
    d'objectif. Les exposer comme des scores réels laisserait croire qu'on peut les comparer
    à ceux d'un GreedyRanker. Elles sont rangées dans meta["predicted"].
    """

    name = "siamese"

    def __init__(self, model):
        self.model = model

    def rank(self, chromosomes, problem, decode) -> Ranking:
        # Import local et non en tête de module : c'est le seul point de rankers.py qui
        # dépend de PyTorch, et un GA en mode greedy n'a aucune raison de le charger.
        import torch
        from part_preparation.solvers.bilevel.ga.surrogate.features import extract_features

        started = time.perf_counter()
        allocations = [decode(chromosome, problem) for chromosome in chromosomes]
        features = torch.tensor(
            [extract_features(allocation, problem) for allocation in allocations],
            dtype=torch.float32,
        )

        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                residuals = self.model.predict_residual(features).squeeze(-1).tolist()
        finally:
            self.model.train(was_training)

        predicted = [
            analytic_score_from_allocation(allocation, problem) + residual
            for allocation, residual in zip(allocations, residuals)
        ]

        return Ranking(
            order=order_by_score(predicted),
            scores=None,  # prédictions, pas des valeurs d'objectif : cf. docstring
            solutions=None,  # aucun planning n'est construit
            meta={
                "ranker": self.name,
                "wall_time": time.perf_counter() - started,
                "predicted": predicted,
            },
        )


class GnnSiameseRanker(Ranker):
    """
    Comme SiameseRanker, mais l'encodeur est un GNN et le résidu passe par un RECALAGE.

    Deux différences avec SiameseRanker, l'une de forme et l'autre de fond.

    De forme : l'entrée n'est plus un vecteur plat de features mais un graphe hétérogène par
    allocation (cf. surrogate.gnn.build_graph). Le vecteur plat donnait au moteur n°7 ses
    propres poids, indépendants de ceux du n°8, et l'emplacement i ne désignait pas le même
    moteur d'une instance à l'autre : 96% d'accuracy d'entraînement pour 51% en validation sur
    des instances non vues, c'est-à-dire le hasard. Le GNN partage ses poids entre tous les
    nœuds et fait circuler la congestion le long des arêtes.

    De fond : `SiameseAllocationNet.predict_residual` rend des euros, parce qu'il a une tête de
    régression entraînée comme telle. Un siamois GNN entraîné en BCE sur des paires n'a de
    contrainte que sur la DIFFÉRENCE de ses scores, en unités de logit. Additionner ces logits
    à la part analytique -- qui est en euros et pèse ~98% du total -- donnerait un classement
    arbitraire. `calibration` porte la pente qui convertit les uns dans les autres, estimée
    hors ligne sur des allocations étiquetées (cf. surrogate.calibration). L'ordonnée à
    l'origine ne change aucun rang, tous les individus d'une population appartenant à la même
    instance ; la pente, elle, fixe l'arbitrage entre « transporte moins » et « ordonnance
    mieux », et c'est la seule quantité que ce ranker exige d'être juste.

    Comme pour SiameseRanker, `scores` reste None : ce sont des prédictions et non des valeurs
    d'objectif, donc une comparaison d'approches doit passer par un arbitre commun.
    """

    name = "gnn_siamese"

    def __init__(self, model, calibration, batch_size: int = 16):
        """
        Args:
            model: un `SiameseGNN` ou le `SchedulingGNN` nu -- surrogate.calibration.raw_scores
                accepte les deux.
            calibration: `surrogate.calibration.Calibration`. Obligatoire et sans valeur par
                défaut : un défaut « identité » ferait tourner le GA sur des logits additionnés
                à des euros, sans rien lever et sans que le tableau de comparaison ne montre
                autre chose qu'une méthode médiocre.
            batch_size: nombre de graphes par forward. La population entière tiendrait en un
                lot pour `medium`, mais un graphe xlarge pèse ~0,2 Mo et une population de 60
                en pleine mémoire d'activations ne passe pas partout.
        """
        self.model = model
        self.calibration = calibration
        self.batch_size = batch_size

    @classmethod
    def from_files(cls, weights_path, calibration_path=None, hidden: int = 64,
                   nb_layers: int = 3, batch_size: int = 16) -> "GnnSiameseRanker":
        """
        Charge poids et recalage depuis le disque.

        `calibration_path` vaut par défaut `<weights>.calib.json`, la convention qu'écrit le CLI
        de surrogate.calibration. Son absence est une ERREUR et non un cas dégradé : des poids
        sans recalage sont utilisables pour mesurer une accuracy par paires, pas pour classer
        une population.
        """
        import torch

        from part_preparation.solvers.bilevel.ga.surrogate.calibration import Calibration
        from part_preparation.solvers.bilevel.ga.surrogate.gnn_siamese import SiameseGNN

        calibration_path = calibration_path or Calibration.path_for(weights_path)
        model = SiameseGNN(hidden=hidden, nb_layers=nb_layers)
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        return cls(model, Calibration.load(calibration_path), batch_size=batch_size)

    def rank(self, chromosomes, problem, decode) -> Ranking:
        # Import local, même raison que dans SiameseRanker : rankers.py doit rester importable
        # sans PyTorch, pour qu'un GA en mode greedy ne le charge pas pour rien.
        from part_preparation.solvers.bilevel.ga.surrogate.calibration import (
            _encoder_of, raw_scores,
        )
        from part_preparation.solvers.bilevel.ga.surrogate.gnn import build_graph

        started = time.perf_counter()
        allocations = [decode(chromosome, problem) for chromosome in chromosomes]
        graphs = [build_graph(problem, allocation) for allocation in allocations]

        scores = raw_scores(_encoder_of(self.model), graphs, batch_size=self.batch_size)
        residuals = self.calibration.to_euros(scores).tolist()

        predicted = [
            analytic_score_from_allocation(allocation, problem) + residual
            for allocation, residual in zip(allocations, residuals)
        ]

        return Ranking(
            order=order_by_score(predicted),
            scores=None,  # prédictions, pas des valeurs d'objectif
            solutions=None,  # aucun planning n'est construit
            meta={
                "ranker": self.name,
                "wall_time": time.perf_counter() - started,
                "predicted": predicted,
                "predicted_residuals": residuals,
                "calibration_slope": self.calibration.slope,
            },
        )
