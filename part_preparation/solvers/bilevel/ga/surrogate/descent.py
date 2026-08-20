"""
Descente de gradient sur l'allocation, à travers le surrogate : une ALTERNATIVE au GA.

L'IDÉE. Le GA explore l'espace des allocations par variations aveugles -- croisement, mutation,
sélection -- et n'utilise du niveau bas que le CLASSEMENT qu'il en tire. Un surrogate
différentiable donne davantage : la DÉRIVÉE du coût par rapport à la décision. On peut alors
descendre directement, au lieu d'échantillonner. C'est une méthode concurrente du GA, jugée sur
le même budget de temps et par le même arbitre.

CE QU'ON DESCEND. Pas le résidu prédit seul, qui a un optimum trivial (ne rien allouer annule
tous les retards), mais la somme complète :

    coût(A, B) = surrogate_résidu(A, B) + transport(A, B) + 1e6 * MSN_non_couverts(A, B)

Le premier terme est le réseau, les deux autres sont exacts et calculés analytiquement
(cf. relaxation.expected_analytic). Les trois sont différentiables par rapport aux logits.

LE PIÈGE, ET IL EST STRUCTUREL. Optimiser une entrée contre un réseau gelé trouve les points où
le RÉSEAU SE TROMPE, pas ceux où le coût réel est bas -- c'est la fabrication d'exemples
adverses, à ceci près qu'on ne la souhaite pas. Le surrogate n'a vu que des allocations issues de
trajectoires de GA ; la descente s'en éloigne, et rien dans le modèle ne signale qu'il
extrapole. `DescentResult` rapporte donc TOUJOURS le coût prédit ET le coût réel : leur écart
est la mesure de l'exploitation, et c'est le seul chiffre qui dise si la méthode fonctionne.
Une descente qui améliore beaucoup le prédit et dégrade le réel n'a pas échoué à optimiser --
elle a réussi à tromper.

DEUX GARDE-FOUS, tous deux optionnels et désactivés par défaut pour que leur effet se mesure :
  - `trust_radius` pénalise l'écart aux logits de départ, ce qui garde la recherche près du
    domaine d'entraînement ;
  - `exclusivity_weight` pénalise les MSN revendiqués par plusieurs moteurs. Un softmax par
    moteur garantit « au plus un MSN par moteur », jamais l'inverse : sans ce terme, la descente
    obtient un coût prédit très bas en affectant trois moteurs au même MSN, et le décodage
    détruit ensuite ce gain en les redistribuant.
"""
from dataclasses import dataclass
from typing import Optional
import time

import torch

from part_preparation.solvers.bilevel.lower import (
    analytic_score_from_allocation, score_solution, simulate_greedy_from_allocation,
)
from part_preparation.solvers.bilevel.ga.surrogate.relaxation import (
    CandidateSet, build_relaxed_graph, decode_relaxed, expected_analytic, _probabilities,
)


@dataclass
class DescentResult:
    """
    Ce qu'une descente rapporte. Les deux colonnes qui comptent sont `start_true` et `true`.

    `predicted` est ce que le surrogate croit ; le comparer à `true` mesure l'exploitation du
    modèle, pas sa qualité. Un `predicted` très inférieur à `true` est le signal que la descente
    est sortie du domaine où le réseau a appris quelque chose.
    """
    start_true: float          # score réel de l'allocation de départ
    true: float                # score réel de l'allocation décodée
    predicted: float           # ce que le surrogate annonçait au point relaxé
    start_predicted: float
    nb_steps: int
    wall_time: float
    nb_allocated: int
    nb_allocated_start: int

    @property
    def true_gain(self) -> float:
        return (self.start_true - self.true) / abs(self.start_true) if self.start_true else 0.0

    @property
    def predicted_gain(self) -> float:
        return ((self.start_predicted - self.predicted) / abs(self.start_predicted)
                if self.start_predicted else 0.0)

    def describe(self) -> str:
        return (
            f"  départ  : réel {self.start_true:>14,.0f} | prédit {self.start_predicted:>14,.0f}"
            f" | {self.nb_allocated_start} moteurs alloués\n"
            f"  arrivée : réel {self.true:>14,.0f} | prédit {self.predicted:>14,.0f}"
            f" | {self.nb_allocated} moteurs alloués\n"
            f"  gain RÉEL     : {self.true_gain:+.2%}\n"
            f"  gain PRÉDIT   : {self.predicted_gain:+.2%}"
            f"   <- l'écart entre les deux mesure l'exploitation du surrogate\n"
            f"  {self.nb_steps} pas en {self.wall_time:.1f}s"
        )


def relaxed_objective(model, candidates, msn_logits, factory_logits, target_scale):
    """Coût total en euros au point relaxé : surrogate + analytique exact."""
    graph = build_relaxed_graph(candidates, msn_logits, factory_logits)
    residual = model(graph)["total_cost"][0] * target_scale
    return residual + expected_analytic(candidates, msn_logits, factory_logits), residual


def _exclusivity_penalty(candidates, msn_logits, factory_logits):
    """
    Pénalise la masse revendiquée au-delà de 1 sur chaque MSN.

    relu(sum_e A[e,m] - 1) et non le carré : on ne veut pénaliser que le DÉPASSEMENT, et
    laisser strictement gratuit tout ce qui est déjà réalisable. Un terme quadratique
    pousserait aussi les revendications légitimes vers zéro.
    """
    a_msn, _, _ = _probabilities(msn_logits, factory_logits)
    nb_engines, msn_cols = a_msn.shape

    cand_idx = torch.zeros((nb_engines, msn_cols), dtype=torch.long)
    for e, candidate_list in enumerate(candidates.msn_candidates):
        for k, msn_i in enumerate(candidate_list[:msn_cols]):
            cand_idx[e, k] = msn_i

    claimed = torch.zeros(len(candidates.msns)).index_add(
        0, cand_idx.reshape(-1), a_msn.reshape(-1))
    return torch.relu(claimed - 1.0).sum()


def descend(problem, model, target_scale, start_allocation=None, nb_steps: int = 200,
            lr: float = 0.1, nb_candidates: Optional[int] = 12, sharpness: float = 6.0,
            trust_radius: float = 0.0, exclusivity_weight: float = 0.0,
            evaluate=None, verbose: bool = True) -> DescentResult:
    """
    Args:
        start_allocation: allocation de départ. `None` part de l'uniforme, ce qui est le cas le
            plus défavorable : le surrogate n'a jamais vu d'allocation uniforme et ses
            prédictions y sont sans garantie. Partir d'une allocation existante (greedy, ou un
            individu de GA) garde la descente près du domaine d'entraînement.
        sharpness: concentration des logits initiaux. Élevée = on part très près du one-hot,
            donc d'un point où la relaxation coïncide avec le graphe dur, mais où les gradients
            sont écrasés par la saturation du softmax. 6.0 est un compromis : la probabilité du
            choix initial vaut ~1 - e^-6, et le gradient reste exploitable.
        evaluate: fonction allocation -> score réel. Par défaut le greedy, quasi gratuit. Passer
            un CP-SAT donne une mesure plus fidèle, au prix de 20-600 s par appel.

    L'optimiseur est Adam et non une descente simple : les trois termes de l'objectif diffèrent
    de six ordres de grandeur (1e6 par MSN découvert contre ~1 par jour de retard), donc un pas
    unique ne peut pas convenir aux deux. Adam normalise par coordonnée.
    """
    evaluate = evaluate or (lambda alloc: score_solution(
        simulate_greedy_from_allocation(alloc, problem), problem))

    started = time.perf_counter()
    candidates = CandidateSet(problem, nb_candidates=nb_candidates)
    msn_logits, factory_logits = candidates.initial_logits(start_allocation, sharpness=sharpness)

    # Les logits de départ sont conservés AVANT d'activer le gradient : ils servent de centre à
    # la région de confiance, et doivent rester constants.
    anchor_msn = msn_logits.clone()
    anchor_factory = factory_logits.clone()

    msn_logits = msn_logits.clone().requires_grad_(True)
    factory_logits = factory_logits.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([msn_logits, factory_logits], lr=lr)

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)  # les poids sont GELÉS : on optimise l'entrée

    start_allocation_decoded = decode_relaxed(candidates, msn_logits.detach(),
                                              factory_logits.detach())
    start_true = evaluate(start_allocation_decoded)
    with torch.no_grad():
        start_predicted = float(relaxed_objective(
            model, candidates, msn_logits, factory_logits, target_scale)[0])

    for step in range(nb_steps):
        optimizer.zero_grad()
        total, _ = relaxed_objective(model, candidates, msn_logits, factory_logits, target_scale)
        loss = total

        if exclusivity_weight > 0:
            loss = loss + exclusivity_weight * _exclusivity_penalty(
                candidates, msn_logits, factory_logits)
        if trust_radius > 0:
            loss = loss + trust_radius * (
                (msn_logits - anchor_msn).clamp(min=-1e6, max=1e6).pow(2).sum()
                + (factory_logits - anchor_factory).clamp(min=-1e6, max=1e6).pow(2).sum())

        loss.backward()
        # Les colonnes hors candidats portent -inf, dont le gradient est NaN : on les remet à
        # zéro plutôt que de les laisser contaminer Adam, qui propagerait le NaN à toute la ligne.
        for tensor in (msn_logits, factory_logits):
            torch.nan_to_num_(tensor.grad, nan=0.0, posinf=0.0, neginf=0.0)
        optimizer.step()

        if verbose and (step % max(1, nb_steps // 10) == 0 or step == nb_steps - 1):
            print(f"    pas {step:>4} | prédit {float(total):>14,.0f} €", flush=True)

    with torch.no_grad():
        final_predicted = float(relaxed_objective(
            model, candidates, msn_logits, factory_logits, target_scale)[0])
    final_allocation = decode_relaxed(candidates, msn_logits.detach(), factory_logits.detach())

    return DescentResult(
        start_true=start_true,
        true=evaluate(final_allocation),
        predicted=final_predicted,
        start_predicted=start_predicted,
        nb_steps=nb_steps,
        wall_time=time.perf_counter() - started,
        nb_allocated=sum(1 for d in final_allocation.values() if d["msn_id"] is not None),
        nb_allocated_start=sum(
            1 for d in start_allocation_decoded.values() if d["msn_id"] is not None),
    )
