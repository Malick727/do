import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter
import logging
from discrete_optimization.generic_tools.do_problem import Solution, ParamsObjectiveFunction
from scipy.optimize import linear_sum_assignment

from part_preparation.problem import (
    EnginePreparationProblem,
    EnginePreparationSolution,
    Engine,
    MSN,
    PreparationFactory,
    AssemblyLine,
)
import numpy as np
logger = logging.getLogger(__name__)

def _scorer(input_dim, hidden_dim):
    """
    Note UN candidat : (..., input_dim) -> (..., 1).

    Une couche cachee au minimum. Un nn.Linear seul est un modele LINEAIRE : il ne peut pas
    representer « quand le moteur arrive tard, la congestion pese plus que le transport »,
    qui est un effet croise entre deux features.
    """
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )


class ActorCriticNetwork(nn.Module):
    """
    Trois tetes d'acteur qui NOTENT DES CANDIDATS, une tete critique sur l'etat global.

    Les tetes d'acteur sortent 1 scalaire par candidat, jamais un nombre fixe de logits : la
    taille de sortie suit donc le nombre de candidats. Indispensable ici, ou les ensembles
    legaux retrecissent au fil de l'episode (1023 moteurs -> 0) et ou les deux types de moteurs
    ont des usines DISJOINTES. Le reseau apprend une regle de comparaison, pas une table
    d'identites -- il ne sait pas ce qu'est « Toulouse ».

    Mecaniquement ce sont les candidats qui jouent le role du batch : nn.Linear ne s'applique
    qu'a la derniere dimension, donc le partage de poids entre candidats est automatique.

    Pas de `forward` : l'action est autoregressive, les quatre sorties sont demandees a des
    moments differents du meme pas (les features MSN n'existent qu'apres le choix du moteur,
    celles des usines qu'apres le choix du MSN). Les parametres restent enregistres par
    l'affectation dans __init__, donc .parameters(), .to() et state_dict() fonctionnent.
    """

    def __init__(self, engine_input_dim, msn_input_dim, factory_input_dim,
                 state_input_dim, hidden_dim=64):
        super().__init__()
        self.engine = _scorer(engine_input_dim, hidden_dim)
        self.msn = _scorer(msn_input_dim, hidden_dim)
        self.factory = _scorer(factory_input_dim, hidden_dim)
        # V(s) est un SCALAIRE, et il prend l'ETAT -- pas des candidats. Seule tete dont
        # l'entree doit etre de taille fixe.
        self.value = nn.Sequential(
            nn.Linear(state_input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    # Les tetes rendent des LOGITS bruts, pas un softmax : Categorical(logits=...) est
    # numeriquement plus stable, et c'est sur les logits qu'on masque le cas echeant.
    #
    # squeeze(-1) n'est pas cosmetique. nn.Linear(h, 1) sort (n_cand, 1), et
    # Categorical(logits=(n_cand, 1)) construit n_cand distributions a UNE issue : sample()
    # rend toujours 0, sans lever la moindre erreur.
    def score_engines(self, feats):
        """(n_candidats, engine_input_dim) -> (n_candidats,)"""
        return self.engine(feats).squeeze(-1)

    def score_msns(self, feats):
        """(n_candidats, msn_input_dim) -> (n_candidats,)"""
        return self.msn(feats).squeeze(-1)

    def score_factories(self, feats):
        """(n_candidats, factory_input_dim) -> (n_candidats,)"""
        return self.factory(feats).squeeze(-1)

    def state_value(self, state):
        """
        (state_input_dim,) -> ()   ou   (B, state_input_dim) -> (B,)

        Appelee UNE fois par pas, sur l'etat d'AVANT l'action. Ne jamais la calculer depuis une
        representation qui depend du moteur ou du MSN deja choisi : une baseline doit etre
        fonction du seul etat, sinon le gradient de REINFORCE devient biaise.

        Le squeeze evite ici un autre echec silencieux : V de forme (B, 1) contre des retours
        de forme (B,) fait broadcaster mse_loss en (B, B).
        """
        return self.value(state).squeeze(-1)

# features per candidate, declared once so the encoder and the network agree
ENGINE_FEATURES = 6         # delivery, lateness in pool, slack, reachable MSNs,
                            # MSNs remaining, cost of leaving it unplaced
MSN_FEATURES = 5            # deadline, raw slack, cheapest reachable travel, net slack, delivery
FACTORY_BASE_FEATURES = 8   # prep, capacity, queue wait, travel cost, travel time,
                            # immediate reward, delay, storage


class Agent:
    """One network, one episode per engine type (the problem splits exactly by type)."""

    def __init__(self, problem: EnginePreparationProblem, engine_type: str, hidden_dim: int = 64):
        self.problem = problem
        self.engine_type = engine_type
        self.factories = [f for f in problem.factories_list if f.can_prepare(engine_type)]
        self.engines = [e for e in problem.engines_list if e.engine_type == engine_type]
        self.msns = [m for m in problem.msns_list if m.engine_type == engine_type]
        self.fals = problem.assembly_lines_list

        # one available date per station, SORTED and padded to the largest capacity: capacities
        # differ (1 to 4), and every candidate must have the same feature count. Sorted because
        # station identity means nothing; padded with the horizon so a missing station reads as
        # never available. Max taken over ALL factories so both types share the dimension.
        self.maxCapacity = max(f.default_capacity for f in problem.factories_list)
        self.factoryFeatures = FACTORY_BASE_FEATURES + self.maxCapacity

        # global state for the critic. min/mean/max over stations rather than one input each:
        # type 1 has 8 stations and type 2 has 10, and one network must serve both.
        self.stateDim = (
            3                                           # station release dates, min/mean/max
            + 2 * len(self.factories) * len(self.fals)  # travel cost and time to each FAL
            + 2                                         # remaining fraction of engines, of MSNs
            + 1                                         # progress
            + 3                                         # remaining MSN deadlines, min/mean/max
            + 3                                         # remaining engine deliveries, min/mean/max
        )

        self.network = ActorCriticNetwork(
            ENGINE_FEATURES, MSN_FEATURES, self.factoryFeatures, self.stateDim, hidden_dim)
