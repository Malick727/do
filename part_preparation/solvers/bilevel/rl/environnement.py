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
    AGGREGATED_OBJECTIVES,
    AGGREGATED_WEIGHTS,
)
import math
import numpy as np
logger = logging.getLogger(__name__)

# Poids lus depuis problem.py plutot que recopies : c'est la source unique que partagent CP-SAT,
# le GA et l'evaluation. Les recopier ici garantirait qu'un jour la recompense du RL et
# aggregate_score divergent sans que rien ne le signale -- ce qui vient d'arriver quand
# storage_cost est passe de 0 a 1.
_WEIGHT = dict(zip(AGGREGATED_OBJECTIVES, AGGREGATED_WEIGHTS))
WEIGHT_DELAY = _WEIGHT["arrival_delays"]
WEIGHT_TRAVEL = _WEIGHT["travel_cost"]
WEIGHT_STORAGE = _WEIGHT["storage_cost"]


@dataclass
class State:
    engineToFactory: Dict[str, str]
    engineToMsn: Dict[str, str]
    engineStart: Dict[str, int]
    # (factory_id, station_id) -> date a laquelle la station se libere.
    # Construit dans Env.reset() et non ici : la compréhension avait besoin de `self`, qui
    # n'existe pas dans un corps de classe. Et sans annotation de type, dataclass en aurait
    # fait un attribut de CLASSE, donc partagé par tous les State.
    stationAvailableDate: Dict[Tuple[str, int], int] = field(default_factory=dict)


class Env():
    """
    Action TRIPLE et autoregressive : (moteur, MSN, usine), dans cet ordre.

    L'ordre n'est pas neutre. Le transport depend du COUPLE (usine, FAL), donc un des deux
    niveaux decide forcement sans connaitre l'autre. En placant l'usine en DERNIER, le dernier
    niveau connait la FAL et la file : `travel_cost` et `delay` y sont EXACTS, et la recompense
    immediate de chaque usine candidate est calculable avant de choisir -- c'est la feature 6
    de factory_candidates. Dans l'ordre inverse, ce serait le niveau MSN qui deciderait a
    l'aveugle sur 99.6% du cout.
    """

    def __init__(self, problem: EnginePreparationProblem, engine_type: str):
        self.problem = problem
        self.engine_type = engine_type
        self.state: State = None
        self._build_index()
        self.action_dim = len(self.factories)

    def _build_index(self):
        """
        Entities and numeric attributes for ONE engine type, built once per instance.

        Everything is restricted to `engine_type` because the problem splits exactly by type:
        factories are type-exclusive, every MSN has a single type, and the FALs -- the only
        shared entity -- carry no capacity, they are just a lookup key for travel. So one Env
        is one independent sub-problem, and score(solution) = sum over types.

        The arrays are not needed for correctness, only for speed: building the candidate
        features with Python loops would cost ~1.5M iterations per episode.
        """
        p, t = self.problem, self.engine_type

        self.engines = [e for e in p.engines_list if e.engine_type == t]
        self.msns = [m for m in p.msns_list if m.engine_type == t]
        self.factories = [f for f in p.factories_list if f.can_prepare(t)]
        self.fals = list(p.assembly_lines_list)

        # id -> position, so step() keeps taking ids while the masks stay positional
        self.engine_index = {e.id: i for i, e in enumerate(self.engines)}
        self.msn_index = {m.id: i for i, m in enumerate(self.msns)}
        self.factory_index = {f.id: i for i, f in enumerate(self.factories)}

        fal_of = {a.id: i for i, a in enumerate(self.fals)}
        self.msn_fal = np.array([fal_of[m.assembly_line_id] for m in self.msns])
        self.deadline = np.array([m.engine_installation_date for m in self.msns], dtype=float)
        self.delivery = np.array(
            [e.delivery_time if e.delivery_time is not None else 0 for e in self.engines],
            dtype=float)

        n_f, n_a = len(self.factories), len(self.fals)
        self.travel_cost = np.zeros((n_f, n_a))
        self.travel_time = np.zeros((n_f, n_a))
        for i, f in enumerate(self.factories):
            for j, a in enumerate(self.fals):
                self.travel_cost[i, j] = p.get_travel_cost(f, a)
                self.travel_time[i, j] = p.get_travel_time(f, a)

        self.prep = np.array([np.ceil(f.get_preparation_duration(t)) for f in self.factories])
        self.capacity = np.array([f.default_capacity or 1 for f in self.factories])

        # Cheapest travel each MSN could ever get, over all compatible factories. This is what
        # keeps the MSN level from choosing blind: the factory is not picked yet, but the agent
        # still knows what that MSN can cost at best.
        self.min_travel = self.travel_cost[:, self.msn_fal].min(axis=0)
        self.min_travel_time = self.travel_time[:, self.msn_fal].min(axis=0)

        self.horizon = float(p.time_horizon)
        self.max_cost = max(self.travel_cost.max(), 1.0)
        # Largest capacity among THIS TYPE's factories (3 for type 1, 4 for type 2). Do not use
        # it to pad per-station features: the padded width must be the same for every type, or
        # a shared network gets 10 factory features on one type and 11 on the other. Pad with
        # Agent.maxCapacity, which is the max over all factories.
        self.type_max_cap = int(self.capacity.max())
        self.max_prep = max(self.prep.max(), 1.0)
        # Longest possible episode: one engine per MSN, whichever side runs out first.
        self.max_steps = min(len(self.engines), len(self.msns))

    def reset(self) -> State:
        self.state = State(
            engineToFactory={},
            engineToMsn={},
            engineStart={},
            stationAvailableDate={
                (f.id, station): 0
                for f in self.factories for station in range(f.default_capacity or 1)
            },
        )
        self.engine_free = np.ones(len(self.engines), dtype=bool)
        self.msn_free = np.ones(len(self.msns), dtype=bool)
        return self.state

    def get_start_date(self, engine_delivery, engine_type, factory_id, installation_date,
                       travel_time):
        """
        Date de debut de preparation, et station retenue.

        engine_type est necessaire : la duree de preparation depend du TYPE du moteur
        (factory.get_preparation_duration(type)), il n'existe pas de factory.duration.
        """
        factory = self.problem.factories[factory_id]
        prep_duration = factory.get_preparation_duration(engine_type)

        # np.array et non list : `list - float` leve TypeError.
        avalaibleDates = np.array([
            self.state.stationAvailableDate[(factory_id, station_id)]
            for station_id in range(factory.default_capacity or 1)
        ])
        ideal_start = installation_date - travel_time - prep_duration

        # Station libre le PLUS TOT, et non la plus proche de ideal_start : argmin(abs(...))
        # traitait « libre 50 jours trop tard » comme « libre 50 jours trop tot », alors que
        # seule la premiere coute du retard. C'est aussi ce que fait l'ordonnanceur greedy.
        seletedStation = int(np.argmin(avalaibleDates))

        # Jamais avant la livraison du moteur, jamais avant que la station se libere.
        # math.ceil et NON int : les dates sont fractionnaires (22.75), et int() tronque vers le
        # bas, ce qui laissait un moteur demarrer a 22 sur une station occupee jusqu'a 22.75 --
        # la capacite de l'usine sautait sans que rien ne le signale. C'est aussi ce que fait
        # l'ordonnanceur de reference, cf. bilevel.lower.simulate_greedy_from_allocation.
        start = max(ideal_start, math.ceil(avalaibleDates[seletedStation]), engine_delivery)

        return start, seletedStation, prep_duration

    def step(self, action: Tuple[str, str, str]):  # action is (engine, factory, msn)
        engine_id = action[0]
        factory_id = action[1]
        msn_id = action[2]

        engine = self.problem.engines[engine_id]
        prepFactory = self.problem.factories[factory_id]
        msn = self.problem.msns[msn_id]
        assemblyLineId = msn.assembly_line_id
        assemblyLine = self.problem.assembly_lines[assemblyLineId]

        # get_travel_cost / get_travel_time rendent des nombres. travel_lookup[...] rendait un
        # objet TravelInfo, et gere le couple absent (0.0) au lieu de lever KeyError.
        travel_cost = self.problem.get_travel_cost(prepFactory, assemblyLine)
        travel_time = self.problem.get_travel_time(prepFactory, assemblyLine)

        engine_delivery = engine.delivery_time if engine.delivery_time is not None else 0
        start, station, prep_duration = self.get_start_date(
            engine_delivery, engine.engine_type, factory_id,
            msn.engine_installation_date, travel_time,
        )

        arrival = start + prep_duration + travel_time
        delay = max(0, arrival - msn.engine_installation_date)

        # Stockage, terme a terme comme problem.evaluate. Le troisieme terme de evaluate --
        # l'attente entre la fin de preparation et le depart -- vaut TOUJOURS zero ici : notre
        # arrival est prep_end + travel_time, donc departure == prep_end par construction.
        storage_rate = engine.engine_type_object.cost_storage_per_time
        storage = storage_rate * (
            max(0, start - engine_delivery)                          # attente avant preparation
            + max(0, msn.engine_installation_date - arrival)         # attente a la FAL
        )

        # Le RL MAXIMISE : la recompense est l'oppose du cout. Les trois termes sont ceux de
        # AGGREGATED_OBJECTIVES, avec leurs poids -- sauf unallocated_msns, nul par construction
        # puisque l'episode s'arrete au couplage maximum.
        reward = -(WEIGHT_DELAY * delay
                   + WEIGHT_TRAVEL * travel_cost
                   + WEIGHT_STORAGE * storage)

        # Without these updates step() changed nothing: the station stayed free (capacity
        # ignored) and nothing stopped the same MSN from being taken twice.
        self.state.engineToFactory[engine_id] = factory_id
        self.state.engineToMsn[engine_id] = msn_id
        self.state.engineStart[engine_id] = start
        self.state.stationAvailableDate[(factory_id, station)] = start + prep_duration
        self.engine_free[self.engine_index[engine_id]] = False
        self.msn_free[self.msn_index[msn_id]] = False

        # One type only, so the stopping rule is simple: over as soon as either side runs out.
        done = not (self.engine_free.any() and self.msn_free.any())

        if done:
            # Les moteurs jamais alloues paient leur stockage de la livraison a l'horizon
            # (cf. problem.evaluate). Ce n'est pas une constante : la politique CHOISIT lesquels
            # rester sur le carreau, et en laisser un livre tot coute plus cher qu'un livre
            # tard. Le facturer ici est ce qui maintient somme(rewards) == -aggregate_score.
            leftover = 0.0
            for i in np.flatnonzero(self.engine_free):
                rate = self.engines[i].engine_type_object.cost_storage_per_time
                leftover += rate * max(0.0, self.horizon - self.delivery[i])
            reward -= WEIGHT_STORAGE * leftover

        info = {"engine": engine_id, "factory": factory_id, "msn": msn_id,
                "start": start, "arrival": arrival, "delay": delay,
                "travel_cost": travel_cost}
        return self.state, reward, done, info
