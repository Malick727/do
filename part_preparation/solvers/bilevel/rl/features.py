import functools
from typing import NamedTuple

import numpy as np

from part_preparation.solvers.bilevel.rl.environnement import (
    WEIGHT_DELAY, WEIGHT_STORAGE, WEIGHT_TRAVEL)

ENGINE_FEATURES = 6
MSN_FEATURES = 5
FACTORY_BASE_FEATURES = 8   # + une date par station (max_capacity)


class Scales(NamedTuple):
    horizon: float
    max_cost: float
    max_prep: float
    max_capacity: int
    reward_scale: float


@functools.lru_cache(maxsize=None)
def scales(problem):
    horizon = max(float(problem.time_horizon), 1.0)
    max_cost = max(
        (problem.get_travel_cost(f, a)
         for f in problem.factories_list for a in problem.assembly_lines_list),
        default=1.0) or 1.0
    max_prep = max(
        (f.get_preparation_duration(t)
         for f in problem.factories_list for t in f.compatible_engine_types),
        default=1.0) or 1.0
    max_capacity = max(f.default_capacity or 1 for f in problem.factories_list)

    max_rate = max((e.engine_type_object.cost_storage_per_time
                    for e in problem.engines_list), default=1.0)
    reward_scale = max(
        WEIGHT_TRAVEL * max_cost + (WEIGHT_STORAGE + WEIGHT_DELAY) * max_rate * horizon, 1.0)

    return Scales(horizon, max_cost, max_prep, max_capacity, reward_scale)


def occupancy(env, state):
    engine_free = np.ones(len(env.engines), dtype=bool)
    msn_free = np.ones(len(env.msns), dtype=bool)
    for engine_id, msn_id in state.engineToMsn.items():
        engine_free[env.engine_index[engine_id]] = False
        msn_free[env.msn_index[msn_id]] = False
    return engine_free, msn_free


def engine_candidates(env, state):
    s = scales(env.problem)
    engine_free, msn_free = occupancy(env, state)
    indices = np.flatnonzero(engine_free)
    if len(indices) == 0 or not msn_free.any():
        return np.zeros((0, ENGINE_FEATURES), dtype=np.float32), indices[:0]

    delivery = env.delivery[indices]
    free_deadlines = np.sort(env.deadline[msn_free])
    n_free_msn = len(free_deadlines)

    earliest_arrival = delivery + env.prep.min() + env.travel_time.min()
    reachable = n_free_msn - np.searchsorted(free_deadlines, earliest_arrival, side="left")

    rate = env.engines[indices[0]].engine_type_object.cost_storage_per_time
    skip_cost = rate * np.maximum(0.0, s.horizon - delivery)

    feats = np.stack([
        delivery / s.horizon,                               # 0 date de livraison
        (delivery - delivery.min()) / s.horizon,            # 1 retard dans le pool
        (free_deadlines[0] - delivery) / s.horizon,         # 2 marge vs MSN le plus urgent
        reachable / n_free_msn,                             # 3 MSN atteignables a temps
        np.full(len(indices), n_free_msn / len(env.msns)),  # 4 MSN restants
        skip_cost / s.reward_scale,                         # 5 cout de non-placement
    ], axis=1)
    return feats.astype(np.float32), indices


def msn_candidates(env, state, engine_index):
    s = scales(env.problem)
    engine_free, msn_free = occupancy(env, state)
    indices = np.flatnonzero(msn_free)
    if len(indices) == 0:
        return np.zeros((0, MSN_FEATURES), dtype=np.float32), indices

    delivery = env.delivery[engine_index]
    deadlines = env.deadline[indices]
    travel_cost = env.min_travel[indices]
    travel_time = env.min_travel_time[indices]
    prep_min = env.prep.min()

    free_deliveries = np.sort(env.delivery[engine_free])
    latest_ok = deadlines - prep_min - travel_time
    servers = np.searchsorted(free_deliveries, latest_ok, side="right")

    feats = np.stack([
        deadlines / s.horizon,                                        # 0 urgence absolue
        (deadlines - delivery) / s.horizon,                           # 1 marge brute
        travel_cost / s.max_cost,                                     # 2 transport min
        (deadlines - delivery - prep_min - travel_time) / s.horizon,  # 3 marge nette
        servers / max(len(free_deliveries), 1),                       # 4 concurrence
    ], axis=1)
    return feats.astype(np.float32), indices


def factory_candidates(env, state, engine_index, msn_index):
    s = scales(env.problem)
    indices = np.arange(len(env.factories))
    deadline = env.deadline[msn_index]
    delivery = env.delivery[engine_index]
    fal = env.msn_fal[msn_index]

    travel_cost = env.travel_cost[:, fal]
    travel_time = env.travel_time[:, fal]
    prep = env.prep

    per_factory = [sorted(state.stationAvailableDate[(f.id, k)]
                          for k in range(f.default_capacity or 1)) for f in env.factories]
    pad = max(s.horizon, max(max(d) for d in per_factory)) + 1.0
    stations = np.full((len(env.factories), s.max_capacity), pad)
    for i, dates in enumerate(per_factory):
        stations[i, :len(dates)] = dates

    earliest = stations[:, 0]

    start = np.maximum(np.maximum(deadline - travel_time - prep, np.ceil(earliest)), delivery)
    arrival = start + prep + travel_time
    delay = np.maximum(0.0, arrival - deadline)

    rate = env.engines[engine_index].engine_type_object.cost_storage_per_time
    storage = rate * (np.maximum(0.0, start - delivery)
                      + np.maximum(0.0, deadline - arrival))

    feats = np.concatenate([
        np.stack([
            prep / s.max_prep,                                      # 0 duree de preparation
            env.capacity / s.max_capacity,                          # 1 capacite
            np.maximum(0.0, earliest - delivery) / s.horizon,       # 2 attente en file
            travel_cost / s.max_cost,                               # 3 cout de transport
            travel_time / s.horizon,                                # 4 temps de transport
            -(WEIGHT_TRAVEL * travel_cost + WEIGHT_DELAY * delay
              + WEIGHT_STORAGE * storage) / s.reward_scale,
            delay / s.horizon,                                      # 6 retard
            storage / s.reward_scale,                               # 7 stockage
        ], axis=1),
        stations / s.horizon,                                       # 8.. dates des stations
    ], axis=1)
    return feats.astype(np.float32), indices


def state_features(env, state):
    s = scales(env.problem)
    engine_free, msn_free = occupancy(env, state)

    def stats(values, empty_value=0.0):
        if len(values) == 0:
            return np.full(3, empty_value)
        return np.array([values.min(), values.mean(), values.max()])

    station_dates = np.fromiter(state.stationAvailableDate.values(), dtype=float)

    parts = [
        stats(station_dates) / s.horizon,                             # 3  files d'attente
        env.travel_cost.ravel() / s.max_cost,                         # 16 instance
        env.travel_time.ravel() / s.horizon,                          # 16 instance
        np.array([engine_free.sum() / len(env.engines),               # 2  ce qui reste
                  msn_free.sum() / len(env.msns)]),
        np.array([len(state.engineToMsn) / max(env.max_steps, 1)]),   # 1  progression
        stats(env.deadline[msn_free], s.horizon) / s.horizon,         # 3  echeances restantes
        stats(env.delivery[engine_free]) / s.horizon,                 # 3  livraisons restantes
    ]
    return np.concatenate(parts).astype(np.float32)
