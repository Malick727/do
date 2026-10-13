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

_WEIGHT = dict(zip(AGGREGATED_OBJECTIVES, AGGREGATED_WEIGHTS))
WEIGHT_DELAY = _WEIGHT["arrival_delays"]
WEIGHT_TRAVEL = _WEIGHT["travel_cost"]
WEIGHT_STORAGE = _WEIGHT["storage_cost"]


@dataclass
class State:
    engineToFactory: Dict[str, str]
    engineToMsn: Dict[str, str]
    engineStart: Dict[str, int]
    stationAvailableDate: Dict[Tuple[str, int], int] = field(default_factory=dict)


class Env():

    def __init__(self, problem: EnginePreparationProblem, engine_type: str):
        self.problem = problem
        self.engine_type = engine_type
        self.state: State = None
        self._build_index()
        self.action_dim = len(self.factories)

    def _build_index(self):
        p, t = self.problem, self.engine_type

        self.engines = [e for e in p.engines_list if e.engine_type == t]
        self.msns = [m for m in p.msns_list if m.engine_type == t]
        self.factories = [f for f in p.factories_list if f.can_prepare(t)]
        self.fals = list(p.assembly_lines_list)

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

        self.min_travel = self.travel_cost[:, self.msn_fal].min(axis=0)
        self.min_travel_time = self.travel_time[:, self.msn_fal].min(axis=0)

        self.horizon = float(p.time_horizon)
        self.max_cost = max(self.travel_cost.max(), 1.0)
        self.type_max_cap = int(self.capacity.max())
        self.max_prep = max(self.prep.max(), 1.0)
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
        factory = self.problem.factories[factory_id]
        prep_duration = factory.get_preparation_duration(engine_type)

        avalaibleDates = np.array([
            self.state.stationAvailableDate[(factory_id, station_id)]
            for station_id in range(factory.default_capacity or 1)
        ])
        ideal_start = installation_date - travel_time - prep_duration

        seletedStation = int(np.argmin(avalaibleDates))

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

        travel_cost = self.problem.get_travel_cost(prepFactory, assemblyLine)
        travel_time = self.problem.get_travel_time(prepFactory, assemblyLine)

        engine_delivery = engine.delivery_time if engine.delivery_time is not None else 0
        start, station, prep_duration = self.get_start_date(
            engine_delivery, engine.engine_type, factory_id,
            msn.engine_installation_date, travel_time,
        )

        arrival = start + prep_duration + travel_time
        delay = max(0, arrival - msn.engine_installation_date)

        storage_rate = engine.engine_type_object.cost_storage_per_time
        storage = storage_rate * (
            max(0, start - engine_delivery)                          # attente avant preparation
            + max(0, msn.engine_installation_date - arrival)         # attente a la FAL
        )

        reward = -(WEIGHT_DELAY * delay
                   + WEIGHT_TRAVEL * travel_cost
                   + WEIGHT_STORAGE * storage)

        self.state.engineToFactory[engine_id] = factory_id
        self.state.engineToMsn[engine_id] = msn_id
        self.state.engineStart[engine_id] = start
        self.state.stationAvailableDate[(factory_id, station)] = start + prep_duration
        self.engine_free[self.engine_index[engine_id]] = False
        self.msn_free[self.msn_index[msn_id]] = False

        done = not (self.engine_free.any() and self.msn_free.any())

        if done:
            leftover = 0.0
            for i in np.flatnonzero(self.engine_free):
                rate = self.engines[i].engine_type_object.cost_storage_per_time
                leftover += rate * max(0.0, self.horizon - self.delivery[i])
            reward -= WEIGHT_STORAGE * leftover

        info = {"engine": engine_id, "factory": factory_id, "msn": msn_id,
                "start": start, "arrival": arrival, "delay": delay,
                "travel_cost": travel_cost}
        return self.state, reward, done, info
