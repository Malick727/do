"""
Feature extraction: (problem, state) -> the tensors the network consumes.

This is the MODELLING side. The environment only owns dynamics, state and reward; what the
agent looks at, and how it is scaled, lives here. That separation is what lets the MLP be
swapped for a GNN without touching the environment.

Four pure functions, each returning (feats, indices). Only LEGAL candidates are returned, so
row k of `feats` describes entity `indices[k]` and the network can never sample an illegal
action. An empty `indices` means the level has no candidate -- the caller must check it before
building a Categorical, which returns NaN on an empty set instead of raising.
"""
import numpy as np

from part_preparation.solvers.bilevel.rl.environnement import (
    WEIGHT_DELAY, WEIGHT_STORAGE, WEIGHT_TRAVEL)


class FeatureExtractor:
    """
    Reads the env's static tables (instance data) and the live `state`. Holds no state of its
    own, so two calls on the same state give the same answer.

    Normalisation constants are computed over the WHOLE problem, not over the type this env
    covers. A single network serves both engine types, so "0.5" must mean the same number of
    euros on both sides -- scaling type 1 by its own max and type 2 by its own would make the
    features silently incomparable.
    """

    def __init__(self, env):
        self.env = env
        problem = env.problem

        # padding width for per-station features: max over ALL factories, so type 1 (3 stations
        # at most) and type 2 (4) produce the same feature count. env.type_max_cap is the
        # per-type max and must NOT be used here.
        self.max_capacity = max(f.default_capacity or 1 for f in problem.factories_list)

        self.horizon = max(float(problem.time_horizon), 1.0)
        self.max_cost = max(
            (problem.get_travel_cost(f, a)
             for f in problem.factories_list for a in problem.assembly_lines_list),
            default=1.0) or 1.0
        self.max_prep = max(
            (f.get_preparation_duration(t)
             for f in problem.factories_list for t in f.compatible_engine_types),
            default=1.0) or 1.0
        self.max_capacity_value = max(
            (f.default_capacity or 1) for f in problem.factories_list)

        # Scale for the immediate reward, which now mixes euros of travel with euros of storage
        # (rate x days). Upper bound on what one engine can cost in a single step, so feature 5
        # lands in roughly [-1, 0] instead of the thousands.
        max_rate = max((e.engine_type_object.cost_storage_per_time
                        for e in problem.engines_list), default=1.0)
        self.reward_scale = max(
            WEIGHT_TRAVEL * self.max_cost
            + (WEIGHT_STORAGE + WEIGHT_DELAY) * max_rate * self.horizon, 1.0)

    def occupancy(self, state):
        """
        Which engines and MSNs are still free, derived from state.engineToMsn.

        Derived rather than cached: the extractor stays a pure function of the state, so it
        cannot drift out of sync with the environment. Cost is O(number already placed), since
        we start from all-free and only clear what has been used.
        """
        env = self.env
        engine_free = np.ones(len(env.engines), dtype=bool)
        msn_free = np.ones(len(env.msns), dtype=bool)
        for engine_id, msn_id in state.engineToMsn.items():
            engine_free[env.engine_index[engine_id]] = False
            msn_free[env.msn_index[msn_id]] = False
        return engine_free, msn_free

    def engine_candidates(self, state):
        """
        Level 1: which engine to place next.

        Within one type, two engines differ ONLY by their delivery date -- everything else on
        the Engine dataclass is the type itself or unused baseline fields. So type-level counts
        are the same for every candidate and cannot rank anything; they belong to the critic's
        state vector. Every feature here is therefore a function of delivery(e) and the state.

        Feature 3 is the important one: how many still-free MSNs this engine can reach on time.
        It measures how CONSTRAINED the engine is, which is the "most constrained first"
        scheduling principle made observable -- an engine that can still serve 2 MSNs should be
        placed before one that has 200 ahead of it.

        The early return guards this function's own arithmetic (indexing an empty array,
        dividing by zero). It is NOT the termination signal: following the Gym convention that
        stays with the environment, which already sets `done` on the same condition.
        """
        env = self.env
        engine_free, msn_free = self.occupancy(state)
        indices = np.flatnonzero(engine_free)
        if len(indices) == 0 or not msn_free.any():
            return np.zeros((0, 5), dtype=np.float32), indices[:0]

        delivery = env.delivery[indices]
        free_deadlines = np.sort(env.deadline[msn_free])
        n_free_msn = len(free_deadlines)

        # earliest a FAL could ever be reached, with the fastest factory and the fastest route
        earliest_arrival = delivery + env.prep.min() + env.travel_time.min()
        # how many free MSNs remain reachable on time, by binary search on the sorted deadlines
        reachable = n_free_msn - np.searchsorted(free_deadlines, earliest_arrival, side="left")

        # What this engine costs if it is NEVER placed: storage from delivery to the horizon.
        # Worth 83,600 on a medium replicate, i.e. 29% of the objective and 1.4x the whole
        # travel bill -- yet nothing in the first five features saw it. An engine delivered
        # early is expensive to leave out, one delivered late is nearly free.
        rate = env.engines[indices[0]].engine_type_object.cost_storage_per_time
        skip_cost = rate * np.maximum(0.0, self.horizon - delivery)

        feats = np.stack([
            delivery / self.horizon,                            # 0 when it becomes available
            (delivery - delivery.min()) / self.horizon,         # 1 lateness within the pool
            (free_deadlines[0] - delivery) / self.horizon,      # 2 slack vs the tightest MSN
            reachable / n_free_msn,                             # 3 how CONSTRAINED it is
            np.full(len(indices), n_free_msn / len(env.msns)),  # 4 how many MSNs remain
            skip_cost / self.reward_scale,                      # 5 cost of NOT placing it
        ], axis=1)
        return feats.astype(np.float32), indices

    def msn_candidates(self, state, engine_index):
        """
        Level 2: which MSN to serve with the engine already chosen.

        Within one type, two MSNs differ by exactly two things: their FAL, which sets the travel
        cost, and their installation date. entry_time and end_time never enter the objective.

        The factory is not picked yet, so the real travel cost is unknown at this level.
        Features 2 and 3 stand in for it with the CHEAPEST reachable route, which is what keeps
        this decision from being blind on 99.6% of the cost.

        Feature 4 mirrors the engine level: how many free engines could still serve this MSN on
        time. Low means almost nobody can take it any more -- serve it now or it becomes a
        guaranteed delay. High means it can safely be left to another engine.
        """
        env = self.env
        engine_free, msn_free = self.occupancy(state)
        indices = np.flatnonzero(msn_free)
        if len(indices) == 0:
            return np.zeros((0, 5), dtype=np.float32), indices

        delivery = env.delivery[engine_index]
        deadlines = env.deadline[indices]
        # cheapest route to each MSN's FAL, over the factories that can prepare this type
        travel_cost = env.min_travel[indices]
        travel_time = env.min_travel_time[indices]
        prep_min = env.prep.min()

        # Latest delivery date that still makes this MSN on time, then count the free engines
        # at or before it. side="right" so an engine delivered exactly on that date counts:
        # the objective charges max(0, arrival - deadline), so equality is zero delay.
        free_deliveries = np.sort(env.delivery[engine_free])
        latest_ok = deadlines - prep_min - travel_time
        servers = np.searchsorted(free_deliveries, latest_ok, side="right")

        feats = np.stack([
            deadlines / self.horizon,                                        # 0 absolute urgency
            (deadlines - delivery) / self.horizon,                           # 1 slack, THIS engine
            travel_cost / self.max_cost,                                     # 2 cheapest possible
            (deadlines - delivery - prep_min - travel_time) / self.horizon,  # 3 NET slack
            servers / max(len(free_deliveries), 1),                          # 4 how CONTESTED
        ], axis=1)
        return feats.astype(np.float32), indices

    def factory_candidates(self, state, engine_index, msn_index):
        """
        Level 3: which factory. Engine and MSN are fixed, so the FAL and the queue are known,
        and `travel_cost` and `delay` are EXACT here -- no proxy at all.

        Feature 5 is the immediate reward env.step will return if this factory is picked. The
        agent has nothing to reconstruct at this level; its only real job is to arbitrate that
        immediate cost against the CONGESTION it creates for later engines (feature 2 and the
        station dates), which is the one part an immediate reward cannot show.

        The start-date formula below MUST match Env.get_start_date exactly. If the two drift --
        a missing max, delivery left out -- feature 5 announces a reward that step will never
        pay, and the agent trains on a lie. First place to look if learning misbehaves.

        No feasibility notion here: a saturated factory stays legal, it merely costs waiting.
        Capacity is a cost constraint, not a feasibility one.
        """
        env = self.env
        indices = np.arange(len(env.factories))
        deadline = env.deadline[msn_index]
        delivery = env.delivery[engine_index]
        fal = env.msn_fal[msn_index]

        travel_cost = env.travel_cost[:, fal]
        travel_time = env.travel_time[:, fal]
        prep = env.prep

        # Station release dates, sorted per factory then padded to the width of the LARGEST
        # factory across all types, so type 1 (3 stations) and type 2 (4) yield the same feature
        # count and one network serves both. Sorted because station identity is meaningless:
        # swapping two stations must not change the input.
        #
        # The pad is strictly above every real date. Zero would be the worst possible choice --
        # it reads as "free right now", the exact opposite of "does not exist", and would make
        # Singapore (capacity 1) look like the least congested factory of the park.
        per_factory = [sorted(state.stationAvailableDate[(f.id, s)]
                              for s in range(f.default_capacity or 1)) for f in env.factories]
        pad = max(self.horizon, max(max(d) for d in per_factory)) + 1.0
        stations = np.full((len(env.factories), self.max_capacity), pad)
        for i, dates in enumerate(per_factory):
            stations[i, :len(dates)] = dates

        earliest = stations[:, 0]  # dates are sorted ascending, so column 0 is the real minimum

        # Exactly Env.get_start_date: never before the ideal date, before the station frees up,
        # nor before the engine is delivered.
        # np.ceil on the station date, to mirror Env.get_start_date exactly. Without it the
        # feature would promise a reward the environment does not pay.
        start = np.maximum(np.maximum(deadline - travel_time - prep, np.ceil(earliest)), delivery)
        arrival = start + prep + travel_time
        delay = np.maximum(0.0, arrival - deadline)

        # Storage, the same two terms Env.step charges. It now DOMINATES the objective -- 78% of
        # it, against 21% for travel and 0.7% for delays -- so leaving it out of feature 5 meant
        # the agent chose its factory on the two smallest components.
        #
        # In practice the FAL term is almost always zero: get_start_date never starts before the
        # ideal date, so nothing arrives early, and 74% of engines arrive late anyway. What is
        # left is rate x queue wait, which makes congestion a direct cost rather than a proxy
        # for lateness.
        rate = self.env.engines[engine_index].engine_type_object.cost_storage_per_time
        storage = rate * (np.maximum(0.0, start - delivery)
                          + np.maximum(0.0, deadline - arrival))

        feats = np.concatenate([
            np.stack([
                prep / self.max_prep,                                   # 0 preparation duration
                env.capacity / self.max_capacity_value,                 # 1 capacity
                np.maximum(0.0, earliest - delivery) / self.horizon,    # 2 CONGESTION: the wait
                travel_cost / self.max_cost,                            # 3 EXACT travel cost
                travel_time / self.horizon,                             # 4 transit time
                # 5 IMMEDIATE reward -- exactly what env.step will pay, weights included
                -(WEIGHT_TRAVEL * travel_cost + WEIGHT_DELAY * delay
                  + WEIGHT_STORAGE * storage) / self.reward_scale,
                delay / self.horizon,                                   # 6 EXACT delay
                storage / self.reward_scale,                            # 7 EXACT storage
            ], axis=1),
            stations / self.horizon,                                    # 8.. station dates
        ], axis=1)
        return feats.astype(np.float32), indices

    def state_features(self, state):
        """
        The critic's view: one fixed-size vector describing the whole state.

        Two rules behind every term. Everything must be known BEFORE the action is chosen, and
        nothing may scale with instance size -- hence min/mean/max statistics rather than one
        input per entity. That is what lets a single network train on medium and run on xlarge,
        and serve both engine types (8 stations on one, 10 on the other).

        The dominant term is HOW MUCH IS LEFT. The return is roughly (steps remaining) x
        (average cost per step), so it spans an order of magnitude across an episode. Without
        the remaining fractions and the progress, V cannot predict the return at all, and the
        advantage ends up measuring "where am I in the episode" rather than "was that action
        good" -- which is precisely how the earlier attempt failed.

        The travel tables are static, so they carry no information WITHIN an episode. They are
        here to condition on the INSTANCE: the generator redraws travel prices for every
        replicate, and a memoryless network cannot learn a quantity that changes between
        instances -- it has to come in through the input.
        """
        env = self.env
        engine_free, msn_free = self.occupancy(state)

        def stats(values, empty_value=0.0):
            """min/mean/max -- size-invariant by construction. Empty set -> neutral constant."""
            if len(values) == 0:
                return np.full(3, empty_value)
            return np.array([values.min(), values.mean(), values.max()])

        station_dates = np.fromiter(state.stationAvailableDate.values(), dtype=float)

        parts = [
            stats(station_dates) / self.horizon,                          # 3  queue state
            env.travel_cost.ravel() / self.max_cost,                      # 16 instance
            env.travel_time.ravel() / self.horizon,                       # 16   conditioning
            np.array([engine_free.sum() / len(env.engines),               # 2  what is left
                      msn_free.sum() / len(env.msns)]),
            np.array([len(state.engineToMsn) / max(env.max_steps, 1)]),   # 1  progress
            stats(env.deadline[msn_free], self.horizon) / self.horizon,   # 3  pressure left
            stats(env.delivery[engine_free]) / self.horizon,              # 3  supply left
        ]
        return np.concatenate(parts).astype(np.float32)