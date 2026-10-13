import math

import didppy as dp


class DidpModel:
    def __init__(
        self,
        nb_engine,
        nb_factory,
        nb_msn,
        release_engine,
        msn_inst_date,
        factory_msn_distance,
        factory_capacity,
        factory_prep_duration,
        factory_msn_travel_cost,
        engine_storage_cost,
        engine_msn_compatible,
        engine_factory_compatible,
        horizon,
    ):
        self.nb_engine = nb_engine
        self.nb_factory = nb_factory
        self.nb_msn = nb_msn
        self.release_engine = release_engine
        self.msn_inst_date = msn_inst_date
        self.factory_msn_distance = factory_msn_distance
        self.factory_capacity = factory_capacity
        self.factory_prep_duration = factory_prep_duration
        self.factory_msn_travel_cost = factory_msn_travel_cost
        self.engine_storage_cost = engine_storage_cost
        self.engine_msn_compatible = engine_msn_compatible
        self.engine_factory_compatible = engine_factory_compatible
        self.horizon = horizon
        self.model = dp.Model(maximize=False, float_cost=True)

    def add_tables(self):
        self.release_engine_table = self.model.add_float_table(self.release_engine)
        self.msn_inst_date_table = self.model.add_float_table(self.msn_inst_date)
        self.factory_msn_distance_table = self.model.add_float_table(self.factory_msn_distance)
        self.factory_capacity_table = self.model.add_int_table(self.factory_capacity)
        self.factory_prep_duration_table = self.model.add_float_table(self.factory_prep_duration)
        self.factory_msn_travel_cost_table = self.model.add_float_table(self.factory_msn_travel_cost)
        self.engine_storage_cost_table = self.model.add_float_table(self.engine_storage_cost)
        self.leftover_storage_table = self.model.add_float_table([
            h * max(0.0, self.horizon - r) for h, r in zip(self.engine_storage_cost, self.release_engine)
        ])

    def add_variables(self):
        self.engine = self.model.add_object_type(number=self.nb_engine)
        self.factory = self.model.add_object_type(number=self.nb_factory)
        self.msn = self.model.add_object_type(number=self.nb_msn)
        self.phase =self.model.add_int_var(target = 0)


        self.unchosen_msn = self.model.add_set_var(object_type=self.msn, target=list(range(self.nb_msn)))
        self.unchosen_engine = self.model.add_set_var(object_type=self.engine, target=list(range(self.nb_engine)))
        self.factory_select = self.model.add_element_var(object_type=self.factory, target=0)
        self.engine_select = self.model.add_element_var(object_type=self.engine, target=0)
        self.msn_select = self.model.add_element_var(object_type=self.msn, target=0)

        self.station_free = [
            [self.model.add_float_resource_var(target=0.0, less_is_better=True) for _ in range(self.factory_capacity[f])]
            for f in range(self.nb_factory)
        ]

    def is_earliest_station(self, f, k):
        stations = self.station_free[f]
        before = [stations[k] < stations[j] for j in range(k)]
        after = [stations[k] <= stations[j] for j in range(k + 1, len(stations))]
        return before + after

    def add_transition(self):
        for e in range(self.nb_engine):
            for f in range(self.nb_factory):
                if not self.engine_factory_compatible[e][f]:
                    continue
                for m in range(self.nb_msn):
                    if not self.engine_msn_compatible[e][m]:
                        continue
                    for k in range(self.factory_capacity[f]):
                        self.add_assignment(e, f, m, k)

        self.model.add_base_case([self.unchosen_engine.is_empty()])
        self.model.add_base_case(
            [self.unchosen_msn.is_empty()],
            cost=self.leftover_storage_table[self.unchosen_engine],
        )

    def add_multi_phase_transition(self):
        for e in range(self.nb_engine):
            self.engine_selection(e)
        for m in range(self.nb_msn):
            self.msn_selection(m)
        for f in range(self.nb_factory):
            for k in range(self.factory_capacity[f]):
                self.factory_selection(f, k)

        self.model.add_base_case([self.phase == 0, self.unchosen_engine.is_empty()])
        self.model.add_base_case(
            [self.phase == 0, self.unchosen_msn.is_empty()],
            cost=self.leftover_storage_table[self.unchosen_engine],
        )

    def engine_selection(self, e):
        transition = dp.Transition(
            name=f"select_engine e={e}",
            cost=dp.FloatExpr.state_cost(),
            effects=[
                (self.unchosen_engine, self.unchosen_engine.remove(e)),
                (self.engine_select, e),
                (self.phase, 1),
            ],
            preconditions=[
                self.unchosen_engine.contains(e),
                self.phase == 0,
            ],
        )
        self.model.add_transition(transition)

    def msn_selection(self, m):
        transition = dp.Transition(
            name=f"select_msn m={m}",
            cost=dp.FloatExpr.state_cost(),
            effects=[
                (self.unchosen_msn, self.unchosen_msn.remove(m)),
                (self.msn_select, m),
                (self.phase, 2),
            ],
            preconditions=[
                self.unchosen_msn.contains(m),
                self.phase == 1,
            ],
        )
        self.model.add_transition(transition)

    def factory_selection(self, f, k):
        d_m = self.msn_inst_date_table[self.msn_select]
        theta = self.factory_msn_distance_table[f, self.msn_select]
        p_f = self.factory_prep_duration_table[f]
        r_e = self.release_engine_table[self.engine_select]
        h_e = self.engine_storage_cost_table[self.engine_select]
        c_fa = self.factory_msn_travel_cost_table[f, self.msn_select]
        sigma = self.station_free[f][k]

        start = dp.max(dp.max(d_m - theta - p_f, dp.float(math.ceil(sigma))), r_e)
        arrival = start + p_f + theta

        delay = dp.max(arrival - d_m, 0)
        storage = h_e * dp.max(start - r_e, 0.0)

        transition = dp.Transition(
            name=f"select_factory f={f} k={k}",
            cost=delay + c_fa + storage + dp.FloatExpr.state_cost(),
            effects=[
                (sigma, start + p_f),
                (self.engine_select, 0),
                (self.msn_select, 0),
                (self.phase, 0),
            ],
            preconditions=[self.phase == 2] + self.is_earliest_station(f, k),
        )
        self.model.add_transition(transition)

    def add_assignment(self, e, f, m, k):
        d_m = self.msn_inst_date_table[m]
        theta = self.factory_msn_distance_table[f, m]
        p_f = self.factory_prep_duration_table[f]
        r_e = self.release_engine_table[e]
        h_e = self.engine_storage_cost_table[e]
        c_fa = self.factory_msn_travel_cost_table[f, m]
        sigma = self.station_free[f][k]

        start = dp.max(dp.max(d_m - theta - p_f, dp.float(math.ceil(sigma))), r_e)
        arrival = start + p_f + theta

        delay = dp.max(arrival - d_m, 0)
        storage = h_e * dp.max(start - r_e, 0.0)

        transition = dp.Transition(
            name=f"assign e={e} f={f} m={m} k={k}",
            cost=delay + c_fa + storage + dp.FloatExpr.state_cost(),
            effects=[
                (self.unchosen_engine, self.unchosen_engine.remove(e)),
                (self.unchosen_msn, self.unchosen_msn.remove(m)),
                (sigma, start + p_f),
            ],
            preconditions=[
                self.unchosen_engine.contains(e),
                self.unchosen_msn.contains(m),
            ] + self.is_earliest_station(f, k),
        )
        self.model.add_transition(transition)
