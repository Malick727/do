#  Copyright (c) 2024 AIRBUS and its affiliates.
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import logging
from collections import defaultdict
from typing import Any, List, Optional

import clingo

from discrete_optimization.generic_tools.asp_tools import AspClingoSolver
from discrete_optimization.generic_tools.result_storage.result_storage import ResultStorage
from multibatching.problem import (
    MultibatchingProblem,
    MultibatchingSolution,
    PackingTransport,
)

logger = logging.getLogger(__name__)


class AspMultibatchingSolver(AspClingoSolver):
    """
    Solver based on Answer Set Programming formulation and clingo solver.
    It solves the Multibatching problem using a Flow formulation (aggregated items).
    """
    problem: MultibatchingProblem

    def __init__(self, problem: MultibatchingProblem, **kwargs: Any):
        super().__init__(problem, **kwargs)
        self.scaling_factor = kwargs.get("scaling_factor", 100)

    def init_model(self, **kwargs: Any) -> None:
        # Default bounds (ensure these are integers)
        max_flow = int(kwargs.get("max_flow_per_product", 50))
        max_trips = int(kwargs.get("max_trips_per_link", 20))

        # ASP Program
        # Note: We use {{ }} for literal braces in f-strings.
        # We added keys (e.g., ", P") to #sum aggregates to ensure distinct items are summed correctly.
        basic_model = f"""
        #const max_flow={max_flow}.
        #const max_trips={max_trips}.

        % --- Generation ---

        % For every valid link-product combination, choose exactly one flow quantity Q.
        1 {{ flow(L, P, Q) : Q=0..max_flow }} 1 :- valid_transport(L, P).

        % --- Constraints ---

        % 1. Flow Conservation
        % Calculate Total Inflow for a (Node, Product)
        flow_in(Node, P, TotalIn) :- 
            location(Node), product(P, _, _),
            TotalIn = #sum {{ Q, L : flow(L, P, Q), link_dest(L, Node) }}.

        % Calculate Total Outflow for a (Node, Product)
        flow_out(Node, P, TotalOut) :- 
            location(Node), product(P, _, _),
            TotalOut = #sum {{ Q, L : flow(L, P, Q), link_src(L, Node) }}.

        % Enforce: In + Supply == Out
        :- location(Node), product(P, _, _), 
           net_supply(Node, P, Supply), 
           flow_in(Node, P, In), 
           flow_out(Node, P, Out), 
           In + Supply != Out.

        % 2. Capacity and Trips
        % Calculate volume used on link L. 
        % Key 'P' is added to #sum tuple {{ Q*Size, P : ... }} to avoid merging identical values.
        volume_on_link(L, Vol) :- 
            link(L, _, _, _, _, _, _),
            Vol = #sum {{ Q * Size, P : flow(L, P, Q), product(P, Size, _) }}.

        % Determine number of trips needed (Integer Ceiling Division).
        % Trips >= Volume / Capacity  =>  Trips = (Volume + Capacity - 1) / Capacity
        trips(L, T) :- 
            link(L, _, _, Cap, _, _, _), Cap > 0,
            volume_on_link(L, Vol),
            T = (Vol + Cap - 1) / Cap.

        % Enforce max trips bound
        :- trips(L, T), T > max_trips.

        % --- Objective Function ---

        % Transport Cost: Sum of (Trips * Distance * CostFactor) for each Link L
        cost_transport(Cost) :- 
            Cost = #sum {{ T * Dist * C, L : trips(L, T), link(L, _, _, _, C, _, Dist) }}.

        % Emission Cost: Sum of (Trips * Distance * EmissionFactor) for each Link L
        cost_emission(Cost) :- 
            Cost = #sum {{ T * Dist * E, L : trips(L, T), link(L, _, _, _, _, E, Dist) }}.

        % Holding/Capital Cost: Sum of (Quantity * HoldingCoeff) for each Link L, Product P
        cost_holding(Cost) :-
            Cost = #sum {{ Q * K, L, P : flow(L, P, Q), holding_coeff(L, P, K) }}.

        % Total Objective to Minimize
        total_cost(C) :- 
            cost_transport(CT), cost_emission(CE), cost_holding(CH),
            C = CT + CE + CH.

        #minimize {{ C : total_cost(C) }}.

        #show flow/3.
        #show trips/2.
        """

        self.ctl = clingo.Control(
            ["--warn=no-atom-undefined", "--opt-mode=optN"]
        )
        self.ctl.add("base", [], basic_model)

        # Add the data facts
        string_data_input = self.build_string_data_input()
        self.ctl.add("base", [], string_data_input)

    def build_string_data_input(self) -> str:
        facts = []

        # 1. Products: product(ID, Size, Value).
        # Force integer casting to avoid "10.0" which crashes Clingo
        for p in self.problem.products:
            p_idx = self.problem.product_to_index[p]
            facts.append(f"product({p_idx}, {int(p.size)}, {int(p.value)}).")

        # 2. Locations: location(ID).
        for loc in self.problem.locations:
            l_idx = self.problem.locations_to_index[loc]
            facts.append(f"location({l_idx}).")

            # Net Supply: net_supply(LocID, ProdID, Amount).
            for p, amount in loc.net_supply.items():
                if amount != 0:
                    p_idx = self.problem.product_to_index[p]
                    # 'amount' can sometimes be float in input data, force int
                    facts.append(f"net_supply({l_idx}, {p_idx}, {int(amount)}).")

        # 3. Links: link(ID, Src, Dst, Cap, Cost, Emission, Dist).
        for link in self.problem.transport_links:
            l_idx = self.problem.transport_links_to_index[link]
            src_idx = self.problem.locations_to_index[link.location_l1]
            dst_idx = self.problem.locations_to_index[link.location_l2]

            tt = link.transport_type

            # Helper predicates
            facts.append(f"link_src({l_idx}, {src_idx}).")
            facts.append(f"link_dest({l_idx}, {dst_idx}).")

            # Main link fact: Force int() on ALL numerical fields
            facts.append(f"link({l_idx}, {src_idx}, {dst_idx}, "
                         f"{int(tt.capacity)}, {int(tt.cost)}, {int(tt.emissions)}, {int(link.distance)}).")

            # 4. Valid Transports & Holding Costs
            # Calculate time safely
            speed = tt.speed if tt.speed > 0 else 0.0001  # avoid div by zero
            time = link.distance / speed

            for p in self.problem.products:
                if tt in p.valid_transports and tt.capacity >= p.size:
                    p_idx = self.problem.product_to_index[p]
                    facts.append(f"valid_transport({l_idx}, {p_idx}).")

                    # Pre-calculate holding cost coefficient
                    # Formula: Value * ((1+alpha)^time - 1)
                    capital_factor = self.problem.capital_factor
                    holding_val = p.value * ((1 + capital_factor) ** time - 1)

                    # Scale and cast to int
                    holding_coeff = int(holding_val * self.scaling_factor)
                    facts.append(f"holding_coeff({l_idx}, {p_idx}, {holding_coeff}).")

        return " ".join(facts)

    def retrieve_solution(self, model: clingo.Model) -> MultibatchingSolution:
        symbols = model.symbols(atoms=True)

        # Parse flows: flow(LinkIdx, ProdIdx, Qty)
        flows = {}
        trips = {}

        for atom in symbols:
            if atom.name == "flow":
                # args: L, P, Q
                l_idx = atom.arguments[0].number
                p_idx = atom.arguments[1].number
                qty = atom.arguments[2].number
                if qty > 0:
                    if l_idx not in flows:
                        flows[l_idx] = {}
                    flows[l_idx][self.problem.products[p_idx]] = qty

            if atom.name == "trips":
                # args: L, T
                l_idx = atom.arguments[0].number
                t_count = atom.arguments[1].number
                trips[l_idx] = t_count

        # Reconstruct PackingTransport objects
        packing_solution = []
        for l_idx, product_quantities in flows.items():
            nb_trips = trips.get(l_idx, 1)  # Should exist if flow > 0
            if nb_trips == 0:
                continue

            # We create a homogenized packing where every trip on this link
            # carries the average amount of products. 
            # (Similar to CpsatMultibatchingSolver.retrieve_solution Flow mode)
            cur_packing = {}
            for prod, total_qty in product_quantities.items():
                cur_packing[prod] = total_qty / nb_trips

            if len(cur_packing) > 0:
                link = self.problem.transport_links[l_idx]
                packing_solution.append(
                    PackingTransport(
                        transport_link=link,
                        product_packing=cur_packing,
                        nb_packing=nb_trips
                    )
                )
        print("hey")
        return MultibatchingSolution(problem=self.problem, list_flows=packing_solution)