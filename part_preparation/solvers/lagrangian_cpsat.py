"""
MSN-centric Aggregated Lagrangian Relaxation Solver
Compatible with the discrete-optimization pipeline.
"""
import time
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from ortools.sat.python import cp_model
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction

from part_preparation.problem import (
    EnginePreparationProblem,
    EnginePreparationSolution,
    Engine,
    MSN,
    PreparationFactory,
)

logger = logging.getLogger(__name__)


class LagrangianResultStorage:
    """Simule le ResultStorage requis par le script de visualisation."""
    def __init__(self, solution: EnginePreparationSolution):
        self.solution = solution

    def get_best_solution(self) -> EnginePreparationSolution:
        return self.solution


class CPSatEngineSolver:
    """
    Solveur par décomposition et relaxation lagrangienne par MSN.
    Nommé à l'identique pour un remplacement transparent par l'import.
    """
    def __init__(
        self,
        problem: EnginePreparationProblem,
        params_objective_function: ParamsObjectiveFunction = None,
        **kwargs: Any
    ):
        self.problem: EnginePreparationProblem = problem
        self.params_objective_function = params_objective_function
        
        # Agrégation des types de moteurs et calcul des quotas (stocks)
        self.engine_types = list(set(e.engine_type for e in problem.engines_list))
        self.total_stock = {
            t: sum(1 for e in problem.engines_list if e.engine_type == t) 
            for t in self.engine_types
        }
        self.scale_factor = 100.0
        self.lambdas: Dict[Tuple[str, int], float] = {}
        self.gammas: Dict[str, float] = {}

    def init_model(self, **kwargs: Any) -> None:
        """Initialise les tables de prix duaux (Lagrange)."""
        self.lambdas = {
            (f.id, t): 0.0 
            for f in self.problem.factories_list 
            for t in range(self.problem.time_horizon)
        }
        self.gammas = {t: 0.0 for t in self.engine_types}

    def _solve_msn_subproblem(self, msn: MSN, sub_timeout: float) -> Dict[str, Any]:
        """Résout le sous-problème CP-SAT d'un unique MSN."""
        model = cp_model.CpModel()
        
        is_allocated = model.NewBoolVar(f"allocated_msn_{msn.id}")
        assign_type = {t: model.NewBoolVar(f"choose_type_{t}") for t in self.engine_types}
        assign_f = {f: model.NewBoolVar(f"choose_f_{f.id}") for f in self.problem.factories_list}
        
        prep_start = model.NewIntVar(0, self.problem.time_horizon, f"start_msn_{msn.id}")
        prep_end = model.NewIntVar(0, self.problem.time_horizon, f"end_msn_{msn.id}")
        arrival = model.NewIntVar(0, self.problem.time_horizon, f"arrival_msn_{msn.id}")
        
        model.AddAtMostOne(assign_type.values())
        model.AddAtMostOne(assign_f.values())
        model.Add(sum(assign_type.values()) == 1).OnlyEnforceIf(is_allocated)
        model.Add(sum(assign_f.values()) == 1).OnlyEnforceIf(is_allocated)
        model.Add(sum(assign_type.values()) == 0).OnlyEnforceIf(is_allocated.negated())
        model.Add(sum(assign_f.values()) == 0).OnlyEnforceIf(is_allocated.negated())

        assembly_line = self.problem.assembly_lines[msn.assembly_line_id]
        
        for t_name in self.engine_types:
            for f in self.problem.factories_list:
                both = model.NewBoolVar(f"both_{t_name}_{f.id}")
                model.AddBoolAnd([assign_type[t_name], assign_f[f]]).OnlyEnforceIf(both)
                
                duration = f.get_preparation_duration(t_name)
                model.Add(prep_end == prep_start + duration).OnlyEnforceIf(both)
                
                travel_time = self.problem.get_travel_time(f, assembly_line)
                model.Add(arrival == prep_end + travel_time).OnlyEnforceIf(both)

        obj_terms = []
        
        # Objectif métier : Retard par rapport à l'installation planifiée
        lateness = model.NewIntVar(0, self.problem.time_horizon, "lateness")
        model.Add(lateness >= arrival - msn.engine_installation_date).OnlyEnforceIf(is_allocated)
        obj_terms.append(1000 * lateness + 100000*(1- is_allocated))
        
        # Pénalité Stock non renouvelable (Type de moteur)
        for t_name, var_type in assign_type.items():
            gamma_val = self.gammas.get(t_name, 0.0)
            scaled_gamma = int(round(gamma_val * self.scale_factor))
            if scaled_gamma > 0:
                obj_terms.append(scaled_gamma * var_type)
                
        # Pénalité Capacité renouvelable (Usine au temps t)
        for f in self.problem.factories_list:
            for t in range(self.problem.time_horizon):
                lambda_val = self.lambdas.get((f.id, t), 0.0)
                if lambda_val <= 1e-5:
                    continue
                is_active_at_t = model.NewBoolVar(f"active_f{f.id}_t{t}")
                after_start = model.NewBoolVar(f"after_t{t}")
                before_end = model.NewBoolVar(f"before_t{t}")
                model.Add(prep_start <= t).OnlyEnforceIf(after_start)
                model.Add(prep_start > t).OnlyEnforceIf(after_start.negated())
                model.Add(prep_end > t).OnlyEnforceIf(before_end)
                model.Add(prep_end <= t).OnlyEnforceIf(before_end.negated())
                model.AddBoolAnd([after_start, before_end, assign_f[f]]).OnlyEnforceIf(is_active_at_t)
                
                scaled_lambda = int(round(lambda_val * self.scale_factor))
                if scaled_lambda > 0:
                    obj_terms.append(scaled_lambda * is_active_at_t)

        if obj_terms:
            model.Minimize(sum(obj_terms))
            
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = sub_timeout
        status = solver.Solve(model)
        
        res = {"allocated": False, "engine_type": None, "factory": None, "start": 0, "end": 0, "arrival": 0}
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) and solver.Value(is_allocated) == 1:
            res["allocated"] = True
            res["start"] = solver.Value(prep_start)
            res["end"] = solver.Value(prep_end)
            res["arrival"] = solver.Value(arrival)
            res["engine_type"] = next((t for t in self.engine_types if solver.Value(assign_type[t]) == 1), None)
            res["factory"] = next((f for f in self.problem.factories_list if solver.Value(assign_f[f]) == 1), None)
        return res

    def solve(self, time_limit: int = 300, max_iters: int = 40, **kwargs: Any) -> LagrangianResultStorage:
        """Boucle duale adaptative respectant l'enveloppe de temps globale."""
        t0 = time.perf_counter()
        step_size = 4.0
        decay = 0.93
        msn_decisions = {}

        for iteration in range(max_iters):
            # Calcul du temps restant pour éviter les coupures brutes
            elapsed = time.perf_counter() - t0
            remaining_time = time_limit - elapsed
            if remaining_time <= 5.0:
                logger.warning(f"Temps limite presque atteint ({elapsed:.1f}s). Arrêt de la boucle duale.")
                break

            type_consumption = {t: 0 for t in self.engine_types}
            factory_load_profiles = {f.id: np.zeros(self.problem.time_horizon) for f in self.problem.factories_list}
            msn_decisions = {}

            # Calcul du budget temps alloué à chaque sous-problème MSN pour cette itération
            sub_timeout = max(30, min(0.5, remaining_time / len(self.problem.msns_list)))

            for msn in self.problem.msns_list:
                dec = self._solve_msn_subproblem(msn, sub_timeout)
                msn_decisions[msn] = dec
                if dec["allocated"]:
                    type_consumption[dec["engine_type"]] += 1
                    factory_load_profiles[dec["factory"].id][dec["start"]:dec["end"]] += 1

            # Ajustement des sous-gradients (Pénalités)
            for t in self.engine_types:
                viol_stock = max(0, type_consumption[t] - self.total_stock[t])
                self.gammas[t] = max(0.0, self.gammas[t] + step_size * viol_stock)
                
            for f in self.problem.factories_list:
                if f.default_capacity is None:
                    continue
                for t in range(self.problem.time_horizon):
                    viol_fac = max(0.0, factory_load_profiles[f.id][t] - f.default_capacity)
                    self.lambdas[(f.id, t)] = max(0.0, self.lambdas[(f.id, t)] + step_size * viol_fac)

            step_size *= decay

        # --- Reconstruction physique finale ---
        engine_to_msn = {}
        engine_to_factory = {}
        prep_start_times = {}
        prep_end_times = {}
        fal_arrival_times = {}
        delivery_time = {}

        available_engines_by_type = {
            t: [e for e in self.problem.engines_list if e.engine_type == t] 
            for t in self.engine_types
        }

        for msn, dec in msn_decisions.items():
            if dec["allocated"] and available_engines_by_type[dec["engine_type"]]:
                engine = available_engines_by_type[dec["engine_type"]].pop(0)
                engine_to_msn[engine] = msn
                engine_to_factory[engine] = dec["factory"]
                prep_start_times[engine] = dec["start"]
                prep_end_times[engine] = dec["end"]
                fal_arrival_times[engine] = dec["arrival"]
                delivery_time[engine] = int(engine.delivery_time) if engine.delivery_time is not None else 0

        # Moteurs physiques restants
        for t, engines in available_engines_by_type.items():
            for engine in engines:
                prep_start_times[engine] = 0
                prep_end_times[engine] = 0
                fal_arrival_times[engine] = 0
                delivery_time[engine] = int(engine.delivery_time) if engine.delivery_time is not None else 0

        final_solution = EnginePreparationSolution(
            problem=self.problem,
            engine_to_msn=engine_to_msn,
            engine_to_factory=engine_to_factory,
            prep_start_times=prep_start_times,
            prep_end_times=prep_end_times,
            fal_arrival_times=fal_arrival_times,
            delivery_time=delivery_time
        )

        return LagrangianResultStorage(final_solution)