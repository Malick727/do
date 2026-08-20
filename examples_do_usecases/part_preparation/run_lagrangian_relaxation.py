import logging
import sys
import os
import numpy as np
from pathlib import Path
from datetime import datetime

# Alignement du sys.path pour les imports locaux
root_path = Path(__file__).parent.parent.parent
sys.path.append(str(root_path))

from part_preparation.paths import SKYWISE_CSV
from part_preparation.instance_generator import generate_instance_from_csv
from part_preparation.problem import (
    EnginePreparationProblem,
    EnginePreparationSolution,
    Engine,
    MSN,
    PreparationFactory,
)
from discrete_optimization.generic_tools.do_problem import ParamsObjectiveFunction, ObjectiveHandling, ModeOptim

# On ré-importe le micro-solveur et l'orchestrateur définis précédemment
from ortools.sat.python import cp_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── 1. Rappel des composants de la Décomposition Lagrangianne ─────────────────

class EngineLagrangianMicroSolver:
    """Solveur CP-SAT local et isolé pour un unique moteur."""
    def __init__(self, problem: EnginePreparationProblem, engine: Engine, lambdas: dict, mus: dict, scale_factor: float = 100.0):
        self.problem = problem
        self.engine = engine
        self.lambdas = lambdas
        self.mus = mus
        self.scale_factor = scale_factor
        self.model = cp_model.CpModel()
        
        self.is_allocated = self.model.NewBoolVar(f"allocated_{engine.id}")
        self.compatible_msns = self.problem.engine_compatible_msns.get(engine, [])
        self.assign_m = {msn: self.model.NewBoolVar(f"assign_m_{msn.id}") for msn in self.compatible_msns}
        self.compatible_factories = self.problem.engine_compatible_factories.get(engine, [])
        self.assign_f = {fac: self.model.NewBoolVar(f"assign_f_{fac.id}") for fac in self.compatible_factories}
        
        self.prep_start = self.model.NewIntVar(0, self.problem.time_horizon, f"start_{engine.id}")
        self.prep_end = self.model.NewIntVar(0, self.problem.time_horizon, f"end_{engine.id}")
        self.arrival = self.model.NewIntVar(0, self.problem.time_horizon, f"arrival_{engine.id}")

    def build_and_solve(self) -> dict:
        self.model.AddAtMostOne(self.assign_m.values())
        self.model.AddAtMostOne(self.assign_f.values())
        self.model.Add(sum(self.assign_m.values()) == 1).OnlyEnforceIf(self.is_allocated)
        self.model.Add(sum(self.assign_m.values()) == 0).OnlyEnforceIf(self.is_allocated.negated())
        self.model.Add(sum(self.assign_f.values()) == 1).OnlyEnforceIf(self.is_allocated)
        self.model.Add(sum(self.assign_f.values()) == 0).OnlyEnforceIf(self.is_allocated.negated())

        if self.engine.delivery_time is not None:
            self.model.Add(self.prep_start >= int(self.engine.delivery_time)).OnlyEnforceIf(self.is_allocated)
            
        for fac in self.compatible_factories:
            duration = fac.get_preparation_duration(self.engine.engine_type)
            self.model.Add(self.prep_end == self.prep_start + duration).OnlyEnforceIf(self.assign_f[fac])
            for msn in self.compatible_msns:
                assembly_line = self.problem.assembly_lines[msn.assembly_line_id]
                travel_time = self.problem.get_travel_time(fac, assembly_line)
                both = self.model.NewBoolVar(f"both_{fac.id}_{msn.id}")
                self.model.AddBoolAnd([self.assign_f[fac], self.assign_m[msn]]).OnlyEnforceIf(both)
                self.model.Add(self.arrival == self.prep_end + travel_time).OnlyEnforceIf(both)

        obj_terms = []
        for msn, var_m in self.assign_m.items():
            mu_val = self.mus.get(msn.id, 0.0)
            scaled_mu = int(round(mu_val * self.scale_factor))
            if scaled_mu > 0:
                obj_terms.append(scaled_mu * var_m)
                
        for fac in self.compatible_factories:
            for t in range(self.problem.time_horizon):
                lambda_val = self.lambdas.get((fac.id, t), 0.0)
                if lambda_val <= 1e-5:
                    continue
                is_active_at_t = self.model.NewBoolVar(f"active_f{fac.id}_t{t}")
                after_start = self.model.NewBoolVar(f"after_t{t}")
                before_end = self.model.NewBoolVar(f"before_t{t}")
                self.model.Add(self.prep_start <= t).OnlyEnforceIf(after_start)
                self.model.Add(self.prep_start > t).OnlyEnforceIf(after_start.negated())
                self.model.Add(self.prep_end > t).OnlyEnforceIf(before_end)
                self.model.Add(self.prep_end <= t).OnlyEnforceIf(before_end.negated())
                self.model.AddBoolAnd([after_start, before_end, self.assign_f[fac]]).OnlyEnforceIf(is_active_at_t)
                
                scaled_lambda = int(round(lambda_val * self.scale_factor))
                if scaled_lambda > 0:
                    obj_terms.append(scaled_lambda * is_active_at_t)

        if obj_terms:
            self.model.Minimize(sum(obj_terms))
            
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 1.0
        status = solver.Solve(self.model)
        
        res = {"allocated": False, "msn": None, "factory": None, "start": 0, "end": 0, "arrival": 0}
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) and solver.Value(self.is_allocated) == 1:
            res["allocated"] = True
            res["start"] = solver.Value(self.prep_start)
            res["end"] = solver.Value(self.prep_end)
            res["arrival"] = solver.Value(self.arrival)
            res["msn"] = next((m for m in self.compatible_msns if solver.Value(self.assign_m[m]) == 1), None)
            res["factory"] = next((f for f in self.compatible_factories if solver.Value(self.assign_f[f]) == 1), None)
        return res


def solve_full_decomposition_lagrangian(problem: EnginePreparationProblem, max_iters: int = 5, step_size: float = 2.0, decay: float = 0.90):
    """Orchestrateur global de la décomposition."""
    lambdas = {}
    for fac in problem.factories_list:
        if fac.default_capacity is not None:
            for t in range(problem.time_horizon):
                lambdas[(fac.id, t)] = 0.0
                
    mus = {msn.id: 0.0 for msn in problem.msns_list}

    for iteration in range(max_iters):
        print(f"  -> Itération Lagrangienne {iteration+1}/{max_iters} (Step Size: {step_size:.3f})")
        msn_usage_counts = {msn.id: 0 for msn in problem.msns_list}
        factory_load_profiles = {fac.id: np.zeros(problem.time_horizon) for fac in problem.factories_list}
        current_iteration_solutions = {}

        # Résolution isolée de chaque moteur
        for engine in problem.engines_list:
            micro_solver = EngineLagrangianMicroSolver(problem, engine, lambdas, mus)
            decision = micro_solver.build_and_solve()
            current_iteration_solutions[engine] = decision
            
            if decision["allocated"]:
                msn_usage_counts[decision["msn"].id] += 1
                start_t, end_t = decision["start"], decision["end"]
                factory_load_profiles[decision["factory"].id][start_t:end_t] += 1

        # Mise à jour des multiplicateurs par sous-gradient
        for msn in problem.msns_list:
            violation_msn = msn_usage_counts[msn.id] - 1
            mus[msn.id] = max(0.0, mus[msn.id] + step_size * violation_msn)
            
        for fac in problem.factories_list:
            if fac.default_capacity is None:
                continue
            for t in range(problem.time_horizon):
                violation_cap = factory_load_profiles[fac.id][t] - fac.default_capacity
                lambdas[(fac.id, t)] = max(0.0, lambdas[(fac.id, t)] + step_size * violation_cap)

        step_size *= decay
        max_msn_conflict = max(msn_usage_counts.values(), default=0)
        print(f"     Conflit MSN max : {max_msn_conflict} moteurs sur une même affectation")

    return current_iteration_solutions


# ── 2. Script de Run de Test Principal ─────────────────────────────────────────

def main():
    csv_path = str(SKYWISE_CSV)
    start_filter = datetime(2026, 9, 15, 0, 0)
    end_filter = datetime(2027, 10, 1, 23, 59)
    
    print("⏳ Chargement de l'instance depuis le fichier CSV...")
    problem = generate_instance_from_csv(
        csv_file_path=csv_path,
        start_date=start_filter,
        end_date=end_filter
    )
    
    print(f"\nTaille du problème chargé :")
    print(f"  - {len(problem.assembly_lines_list)} lignes d'assemblage")
    print(f"  - {len(problem.msns_list)} MSNs")
    print(f"  - {len(problem.engines_list)} moteurs")
    print(f"  - {len(problem.factories_list)} usines de préparation")
    print(f"  - Horizon de temps : {problem.time_horizon} jours")

    print("\n🚀 Lancement du test de relaxation lagrangienne par blocs de moteurs...")
    # Lancement sur un petit nombre d'itérations pour valider l'exécution
    raw_decisions = solve_full_decomposition_lagrangian(problem, max_iters=5, step_size=5.0)

    print("\n🧩 Reconstruction de la solution globale pour évaluation...")
    
    # Extraction et re-mappage vers les structures d'origine du framework
    engine_to_msn = {}
    engine_to_factory = {}
    prep_start_times = {}
    prep_end_times = {}
    fal_arrival_times = {}
    delivery_time = {}

    for engine, dec in raw_decisions.items():
        # Remplissage par défaut requis par l'objet EnginePreparationSolution
        delivery_time[engine] = int(engine.delivery_time) if engine.delivery_time is not None else 0
        prep_start_times[engine] = dec["start"]
        prep_end_times[engine] = dec["end"]
        fal_arrival_times[engine] = dec["arrival"]
        
        if dec["allocated"]:
            engine_to_msn[engine] = dec["msn"]
            engine_to_factory[engine] = dec["factory"]

    # Instanciation de l'objet solution de discrete-optimization
    solution = EnginePreparationSolution(
        problem=problem,
        engine_to_msn=engine_to_msn,
        engine_to_factory=engine_to_factory,
        prep_start_times=prep_start_times,
        prep_end_times=prep_end_times,
        fal_arrival_times=fal_arrival_times,
        delivery_time=delivery_time
    )

    print("\n" + "=" * 80)
    print("RÉSULTATS DU PETIT TEST LAGRANGIEN")
    print("=" * 80)

    feasible = problem.satisfy(solution)
    kpis = problem.evaluate(solution)

    print(f"Faisabilité stricte de la solution : {'✅ Validée' if feasible else '❌ Non validée (Conflits résiduels)'}")
    print("\nValeurs des indicateurs de performance (KPIs) :")
    for name, val in kpis.items():
        print(f"  - {name:20s}: {val:.2f}")

    allocated_count = len(engine_to_msn)
    print(f"\nTaux d'allocation obtenu : {allocated_count}/{len(problem.engines_list)} moteurs planifiés.")

if __name__ == "__main__":
    main()