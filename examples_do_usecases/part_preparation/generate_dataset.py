import random
import itertools
import json
import time
from ortools.sat.python import cp_model
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from part_preparation.instance_generator import generate_instance, InstanceParams, FactoryDefinition

# On reproduit EXACTEMENT les paramètres utilisés dans run_ga_and_visualize.py
manual_factories = [
    FactoryDefinition(engine_type="EngineType_1", home_fal_index=0, capacity=3, preparation_duration=11, location="Derby"),
    FactoryDefinition(engine_type="EngineType_1", home_fal_index=1, capacity=2, preparation_duration=10, location="Toulouse"),
    FactoryDefinition(engine_type="EngineType_1", home_fal_index=2, capacity=2, preparation_duration=13, location="Virginia"),
    FactoryDefinition(engine_type="EngineType_1", home_fal_index=None, capacity=1, preparation_duration=14, location="Singapore"),
    FactoryDefinition(engine_type="EngineType_2", home_fal_index=0, capacity=4, preparation_duration=15, location="Durham"),
    FactoryDefinition(engine_type="EngineType_2", home_fal_index=1, capacity=3, preparation_duration=15, location="Bordeaux"),
    FactoryDefinition(engine_type="EngineType_2", home_fal_index=2, capacity=1, preparation_duration=15, location="Cincinnati"),
    FactoryDefinition(engine_type="EngineType_2", home_fal_index=3, capacity=2, preparation_duration=12, location="Shanghai"),
]

TARGET_INSTANCE = InstanceParams(
    n_fals=4,
    n_msns_per_fal=40, # 40 MSNs per FAL (donne environ 162 moteurs)
    n_engine_types=2,
    time_horizon=300,
    fal_takt_paces=[2, 2, 5, 5],
    factory_definitions=manual_factories,
    engine_to_msn_ratio={"EngineType_1": 1.1, "EngineType_2": 0.95},
    supplier_regularity={"EngineType_1": 1.0, "EngineType_2": 0.8},
    supplier_production_rate={"EngineType_1": 1.0, "EngineType_2": 0.9},
    flexible_delivery_prob=0
)

def generate_random_allocation(problem) -> dict:
    allocation = {}
    msns_by_type = {}
    for m in problem.msns_list:
        msns_by_type.setdefault(m.engine_type, []).append(m)

    factories_by_type = {}
    for f in problem.factories_list:
        for et in f.compatible_engine_types:
            factories_by_type.setdefault(et, []).append(f)

    assigned_msns = set()
    engines = list(problem.engines_list)
    random.shuffle(engines)

    for engine in engines:
        etype = engine.engine_type
        available_msns = [m for m in msns_by_type.get(etype, []) if m.id not in assigned_msns]
        if available_msns:
            msn_id = random.choice(available_msns).id
            assigned_msns.add(msn_id)
        else:
            msn_id = None
        compatible_factories = factories_by_type.get(etype, [])
        factory_id = random.choice(compatible_factories).id if compatible_factories else None

        allocation[engine.id] = {
            "msn_id": msn_id,
            "factory_id": factory_id
        }
    return allocation

def evaluate_allocation_with_cpsat(problem, allocation) -> dict:
    model = cp_model.CpModel()
    max_time = int(problem.time_horizon + 200)
    start_vars, end_vars, interval_vars, delay_vars = {}, {}, {}, {}

    msn_dict = problem.msns
    factory_dict = problem.factories
    engine_dict = problem.engines
    assembly_lines_dict = problem.assembly_lines

    factory_intervals = {f_id: [] for f_id in factory_dict.keys()}
    factory_demands = {f_id: [] for f_id in factory_dict.keys()}

    fixed_travel_cost = 0.0
    allocated_msns = set()

    for engine_id, decisions in allocation.items():
        engine = engine_dict[engine_id]
        msn_id = decisions.get("msn_id")
        factory_id = decisions.get("factory_id")

        if msn_id is None or factory_id is None:
            continue

        allocated_msns.add(msn_id)
        msn = msn_dict[msn_id]
        factory = factory_dict[factory_id]
        assembly_line = assembly_lines_dict[msn.assembly_line_id]

        raw_min_start = engine.delivery_time if engine.delivery_time is not None else 0
        min_start = int(round(raw_min_start))

        S = model.NewIntVar(min_start, max_time, f"S_{engine_id}")
        E = model.NewIntVar(min_start, max_time, f"E_{engine_id}")

        prep_duration = int(round(factory.get_preparation_duration(engine.engine_type)))
        I = model.NewIntervalVar(S, prep_duration, E, f"I_{engine_id}")

        start_vars[engine_id] = S
        end_vars[engine_id] = E

        factory_intervals[factory_id].append(I)
        factory_demands[factory_id].append(1)

        travel_time = int(round(problem.get_travel_time(factory, assembly_line)))
        fixed_travel_cost += problem.get_travel_cost(factory, assembly_line)

        arrival_time = E + travel_time
        target_date = int(round(msn.engine_installation_date))

        D = model.NewIntVar(0, max_time, f"D_{engine_id}")
        model.AddMaxEquality(D, [0, arrival_time - target_date])
        delay_vars[engine_id] = D

    for factory_id, intervals in factory_intervals.items():
        if intervals:
            capacity = factory_dict[factory_id].default_capacity
            if capacity is not None:
                model.AddCumulative(intervals, factory_demands[factory_id], int(capacity))

    total_delay_penalty = sum(D for D in delay_vars.values())
    unallocated_count = len(msn_dict) - len(allocated_msns)
    unallocated_penalty = unallocated_count * 100000.0

    model.Minimize(total_delay_penalty)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.Solve(model)

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        total_score = solver.ObjectiveValue() + fixed_travel_cost + unallocated_penalty
        return {
            "is_valid": True,
            "total_cost": total_score
        }
        
    return {"is_valid": False, "total_cost": float('inf')}

def generate_full_dataset(output_file="dataset_matches.json"):
    # ON NE GENERE PLUS QU'UNE SEULE TAILLE !
    scenarios = [
        {"params": TARGET_INSTANCE, "count": 10, "name": "Target (Run GA Default)"}
    ]

    allocs_per_problem = 30
    dataset = []
    total_instances = sum(s["count"] for s in scenarios)
    current_instance = 1

    start_time = time.time()
    print(f"🚀 Lancement de la génération... (Cible : {total_instances} instances TARGET)")

    for scenario in scenarios:
        for _ in range(scenario["count"]):
            print(f"\n--- Instance {current_instance}/{total_instances} ---")
            problem = generate_instance(scenario["params"])

            evaluated = []
            for a_idx in range(allocs_per_problem):
                alloc = generate_random_allocation(problem)
                res = evaluate_allocation_with_cpsat(problem, alloc)

                if res["is_valid"]:
                    evaluated.append({
                        "alloc": alloc,
                        "cost": res["total_cost"]
                    })

            match_count = 0
            for alloc_A, alloc_B in itertools.combinations(evaluated, 2):
                if alloc_A["cost"] == alloc_B["cost"]:
                    continue
                label = 1.0 if alloc_A["cost"] < alloc_B["cost"] else 0.0
                dataset.append({
                    "scenario": scenario["name"],
                    "alloc_A": alloc_A["alloc"],
                    "alloc_B": alloc_B["alloc"],
                    "cost_A": alloc_A["cost"],
                    "cost_B": alloc_B["cost"],
                    "label": label
                })
                match_count += 1

            print(f"✅ {len(evaluated)} allocations valides -> {match_count} matchs générés.")
            current_instance += 1

    elapsed = time.time() - start_time
    print(f"\n🎉 Terminé en {elapsed:.1f} secondes !")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)

if __name__ == "__main__":
    chemin_absolu = r"C:\Users\ESOWNN21\Documents\discrete-optimisation-use-cases-ld\examples_do_usecases\part_preparation\dataset_matches.json"
    generate_full_dataset(output_file=chemin_absolu)