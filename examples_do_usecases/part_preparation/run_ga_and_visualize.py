"""
Run Genetic Algorithm solver on engine preparation problem and launch visualization dashboards

Usage:
    uv run python run_ga_and_visualize.py --instance medium --ngen 100 --pop-size 50 --eval-mode ia
    uv run python run_ga_and_visualize.py --instance medium --ngen 100 --pop-size 50 --eval-mode cpsat
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import pickle
import subprocess
import torch

from part_preparation.instance_generator import generate_instance, InstanceParams, FactoryDefinition

# Import des classes du solver IA
from part_preparation.solvers.ga_solver import GAEngineSolver, SiameseAllocationNet


def main():
    parser = argparse.ArgumentParser(description="Run Genetic Algorithm solver and visualize results")
    parser.add_argument('--instance', choices=['medium', 'large'], default='medium', help='Instance size to solve')
    parser.add_argument('--ngen', type=int, default=100, help='Number of generations for GA')
    parser.add_argument('--pop-size', type=int, default=1000, help='Population size for GA')
    # Ajout du choix du mode d'évaluation
    parser.add_argument('--eval-mode', choices=['cpsat', 'greedy', 'ia'], default='ia', help='Evaluation engine to use')

    args = parser.parse_args()

    print("=" * 80)
    print(f"GENERATING {args.instance.upper()} INSTANCE")
    print("=" * 80)

    # Factory definitions
    manual_factories = [
        FactoryDefinition(engine_type="EngineType_1", home_fal_index=0, capacity=3, preparation_duration=11,
                          location="Derby"),
        FactoryDefinition(engine_type="EngineType_1", home_fal_index=1, capacity=2, preparation_duration=10,
                          location="Toulouse"),
        FactoryDefinition(engine_type="EngineType_1", home_fal_index=2, capacity=2, preparation_duration=13,
                          location="Virginia"),
        FactoryDefinition(engine_type="EngineType_1", home_fal_index=None, capacity=1, preparation_duration=14,
                          location="Singapore"),
        FactoryDefinition(engine_type="EngineType_2", home_fal_index=0, capacity=4, preparation_duration=15,
                          location="Durham"),
        FactoryDefinition(engine_type="EngineType_2", home_fal_index=1, capacity=3, preparation_duration=15,
                          location="Bordeaux"),
        FactoryDefinition(engine_type="EngineType_2", home_fal_index=2, capacity=1, preparation_duration=15,
                          location="Cincinnati"),
        FactoryDefinition(engine_type="EngineType_2", home_fal_index=3, capacity=2, preparation_duration=12,
                          location="Shanghai"),
    ]

    n_msns = 80 if args.instance == 'large' else 40
    time_horizon = 1000 if args.instance == 'large' else 300

    params = InstanceParams(
        n_fals=4,
        n_msns_per_fal=n_msns,
        n_engine_types=2,
        time_horizon=time_horizon,
        fal_takt_paces=[2, 2, 5, 5],
        factory_definitions=manual_factories,
        engine_to_msn_ratio={"EngineType_1": 1.1, "EngineType_2": 0.95},
        supplier_regularity={"EngineType_1": 1.0, "EngineType_2": 0.8},
        supplier_production_rate={"EngineType_1": 1.0, "EngineType_2": 0.9},
        flexible_delivery_prob=0
    )

    problem = generate_instance(params)

    # ==========================================
    # CHARGEMENT DU MODÈLE IA (si sélectionné)
    # ==========================================
    ia_model = None
    if args.eval_mode == 'ia':
        print("\n" + "=" * 80)
        print("LOADING SURROGATE AI MODEL")
        print("=" * 80)

        nb_features = 2 * len(problem.engines_list)
        ia_model = SiameseAllocationNet(input_size=nb_features)

        # ---> ON FORCE LE CHEMIN ABSOLU EXACT DE TON PC <---
        weights_path = Path(r"C:\Users\ESOWNN21\Documents\discrete-optimisation-use-cases-ld\siamese_weights_2cpu.pth")

        if weights_path.exists():
            try:

                ia_model.load_state_dict(torch.load(str(weights_path)))
                ia_model.eval() # Mode évaluation (Inférence)
                print(f"✅ Modèle chargé avec succès depuis : {weights_path}")
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    print("\n❌ ERREUR DE DIMENSION (Size Mismatch) ❌")
                    print(f"Le fichier de poids '{weights_path}' contient une ancienne architecture.")
                    print(f"Ton GA attend {nb_features} variables en entrée, mais le fichier .pth a été entraîné sur une autre taille.")
                    print("\n👉 ACTION REQUISE :")
                    print("1. Va dans ton dossier et SUPPRIME le fichier 'siamese_weights_cpu.pth'.")
                    print("2. Relance le script d'entraînement pour recréer un modèle à la bonne taille :")
                    print("   uv run python examples_do_usecases/part_preparation/solvers/siamese_model.py")
                    print("3. Une fois terminé, relance cette commande GA.")
                    sys.exit(1)
                else:
                    raise e
        else:
            print(f"❌ Erreur : Impossible de trouver le fichier de poids '{weights_path}'")
            print("As-tu bien lancé l'entraînement au préalable ?")
            sys.exit(1)


    print("\n" + "=" * 80)
    print(f"SOLVING WITH GA (pop: {args.pop_size}, ngen: {args.ngen}, mode: {args.eval_mode.upper()})")
    print("=" * 80)

    # Instantiation of our separated Solver, with new arguments
    solver = GAEngineSolver(
        problem=problem,
        pop_size=args.pop_size,
        ngen=args.ngen,
        evaluation_mode=args.eval_mode,
        ia_model=ia_model
    )

    # Run resolution
    solution, solution_history = solver.solve()

    print("\n" + "=" * 80)
    print("SOLVER FINISHED")
    print("=" * 80)

    if solution is None:
        print("\n❌ No solution found!")
        return

    print("\n✅ SOLUTION FOUND")

    # Evaluate solution
    kpis = problem.evaluate(solution)
    feasible = problem.satisfy(solution, debug=True)

    print(f"\nFeasible: {'✅ Yes' if feasible else '❌ No'}")

    print(f"\nObjective values:")
    for obj_name, obj_value in kpis.items():
        print(f"  - {obj_name}: {obj_value:.2f}")

    # Count allocated engines and MSNs
    allocated_engines = sum(
        1 for e in problem.engines_list if e in solution.engine_to_msn and solution.engine_to_msn[e] is not None)
    unallocated_engines = len(problem.engines_list) - allocated_engines

    print(f"\nAllocation:")
    print(f"  - Allocated engines: {allocated_engines}/{len(problem.engines_list)}")
    print(f"  - Unallocated engines: {unallocated_engines}")
    print(f"  - Unallocated MSNs: {kpis['unallocated_msns']:.0f}/{len(problem.msns_list)}")

    # Save problem and solution for dashboard
    data_dir = Path(__file__).parent / "solution_data"
    data_dir.mkdir(exist_ok=True)

    data_file = data_dir / f"{args.instance}_ga_solution.pkl"
    with open(data_file, 'wb') as f:
        pickle.dump({
            'problem': problem,
            'solution': solution,
            'kpis': kpis,
            'feasible': feasible,
            'solution_history': solution_history,
        }, f)

    print(f"\n✓ Solution saved to: {data_file}")

    # Launch dashboard
    print("\n" + "=" * 80)
    print("LAUNCHING DASHBOARD")
    print("=" * 80)

    dashboard_launcher = Path(__file__).parent / "launch_unified_dashboard.py"

    launcher_code = f"""
import streamlit as st
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

data_file = Path(__file__).parent / "solution_data" / "{args.instance}_ga_solution.pkl"
with open(data_file, 'rb') as f:
    data = pickle.load(f)

st.session_state.problem = data['problem']
st.session_state.solution = data['solution']
st.session_state.solution_history = data.get('solution_history', [])

from part_preparation.unified_dashboard import main
main()
    """

    with open(dashboard_launcher, 'w') as f:
        f.write(launcher_code)

    print(f"Dashboard launcher created: {dashboard_launcher}")
    print(f"\nRun: streamlit run {dashboard_launcher}")

    subprocess.run(["streamlit", "run", str(dashboard_launcher)])


if __name__ == "__main__":
    main()