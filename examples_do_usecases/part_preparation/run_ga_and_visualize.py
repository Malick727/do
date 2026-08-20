"""
Run Genetic Algorithm solver on engine preparation problem and launch visualization dashboards

Usage:
    uv run python run_ga_and_visualize.py --instance medium --ngen 100 --pop-size 50 --ranker siamese
    uv run python run_ga_and_visualize.py --instance medium --ngen 100 --pop-size 50 --ranker cpsat
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import pickle
import subprocess
import torch

from part_preparation import instances
from part_preparation.paths import solution_file, ensure_dirs
from part_preparation.solvers.bilevel.ga.surrogate.paths import SIAMESE_WEIGHTS
from part_preparation.solvers.bilevel.ga.solver import BilevelGASolverD1
from part_preparation.solvers.bilevel.ga.rankers import (
    CpSatExactRanker, CpSatRanker, GreedyRanker, SiameseRanker,
)


def main():
    parser = argparse.ArgumentParser(description="Run the bilevel D1 genetic algorithm and visualize results")
    parser.add_argument('--instance', choices=sorted(instances.INSTANCES),
                        default=instances.DEFAULT_INSTANCE, help='Instance to solve')
    parser.add_argument('--ngen', type=int, default=100, help='Number of generations')
    parser.add_argument('--pop-size', type=int, default=100, help='Population size')
    # Le ranker EST la méthode de résolution du niveau inférieur : c'est le seul paramètre
    # à faire varier pour comparer les approches de la décomposition 1.
    parser.add_argument('--ranker', choices=['greedy', 'cpsat', 'siamese', 'gnn_siamese'],
                        default='greedy', help='Lower-level solving method')
    # `siamese` charge le MLP sur vecteur plat (51% en validation sur instances non vues) ;
    # `gnn_siamese` charge l'encodeur GNN recalé (96.8%). Les deux cohabitent le temps de
    # documenter la comparaison, cf. surrogate/__init__.py.
    parser.add_argument('--gnn-weights', type=Path, default=None,
                        help="Poids du siamois GNN. Défaut : MODELS_DIR/gnn_siamese_xlarge.pth")
    parser.add_argument('--gnn-calibration', type=Path, default=None,
                        help="Défaut : <gnn-weights>.calib.json, écrit par surrogate.calibration")
    parser.add_argument('--replicate', type=int, default=None,
                        help="Réplicat de l'instance. Indispensable avec un surrogate : sans "
                             "lui on résout l'instance de BASE, et rien ne dit si le réseau "
                             "l'avait vue à l'entraînement")
    parser.add_argument('--referee', choices=['greedy', 'cpsat', 'cpsat_exact'], default='cpsat',
                        help='Ranker used to arbitrate the finalists on a common scale')
    parser.add_argument('--cpsat-time-limit', type=float, default=10.0)
    # Budget en TEMPS de la boucle. C'est lui, et non --ngen, qui rend un run comparable à un
    # run CP-SAT : à ngen fixé on ne mesure pas les méthodes mais leurs coûts unitaires (cf.
    # le protocole détaillé dans solvers.compare). --ngen sert alors de plafond de sécurité.
    parser.add_argument('--time-limit', type=float, default=None,
                        help="Secondes allouées à la boucle du GA (l'arbitrage final s'y ajoute)")
    parser.add_argument('--seed', type=int, default=None,
                        help='Graine du GA, pour rejouer un run à l\'identique')
    # Sans ce drapeau le script se termine sur un `streamlit run` bloquant, donc il n'est
    # exécutable ni dans un test ni dans un batch de comparaison.
    parser.add_argument('--no-dashboard', action='store_true',
                        help="Résout et sauvegarde, sans lancer Streamlit")

    args = parser.parse_args()

    print("=" * 80)
    print(f"GENERATING {args.instance.upper()} INSTANCE")
    print("=" * 80)

    problem = instances.build(args.instance, replicate_id=args.replicate)

    # ==========================================
    # RANKER = niveau inférieur
    # ==========================================
    def build_ranker(name):
        if name == 'greedy':
            return GreedyRanker()
        if name == 'cpsat':
            return CpSatRanker(time_limit=args.cpsat_time_limit)
        if name == 'cpsat_exact':
            return CpSatExactRanker(time_limit=600.0)
        raise ValueError(name)

    if args.ranker == 'gnn_siamese':
        print("\n" + "=" * 80)
        print("LOADING GNN SURROGATE")
        print("=" * 80)

        from part_preparation.solvers.bilevel.ga.rankers import GnnSiameseRanker
        from part_preparation.solvers.bilevel.ga.surrogate.calibration import Calibration
        from part_preparation.solvers.bilevel.ga.surrogate.paths import MODELS_DIR

        weights = args.gnn_weights or (MODELS_DIR / "gnn_siamese_xlarge.pth")
        calibration = args.gnn_calibration or Calibration.path_for(weights)

        if not Path(weights).exists():
            print(f"Poids introuvables : {weights}")
            print("Les produire d'abord :")
            print("  uv run python -m part_preparation.solvers.bilevel.ga.surrogate.gnn_siamese \\")
            print("    --dataset data/part_preparation/datasets/dataset_xlarge.jsonl \\")
            print("    --target residual --epochs 20 --pairs-per-instance 100")
            sys.exit(1)

        # Le recalage est vérifié SÉPARÉMENT des poids, et son absence est fatale : sans lui le
        # ranker additionnerait des logits à des euros. Cf. GnnSiameseRanker.__init__.
        if not Path(calibration).exists():
            print(f"Recalage introuvable : {calibration}")
            print("Le produire d'abord :")
            print("  uv run python -m part_preparation.solvers.bilevel.ga.surrogate.calibration \\")
            print(f"    --weights {weights} \\")
            print("    --dataset data/part_preparation/datasets/dataset_xlarge.jsonl \\")
            print("    --target residual --on val --seed 42")
            sys.exit(1)

        ranker = GnnSiameseRanker.from_files(weights, calibration)
        print(f"Poids    : {weights}")
        print(f"Recalage : {calibration} "
              f"(pente {ranker.calibration.slope:,.1f} €/logit, "
              f"R² intra {ranker.calibration.within_r2:.3f})")
    elif args.ranker == 'siamese':
        print("\n" + "=" * 80)
        print("LOADING SURROGATE MODEL")
        print("=" * 80)

        from part_preparation.solvers.bilevel.ga.surrogate.features import (
            NB_FEATURES_PER_ENGINE, NB_GLOBAL_FEATURES,
        )
        from part_preparation.solvers.bilevel.ga.surrogate.siamese import SiameseAllocationNet

        nb_features = NB_FEATURES_PER_ENGINE * len(problem.engines_list) + NB_GLOBAL_FEATURES
        model = SiameseAllocationNet(input_size=nb_features)

        if not SIAMESE_WEIGHTS.exists():
            print(f"Poids introuvables : {SIAMESE_WEIGHTS}")
            print("Lancer d'abord l'entraînement :")
            print("  uv run python -m part_preparation.solvers.bilevel.ga.surrogate.siamese")
            sys.exit(1)

        try:
            model.load_state_dict(torch.load(str(SIAMESE_WEIGHTS)))
        except RuntimeError as error:
            if "size mismatch" not in str(error):
                raise
            print("\nERREUR DE DIMENSION")
            print(f"Le GA attend {nb_features} features en entrée, mais {SIAMESE_WEIGHTS} a été "
                  "entraîné sur une autre taille -- probablement une autre instance.")
            print("Supprimer le fichier de poids et relancer l'entraînement.")
            sys.exit(1)

        model.eval()
        print(f"Modèle chargé depuis : {SIAMESE_WEIGHTS}")
        ranker = SiameseRanker(model)
    else:
        ranker = build_ranker(args.ranker)

    print("\n" + "=" * 80)
    print(f"SOLVING (pop: {args.pop_size}, ngen: {args.ngen}, "
          f"ranker: {ranker.name}, arbitre: {args.referee})")
    print("=" * 80)

    solver = BilevelGASolverD1(
        problem=problem,
        ranker=ranker,
        pop_size=args.pop_size,
        ngen=args.ngen,
        referee=build_ranker(args.referee),
        seed=args.seed,
        time_limit=args.time_limit,
    )

    result = solver.solve()
    solution = result.best_solution
    solution_history = [
        {
            "generation": record.generation,
            "best_true_score": record.best_true_score,
            "mean_true_score": record.mean_true_score,
            "rank_wall_time": record.rank_wall_time,
            "ranker_meta": record.ranker_meta,
        }
        for record in result.history
    ]
    print(f"\nScore arbitré du vainqueur : {result.best_score:,.2f} "
          f"(total {result.total_wall_time:.1f}s)")

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
    ensure_dirs()

    data_file = solution_file(f"{args.instance}_ga")
    with open(data_file, 'wb') as f:
        pickle.dump({
            'problem': problem,
            'solution': solution,
            'kpis': kpis,
            'feasible': feasible,
            'solution_history': solution_history,
        }, f)

    print(f"\n✓ Solution saved to: {data_file}")

    if args.no_dashboard:
        return

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

data_file = Path(r"{data_file}")
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