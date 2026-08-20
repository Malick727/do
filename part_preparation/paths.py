"""
Emplacements des données de part_preparation, déclarés UNE SEULE FOIS.

Les chemins étaient auparavant écrits en dur dans une quinzaine de scripts, sous la forme
`Path(__file__).parent / "solution_data"` ou `"examples_do_usecases/part_preparation/..."`.
Deux problèmes : le résultat dépendait du répertoire depuis lequel on lançait le script, et
les données (1,3 Go de datasets, de pickles et d'un export CSV) vivaient au milieu du code,
dont une partie versionnée dans git.

Tout est désormais sous `data/part_preparation/`, en dehors de l'arbre des sources et ignoré
par git. Les chemins sont absolus, donc indépendants du répertoire courant.
"""
from pathlib import Path

# part_preparation/paths.py -> racine du dépôt
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "part_preparation"
RAW_DIR = DATA_DIR / "raw"              # exports fournis (Skywise...), en lecture seule
SOLUTIONS_DIR = DATA_DIR / "solutions"  # solutions picklées pour les dashboards
RESULTS_DIR = DATA_DIR / "results"      # tableaux de comparaison d'approches (CSV)

# Les datasets d'entraînement et les poids ne sont PAS déclarés ici : ils ne décrivent pas le
# problème mais une façon d'approximer le niveau bas d'une décomposition donnée. Ils vivent
# donc avec le code qui les produit et les consomme, cf.
# part_preparation.solvers.bilevel.ga.surrogate.paths.

# --- fichiers nommés ---------------------------------------------------------------------
SKYWISE_CSV = RAW_DIR / "contour-export-06-07-2026.csv"


def solution_file(name: str) -> Path:
    """Chemin du pickle de solution `name` (ex: "medium", "medium_ga")."""
    return SOLUTIONS_DIR / f"{name}_solution.pkl"


def ensure_dirs() -> None:
    """Crée les répertoires de données s'ils manquent. À appeler avant toute écriture."""
    for directory in (RAW_DIR, SOLUTIONS_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
