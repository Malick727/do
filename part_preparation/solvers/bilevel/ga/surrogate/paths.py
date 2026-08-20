"""
Emplacements des DONNÉES DE SURROGATE de la décomposition 1 : datasets d'entraînement et poids.

Ces chemins vivaient dans part_preparation/paths.py, à côté de ceux du problème lui-même. Ils
sont pourtant d'une autre nature : un dataset `allocation -> coût` et des poids de réseau ne
décrivent pas le problème, ils décrivent une façon d'approximer le NIVEAU BAS DE D1. Un
utilisateur qui résout en CP-SAT n'a rien à en connaître, et la décomposition 2 aura les siens,
d'un autre format -- son surrogate n'a pas la même entrée (cf. surrogate.features).

Les répertoires restent sous data/part_preparation/ (hors de l'arbre des sources, ignoré par
git) : c'est la DÉCLARATION qui se rapproche du code qui l'utilise, pas les fichiers.
"""
from pathlib import Path

from part_preparation.paths import DATA_DIR

DATASETS_DIR = DATA_DIR / "datasets"  # datasets d'apprentissage générés
MODELS_DIR = DATA_DIR / "models"      # poids entraînés

# --- datasets -----------------------------------------------------------------------------
# Étiquetés par dataset_builder.py : une allocation, son coût, et la décomposition
# analytique/résiduel de ce coût.
SAMPLES_DATASET = DATASETS_DIR / "dataset_samples.jsonl"

# Étiquetés par schedule_builder.py : une allocation et le PLANNING optimal correspondant.
# Cible d'un modèle génératif, pas d'un régresseur.
SCHEDULES_DATASET = DATASETS_DIR / "dataset_schedules.jsonl"

# Paires pré-calculées (legacy) : recopiait les allocations dans chaque paire, d'où 395 Mo
# pour strictement moins d'information que le fichier plat dont il dérivait. Conservé pour
# relire d'anciens fichiers ; build_instance_pairs le remplace.
MATCHES_DATASET = DATASETS_DIR / "dataset_matches.json"

# --- poids --------------------------------------------------------------------------------
SIAMESE_WEIGHTS = MODELS_DIR / "siamese_weights_2cpu.pth"
GNN_WEIGHTS = MODELS_DIR / "gnn_weights.pth"


def ensure_dirs() -> None:
    """Crée les répertoires de surrogate s'ils manquent. À appeler avant toute écriture."""
    for directory in (DATASETS_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
