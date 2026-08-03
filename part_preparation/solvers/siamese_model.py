import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
import os
import time

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from part_preparation.problem import (
    EnginePreparationProblem,
    EnginePreparationSolution,
    Engine,
    MSN,
    PreparationFactory,
    AssemblyLine,
)
from part_preparation.instance_generator import generate_instance, InstanceParams, FactoryDefinition

logger = logging.getLogger(__name__)


# ==========================================
# 1. FEATURE ENGINEERING (RAW ALLOCATION)
# ==========================================
def extract_features(allocation_json, problem):
    """
    Transforme l'allocation brute en un vecteur numérique aligné avec le chromosome du GA.
    Pour chaque moteur, on stocke l'index de son MSN et l'index de son usine.
    Vecteur de sortie : [msn_idx_e1, factory_idx_e1, msn_idx_e2, factory_idx_e2, ...]
    """
    features = []

    # Dictionnaires pour retrouver rapidement l'index (0, 1, 2...) à partir de l'ID ("MSN_1", "Fact_A")
    msn_to_idx = {m.id: i for i, m in enumerate(problem.msns_list)}
    factory_to_idx = {f.id: i for i, f in enumerate(problem.factories_list)}

    # On parcourt les moteurs DANS UN ORDRE FIXE (très important pour les NNs)
    for engine in problem.engines_list:
        data = allocation_json.get(engine.id, {})
        msn_id = data.get("msn_id")
        factory_id = data.get("factory_id")

        # Conversion en index. -1 représente une non-allocation.
        m_idx = msn_to_idx.get(msn_id, -1.0)
        f_idx = factory_to_idx.get(factory_id, -1.0)

        # On ajoute ces deux valeurs brutes aux features
        features.extend([float(m_idx), float(f_idx)])

    return features


# ==========================================
# 2. PYTORCH DATASET
# ==========================================
class AllocationDataset(Dataset):
    def __init__(self, json_filepath, problem):
        """
        json_filepath: chemin vers 'dataset_matches.json'
        problem: instance de la classe Problem pour reconstituer les règles métier
        """
        self.problem = problem
        with open(json_filepath, 'r') as f:
            self.matches = json.load(f)

        # La taille du vecteur d'entrée est exactement 2 cases (msn + factory) par moteur.
        self.input_features_size = 2 * len(problem.engines_list)

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, idx):
        match = self.matches[idx]

        features_a = extract_features(match["alloc_A"], self.problem)
        features_b = extract_features(match["alloc_B"], self.problem)

        label = float(match["label"])

        return (
            torch.tensor(features_a, dtype=torch.float32),
            torch.tensor(features_b, dtype=torch.float32),
            torch.tensor([label], dtype=torch.float32)
        )


# ==========================================
# 3. SIAMESE NETWORK ARCHITECTURE
# ==========================================
class SiameseAllocationNet(nn.Module):
    def __init__(self, input_size):
        super(SiameseAllocationNet, self).__init__()

        # Le "Juge" 
        # On utilise bien 128 neurones en entrée de la première couche cachée.
        self.scorer = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Un seul score de sortie
        )

    def forward(self, alloc_a, alloc_b):
        score_a = self.scorer(alloc_a)
        score_b = self.scorer(alloc_b)
        # Probabilité que A soit meilleur que B
        return torch.sigmoid(score_b - score_a)

    def predict_score(self, alloc_features):
        """Méthode pour l'inférence dans l'Algorithme Génétique"""
        return self.scorer(alloc_features)


# ==========================================
# 4. TRAINING LOOP
# ==========================================
def train_siamese_network(dataset_path, problem, epochs=30, batch_size=32, lr=0.001):
    print(f"Loading dataset from {dataset_path}...")
    dataset = AllocationDataset(dataset_path, problem)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SiameseAllocationNet(input_size=dataset.input_features_size)

    criterion = nn.BCELoss()  # Binary Cross Entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Starting training for {epochs} epochs...")
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for alloc_a, alloc_b, label in dataloader:
            optimizer.zero_grad()

            # Forward pass
            prob_a_wins = model(alloc_a, alloc_b)

            # Loss computation
            loss = criterion(prob_a_wins, label)

            # Backward pass & optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy computation
            predictions = (prob_a_wins >= 0.5).float()
            correct_predictions += (predictions == label).sum().item()
            total_samples += label.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = (correct_predictions / total_samples) * 100

        print(f"Epoch {epoch + 1:02d}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.1f}%")

    print("Training complete!")
    return model


# ==========================================
# 5. EXECUTION EXAMPLE
# ==========================================
if __name__ == "__main__":
    # 1. Vérification de l'environnement (par défaut sur CPU)
    device = torch.device("cpu")
    print(f"🚀 Initialisation de l'entraînement sur : {device}")

    # 2. Instanciation du problème métier
    try:
        from part_preparation.instance_generator import MEDIUM_INSTANCE, generate_instance

        print("Génération de l'instance de référence pour le Feature Engineering...")
        my_problem = generate_instance(MEDIUM_INSTANCE)
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
        params = InstanceParams(
        n_fals=4, n_msns_per_fal=40, n_engine_types=2, time_horizon=300,
        fal_takt_paces=[2, 2, 5, 5], factory_definitions=manual_factories,
        engine_to_msn_ratio={"EngineType_1": 1.1, "EngineType_2": 0.95},
        supplier_regularity={"EngineType_1": 1.0, "EngineType_2": 0.8},
        supplier_production_rate={"EngineType_1": 1.0, "EngineType_2": 0.9},
        flexible_delivery_prob=0
    )
        my_problem = generate_instance(params)
    except ImportError as e:
        print(f"⚠️ Erreur d'importation : {e}")
        exit(1)

    # 3. Localisation du dataset
    # On essaye d'abord un chemin relatif robuste
    base_dir = Path(__file__).parent.parent.parent.parent
    dataset_file = base_dir / "examples_do_usecases" / "part_preparation" / "dataset_matches.json"

    # Fallback sur le chemin absolu que tu as fourni si le relatif ne marche pas
    if not dataset_file.exists():
        dataset_file = Path(r"C:\Users\ESOWNN21\Documents\discrete-optimisation-use-cases-ld\examples_do_usecases\part_preparation\dataset_matches.json")

    if not dataset_file.exists():
        print(f"❌ Impossible de trouver le fichier de dataset.")
        print(f"Chemin cherché : {dataset_file}")
    else:
        # 4. Lancement de l'entraînement
        print("\n=== DÉBUT DE L'ENTRAÎNEMENT ===")
        trained_model = train_siamese_network(
            dataset_path=dataset_file,
            problem=my_problem,
            epochs=50,
            batch_size=32,
            lr=0.001
        )

        # 5. Sauvegarde du modèle entraîné
        # On le sauvegarde dans le dossier parent (part_preparation) pour que le GA le trouve facilement
        save_path = Path(r"C:\Users\ESOWNN21\Documents\discrete-optimisation-use-cases-ld\siamese_weights_2cpu.pth")
        torch.save(trained_model.state_dict(), save_path)
        print(f"\n✅ Modèle sauvegardé avec succès sous : {save_path}")

        # 6. Petit test d'inférence
        print("\n=== TEST D'INFÉRENCE RAPIDE ===")
        model_eval = SiameseAllocationNet(input_size=trained_model.scorer[0].in_features)
        model_eval.load_state_dict(torch.load(save_path))
        model_eval.eval()

        dummy_features = torch.zeros(1, model_eval.scorer[0].in_features)

        start_time = time.perf_counter()
        with torch.no_grad():
            score = model_eval.predict_score(dummy_features)
        end_time = time.perf_counter()

        print(f"Score brut (fitness IA) pour un vecteur vide : {score.item():.4f}")
        print(f"Temps d'inférence pur : {(end_time - start_time) * 1000:.4f} millisecondes")