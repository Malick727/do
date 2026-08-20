"""
Surrogate siamois de la décomposition 1 : architecture et entraînement.

Le réseau approxime le NIVEAU BAS de D1 -- le coût d'ordonnancement à allocation figée. Il
est utilisé par bilevel.ga.rankers.SiameseRanker, qui l'interroge sur une POPULATION entière
en une passe plutôt qu'individu par individu.
"""
import contextlib
import copy
import io
import logging
import os
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from part_preparation.solvers.bilevel.ga.surrogate.paths import (
    SAMPLES_DATASET, SIAMESE_WEIGHTS, ensure_dirs,
)
# instance_generator n'est plus importé : les problems sont reconstruits par
# part_preparation.instances (cf. problems_from_dataset), qui est la source unique. L'ancien
# __main__ redéclarait ici ses propres InstanceParams et FactoryDefinition, ce qui reconstruisait
# `medium` quel que soit le dataset chargé.
from part_preparation.solvers.bilevel.ga.surrogate.features import (
    NB_FEATURES_PER_ENGINE, NB_GLOBAL_FEATURES, TARGET_KEY, extract_features,
)
from part_preparation.solvers.bilevel.ga.surrogate.dataset import (
    AllocationDataset, PairwiseAllocationDataset, SingleAllocationDataset,
    build_instance_pairs, load_dataset_matches, load_regression_samples, split_by_instance,
)

logger = logging.getLogger(__name__)


# ==========================================
# 3. SIAMESE NETWORK ARCHITECTURE
# ==========================================
class SiameseAllocationNet(nn.Module):
    """
    Le réseau prédit le RÉSIDU de l'objectif (la part qui dépend du planning), pas le score
    total : cf. problem.residual_score. L'appelant y rajoute la part analytique exacte.

    Les paramètres de standardisation de la cible sont enregistrés comme buffers, donc
    sauvegardés et rechargés avec state_dict(). C'est indispensable ici : la prédiction est
    ADDITIONNÉE à un terme en unités réelles, donc on doit pouvoir la dénormaliser au moment
    de l'inférence. Tant que le réseau ne servait qu'à classer, l'échelle était sans effet et
    ces paramètres pouvaient rester dans le script d'entraînement.
    """

    def __init__(self, input_size, target_mean: float = 0.0, target_std: float = 1.0):
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
        self.register_buffer("target_mean", torch.tensor(float(target_mean)))
        self.register_buffer("target_std", torch.tensor(float(target_std)))

    def set_target_normalization(self, mean: float, std: float) -> None:
        """Fixe la standardisation de la cible après calcul sur le split d'entraînement."""
        self.target_mean.fill_(float(mean))
        self.target_std.fill_(float(std))

    def forward(self, alloc_a, alloc_b):
        score_a = self.scorer(alloc_a)
        score_b = self.scorer(alloc_b)
        # Probabilité que A soit meilleur que B
        return torch.sigmoid(score_b - score_a)

    def predict_score(self, alloc_features):
        """Sortie BRUTE (standardisée). Suffit pour classer, pas pour additionner."""
        return self.scorer(alloc_features)

    def predict_residual(self, alloc_features):
        """
        Résidu prédit, en unités RÉELLES de l'objectif. C'est ce qu'il faut utiliser dès qu'on
        combine la prédiction avec la part analytique (cf. ga.rankers.SiameseRanker).
        """
        return self.scorer(alloc_features) * self.target_std + self.target_mean


# ==========================================
# 4. TRAINING LOOP
# ==========================================
def _run_epoch(model, dataloader, criterion, optimizer=None):
    """Une passe sur le dataloader. Si optimizer est fourni, on entraîne ; sinon, on évalue seulement."""
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss, correct_predictions, total_samples = 0.0, 0, 0

    with torch.set_grad_enabled(is_training):
        for alloc_a, alloc_b, label in dataloader:
            if is_training:
                optimizer.zero_grad()

            prob_a_wins = model(alloc_a, alloc_b)
            loss = criterion(prob_a_wins, label)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            predictions = (prob_a_wins >= 0.5).float()
            correct_predictions += (predictions == label).sum().item()
            total_samples += label.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = (correct_predictions / total_samples) * 100
    return avg_loss, accuracy


def train_siamese_network(dataset_path, problems_by_instance, epochs=30, batch_size=32, lr=0.001,
                           val_ratio=0.2, seed=42, pairs_per_instance=None,
                           min_relative_gap=0.0, target_key=TARGET_KEY):
    """
    Entraîne SiameseAllocationNet à COMPARER deux allocations, à partir du dataset plat
    (dataset_samples.jsonl) : les paires sont construites ici, pas lues depuis un fichier.

    L'ordre des opérations n'est pas anodin : on sépare d'abord les INSTANCES en train/val,
    puis on construit les paires de chaque côté. Construire les paires d'abord puis découper
    laisserait deux paires partageant une même allocation tomber de part et d'autre du split --
    le réseau aurait déjà vu en entraînement une des deux allocations qu'on lui demande de
    départager en validation.

    Le modèle retourné est celui de la meilleure loss de validation, pas du dernier epoch.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    print(f"Loading dataset from {dataset_path}...")
    all_samples = load_regression_samples(dataset_path)
    if all_samples and target_key not in all_samples[0]:
        raise ValueError(
            f"Le dataset {dataset_path} ne contient pas le champ '{target_key}' : il a été "
            f"généré avant le passage à la cible résiduelle, relancer generate_dataset.py. "
            f"Champs disponibles : {sorted(all_samples[0].keys())}"
        )

    train_samples, val_samples = split_by_instance(all_samples, val_ratio=val_ratio, seed=seed)
    print(f"Instances -> train: {len({s['instance_id'] for s in train_samples})}, "
          f"val: {len({s['instance_id'] for s in val_samples})}")
    print(f"Échantillons -> train: {len(train_samples)}, val: {len(val_samples)}")

    train_pairs, skipped_train = build_instance_pairs(
        train_samples, pairs_per_instance, min_relative_gap, target_key, seed)
    val_pairs, skipped_val = build_instance_pairs(
        val_samples, pairs_per_instance, min_relative_gap, target_key, seed + 1)
    print(f"Paires (cible '{target_key}', écart mini {min_relative_gap:.0%}) -> "
          f"train: {len(train_pairs)}, val: {len(val_pairs)}")
    print(f"   paires trop serrées écartées -> train: {skipped_train}, val: {skipped_val}")
    if train_pairs:
        share = sum(p[2] for p in train_pairs) / len(train_pairs)
        print(f"   équilibre des labels (train) : {share:.1%} de label=1")

    train_dataset = PairwiseAllocationDataset(train_samples, train_pairs, problems_by_instance)
    val_dataset = PairwiseAllocationDataset(val_samples, val_pairs, problems_by_instance)
    # drop_last sur le TRAIN seulement : `scorer` contient un BatchNorm1d, qui lève
    # "Expected more than 1 value per channel" si le dernier batch d'un epoch se retrouve à un
    # seul élément -- ce qui arrive dès que le nombre de paires est congru à 1 modulo
    # batch_size. Le plantage surviendrait en fin de premier epoch, après le chargement et
    # l'extraction des features. La validation n'en a pas besoin : elle tourne sous
    # model.eval(), où BatchNorm utilise ses statistiques courantes et accepte un batch de 1.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SiameseAllocationNet(input_size=train_dataset.input_features_size)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Le modèle retenu est celui de la meilleure ACCURACY de validation, pas de la meilleure
    # loss. Les deux divergent, et pas marginalement : la BCE pénalise la confiance mal placée,
    # donc un réseau qui classe mieux mais plus franchement voit sa loss monter pendant que son
    # accuracy progresse. Or ce modèle ne sert qu'à ORDONNER une population -- seul compte le
    # nombre de paires correctement départagées, jamais la calibration des probabilités.
    #
    # Sélectionner sur la loss avait un effet concret et invisible : sur un run où la val loss
    # était minimale au premier epoch et montait ensuite, le fichier de poids sauvegardé était
    # celui de l'epoch 1, c'est-à-dire un réseau à peine entraîné, alors que 49 epochs avaient
    # tourné.
    best_val_acc = -1.0
    best_val_loss = float('inf')
    best_state = None

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, optimizer=None)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        marker = " (best)" if is_best else ""
        print(f"Epoch {epoch + 1:02d}/{epochs} | "
              f"train loss: {train_loss:.4f} acc: {train_acc:.1f}% | "
              f"val loss: {val_loss:.4f} acc: {val_acc:.1f}%{marker}")

    print(f"Training complete! Best val accuracy: {best_val_acc:.1f}% "
          f"(loss {best_val_loss:.4f} à cet epoch) — hasard = 50%")
    model.load_state_dict(best_state)
    return model



def train_regression_network(dataset_path, problems_by_instance, epochs=50, batch_size=32, lr=0.001,
                              val_ratio=0.2, seed=42, target_key=TARGET_KEY):
    """
    Entraîne SiameseAllocationNet (via predict_score) à prédire le RÉSIDU de l'objectif -- la
    part qui dépend du planning -- plutôt que le score total. Le transport et les MSN non
    couverts, qui pèsent >90% du score, sont calculés exactement à l'inférence par
    bilevel.lower.analytic_score_from_allocation : les faire apprendre au réseau reviendrait à
    remplacer un calcul exact par une approximation.

    La cible est standardisée (moyenne/écart-type calculés sur le TRAIN uniquement, pour ne pas
    fuiter la validation). Ces deux paramètres sont stockés dans le modèle et non plus
    seulement ici : la prédiction devant être additionnée à un terme en unités réelles, il faut
    pouvoir la dénormaliser à l'inférence (cf. SiameseAllocationNet.predict_residual).
    """
    random.seed(seed)
    torch.manual_seed(seed)

    print(f"Loading dataset from {dataset_path}...")
    all_samples = load_regression_samples(dataset_path)

    missing = [s for s in all_samples[:1] if target_key not in s]
    if missing:
        raise ValueError(
            f"Le dataset {dataset_path} ne contient pas le champ '{target_key}'. Il a été "
            f"généré avant le passage à la cible résiduelle : relancer generate_dataset.py. "
            f"Champs disponibles : {sorted(all_samples[0].keys())}"
        )

    train_samples, val_samples = split_by_instance(all_samples, val_ratio=val_ratio, seed=seed)

    train_targets = [s[target_key] for s in train_samples]
    target_mean = sum(train_targets) / len(train_targets)
    target_std = (sum((c - target_mean) ** 2 for c in train_targets) / len(train_targets)) ** 0.5
    if target_std == 0:
        raise ValueError(
            f"La cible '{target_key}' est constante sur le train : rien à apprendre."
        )

    print(f"Instances -> train: {len({s['instance_id'] for s in train_samples})}, "
          f"val: {len({s['instance_id'] for s in val_samples})}")
    print(f"Échantillons -> train: {len(train_samples)}, val: {len(val_samples)}")
    print(f"Cible '{target_key}' (train) -> moyenne: {target_mean:.0f}, écart-type: {target_std:.0f}")

    train_dataset = SingleAllocationDataset(train_samples, problems_by_instance,
                                            target_mean, target_std, target_key)
    val_dataset = SingleAllocationDataset(val_samples, problems_by_instance,
                                          target_mean, target_std, target_key)
    # drop_last : même raison qu'en mode ranking, cf. train_siamese_network.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SiameseAllocationNet(input_size=train_dataset.input_features_size,
                                 target_mean=target_mean, target_std=target_std)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_state = None

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss_total = 0.0
        for feats, target in train_loader:
            optimizer.zero_grad()
            pred = model.predict_score(feats)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()
        train_loss = train_loss_total / len(train_loader)

        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for feats, target in val_loader:
                pred = model.predict_score(feats)
                loss = criterion(pred, target)
                val_loss_total += loss.item()
        val_loss = val_loss_total / len(val_loader)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        marker = " (best)" if is_best else ""
        print(f"Epoch {epoch + 1:02d}/{epochs} | train MSE: {train_loss:.4f} | val MSE: {val_loss:.4f}{marker}")

    print(f"Training complete! Best val MSE: {best_val_loss:.4f} "
          f"(RMSE sur le résidu réel ≈ {(best_val_loss ** 0.5) * target_std:.0f})")
    model.load_state_dict(best_state)
    # best_state provient d'un deepcopy du state_dict, buffers de normalisation compris :
    # le modèle rendu sait donc dénormaliser sans intervention de l'appelant.
    model.set_target_normalization(target_mean, target_std)
    return model



# ==========================================
# 5. CLI
# ==========================================
def problems_from_dataset(samples, instance_name=None):
    """
    Reconstruit un `problem` par `instance_id` présent dans le dataset.

    Les problems ne sont pas sérialisés avec le dataset et n'ont pas à l'être : `instances.build`
    est déterministe, donc `(nom, replicate_id)` suffit à régénérer à l'identique celui qui a
    produit chaque échantillon. C'est ce qui permet de garder un dataset de quelques centaines
    de Mo au lieu d'y recopier une instance par ligne.

    Deux garde-fous, parce que se tromper d'instance ici ne lève RIEN de visible -- les features
    seraient calculées sur des dates de MSN qui n'ont pas produit ces coûts, et le réseau
    apprendrait un bruit avec une loss d'apparence normale :

      1. le nom est lu dans le dataset (champ écrit par dataset_builder.deduplicate) ; l'argument
         `instance_name` ne sert qu'aux fichiers produits avant ce champ, et il est refusé s'il
         contredit ce que le dataset déclare ;
      2. le nombre de moteurs du problem reconstruit est confronté à la taille des allocations
         stockées. C'est la vérification qui attrape effectivement une erreur de nom, puisque le
         vecteur de features est en NB_FEATURES_PER_ENGINE x nb_moteurs.
    """
    from part_preparation import instances

    declared = {sample["instance"] for sample in samples if "instance" in sample}
    if len(declared) > 1:
        raise ValueError(
            f"Le dataset mélange plusieurs instances nommées ({sorted(declared)}). Le vecteur "
            "de features est de taille fixe : un dataset = un nom d'instance."
        )

    if declared:
        name_from_dataset = declared.pop()
        if instance_name is not None and instance_name != name_from_dataset:
            raise ValueError(
                f"--instance {instance_name!r} contredit le dataset, qui déclare "
                f"{name_from_dataset!r}. Ne pas passer --instance : le dataset fait foi."
            )
        instance_name = name_from_dataset
    elif instance_name is None:
        raise ValueError(
            "Dataset sans champ 'instance' (généré avant son ajout) et --instance non fourni. "
            "Relancer dataset_builder, ou préciser --instance en sachant que rien ne le vérifie."
        )

    # generate_instance imprime un rapport d'une quarantaine de lignes à chaque appel, sans
    # interrupteur : les print y sont en dur en 25 endroits. Utile quand on construit UNE
    # instance à la main, insupportable ici où l'on en reconstruit cent -- 4000 lignes de bruit
    # avant le premier epoch, au milieu desquelles l'équilibre des labels et la loss de
    # validation passent inaperçus. On coupe donc la sortie au point d'appel, ce qui est local
    # et réversible, plutôt que de guarder les 25 print du générateur.
    instance_ids = sorted({sample["instance_id"] for sample in samples})
    with contextlib.redirect_stdout(io.StringIO()):
        problems = {
            instance_id: instances.build(instance_name, replicate_id=instance_id)
            for instance_id in instance_ids
        }

    nb_engines = {len(sample["alloc"]) for sample in samples}
    expected = {len(problem.engines_list) for problem in problems.values()}
    if not nb_engines <= expected:
        raise ValueError(
            f"Les allocations du dataset portent sur {sorted(nb_engines)} moteurs, mais "
            f"l'instance {instance_name!r} en compte {sorted(expected)}. Le dataset et "
            "l'instance ne correspondent pas."
        )

    return instance_name, problems


def _cli():
    import argparse
    from pathlib import Path

    from part_preparation import instances

    parser = argparse.ArgumentParser(
        description="Entraîne le surrogate siamois du niveau bas de D1."
    )
    parser.add_argument("--dataset", type=Path, default=SAMPLES_DATASET,
                        help="Dataset produit par dataset_builder")
    # Deux objectifs d'entraînement pour le MÊME réseau, et le choix n'est pas neutre :
    #   regression -- apprend la valeur du résidu (MSE). Nécessaire si la prédiction doit être
    #     additionnée à la part analytique, ce que fait SiameseRanker.
    #   ranking    -- apprend seulement à comparer deux allocations (BCE sur des paires). Plus
    #     facile, et suffisant pour un GA dont la fitness est un RANG -- mais le modèle rendu
    #     n'a pas d'échelle, donc predict_residual n'y a aucun sens.
    # SiameseRanker ajoutant la part analytique aux prédictions, c'est `regression` qui est
    # utilisable tel quel ; d'où le défaut.
    parser.add_argument("--mode", choices=["regression", "ranking"], default="regression",
                        help="regression = prédit la valeur du résidu ; "
                             "ranking = apprend seulement à comparer (sans échelle)")
    parser.add_argument("--instance", choices=sorted(instances.INSTANCES), default=None,
                        help="Uniquement pour les datasets antérieurs au champ 'instance'")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pairs-per-instance", type=int, default=None,
                        help="Mode ranking : plafond par instance. Défaut = toutes les paires, "
                             "soit C(n,2) = 435 pour 30 allocations")
    parser.add_argument("--min-relative-gap", type=float, default=0.0,
                        help="Mode ranking : écarte les paires dont les cibles diffèrent de "
                             "moins de ce ratio. 0.0 = ne garder que les égalités strictes hors")
    parser.add_argument("--output", type=Path, default=SIAMESE_WEIGHTS)
    args = parser.parse_args()

    if not args.dataset.exists():
        parser.error(
            f"Dataset introuvable : {args.dataset}\n"
            "Le générer d'abord :\n"
            "  uv run python -m part_preparation.solvers.bilevel.ga.surrogate.dataset_builder "
            "--instance medium --replicates 100"
        )

    samples = load_regression_samples(args.dataset)
    if not samples:
        parser.error(f"Dataset vide : {args.dataset}")

    instance_name, problems_by_instance = problems_from_dataset(samples, args.instance)
    print(f"Instance '{instance_name}' | {len(problems_by_instance)} réplicats | "
          f"{len(samples):,} échantillons")

    if args.mode == "regression":
        model = train_regression_network(
            dataset_path=args.dataset,
            problems_by_instance=problems_by_instance,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            val_ratio=args.val_ratio, seed=args.seed,
        )
    else:
        model = train_siamese_network(
            dataset_path=args.dataset,
            problems_by_instance=problems_by_instance,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            val_ratio=args.val_ratio, seed=args.seed,
            pairs_per_instance=args.pairs_per_instance,
            min_relative_gap=args.min_relative_gap,
        )

    ensure_dirs()
    torch.save(model.state_dict(), args.output)
    print(f"\nPoids sauvegardés : {args.output}")
    if args.mode == "ranking":
        print("ATTENTION : modèle entraîné en mode ranking, donc sans échelle. "
              "SiameseRanker additionne predict_residual() à la part analytique et donnera "
              "des classements faux. Utiliser --mode regression pour un modèle utilisable "
              "dans le GA.")


if __name__ == "__main__":
    _cli()
