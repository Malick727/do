"""
Siamois à encodeur GNN : apprendre à CLASSER deux allocations, pas à chiffrer leur coût.

Le siamois n'est pas une architecture, c'est un schéma d'entraînement -- un encodeur partagé
appliqué à deux entrées, puis une comparaison. Ce module ne fait que remplacer l'encodeur de
`siamese.SiameseAllocationNet` (un MLP sur un vecteur plat) par `gnn.SchedulingGNN`. Le reste
de la chaîne est réutilisé tel quel : `build_instance_pairs` pour les paires, `split_by_instance`
pour le découpage, la même convention de label.

POURQUOI CHANGER D'ENCODEUR. Le MLP reçoit un vecteur où les positions 0-4 décrivent le moteur
n°0, 5-9 le moteur n°1, etc., et sa première couche a un jeu de poids par emplacement. Trois
conséquences : ce qu'il apprend sur un moteur ne sert à aucun autre ; le déterminant réel du
résidu est RELATIONNEL (quels moteurs se disputent la même usine sur des fenêtres qui se
chevauchent), donc une propriété de paires d'emplacements et non d'un emplacement ; et
l'emplacement i ne désigne pas le même moteur d'une instance à l'autre. Mesuré sur xlarge :
96% d'accuracy d'entraînement, 51% en validation sur des instances non vues -- c'est-à-dire le
hasard. Ajouter la date d'installation aux features n'y a rien changé, ce qui écarte
l'hypothèse d'une information manquante.

Le GNN applique les MÊMES poids à tous les nœuds (équivariance par permutation) et fait
circuler l'information le long des arêtes : un moteur envoie sa demande à son usine, l'usine
agrège ce qu'elle reçoit et renvoie sa charge. La congestion devient une opération du modèle au
lieu d'un motif à mémoriser, et un motif appris sur un moteur vaut pour les 1023 et pour toute
instance.

CE QUE CE MODULE NE RÉSOUT PAS. Si les étiquettes sont du bruit -- CP-SAT plafonné qui ne prouve
jamais l'optimalité, donc des bornes supérieures dont l'écart peut être dominé par la chance du
solveur -- le GNN échouera exactement comme le MLP. Une accuracy de validation qui reste à 50%
ici ne démontre donc pas que l'encodeur est en cause : elle laisse les deux hypothèses ouvertes.

Usage:
    python -m part_preparation.solvers.bilevel.ga.surrogate.gnn_siamese \\
        --dataset data/part_preparation/datasets/schedules_xlarge.jsonl --epochs 30
"""
import contextlib
import copy
import io
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from part_preparation.solvers.bilevel.ga.surrogate.dataset import (
    build_instance_pairs, load_regression_samples, split_by_instance,
)
from part_preparation.solvers.bilevel.ga.surrogate.gnn import (
    SchedulingGNN, build_graph, collate_graphs,
)
from part_preparation.solvers.bilevel.ga.surrogate.paths import MODELS_DIR, ensure_dirs


class SiameseGNN(nn.Module):
    """
    Encodeur GNN partagé + comparaison par différence de scores.

    `SchedulingGNN.forward` rend déjà un `total_cost` scalaire par graphe : une somme, via
    `index_add_`, de contributions par moteur passées par un softplus. C'est directement la
    fonction de score qu'un siamois demande, et elle est invariante par permutation par
    construction -- exactement la propriété qui manquait au MLP.

    `forward` reprend la convention de siamese.SiameseAllocationNet : la sortie est
    P(a meilleure que b) = sigmoid(score_b - score_a), avec label 1.0 quand a est meilleure
    (cible plus faible). Les deux modèles sont donc interchangeables du point de vue de la
    boucle d'entraînement et de la construction des paires.

    ATTENTION À L'ÉCHELLE. Entraîné en BCE sur des paires, seule la DIFFÉRENCE des scores est
    contrainte, et en unités de logit -- pas en euros. `total_cost` a beau être homogène à un
    coût par construction (somme de termes positifs), rien dans cette perte ne l'y ramène.
    Le modèle rendu classe, il ne chiffre pas : l'additionner à la part analytique comme le fait
    SiameseRanker donnerait des classements faux. Pour un modèle utilisable tel quel dans le GA,
    il faut entraîner `SchedulingGNN` en régression sur `arrival_delays_engines`, ce que ce
    module ne fait pas.
    """

    def __init__(self, hidden: int = 64, nb_layers: int = 3, nb_engine_types: int = 2):
        super().__init__()
        self.encoder = SchedulingGNN(
            hidden=hidden, nb_layers=nb_layers, nb_engine_types=nb_engine_types
        )

    def score(self, batch):
        return self.encoder(batch)["total_cost"]

    def forward(self, batch_a, batch_b):
        return torch.sigmoid(self.score(batch_b) - self.score(batch_a))


class PairwiseGraphDataset(Dataset):
    """
    Paires d'allocations livrées sous forme de GRAPHES.

    Les graphes sont construits UNE FOIS par allocation puis mis en cache, comme le fait
    `PairwiseAllocationDataset` pour les vecteurs de features. Ce n'est pas la même économie :
    un vecteur coûte quelques centaines de microsecondes, un graphe hétérogène de 1023 moteurs
    en coûte plusieurs dizaines de millisecondes. Sans cache, avec 435 paires par instance et
    une allocation apparaissant dans 29 d'entre elles, on le reconstruirait 29 fois par epoch.

    Le prix est la mémoire : seules les allocations réellement citées par une paire sont
    converties, jamais tout le dataset.
    """

    def __init__(self, samples, pairs, problems_by_instance):
        self.pairs = pairs
        used = sorted({index for pair in pairs for index in pair[:2]})
        self._graphs = {}
        for index in used:
            sample = samples[index]
            problem = problems_by_instance[sample["instance_id"]]
            self._graphs[index] = build_graph(problem, sample["alloc"])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, k):
        i, j, label = self.pairs[k]
        return self._graphs[i], self._graphs[j], label


def collate_pairs(items):
    """
    Collation d'un lot de paires : deux batches de graphes alignés, plus les labels.

    `collate_graphs` fusionne des graphes en décalant les indices d'arêtes. On l'applique
    séparément aux deux côtés, en gardant l'ordre : la k-ième entrée de `batch_a` et la k-ième
    de `batch_b` forment bien la k-ième paire, ce dont dépend la soustraction des scores.
    """
    graphs_a, graphs_b, labels = zip(*items)
    return (
        collate_graphs(list(graphs_a)),
        collate_graphs(list(graphs_b)),
        torch.tensor(labels, dtype=torch.float32),
    )


def problems_from_schedule_dataset(samples, instance_name=None):
    """
    Reconstruit un problem par `instance_id`, à partir du nom porté par le dataset.

    Les enregistrements de schedule_builder utilisent `problem_id` là où ceux de dataset_builder
    utilisent `instance_id`. On normalise ici plutôt que dans chaque fonction en aval, pour que
    `split_by_instance` et `build_instance_pairs` fonctionnent sur les deux formats sans
    connaître leur provenance.
    """
    from part_preparation import instances

    for sample in samples:
        if "instance_id" not in sample:
            sample["instance_id"] = sample["problem_id"]

    declared = {s["instance"] for s in samples if "instance" in s}
    if len(declared) > 1:
        raise ValueError(f"Le dataset mélange plusieurs instances : {sorted(declared)}")
    if declared:
        name_from_dataset = declared.pop()
        if instance_name is not None and instance_name != name_from_dataset:
            raise ValueError(
                f"--instance {instance_name!r} contredit le dataset ({name_from_dataset!r})."
            )
        instance_name = name_from_dataset
    elif instance_name is None:
        raise ValueError(
            "Dataset sans champ 'instance' (généré avant son ajout) et --instance non fourni."
        )

    # generate_instance imprime une quarantaine de lignes par appel, sans interrupteur : on
    # coupe, sinon 100 réplicats noient la sortie d'entraînement sous 4000 lignes.
    with contextlib.redirect_stdout(io.StringIO()):
        problems = {
            instance_id: instances.build(instance_name, replicate_id=instance_id)
            for instance_id in sorted({s["instance_id"] for s in samples})
        }
    return instance_name, problems


def _run_epoch(model, dataloader, criterion, optimizer=None):
    """Une passe. Avec optimizer on entraîne, sans lui on évalue."""
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(is_training):
        for batch_a, batch_b, labels in dataloader:
            if is_training:
                optimizer.zero_grad()

            prob_a_wins = model(batch_a, batch_b)
            loss = criterion(prob_a_wins, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            correct += ((prob_a_wins >= 0.5).float() == labels).sum().item()
            seen += labels.size(0)

    return total_loss / len(dataloader), (correct / seen) * 100


def train_gnn_siamese(samples, problems_by_instance, epochs=30, batch_size=8, lr=1e-3,
                      val_ratio=0.2, seed=42, hidden=64, nb_layers=3,
                      pairs_per_instance=None, min_relative_gap=0.0,
                      target_key="arrival_delays_engines"):
    """
    Args:
        samples: échantillons DÉJÀ chargés, et non un chemin. Deux raisons : le format de
            schedule_builder nomme sa clé `problem_id` là où celui de dataset_builder la nomme
            `instance_id`, et la normalisation est faite en amont par
            problems_from_schedule_dataset -- relire le fichier ici en perdrait le bénéfice.
            Accessoirement, un dataset xlarge pèse ~270 Mo : le parser deux fois coûte autant
            de RAM que de patience.
        batch_size: 8 par défaut, contre 32 pour le MLP. Un élément de lot est ici une PAIRE de
            graphes de ~1023 moteurs, pas deux vecteurs -- le coût mémoire d'un pas est deux
            ordres de grandeur au-dessus.
        target_key: `arrival_delays_engines` et non `residual`. Les enregistrements de
            schedule_builder portent la décomposition terme à terme et non le résidu agrégé ;
            sous la pondération canonique, `arrival_delays_engines` EST le résidu (storage_cost
            y pèse 0). Passer `residual` sur un dataset de dataset_builder reste possible.

    Le découpage est fait par INSTANCE, comme pour le MLP : les deux allocations d'une paire
    viennent de la même instance, et deux paires partageant une allocation ne doivent pas tomber
    de part et d'autre du split.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    if samples and target_key not in samples[0]:
        raise ValueError(
            f"Le dataset ne contient pas '{target_key}'. Champs disponibles : "
            f"{sorted(samples[0].keys())}"
        )

    train_samples, val_samples = split_by_instance(samples, val_ratio=val_ratio, seed=seed)
    print(f"Instances -> train: {len({s['instance_id'] for s in train_samples})}, "
          f"val: {len({s['instance_id'] for s in val_samples})}")

    train_pairs, skipped_train = build_instance_pairs(
        train_samples, pairs_per_instance, min_relative_gap, target_key, seed)
    val_pairs, skipped_val = build_instance_pairs(
        val_samples, pairs_per_instance, min_relative_gap, target_key, seed + 1)
    print(f"Paires (cible '{target_key}') -> train: {len(train_pairs)}, val: {len(val_pairs)}")
    print(f"   écartées -> train: {skipped_train}, val: {skipped_val}")
    if train_pairs:
        print(f"   équilibre des labels (train) : "
              f"{sum(p[2] for p in train_pairs) / len(train_pairs):.1%} de label=1")

    print("Construction des graphes (mis en cache)...", flush=True)
    train_dataset = PairwiseGraphDataset(train_samples, train_pairs, problems_by_instance)
    val_dataset = PairwiseGraphDataset(val_samples, val_pairs, problems_by_instance)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_pairs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_pairs)

    model = SiameseGNN(hidden=hidden, nb_layers=nb_layers)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"Paramètres : {sum(p.numel() for p in model.parameters()):,} "
          f"(le MLP en avait ~534 000 pour la seule première couche)")

    # Sélection sur l'ACCURACY et non la loss : ce modèle ne sert qu'à ordonner, et la BCE
    # pénalise la confiance mal placée -- un réseau qui classe mieux mais plus franchement voit
    # sa loss monter pendant que son accuracy progresse.
    best_val_acc, best_val_loss, best_state = -1.0, float("inf"), None

    print(f"Starting training for {epochs} epochs...", flush=True)
    for epoch in range(epochs):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, optimizer=None)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc, best_val_loss = val_acc, val_loss
            best_state = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch + 1:02d}/{epochs} | "
              f"train loss: {train_loss:.4f} acc: {train_acc:.1f}% | "
              f"val loss: {val_loss:.4f} acc: {val_acc:.1f}%"
              f"{' (best)' if is_best else ''}", flush=True)

    print(f"Training complete! Best val accuracy: {best_val_acc:.1f}% "
          f"(loss {best_val_loss:.4f}) — hasard = 50%")
    model.load_state_dict(best_state)
    return model


def _cli():
    import argparse
    from pathlib import Path

    from part_preparation import instances

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Dataset de schedule_builder ou de dataset_builder")
    parser.add_argument("--instance", choices=sorted(instances.INSTANCES), default=None,
                        help="Uniquement pour les datasets antérieurs au champ 'instance'")
    parser.add_argument("--target", default="arrival_delays_engines",
                        help="Clé de la cible servant à étiqueter les paires "
                             "('residual' pour un dataset de dataset_builder)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pairs-per-instance", type=int, default=None)
    parser.add_argument("--min-relative-gap", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=MODELS_DIR / "gnn_siamese.pth")
    args = parser.parse_args()

    if not args.dataset.exists():
        parser.error(
            f"Dataset introuvable : {args.dataset}\n"
            "Le générer d'abord :\n"
            "  uv run python -m part_preparation.solvers.bilevel.ga.surrogate.schedule_builder "
            "--instance xlarge --problems 100 --allocations 30"
        )

    samples = load_regression_samples(args.dataset)
    if not samples:
        parser.error(f"Dataset vide : {args.dataset}")

    instance_name, problems = problems_from_schedule_dataset(samples, args.instance)
    print(f"Instance '{instance_name}' | {len(problems)} réplicats | {len(samples):,} échantillons")

    model = train_gnn_siamese(
        samples=samples,
        problems_by_instance=problems,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        val_ratio=args.val_ratio, seed=args.seed,
        hidden=args.hidden, nb_layers=args.layers,
        pairs_per_instance=args.pairs_per_instance,
        min_relative_gap=args.min_relative_gap,
        target_key=args.target,
    )

    ensure_dirs()
    torch.save(model.state_dict(), args.output)
    print(f"\nPoids sauvegardés : {args.output}")
    print("ATTENTION : entraîné en pairwise, donc sans échelle en euros. Ce modèle CLASSE, il "
          "ne chiffre pas — SiameseRanker additionne predict_residual() à la part analytique et "
          "donnerait des classements faux. Pour le brancher dans le GA il faut soit un recalage "
          "a posteriori, soit un entraînement en régression de SchedulingGNN.")


if __name__ == "__main__":
    _cli()
