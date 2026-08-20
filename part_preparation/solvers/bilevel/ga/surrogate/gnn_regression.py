"""
Entraînement de SchedulingGNN en RÉGRESSION : prédire le résidu en euros, pas le classer.

POURQUOI UNE SECONDE FAÇON D'ENTRAÎNER LE MÊME RÉSEAU. `gnn_siamese` optimise une BCE sur des
paires : seule la différence de deux scores est contrainte, en unités de logit. Cela suffit à
classer une population de GA, et `calibration` suffit à ramener ces logits en euros par une
affine (R² intra-instance 0.986 mesuré). Ce que le pairwise ne donne pas, c'est un PAYSAGE de
valeurs exploitable -- une fonction dont on puisse descendre le gradient par rapport à l'entrée
pour chercher une meilleure allocation. Rien dans la perte par paires ne contraint ce qui se
passe entre deux allocations observées ; la régression, elle, ajuste la valeur en chaque point.

CE QUE LA CIBLE EST VRAIMENT. `residual` (ou `arrival_delays_engines`) provient de
`schedule_allocation_with_cpsat` sous PLAFOND DE TEMPS -- un incumbent, pas un optimum prouvé.
Pour un classement, un biais systématique se compense en grande partie. Pour une valeur qu'une
descente de gradient va exploiter activement, il devient une direction dans laquelle descendre :
le réseau apprendra les régularités du solveur autant que celles du problème. C'est une limite
des étiquettes, qu'aucun réglage d'entraînement ne corrige.

L'ÉCHELLE, ET POURQUOI ELLE EST TRAITÉE ICI. `SchedulingGNN.forward` somme un softplus par
moteur : à l'initialisation chaque moteur rend ~0.7 et le total vaut ~700 sur une instance
xlarge, pour une cible de ~230 000 €. Entraîner tel quel demande au réseau de traverser cinq
ordres de grandeur avant de commencer à apprendre quoi que ce soit. On normalise donc la cible
par sa moyenne d'entraînement (`target_scale`), et on initialise le biais de la tête `delay`
pour que la somme initiale vaille ~1. Les deux ensemble mettent le réseau en position de partir,
et `target_scale` est sauvegardée avec les poids -- sans elle les prédictions ne sont plus des
euros et le modèle est inutilisable.

Usage:
    python -m part_preparation.solvers.bilevel.ga.surrogate.gnn_regression \\
        --dataset data/part_preparation/datasets/dataset_xlarge.jsonl \\
        --target residual --epochs 30
"""
from dataclasses import dataclass
import copy
import math
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from part_preparation.solvers.bilevel.ga.rank_quality import spearman
from part_preparation.solvers.bilevel.ga.surrogate.dataset import (
    load_regression_samples, split_by_instance,
)
from part_preparation.solvers.bilevel.ga.surrogate.gnn import (
    SchedulingGNN, build_graph, collate_graphs,
)
from part_preparation.solvers.bilevel.ga.surrogate.paths import MODELS_DIR, ensure_dirs


@dataclass
class RegressionModel:
    """
    Les poids ET l'échelle, qui n'ont aucun sens l'un sans l'autre.

    Le réseau travaille en unités normalisées ; `predict_euros` est le seul point qui rend des
    euros. Sauvegarder les deux dans le même fichier évite le mode de défaillance de
    `gnn_siamese`, où les poids et leur recalage vivent dans deux fichiers séparés qu'il est
    possible de désapparier sans que rien ne le signale.
    """
    model: SchedulingGNN
    target_scale: float
    target_key: str

    @torch.no_grad()
    def predict_euros(self, graphs, batch_size: int = 16):
        was_training = self.model.training
        self.model.eval()
        try:
            out = []
            for start in range(0, len(graphs), batch_size):
                batch = collate_graphs(graphs[start:start + batch_size])
                out.append(self.model(batch)["total_cost"] * self.target_scale)
            return torch.cat(out)
        finally:
            self.model.train(was_training)

    def save(self, path) -> None:
        torch.save({
            "state_dict": self.model.state_dict(),
            "target_scale": self.target_scale,
            "target_key": self.target_key,
            "hidden": self.model.delay_head[0].in_features,
            "nb_layers": len(self.model.layers),
        }, path)

    @staticmethod
    def load(path) -> "RegressionModel":
        blob = torch.load(path, map_location="cpu")
        model = SchedulingGNN(hidden=blob["hidden"], nb_layers=blob["nb_layers"])
        model.load_state_dict(blob["state_dict"])
        return RegressionModel(model, blob["target_scale"], blob["target_key"])


class GraphRegressionDataset(Dataset):
    """Un graphe par allocation, construit une fois puis mis en cache (cf. PairwiseGraphDataset)."""

    def __init__(self, samples, problems_by_instance, target_key, target_scale):
        self.graphs = [
            build_graph(problems_by_instance[s["instance_id"]], s["alloc"]) for s in samples
        ]
        self.targets = [float(s[target_key]) / target_scale for s in samples]
        self.instance_ids = [s["instance_id"] for s in samples]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, k):
        return self.graphs[k], self.targets[k]


def collate_regression(items):
    graphs, targets = zip(*items)
    return collate_graphs(list(graphs)), torch.tensor(targets, dtype=torch.float32)


def _init_output_bias(model: SchedulingGNN, nb_engines: int) -> None:
    """
    Cale le biais de sortie pour que la somme initiale sur les moteurs vaille ~1.

    Sans cela le réseau démarre à ~0.7 x nb_engines (soit ~700 sur xlarge) pour une cible
    normalisée à ~1, et les premières époques ne servent qu'à redescendre cette constante --
    en tirant sur TOUS les poids, puisque rien ne distingue le biais du reste. Le softplus est
    inversé analytiquement : softplus(b) = 1/n  <=>  b = log(exp(1/n) - 1).
    """
    per_engine = 1.0 / max(nb_engines, 1)
    bias = math.log(math.expm1(per_engine))
    with torch.no_grad():
        model.delay_head[-1].bias.fill_(bias)


def _run_epoch(model, dataloader, criterion, optimizer=None):
    """Une passe. Rend (loss moyenne, MAE en unités normalisées, prédictions, cibles)."""
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss, total_abs, seen = 0.0, 0.0, 0
    predictions, targets = [], []

    with torch.set_grad_enabled(is_training):
        for batch, y in dataloader:
            if is_training:
                optimizer.zero_grad()

            predicted = model(batch)["total_cost"]
            loss = criterion(predicted, y)

            if is_training:
                loss.backward()
                # Le total est une SOMME sur ~1000 moteurs : un lot atypique produit un gradient
                # d'un ordre de grandeur au-dessus des autres, et sans plafond un seul pas
                # suffit à renvoyer le réseau dans le régime plat du softplus, dont il ne
                # revient pas.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * y.numel()
            total_abs += (predicted.detach() - y).abs().sum().item()
            seen += y.numel()
            predictions.extend(predicted.detach().tolist())
            targets.extend(y.tolist())

    return total_loss / seen, total_abs / seen, predictions, targets


def _mean_within_instance_spearman(predictions, targets, instance_ids) -> float:
    """
    Spearman moyen CALCULÉ PAR INSTANCE, et non sur l'ensemble des échantillons.

    Un Spearman global serait porté par les écarts ENTRE instances, qui sont grands et que
    personne n'a besoin de prédire : le GA ne compare jamais que des allocations d'une même
    instance. C'est aussi la métrique qui permet de comparer honnêtement ce modèle au siamois,
    dont l'accuracy par paires est par construction intra-instance.
    """
    by_instance = {}
    for prediction, target, instance_id in zip(predictions, targets, instance_ids):
        by_instance.setdefault(instance_id, ([], []))
        by_instance[instance_id][0].append(prediction)
        by_instance[instance_id][1].append(target)

    scores = [
        spearman(p, t) for p, t in by_instance.values()
        if len(p) > 2 and len(set(t)) > 1
    ]
    scores = [s for s in scores if s == s]  # écarte les NaN (variance nulle)
    return sum(scores) / len(scores) if scores else float("nan")


def train_gnn_regression(samples, problems_by_instance, epochs=30, batch_size=16, lr=1e-3,
                         val_ratio=0.2, seed=42, hidden=64, nb_layers=3,
                         target_key="residual", huber_delta=1.0):
    """
    Args:
        huber_delta: Huber plutôt que MSE. Les cibles viennent d'un solveur sous plafond, donc
            une allocation mal ordonnancée par malchance produit une cible aberrante ; la MSE
            lui donnerait un poids quadratique et le réseau passerait sa capacité à mémoriser
            des accidents du solveur.

    La sélection se fait sur la MAE de validation, pas sur le Spearman intra-instance. Ce
    modèle est entraîné pour rendre une VALEUR -- c'est tout son intérêt par rapport au
    siamois -- et sélectionner sur le rang reviendrait à retenir un modèle qui ordonne bien en
    chiffrant mal, c'est-à-dire précisément ce qu'on a déjà.
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

    # L'échelle est estimée sur le TRAIN seul : la calculer sur tout le dataset ferait entrer
    # la distribution de validation dans la normalisation, donc dans le modèle.
    target_scale = sum(float(s[target_key]) for s in train_samples) / max(len(train_samples), 1)
    if target_scale <= 0:
        raise ValueError(f"Échelle non positive ({target_scale}) : cible '{target_key}' douteuse.")
    print(f"Échelle de normalisation ('{target_key}') : {target_scale:,.0f} €")

    print("Construction des graphes (mise en cache)...", flush=True)
    train_dataset = GraphRegressionDataset(train_samples, problems_by_instance,
                                           target_key, target_scale)
    val_dataset = GraphRegressionDataset(val_samples, problems_by_instance,
                                         target_key, target_scale)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_regression)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_regression)

    model = SchedulingGNN(hidden=hidden, nb_layers=nb_layers)
    nb_engines = len(next(iter(problems_by_instance.values())).engines_list)
    _init_output_bias(model, nb_engines)

    criterion = nn.HuberLoss(delta=huber_delta)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"Paramètres : {sum(p.numel() for p in model.parameters()):,} | "
          f"biais de sortie calé sur {nb_engines} moteurs")

    best_val_mae, best_state, best_rho = float("inf"), None, float("nan")

    print(f"Starting training for {epochs} epochs...", flush=True)
    for epoch in range(epochs):
        train_loss, train_mae, _, _ = _run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_mae, val_pred, val_true = _run_epoch(model, val_loader, criterion)
        rho = _mean_within_instance_spearman(val_pred, val_true, val_dataset.instance_ids)

        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae, best_rho = val_mae, rho
            best_state = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch + 1:02d}/{epochs} | "
              f"train MAE: {train_mae * target_scale:>10,.0f} € | "
              f"val MAE: {val_mae * target_scale:>10,.0f} € "
              f"({val_mae:.1%} de l'échelle) | rho intra: {rho:.3f}"
              f"{' (best)' if is_best else ''}", flush=True)

    print(f"\nTraining complete! Meilleure MAE de validation : "
          f"{best_val_mae * target_scale:,.0f} € ({best_val_mae:.1%} de l'échelle), "
          f"rho intra-instance {best_rho:.3f}")
    model.load_state_dict(best_state)
    return RegressionModel(model, target_scale, target_key)


def _cli():
    import argparse
    from pathlib import Path

    from part_preparation import instances
    from part_preparation.solvers.bilevel.ga.surrogate.gnn_siamese import (
        problems_from_schedule_dataset,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--instance", choices=sorted(instances.INSTANCES), default=None)
    parser.add_argument("--target", default="residual",
                        help="'residual' (dataset_builder) ou 'arrival_delays_engines' "
                             "(schedule_builder)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42,
                        help="MÊME graine que le run siamois pour un découpage identique, "
                             "sans quoi les deux modèles ne sont pas comparables")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=MODELS_DIR / "gnn_regression_xlarge.pth")
    args = parser.parse_args()

    if not args.dataset.exists():
        parser.error(f"Dataset introuvable : {args.dataset}")

    samples = load_regression_samples(args.dataset)
    if not samples:
        parser.error(f"Dataset vide : {args.dataset}")

    instance_name, problems = problems_from_schedule_dataset(samples, args.instance)
    print(f"Instance '{instance_name}' | {len(problems)} réplicats | {len(samples):,} échantillons")

    trained = train_gnn_regression(
        samples=samples, problems_by_instance=problems,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        val_ratio=args.val_ratio, seed=args.seed,
        hidden=args.hidden, nb_layers=args.layers,
        target_key=args.target, huber_delta=args.huber_delta,
    )

    ensure_dirs()
    trained.save(args.output)
    print(f"\nPoids + échelle sauvegardés : {args.output}")
    print("Ce modèle rend des EUROS (RegressionModel.predict_euros) : aucun recalage séparé "
          "n'est nécessaire, contrairement au siamois.")


if __name__ == "__main__":
    _cli()
