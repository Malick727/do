"""
Recalage d'un siamois GNN : passer d'un score en LOGITS à un résidu en EUROS.

POURQUOI IL EN FAUT UN. `gnn_siamese` entraîne l'encodeur en BCE sur des paires, donc seule la
DIFFÉRENCE des scores de deux allocations est contrainte, et en unités de logit. Le réseau rendu
classe, il ne chiffre pas. Or le GA ne classe pas sur le résidu seul : il classe sur
`analytic + résidu`, où la part analytique est exacte, en euros, et pèse ~98% du total (cf.
bilevel.lower.exact_allocation_terms). Additionner des logits à des euros donnerait un classement
arbitraire -- il suffirait que l'échelle des logits soit dix fois trop grande pour que le résidu
prédit écrase la part exacte, ou dix fois trop petite pour qu'il ne serve à rien.

CE QUI DOIT ÊTRE JUSTE, ET CE QUI NE COMPTE PAS. Le recalage est affine, `résidu ≈ a·s + b`.
Dans une exécution du GA, tous les individus appartiennent à LA MÊME instance : `b` les décale
tous d'autant et ne change aucun rang. C'est la PENTE `a` qui porte tout -- elle fixe le taux de
change entre un logit du réseau et un euro de la part analytique, donc l'arbitrage entre
« cette allocation transporte moins » et « cette allocation ordonnance mieux ».

D'OÙ LA PENTE EST ESTIMÉE. Par moindres carrés sur les données CENTRÉES PAR INSTANCE, et non
sur les données brutes. L'entraînement par paires ne compare que des allocations d'une même
instance (cf. dataset.build_instance_pairs) : rien n'a jamais demandé au réseau d'aligner ses
scores D'UNE INSTANCE À L'AUTRE, et il peut parfaitement placer l'instance #3 mille logits plus
haut que l'instance #7 sans qu'une seule paire d'entraînement s'en plaigne. Une régression
globale prendrait ces décalages pour du signal et estimerait une pente qui les explique, au lieu
de la pente qui explique les écarts INTERNES à une instance -- les seuls que le GA voit.

`within_r2` et `global_r2` sont rapportés tous les deux, et leur ÉCART est le diagnostic :
- `within_r2` élevé, `global_r2` bas -> le réseau ordonne bien à l'intérieur d'une instance mais
  ne cale pas ses niveaux entre instances. C'est le régime ATTENDU, et il est sans conséquence
  pour le GA.
- `within_r2` bas -> le recalage affine ne tient pas, et un GnnSiameseRanker sur ce modèle
  classera mal même si l'accuracy par paires était bonne. Une accuracy de 90% sur des paires
  faciles est compatible avec une relation score/euros fortement non linéaire.

Usage:
    python -m part_preparation.solvers.bilevel.ga.surrogate.calibration \\
        --weights data/part_preparation/models/gnn_siamese_xlarge.pth \\
        --dataset data/part_preparation/datasets/dataset_xlarge.jsonl
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import json

import torch

from part_preparation.solvers.bilevel.ga.surrogate.gnn import build_graph, collate_graphs


@dataclass
class Calibration:
    """
    L'affine `résidu_euros = slope * score_brut + intercept`, et de quoi juger sa validité.

    Sérialisée en JSON à côté des poids plutôt qu'à l'intérieur : ce sont trois flottants
    estimés sur un dataset donné, pas des paramètres appris. Les garder lisibles permet de
    constater d'un coup d'œil qu'une pente est négative -- signe que le modèle a été chargé
    avec la mauvaise convention de label, ce qu'aucune métrique d'entraînement ne montrerait.
    """
    slope: float
    intercept: float
    within_r2: float      # qualité du recalage APRÈS centrage par instance : ce qui compte
    global_r2: float      # sans centrage : diagnostic d'alignement inter-instances
    nb_samples: int
    nb_instances: int
    target_key: str

    def to_euros(self, raw_score):
        """Score brut (logits) -> résidu en euros. Accepte un flottant ou un tenseur."""
        return self.slope * raw_score + self.intercept

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @staticmethod
    def load(path) -> "Calibration":
        return Calibration(**json.loads(Path(path).read_text(encoding="utf-8")))

    @staticmethod
    def path_for(weights_path) -> Path:
        """`.../gnn_siamese_xlarge.pth` -> `.../gnn_siamese_xlarge.calib.json`."""
        return Path(weights_path).with_suffix(".calib.json")


@torch.no_grad()
def raw_scores(model, graphs, batch_size: int = 16):
    """
    Scores bruts d'une liste de graphes, par lots.

    `model.eval()` est posé explicitement : l'encodeur contient des LayerNorm, dont le
    comportement ne dépend pas du lot, mais la convention doit valoir pour tout modèle qu'on
    voudrait recaler ici -- y compris un futur encodeur à BatchNorm, où évaluer par lots de 1
    fausserait tout.
    """
    was_training = model.training
    model.eval()
    try:
        scores = []
        for start in range(0, len(graphs), batch_size):
            batch = collate_graphs(graphs[start:start + batch_size])
            output = model(batch)
            # SiameseGNN expose .score(), SchedulingGNN rend un dict : on accepte les deux, pour
            # pouvoir recaler aussi bien un modèle entraîné en paires qu'un encodeur nu.
            scores.append(output["total_cost"] if isinstance(output, dict) else output)
        return torch.cat(scores)
    finally:
        model.train(was_training)


def _encoder_of(model):
    """Rend l'objet dont le forward accepte un batch de graphes et produit `total_cost`."""
    return getattr(model, "encoder", model)


def fit_calibration(model, samples, problems_by_instance, target_key: str = "residual",
                    batch_size: int = 16, verbose: bool = True) -> Calibration:
    """
    Estime l'affine score -> euros sur des échantillons ÉTIQUETÉS.

    Args:
        samples: échantillons portant `alloc`, `instance_id` et `target_key`. Ce sont
            normalement les instances de VALIDATION du run d'entraînement : recaler sur les
            instances d'entraînement donnerait une pente optimiste, du même optimisme que
            l'accuracy d'entraînement. Le passage par
            `gnn_siamese.problems_from_schedule_dataset` normalise `problem_id` en
            `instance_id` et doit avoir eu lieu en amont.

    La pente est estimée sur les écarts à la MOYENNE PAR INSTANCE (cf. docstring du module) :

        a = Σ (s_i - s̄_inst)·(y_i - ȳ_inst) / Σ (s_i - s̄_inst)²

    puis `b` est ce qui recale la moyenne globale, pour que `to_euros` reste interprétable comme
    un résidu et pas seulement comme un rang. Les instances à un seul échantillon ne contribuent
    rien après centrage -- elles sont comptées mais n'influencent pas `a`.
    """
    encoder = _encoder_of(model)

    graphs, targets, instance_ids = [], [], []
    for sample in samples:
        problem = problems_by_instance[sample["instance_id"]]
        graphs.append(build_graph(problem, sample["alloc"]))
        targets.append(float(sample[target_key]))
        instance_ids.append(sample["instance_id"])

    scores = raw_scores(encoder, graphs, batch_size=batch_size).double()
    y = torch.tensor(targets, dtype=torch.float64)

    # --- centrage par instance ---
    unique_ids = sorted(set(instance_ids))
    id_to_row = {instance_id: k for k, instance_id in enumerate(unique_ids)}
    rows = torch.tensor([id_to_row[i] for i in instance_ids], dtype=torch.long)

    counts = torch.zeros(len(unique_ids), dtype=torch.float64).index_add_(
        0, rows, torch.ones_like(y))
    score_means = torch.zeros(len(unique_ids), dtype=torch.float64).index_add_(
        0, rows, scores) / counts
    target_means = torch.zeros(len(unique_ids), dtype=torch.float64).index_add_(
        0, rows, y) / counts

    centered_scores = scores - score_means[rows]
    centered_targets = y - target_means[rows]

    denominator = (centered_scores ** 2).sum()
    if denominator <= 1e-12:
        raise ValueError(
            "Le modèle rend le même score à toutes les allocations d'une même instance : "
            "aucune pente n'est estimable. Poids non entraînés, ou entraînement effondré."
        )
    slope = float((centered_scores * centered_targets).sum() / denominator)
    intercept = float(y.mean() - slope * scores.mean())

    def r_squared(predicted, actual):
        total = ((actual - actual.mean()) ** 2).sum()
        return float(1.0 - ((actual - predicted) ** 2).sum() / total) if total > 0 else float("nan")

    within_r2 = r_squared(slope * centered_scores, centered_targets)
    global_r2 = r_squared(slope * scores + intercept, y)

    calibration = Calibration(
        slope=slope, intercept=intercept,
        within_r2=within_r2, global_r2=global_r2,
        nb_samples=len(samples), nb_instances=len(unique_ids),
        target_key=target_key,
    )

    if verbose:
        print(f"Recalage sur {len(samples)} échantillons / {len(unique_ids)} instances")
        print(f"  pente     : {slope:,.1f} € par logit")
        print(f"  ordonnée  : {intercept:,.1f} €")
        print(f"  R² intra-instance : {within_r2:.3f}   <- ce qui décide du classement dans le GA")
        print(f"  R² global         : {global_r2:.3f}   <- alignement des niveaux entre instances")
        if slope < 0:
            print("  ⚠ pente NÉGATIVE : le modèle ordonne à l'envers. Vérifier la convention "
                  "de label (1.0 = i MEILLEURE que j).")
        if within_r2 < 0.5:
            print("  ⚠ R² intra-instance faible : la relation score/euros n'est pas affine. "
                  "Le ranker classera mal malgré une bonne accuracy par paires.")
    return calibration


def _cli():
    import argparse

    from part_preparation.solvers.bilevel.ga.surrogate.dataset import (
        load_regression_samples, split_by_instance,
    )
    from part_preparation.solvers.bilevel.ga.surrogate.gnn_siamese import (
        SiameseGNN, problems_from_schedule_dataset,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--instance", default=None,
                        help="Uniquement pour les datasets antérieurs au champ 'instance'")
    parser.add_argument("--target", default="residual")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42,
                        help="DOIT être la graine du run d'entraînement : c'est elle qui définit "
                             "le découpage, et donc quelles instances sont réellement tenues à "
                             "l'écart")
    parser.add_argument("--on", choices=["val", "train", "all"], default="val",
                        help="Partie du dataset servant au recalage. 'val' par défaut : recaler "
                             "sur les instances vues à l'entraînement donne une pente optimiste")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=None,
                        help="Défaut : <weights>.calib.json")
    args = parser.parse_args()

    samples = load_regression_samples(args.dataset)
    instance_name, problems = problems_from_schedule_dataset(samples, args.instance)
    print(f"Instance '{instance_name}' | {len(problems)} réplicats | {len(samples):,} échantillons")

    train_samples, val_samples = split_by_instance(
        samples, val_ratio=args.val_ratio, seed=args.seed)
    chosen = {"val": val_samples, "train": train_samples, "all": samples}[args.on]
    print(f"Recalage sur la partie '{args.on}' : {len(chosen):,} échantillons")

    model = SiameseGNN(hidden=args.hidden, nb_layers=args.layers)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))

    calibration = fit_calibration(model, chosen, problems, target_key=args.target,
                                  batch_size=args.batch_size)

    output = args.output or Calibration.path_for(args.weights)
    calibration.save(output)
    print(f"\n-> {output}")


if __name__ == "__main__":
    _cli()
