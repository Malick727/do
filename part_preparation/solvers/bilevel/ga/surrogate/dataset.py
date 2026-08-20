"""
Datasets d'entraînement des surrogates de D1 : lecture, découpage, construction de paires.

L'étiquetage lui-même (faire tourner un solveur de niveau bas sur des allocations pour en
mesurer le coût) est dans dataset_builder.py -- ici on ne fait que consommer les fichiers
produits.
"""
import itertools
import json
import random

import torch
from torch.utils.data import Dataset

from part_preparation.solvers.bilevel.ga.surrogate.features import (
    NB_FEATURES_PER_ENGINE, NB_GLOBAL_FEATURES, TARGET_KEY, extract_features,
)


# ==========================================
# 2. PYTORCH DATASET
# ==========================================
def load_dataset_matches(json_filepath):
    with open(json_filepath, 'r') as f:
        return json.load(f)


def load_regression_samples(jsonl_filepath):
    """
    Lit le dataset APLATI au format JSON Lines (un échantillon {instance_id, alloc, cost}
    par ligne). Lecture ligne par ligne : à ~450 Mo, un json.load() sur un tableau unique
    faisait exploser la RAM alors que seul le contenu déjà parsé nous intéresse.
    """
    samples = []
    with open(jsonl_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def split_by_instance(records, val_ratio=0.2, seed=42):
    """
    Sépare les enregistrements par instance_id (pas par enregistrement) pour éviter la fuite
    de données : deux enregistrements de la même instance partagent la même structure de coûts,
    donc les laisser se répartir entre train et validation biaiserait le score de validation.
    Fonctionne aussi bien sur des paires (ranking) que sur des échantillons plats (régression).
    """
    instance_ids = sorted({r["instance_id"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(instance_ids)

    n_val = max(1, round(len(instance_ids) * val_ratio))
    val_ids = set(instance_ids[:n_val])
    train_ids = set(instance_ids[n_val:])

    train_records = [r for r in records if r["instance_id"] in train_ids]
    val_records = [r for r in records if r["instance_id"] in val_ids]
    return train_records, val_records


# ==========================================
# 2bis. CONSTRUCTION DES PAIRES
# ==========================================
def build_instance_pairs(samples, pairs_per_instance=None, min_relative_gap=0.0,
                         target_key=TARGET_KEY, seed=42):
    """
    Construit TOUTES les paires (i, j, label) d'INDICES dans `samples`, pour l'entraînement par
    comparaison.

    L'énumération est exhaustive : C(n, 2) paires pour n allocations d'une même instance, soit
    435 pour n=30. La version précédente TIRAIT `pairs_per_instance` paires au hasard AVEC
    REMISE, avec un défaut de 2000 -- soit, à n=30, 2000 tirages dans un vivier de 435. Chaque
    paire y apparaissait ~4,6 fois avec des multiplicités poissonniennes : le dataset n'était
    pas plus grand, il était repondéré au hasard, certaines paires vues dix fois et d'autres
    une seule.

    Trois propriétés, chacune pour une raison précise :

    1. Les deux membres d'une paire viennent de la MÊME instance. Deux allocations d'instances
       différentes n'ont ni les mêmes MSN, ni les mêmes dates d'installation : leurs coûts ne
       sont pas sur une échelle commune. Les comparer apprendrait au réseau à reconnaître
       l'instance -- signal réel mais inutile -- au lieu de départager deux allocations.

    2. On ne stocke que des INDICES. L'ancien dataset_matches.json recopiait les allocations
       dans chaque paire, d'où 395 Mo pour strictement moins d'information que le fichier plat
       dont il dérivait. Ici une allocation peut apparaître dans 100 paires sans coûter un
       octet de plus.

    3. Le label est décidé sur `target_key` -- le RÉSIDU par défaut, pas le coût total. La part
       analytique est exacte et connue à l'inférence (cf. problem.analytic_score) : la faire
       départager par le réseau reviendrait à lui faire réapprendre une soustraction, et comme
       elle pèse >99% du score, elle trancherait à elle seule la moitié des paires.

    label = 1.0 si l'allocation i est MEILLEURE (cible plus faible) que j, 0.0 sinon. Cette
    convention est imposée par SiameseAllocationNet.forward, qui renvoie
    sigmoid(score_j - score_i) = P(i meilleure que j).

    Args:
        pairs_per_instance: `None` = toutes les paires. Un entier plafonne par échantillonnage
            SANS remise, pour les instances où C(n, 2) deviendrait ingérable (n=1000 donne un
            demi-million de paires).
        min_relative_gap: 0.0 par défaut, donc rien n'est écarté hors égalités strictes. Le
            relever écarte les paires trop serrées, dont le label reflète surtout le bruit
            d'ordonnancement -- utile quand les cibles sont des optima prouvés, discutable
            quand ce sont des bornes supérieures, où l'écart mesuré est déjà du bruit.

    Returns:
        (pairs, skipped). `skipped` compte les paires écartées, égalités strictes comprises.
    """
    by_instance = {}
    for idx, sample in enumerate(samples):
        by_instance.setdefault(sample["instance_id"], []).append(idx)

    rng = random.Random(seed)
    pairs = []
    skipped_ties = 0

    for instance_id, indices in sorted(by_instance.items()):
        if len(indices) < 2:
            continue

        instance_pairs = []
        for i, j in itertools.combinations(indices, 2):
            ti, tj = samples[i][target_key], samples[j][target_key]

            # Égalité stricte : le label n'est pas « difficile », il est INDÉFINI. La convention
            # ci-dessous trancherait arbitrairement à 0.0 et demanderait au réseau d'apprendre
            # un pile-ou-face. Écarté quel que soit min_relative_gap.
            if ti == tj:
                skipped_ties += 1
                continue

            if min_relative_gap > 0.0:
                denominator = max(abs(ti), abs(tj), 1e-9)
                if abs(ti - tj) / denominator < min_relative_gap:
                    skipped_ties += 1
                    continue

            # ORIENTATION TIRÉE AU SORT, et ce n'est pas cosmétique. itertools.combinations rend
            # toujours i < j dans l'ordre de stockage ; or en mode "ga" les échantillons sont
            # écrits génération par génération, donc à coût DÉCROISSANT. Sans ce tirage, ti > tj
            # presque toujours et le label vaudrait 0.0 sur ~95% des paires : le réseau
            # apprendrait à répondre « non » sans regarder l'entrée, avec une accuracy de 95% et
            # une loss d'apparence saine. L'ancien tirage aléatoire masquait le problème en
            # tirant aussi l'ordre.
            if rng.random() < 0.5:
                i, j, ti, tj = j, i, tj, ti

            instance_pairs.append((i, j, 1.0 if ti < tj else 0.0))

        if pairs_per_instance is not None and len(instance_pairs) > pairs_per_instance:
            # Sans remise : un plafond ne doit pas réintroduire les doublons qu'on vient
            # d'éliminer.
            instance_pairs = rng.sample(instance_pairs, pairs_per_instance)

        pairs.extend(instance_pairs)

    rng.shuffle(pairs)
    return pairs, skipped_ties


class PairwiseAllocationDataset(Dataset):
    """
    Dataset de paires bâti sur les échantillons PLATS (dataset_samples.jsonl), là où
    AllocationDataset consomme un fichier de paires pré-calculé.

    Les features sont calculées UNE FOIS par allocation puis mises en cache. En mode paires,
    une même allocation apparaît dans des dizaines de paires et à chaque epoch : sans cache,
    extract_features serait rappelé des centaines de fois sur des entrées identiques.
    """

    def __init__(self, samples, pairs, problems_by_instance):
        self.samples = samples
        self.pairs = pairs
        self.problems_by_instance = problems_by_instance
        any_problem = next(iter(problems_by_instance.values()))
        self.input_features_size = NB_FEATURES_PER_ENGINE * len(any_problem.engines_list) + NB_GLOBAL_FEATURES

        # On ne calcule que les allocations réellement référencées par au moins une paire.
        used = sorted({i for pair in pairs for i in pair[:2]})
        self._features = {}
        for idx in used:
            sample = samples[idx]
            problem = problems_by_instance[sample["instance_id"]]
            self._features[idx] = torch.tensor(
                extract_features(sample["alloc"], problem), dtype=torch.float32
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, k):
        i, j, label = self.pairs[k]
        return (
            self._features[i],
            self._features[j],
            torch.tensor([label], dtype=torch.float32),
        )


class AllocationDataset(Dataset):
    def __init__(self, matches, problems_by_instance):
        """
        matches: liste de paires (dicts avec alloc_A/alloc_B/label/instance_id), déjà splittée
        problems_by_instance: {instance_id: Problem} -- CHAQUE instance a ses propres MSN/dates
        (seeds différents), donc il faut utiliser le bon problem pour calculer les features
        de chaque entrée, pas un problem global unique.
        """
        self.problems_by_instance = problems_by_instance
        self.matches = matches
        any_problem = next(iter(problems_by_instance.values()))
        self.input_features_size = NB_FEATURES_PER_ENGINE * len(any_problem.engines_list) + NB_GLOBAL_FEATURES

    def __len__(self):
        return len(self.matches)

    def __getitem__(self, idx):
        match = self.matches[idx]
        problem = self.problems_by_instance[match["instance_id"]]

        features_a = extract_features(match["alloc_A"], problem)
        features_b = extract_features(match["alloc_B"], problem)

        label = float(match["label"])

        return (
            torch.tensor(features_a, dtype=torch.float32),
            torch.tensor(features_b, dtype=torch.float32),
            torch.tensor([label], dtype=torch.float32)
        )


# ==========================================
# 4bis. RÉGRESSION DIRECTE (alternative au ranking)
# ==========================================
class SingleAllocationDataset(
    
    
):
    """Dataset pour la régression directe : une allocation -> sa cible (normalisée)."""
    def __init__(self, samples, problems_by_instance, target_mean, target_std,
                 target_key=TARGET_KEY):
        self.samples = samples
        self.problems_by_instance = problems_by_instance
        self.target_mean = target_mean
        self.target_std = target_std
        self.target_key = target_key
        any_problem = next(iter(problems_by_instance.values()))
        self.input_features_size = NB_FEATURES_PER_ENGINE * len(any_problem.engines_list) + NB_GLOBAL_FEATURES

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        problem = self.problems_by_instance[sample["instance_id"]]
        feats = extract_features(sample["alloc"], problem)
        normalized = (sample[self.target_key] - self.target_mean) / self.target_std
        return (
            torch.tensor(feats, dtype=torch.float32),
            torch.tensor([normalized], dtype=torch.float32),
        )

