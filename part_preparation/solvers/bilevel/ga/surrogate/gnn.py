"""
Surrogate GNN : allocation -> valeur optimale d'ordonnancement.

Motivation. La valeur de référence d'une allocation est l'optimum du modèle CP-SAT complet
(cf. bilevel.lower.solve_allocation_with_full_cpsat), qui coûte 20s à 600s+ par évaluation et est
donc inutilisable dans une boucle de GA. Ce module apprend à l'approcher.

Trois choix structurants, tous justifiés par les mesures faites sur le MLP précédent :

1. On n'apprend QUE le résidu. `travel_cost` et `unallocated_msns` représentent ~98% du score
   GA mais sont exactement calculables depuis l'allocation seule (bilevel.lower.exact_allocation_terms).
   Les faire prédire par un réseau revenait à lui faire apprendre un comptage, ce qui explique
   qu'un MLP à 2 époques atteignait déjà son meilleur score de validation. Le GNN ne prédit que
   `arrival_delays + storage_cost`, la seule part que l'ordonnancement influence.

2. Partage de poids au lieu de 650 entrées à plat. Le MLP donnait au moteur #7 ses propres poids,
   indépendants de ceux du moteur #8 : il devait réapprendre 162 fois le même motif, avec seulement
   8 structures de problème distinctes pour l'apprendre. Ici tous les moteurs passent par le même
   encodeur, et le modèle devient invariant à la taille de l'instance -- on peut entraîner sur des
   petites instances (rapides à étiqueter par CP-SAT) et inférer sur les grandes.

3. Prédiction PAR MOTEUR, puis somme. La supervision passe de 1 signal par échantillon à un par
   moteur, et la décomposition additive de l'objectif devient explicite dans l'architecture.
"""
import torch
import torch.nn as nn

from part_preparation.solvers.bilevel.ga.surrogate.features import _max_travel_cost


# ==========================================
# 1. STRUCTURE DU GRAPHE
# ==========================================
NODE_TYPES = ("engine", "factory", "msn", "fal")

# (type source, nom, type destination, dimension des features d'arête)
# Chaque relation est déclarée dans LES DEUX SENS avec des poids distincts : un moteur doit
# apprendre l'état de congestion de son usine, et l'usine doit agréger les moteurs qui se la
# disputent -- ce sont deux fonctions différentes, pas une seule symétrique.
EDGE_SPECS = (
    ("engine", "prepared_at", "factory", 1),
    ("factory", "prepares", "engine", 1),
    ("engine", "assigned_to", "msn", 3),
    ("msn", "served_by", "engine", 3),
    ("factory", "serves", "fal", 2),
    ("fal", "supplied_by", "factory", 2),
    ("msn", "on_line", "fal", 1),
    ("fal", "hosts", "msn", 1),
)


def node_feature_dims(nb_engine_types=2):
    return {
        "engine": 4 + nb_engine_types,
        "factory": 5,
        "msn": 2 + nb_engine_types,
        "fal": 3,
    }


def build_graph(problem, allocation, nb_engine_types=2):
    """
    Construit le graphe hétérogène d'UNE allocation.

    `allocation` est au format plat {engine_id: {"msn_id":..., "factory_id":...}}.
    L'ordre des nœuds `engine` suit problem.engines_list -- le MÊME ordre fixe que
    extract_features et que le vecteur `prep_starts` du dataset de plannings, pour que
    prédictions et cibles se correspondent index par index.
    """
    horizon = float(problem.time_horizon or 1.0)
    max_travel = _max_travel_cost(problem)

    engines = problem.engines_list
    factories = problem.factories_list
    msns = problem.msns_list
    fals = problem.assembly_lines_list

    engine_index = {e.id: i for i, e in enumerate(engines)}
    factory_index = {f.id: i for i, f in enumerate(factories)}
    msn_index = {m.id: i for i, m in enumerate(msns)}
    fal_index = {l.id: i for i, l in enumerate(fals)}

    engine_types = sorted({e.engine_type for e in engines})
    type_index = {t: i for i, t in enumerate(engine_types)}

    def type_onehot(engine_type):
        vec = [0.0] * nb_engine_types
        idx = type_index.get(engine_type)
        if idx is not None and idx < nb_engine_types:
            vec[idx] = 1.0
        return vec

    # --- charge par usine : c'est la quantité qui crée les files d'attente ---
    factory_loads = {}
    for data in allocation.values():
        factory_id = data.get("factory_id")
        if data.get("msn_id") is not None and factory_id is not None:
            factory_loads[factory_id] = factory_loads.get(factory_id, 0) + 1

    # --- nœuds engine ---
    engine_feats = []
    for engine in engines:
        data = allocation.get(engine.id, {})
        msn_id, factory_id = data.get("msn_id"), data.get("factory_id")
        is_allocated = 1.0 if msn_id is not None else 0.0
        delivery = float(engine.delivery_time or 0.0) / horizon

        prep_duration, slack = 0.0, 0.0
        if is_allocated and factory_id is not None:
            factory = problem.factories[factory_id]
            msn = problem.msns[msn_id]
            assembly_line = problem.assembly_lines[msn.assembly_line_id]
            duration = factory.get_preparation_duration(engine.engine_type)
            travel = problem.get_travel_time(factory, assembly_line)
            prep_duration = duration / horizon
            # Marge disponible avant de devoir démarrer : négative => retard garanti même
            # sans aucune file d'attente. C'est le signal local le plus prédictif du retard.
            slack = (msn.engine_installation_date - travel - duration
                     - float(engine.delivery_time or 0.0)) / horizon

        engine_feats.append([is_allocated, delivery, prep_duration, slack] + type_onehot(engine.engine_type))

    # --- nœuds factory ---
    factory_feats = []
    max_capacity = max((f.default_capacity or 1) for f in factories)
    for factory in factories:
        capacity = float(factory.default_capacity or 1)
        load = float(factory_loads.get(factory.id, 0))
        durations = [factory.get_preparation_duration(t) for t in engine_types]
        factory_feats.append([
            capacity / max_capacity,
            load / max(capacity, 1.0),          # taux d'occupation -> congestion
            load / max(len(engines), 1),
            (sum(durations) / len(durations)) / horizon,
            float(factory.takt or 0.0) / horizon,
        ])

    # --- nœuds msn ---
    covered_msn_ids = {d.get("msn_id") for d in allocation.values() if d.get("msn_id") is not None}
    msn_feats = []
    for msn in msns:
        msn_feats.append(
            [float(msn.engine_installation_date) / horizon,
             1.0 if msn.id in covered_msn_ids else 0.0]
            + type_onehot(getattr(msn, "engine_type", None))
        )

    # --- nœuds fal ---
    msns_per_fal = {}
    for msn in msns:
        msns_per_fal[msn.assembly_line_id] = msns_per_fal.get(msn.assembly_line_id, 0) + 1
    fal_feats = []
    for line in fals:
        fal_feats.append([
            float(getattr(line, "takt_pace", 0.0) or 0.0) / horizon,
            msns_per_fal.get(line.id, 0) / max(len(msns), 1),
            float(len([m for m in covered_msn_ids if problem.msns[m].assembly_line_id == line.id]))
            / max(msns_per_fal.get(line.id, 1), 1),
        ])

    # --- arêtes ---
    edges = {spec[1]: ([], [], []) for spec in EDGE_SPECS}

    for engine in engines:
        data = allocation.get(engine.id, {})
        msn_id, factory_id = data.get("msn_id"), data.get("factory_id")
        if msn_id is None or factory_id is None:
            continue

        e_i = engine_index[engine.id]
        f_i = factory_index[factory_id]
        m_i = msn_index[msn_id]

        factory = problem.factories[factory_id]
        msn = problem.msns[msn_id]
        assembly_line = problem.assembly_lines[msn.assembly_line_id]
        duration = factory.get_preparation_duration(engine.engine_type)
        travel_time = problem.get_travel_time(factory, assembly_line)
        ideal_start = msn.engine_installation_date - travel_time - duration
        delay_lb = max(0.0, float(engine.delivery_time or 0.0) - ideal_start)

        for name, src, dst in (("prepared_at", e_i, f_i), ("prepares", f_i, e_i)):
            edges[name][0].append(src)
            edges[name][1].append(dst)
            edges[name][2].append([duration / horizon])

        msn_attr = [ideal_start / horizon, travel_time / horizon, delay_lb / horizon]
        for name, src, dst in (("assigned_to", e_i, m_i), ("served_by", m_i, e_i)):
            edges[name][0].append(src)
            edges[name][1].append(dst)
            edges[name][2].append(msn_attr)

    for factory in factories:
        for line in fals:
            attr = [problem.get_travel_time(factory, line) / horizon,
                    problem.get_travel_cost(factory, line) / max_travel]
            f_i, l_i = factory_index[factory.id], fal_index[line.id]
            for name, src, dst in (("serves", f_i, l_i), ("supplied_by", l_i, f_i)):
                edges[name][0].append(src)
                edges[name][1].append(dst)
                edges[name][2].append(attr)

    for msn in msns:
        m_i, l_i = msn_index[msn.id], fal_index[msn.assembly_line_id]
        attr = [float(msn.engine_installation_date) / horizon]
        for name, src, dst in (("on_line", m_i, l_i), ("hosts", l_i, m_i)):
            edges[name][0].append(src)
            edges[name][1].append(dst)
            edges[name][2].append(attr)

    graph = {
        "x": {
            "engine": torch.tensor(engine_feats, dtype=torch.float32),
            "factory": torch.tensor(factory_feats, dtype=torch.float32),
            "msn": torch.tensor(msn_feats, dtype=torch.float32),
            "fal": torch.tensor(fal_feats, dtype=torch.float32),
        },
        "edges": {},
    }
    for _, name, _, attr_dim in EDGE_SPECS:
        src, dst, attr = edges[name]
        graph["edges"][name] = (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(dst, dtype=torch.long),
            torch.tensor(attr, dtype=torch.float32).reshape(len(src), attr_dim),
        )
    return graph


def collate_graphs(graphs):
    """
    Fusionne plusieurs graphes en un seul en décalant les indices d'arêtes (batching à la PyG).
    Le champ `engine_batch` indique à quel échantillon appartient chaque nœud engine, ce qui
    permet de resommer les prédictions par moteur en un total par allocation.
    """
    batched_x = {t: [] for t in NODE_TYPES}
    batched_edges = {spec[1]: ([], [], []) for spec in EDGE_SPECS}
    offsets = {t: 0 for t in NODE_TYPES}
    engine_batch = []

    for sample_idx, graph in enumerate(graphs):
        for src_type, name, dst_type, _ in EDGE_SPECS:
            src, dst, attr = graph["edges"][name]
            batched_edges[name][0].append(src + offsets[src_type])
            batched_edges[name][1].append(dst + offsets[dst_type])
            batched_edges[name][2].append(attr)

        for node_type in NODE_TYPES:
            batched_x[node_type].append(graph["x"][node_type])

        engine_batch.append(
            torch.full((graph["x"]["engine"].shape[0],), sample_idx, dtype=torch.long)
        )
        for node_type in NODE_TYPES:
            offsets[node_type] += graph["x"][node_type].shape[0]

    return {
        "x": {t: torch.cat(batched_x[t], dim=0) for t in NODE_TYPES},
        "edges": {
            name: (torch.cat(s), torch.cat(d), torch.cat(a))
            for name, (s, d, a) in batched_edges.items()
        },
        "engine_batch": torch.cat(engine_batch),
        "nb_graphs": len(graphs),
    }


# ==========================================
# 2. MESSAGE PASSING
# ==========================================
class HeteroLayer(nn.Module):
    """
    Une couche de message passing hétérogène, en torch pur.

    Agrégation par somme ET par moyenne concaténées : la moyenne décrit le profil typique des
    voisins, la somme leur nombre. Pour la congestion d'une usine, les deux comptent -- trois
    moteurs serrés ou dix moteurs étalés ne produisent pas la même file d'attente, et une
    moyenne seule ne saurait pas les distinguer.
    """

    def __init__(self, hidden):
        super().__init__()
        self.message_nets = nn.ModuleDict({
            name: nn.Linear(hidden + attr_dim, hidden)
            for _, name, _, attr_dim in EDGE_SPECS
        })
        # 1 (état courant) + 2 par relation entrante (somme + moyenne)
        incoming = {t: 0 for t in NODE_TYPES}
        for _, _, dst_type, _ in EDGE_SPECS:
            incoming[dst_type] += 1
        self.update_nets = nn.ModuleDict({
            t: nn.Linear(hidden * (1 + 2 * incoming[t]), hidden) for t in NODE_TYPES
        })
        self.norms = nn.ModuleDict({t: nn.LayerNorm(hidden) for t in NODE_TYPES})

    def forward(self, h, edges, edge_weights=None):
        """
        Args:
            edge_weights: poids par arête, {nom_de_relation: tenseur [nb_arêtes]}, ou None.

                None = tous les poids valent 1, soit exactement le comportement d'origine :
                une arête existe ou n'existe pas. C'est le cas d'un graphe construit par
                `build_graph`, où l'allocation est figée dans la TOPOLOGIE.

                Des poids permettent la relaxation continue dont a besoin une descente de
                gradient sur l'allocation : on crée toutes les arêtes CANDIDATES et on pondère
                chacune par la probabilité de l'affectation correspondante. Une arête de poids 0
                se comporte alors comme une arête absente, et le poids est différentiable là où
                l'existence d'une arête ne l'est pas.

                La moyenne devient une moyenne PONDÉRÉE (somme des poids au dénominateur, et non
                le nombre d'arêtes). Sans cela, ajouter des candidats de poids nul ferait chuter
                la moyenne vers zéro et la relaxation ne coïnciderait plus avec le graphe dur au
                point one-hot -- c'est cette coïncidence qui rend le modèle entraîné réutilisable
                tel quel.
        """
        gathered = {t: [h[t]] for t in NODE_TYPES}

        for src_type, name, dst_type, _ in EDGE_SPECS:
            src, dst, attr = edges[name]
            n_dst = h[dst_type].shape[0]

            if src.numel() == 0:
                zeros = h[dst_type].new_zeros(n_dst, h[dst_type].shape[1])
                gathered[dst_type].extend([zeros, zeros])
                continue

            messages = self.message_nets[name](torch.cat([h[src_type][src], attr], dim=1))
            messages = torch.relu(messages)

            weights = None if edge_weights is None else edge_weights.get(name)
            if weights is not None:
                messages = messages * weights.unsqueeze(1)

            summed = h[dst_type].new_zeros(n_dst, messages.shape[1])
            summed = summed.index_add(0, dst, messages)

            counts = h[dst_type].new_zeros(n_dst)
            contributions = (torch.ones_like(dst, dtype=messages.dtype) if weights is None
                             else weights)
            counts = counts.index_add(0, dst, contributions)
            mean = summed / counts.clamp(min=1.0).unsqueeze(1)

            gathered[dst_type].extend([summed, mean])

        # Connexion résiduelle : sans elle, empiler des couches sur un graphe aussi dense
        # lisse tous les nœuds vers un même vecteur (over-smoothing).
        return {
            t: self.norms[t](h[t] + torch.relu(self.update_nets[t](torch.cat(gathered[t], dim=1))))
            for t in NODE_TYPES
        }


class SchedulingGNN(nn.Module):
    """
    Encodeur partagé + deux têtes par moteur :
      - `delay`  : contribution du moteur au coût d'ordonnancement (retards + storage)
      - `start`  : date de démarrage de préparation, normalisée par l'horizon

    Les deux têtes partagent l'encodeur (multi-tâche) : prédire quand démarrer et prédire le
    coût qui en résulte reposent sur la même compréhension des files d'attente, et la tâche
    `start` fournit une supervision beaucoup plus dense au dataset de plannings.
    """

    def __init__(self, hidden=64, nb_layers=3, nb_engine_types=2):
        super().__init__()
        dims = node_feature_dims(nb_engine_types)
        self.encoders = nn.ModuleDict({
            t: nn.Sequential(nn.Linear(dims[t], hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            for t in NODE_TYPES
        })
        self.layers = nn.ModuleList([HeteroLayer(hidden) for _ in range(nb_layers)])
        self.delay_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.start_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, batch):
        # `edge_weights` est absent des graphes durs (build_graph) et présent dans les graphes
        # relaxés (descent.build_relaxed_graph). Le même modèle entraîné sert donc aux deux,
        # sans conversion ni ré-entraînement.
        edge_weights = batch.get("edge_weights")

        h = {t: self.encoders[t](batch["x"][t]) for t in NODE_TYPES}
        for layer in self.layers:
            h = layer(h, batch["edges"], edge_weights)

        engine_h = h["engine"]
        # softplus : un coût d'ordonnancement est positif par construction. L'imposer par
        # l'architecture évite au réseau de devoir l'apprendre et supprime les prédictions
        # négatives, qui n'ont aucun sens et perturberaient le classement dans le GA.
        per_engine_delay = torch.nn.functional.softplus(self.delay_head(engine_h)).squeeze(-1)
        per_engine_start = self.start_head(engine_h).squeeze(-1)

        total = engine_h.new_zeros(batch["nb_graphs"])
        total.index_add_(0, batch["engine_batch"], per_engine_delay)

        return {
            "total_cost": total,
            "per_engine_delay": per_engine_delay,
            "per_engine_start": per_engine_start,
        }
