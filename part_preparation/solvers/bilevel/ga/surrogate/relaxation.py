"""
Relaxation CONTINUE de l'allocation : de quoi descendre un gradient là où il n'y avait qu'un choix.

LE PROBLÈME QUE CECI RÉSOUT. `build_graph` inscrit l'allocation dans la TOPOLOGIE : les arêtes
`engine->msn` et `engine->factory` ne sont créées que pour les moteurs alloués, et `factory.load`
est un COMPTAGE. Une descente de gradient sur les features ne peut donc pas déplacer un moteur
d'un MSN vers un autre -- créer ou supprimer une arête n'est pas une opération différentiable.
Descendre sur les features seules produirait un graphe auquel ne correspond aucune allocation.

CE QU'ON FAIT À LA PLACE. On crée toutes les arêtes CANDIDATES et on pondère chacune par la
probabilité de l'affectation correspondante :

    A[e, m]  probabilité que le moteur e serve le MSN m   (softmax sur ses candidats + « aucun »)
    B[e, f]  probabilité qu'il soit préparé par l'usine f (softmax sur ses usines compatibles)

Les features qui dépendaient de l'allocation deviennent des ESPÉRANCES sous A et B : la charge
d'une usine est une somme de probabilités et non un compte, un MSN est couvert à hauteur de la
probabilité qu'un moteur lui soit affecté. Tout est alors différentiable par rapport aux logits
de A et B, qui sont les vraies variables de décision.

LA PROPRIÉTÉ QUI REND CECI LÉGITIME. Au point ONE-HOT -- A et B concentrées sur un seul candidat
-- le graphe relaxé doit redonner EXACTEMENT le graphe dur, donc la même prédiction du réseau
entraîné. Sans cette coïncidence, la descente optimiserait une fonction différente de celle qui
a été apprise, et le modèle entraîné sur `build_graph` ne serait pas réutilisable ici. C'est ce
que vérifie `check_consistency`, et c'est le test à faire passer AVANT d'écrire quoi que ce soit
d'autre.

CE QUE CE MODULE NE FAIT PAS. Il ne contraint pas l'EXCLUSIVITÉ : un softmax par moteur garantit
« au plus un MSN par moteur », jamais « au plus un moteur par MSN ». Deux moteurs peuvent donc
revendiquer le même MSN avec probabilité 1 chacun, ce qu'aucune allocation ne réalise. C'est une
contrainte d'appariement, qui demande une normalisation de type Sinkhorn (ou une pénalité), et
elle appartient à la boucle de descente, pas à la construction du graphe.
"""
from typing import Dict, List, Optional, Sequence

import torch

from part_preparation.solvers.bilevel.ga.surrogate.features import _max_travel_cost
from part_preparation.solvers.bilevel.ga.surrogate.gnn import EDGE_SPECS, NODE_TYPES


class CandidateSet:
    """
    Les affectations ENVISAGÉES, et elles seules.

    Le produit complet moteur x MSN vaut 1023 x 998 ~ 1 million d'arêtes sur xlarge, pour un
    graphe dur qui en compte ~4000 : la relaxation exhaustive coûterait deux ordres de grandeur
    de plus en mémoire comme en calcul. On restreint chaque moteur à ses `nb_candidates` MSN les
    plus PLAUSIBLES, au sens de la marge disponible entre livraison et installation.

    Le critère est le même `slack` qui sert de feature au nœud engine (cf. gnn.build_graph) :
    un MSN dont la date d'installation précède la livraison du moteur est infaisable, un MSN très
    postérieur impose du stockage. Trier par |slack| retient les appariements serrés, qui sont
    ceux que l'optimisation a une raison de considérer.

    C'est une restriction de l'espace de recherche, donc une perte d'optimalité possible : la
    meilleure allocation peut utiliser un appariement écarté. `nb_candidates` est le curseur qui
    arbitre, et le mettre à None retire la restriction.
    """

    def __init__(self, problem, nb_candidates: Optional[int] = 12):
        self.problem = problem
        self.engines = problem.engines_list
        self.factories = problem.factories_list
        self.msns = problem.msns_list

        self.msn_index = {m.id: i for i, m in enumerate(self.msns)}
        self.factory_index = {f.id: i for i, f in enumerate(self.factories)}

        self.msn_candidates: List[List[int]] = []
        self.factory_candidates: List[List[int]] = []

        for engine in self.engines:
            compatible = problem.engine_compatible_msns.get(engine, [])
            delivery = float(engine.delivery_time or 0.0)

            scored = []
            for msn in compatible:
                # Marge brute, usine ignorée : le classement ne sert qu'à présélectionner, et
                # faire intervenir l'usine ici demanderait de choisir avant d'avoir optimisé.
                scored.append((abs(float(msn.engine_installation_date) - delivery), msn.id))
            scored.sort()
            if nb_candidates is not None:
                scored = scored[:nb_candidates]
            self.msn_candidates.append([self.msn_index[msn_id] for _, msn_id in scored])

            factories = problem.engine_compatible_factories.get(engine, []) or self.factories
            self.factory_candidates.append([self.factory_index[f.id] for f in factories])

    def initial_logits(self, allocation=None, sharpness: float = 12.0):
        """
        Logits de départ. Sans allocation, uniformes ; avec, concentrés dessus.

        Partir d'une allocation connue n'est pas un détail de confort : le surrogate n'a été
        entraîné que sur des allocations issues de trajectoires de GA, et un point de départ
        uniforme est très loin de cette distribution -- exactement là où ses prédictions ne
        valent rien. Descendre depuis une allocation existante garde la recherche près du
        domaine où le modèle a appris quelque chose.

        Le dernier logit de chaque ligne de A est l'option « NON ALLOUÉ », qui doit rester
        atteignable : le déséquilibre moteurs/MSN par type rend une partie des moteurs
        structurellement non affectables (cf. problem.uncoverable_msns).
        """
        nb_engines = len(self.engines)

        # PREMIÈRE PASSE : compléter les listes de candidats AVANT de dimensionner le tenseur.
        # Une allocation de départ peut utiliser un appariement que la présélection a écarté ;
        # l'ajouter après coup décalerait la colonne « non alloué », que `_probabilities` lit
        # comme la DERNIÈRE. Un seul ajout suffisait alors à désaligner tous les indices, et le
        # symptôme était un écart de 1.0 sur `is_allocated` -- pas une erreur numérique, une
        # colonne lue pour une autre.
        if allocation is not None:
            for e, engine in enumerate(self.engines):
                msn_id = allocation.get(engine.id, {}).get("msn_id")
                target = self.msn_index.get(msn_id) if msn_id is not None else None
                if target is not None and target not in self.msn_candidates[e]:
                    self.msn_candidates[e].append(target)

        max_msn = max(len(c) for c in self.msn_candidates) if self.msn_candidates else 0
        max_factory = max(len(c) for c in self.factory_candidates)

        # -inf sur les colonnes hors candidats : après softmax elles valent exactement 0, ce qui
        # est plus sûr qu'un masque appliqué après coup et oublié quelque part.
        msn_logits = torch.full((nb_engines, max_msn + 1), float("-inf"))
        factory_logits = torch.full((nb_engines, max_factory), float("-inf"))

        for e in range(nb_engines):
            msn_logits[e, : len(self.msn_candidates[e])] = 0.0
            msn_logits[e, max_msn] = 0.0  # option « non alloué », toujours ouverte
            factory_logits[e, : len(self.factory_candidates[e])] = 0.0

        if allocation is not None:
            for e, engine in enumerate(self.engines):
                data = allocation.get(engine.id, {})
                msn_id, factory_id = data.get("msn_id"), data.get("factory_id")

                if msn_id is not None and self.msn_index.get(msn_id) is not None:
                    # Présent par construction : la première passe l'a ajouté si nécessaire.
                    msn_logits[e, self.msn_candidates[e].index(self.msn_index[msn_id])] = sharpness
                else:
                    msn_logits[e, max_msn] = sharpness  # colonne « non alloué », toujours la dernière

                if factory_id is not None and self.factory_index.get(factory_id) is not None:
                    target = self.factory_index[factory_id]
                    if target in self.factory_candidates[e]:
                        factory_logits[e, self.factory_candidates[e].index(target)] = sharpness

        return msn_logits, factory_logits


def _probabilities(msn_logits, factory_logits):
    """Softmax par moteur. La dernière colonne de A porte « non alloué »."""
    a_full = torch.softmax(msn_logits, dim=1)
    return a_full[:, :-1], a_full[:, -1], torch.softmax(factory_logits, dim=1)


def build_relaxed_graph(candidates: CandidateSet, msn_logits, factory_logits,
                        nb_engine_types: int = 2):
    """
    Graphe pondéré, différentiable par rapport aux logits.

    Chaque quantité qui dépendait de l'allocation devient son espérance :
      - `is_allocated` -> probabilité d'être alloué
      - `factory.load` -> somme des probabilités d'affectation, au lieu d'un comptage
      - `msn.is_covered` -> probabilité qu'un moteur au moins le serve
      - `slack`, `ideal_start`, `delay_lb` -> espérances sous A (le MSN) et B (l'usine)

    Les arêtes instance (factory<->fal, msn<->fal) sont IDENTIQUES au graphe dur et de poids 1 :
    elles ne dépendent pas de la décision (cf. gnn.build_graph), donc ni la relaxation ni le
    gradient n'ont à les toucher.
    """
    problem = candidates.problem
    horizon = float(problem.time_horizon or 1.0)
    max_travel = _max_travel_cost(problem)

    engines, factories = candidates.engines, candidates.factories
    msns, fals = candidates.msns, problem.assembly_lines_list
    fal_index = {l.id: i for i, l in enumerate(fals)}

    a_msn, p_none, b_factory = _probabilities(msn_logits, factory_logits)
    p_alloc = 1.0 - p_none

    engine_types = sorted({e.engine_type for e in engines})
    type_index = {t: i for i, t in enumerate(engine_types)}

    # ---- tables constantes : durées, trajets, dates. Aucune ne dépend des logits ----
    durations = torch.tensor(
        [[float(f.get_preparation_duration(e.engine_type)) for f in factories] for e in engines],
        dtype=torch.float32)
    install = torch.tensor([float(m.engine_installation_date) for m in msns], dtype=torch.float32)
    delivery = torch.tensor([float(e.delivery_time or 0.0) for e in engines], dtype=torch.float32)
    travel_ff = torch.tensor(
        [[float(problem.get_travel_time(f, l)) for l in fals] for f in factories],
        dtype=torch.float32)
    msn_fal = torch.tensor([fal_index[m.assembly_line_id] for m in msns], dtype=torch.long)

    nb_engines = len(engines)
    msn_cols = a_msn.shape[1]

    # Indices des candidats, -1 pour le remplissage. Les colonnes de remplissage portent une
    # probabilité nulle (logit -inf), donc leurs contributions s'annulent ; on les pointe sur 0
    # uniquement pour que l'indexation reste licite.
    cand_idx = torch.zeros((nb_engines, msn_cols), dtype=torch.long)
    for e, candidate_list in enumerate(candidates.msn_candidates):
        for k, msn_i in enumerate(candidate_list[:msn_cols]):
            cand_idx[e, k] = msn_i

    cand_install = install[cand_idx]                      # [E, K]
    cand_fal = msn_fal[cand_idx]                          # [E, K]

    # B est indexée sur les usines CANDIDATES de chaque moteur (4 du bon type), pas sur les 8
    # usines du parc. Toute table consommée avec B doit donc être ré-indexée de la même façon,
    # sans quoi on multiplierait des probabilités par les durées d'usines d'un autre type.
    nb_factory_cols = b_factory.shape[1]
    fact_idx = torch.zeros((nb_engines, nb_factory_cols), dtype=torch.long)
    for e, factory_list in enumerate(candidates.factory_candidates):
        for c, factory_i in enumerate(factory_list[:nb_factory_cols]):
            fact_idx[e, c] = factory_i

    cand_durations = durations.gather(1, fact_idx)                          # [E, Fc]
    # travel(usine candidate c, FAL du MSN candidat k) -> [E, K, Fc]
    travel_by_candidate = travel_ff[fact_idx]                               # [E, Fc, L]
    cand_travel = travel_by_candidate.gather(
        2, cand_fal.unsqueeze(1).expand(nb_engines, nb_factory_cols, msn_cols)
    ).permute(0, 2, 1)                                                      # [E, K, Fc]

    expected_duration = (b_factory * cand_durations).sum(dim=1)             # [E]
    # E_A,B[travel] = sum_k sum_c A[e,k] B[e,c] travel[e,k,c]
    expected_travel = torch.einsum("ek,ec,ekc->e", a_msn, b_factory, cand_travel)
    expected_install = (a_msn * cand_install).sum(dim=1)                    # [E]

    # ideal_start = install - travel - duration ; slack = ideal_start - delivery
    expected_ideal_start = expected_install - expected_travel - expected_duration
    slack = (expected_ideal_start - delivery) * p_alloc / horizon
    delay_lb = torch.relu(delivery - expected_ideal_start) * p_alloc / horizon

    def type_onehot_rows(items, getter):
        rows = torch.zeros(len(items), nb_engine_types)
        for i, item in enumerate(items):
            idx = type_index.get(getter(item))
            if idx is not None and idx < nb_engine_types:
                rows[i, idx] = 1.0
        return rows

    engine_x = torch.cat([
        p_alloc.unsqueeze(1),
        (delivery / horizon).unsqueeze(1),
        (expected_duration * p_alloc / horizon).unsqueeze(1),
        slack.unsqueeze(1),
        type_onehot_rows(engines, lambda e: e.engine_type),
    ], dim=1)

    # ---- usines : la charge devient une somme de probabilités ----
    # Redispatchée des colonnes CANDIDATES vers les usines réelles : `b_factory[e, 2]` ne
    # désigne pas la même usine pour deux moteurs de types différents.
    load = torch.zeros(len(factories)).index_add(
        0, fact_idx.reshape(-1), (p_alloc.unsqueeze(1) * b_factory).reshape(-1))
    capacities = torch.tensor([float(f.default_capacity or 1) for f in factories])
    max_capacity = capacities.max().clamp(min=1.0)
    mean_durations = torch.tensor(
        [sum(f.get_preparation_duration(t) for t in engine_types) / len(engine_types)
         for f in factories], dtype=torch.float32)
    takts = torch.tensor([float(f.takt or 0.0) for f in factories], dtype=torch.float32)

    factory_x = torch.stack([
        capacities / max_capacity,
        load / capacities.clamp(min=1.0),
        load / max(len(engines), 1),
        mean_durations / horizon,
        takts / horizon,
    ], dim=1)

    # ---- MSN : couverture = probabilité qu'AU MOINS un moteur le serve ----
    # 1 - prod_e (1 - A[e, m]) plutôt que la somme : la somme dépasse 1 dès que deux moteurs
    # revendiquent le même MSN, et sortirait de l'échelle [0, 1] sur laquelle le réseau a été
    # entraîné. Calculé en log pour rester stable quand 1023 facteurs se multiplient.
    coverage = torch.zeros(len(msns))
    log_absent = torch.zeros(len(msns))
    log_absent = log_absent.index_add(
        0, cand_idx.reshape(-1), torch.log1p(-a_msn.reshape(-1).clamp(max=1 - 1e-6)))
    coverage = 1.0 - torch.exp(log_absent)

    msn_x = torch.cat([
        (install / horizon).unsqueeze(1),
        coverage.unsqueeze(1),
        type_onehot_rows(msns, lambda m: getattr(m, "engine_type", None)),
    ], dim=1)

    # ---- FAL : identiques au graphe dur, sauf le taux de couverture ----
    msns_per_fal = torch.zeros(len(fals))
    msns_per_fal = msns_per_fal.index_add(0, msn_fal, torch.ones(len(msns)))
    covered_per_fal = torch.zeros(len(fals)).index_add(0, msn_fal, coverage)

    fal_x = torch.stack([
        torch.tensor([float(getattr(l, "takt_pace", 0.0) or 0.0) for l in fals]) / horizon,
        msns_per_fal / max(len(msns), 1),
        covered_per_fal / msns_per_fal.clamp(min=1.0),
    ], dim=1)

    # ==================== arêtes ====================
    edges: Dict[str, tuple] = {}
    weights: Dict[str, torch.Tensor] = {}

    # engine <-> factory : toutes les usines compatibles, pondérées par B
    ef_src, ef_dst, ef_w, ef_attr = [], [], [], []
    for e in range(nb_engines):
        for k, f in enumerate(candidates.factory_candidates[e]):
            ef_src.append(e)
            ef_dst.append(f)
            ef_w.append(b_factory[e, k] * p_alloc[e])
            ef_attr.append(durations[e, f] / horizon)

    ef_src_t = torch.tensor(ef_src, dtype=torch.long)
    ef_dst_t = torch.tensor(ef_dst, dtype=torch.long)
    ef_weight = torch.stack(ef_w) if ef_w else torch.zeros(0)
    ef_attr_t = (torch.stack(ef_attr) if ef_attr else torch.zeros(0)).reshape(-1, 1)

    edges["prepared_at"] = (ef_src_t, ef_dst_t, ef_attr_t)
    edges["prepares"] = (ef_dst_t, ef_src_t, ef_attr_t)
    weights["prepared_at"] = weights["prepares"] = ef_weight

    # engine <-> msn : les candidats, pondérés par A. L'attribut porte l'ESPÉRANCE sur l'usine,
    # puisque ideal_start et travel_time dépendent de l'usine choisie (cf. gnn.build_graph).
    em_src, em_dst, em_w, em_attr = [], [], [], []
    for e in range(nb_engines):
        for k, msn_i in enumerate(candidates.msn_candidates[e][:msn_cols]):
            travel_ek = (b_factory[e] * cand_travel[e, k]).sum()
            duration_e = expected_duration[e]
            ideal = cand_install[e, k] - travel_ek - duration_e
            em_src.append(e)
            em_dst.append(msn_i)
            em_w.append(a_msn[e, k])
            em_attr.append(torch.stack([
                ideal / horizon,
                travel_ek / horizon,
                torch.relu(delivery[e] - ideal) / horizon,
            ]))

    em_src_t = torch.tensor(em_src, dtype=torch.long)
    em_dst_t = torch.tensor(em_dst, dtype=torch.long)
    em_weight = torch.stack(em_w) if em_w else torch.zeros(0)
    em_attr_t = (torch.stack(em_attr) if em_attr else torch.zeros(0)).reshape(-1, 3)

    edges["assigned_to"] = (em_src_t, em_dst_t, em_attr_t)
    edges["served_by"] = (em_dst_t, em_src_t, em_attr_t)
    weights["assigned_to"] = weights["served_by"] = em_weight

    # factory <-> fal et msn <-> fal : indépendantes de la décision, poids 1
    ff_src, ff_dst, ff_attr = [], [], []
    for fi, factory in enumerate(factories):
        for li, line in enumerate(fals):
            ff_src.append(fi)
            ff_dst.append(li)
            ff_attr.append([problem.get_travel_time(factory, line) / horizon,
                            problem.get_travel_cost(factory, line) / max_travel])
    ff_src_t = torch.tensor(ff_src, dtype=torch.long)
    ff_dst_t = torch.tensor(ff_dst, dtype=torch.long)
    ff_attr_t = torch.tensor(ff_attr, dtype=torch.float32).reshape(-1, 2)
    edges["serves"] = (ff_src_t, ff_dst_t, ff_attr_t)
    edges["supplied_by"] = (ff_dst_t, ff_src_t, ff_attr_t)

    mf_src = torch.arange(len(msns), dtype=torch.long)
    mf_attr = (install / horizon).reshape(-1, 1)
    edges["on_line"] = (mf_src, msn_fal, mf_attr)
    edges["hosts"] = (msn_fal, mf_src, mf_attr)

    for _, name, _, _ in EDGE_SPECS:
        weights.setdefault(name, torch.ones(edges[name][0].shape[0]))

    return {
        "x": {"engine": engine_x, "factory": factory_x, "msn": msn_x, "fal": fal_x},
        "edges": edges,
        "edge_weights": weights,
        "engine_batch": torch.zeros(nb_engines, dtype=torch.long),
        "nb_graphs": 1,
    }


def expected_analytic(candidates: CandidateSet, msn_logits, factory_logits):
    """
    Part ANALYTIQUE du score, en euros et DIFFÉRENTIABLE : transport + MSN non couverts.

    Indispensable, et pas seulement pour l'exactitude du chiffre. Le surrogate ne prédit que le
    RÉSIDU, et le résidu a un optimum trivial : ne rien allouer. `p_alloc = 0` annule tous les
    slacks, donc tous les retards, donc la cible. Une descente sur le seul résidu converge vers
    l'allocation vide en quelques dizaines de pas -- avec une prédiction excellente et un score
    réel catastrophique.

    Ce qui l'en empêche est le poids de `unallocated_msns` : 1 000 000 € par MSN découvert
    évitable, contre ~1 € par jour de retard (cf. problem.AGGREGATED_WEIGHTS). Les deux termes
    ne sont pas du même ordre, et c'est le second qui tient la solution ensemble.

    `uncoverable_msns` est retranché comme le fait problem.evaluate() : une partie des MSN est
    structurellement non couvrable (déséquilibre moteurs/MSN par type), et les facturer rendrait
    le gradient constant sur cette part -- du bruit ajouté à une quantité déjà énorme.
    """
    from part_preparation.problem import AGGREGATED_OBJECTIVES, AGGREGATED_WEIGHTS

    weights = dict(zip(AGGREGATED_OBJECTIVES, AGGREGATED_WEIGHTS))
    problem = candidates.problem
    factories, msns, fals = candidates.factories, candidates.msns, problem.assembly_lines_list
    fal_index = {l.id: i for i, l in enumerate(fals)}

    a_msn, p_none, b_factory = _probabilities(msn_logits, factory_logits)
    p_alloc = 1.0 - p_none
    nb_engines, msn_cols = a_msn.shape
    nb_factory_cols = b_factory.shape[1]

    cand_idx = torch.zeros((nb_engines, msn_cols), dtype=torch.long)
    for e, candidate_list in enumerate(candidates.msn_candidates):
        for k, msn_i in enumerate(candidate_list[:msn_cols]):
            cand_idx[e, k] = msn_i

    fact_idx = torch.zeros((nb_engines, nb_factory_cols), dtype=torch.long)
    for e, factory_list in enumerate(candidates.factory_candidates):
        for c, factory_i in enumerate(factory_list[:nb_factory_cols]):
            fact_idx[e, c] = factory_i

    msn_fal = torch.tensor([fal_index[m.assembly_line_id] for m in msns], dtype=torch.long)
    cost_fl = torch.tensor(
        [[float(problem.get_travel_cost(f, l)) for l in fals] for f in factories],
        dtype=torch.float32)

    cand_fal = msn_fal[cand_idx]                                     # [E, K]
    cost_by_candidate = cost_fl[fact_idx]                            # [E, Fc, L]
    cand_cost = cost_by_candidate.gather(
        2, cand_fal.unsqueeze(1).expand(nb_engines, nb_factory_cols, msn_cols)
    ).permute(0, 2, 1)                                               # [E, K, Fc]

    travel_cost = torch.einsum("ek,ec,ekc->", a_msn, b_factory, cand_cost)

    log_absent = torch.zeros(len(msns)).index_add(
        0, cand_idx.reshape(-1), torch.log1p(-a_msn.reshape(-1).clamp(max=1 - 1e-6)))
    expected_covered = (1.0 - torch.exp(log_absent)).sum()

    unallocated = len(msns) - expected_covered
    avoidable = torch.relu(unallocated - float(problem.uncoverable_msns))

    return (weights["travel_cost"] * travel_cost
            + weights["unallocated_msns"] * avoidable)


def decode_relaxed(candidates: CandidateSet, msn_logits, factory_logits):
    """
    Point relaxé -> allocation ENTIÈRE et réalisable.

    Deux étapes, et la seconde est celle qui compte. L'argmax par moteur ne respecte pas
    l'exclusivité : rien n'empêche deux moteurs de choisir le même MSN, puisque leurs softmax
    sont indépendants. On résout donc les conflits par avidité, en servant d'abord les moteurs
    les plus SÛRS de leur choix -- une probabilité élevée signifie que la descente a concentré sa
    masse, et céder ces appariements-là aux moteurs hésitants détruirait le plus d'information.

    Un moteur évincé reprend son meilleur candidat encore libre, et reste non alloué s'il n'en
    a plus -- même politique de réparation que `encoding.decode`, pour que les allocations
    produites ici soient du même type que celles vues à l'entraînement.
    """
    a_msn, p_none, b_factory = _probabilities(msn_logits, factory_logits)

    claims = []
    for e in range(len(candidates.engines)):
        candidate_list = candidates.msn_candidates[e][:a_msn.shape[1]]
        if not candidate_list:
            claims.append((0.0, e, None))
            continue
        best_k = int(torch.argmax(a_msn[e, : len(candidate_list)]).item())
        confidence = float(a_msn[e, best_k])
        if confidence <= float(p_none[e]):
            claims.append((0.0, e, None))
        else:
            claims.append((confidence, e, best_k))

    claims.sort(key=lambda c: -c[0])

    taken = set()
    allocation: Dict[str, Dict] = {}
    for _, e, best_k in claims:
        engine = candidates.engines[e]
        candidate_list = candidates.msn_candidates[e][:a_msn.shape[1]]

        chosen_msn = None
        if best_k is not None:
            order = torch.argsort(a_msn[e, : len(candidate_list)], descending=True).tolist()
            for k in order:
                msn_i = candidate_list[k]
                if msn_i not in taken:
                    chosen_msn, _ = candidates.msns[msn_i], taken.add(msn_i)
                    break

        factory_list = candidates.factory_candidates[e]
        best_f = int(torch.argmax(b_factory[e, : len(factory_list)]).item())
        factory = candidates.factories[factory_list[best_f]]

        allocation[engine.id] = {
            "msn_id": chosen_msn.id if chosen_msn is not None else None,
            "factory_id": factory.id,
            "engine_obj": engine,
            "msn_obj": chosen_msn,
            "factory_obj": factory,
        }
    return allocation


def check_consistency(problem, allocation, model, nb_candidates: Optional[int] = 12,
                      sharpness: float = 30.0, verbose: bool = True):
    """
    Vérifie qu'AU POINT ONE-HOT le graphe relaxé redonne la prédiction du graphe dur.

    C'est LE test de correction de ce module, et il doit passer avant qu'une descente soit
    écrite. S'il échoue, la relaxation optimise une fonction différente de celle qu'on a
    entraînée : le gradient serait exact... pour un autre problème, ce qu'aucune métrique en
    aval ne révélerait -- une descente qui améliore une fonction fausse a exactement l'allure
    d'une descente qui marche.

    `sharpness` élevé approche le one-hot sans l'atteindre (softmax ne vaut jamais exactement 1),
    donc un écart résiduel de l'ordre de e^-sharpness est attendu et n'est pas une erreur.
    """
    from part_preparation.solvers.bilevel.ga.surrogate.gnn import build_graph, collate_graphs

    hard_graph = collate_graphs([build_graph(problem, allocation)])
    candidates = CandidateSet(problem, nb_candidates=nb_candidates)
    msn_logits, factory_logits = candidates.initial_logits(allocation, sharpness=sharpness)
    relaxed_graph = build_relaxed_graph(candidates, msn_logits, factory_logits)

    model.eval()
    with torch.no_grad():
        hard_cost = float(model(hard_graph)["total_cost"][0])
        relaxed_cost = float(model(relaxed_graph)["total_cost"][0])

    gap = abs(relaxed_cost - hard_cost) / max(abs(hard_cost), 1e-9)

    if verbose:
        print(f"  graphe dur     : {hard_cost:.6f}")
        print(f"  graphe relaxé  : {relaxed_cost:.6f}")
        print(f"  écart relatif  : {gap:.3%}")

        # Diagnostic feature par feature : un écart global ne dit pas d'où il vient, et les
        # espérances les plus susceptibles de diverger (slack, charge des usines) ne sont pas
        # celles qu'on suspecterait en premier.
        for node_type in NODE_TYPES:
            diff = (relaxed_graph["x"][node_type] - hard_graph["x"][node_type]).abs()
            worst = diff.max(dim=0).values
            print(f"  x[{node_type:<7}] écart max par feature : "
                  f"{[round(float(v), 4) for v in worst]}")

    return gap, hard_cost, relaxed_cost
