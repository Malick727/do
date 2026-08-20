"""
Encodage d'une allocation D1 en vecteur numérique, pour les surrogates.

C'est le point où le surrogate devient SPÉCIFIQUE À LA DÉCOMPOSITION 1 : l'entrée est une
allocation complète (moteur -> MSN ET moteur -> usine), c'est-à-dire exactement ce que le
niveau supérieur de D1 a figé. En décomposition 2 le niveau supérieur ne fixe que les usines,
donc l'entrée serait différente et cet encodage ne s'appliquerait pas.
"""
# ==========================================
# 1. FEATURE ENGINEERING (RAW ALLOCATION)
# ==========================================
NB_FEATURES_PER_ENGINE = 5
NB_GLOBAL_FEATURES = 2  # taux d'allocation global + congestion moyenne des usines

# Clé du dataset servant de cible, en régression comme en comparaison par paires. Par défaut
# le RÉSIDU et non le score total : cf. problem.residual_score et SiameseAllocationNet.
TARGET_KEY = "residual"


def _compute_factory_loads(allocation_json):
    """Nombre de moteurs routés vers chaque usine dans cette allocation (feature globale)."""
    loads = {}
    for data in allocation_json.values():
        factory_id = data.get("factory_id")
        if factory_id is not None:
            loads[factory_id] = loads.get(factory_id, 0) + 1
    return loads


def _max_travel_cost(problem):
    """Coût de transport maximum observable, pour normaliser travel_cost dans [0, 1]."""
    costs = [
        problem.get_travel_cost(factory, assembly_line)
        for factory in problem.factories_list
        for assembly_line in problem.assembly_lines.values()
    ]
    return max(costs) if costs and max(costs) > 0 else 1.0


def extract_features(allocation_json, problem):
    """
    Transforme l'allocation brute en un vecteur numérique aligné avec le chromosome du GA.
    Pour chaque moteur, 5 features à sémantique métier (au lieu d'indices bruts) :
      - is_allocated: 1.0 si un MSN est assigné, 0.0 sinon
      - travel_cost_norm: coût de transport usine -> chaîne d'assemblage, normalisé
      - delay_lb_norm: borne basse du retard (sans simuler la file d'attente), normalisée
      - factory_load_ratio: charge de l'usine assignée (nb moteurs / capacité, clippé à 3x)
      - installation_date_norm: date d'installation prévue du MSN servi, normalisée
    Puis 2 features GLOBALES (une seule fois, à la fin du vecteur), pour donner au réseau
    un contexte sur l'allocation entière et pas seulement moteur par moteur :
      - global_allocation_rate: proportion de moteurs alloués sur l'ensemble du chromosome
      - global_congestion_rate: congestion moyenne des usines sur l'ensemble du chromosome
    Vecteur de sortie :
      [is_alloc_e1, travel_e1, delay_e1, load_e1, install_e1, ..., alloc_rate, congestion_rate]

    POURQUOI LA DATE D'INSTALLATION. La cible du surrogate est le RÉSIDU, c'est-à-dire la part
    du coût due aux FILES D'ATTENTE dans les usines : qui se dispute quelle usine, et surtout à
    quel moment. Or les quatre premières features sont atemporelles -- `factory_load_ratio`
    compte les moteurs routés vers une usine sans dire s'ils y arrivent le même mois ou étalés
    sur deux ans, et `delay_lb` est explicitement une borne « sans simuler la file d'attente ».
    Deux allocations de congestion très différentes pouvaient donc produire des vecteurs
    quasi identiques : le réseau n'avait aucun moyen de distinguer une usine saturée d'une
    usine simplement très employée, et ne pouvait plus qu'apprendre par cœur (train 96%,
    validation 51% sur des instances non vues).

    La date d'installation est la seule grandeur qui situe chaque moteur dans le temps. Croisée
    avec l'usine assignée, elle rend la congestion temporelle INFÉRABLE -- sans la garantir :
    il reste au réseau à agréger lui-même 1023 couples (usine, date) en une mesure de file
    d'attente, ce qu'un MLP dense sur des positions fixes fait mal. Si la validation reste
    plate après ce changement, c'est l'encodage positionnel qu'il faudra abandonner (cf.
    surrogate.gnn).
    """
    features = []

    factory_loads = _compute_factory_loads(allocation_json)
    max_travel_cost = _max_travel_cost(problem)
    horizon = problem.time_horizon or 1.0

    allocated_count = 0
    load_ratios = []

    # On parcourt les moteurs DANS UN ORDRE FIXE (très important pour les NNs)
    for engine in problem.engines_list:
        data = allocation_json.get(engine.id, {})
        msn_id = data.get("msn_id")
        factory_id = data.get("factory_id")

        factory = problem.factories.get(factory_id) if factory_id is not None else None
        load_ratio = 0.0
        if factory is not None and factory.default_capacity:
            raw_ratio = factory_loads.get(factory_id, 0) / factory.default_capacity
            # Clip à 3x la capacité (usine très surchargée) puis normalise dans [0, 1]
            load_ratio = min(raw_ratio, 3.0) / 3.0
        load_ratios.append(load_ratio)

        if msn_id is None or factory is None:
            # Pas d'allocation MSN (ou pas d'usine valide) : ni coût, ni délai, ni date à
            # évaluer. Le 0.0 de la date est lisible sans ambiguïté puisque is_allocated vaut
            # 0.0 juste à côté -- le réseau dispose du drapeau pour ne pas la confondre avec un
            # MSN installé au tout début de l'horizon.
            features.extend([0.0, 0.0, 0.0, load_ratio, 0.0])
            continue

        allocated_count += 1
        msn = problem.msns[msn_id]
        assembly_line = problem.assembly_lines[msn.assembly_line_id]

        travel_time = problem.get_travel_time(factory, assembly_line)
        travel_cost = problem.get_travel_cost(factory, assembly_line)
        prep_duration = factory.get_preparation_duration(engine.engine_type)

        # Date à laquelle la préparation DEVRAIT démarrer pour arriver à l'heure
        ideal_start = msn.engine_installation_date - travel_time - prep_duration
        delivery_time = engine.delivery_time if engine.delivery_time is not None else 0.0
        # Retard minimum garanti, même sans aucune file d'attente dans l'usine
        delay_lb = max(0.0, delivery_time - ideal_start)

        features.extend([
            1.0,
            travel_cost / max_travel_cost,
            min(delay_lb / horizon, 1.0),
            load_ratio,
            # Normalisée par l'horizon, comme les autres grandeurs temporelles : les dates
            # brutes vont jusqu'à 1600 sur xlarge et écraseraient les features dans [0, 1] à
            # l'entrée de la première couche. `min(..., 1.0)` par sécurité -- une date
            # d'installation au-delà de l'horizon est possible sur des instances issues de CSV.
            min(msn.engine_installation_date / horizon, 1.0),
        ])

    nb_engines = len(problem.engines_list)
    global_allocation_rate = allocated_count / nb_engines if nb_engines else 0.0
    global_congestion_rate = sum(load_ratios) / len(load_ratios) if load_ratios else 0.0
    features.extend([global_allocation_rate, global_congestion_rate])

    return features

