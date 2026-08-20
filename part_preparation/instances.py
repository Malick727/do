"""
Registre des instances NOMMÉES du problème de préparation moteurs.

Source unique de vérité. Les mêmes définitions vivaient auparavant, recopiées mot pour mot,
dans generate_dataset.py et run_ga_and_visualize.py -- deux copies d'une définition d'instance,
c'est la garantie de comparer un jour deux approches sur des instances qu'on croit identiques
et qui ne le sont pas.

Deux besoins que ce module sert :

  - comparer des approches (greedy / CP-SAT / surrogate) n'a de sens qu'à instance FIXÉE, donc
    il faut pouvoir désigner une instance par un nom et la régénérer à l'identique ;
  - l'étiquetage d'un dataset et l'entraînement qui le consomme doivent construire EXACTEMENT
    les mêmes problems, sinon les features seraient recalculées sur des dates de MSN
    différentes de celles ayant produit les coûts. D'où `instance_seed`, ici et nulle part
    ailleurs.
"""
import dataclasses
import random
from typing import Dict

from part_preparation.instance_generator import (
    FactoryDefinition,
    InstanceParams,
    generate_instance,
)
from part_preparation.problem import EnginePreparationProblem

# Parc d'usines commun à toutes les instances nommées : 4 usines par type de moteur, de
# capacités et de durées de préparation différentes, dont une sans FAL de rattachement
# (Singapore) pour que le choix d'usine ait un vrai coût de transport.
DEFAULT_FACTORIES = [
    FactoryDefinition(engine_type="EngineType_1", home_fal_index=0, capacity=3, preparation_duration=11, location="Derby"),
    FactoryDefinition(engine_type="EngineType_1", home_fal_index=1, capacity=2, preparation_duration=10, location="Toulouse"),
    FactoryDefinition(engine_type="EngineType_1", home_fal_index=2, capacity=2, preparation_duration=13, location="Virginia"),
    FactoryDefinition(engine_type="EngineType_1", home_fal_index=None, capacity=1, preparation_duration=14, location="Singapore"),
    FactoryDefinition(engine_type="EngineType_2", home_fal_index=0, capacity=4, preparation_duration=15, location="Durham"),
    FactoryDefinition(engine_type="EngineType_2", home_fal_index=1, capacity=3, preparation_duration=15, location="Bordeaux"),
    FactoryDefinition(engine_type="EngineType_2", home_fal_index=2, capacity=1, preparation_duration=15, location="Cincinnati"),
    FactoryDefinition(engine_type="EngineType_2", home_fal_index=3, capacity=2, preparation_duration=12, location="Shanghai"),
]


FAL_TAKT_PACES = [2, 2, 5, 5]


def _params(n_msns_per_fal: int, time_horizon: int) -> InstanceParams:
    return InstanceParams(
        n_fals=4,
        n_msns_per_fal=n_msns_per_fal,
        n_engine_types=2,
        time_horizon=time_horizon,
        fal_takt_paces=list(FAL_TAKT_PACES),
        factory_definitions=DEFAULT_FACTORIES,
        # Type 1 excédentaire (1.1), type 2 déficitaire (0.95) : le déséquilibre par type est
        # ce qui crée un plancher structurel de MSN non couvrables (cf.
        # EnginePreparationProblem.uncoverable_msns).
        engine_to_msn_ratio={"EngineType_1": 1.1, "EngineType_2": 0.95},
        supplier_regularity={"EngineType_1": 1.0, "EngineType_2": 0.8},
        supplier_production_rate={"EngineType_1": 1.0, "EngineType_2": 0.9},
        flexible_delivery_prob=0,
    )


# "medium" : ~162 moteurs pour 160 MSN. C'est l'instance de référence des datasets.
# "xlarge" : 1000 MSN (250 par FAL sur 4 FAL), ~1030 moteurs. L'horizon suit la cadence : la
# FAL la plus lente a un takt de 5 jours, donc 250 MSN y occupent 1250 jours ; 1600 laisse de
# la marge pour la préparation et le transport en fin d'horizon.
INSTANCES: Dict[str, InstanceParams] = {
    "small": _params(n_msns_per_fal=10, time_horizon=150),
    "medium": _params(n_msns_per_fal=40, time_horizon=300),
    "large": _params(n_msns_per_fal=80, time_horizon=1000),
    "xlarge": _params(n_msns_per_fal=250, time_horizon=1600),
}

DEFAULT_INSTANCE = "medium"


def sized_params(n_msns_per_fal: int) -> InstanceParams:
    """
    Instance AD HOC d'une taille quelconque, avec un horizon DÉDUIT et non choisi.

    L'horizon n'est pas un paramètre libre : la FAL la plus lente a un takt de 5 jours, donc
    n MSN y occupent 5n jours, auxquels s'ajoutent la préparation et le transport en fin
    d'horizon. Le fixer trop court ne rend pas l'instance infaisable, ce qui la rendrait
    visible -- il rend une part croissante des MSN structurellement non couvrables, et le score
    se met à mesurer l'horizon plutôt que la méthode. C'est le piège qui faisait passer
    `n_msns_per_fal=1000, time_horizon=300` pour une instance valide.

    Le facteur 1.3 et la marge de 100 jours reproduisent le rapport des instances nommées
    (medium : 40 MSN/FAL pour 300 jours ; xlarge : 250 pour 1600), pour qu'une taille ad hoc
    soit du même régime qu'elles et non d'un régime plus tendu ou plus lâche.

    À réserver à l'exploration. Toute comparaison destinée à être publiée doit passer par une
    instance NOMMÉE : celles-ci sont stables dans le temps, alors que cette fonction suit ce
    qu'on lui passe.
    """
    return _params(
        n_msns_per_fal=n_msns_per_fal,
        time_horizon=int(max(FAL_TAKT_PACES) * n_msns_per_fal * 1.3) + 100,
    )


def varied_params(seed: int, n_msns_per_fal: int = None,
                  size_range: tuple = (20, 120)) -> InstanceParams:
    """
    Instance de MÊME STRUCTURE que les instances nommées, mais aux paramètres TIRÉS.

    Destinée à l'entraînement d'une politique, pas à la comparaison de méthodes. Les instances
    nommées gardent des paramètres figés (cf. INSTANCES) et restent la seule référence pour
    publier un résultat ; ici on fait varier ce que la politique ne doit surtout pas apprendre
    par coeur.

    Deux paramètres varient, et chacun débloque une décision aujourd'hui presque vide de sens :

    `supplier_regularity` -- l'écart entre la livraison d'un moteur et l'échéance de son MSN de
        référence. Au réglage actuel (1.0 pour le type 1) cet écart vaut 17.6 jours à ±2 jours
        près : tous les moteurs sont pratiquement interchangeables, donc CHOISIR le MSN ne
        change presque rien. À 0.5 l'écart va de 8 à 39 jours -- un moteur livré avec 8 jours de
        marge alors que la préparation en prend 10 à 15 est structurellement en retard, et il
        devient nécessaire de décider quel MSN lui donner et lequel réserver.

    `fal_takt_paces` -- la cadence de chaque FAL, donc le profil de congestion. Fixé à
        [2, 2, 5, 5] dans toutes les instances nommées, il est identique d'un réplicat à
        l'autre : une politique peut le mémoriser au lieu de lire la charge courante.

    L'horizon reste DÉDUIT des takts, jamais choisi -- même piège que dans sized_params : un
    horizon trop court ne rend pas l'instance infaisable, il rend une part croissante des MSN
    structurellement non couvrables et le score se met à mesurer l'horizon.
    """
    rng = random.Random(seed)
    takts = [rng.choice([2, 3, 4, 5, 6]) for _ in range(4)]
    engine_types = ["EngineType_1", "EngineType_2"]

    # La TAILLE varie aussi, sauf si elle est imposée. C'est ce qui manquait : une politique
    # entraînée uniquement sur ~80 moteurs par type doit ensuite en traiter 500 sur xlarge, et
    # plusieurs features (rangs normalisés, fractions de vivier) ne se comportent pas de la même
    # façon selon la taille. Tirer la taille met ces régimes dans l'entraînement.
    if n_msns_per_fal is None:
        n_msns_per_fal = rng.randint(*size_range)

    params = _params(
        n_msns_per_fal=n_msns_per_fal,
        time_horizon=int(max(takts) * n_msns_per_fal * 1.3) + 100,
    )
    return dataclasses.replace(
        params,
        seed=seed,
        fal_takt_paces=takts,
        supplier_regularity={t: rng.uniform(0.4, 1.0) for t in engine_types},
        # Le déséquilibre par type est ce qui crée le plancher de MSN non couvrables ; on le
        # fait varier autour des valeurs nommées sans jamais l'annuler, pour que la politique
        # rencontre les deux régimes (excédent de moteurs, excédent de MSN).
        engine_to_msn_ratio={
            "EngineType_1": rng.uniform(1.0, 1.2),
            "EngineType_2": rng.uniform(0.9, 1.05),
        },
    )


def instance_seed(replicate_id: int, params: InstanceParams) -> int:
    """
    Seed du réplicat `replicate_id` (0-indexé) d'une instance nommée.

    Sert à produire une FAMILLE d'instances de même structure et de dates différentes : c'est
    ce dont l'apprentissage a besoin (un surrogate entraîné sur une seule instance
    n'apprendrait que ses dates), et ce dont une comparaison d'approches a besoin pour ne pas
    conclure sur un coup de chance.

    Le décalage de +1 garantit que le réplicat 0 diffère de l'instance de base.
    """
    return params.seed + (replicate_id + 1)


def get_params(name: str = DEFAULT_INSTANCE) -> InstanceParams:
    if name not in INSTANCES:
        raise ValueError(f"Instance inconnue : {name!r}. Disponibles : {sorted(INSTANCES)}")
    return INSTANCES[name]


def build(name: str = DEFAULT_INSTANCE, replicate_id: int = None) -> EnginePreparationProblem:
    """
    Construit l'instance nommée `name`, ou son réplicat `replicate_id`.

    Deux appels avec les mêmes arguments donnent le MÊME problem : generate_instance() repose
    sur random.seed(params.seed). C'est cette propriété qui rend une comparaison d'approches
    reproductible, et qui permet à l'étiquetage et à l'entraînement de partager des instances
    sans se transmettre de fichier.
    """
    params = get_params(name)
    if replicate_id is not None:
        params = dataclasses.replace(params, seed=instance_seed(replicate_id, params))
    return generate_instance(params)
