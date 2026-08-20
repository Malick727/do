"""
Encodage du niveau SUPÉRIEUR de la décomposition 1 : allocation moteur -> MSN et moteur -> usine.

Chromosome : 2 gènes par moteur, dans l'ordre de problem.engines_list.

    [msn_idx_0, factory_idx_0, msn_idx_1, factory_idx_1, ...]

`msn_idx` est un index LOCAL dans les MSN COMPATIBLES du moteur, pas un index global. Avec un
index global, `random.randint(-1, nb_msns - 1)` tombait une fois sur deux sur un MSN d'un autre
type : la moitié des gènes était invalide dès le tirage et le décodeur se contentait de jeter
le moteur. Ici, un gène ne peut plus être type-invalide. (-1 = ne pas allouer.)

Ce module est propre à D1. En décomposition 2 le chromosome n'encoderait que les usines, donc
un autre encodage -- c'est pour ça qu'il vit dans le sous-package de la décomposition et non
à la racine de bilevel/.
"""
import random
from typing import Any, Dict, List, Sequence

from part_preparation.problem import EnginePreparationProblem


def cheapest_factory_index(problem: EnginePreparationProblem, engine, msn) -> int:
    """
    Index GLOBAL de l'usine compatible la moins chère en transport vers la FAL de `msn`.

    Le coût de transport ne dépend que du couple (usine, FAL) : c'est une table, connue avant
    toute résolution (cf. problem.travel_lookup). Le tirer au hasard était donc du gaspillage
    pur -- avec 4 usines compatibles par moteur et 1023 moteurs, la probabilité qu'un tirage
    uniforme approche l'affectation la moins chère est nulle, et aucune mutation gène à gène
    ne la retrouve : il faudrait corriger ~750 gènes simultanément.

    Mesuré sur xlarge#1 : appliquer cette règle à une allocation de dataset fait passer le
    score de 2 539 191 € à 932 391 €, soit 63%. Le meilleur GA après 3 heures et 3 graines
    rendait 2 330 657 €, c'est-à-dire 2.5x pire qu'une règle d'une ligne.

    Ce n'est PAS l'affectation optimale, et elle ne prétend pas l'être : elle ignore la capacité
    des usines, donc les files d'attente. Sous cette règle, Toulouse monte à 89.5 moteurs par
    unité de capacité pendant que Singapore n'en reçoit aucun. L'arbitrage transport/congestion
    reste à faire, et c'est ce que la mutation -- ou un surrogate -- doit pouvoir explorer :
    d'où un POINT DE DÉPART et non une contrainte.
    """
    compatible = problem.engine_compatible_factories.get(engine) or problem.factories_list
    assembly_line = problem.assembly_lines[msn.assembly_line_id]
    best = min(compatible, key=lambda f: problem.get_travel_cost(f, assembly_line))
    return problem.factories_list.index(best)


def create_chromosome(problem: EnginePreparationProblem) -> List[int]:
    """
    MSN tiré au hasard, usine DÉDUITE : la moins chère vers la FAL du MSN retenu.

    Le gène d'usine reste un gène -- mutable, échangeable par croisement -- mais il ne part plus
    d'un tirage uniforme. La diversité initiale porte ainsi sur la seule dimension où elle a un
    sens, l'affectation des MSN ; celle des usines est réintroduite par la mutation, là où elle
    peut payer (décharger une usine saturée), au lieu d'être imposée partout au départ.
    """
    chromosome: List[int] = []
    nb_factories = len(problem.factories_list)

    for engine in problem.engines_list:
        compatible_msns = problem.engine_compatible_msns.get(engine, [])
        msn_idx = random.randint(-1, len(compatible_msns) - 1) if compatible_msns else -1
        factory_idx = (
            cheapest_factory_index(problem, engine, compatible_msns[msn_idx])
            if msn_idx >= 0 else random.randint(0, nb_factories - 1)
        )
        chromosome.extend([msn_idx, factory_idx])
    return chromosome


def crossover_blocks(ind1, ind2, problem: EnginePreparationProblem):
    """
    Croisement en un point, coupé sur une FRONTIÈRE DE MOTEUR et jamais au milieu d'un couple
    (msn, usine). Couper à l'intérieur d'un bloc échangerait le MSN d'un parent avec l'usine
    de l'autre, ce qui détruit l'appariement transport/échéance que le parent avait trouvé.
    """
    nb_engines = len(problem.engines_list)
    if nb_engines < 2:
        return ind1, ind2
    cut_idx = random.randint(1, nb_engines - 1) * 2
    ind1[cut_idx:], ind2[cut_idx:] = ind2[cut_idx:], ind1[cut_idx:]
    return ind1, ind2


def mutate_blocks(individual, indpb: float, problem: EnginePreparationProblem,
                  resync_factory: bool = True):
    """
    Mutation gène à gène, en respectant la convention d'index LOCAL pour les MSN.

    Args:
        resync_factory: quand le gène MSN mute, réaligner l'usine sur la moins chère du NOUVEAU
            MSN. Sans cela le gène d'usine devient périmé à chaque mutation de MSN -- il reste
            optimal pour une FAL que le moteur ne dessert plus -- et l'avantage du démarrage
            se dissout au rythme des mutations. Avec, l'usine SUIT le MSN par défaut, et ne
            s'en écarte que si sa propre mutation le décide.

    Les deux mutations restent indépendantes : après un réalignement, le tirage suivant peut
    toujours envoyer le moteur vers une usine plus chère. C'est la seule façon d'explorer
    l'arbitrage transport/congestion, que la règle du moins cher ignore par construction.
    """
    nb_factories = len(problem.factories_list)

    for engine_pos, engine in enumerate(problem.engines_list):
        i = engine_pos * 2
        compatible_msns = problem.engine_compatible_msns.get(engine, [])

        if random.random() < indpb:
            individual[i] = random.randint(-1, len(compatible_msns) - 1) if compatible_msns else -1
            if resync_factory and individual[i] >= 0:
                individual[i + 1] = cheapest_factory_index(
                    problem, engine, compatible_msns[individual[i]])

        if random.random() < indpb:
            individual[i + 1] = random.randint(0, nb_factories - 1)
    return (individual,)


def decode(
    chromosome: Sequence[int], problem: EnginePreparationProblem
) -> Dict[str, Dict[str, Any]]:
    """
    Décode le chromosome en allocation, en RÉPARANT réellement les conflits.

    Une version qui se contente de rejeter laisse un moteur dont le gène visait un MSN déjà
    pris définitivement non alloué (~63 moteurs alloués sur 162 en pratique). Ici les moteurs
    évincés par une collision sont réappariés à un MSN compatible encore libre, et ne restent
    non alloués que s'il n'en existe plus aucun. Le plafond de couverture vient alors du seul
    déséquilibre moteurs/MSN par type, pas du décodeur.

    Le décodage est DÉTERMINISTE : le RNG de réparation est semé à partir du chromosome. Avec
    `random.choice` sur le RNG global, un même chromosome donnait deux allocations -- donc deux
    fitness -- différentes, ce qui fausse la sélection et rend impossible l'étiquetage d'un
    dataset. C'est aussi ce qui autorise un ranker à mémoïser sur le chromosome.

    Returns:
        engine_id -> {"msn_id", "factory_id", "engine_obj", "msn_obj", "factory_obj"}
    """
    # hash() sur un tuple d'entiers est stable d'une exécution à l'autre (PYTHONHASHSEED ne
    # randomise que les str/bytes), donc la graine est bien reproductible.
    rng = random.Random(hash(tuple(chromosome)))
    allocation: Dict[str, Dict[str, Any]] = {}
    claimed_msn_ids = set()
    engines_needing_msn = []

    for i, engine in enumerate(problem.engines_list):
        msn_idx = chromosome[i * 2]
        factory_idx = chromosome[i * 2 + 1]

        # Réparation de l'usine : un gène peut désigner une usine incompatible.
        factory = problem.factories_list[factory_idx]
        if not factory.can_prepare(engine.engine_type):
            compatible_factories = problem.engine_compatible_factories.get(engine, [])
            if compatible_factories:
                factory = rng.choice(compatible_factories)

        # Passe 1 : honorer le gène quand le MSN visé est encore libre. Le gène étant un index
        # LOCAL aux MSN compatibles, le type est compatible d'office ; le modulo protège les
        # chromosomes issus d'une ancienne convention.
        compatible_msns = problem.engine_compatible_msns.get(engine, [])
        msn = None
        if msn_idx >= 0 and compatible_msns:
            candidate_msn = compatible_msns[msn_idx % len(compatible_msns)]
            if candidate_msn.id not in claimed_msn_ids:
                msn = candidate_msn
                claimed_msn_ids.add(msn.id)
            else:
                engines_needing_msn.append(engine)
        elif compatible_msns:
            # msn_idx == -1 : non-allocation explicite. Traitée comme une PRÉFÉRENCE et non
            # comme une contrainte, donc le moteur passe quand même en passe 2. Laisser un
            # moteur inutilisé alors qu'un MSN compatible reste libre est strictement dominé :
            # on perd la couverture, ET un moteur non alloué coûte PLUS de stockage qu'un
            # moteur alloué (horizon - livraison, cf. problem.evaluate).
            engines_needing_msn.append(engine)

        allocation[engine.id] = {
            "msn_id": msn.id if msn else None,
            "factory_id": factory.id if factory else None,
            "engine_obj": engine,
            "msn_obj": msn,
            "factory_obj": factory,
        }

    # Passe 2 : réapparier les moteurs sans MSN -- victimes d'une collision OU portant un gène
    # à -1. On sert d'abord les MSN dont la date d'installation est la plus proche : ce sont
    # ceux dont la non-couverture coûterait le plus longtemps.
    #
    # À la sortie, un moteur ne reste sans MSN que si AUCUN MSN compatible n'est libre : le
    # couplage est donc MAXIMAL. Et comme un moteur de type T convient à n'importe quel MSN de
    # type T (graphe biparti COMPLET par type), maximal implique MAXIMUM -- la couverture est
    # optimale sans avoir à calculer un couplage maximum.
    for engine in engines_needing_msn:
        candidates = sorted(
            (m for m in problem.engine_compatible_msns.get(engine, [])
             if m.id not in claimed_msn_ids),
            key=lambda m: m.engine_installation_date,
        )
        if not candidates:
            continue
        msn = candidates[0]
        claimed_msn_ids.add(msn.id)
        allocation[engine.id]["msn_id"] = msn.id
        allocation[engine.id]["msn_obj"] = msn

    return allocation
