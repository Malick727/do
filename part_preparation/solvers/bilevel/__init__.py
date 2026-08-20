"""
Décomposition bilevel du problème de préparation moteurs.

    niveau SUPÉRIEUR : allocation moteur -> MSN  ET  moteur -> usine
    niveau INFÉRIEUR : ordonnancement de la préparation, allocation FIXÉE (cf. bilevel.lower)

À allocation figée, le transport et le nombre de MSN non couverts sont des CONSTANTES
additives : seuls les retards et le stockage varient. C'est cette propriété qui rend le niveau
bas si peu coûteux, et c'est elle qui autorise deux sous-solveurs très différents à partager
exactement le même niveau inférieur.

    bilevel.lower      les trois façons de résoudre le niveau bas (greedy, CP-SAT réduit,
                       CP-SAT complet). PARTAGÉ par les deux sous-solveurs.

    bilevel.ga         SOUS-SOLVEUR 1 : algorithme génétique sur l'allocation, avec son
                       encodage, ses rankers et ses surrogates.
    bilevel.rl         SOUS-SOLVEUR 2 : apprentissage par renforcement.

Les deux sous-solveurs n'ont pas à décider la même chose, et c'est ce qui rend la comparaison
intéressante. Elle n'est cependant valide que s'ils passent par le MÊME niveau bas et le MÊME
score : cf. bilevel.lower.score_solution, unique point de vérité de l'architecture.
"""
