"""
SOUS-SOLVEUR 1 : algorithme génétique sur le niveau supérieur.

Le chromosome encode l'allocation COMPLÈTE -- moteur -> MSN et moteur -> usine (cf. encoding) ;
le niveau bas complète en ordonnançant. Le GA ne connaît du niveau bas qu'un objet `Ranker` :
changer de méthode de résolution du niveau bas, c'est changer cet objet, rien d'autre.

    encoding      chromosome, opérateurs de variation, décodage
    ranking       le contrat `Ranker` (population -> classement)
    rankers       les Ranker concrets : greedy, CP-SAT, siamois, GNN
    solver        la boucle générationnelle
    rank_quality  mesure de la qualité d'un classement (Spearman...)
    surrogate     réseaux approchant le niveau bas, consommés par les rankers
"""
