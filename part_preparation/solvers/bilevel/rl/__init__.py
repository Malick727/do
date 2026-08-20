"""
SOUS-SOLVEUR 2 : apprentissage par renforcement sur l'affectation des usines.

Partage des décisions du niveau supérieur -- c'est ce qui distingue ce sous-solveur du GA :

    moteur -> usine        POLITIQUE APPRISE      (ce package)
    moteur -> MSN          LAP exact              (bilevel.matching)
    dates de préparation   greedy                 (bilevel.lower)

L'agent n'apprend donc QUE les usines. Les deux autres décisions ont une solution exacte ou
déjà bonne, et les faire apprendre reviendrait à dépenser de la capacité pour réapprendre ce
qu'on sait calculer -- même raisonnement que la décomposition analytique/résiduelle des
surrogates (cf. problem.ANALYTIC_OBJECTIVES).

À écrire.
"""
