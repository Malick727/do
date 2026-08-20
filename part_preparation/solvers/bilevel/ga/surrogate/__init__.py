"""
Surrogates du niveau bas de la DÉCOMPOSITION 1.

Un surrogate remplace la résolution du niveau inférieur par une prédiction : au lieu
d'ordonnancer pour connaître le coût d'une allocation, on l'estime. C'est ce qui permet au GA
d'explorer des ordres de grandeur plus d'allocations, au prix d'un classement approché.

    features.py         allocation D1 -> vecteur  (le point qui rend tout ceci propre à D1)
    dataset.py          lecture des datasets, découpage par instance, paires
    dataset_builder.py  étiquetage : produit les datasets en appelant le vrai niveau bas
    schedule_builder.py étiquetage : allocation -> planning optimal (cible par moteur)
    siamese.py          SiameseAllocationNet : MLP sur vecteur plat, architecture + entraînement
    gnn.py              graphe hétérogène d'une allocation + SchedulingGNN (encodeur partagé)
    gnn_siamese.py      entraînement PAR PAIRES de SchedulingGNN : apprend à classer
    calibration.py      recalage du score du siamois, des logits vers les euros

L'ORDRE DE CES MODULES EST UNE HISTOIRE. `siamese.py` encode l'allocation en un vecteur plat et
donne au moteur n°7 des poids distincts de ceux du n°8 : 96% d'accuracy d'entraînement pour 51%
en validation sur des instances non vues, soit le hasard. `gnn.py` remplace cet encodeur par un
réseau à poids partagés où la congestion circule le long des arêtes ; entraîné par
`gnn_siamese.py` sur le même dataset et le même découpage, il passe à ~95% en validation. C'était
donc bien l'encodeur, et non des étiquettes trop bruitées.

Reste que l'entraînement par paires ne contraint que des DIFFÉRENCES de scores, en unités de
logit : le modèle classe mais ne chiffre pas, alors que le GA doit ajouter le résidu prédit à
une part analytique exacte en euros. `calibration.py` estime la pente qui convertit les uns dans
les autres, et `ga.rankers.GnnSiameseRanker` est ce qui en résulte.

Ce sous-package est le seul de d1/ à dépendre de PyTorch. C'est délibéré : lower.py et
rankers.py doivent rester importables sans torch, pour qu'un GA en mode greedy (~0.3 ms par
individu) ne le charge pas pour rien.
"""
