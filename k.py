import torch

# 1. Remplace ça par le chemin EXACT que ton GA essaie d'ouvrir
chemin_vers_ton_fichier = r"C:\Users\ESOWNN21\Documents\discrete-optimisation-use-cases-ld\siamese_weights_2cpu.pth"

print(f"Tentative de lecture du fichier : {chemin_vers_ton_fichier}")

try:
    # 2. On charge le fichier brut (sans avoir besoin de la classe du modèle !)
    state_dict = torch.load(chemin_vers_ton_fichier, map_location=torch.device('cpu'))
    
    print("\n✅ Fichier chargé avec succès ! Voici ce qu'il contient :")
    print("-" * 50)
    
    # 3. On affiche la forme (shape) de chaque couche
    for nom_de_la_couche, tenseur_des_poids in state_dict.items():
        print(f"Couche : {nom_de_la_couche.ljust(25)} | Dimensions : {tenseur_des_poids.shape}")

except FileNotFoundError:
    print("\n❌ ERREUR : Le fichier n'existe pas à cet endroit.")
except Exception as e:
    print(f"\n❌ ERREUR inattendue : {e}")