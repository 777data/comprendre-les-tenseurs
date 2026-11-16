# Saison 1 — Les tenseurs

| # | Épisode | Script | Ce que vous allez voir |
|---|---|---|---|
| 1 | [De l'image à la donnée numérique](episode-1-image-en-nombres/) | `image_en_nombres.py` | Une image n'est qu'une grille de 64 nombres (Python seul, sans bibliothèque) |
| 2 | [Créer son premier tenseur avec Python](episode-2-premier-tenseur/) | `premier_tenseur.py` | `torch.tensor`, forme, type, lire un pixel, calculer sur 64 valeurs à la fois |
| 3 | [Comprendre les dimensions d'un tenseur](episode-3-dimensions/) | `dimensions.py` | Rangs 0 à 4, axes, `flatten` et `reshape`, canaux de couleur, lot d'images |
| 4 | [Comment un réseau de neurones utilise les tenseurs](episode-4-reseau-de-neurones/) | `reseau.py`, `entrainer.py` | Un réseau 64 → 16 → 10 : ses poids sont des tenseurs, il reconnaît le 7 une fois entraîné |
| 5 | [Comment l'IA apprend à partir des données](episode-5-apprentissage/) | `apprentissage.py` | La perte, les gradients et la boucle d'entraînement, sur 1 437 vrais chiffres manuscrits |

Chaque dossier contient le script et `resultats_attendus.txt`, la sortie exacte montrée dans la vidéo.

## Les résultats clés

- **Épisode 4** : avec des poids au hasard, le réseau prend le 7 pour un 1. Une fois entraîné, il répond 7 avec une probabilité de 95,7 %.
- **Épisode 5** : la perte passe de 2,348 à 0,015 en 400 passes, et le réseau reconnaît 96,7 % des 360 images qu'il n'a jamais vues.

## Pour aller plus loin

- Dans `premier_tenseur.py`, changez un pixel puis réaffichez l'image.
- Dans `entrainer.py`, remplacez `nn.Linear(64, 16)` par `nn.Linear(64, 32)` (et la couche suivante par `nn.Linear(32, 10)`), puis comparez l'exactitude.
- Dans `apprentissage.py`, essayez `lr=0.1` ou `lr=0.0001` pour voir un apprentissage trop brutal ou trop lent.

Attention : si vous modifiez un script, sa sortie ne correspondra plus à `resultats_attendus.txt` et `pytest` le signalera. C'est normal.
