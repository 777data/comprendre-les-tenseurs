# Épisode 3 — Comprendre les dimensions d'un tenseur

Un pixel (rang 0), une ligne (rang 1), l'image (rang 2), une image en couleurs (rang 3, forme [3, 8, 8]) et un lot d'images (rang 4, forme [1, 3, 8, 8]). On additionne selon chaque axe, on aplatit puis on remodèle.

```bash
python dimensions.py
```

La sortie attendue est dans [resultats_attendus.txt](resultats_attendus.txt).
