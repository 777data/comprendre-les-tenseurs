# Épisode 5 — Comment l'IA apprend à partir des données

Le même réseau qu'à l'épisode 4, entraîné devant vous sur les chiffres manuscrits de scikit-learn. La perte mesure l'erreur, `backward()` calcule les gradients (des tenseurs de même forme que les poids), et l'optimiseur corrige les poids 400 fois. Le script écrit aussi `journal_entrainement.json` : la perte, l'exactitude et la confiance sur le 7 à chaque passe.

```bash
python apprentissage.py
```

La sortie attendue est dans [resultats_attendus.txt](resultats_attendus.txt).
